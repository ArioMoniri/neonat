#!/usr/bin/env bash
# ============================================================================
# plug_and_train.sh — COMPLETE, self-contained "plug & train" pipeline.
#
#   open literature  ->  teacher LLM distills grounded TR cards  ->  QLoRA
#   fine-tune Kumru-2B (synthetic mode)  ->  research gate.
#
# No clinician data and no API keys required (the teacher runs on your GPU).
# The result is a RESEARCH PROTOTYPE trained on machine-generated data — it is
# NOT clinician-reviewed and NOT for clinical use.
#
# Configure via env vars (all optional) — turn these UP to scale the model:
#   SOURCES    (default "europepmc,pubmed")  corpus sources: europepmc,pubmed,urls,exa
#   LIMIT      (default 400)                  max passages pulled
#   PER_TOPIC  (default 8)                    articles fetched per topic (more = bigger corpus)
#   VARIANTS   (default 1)                    cards generated PER passage (multiplies data)
#   TEACHER    (default Qwen/Qwen2.5-72B-Instruct)
#   EPOCHS     (default 2)   LORA_R (default 16)   LR (default 1e-4)   MAXSEQ (default 2048)
#   URLS_FILE  (optional)                     file of guideline URLs for SOURCES=...,urls
#   RUN        (default synth-run)            adapter run name
#
# Example "much bigger" run (≈ thousands of cards, higher-capacity adapter):
#   LIMIT=2000 PER_TOPIC=40 VARIANTS=3 EPOCHS=3 LORA_R=32 bash scripts/plug_and_train.sh
#
# Usage:
#   bash scripts/plug_and_train.sh                 # full pipeline
#   bash scripts/plug_and_train.sh --plumbing      # offline: stub corpus + stub teacher,
#                                                  # validates the data path (no GPU/network)
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"

PLUMBING=0
[ "${1:-}" = "--plumbing" ] && PLUMBING=1

if [ ! -f "$PROJECT/env.sh" ] || [ ! -d "$PROJECT/.venv" ]; then
  echo "ERROR: run scripts/setup_server.sh first (no venv/env.sh found)." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$PROJECT/env.sh"
set +u
# shellcheck disable=SC1091
source "$PROJECT/.venv/bin/activate"
set -u
cd "$PROJECT"

SOURCES="${SOURCES:-europepmc,pubmed}"
LIMIT="${LIMIT:-400}"
PER_TOPIC="${PER_TOPIC:-8}"
VARIANTS="${VARIANTS:-1}"
TEACHER="${TEACHER:-Qwen/Qwen3-32B}"   # apache-2.0 dense; clean distillation rights
EPOCHS="${EPOCHS:-2}"
LORA_R="${LORA_R:-16}"
LR="${LR:-1e-4}"
MAXSEQ="${MAXSEQ:-2048}"
RUN="${RUN:-synth-run}"
CORPUS="data/corpus/passages.jsonl"
SYNTH="data/processed/task_sft.synth.jsonl"
ADAPTER="models/kumru-neoperi-lora-${RUN}"
TRAIN_ARGS=(--epochs "$EPOCHS" --lora-r "$LORA_R" --lr "$LR" --max-seq-len "$MAXSEQ")

echo "============================================================"
echo "plug_and_train  sources=$SOURCES limit=$LIMIT per_topic=$PER_TOPIC variants=$VARIANTS"
echo "  teacher=$TEACHER  epochs=$EPOCHS lora_r=$LORA_R lr=$LR maxseq=$MAXSEQ  run=$RUN"
echo "  (plumbing=$PLUMBING)"
echo "============================================================"

if [ "$PLUMBING" -eq 1 ]; then
  echo "==> [1/4] corpus (offline selftest)"
  python scripts/build_corpus.py --selftest --out "$CORPUS"
  echo "==> [2/4] synthesize (dry-run stub teacher)"
  python scripts/synthesize_cards.py --passages "$CORPUS" --out "$SYNTH" \
      --teacher "$TEACHER" --limit "$LIMIT" --dry-run
  echo "==> [3/4] validate the synthetic data through the training gate (smoke would need a GPU)"
  python - "$SYNTH" <<'PY'
import importlib.util, sys
s=importlib.util.spec_from_file_location("tl","scripts/train_lora.py")
tl=importlib.util.module_from_spec(s); s.loader.exec_module(tl)
rows, synth = tl.load_data(sys.argv[1], allow_synthetic=True)
print(f"    data-path OK: {len(rows)} rows, synthetic_run={synth}")
PY
  echo "==> [4/4] plumbing OK. Real run: omit --plumbing (needs GPU + downloads)."
  exit 0
fi

EXTRA_ARGS=()
[ -n "${URLS_FILE:-}" ] && EXTRA_ARGS+=(--urls "$URLS_FILE")
[ "${ACCEPT_UNVETTED:-0}" = "1" ] && EXTRA_ARGS+=(--accept-unvetted-license)

echo "==> [1/4] Building open-literature corpus -> $CORPUS"
python scripts/build_corpus.py --out "$CORPUS" --sources "$SOURCES" \
    --limit "$LIMIT" --per-topic "$PER_TOPIC" "${EXTRA_ARGS[@]}"

echo "==> [2/4] Distilling grounded TR cards with teacher -> $SYNTH"
python scripts/synthesize_cards.py --passages "$CORPUS" --out "$SYNTH" \
    --teacher "$TEACHER" --limit "$LIMIT" --variants "$VARIANTS"

echo "==> [3/4] Smoke + full QLoRA fine-tune (SYNTHETIC mode) -> $ADAPTER"
python scripts/train_lora.py "$SYNTH" "$RUN" --allow-synthetic --smoke-only "${TRAIN_ARGS[@]}"
python scripts/train_lora.py "$SYNTH" "$RUN" --allow-synthetic "${TRAIN_ARGS[@]}"

echo "==> [4/4] Research gate (eval + red-team) on $ADAPTER"
set +e
python scripts/evaluate.py --adapter "$ADAPTER" \
    --redteam data/redteam/redteam.example.jsonl --train "$SYNTH"
EVAL_RC=$?
set -e

echo ""
echo "==> DONE. Adapter: $ADAPTER"
if [ "$EVAL_RC" -eq 0 ]; then
  echo "    GATE: PASS (RESEARCH_GATE_OK) — see $ADAPTER/RESEARCH_GATE_OK"
else
  echo "    GATE: BLOCKED — see $ADAPTER/RESEARCH_GATE_BLOCKED (rc=$EVAL_RC)"
fi
echo "    Artifacts: metrics.json, PROVENANCE.json in $ADAPTER"
echo "    This is a RESEARCH PROTOTYPE trained on machine-generated data — NOT for clinical use."
exit "$EVAL_RC"
