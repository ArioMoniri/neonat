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
# Configure via env vars (all optional):
#   SOURCES   (default "europepmc,pubmed")   corpus sources: europepmc,pubmed,urls,exa
#   LIMIT     (default 400)                   max passages / training examples
#   TEACHER   (default Qwen/Qwen2.5-72B-Instruct)
#   URLS_FILE (optional)                      file of guideline URLs for SOURCES=...,urls
#   RUN       (default synth-run)             adapter run name
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
TEACHER="${TEACHER:-Qwen/Qwen2.5-72B-Instruct}"
RUN="${RUN:-synth-run}"
CORPUS="data/corpus/passages.jsonl"
SYNTH="data/processed/task_sft.synth.jsonl"
ADAPTER="models/kumru-neoperi-lora-${RUN}"

echo "============================================================"
echo "plug_and_train  sources=$SOURCES limit=$LIMIT teacher=$TEACHER run=$RUN"
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
    --limit "$LIMIT" "${EXTRA_ARGS[@]}"

echo "==> [2/4] Distilling grounded TR cards with teacher -> $SYNTH"
python scripts/synthesize_cards.py --passages "$CORPUS" --out "$SYNTH" \
    --teacher "$TEACHER" --limit "$LIMIT"

echo "==> [3/4] Smoke + full QLoRA fine-tune (SYNTHETIC mode) -> $ADAPTER"
python scripts/train_lora.py "$SYNTH" "$RUN" --allow-synthetic --smoke-only
python scripts/train_lora.py "$SYNTH" "$RUN" --allow-synthetic

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
