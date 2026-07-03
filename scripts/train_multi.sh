#!/usr/bin/env bash
# ============================================================================
# train_multi.sh — QLoRA fine-tune SEVERAL base families on the same synthetic
# data, so they can be benchmarked against each other.
#
# Reads config/models.conf (name | hf_id | gated | train_flags). Gated models
# (Gemma) are skipped unless HF_TOKEN is set. Each model -> its own adapter at
# models/<name>-neoperi-<RUN>/.
#
# Usage:
#   bash scripts/train_multi.sh data/processed/task_sft.synth.full.jsonl
#   MODELS="kumru,gemma3-4b" EPOCHS=3 LORA_R=32 bash scripts/train_multi.sh <data>
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT"

if [ -f "$PROJECT/env.sh" ] && [ -d "$PROJECT/.venv" ]; then
  # shellcheck disable=SC1091
  source "$PROJECT/env.sh"; set +u; source "$PROJECT/.venv/bin/activate"; set -u
fi

DATA="${1:-data/processed/task_sft.synth.full.jsonl}"
[ -f "$DATA" ] || { echo "ERROR: training data not found: $DATA" >&2; exit 1; }

RUN="${RUN:-synth}"
EPOCHS="${EPOCHS:-2}"; LORA_R="${LORA_R:-16}"; LORA_ALPHA="${LORA_ALPHA:-$((LORA_R*2))}"
LR="${LR:-1e-4}"; MAXSEQ="${MAXSEQ:-2048}"
ONLY="${MODELS:-}"     # optional comma list to restrict which registry rows run
CONF="$PROJECT/config/models.conf"
[ -f "$CONF" ] || { echo "ERROR: $CONF missing" >&2; exit 1; }

# Detect whether we have an HF token (for gated Gemma) — several env names.
HF_TOK="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-${HUGGINGFACE_TOKEN:-}}}"

echo "============================================================"
echo "train_multi  data=$DATA run=$RUN epochs=$EPOCHS lora_r=$LORA_R"
echo "  gated models $( [ -n "$HF_TOK" ] && echo ENABLED || echo 'SKIPPED (no HF_TOKEN)')"
echo "============================================================"

trained=(); skipped=()
while IFS='|' read -r name hf_id gated flags; do
  name="$(echo "$name" | xargs)"; hf_id="$(echo "$hf_id" | xargs)"
  gated="$(echo "$gated" | xargs)"; flags="$(echo "${flags:-}" | xargs)"
  [ -z "$name" ] && continue
  case "$name" in \#*) continue ;; esac                 # comment line
  if [ -n "$ONLY" ] && ! echo ",$ONLY," | grep -q ",$name,"; then continue; fi
  if [ "$gated" = "1" ] && [ -z "$HF_TOK" ]; then
    echo "==> SKIP $name ($hf_id): gated, no HF token."
    skipped+=("$name"); continue
  fi
  out="models/${name}-neoperi-${RUN}"
  echo ""
  echo "############### FINE-TUNE $name  ($hf_id) ###############"
  # shellcheck disable=SC2086
  if python scripts/train_lora.py "$DATA" --allow-synthetic \
        --base-model "$hf_id" --output-dir "$out" \
        --epochs "$EPOCHS" --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" \
        --lr "$LR" --max-seq-len "$MAXSEQ" \
        $flags; then
    trained+=("$name")
  else
    echo "==> WARNING: training FAILED for $name (continuing with the rest)."
    skipped+=("$name")
  fi
done < "$CONF"

echo ""
echo "==> train_multi done. trained: ${trained[*]:-none}   skipped: ${skipped[*]:-none}"
echo "    Next: bash scripts/run_benchmark.sh"
