#!/usr/bin/env bash
# Build the held-out benchmark (if missing) and score all registry models.
#   RUN=synth bash scripts/run_benchmark.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT"
if [ -f "$PROJECT/env.sh" ] && [ -d "$PROJECT/.venv" ]; then
  # shellcheck disable=SC1091
  source "$PROJECT/env.sh"; set +u; source "$PROJECT/.venv/bin/activate"; set -u
fi

RUN="${RUN:-synth}"
SYNTH="${SYNTH:-data/processed/task_sft.synth.full.jsonl}"
BENCH="data/benchmark/benchmark.jsonl"

if [ ! -f "$BENCH" ]; then
  echo "==> Building held-out benchmark (grounded=${GROUNDED:-120})"
  python scripts/build_benchmark.py --passages data/corpus/passages.jsonl \
      --train "$SYNTH" --redteam data/redteam/redteam.example.jsonl \
      --grounded "${GROUNDED:-120}" --out "$BENCH"
fi
# Frontier CLOSED comparators (Anthropic/OpenAI/Google) — only if a key is set. Generated
# via API, then folded into the SAME leaderboard with --precomputed. Non-fatal.
PRE=()
if [ -n "${ANTHROPIC_API_KEY:-}${OPENAI_API_KEY:-}${GOOGLE_API_KEY:-}" ]; then
  # Install ONLY the SDKs for providers whose key is set (not in the base install, to
  # keep setup lean). Idempotent + non-fatal.
  pkgs=""
  [ -n "${ANTHROPIC_API_KEY:-}" ] && pkgs="$pkgs anthropic"
  [ -n "${OPENAI_API_KEY:-}" ]    && pkgs="$pkgs openai"
  [ -n "${GOOGLE_API_KEY:-}" ]    && pkgs="$pkgs google-genai"
  if [ -n "$pkgs" ]; then echo "==> Ensuring API SDKs:$pkgs"; pip install -q $pkgs || true; fi
  echo "==> Generating frontier API comparator cards (keys detected)..."
  if python scripts/api_comparators.py --benchmark "$BENCH" \
        --out data/benchmark/api_outputs.jsonl; then
    [ -f data/benchmark/api_outputs.jsonl ] && PRE=(--precomputed data/benchmark/api_outputs.jsonl)
  else
    echo "==> API comparator generation failed (non-fatal); open comparators still scored."
  fi
fi
echo "==> Scoring registry models + external baselines (run=$RUN, paraphrases=${PARAPHRASES:-1})"
EXTRA="config/benchmark_models.conf"
MCQ="data/benchmark/mcq.jsonl"
python scripts/benchmark.py --benchmark "$BENCH" --from-registry "$RUN" \
    --paraphrases "${PARAPHRASES:-1}" \
    ${EXTRA:+--extra-registry "$EXTRA"} \
    ${PRE[@]+"${PRE[@]}"} \
    $( [ -f "$MCQ" ] && echo "--mcq $MCQ" )
