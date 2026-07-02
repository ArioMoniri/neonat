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
  echo "==> Building held-out benchmark"
  python scripts/build_benchmark.py --passages data/corpus/passages.jsonl \
      --train "$SYNTH" --redteam data/redteam/redteam.example.jsonl --out "$BENCH"
fi
echo "==> Scoring registry models + external baselines (run=$RUN)"
EXTRA="config/benchmark_models.conf"
MCQ="data/benchmark/mcq.jsonl"
python scripts/benchmark.py --benchmark "$BENCH" --from-registry "$RUN" \
    ${EXTRA:+--extra-registry "$EXTRA"} \
    $( [ -f "$MCQ" ] && echo "--mcq $MCQ" )
