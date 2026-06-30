#!/usr/bin/env bash
# Run the eval + red-team release gate inside the project venv.
#   bash scripts/run_eval.sh --adapter models/kumru-neoperi-lora-run-01 \
#       --redteam data/redteam/redteam.example.jsonl [--eval data/eval/heldout.jsonl]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ ! -f "$PROJECT/env.sh" ] || [ ! -d "$PROJECT/.venv" ]; then
  echo "ERROR: run scripts/setup_server.sh first (no venv/env.sh found)." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$PROJECT/env.sh"
set +u  # venv activate may reference unbound vars
# shellcheck disable=SC1091
source "$PROJECT/.venv/bin/activate"
set -u

cd "$PROJECT"
echo "==> python scripts/evaluate.py $* (HF_HOME=$HF_HOME)"
exec python scripts/evaluate.py "$@"
