#!/usr/bin/env bash
# Run training inside the project venv with the project-local model cache.
# All args are passed through to train_lora.py, e.g.:
#   bash scripts/run_train.sh data/processed/task_sft.jsonl run-01
#   bash scripts/run_train.sh data/processed/task_sft.jsonl --smoke-only
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

cd "$PROJECT"   # outputs (models/, data/) resolve relative to the project
echo "==> python scripts/train_lora.py $* (HF_HOME=$HF_HOME)"
exec python scripts/train_lora.py "$@"
