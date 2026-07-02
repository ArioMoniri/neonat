#!/usr/bin/env bash
# ============================================================================
# reset_runs.sh — clear FAILED/partial run artifacts to start clean, while
# KEEPING the expensive things: your synthetic cards, the corpus, downloaded
# model weights (.hf_cache), and the venv.
#
# Removes:  models/  (failed/partial adapters),  data/benchmark/  (leaderboard +
#           mcq drafts),  **/__pycache__.
# Keeps:    data/processed/*.jsonl (your 390 cards), data/corpus/, .hf_cache,
#           .pip_cache, .venv, env.sh.
#
#   bash scripts/reset_runs.sh          # asks first
#   bash scripts/reset_runs.sh --yes
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"
case "$PROJECT" in ""|"/"|"$HOME") echo "refusing: PROJECT='$PROJECT'"; exit 3 ;; esac

echo "Will remove (KEEPS cards/corpus/caches/venv):"
echo "    $PROJECT/models         (failed/partial adapters)"
echo "    $PROJECT/data/benchmark  (leaderboard + mcq drafts)"
echo "    **/__pycache__"
if [ "${1:-}" != "--yes" ] && [ "${1:-}" != "-y" ]; then
  printf "Proceed? [y/N] "; read -r a; case "$a" in y|Y|yes) ;; *) echo "Aborted."; exit 0 ;; esac
fi
rm -rf -- "$PROJECT/models" "$PROJECT/data/benchmark"
find "$PROJECT" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
echo "==> Clean. Kept: data/processed (cards), data/corpus, .hf_cache, .venv."
echo "    Re-run:  SYNTH=data/processed/task_sft.synth.full.jsonl bash scripts/neoperi_launch.sh all"
