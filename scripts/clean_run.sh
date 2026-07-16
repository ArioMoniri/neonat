#!/usr/bin/env bash
# clean_run.sh — remove a PREVIOUS run's outputs so you can retrain from scratch on the
# SAME device. Adapters + generated dataset + benchmark + corpus + logs by default;
# model/pip caches and the venv are opt-in (they force a re-download / reinstall).
#
#   bash scripts/clean_run.sh              # adapters, synth dataset, benchmark, corpus, logs
#   CACHE=1 bash scripts/clean_run.sh      # ALSO .hf_cache/.pip_cache  (re-download weights)
#   VENV=1  bash scripts/clean_run.sh      # ALSO .venv                 (full reinstall)
#   KEEP_CORPUS=1 bash scripts/clean_run.sh  # keep data/corpus (avoid re-hitting PMC/EuropePMC)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT"
echo "==> Cleaning previous run under $PROJECT"
rm -rf models/*-neoperi-* 2>/dev/null || true
rm -f  data/processed/*.synth*.jsonl 2>/dev/null || true
rm -f  data/benchmark/*.jsonl data/benchmark/leaderboard.* 2>/dev/null || true
rm -f  state/*.log 2>/dev/null || true
echo "   removed: adapters, synth dataset(s), benchmark, run logs"
if [ "${KEEP_CORPUS:-0}" != "1" ]; then
  rm -f data/corpus/passages.jsonl 2>/dev/null || true
  echo "   removed: data/corpus/passages.jsonl (rebuilt from sources next run)"
fi
if [ "${CACHE:-0}" = "1" ]; then rm -rf .hf_cache .pip_cache && echo "   removed: .hf_cache .pip_cache (weights will re-download)"; fi
if [ "${VENV:-0}" = "1" ]; then rm -rf .venv && echo "   removed: .venv (setup_server.sh will reinstall)"; fi
echo "==> Clean complete. Disk:"; df -h "$PROJECT" 2>/dev/null | awk 'NR<=2'
