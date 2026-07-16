#!/usr/bin/env bash
# bootstrap.sh — ONE command to clean (optional), pull latest from GitHub, set up the
# venv, and run the whole neoperi pipeline on the server. Idempotent + self-healing.
#
# RECOMMENDED (keeps your terminal for the interactive key wizard + tmux):
#   export HF_TOKEN=hf_...                       # gated Gemma-4 + optional pushes
#   export NCBI_API_KEY=...  NCBI_EMAIL=you@x    # optional: faster PMC/EuropePMC pulls
#   # optional frontier benchmark comparators:  ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY
#   CLEAN=1 bash <(curl -fsSL https://raw.githubusercontent.com/ArioMoniri/neonat/main/scripts/bootstrap.sh)
#
# (Plain `curl ... | bash` also works but can't show the interactive key wizard — set the
#  keys as env vars first if you pipe.)
#
# Env knobs: PROJECT(/data/neoperi-cdss) STAGE(all) BRANCH(main) CLEAN(0) CACHE(0) VENV(0)
#   run tuning: TEACHER TEACHER_FALLBACK REFUSAL_RATIO MODELS LIMIT VARIANTS SOURCES EPOCHS
set -euo pipefail
REPO_URL="${REPO_URL:-https://github.com/ArioMoniri/neonat.git}"
BRANCH="${BRANCH:-main}"
PROJECT="${PROJECT:-/data/neoperi-cdss}"
STAGE="${STAGE:-all}"

echo "=================================================================="
echo " neoperi bootstrap  repo=$REPO_URL  branch=$BRANCH"
echo "                    project=$PROJECT  stage=$STAGE  clean=${CLEAN:-0}"
echo "=================================================================="

# 1) clone or hard-update to latest (survives a dirty/stale checkout)
# Git 2.35+ refuses a repo owned by another uid ("dubious ownership") — allow this one.
git config --global --add safe.directory "$PROJECT" 2>/dev/null || true
git config --global --add safe.directory '*' 2>/dev/null || true

if [ ! -d "$PROJECT" ] || [ -z "$(ls -A "$PROJECT" 2>/dev/null)" ]; then
  echo "==> Cloning $REPO_URL -> $PROJECT"
  mkdir -p "$(dirname "$PROJECT")"
  git clone --branch "$BRANCH" "$REPO_URL" "$PROJECT"
else
  # Existing dir — git, scp'd, or HALF-initialized (a partial prior run may have created
  # .git but no 'origin'). Make it a clean checkout of origin/$BRANCH in place, preserving
  # untracked caches/venv/data (no multi-hour weight re-download). Idempotent + self-healing.
  echo "==> Syncing existing $PROJECT to origin/$BRANCH (preserving caches/venv)"
  [ -d "$PROJECT/.git" ] || git -C "$PROJECT" init -q
  git -C "$PROJECT" remote add origin "$REPO_URL" 2>/dev/null \
    || git -C "$PROJECT" remote set-url origin "$REPO_URL"
  git -C "$PROJECT" fetch origin "$BRANCH"
  git -C "$PROJECT" reset --hard "origin/$BRANCH"
fi
cd "$PROJECT"
echo "==> code version: $(grep -m1 NEOPERI_VERSION scripts/train_lora.py | sed 's/.*= *//')"

# 2) optional clean of a previous run on the SAME device (fast; rm only)
if [ "${CLEAN:-0}" = "1" ]; then
  CACHE="${CACHE:-0}" VENV="${VENV:-0}" bash scripts/clean_run.sh
fi

# 3) run — the launcher re-execs itself INSIDE tmux first, then (inside tmux) runs
#    setup_server.sh if the venv is missing and activates it. So pip install, model
#    downloads, and training ALL happen inside tmux+venv and survive an SSH drop.
echo "==> Launching stage '$STAGE' (setup + venv + run all happen inside tmux)"
exec bash scripts/neoperi_launch.sh "$STAGE"
