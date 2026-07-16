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
if [ -d "$PROJECT/.git" ]; then
  echo "==> Updating existing git checkout to origin/$BRANCH"
  git -C "$PROJECT" fetch origin "$BRANCH"
  git -C "$PROJECT" reset --hard "origin/$BRANCH"
elif [ -d "$PROJECT" ] && [ -n "$(ls -A "$PROJECT" 2>/dev/null)" ]; then
  # Existing NON-git dir (e.g. scp'd files) — adopt it in place so .hf_cache/.venv are
  # preserved (no multi-hour weight re-download). Tracked repo files are force-updated;
  # untracked caches/generated data are left alone.
  echo "==> Adopting existing non-git $PROJECT in place (preserving caches/venv)"
  git -C "$PROJECT" init -q
  git -C "$PROJECT" remote add origin "$REPO_URL" 2>/dev/null || git -C "$PROJECT" remote set-url origin "$REPO_URL"
  git -C "$PROJECT" fetch origin "$BRANCH"
  git -C "$PROJECT" reset --hard "origin/$BRANCH"
else
  echo "==> Cloning $REPO_URL -> $PROJECT"
  mkdir -p "$(dirname "$PROJECT")"
  git clone --branch "$BRANCH" "$REPO_URL" "$PROJECT"
fi
cd "$PROJECT"
echo "==> code version: $(grep -m1 NEOPERI_VERSION scripts/train_lora.py | sed 's/.*= *//')"

# 2) optional clean of a previous run on the SAME device
if [ "${CLEAN:-0}" = "1" ]; then
  CACHE="${CACHE:-0}" VENV="${VENV:-0}" bash scripts/clean_run.sh
fi

# 3) venv + deps (idempotent; installs the transformers>=4.60 stack that loads Qwen3/Gemma-4)
bash scripts/setup_server.sh

# 4) run — setup wrote env.sh; the launcher self-activates the venv, runs the interactive
#    preflight/key-wizard, and re-execs itself inside tmux so an SSH drop can't kill it.
echo "==> Launching stage '$STAGE'"
exec bash scripts/neoperi_launch.sh "$STAGE"
