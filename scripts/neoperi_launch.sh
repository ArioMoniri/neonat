#!/usr/bin/env bash
# ============================================================================
# neoperi_launch.sh — ONE launcher for the whole pipeline, tmux-safe.
#
#   corpus -> distill (teacher) -> train MANY bases -> benchmark leaderboard
#
# - Prompts for an HF token on the server if a GATED model (Gemma) is in the plan.
# - Re-execs itself inside a tmux session so an SSH drop can't kill the run.
# - Every stage is an existing script; this just orchestrates + passes env knobs.
#
# STAGES (arg 1, default "all"):  corpus | distill | train | bench | all
#
# Env knobs (all optional):
#   SOURCES LIMIT PER_TOPIC VARIANTS TEACHER   (data build/distill)
#   MODELS EPOCHS LORA_R LR MAXSEQ RUN         (multi-model training)
#   NO_TMUX=1  to skip tmux    HF_TOKEN=...  to pre-supply the token
#
# Examples:
#   bash scripts/neoperi_launch.sh all
#   MODELS="kumru,qwen25-3b" EPOCHS=3 VARIANTS=3 bash scripts/neoperi_launch.sh all
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT"

STAGE="${1:-all}"
RUN="${RUN:-synth}"
CORPUS="data/corpus/passages.jsonl"
SYNTH="${SYNTH:-data/processed/task_sft.synth.full.jsonl}"
TOKEN_FILE="$PROJECT/.hf_token"

if [ ! -f "$PROJECT/env.sh" ] || [ ! -d "$PROJECT/.venv" ]; then
  echo "ERROR: run scripts/setup_server.sh first (no venv/env.sh)." >&2; exit 1
fi

# --- Does the plan need a gated (Gemma) model? -------------------------------
needs_token() {
  case "$STAGE" in train|all) ;; *) return 1 ;; esac
  local only="${MODELS:-}"
  while IFS='|' read -r name _hf gated _flags; do
    name="$(echo "$name" | xargs)"; gated="$(echo "$gated" | xargs)"
    [ -z "$name" ] && continue; case "$name" in \#*) continue ;; esac
    if [ -n "$only" ] && ! echo ",$only," | grep -q ",$name,"; then continue; fi
    [ "$gated" = "1" ] && return 0
  done < "$PROJECT/config/models.conf"
  return 1
}

# --- HF token: load cached, else prompt (interactive only) -------------------
[ -z "${HF_TOKEN:-}" ] && [ -f "$TOKEN_FILE" ] && HF_TOKEN="$(cat "$TOKEN_FILE")"
if [ -z "${HF_TOKEN:-}" ] && needs_token; then
  if [ -t 0 ]; then
    echo "A GATED model (Gemma) is in the plan and needs a Hugging Face token."
    echo "Create one at https://huggingface.co/settings/tokens and accept the Gemma"
    echo "license on its model page first. Leave empty to just SKIP gated models."
    printf "HF token (input hidden): "
    read -rs HF_TOKEN; echo
    if [ -n "$HF_TOKEN" ]; then
      umask 077; printf '%s' "$HF_TOKEN" > "$TOKEN_FILE"
      echo "==> Saved to $TOKEN_FILE (chmod 600). Gated models enabled."
    else
      echo "==> No token entered; gated models will be skipped."
    fi
  else
    echo "==> Non-interactive and no HF_TOKEN; gated models will be skipped."
  fi
fi
# Persist a provided token to the chmod-600 file so we never pass it on a command
# line (visible in ps); the inner run re-reads it from the file.
if [ -n "${HF_TOKEN:-}" ] && [ ! -f "$TOKEN_FILE" ]; then
  umask 077; printf '%s' "$HF_TOKEN" > "$TOKEN_FILE"
fi
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

# --- Re-exec inside tmux so SSH drops don't kill the run (token via file) -----
if [ -z "${NO_TMUX:-}" ] && [ -z "${TMUX:-}" ] && command -v tmux >/dev/null 2>&1 && [ -t 1 ]; then
  SESS="neoperi"
  echo "==> Launching in tmux session '$SESS' (detach: Ctrl-b d; reattach: tmux attach -t $SESS)"
  exec tmux new-session -A -s "$SESS" \
    "cd '$PROJECT'; NO_TMUX=1 MODELS='${MODELS:-}' \
     SOURCES='${SOURCES:-}' LIMIT='${LIMIT:-}' PER_TOPIC='${PER_TOPIC:-}' \
     VARIANTS='${VARIANTS:-}' TEACHER='${TEACHER:-}' EPOCHS='${EPOCHS:-}' \
     LORA_R='${LORA_R:-}' RUN='$RUN' bash scripts/neoperi_launch.sh '$STAGE'; \
     echo; echo '[stage $STAGE finished — press enter to close]'; read _"
fi

# shellcheck disable=SC1091
source "$PROJECT/env.sh"; set +u; source "$PROJECT/.venv/bin/activate"; set -u

run_corpus() {
  echo "### [corpus] building open-literature corpus"
  local extra=(); [ -n "${URLS_FILE:-}" ] && extra+=(--urls "$URLS_FILE")
  python scripts/build_corpus.py --out "$CORPUS" \
    --sources "${SOURCES:-europepmc,pubmed}" --limit "${LIMIT:-400}" \
    --per-topic "${PER_TOPIC:-8}" "${extra[@]}"
}
run_distill() {
  echo "### [distill] teacher -> grounded TR cards"
  [ -f "$CORPUS" ] || run_corpus
  python scripts/synthesize_cards.py --passages "$CORPUS" --out "$SYNTH" \
    --teacher "${TEACHER:-Qwen/Qwen2.5-72B-Instruct}" --limit "${LIMIT:-400}" \
    --variants "${VARIANTS:-1}"
}
run_train() {
  echo "### [train] fine-tune all registry students"
  # Gemma 4 (multimodal) needs the latest transformers; upgrade if it's planned.
  if grep -qiE '^\s*[^#].*gemma-4' "$PROJECT/config/models.conf"; then
    echo "==> Gemma 4 in plan -> upgrading transformers (needs latest)."
    python -m pip install -U "transformers>=4.56" || \
      echo "    (upgrade failed; if Gemma 4 load errors, run: pip install -U transformers)"
  fi
  [ -f "$SYNTH" ] || run_distill
  RUN="$RUN" EPOCHS="${EPOCHS:-2}" LORA_R="${LORA_R:-16}" MODELS="${MODELS:-}" \
    bash scripts/train_multi.sh "$SYNTH"
}
run_bench() { echo "### [bench] leaderboard"; RUN="$RUN" SYNTH="$SYNTH" bash scripts/run_benchmark.sh; }

case "$STAGE" in
  corpus)  run_corpus ;;
  distill) run_distill ;;
  train)   run_train ;;
  bench)   run_bench ;;
  all)     run_train; run_bench ;;   # train() chains corpus+distill if missing
  *) echo "unknown stage: $STAGE (use corpus|distill|train|bench|all)" >&2; exit 2 ;;
esac
echo "==> neoperi_launch stage '$STAGE' complete."
