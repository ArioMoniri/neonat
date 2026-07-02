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

# Self-bootstrap: set up the venv on first run so this is a true single entrypoint.
if [ ! -f "$PROJECT/env.sh" ] || [ ! -d "$PROJECT/.venv" ]; then
  echo "==> First run: no venv yet — running setup_server.sh ..."
  bash "$PROJECT/scripts/setup_server.sh"
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
  mig_teacher_guard
  [ -f "$CORPUS" ] || run_corpus
  python scripts/synthesize_cards.py --passages "$CORPUS" --out "$SYNTH" \
    --teacher "${TEACHER:-Qwen/Qwen2.5-72B-Instruct}" --limit "${LIMIT:-400}" \
    --variants "${VARIANTS:-1}"
}
# Gemma-4/MedGemma are multimodal: their processor imports timm (vision) and
# librosa/soundfile (audio) even for text-only use, and Gemma-4 needs a very recent
# transformers. Install these only when such a model is actually planned.
ensure_gemma_deps() {
  if grep -qiE '^[[:space:]]*[^#].*gemma' "$PROJECT/config/models.conf" \
     "$PROJECT/config/benchmark_models.conf" 2>/dev/null; then
    echo "==> Gemma/MedGemma planned -> ensuring recent transformers + timm/librosa/soundfile."
    python -m pip install -U "transformers>=4.60" timm librosa soundfile || \
      echo "    (install failed; if a Gemma load errors: pip install -U transformers timm librosa soundfile)"
  fi
}
# Abort early if a 235B teacher is asked for on a too-small (MIG) GPU.
mig_teacher_guard() {
  case "${TEACHER:-}" in *235B*|*235b*) ;; *) return 0 ;; esac
  local freemb
  freemb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  if [ -n "$freemb" ] && [ "$freemb" -lt 130000 ]; then
    echo "ERROR: TEACHER=$TEACHER needs ~130GB (MIG OFF / full H200); this GPU reports ${freemb}MiB." >&2
    echo "       Use TEACHER=Qwen/Qwen2.5-72B-Instruct or Qwen/Qwen3-32B, or disable MIG." >&2
    exit 1
  fi
}
ensure_build_tools() {
  if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1 \
     && command -v apt-get >/dev/null 2>&1; then
    echo "==> Installing a C compiler (build-essential) — self-heal."
    apt-get update -y && apt-get install -y build-essential || true
  fi
}
run_train() {
  echo "### [train] fine-tune all registry students"
  ensure_build_tools
  ensure_gemma_deps
  [ -f "$SYNTH" ] || run_distill
  RUN="$RUN" EPOCHS="${EPOCHS:-2}" LORA_R="${LORA_R:-16}" MODELS="${MODELS:-}" \
    bash scripts/train_multi.sh "$SYNTH"
}
run_mcq() {
  echo "### [mcq] synthetic knowledge probe (teacher-generated)"
  mig_teacher_guard
  [ -f "$CORPUS" ] || run_corpus
  python scripts/build_mcq.py --passages "$CORPUS" --train "$SYNTH" \
    --grounded data/benchmark/benchmark.jsonl \
    --teacher "${TEACHER:-Qwen/Qwen2.5-72B-Instruct}" --n "${MCQ_N:-100}" \
    --out data/benchmark/mcq.jsonl || echo "==> MCQ build skipped/failed (non-fatal)."
}
run_bench() {
  echo "### [bench] leaderboard (+ MCQ probe if present)"
  RUN="$RUN" SYNTH="$SYNTH" bash scripts/run_benchmark.sh
}

# -------- Interactive PREFLIGHT: pop up and ask for anything missing ----------
planned_students() {
  local only="${MODELS:-}"
  while IFS='|' read -r name _hf gated _f; do
    name="$(echo "$name" | xargs)"; [ -z "$name" ] && continue
    case "$name" in \#*) continue ;; esac
    if [ -n "$only" ] && ! echo ",$only," | grep -q ",$name,"; then continue; fi
    echo -n "$name "
  done < "$PROJECT/config/models.conf"
}
preflight() {
  echo "======================= PREFLIGHT ======================="
  local gpu memmb disk
  gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  memmb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  disk="$(df -Pk "$PROJECT" | awk 'NR==2{printf "%.0f", $4/1024/1024}')"
  echo "  GPU: ${gpu:-none}  (${memmb:-?} MiB)     free disk: ${disk:-?} GiB"
  echo "  Teacher: ${TEACHER:-Qwen/Qwen2.5-72B-Instruct}"
  echo "  Students: $(planned_students)"
  echo "  HF token: $([ -n "${HF_TOKEN:-}" ] && echo present || echo 'absent (gated models will be skipped)')"
  # Teacher fit check — offer to downshift if it won't fit.
  local t="${TEACHER:-Qwen/Qwen2.5-72B-Instruct}"
  if [ -n "$memmb" ]; then
    if echo "$t" | grep -qiE '235b' && [ "$memmb" -lt 130000 ]; then
      echo "  ! $t needs ~130GB; this GPU has ${memmb}MiB."
    elif echo "$t" | grep -qiE '72b' && [ "$memmb" -lt 45000 ]; then
      echo "  ! 72B teacher may not fit ${memmb}MiB."
    fi
    if [ "$memmb" -lt 45000 ] && [ -t 0 ] && [ -z "${TEACHER:-}" ]; then
      printf "  Switch teacher to Qwen/Qwen3-32B (fits smaller VRAM)? [Y/n] "
      read -r a; case "$a" in n|N) ;; *) export TEACHER="Qwen/Qwen3-32B"; echo "  -> TEACHER=Qwen3-32B" ;; esac
    fi
  fi
  if [ -n "${disk:-}" ] && [ "${disk%.*}" -lt 60 ]; then
    echo "  ! Low disk (${disk} GiB). Models can need 40-120GB; free space or 'rm -rf .hf_cache'."
  fi
  if [ -t 0 ]; then
    printf "Proceed? [Y/n] "; read -r go; case "$go" in n|N) echo "Aborted."; exit 0 ;; esac
  fi
  echo "========================================================="
}

case "$STAGE" in
  corpus)  run_corpus ;;
  distill) preflight; run_distill ;;
  train)   preflight; run_train ;;
  mcq)     preflight; run_mcq ;;
  bench)   run_bench ;;
  all)     preflight; run_train; run_mcq; run_bench ;;   # train() chains corpus+distill if missing
  *) echo "unknown stage: $STAGE (use corpus|distill|train|mcq|bench|all)" >&2; exit 2 ;;
esac
echo "==> neoperi_launch stage '$STAGE' complete."
