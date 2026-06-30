#!/usr/bin/env bash
# ============================================================================
# uninstall.sh — clean teardown of the project footprint.
#
# Default: removes the heavy GENERATED footprint (venv, model cache, training
#          outputs, logs, __pycache__) but KEEPS your source and your
#          clinician-reviewed data (data/processed, data/staging).
#
#   bash scripts/uninstall.sh            # remove generated footprint (asks first)
#   bash scripts/uninstall.sh --yes      # ... without the prompt
#   bash scripts/uninstall.sh --all      # remove the ENTIRE project dir (incl. data!)
#
# The simplest nuke is always:  rm -rf <project dir>
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Safety guards: never operate on a dangerous root.
case "$PROJECT" in
  ""|"/"|"$HOME") echo "Refusing to run: PROJECT resolves to '$PROJECT'." >&2; exit 3 ;;
  /*/*) : ;;                       # require at least depth-2 absolute path
  *) echo "Refusing to run: PROJECT '$PROJECT' is too shallow." >&2; exit 3 ;;
esac
# Sentinel: only act on something that is actually a neoperi-cdss project.
if [ ! -f "$PROJECT/config/requirements-train.txt" ] || [ ! -f "$PROJECT/scripts/train_lora.py" ]; then
  echo "Refusing to run: '$PROJECT' does not look like a neoperi-cdss project." >&2
  exit 3
fi

ALL=0; YES=0
for a in "$@"; do
  case "$a" in
    --all) ALL=1 ;;
    --yes|-y) YES=1 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

if [ "$ALL" -eq 1 ]; then
  echo "This will DELETE THE ENTIRE PROJECT (including any clinician data):"
  echo "    $PROJECT"
else
  echo "This will remove the generated footprint (KEEPS source + data/):"
  echo "    $PROJECT/.venv"
  echo "    $PROJECT/.hf_cache   (downloaded models)"
  echo "    $PROJECT/.pip_cache  (pip cache)"
  echo "    $PROJECT/models      (trained adapters)"
  echo "    $PROJECT/state, env.sh, **/__pycache__"
fi

if [ "$YES" -ne 1 ]; then
  printf "Proceed? [y/N] "
  read -r ans
  case "$ans" in y|Y|yes|YES) ;; *) echo "Aborted."; exit 0 ;; esac
fi

if [ "$ALL" -eq 1 ]; then
  PARENT="$(cd "$PROJECT/.." && pwd)"
  cd "$PARENT"
  rm -rf -- "$PROJECT"
  echo "==> Removed $PROJECT"
else
  rm -rf -- "$PROJECT/.venv" "$PROJECT/.hf_cache" "$PROJECT/.pip_cache" \
            "$PROJECT/models" "$PROJECT/state" "$PROJECT/env.sh"
  find "$PROJECT" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
  echo "==> Removed generated footprint. Source + data/ kept."
  echo "    (Full nuke: rm -rf \"$PROJECT\")"
fi
