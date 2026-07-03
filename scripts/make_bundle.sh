#!/usr/bin/env bash
# ============================================================================
# make_bundle.sh — build the self-contained transfer tarball (run LOCALLY).
#
# Produces dist/neoperi-cdss-bundle.tar.gz containing ONLY what the server needs:
# code, configs, docs, and the *.example.jsonl files. It deliberately EXCLUDES
# real/clinician data, the venv, model caches, and trained adapters.
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

STAGE_PARENT="$REPO/dist"
STAGE="$STAGE_PARENT/neoperi-cdss"
TARBALL="$STAGE_PARENT/neoperi-cdss-bundle.tar.gz"

rm -rf "$STAGE"
mkdir -p "$STAGE/scripts" "$STAGE/config" "$STAGE/docs" \
         "$STAGE/data/processed" "$STAGE/data/staging" \
         "$STAGE/data/redteam" "$STAGE/data/eval" "$STAGE/.claude/agents"

# Code
cp "$REPO/scripts/"*.py "$STAGE/scripts/"
cp "$REPO/scripts/"*.sh "$STAGE/scripts/"
cp "$REPO/config/requirements-train.txt" "$STAGE/config/"
cp "$REPO/config/models.conf" "$STAGE/config/"
cp "$REPO/config/benchmark_models.conf" "$STAGE/config/"
cp "$REPO/config/hf_datasets.txt" "$STAGE/config/"
cp -R "$REPO/docs/neoperi-cdss" "$STAGE/docs/"
cp -R "$REPO/.claude/agents/." "$STAGE/.claude/agents/"

# Example data ONLY (no real/PHI data ever goes in the bundle).
cp "$REPO/data/processed/task_sft.example.jsonl" "$STAGE/data/processed/"
cp "$REPO/data/redteam/redteam.example.jsonl"    "$STAGE/data/redteam/"

# Keep empty dirs present on extract.
touch "$STAGE/data/staging/.gitkeep" "$STAGE/data/eval/.gitkeep"

# COPYFILE_DISABLE=1 stops macOS from embedding AppleDouble (._*) files; the
# bsdtar flags drop xattrs/metadata so the Linux server gets a clean extract.
TAR_FLAGS=()
if tar --no-mac-metadata --version >/dev/null 2>&1; then
  TAR_FLAGS+=(--no-mac-metadata --no-xattrs)
fi
COPYFILE_DISABLE=1 tar "${TAR_FLAGS[@]}" -czf "$TARBALL" -C "$STAGE_PARENT" neoperi-cdss
SIZE="$(du -h "$TARBALL" | cut -f1)"

echo "==> Built $TARBALL ($SIZE)"
echo ""
echo "Transfer it to the server (adjust port/host/path to yours):"
echo "    scp -P 30405 \"$TARBALL\" root@10.6.110.10:~/"
echo ""
echo "Then on the server:"
echo "    tar xzf neoperi-cdss-bundle.tar.gz"
echo "    cd neoperi-cdss"
echo "    bash scripts/setup_server.sh"
echo ""
echo "Full runbook: docs/neoperi-cdss/TRANSFER.md"
