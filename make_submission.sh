#!/usr/bin/env bash
# Build the course submission ZIP: GroupNo_25_ProjectNo_2.zip
#
#   ./make_submission.sh                 # writes ./GroupNo_25_ProjectNo_2.zip (repository root)
#
# Put the report and slides in report/ first (any file there is included, tracked or not),
# e.g. report/Group25_Report.pdf and report/Group25_Slides.pptx.
#
# Included: every file tracked by git, minus what the course asks to leave out:
#   - raw simulator run folders (results-ca/*/runs/), except one representative run
#   - raw hidden-state prediction dumps (endsem/gpu/results/hs_pred_*.npz), except one
#   - simulator sources (cloned at pinned commits by setup.sh), traces (downloaded by
#     endsem/traces/download.py), build directories and caches are never tracked.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
NAME=GroupNo_25_ProjectNo_2
OUT=${1:-"$ROOT/$NAME.zip"}
KEEP_RUN="results-ca/q2_offload_load_vs_exec/runs/A100_mixtral_b1_base/"
KEEP_NPZ="endsem/gpu/results/hs_pred_b1_ctx1024.npz"

cd "$ROOT"
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/$NAME"

{
  git ls-files | grep -v -E '^results-ca/[^/]+/runs/' | grep -v -E '^endsem/gpu/results/hs_pred_.*\.npz$' \
    | grep -v -x -E 'LLMSimulator|external/LLMSimulator' | grep -v -E '\.zip$'
  git ls-files "$KEEP_RUN"
  echo "$KEEP_NPZ"
  ls endsem/results/*_run.log 2>/dev/null || true          # representative end-semester run logs (untracked, small)
  if [ -d report ]; then find report -type f ! -name '.*'; fi
} | sort -u > "$STAGE/files.txt"

tar -cf - -T "$STAGE/files.txt" | tar -xf - -C "$STAGE/$NAME"
rm -f "$OUT"
(cd "$STAGE" && zip -qr -9 "$OUT" "$NAME" -x '*/__pycache__/*' '*.pyc' '*/.DS_Store')

echo "wrote $OUT ($(du -h "$OUT" | cut -f1), $(wc -l < "$STAGE/files.txt") files)"
if [ ! -d report ] || [ -z "$(find report -type f ! -name '.*' 2>/dev/null)" ]; then
  echo "WARNING: report/ is empty - add the report PDF and slides, then run this again."
fi
