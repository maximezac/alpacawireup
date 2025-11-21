#!/usr/bin/env bash
# scripts/ci/list_artifacts_for_upload.sh
# List files under candidate upload paths and write them to a text file
# Usage: ./scripts/ci/list_artifacts_for_upload.sh [OUT_FILE]

set -euo pipefail

OUT_FILE="${1:-artifacts/upload_included_files.txt}"

# Paths we plan to upload (adjust as needed)
PATHS_TO_CHECK=("artifacts" ".tmp" "data/portfolios")

rm -f "$OUT_FILE"
mkdir -p "$(dirname "$OUT_FILE")"

echo "Listing artifact candidate files" > "$OUT_FILE"
for P in "${PATHS_TO_CHECK[@]}"; do
  echo "---- $P ----" >> "$OUT_FILE"
  if [ -d "$P" ]; then
    # show relative paths
    find "$P" -type f | sed 's#^./##' >> "$OUT_FILE" || true
  else
    echo "(missing)" >> "$OUT_FILE"
  fi
done

echo "Wrote $OUT_FILE"
exit 0
