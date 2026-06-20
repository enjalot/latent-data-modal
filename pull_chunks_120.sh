#!/usr/bin/env bash
# Pull the MiniLM-compatible chunked-120 TEXT parquets from Modal -> /data/chunks.
# These are UNIQUE to Modal (only the .npy embeddings are local) and are needed
# to label the MiniLM Stage-J SAE. Priority order: fineweb (unblocks labeling),
# then redpajama, then pile. Per-FILE get (the proven pattern from pull_chunks.sh);
# idempotent — skips files already on disk with nonzero size.
#
# Layout: /data/chunks/<dataset>-chunked-120/train/*.parquet
set -uo pipefail
CHUNK_DIR="${CHUNK_DIR:-/data/chunks}"

# dataset | modal_volume   (priority order: fineweb first)
DATASETS=(
  "fineweb-edu-sample-10BT-chunked-120|embedding-fineweb-edu"
  "RedPajama-Data-V2-sample-10B-chunked-120|datasets"
  "pile-uncopyrighted-chunked-120|datasets"
)

pull_file() {  # vol remote local_dir
  local vol="$1" remote="$2" local_dir="$3"
  local local_path="$local_dir/$(basename "$remote")"
  mkdir -p "$local_dir"
  [[ -s "$local_path" ]] && return 0
  modal volume get "$vol" "$remote" "$local_path"
}

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r ds vol <<< "$entry"
  echo "[$(date '+%H:%M:%S')] === $ds (vol=$vol) ==="
  files=$(modal volume ls --json "$vol" "${ds}/train" 2>/dev/null | python3 -c "
import json,sys
for e in json.load(sys.stdin):
    fn=e.get('Filename','')
    if fn.endswith('.parquet'): print(fn)
")
  n=$(echo "$files" | grep -c . || true)
  echo "  $n parquet files to pull"
  i=0
  while IFS= read -r fn; do
    [[ -z "$fn" ]] && continue
    i=$((i+1))
    pull_file "$vol" "$fn" "$CHUNK_DIR/$ds/train" || { echo "  FAILED: $fn"; exit 1; }
    [[ $((i % 20)) -eq 0 ]] && echo "  [$(date '+%H:%M:%S')] $i/$n  ($(du -sh "$CHUNK_DIR/$ds" 2>/dev/null | cut -f1))"
  done <<< "$files"
  echo "  [$(date '+%H:%M:%S')] $ds DONE: $(du -sh "$CHUNK_DIR/$ds" 2>/dev/null | cut -f1)"
done
echo "[$(date '+%H:%M:%S')] === ALL DONE ==="
du -sh "$CHUNK_DIR"/*-chunked-120 2>/dev/null
