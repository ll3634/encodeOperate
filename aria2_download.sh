#!/usr/bin/env bash
# aria2_download.sh — fast parallel download for Qwen3-32B via ModelScope CDN
# Skips shards already present in MODEL_DIR.
# Usage: bash aria2_download.sh

MODEL_DIR="/home/featurize/work/models/Qwen3-32B"
BASE_URL="https://modelscope.cn/models/Qwen/Qwen3-32B/resolve/master"
ARIA2_DIR="/home/featurize/work/tmc/aria2_tmp"
DLOG="/home/featurize/work/tmc/qwen3_download.log"

mkdir -p "$ARIA2_DIR"
mkdir -p "$MODEL_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$DLOG"; }
log "=== aria2_download.sh started ==="

# Config/tokenizer files (small, always grab)
SMALL_FILES=(
  "config.json"
  "generation_config.json"
  "tokenizer.json"
  "tokenizer_config.json"
  "vocab.json"
  "merges.txt"
  "model.safetensors.index.json"
)

# Build aria2 input file for small files
SMALL_LIST="$ARIA2_DIR/small_files.txt"
> "$SMALL_LIST"
for f in "${SMALL_FILES[@]}"; do
  echo "${BASE_URL}/${f}" >> "$SMALL_LIST"
  echo "  dir=$MODEL_DIR" >> "$SMALL_LIST"
  echo "  out=${f}" >> "$SMALL_LIST"
  echo "  continue=true" >> "$SMALL_LIST"
done

# Download small files first
log "Downloading config/tokenizer files..."
aria2c -i "$SMALL_LIST" -x4 -s4 --auto-file-renaming=false \
  --log="$ARIA2_DIR/aria2_small.log" --log-level=warn 2>&1 | tail -5

# Build shard download list — skip already-complete shards
SHARD_LIST="$ARIA2_DIR/shards.txt"
> "$SHARD_LIST"
NEED=0
for i in $(seq -f "%05g" 1 17); do
  FNAME="model-${i}-of-00017.safetensors"
  DEST="$MODEL_DIR/$FNAME"
  if [ -f "$DEST" ] && [ "$(stat -c%s "$DEST" 2>/dev/null || echo 0)" -gt 3000000000 ]; then
    log "  skip $FNAME (already complete)"
    continue
  fi
  NEED=$((NEED + 1))
  echo "${BASE_URL}/${FNAME}" >> "$SHARD_LIST"
  echo "  dir=$MODEL_DIR" >> "$SHARD_LIST"
  echo "  out=${FNAME}" >> "$SHARD_LIST"
  echo "  continue=true" >> "$SHARD_LIST"
done

log "Shards needed: $NEED / 17"

if [ "$NEED" -eq 0 ]; then
  log "All shards already present!"
  exit 0
fi

# Fire aria2c with aggressive settings for Chinese CDN
log "Starting aria2c for $NEED shards..."
aria2c -i "$SHARD_LIST" \
  -x 16 -s 16 -j 4 \
  --continue=true \
  --auto-file-renaming=false \
  --file-allocation=none \
  --max-tries=20 \
  --retry-wait=5 \
  --timeout=60 \
  --connect-timeout=20 \
  --max-concurrent-downloads=4 \
  --log="$ARIA2_DIR/aria2_shards.log" \
  --log-level=notice \
  2>&1 | tee -a "$DLOG"

log "aria2c finished. Final check:"
ls "$MODEL_DIR"/model-*.safetensors 2>/dev/null | wc -l
