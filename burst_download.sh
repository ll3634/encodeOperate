#!/usr/bin/env bash
# burst_download.sh  — restart-loop downloader for Qwen3-32B
# Each ModelScope session gives a ~23 MB/s burst for ~30-60 s, then throttles.
# We kill and restart every BURST_SEC seconds to keep riding fresh bursts.
# Compatible with ModelScope's temp-dir resume: already-staged shards are skipped.

MODEL_DIR="/home/featurize/work/models/Qwen3-32B"
DLOG="/home/featurize/work/tmc/qwen3_download.log"
BURST_SEC=45      # restart interval (seconds)
TARGET=17         # number of shards expected

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$DLOG"; }

log "=== burst_download.sh started (restart every ${BURST_SEC}s) ==="

while true; do
    # Count final shards
    N=$(ls "$MODEL_DIR"/model-*.safetensors 2>/dev/null | wc -l)
    T=$(ls "$MODEL_DIR/._____temp/" 2>/dev/null | grep -c safetensors || true)
    log "final=$N/$TARGET  temp=$T"

    if [ "$N" -ge "$TARGET" ]; then
        log "All $TARGET shards present — download complete!"
        exit 0
    fi

    # Measure net speed before burst
    R1=$(awk '/enp1s0/{print $2}' /proc/net/dev)

    # Start one download session
    python3 -c "
from modelscope import snapshot_download
import sys, time
t0 = time.time()
try:
    snapshot_download(
        model_id='Qwen/Qwen3-32B',
        local_dir='$MODEL_DIR',
    )
    print(f'snapshot_download finished in {(time.time()-t0)/60:.1f} min', flush=True)
except Exception as e:
    print(f'session ended: {e}', flush=True)
" >> "$DLOG" 2>&1 &
    DL_PID=$!
    log "burst session PID=$DL_PID"

    # Let it run for BURST_SEC, then kill
    sleep "$BURST_SEC"
    kill -9 "$DL_PID" 2>/dev/null
    sleep 2
    # Force-kill any leftover zombie python holding the same model dir
    pkill -9 -f "snapshot_download" 2>/dev/null || true
    sleep 1

    # Measure achieved speed
    R2=$(awk '/enp1s0/{print $2}' /proc/net/dev)
    KBPS=$(( (R2 - R1) / 1024 / BURST_SEC ))
    log "burst ended — avg ${KBPS} KB/s over ${BURST_SEC}s"
done
