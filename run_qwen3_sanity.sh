#!/usr/bin/env bash
# Auto-launch qwen3_circuit_sanity.py when the local Qwen3-32B snapshot
# finishes downloading. Override MODEL_DIR / LOG via environment variables
# if the model lives elsewhere.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/models/Qwen3-32B}"
LOG="${LOG:-$SCRIPT_DIR/logs/qwen3_sanity.log}"
mkdir -p "$(dirname "$LOG")"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "Waiting for model download to complete..."

# Wait until all 17 safetensors shards are present
while true; do
    N=$(ls "$MODEL_DIR"/model-*.safetensors 2>/dev/null | wc -l || echo 0)
    log "safetensors present: $N/17"
    if [ "$N" -ge 17 ]; then
        log "All shards present. Launching experiment."
        break
    fi
    # Check temp dir progress (ModelScope stages to temp before moving to final)
    T=$(ls "$MODEL_DIR/._____temp/" 2>/dev/null | wc -l || echo 0)
    log "  temp shards: $T  final: $N/17"
    sleep 60
done

echo "[$(date)] Launching qwen3_circuit_sanity.py ..." | tee -a "$LOG"
cd "$SCRIPT_DIR"
python3 scripts/qwen3_circuit_sanity.py \
    --model-path "$MODEL_DIR" \
    --dtype bfloat16 \
    --n-popqa 300 \
    --limit 50 \
    --rhos "0.10,0.20" \
    --out-dir "results/cross_model_qwen3_32b/circuit_sanity" \
    2>&1 | tee -a "$LOG"

echo "[$(date)] Done." | tee -a "$LOG"
