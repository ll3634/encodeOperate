#!/bin/bash
# Evidence Signal Erosion During Thought Generation
# Extracts L20 activations at 5 positions during thought generation for 486 samples.
#
# Runtime: ~2-3 hours (486 samples × 2 model calls each)
# Resume: checkpoints every 50 samples to raw_erosion_data.npz
set -e
cd "$(dirname "$0")"

mkdir -p results/thought_erosion logs

MODE="${1:-full}"   # "dry" for quick test, "full" for all samples

if [ "$MODE" = "dry" ]; then
    DRY="--dry-run"
    echo "=== Thought Erosion Probe [DRY RUN] ==="
else
    DRY=""
    echo "=== Thought Erosion Probe [FULL N=486] ==="
fi

LOG="logs/thought_erosion_$(date +%Y%m%d_%H%M%S).log"
echo "Started: $(date)" | tee "$LOG"

python3 scripts/thought_erosion_probe.py \
    --labels results/phase1_probe/labels.jsonl \
    --baseline results/l20_rho020_n500/baseline_results.jsonl \
    --probe-direction results/phase1_probe/probe_direction_l20.npz \
    --output-dir results/thought_erosion \
    --model Qwen/Qwen2.5-7B-Instruct \
    --layer 20 \
    $DRY \
    2>&1 | tee -a "$LOG"

echo "Finished: $(date)" | tee -a "$LOG"
echo "Key outputs:"
echo "  results/thought_erosion/erosion_results.json"
echo "  results/thought_erosion/erosion_curves.png"
