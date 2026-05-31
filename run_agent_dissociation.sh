#!/bin/bash
# Agent-Specific Dissociation Test runner
# Run from the repository root (the script cd's into its own directory).

set -e
cd "$(dirname "$0")"
mkdir -p results/agent_specific_dissociation

# Ensure GPU is visible (override empty CUDA_VISIBLE_DEVICES if set)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"

echo "=== Agent-Specific Dissociation Test ==="
echo "Model: $MODEL"
echo ""

echo "--- FULL RUN (N=486) ---"
python3 -u scripts/agent_specific_dissociation.py \
    --labels-path results/phase1_probe/labels.jsonl \
    --baseline-path results/l20_rho020_n500/baseline_results.jsonl \
    --output-dir results/agent_specific_dissociation \
    --model "$MODEL" \
    2>&1 | tee results/agent_specific_dissociation/run_log.txt