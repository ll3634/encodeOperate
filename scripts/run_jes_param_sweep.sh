#!/bin/bash
# JES Parameter Sweep: Test different rho_max and tau values
# Goal: Find optimal parameters to improve total success rate

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Configuration
DIRECTION_PATH="${DIRECTION_PATH:-$(pwd)/steering/directions/direction_search_v3.npz}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
N_SAMPLES=500
SEED=42
RESULTS_DIR="results/popqa_500_param_sweep"
POP_LIMIT=100

mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "JES Parameter Sweep Experiment"
echo "========================================"
echo "Direction: $DIRECTION_PATH"
echo "Model: $MODEL"
echo "N samples: $N_SAMPLES"
echo "Results: $RESULTS_DIR"
echo "========================================"

# Parameter combinations to test
# Current: tau=0.2, rho_max=0.25 (74.2% saturated)
# Test: higher rho_max values

echo ""
echo "=== Test 1: rho_max=0.5, tau=0.2 ==="
python runners/run_popqa.py \
    --data-path data/popqa/popqa_test.jsonl \
    --corpus-path data/popqa/corpus.jsonl \
    --direction-path "$DIRECTION_PATH" \
    --output "$RESULTS_DIR/jes_rho0.5_tau0.2.jsonl" \
    --model "$MODEL" \
    --policy jes \
    --n-samples "$N_SAMPLES" \
    --pop-limit "$POP_LIMIT" \
    --seed "$SEED" \
    --tau 0.2 \
    --rho-max 0.5

echo ""
echo "=== Test 2: rho_max=0.5, tau=0.5 ==="
python runners/run_popqa.py \
    --data-path data/popqa/popqa_test.jsonl \
    --corpus-path data/popqa/corpus.jsonl \
    --direction-path "$DIRECTION_PATH" \
    --output "$RESULTS_DIR/jes_rho0.5_tau0.5.jsonl" \
    --model "$MODEL" \
    --policy jes \
    --n-samples "$N_SAMPLES" \
    --pop-limit "$POP_LIMIT" \
    --seed "$SEED" \
    --tau 0.5 \
    --rho-max 0.5

echo ""
echo "=== Test 3: rho_max=0.75, tau=0.2 ==="
python runners/run_popqa.py \
    --data-path data/popqa/popqa_test.jsonl \
    --corpus-path data/popqa/corpus.jsonl \
    --direction-path "$DIRECTION_PATH" \
    --output "$RESULTS_DIR/jes_rho0.75_tau0.2.jsonl" \
    --model "$MODEL" \
    --policy jes \
    --n-samples "$N_SAMPLES" \
    --pop-limit "$POP_LIMIT" \
    --seed "$SEED" \
    --tau 0.2 \
    --rho-max 0.75

echo ""
echo "=== Parameter Sweep Complete ==="
echo "Results saved to: $RESULTS_DIR"

