#!/bin/bash
# Run PopQA experiment with 100 samples after prompt fix
# Goal: Verify that JES now works correctly with aligned prompts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Configuration
DIRECTION_PATH="${DIRECTION_PATH:-$(pwd)/steering/directions/direction_search_v3.npz}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
N_SAMPLES=100
SEED=42
RESULTS_DIR="results/popqa_100_fixed"
POP_LIMIT=100  # Use hard samples (low popularity)

mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "PopQA 100-Sample Experiment (Fixed Prompts)"
echo "========================================"
echo "Direction: $DIRECTION_PATH"
echo "Model: $MODEL"
echo "N samples: $N_SAMPLES"
echo "Pop limit: $POP_LIMIT (hard samples)"
echo "Seed: $SEED"
echo "Results: $RESULTS_DIR"
echo "========================================"

echo ""
echo "=== Running Baseline (100 samples) ==="
python runners/run_popqa.py \
    --data-path data/popqa/popqa_test.jsonl \
    --corpus-path data/popqa/corpus.jsonl \
    --direction-path "$DIRECTION_PATH" \
    --output "$RESULTS_DIR/baseline_100.jsonl" \
    --model "$MODEL" \
    --policy baseline \
    --n-samples "$N_SAMPLES" \
    --pop-limit "$POP_LIMIT" \
    --seed "$SEED"

echo ""
echo "=== Running JES (100 samples) ==="
python runners/run_popqa.py \
    --data-path data/popqa/popqa_test.jsonl \
    --corpus-path data/popqa/corpus.jsonl \
    --direction-path "$DIRECTION_PATH" \
    --output "$RESULTS_DIR/jes_100.jsonl" \
    --model "$MODEL" \
    --policy jes \
    --n-samples "$N_SAMPLES" \
    --pop-limit "$POP_LIMIT" \
    --seed "$SEED" \
    --tau 0.2 \
    --rho-max 0.25

echo ""
echo "=== Running Force Adopt (100 samples) ==="
python runners/run_popqa.py \
    --data-path data/popqa/popqa_test.jsonl \
    --corpus-path data/popqa/corpus.jsonl \
    --direction-path "$DIRECTION_PATH" \
    --output "$RESULTS_DIR/force_adopt_100.jsonl" \
    --model "$MODEL" \
    --policy force_adopt \
    --n-samples "$N_SAMPLES" \
    --pop-limit "$POP_LIMIT" \
    --seed "$SEED"

echo ""
echo "=== Running Force Reject (100 samples) ==="
python runners/run_popqa.py \
    --data-path data/popqa/popqa_test.jsonl \
    --corpus-path data/popqa/corpus.jsonl \
    --direction-path "$DIRECTION_PATH" \
    --output "$RESULTS_DIR/force_reject_100.jsonl" \
    --model "$MODEL" \
    --policy force_reject \
    --n-samples "$N_SAMPLES" \
    --pop-limit "$POP_LIMIT" \
    --seed "$SEED"

echo ""
echo "=== Analyzing Subsets ==="
python analysis/analyze_subsets.py \
    --baseline "$RESULTS_DIR/baseline_100.jsonl" \
    --jes "$RESULTS_DIR/jes_100.jsonl" \
    --force-adopt "$RESULTS_DIR/force_adopt_100.jsonl" \
    --force-reject "$RESULTS_DIR/force_reject_100.jsonl" \
    --output "$RESULTS_DIR/subset_analysis.json"

echo ""
echo "=== PopQA 100 (Fixed) Complete ==="
cat "$RESULTS_DIR/subset_analysis.json"
