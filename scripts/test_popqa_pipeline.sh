#!/bin/bash
# Test PopQA pipeline with 10 samples before running full 500-sample experiment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Configuration
DIRECTION_PATH="$(pwd)/steering/directions/direction_search_v3.npz"
MODEL="Qwen/Qwen2.5-7B-Instruct"
N_SAMPLES=10
SEED=42
RESULTS_DIR="results/popqa_test"
POP_LIMIT=100

mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "PopQA Pipeline Test (10 samples)"
echo "========================================"
echo "Direction: $DIRECTION_PATH"
echo "Model: $MODEL"
echo "N samples: $N_SAMPLES"
echo "Results: $RESULTS_DIR"
echo "========================================"

# Check files
if [ ! -f "data/popqa/popqa_test.jsonl" ]; then
    echo "ERROR: PopQA data not found"
    exit 1
fi

if [ ! -f "$DIRECTION_PATH" ]; then
    echo "ERROR: Direction file not found: $DIRECTION_PATH"
    exit 1
fi

echo ""
echo "=== Running Baseline ==="
python runners/run_popqa.py \
    --data-path data/popqa/popqa_test.jsonl \
    --corpus-path data/popqa/corpus.jsonl \
    --direction-path "$DIRECTION_PATH" \
    --output "$RESULTS_DIR/baseline_10.jsonl" \
    --model "$MODEL" \
    --policy baseline \
    --n-samples "$N_SAMPLES" \
    --pop-limit "$POP_LIMIT" \
    --seed "$SEED"

echo ""
echo "=== Running JES ==="
python runners/run_popqa.py \
    --data-path data/popqa/popqa_test.jsonl \
    --corpus-path data/popqa/corpus.jsonl \
    --direction-path "$DIRECTION_PATH" \
    --output "$RESULTS_DIR/jes_10.jsonl" \
    --model "$MODEL" \
    --policy jes \
    --n-samples "$N_SAMPLES" \
    --pop-limit "$POP_LIMIT" \
    --seed "$SEED" \
    --tau 0.2 \
    --rho-max 0.25

echo ""
echo "=== Running Force Adopt ==="
python runners/run_popqa.py \
    --data-path data/popqa/popqa_test.jsonl \
    --corpus-path data/popqa/corpus.jsonl \
    --direction-path "$DIRECTION_PATH" \
    --output "$RESULTS_DIR/force_adopt_10.jsonl" \
    --model "$MODEL" \
    --policy force_adopt \
    --n-samples "$N_SAMPLES" \
    --pop-limit "$POP_LIMIT" \
    --seed "$SEED"

echo ""
echo "=== Running Force Reject ==="
python runners/run_popqa.py \
    --data-path data/popqa/popqa_test.jsonl \
    --corpus-path data/popqa/corpus.jsonl \
    --direction-path "$DIRECTION_PATH" \
    --output "$RESULTS_DIR/force_reject_10.jsonl" \
    --model "$MODEL" \
    --policy force_reject \
    --n-samples "$N_SAMPLES" \
    --pop-limit "$POP_LIMIT" \
    --seed "$SEED"

echo ""
echo "=== Analyzing Subsets ==="
python analysis/analyze_subsets.py \
    --baseline "$RESULTS_DIR/baseline_10.jsonl" \
    --jes "$RESULTS_DIR/jes_10.jsonl" \
    --force-adopt "$RESULTS_DIR/force_adopt_10.jsonl" \
    --force-reject "$RESULTS_DIR/force_reject_10.jsonl" \
    --output "$RESULTS_DIR/subset_analysis.json"

echo ""
echo "=== Test Complete ==="
cat "$RESULTS_DIR/subset_analysis.json"

