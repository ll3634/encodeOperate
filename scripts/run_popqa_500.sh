#!/bin/bash
# Run PopQA experiment with 500 samples to capture more Red Flag samples
# Goal: Get statistically significant Red Flag data (expect ~50-60 Red Flag samples)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Configuration
DIRECTION_PATH="${DIRECTION_PATH:-$(pwd)/steering/directions/direction_search_v3.npz}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
N_SAMPLES=500
SEED=42
RESULTS_DIR="results/popqa_500"
POP_LIMIT=100  # Use hard samples (low popularity)

mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "PopQA 500-Sample Experiment"
echo "========================================"
echo "Direction: $DIRECTION_PATH"
echo "Model: $MODEL"
echo "N samples: $N_SAMPLES"
echo "Pop limit: $POP_LIMIT (hard samples)"
echo "Seed: $SEED"
echo "Results: $RESULTS_DIR"
echo "========================================"

# Check data files
if [ ! -f "data/popqa/popqa_test.jsonl" ]; then
    echo "ERROR: PopQA data not found at data/popqa/popqa_test.jsonl"
    echo "Please download and convert from akariasai/PopQA"
    exit 1
fi

# Build corpus if needed
if [ ! -f "data/popqa/corpus.jsonl" ]; then
    echo "Building PopQA corpus..."
    python -c "from datasets.popqa import build_popqa_corpus; build_popqa_corpus('data/popqa/popqa_test.jsonl', 'data/popqa/corpus.jsonl')"
fi

echo ""
echo "=== Running Baseline (500 samples) ==="
python runners/run_popqa.py \
    --data-path data/popqa/popqa_test.jsonl \
    --corpus-path data/popqa/corpus.jsonl \
    --direction-path "$DIRECTION_PATH" \
    --output "$RESULTS_DIR/baseline_500.jsonl" \
    --model "$MODEL" \
    --policy baseline \
    --n-samples "$N_SAMPLES" \
    --pop-limit "$POP_LIMIT" \
    --seed "$SEED"

echo ""
echo "=== Running JES (500 samples) ==="
python runners/run_popqa.py \
    --data-path data/popqa/popqa_test.jsonl \
    --corpus-path data/popqa/corpus.jsonl \
    --direction-path "$DIRECTION_PATH" \
    --output "$RESULTS_DIR/jes_500.jsonl" \
    --model "$MODEL" \
    --policy jes \
    --n-samples "$N_SAMPLES" \
    --pop-limit "$POP_LIMIT" \
    --seed "$SEED" \
    --tau 0.2 \
    --rho-max 0.25

echo ""
echo "=== Running Force Adopt (500 samples) ==="
python runners/run_popqa.py \
    --data-path data/popqa/popqa_test.jsonl \
    --corpus-path data/popqa/corpus.jsonl \
    --direction-path "$DIRECTION_PATH" \
    --output "$RESULTS_DIR/force_adopt_500.jsonl" \
    --model "$MODEL" \
    --policy force_adopt \
    --n-samples "$N_SAMPLES" \
    --pop-limit "$POP_LIMIT" \
    --seed "$SEED"

echo ""
echo "=== Running Force Reject (500 samples) ==="
python runners/run_popqa.py \
    --data-path data/popqa/popqa_test.jsonl \
    --corpus-path data/popqa/corpus.jsonl \
    --direction-path "$DIRECTION_PATH" \
    --output "$RESULTS_DIR/force_reject_500.jsonl" \
    --model "$MODEL" \
    --policy force_reject \
    --n-samples "$N_SAMPLES" \
    --pop-limit "$POP_LIMIT" \
    --seed "$SEED"

echo ""
echo "=== Analyzing Subsets (Stealth/Red Flag/Indifferent) ==="
python analysis/analyze_subsets.py \
    --baseline "$RESULTS_DIR/baseline_500.jsonl" \
    --jes "$RESULTS_DIR/jes_500.jsonl" \
    --force-adopt "$RESULTS_DIR/force_adopt_500.jsonl" \
    --force-reject "$RESULTS_DIR/force_reject_500.jsonl" \
    --output "$RESULTS_DIR/subset_analysis.json"

echo ""
echo "=== PopQA 500 Complete ==="
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Expected Red Flag samples: ~50-60 (11% of 500)"
echo "Expected Stealth samples: ~20 (4% of 500)"
echo ""
echo "Next steps:"
echo "1. Check subset_analysis.json for Red Flag count"
echo "2. If Red Flag >= 30, analyze JES protection rate"
echo "3. If Red Flag < 30, consider expanding to 1000 samples"

