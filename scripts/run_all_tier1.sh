#!/bin/bash
# Run all Tier 1 diagnostic experiments
# Tests: HotpotQA (under-reliance) and TruthfulQA (over-reliance)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Configuration
DIRECTION_PATH="${DIRECTION_PATH:-oral_experiment_v7/direction_boundary.npz}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
N_SAMPLES="${N_SAMPLES:-20}"
SEED="${SEED:-42}"
RESULTS_DIR="${RESULTS_DIR:-results/tier1}"

mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "E2E Agent Tier 1 Experiments"
echo "========================================"
echo "Direction: $DIRECTION_PATH"
echo "Model: $MODEL"
echo "N samples: $N_SAMPLES"
echo "Seed: $SEED"
echo "Results: $RESULTS_DIR"
echo "========================================"

# Check data files
if [ ! -f "data/hotpotqa/hotpot_dev_distractor_v1.json" ]; then
    echo "ERROR: HotpotQA data not found at data/hotpotqa/hotpot_dev_distractor_v1.json"
    echo "Please download from https://hotpotqa.github.io/"
    exit 1
fi

# Build corpus if needed
if [ ! -f "data/hotpotqa/corpus.jsonl" ]; then
    echo "Building HotpotQA corpus..."
    python -c "from datasets.hotpotqa import build_hotpotqa_corpus; build_hotpotqa_corpus('data/hotpotqa/hotpot_dev_distractor_v1.json', 'data/hotpotqa/corpus.jsonl')"
fi

echo ""
echo "=== HotpotQA: Under-reliance (should adopt) ==="

# Baseline
echo "Running baseline..."
python runners/run_hotpotqa.py \
    --data-path data/hotpotqa/hotpot_dev_distractor_v1.json \
    --corpus-path data/hotpotqa/corpus.jsonl \
    --direction-path "$DIRECTION_PATH" \
    --output "$RESULTS_DIR/hotpotqa_baseline.jsonl" \
    --model "$MODEL" \
    --policy baseline \
    --n-samples "$N_SAMPLES" \
    --seed "$SEED"

# JES
echo "Running JES..."
python runners/run_hotpotqa.py \
    --data-path data/hotpotqa/hotpot_dev_distractor_v1.json \
    --corpus-path data/hotpotqa/corpus.jsonl \
    --direction-path "$DIRECTION_PATH" \
    --output "$RESULTS_DIR/hotpotqa_jes.jsonl" \
    --model "$MODEL" \
    --policy jes \
    --n-samples "$N_SAMPLES" \
    --seed "$SEED" \
    --tau 0.2 --rho-max 0.25

echo ""
echo "=== TruthfulQA: Over-reliance (should reject) ==="

if [ -f "data/truthfulqa/TruthfulQA.csv" ]; then
    # Baseline
    echo "Running baseline..."
    python runners/run_truthfulqa.py \
        --data-path data/truthfulqa/TruthfulQA.csv \
        --direction-path "$DIRECTION_PATH" \
        --output "$RESULTS_DIR/truthfulqa_baseline.jsonl" \
        --model "$MODEL" \
        --policy baseline \
        --n-samples "$N_SAMPLES" \
        --seed "$SEED"
    
    # JES
    echo "Running JES..."
    python runners/run_truthfulqa.py \
        --data-path data/truthfulqa/TruthfulQA.csv \
        --direction-path "$DIRECTION_PATH" \
        --output "$RESULTS_DIR/truthfulqa_jes.jsonl" \
        --model "$MODEL" \
        --policy jes \
        --n-samples "$N_SAMPLES" \
        --seed "$SEED"
else
    echo "WARNING: TruthfulQA data not found. Skipping."
fi

echo ""
echo "=== Summarizing Results ==="
python analysis/summarize.py --input-dir "$RESULTS_DIR" --output "$RESULTS_DIR/summary.json" --compare

echo ""
echo "=== Tier 1 Complete ==="
echo "Results saved to: $RESULTS_DIR"

