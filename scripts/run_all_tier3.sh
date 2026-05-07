#!/bin/bash
# Run all Tier 3 attribution experiments
# Tests: Corruption sweeps, forced baselines, random controls

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Configuration
DIRECTION_PATH="${DIRECTION_PATH:-oral_experiment_v7/direction_boundary.npz}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
N_SAMPLES="${N_SAMPLES:-20}"
SEED="${SEED:-42}"
RESULTS_DIR="${RESULTS_DIR:-results/tier3}"

mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "E2E Agent Tier 3 Attribution Experiments"
echo "========================================"
echo "Direction: $DIRECTION_PATH"
echo "Model: $MODEL"
echo "N samples: $N_SAMPLES"
echo "Seed: $SEED"
echo "Results: $RESULTS_DIR"
echo "========================================"

# Check data files
if [ ! -f "data/hotpotqa/hotpot_dev_distractor_v1.json" ]; then
    echo "ERROR: HotpotQA data not found."
    exit 1
fi

echo ""
echo "=== Corruption Sweep ==="
echo "Testing JES robustness to tool corruption..."

python runners/run_controls.py \
    --experiment corruption_sweep \
    --data-path data/hotpotqa/hotpot_dev_distractor_v1.json \
    --corpus-path data/hotpotqa/corpus.jsonl \
    --direction-path "$DIRECTION_PATH" \
    --output "$RESULTS_DIR/corruption_sweep.json" \
    --model "$MODEL" \
    --n-samples "$N_SAMPLES" \
    --seed "$SEED"

echo ""
echo "=== Random Control ==="
echo "Comparing decision direction vs random orthogonal..."

python runners/run_controls.py \
    --experiment random_control \
    --data-path data/hotpotqa/hotpot_dev_distractor_v1.json \
    --corpus-path data/hotpotqa/corpus.jsonl \
    --direction-path "$DIRECTION_PATH" \
    --output "$RESULTS_DIR/random_control.json" \
    --model "$MODEL" \
    --n-samples "$N_SAMPLES" \
    --seed "$SEED"

echo ""
echo "=== Generating Corruption Curve ==="
python analysis/plot_corruption_curves.py \
    --input "$RESULTS_DIR/corruption_sweep.json" \
    --output "$RESULTS_DIR/corruption_curve.png" \
    --title "JES Robustness to Tool Corruption"

echo ""
echo "=== Tier 3 Complete ==="
echo "Results saved to: $RESULTS_DIR"

echo ""
echo "=== Summary ==="
echo ""
echo "Corruption Sweep Results:"
cat "$RESULTS_DIR/corruption_sweep.json" | python -m json.tool | head -30

echo ""
echo "Random Control Results:"
cat "$RESULTS_DIR/random_control.json" | python -m json.tool | head -30

