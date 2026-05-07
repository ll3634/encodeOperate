#!/bin/bash
# Run all Tier 2 application experiments
# Tests: GAIA conflict subset (mixed adopt/reject scenarios)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Configuration
DIRECTION_PATH="${DIRECTION_PATH:-oral_experiment_v7/direction_boundary.npz}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
N_SAMPLES="${N_SAMPLES:-50}"
SEED="${SEED:-42}"
RESULTS_DIR="${RESULTS_DIR:-results/tier2}"

mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "E2E Agent Tier 2 Experiments"
echo "========================================"
echo "Direction: $DIRECTION_PATH"
echo "Model: $MODEL"
echo "N samples: $N_SAMPLES"
echo "Seed: $SEED"
echo "Results: $RESULTS_DIR"
echo "========================================"

# Check data files
if [ ! -f "data/gaia/gaia_subset.jsonl" ]; then
    echo "ERROR: GAIA data not found at data/gaia/gaia_subset.jsonl"
    echo "Please prepare the GAIA subset. See README.md for instructions."
    exit 1
fi

echo ""
echo "=== GAIA Conflict Subset ==="

# Baseline
echo "Running baseline..."
python runners/run_gaia_subset.py \
    --data-path data/gaia/gaia_subset.jsonl \
    --direction-path "$DIRECTION_PATH" \
    --output "$RESULTS_DIR/gaia_baseline.jsonl" \
    --model "$MODEL" \
    --policy baseline \
    --n-samples "$N_SAMPLES" \
    --seed "$SEED" \
    --conflict-only

# JES
echo "Running JES..."
python runners/run_gaia_subset.py \
    --data-path data/gaia/gaia_subset.jsonl \
    --direction-path "$DIRECTION_PATH" \
    --output "$RESULTS_DIR/gaia_jes.jsonl" \
    --model "$MODEL" \
    --policy jes \
    --n-samples "$N_SAMPLES" \
    --seed "$SEED" \
    --conflict-only \
    --tau 0.2 --rho-max 0.25

# Force adopt
echo "Running force_adopt..."
python runners/run_gaia_subset.py \
    --data-path data/gaia/gaia_subset.jsonl \
    --direction-path "$DIRECTION_PATH" \
    --output "$RESULTS_DIR/gaia_force_adopt.jsonl" \
    --model "$MODEL" \
    --policy force_adopt \
    --n-samples "$N_SAMPLES" \
    --seed "$SEED" \
    --conflict-only

# Force reject
echo "Running force_reject..."
python runners/run_gaia_subset.py \
    --data-path data/gaia/gaia_subset.jsonl \
    --direction-path "$DIRECTION_PATH" \
    --output "$RESULTS_DIR/gaia_force_reject.jsonl" \
    --model "$MODEL" \
    --policy force_reject \
    --n-samples "$N_SAMPLES" \
    --seed "$SEED" \
    --conflict-only

echo ""
echo "=== Summarizing Results ==="
python analysis/summarize.py --input-dir "$RESULTS_DIR" --output "$RESULTS_DIR/summary.json" --compare

echo ""
echo "=== Generating Pareto Plot ==="
python analysis/plot_pareto.py \
    --input "$RESULTS_DIR/summary.json" \
    --output "$RESULTS_DIR/pareto.png" \
    --title "GAIA: Success vs Cost Pareto"

echo ""
echo "=== Tier 2 Complete ==="
echo "Results saved to: $RESULTS_DIR"

