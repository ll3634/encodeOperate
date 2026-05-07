#!/bin/bash
# Positive Control Validity Check for Condition A and A_v2
# Tests whether the judge conditions exhibit response bias or genuine evidence evaluation.
#
# Runtime: ~10-20 min (104 P1 + 5 P2 + 10 SYNTH = ~120 forward passes)
set -e
cd "$(dirname "$0")"

mkdir -p results/positive_control logs

LOG="logs/positive_control_$(date +%Y%m%d_%H%M%S).log"
echo "=== Positive Control Validity Check ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"

python3 scripts/positive_control_judge.py \
    --labels results/phase1_probe/labels.jsonl \
    --baseline results/l20_rho020_n500/baseline_results.jsonl \
    --hotpotqa data/hotpotqa/hotpot_dev_distractor_v1.json \
    --output-dir results/positive_control \
    --model Qwen/Qwen2.5-7B-Instruct \
    --n-synth 10 \
    2>&1 | tee -a "$LOG"

echo "Finished: $(date)" | tee -a "$LOG"
echo "Results: results/positive_control/positive_control_results.json"
