#!/bin/bash
cd /home/featurize/work/tmc/scripts/e2e_agent

nohup python scripts/run_verify_critical_pipeline.py \
    --dataset hotpotqa \
    --data-path data/hotpotqa/hotpot_dev_distractor_v1.json \
    --corpus-path data/hotpotqa/corpus.jsonl \
    --direction-path steering/directions/direction_search_v3.npz \
    --type-filter bridge \
    --tau-sweep 0.0 \
    --max-rho-sweep 0.25 0.5 1.0 1.5 2.0 \
    --out results/hotpotqa_v3_full \
    > results/hotpotqa_v3_full.log 2>&1 &

echo "PID: $!"

