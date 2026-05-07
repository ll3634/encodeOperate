#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

run() {
    local mid=$1; local out=$2
    echo "=================================================="
    echo "$(date +%H:%M:%S)  Starting: $mid"
    echo "=================================================="
    python -u scripts/cross_model_layer_trajectory.py \
        --model "$mid" \
        --output-dir "results/cross_layer_cos/$out" 2>&1 | tee "results/cross_layer_cos/${out}.log"
}

# qwen25 already done; uncomment to redo
# run "Qwen/Qwen2.5-7B-Instruct"           qwen25
# Order: working models first, llama last (known fp issue under sweep).
run "mistralai/Mistral-7B-Instruct-v0.3"  mistral
run "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" r1distill
run "unsloth/gemma-2-9b-it"               gemma2
run "unsloth/Meta-Llama-3.1-8B-Instruct"  llama31

echo "$(date +%H:%M:%S)  ALL DONE"
