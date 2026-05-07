#!/bin/bash
set -e
cd /home/featurize/work/tmc/scripts/e2e_agent

# Gemma first (headline; strong trap)
for RHO in -0.20 -0.30 -0.60; do
  OUT=results/attack3_closure/codesearch_steer_gemma/rho${RHO}
  mkdir -p $OUT
  echo "==== Gemma codesearch L37 rho=$RHO -> $OUT ===="
  python scripts/run_nonqa_react_codesearch.py \
    --model unsloth/gemma-2-9b-it \
    --n_items 60 --mode prefilled \
    --steer_direction results/gemma_circuit_sanity/exp2_steering/directions.npz \
    --steer_key action_dir \
    --steer_layer 37 \
    --steer_rho $RHO \
    --hidden_rms 11.2718 \
    --output_dir $OUT
  echo "==== done Gemma rho=$RHO ===="
done

# Mistral (margin-predicted NULL; verifies prediction)
for RHO in -0.20 -0.30 -0.60; do
  OUT=results/attack3_closure/codesearch_steer_mistral/rho${RHO}
  mkdir -p $OUT
  echo "==== Mistral codesearch L28 rho=$RHO -> $OUT ===="
  python scripts/run_nonqa_react_codesearch.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --n_items 60 --mode prefilled \
    --steer_direction results/mistral_circuit_sanity/exp2_steering/directions.npz \
    --steer_key action_dir \
    --steer_layer 28 \
    --steer_rho $RHO \
    --hidden_rms 0.3085 \
    --output_dir $OUT
  echo "==== done Mistral rho=$RHO ===="
done
echo "ALL CODESEARCH SWEEPS COMPLETE"
