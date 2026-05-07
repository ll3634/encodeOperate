#!/usr/bin/env python3
"""
Llama Layer-Depth Paired Corruption Test
==========================================
Tests the hypothesis: "Llama's evidence direction at L24 (75% depth) is
less sensitive to paragraph swap than at earlier layers."

For each target layer, runs a full self-contained paired corruption:
  1. Build evidence_dir from step-1 hidden states at that layer
  2. Build action_dir from PopQA margin contrast at that layer
  3. Run paired corruption at that layer (N=200)

Compare A/B (evidence) and A/B (action) across layers.
If L16 shows higher A/B than L24 → layer-depth hypothesis supported.

Usage:
  cd tmc/scripts/e2e_agent
  HF_ENDPOINT=https://hf-mirror.com HF_HUB_CACHE=/tmp/hf_cache \
  python scripts/llama_layer_sweep_corruption.py
"""

import os, sys, json, random, argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import mannwhitneyu

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import get_model_layers
from scripts.paired_corruption_analysis import select_samples, make_corrupted_obs, build_prompt
from scripts.cross_model_full import (
    collect_step1_states, collect_popqa_multilayer,
    extract_action_dir_from_popqa, train_probe, extract_hidden, compute_margin,
)


def run_corruption_at_layer(model, tokenizer, layer, evidence_dir, action_dir,
                             baseline_path, hotpotqa_path, n_samples=200, seed=42):
    """Run paired corruption at a specific layer, return A/B ratios."""
    samples = select_samples(baseline_path, hotpotqa_path, n=n_samples, seed=seed)
    groups = {"A": [], "B": []}
    for gi, group in enumerate(["A", "B"]):
        rng = random.Random(seed + gi * 10000)
        for sample in samples:
            clean_obs, corrupt_obs = make_corrupted_obs(sample, group, rng)
            p_c = build_prompt(tokenizer, sample["question"], sample["step0_query"], clean_obs)
            p_x = build_prompt(tokenizer, sample["question"], sample["step0_query"], corrupt_obs)
            h_c, _ = extract_hidden(model, tokenizer, p_c, layer)
            h_x, _ = extract_hidden(model, tokenizer, p_x, layer)
            diff = h_c - h_x
            groups[group].append({
                "delta_action":   abs(float(np.dot(diff, action_dir))),
                "delta_evidence": abs(float(np.dot(diff, evidence_dir))),
            })
        print(f"    Group {group}: {len(groups[group])} done", flush=True)

    A_act = [d["delta_action"]   for d in groups["A"]]
    B_act = [d["delta_action"]   for d in groups["B"]]
    A_evi = [d["delta_evidence"] for d in groups["A"]]
    B_evi = [d["delta_evidence"] for d in groups["B"]]
    _, p_act = mannwhitneyu(A_act, B_act, alternative="two-sided")
    _, p_evi = mannwhitneyu(A_evi, B_evi, alternative="two-sided")
    return {
        "layer": layer,
        "AB_action":   float(np.mean(A_act) / (np.mean(B_act) + 1e-9)),
        "MW_action_p": float(p_act),
        "AB_evidence": float(np.mean(A_evi) / (np.mean(B_evi) + 1e-9)),
        "MW_evi_p":    float(p_evi),
        "A_mean_act":  float(np.mean(A_act)),
        "B_mean_act":  float(np.mean(B_act)),
        "A_mean_evi":  float(np.mean(A_evi)),
        "B_mean_evi":  float(np.mean(B_evi)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",         default="unsloth/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--popqa-path",    default="data/popqa/popqa_test.jsonl")
    ap.add_argument("--labels-path",   default="results/phase1_probe/labels.jsonl")
    ap.add_argument("--baseline-trace",default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--hotpotqa-data", default="data/hotpotqa/hotpot_dev_distractor_v1.json")
    ap.add_argument("--output-dir",    default="results/llama_layer_corruption")
    ap.add_argument("--layers",        nargs="+", type=int, default=[16, 20, 24])
    ap.add_argument("--n-popqa",       type=int, default=400)
    ap.add_argument("--n-corruption",  type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_token = os.environ.get("HF_TOKEN", None)
    print(f"Loading {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, token=hf_token)
    model.eval()
    print(f"Model loaded. Layers: {len(get_model_layers(model))}", flush=True)

    # Step 1: collect step-1 hidden states at all target layers (single pass)
    print(f"\n=== Step 1: step-1 hidden states at {args.layers} ===", flush=True)
    step1_data = collect_step1_states(
        model, tokenizer, args.labels_path, args.baseline_trace, args.layers)
    print(f"  {len(step1_data)} samples", flush=True)

    # Step 2: collect PopQA at all target layers (single pass)
    print(f"\n=== Step 2: PopQA states N={args.n_popqa} at {args.layers} ===", flush=True)
    popqa_by_layer = collect_popqa_multilayer(
        model, tokenizer, args.popqa_path, args.layers, n=args.n_popqa)

    # Step 3: for each layer, build directions + run corruption
    all_results = []
    for li in args.layers:
        print(f"\n=== Layer {li} ===", flush=True)

        # Evidence probe
        X = np.array([d["hidden"][li] for d in step1_data], dtype=np.float32)
        y = np.array([d["label"]      for d in step1_data], dtype=np.int32)
        evi_dir, cv = train_probe(X, y, return_cv=True)
        print(f"  Evidence AUROC: {cv['auroc_mean']:.3f} ± {cv['auroc_std']:.3f}", flush=True)

        # Action direction + quality
        act_dir, quality, mstats = extract_action_dir_from_popqa(popqa_by_layer[li])
        cos_ae = float(np.dot(act_dir, evi_dir))
        print(f"  Action quality: {quality:.3f}  cos(act,evi): {cos_ae:.4f}", flush=True)

        # Paired corruption
        print(f"  Running corruption N={args.n_corruption} ...", flush=True)
        res = run_corruption_at_layer(
            model, tokenizer, li, evi_dir, act_dir,
            args.baseline_trace, args.hotpotqa_data,
            n_samples=args.n_corruption)
        res["evidence_auroc"]    = cv["auroc_mean"]
        res["action_quality"]    = quality
        res["cos_action_evidence"] = cos_ae
        all_results.append(res)

        print(f"  A/B evidence: {res['AB_evidence']:.3f}x  p={res['MW_evi_p']:.4f}")
        print(f"  A/B action:   {res['AB_action']:.3f}x  p={res['MW_action_p']:.4f}")

    # Summary
    print("\n" + "="*60)
    print(f"LAYER DEPTH TEST SUMMARY: {args.model}")
    print("="*60)
    print(f"{'Layer':<8} {'evi AUROC':<12} {'A/B evi':<12} {'p':<8} {'A/B act':<12} {'p'}")
    for r in all_results:
        depth = r['layer'] / len(get_model_layers(model)) * 100
        print(f"L{r['layer']:<6} {r['evidence_auroc']:<12.3f} "
              f"{r['AB_evidence']:<12.3f} {r['MW_evi_p']:<8.4f} "
              f"{r['AB_action']:<12.3f} {r['MW_action_p']:.4f}  ({depth:.0f}% depth)")
    print("="*60)

    out_path = os.path.join(args.output_dir, "layer_corruption_results.json")
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "timestamp": datetime.now().isoformat(),
                   "results": all_results}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
