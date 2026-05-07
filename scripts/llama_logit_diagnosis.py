#!/usr/bin/env python3
"""
Llama Logit-Level Paired Corruption Diagnosis
===============================================
Tests whether Llama's LOGIT MARGIN responds to evidence corruption (Group A)
vs distractor swap (Group B). This distinguishes:

  H1: Llama ignores observation content  → Δmargin(A) ≈ Δmargin(B)
  H2: Evidence is distributed/non-linear → Δmargin(A) >> Δmargin(B)
                                           (behavioral effect exists but
                                            linear direction can't capture it)

If H1: model ignores observations → this is architecturally interesting.
If H2: need a non-linear evidence measure for cross-subspace routing.

Usage:
  cd tmc/scripts/e2e_agent
  HF_ENDPOINT=https://hf-mirror.com HF_HUB_CACHE=/tmp/hf_cache \
  python scripts/llama_logit_diagnosis.py
"""

import os, sys, json, random, argparse
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu, ttest_rel

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import ACTION_TOKENS
from steering.hook_utils import get_model_layers
from scripts.paired_corruption_analysis import (
    select_samples, make_corrupted_obs, build_prompt,
)


def compute_margin(logits, tokenizer):
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    tool_ids = [tokenizer.encode(t, add_special_tokens=False)[0]
                for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tokenizer.encode(t, add_special_tokens=False)[0]
                for t in ACTION_TOKENS["finish"]]
    return (torch.logsumexp(log_probs[tool_ids], 0) -
            torch.logsumexp(log_probs[fin_ids],  0)).item()


def get_logit_margin(model, tokenizer, prompt):
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :]
    return compute_margin(logits, tokenizer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--baseline-trace",
                    default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--hotpotqa-data",
                    default="data/hotpotqa/hotpot_dev_distractor_v1.json")
    ap.add_argument("--output-dir", default="results/llama_logit_diagnosis")
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_token = os.environ.get("HF_TOKEN", None)
    print(f"Loading {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, token=hf_token)
    model.eval()
    print("Model loaded.", flush=True)

    samples = select_samples(args.baseline_trace, args.hotpotqa_data,
                             n=args.n_samples, seed=args.seed)
    print(f"Selected {len(samples)} samples", flush=True)

    groups = {"A": [], "B": [], "C": []}
    for gi, group in enumerate(["A", "B", "C"]):
        rng = random.Random(args.seed + gi * 10000)
        for i, sample in enumerate(samples):
            clean_obs, corrupt_obs = make_corrupted_obs(sample, group, rng)
            p_clean   = build_prompt(tokenizer, sample["question"],
                                     sample["step0_query"], clean_obs)
            p_corrupt = build_prompt(tokenizer, sample["question"],
                                     sample["step0_query"], corrupt_obs)
            m_clean   = get_logit_margin(model, tokenizer, p_clean)
            m_corrupt = get_logit_margin(model, tokenizer, p_corrupt)
            delta = m_clean - m_corrupt  # positive = evidence removal → less search
            groups[group].append({
                "margin_clean": m_clean,
                "margin_corrupt": m_corrupt,
                "delta_margin": delta,
                "abs_delta": abs(delta),
            })
            if (i + 1) % 25 == 0:
                print(f"  Group {group}: {i+1}/{len(samples)}", flush=True)
        print(f"  Group {group} done. mean_margin_clean={np.mean([d['margin_clean'] for d in groups[group]]):.3f}",
              flush=True)

    A_abs = [d["abs_delta"] for d in groups["A"]]
    B_abs = [d["abs_delta"] for d in groups["B"]]
    A_raw = [d["delta_margin"] for d in groups["A"]]
    B_raw = [d["delta_margin"] for d in groups["B"]]

    _, p_mw   = mannwhitneyu(A_abs, B_abs, alternative="two-sided")
    _, p_mw1  = mannwhitneyu(A_abs, B_abs, alternative="greater")
    ratio     = np.mean(A_abs) / (np.mean(B_abs) + 1e-9)

    print("\n" + "=" * 60)
    print(f"LOGIT DIAGNOSIS: {args.model}")
    print("=" * 60)
    print(f"  Group A mean |Δmargin|: {np.mean(A_abs):.4f}")
    print(f"  Group B mean |Δmargin|: {np.mean(B_abs):.4f}")
    print(f"  A/B ratio:              {ratio:.3f}x")
    print(f"  MW p (two-sided):       {p_mw:.4f}")
    print(f"  MW p (A>B, one-sided):  {p_mw1:.4f}")
    print(f"  A mean raw Δmargin:     {np.mean(A_raw):.4f}  (+ = less search when evidence removed)")
    print(f"  B mean raw Δmargin:     {np.mean(B_raw):.4f}")
    print()
    if ratio > 1.3 and p_mw1 < 0.05:
        print("  → H2 supported: Llama DOES respond at logit level.")
        print("    Evidence routing exists but is non-linear / multi-dimensional.")
    elif ratio < 1.1 and p_mw1 > 0.3:
        print("  → H1 supported: Llama ignores observation content at logit level!")
        print("    This is an agent-level dissociation (model doesn't use retrieved evidence).")
    else:
        print("  → Ambiguous. Check ratio and direction of A_raw.")
    print("=" * 60)

    out = {
        "model": args.model, "n_samples": len(samples),
        "A_mean_abs_delta": float(np.mean(A_abs)),
        "B_mean_abs_delta": float(np.mean(B_abs)),
        "ratio_AB": float(ratio),
        "MW_pval_twosided": float(p_mw),
        "MW_pval_onesided_Agtr": float(p_mw1),
        "A_mean_raw_delta": float(np.mean(A_raw)),
        "B_mean_raw_delta": float(np.mean(B_raw)),
        "per_sample": {g: groups[g] for g in ["A", "B", "C"]},
    }
    out_path = os.path.join(args.output_dir, "logit_diagnosis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
