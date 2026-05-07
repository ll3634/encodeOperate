#!/usr/bin/env python3
"""
PCA Bridge Analysis — The Killer Figure
=========================================
Connects PCA subspace discovery to the broken-bridge narrative.

Key analyses:
1. Projection of action_dir / evidence_dir onto the circuit's PCA subspace
2. Comparison with random baseline (k/D for random vectors)
3. Cross-subspace bridge gain
4. Alignment between PCA subspace and interpretable directions at each k
"""

import os, sys, json, argparse, random
import numpy as np
from pathlib import Path

import torch
sys.path.insert(0, str(Path(__file__).parent.parent))
from steering.hook_utils import get_model_layers
from scripts.paired_corruption_analysis import (
    select_samples, make_corrupted_obs, build_prompt,
)
from scripts.activation_patching import extract_component_outputs

K_VALUES = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]
D = 3584  # model hidden dim


def projection_onto_subspace(vec, Vt, k):
    """||P_k(vec)||^2 = sum of squared dot products with top-k PCs."""
    proj = 0.0
    for j in range(min(k, len(Vt))):
        proj += float(np.dot(vec, Vt[j])) ** 2
    return proj


def run(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    n_layers = len(get_model_layers(model))

    # Load directions
    action_dir = np.load(args.action_dir)["decision_direction"].astype(np.float32)
    action_dir /= np.linalg.norm(action_dir)
    evidence_dir = np.load(args.evidence_dir)["decision_direction"].astype(np.float32)
    evidence_dir /= np.linalg.norm(evidence_dir)
    print(f"  cos(evi, act) = {float(np.dot(evidence_dir, action_dir)):.4f}")

    # Generate random control directions
    rng = np.random.RandomState(42)
    n_rand = 50
    rand_dirs = []
    for _ in range(n_rand):
        rd = rng.randn(D).astype(np.float32)
        rd /= np.linalg.norm(rd)
        rand_dirs.append(rd)

    samples = select_samples(args.baseline_trace, args.hotpotqa_data,
                              n=args.n_samples, seed=args.seed)
    print(f"Using {len(samples)} samples", flush=True)

    # ── Collect diffs ──────────────────────────────────────────────────────
    print("Collecting difference vectors...", flush=True)
    diffs_attn, diffs_mlp = [], []
    for i, sample in enumerate(samples):
        rng_copy = random.Random(args.seed)
        for j in range(i):
            make_corrupted_obs(samples[j], "A", rng_copy)
        clean_obs, corrupted_obs = make_corrupted_obs(sample, "A", rng_copy)
        prompt_clean = build_prompt(tokenizer, sample["question"],
                                     sample["step0_query"], clean_obs)
        prompt_corrupt = build_prompt(tokenizer, sample["question"],
                                       sample["step0_query"], corrupted_obs)
        clean_cache = extract_component_outputs(
            model, tokenizer, prompt_clean, n_layers=n_layers)
        corrupt_cache = extract_component_outputs(
            model, tokenizer, prompt_corrupt, n_layers=n_layers)
        diffs_attn.append((clean_cache[('attn', 18)] - corrupt_cache[('attn', 18)]).astype(np.float32))
        diffs_mlp.append((clean_cache[('mlp', 20)] - corrupt_cache[('mlp', 20)]).astype(np.float32))
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(samples)}]", flush=True)

    # ── PCA ────────────────────────────────────────────────────────────────
    print("Computing PCA...", flush=True)
    mat_attn = np.stack(diffs_attn, axis=0)
    mat_mlp = np.stack(diffs_mlp, axis=0)
    _, S_attn, Vt_attn = np.linalg.svd(mat_attn - mat_attn.mean(0), full_matrices=False)
    _, S_mlp, Vt_mlp = np.linalg.svd(mat_mlp - mat_mlp.mean(0), full_matrices=False)

    # Save PCA directions for future use
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(os.path.join(args.output_dir, "pca_directions.npz"),
             Vt_attn=Vt_attn[:500], S_attn=S_attn,
             Vt_mlp=Vt_mlp[:500], S_mlp=S_mlp)

    # ── Projection analysis ───────────────────────────────────────────────
    print("\n" + "="*70)
    print("PROJECTION OF INTERPRETABLE DIRECTIONS ONTO CIRCUIT PCA SUBSPACE")
    print("="*70)

    results = {}
    for comp_name, Vt in [("attn_L18", Vt_attn), ("mlp_L20", Vt_mlp)]:
        print(f"\n--- {comp_name} ---")
        print(f"{'k':>5} | {'||P_k(act)||²':>14} {'||P_k(evi)||²':>14} "
              f"{'random_mean':>12} {'random_baseline':>16} | "
              f"{'act/rand':>9} {'evi/rand':>9}")
        comp_results = {}
        for k in K_VALUES:
            p_act = projection_onto_subspace(action_dir, Vt, k)
            p_evi = projection_onto_subspace(evidence_dir, Vt, k)
            p_rands = [projection_onto_subspace(rd, Vt, k) for rd in rand_dirs]
            p_rand_mean = np.mean(p_rands)
            random_baseline = k / D  # theoretical baseline for random
            ratio_act = p_act / random_baseline if random_baseline > 0 else 0
            ratio_evi = p_evi / random_baseline if random_baseline > 0 else 0
            comp_results[k] = {
                "proj_action": float(p_act), "proj_evidence": float(p_evi),
                "rand_mean": float(p_rand_mean), "rand_baseline": float(random_baseline),
                "ratio_act_vs_rand": float(ratio_act),
                "ratio_evi_vs_rand": float(ratio_evi),
            }
            print(f"{k:>5} | {p_act:>14.4f} {p_evi:>14.4f} "
                  f"{p_rand_mean:>12.4f} {random_baseline:>16.4f} | "
                  f"{ratio_act:>9.2f}x {ratio_evi:>9.2f}x")
        results[comp_name] = comp_results

    # ── Cross-subspace bridge analysis ────────────────────────────────────
    print(f"\n{'='*70}")
    print("CROSS-SUBSPACE BRIDGE ANALYSIS")
    print("  Bridge gain = sqrt(||P_k(evi)||²) × sqrt(||P_k(act)||²)")
    print("  = geometric mean of how much evidence enters × action exits")
    print(f"{'='*70}")

    # Load behavioral recovery from previous experiment
    prev_path = os.path.join("results/pca_patching/pca_patching_results.json")
    prev_data = None
    if os.path.exists(prev_path):
        prev_data = json.load(open(prev_path))
        print("  (Loaded behavioral recovery from pca_patching_results.json)")

    for comp_name in ["attn_L18", "mlp_L20"]:
        print(f"\n--- {comp_name} ---")
        print(f"{'k':>5} | {'bridge_gain':>12} {'evi_in':>8} {'act_out':>8} "
              f"| {'behav_recov':>12} {'var_explained':>14}")
        Vt = Vt_attn if 'attn' in comp_name else Vt_mlp
        S = S_attn if 'attn' in comp_name else S_mlp
        var_total = float((S**2).sum())
        for k in K_VALUES:
            r = results[comp_name][k]
            evi_in = np.sqrt(r["proj_evidence"])
            act_out = np.sqrt(r["proj_action"])
            bridge = evi_in * act_out
            var_k = float((S[:k]**2).sum()) / var_total
            behav = ""
            if prev_data and f"{comp_name}_pca{k}" in prev_data.get("summary", {}):
                behav = f"{prev_data['summary'][f'{comp_name}_pca{k}']['median_recovery']:+.4f}"
            elif prev_data and k > 500:
                behav = "~full"
            print(f"{k:>5} | {bridge:>12.4f} {evi_in:>8.4f} {act_out:>8.4f} "
                  f"| {behav:>12} {var_k:>14.4f}")

    # ── Narrative summary ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("NARRATIVE SUMMARY")
    print(f"{'='*70}")
    # For mlp_L20, what fraction of action_dir is reachable through the circuit?
    r10 = results["mlp_L20"][10]
    r5 = results["mlp_L20"][5]
    print(f"""
The circuit (attn_L18 → mlp_L20) operates in a ~10D subspace that captures
73% of the behavioral effect (PCA patching).

Within this subspace:
  - action_dir projection: ||P_10(act)||² = {r10['proj_action']:.4f}
    (vs random baseline k/D = {r10['rand_baseline']:.4f}, ratio = {r10['ratio_act_vs_rand']:.1f}x)
  - evidence_dir projection: ||P_10(evi)||² = {r10['proj_evidence']:.4f}
    (vs random baseline = {r10['rand_baseline']:.4f}, ratio = {r10['ratio_evi_vs_rand']:.1f}x)

If action_dir is ABOVE random → the circuit DOES route toward action, just weakly.
If evidence_dir is ABOVE random → the circuit IS processing evidence.
If evidence >> action → broken bridge: evidence enters but can't exit to action.
If evidence ≈ action ≈ random → the circuit uses a different coding scheme entirely.
""")

    # ── Save ──────────────────────────────────────────────────────────────
    output = {"projections": results}
    out_path = os.path.join(args.output_dir, "bridge_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--baseline-trace",
                    default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--hotpotqa-data",
                    default="data/hotpotqa/hotpot_dev_distractor_v1.json")
    ap.add_argument("--output-dir", default="results/pca_bridge")
    ap.add_argument("--action-dir",
                    default="steering/directions/direction_search_v3_layer20.npz")
    ap.add_argument("--evidence-dir",
                    default="results/phase1_probe/probe_direction_l20.npz")
    ap.add_argument("--n-samples", type=int, default=9999)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()

