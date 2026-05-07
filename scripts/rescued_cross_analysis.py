#!/usr/bin/env python3
"""
追加3: A3 Rescued Cross-Analysis
=================================
For the 15 rescued-via-search samples (from A3 steering) that pass the
corruption filter, compare evidence-corruption sensitivity against
matched non-rescued controls.

Measures:
  1. |Δaction|  — L20 activation shift along action-steering direction
  2. |Δmargin| — first-token logit margin shift (Action vs Final)

Hypothesis: rescued samples show HIGHER sensitivity (larger |Δ|) because
they sit at a more malleable point in the evidence→action routing.

Output → results/paired_corruption/rescued_cross_analysis.json
"""

import os, sys, json, argparse, random
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, get_action_token_ids
from scripts.paired_corruption_analysis import (
    select_samples, make_corrupted_obs, build_prompt,
    extract_l20_hidden,
)

# 19 rescued_via_search sample IDs from A3
A3_RESCUED = {
    "5abaee845542996606241696", "5abbcfaf5542993f40c73ba9",
    "5ae2eda355429928c4239570", "5a8782f25542996e4f308818",
    "5a8f51185542992414482a3d", "5a85b2895542994c784ddb49",
    "5ae256435542992decbdccc3", "5ab29956554299194fa9342d",
    "5ae55d1e55429960a22e02cb", "5ab9cfe655429970cfb8ebaf",
    "5a821c95554299676cceb219", "5abdba405542993f32c2a023",
    "5abf92c45542993fe9a41e07", "5ac2a35055429967731025ce",
    "5ae7535c5542997b22f6a6d8", "5ae47cab5542996836b02cb9",
    "5a79311755429970f5fffe67", "5a7e02b75542997cc2c474f3",
    "5a83c2e25542996488c2e4bc",
}


def get_first_token_margin(model, tokenizer, prompt, action_tid, final_tid):
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(input_ids)
    logits = out.logits[0, -1, :]
    return float(logits[action_tid].item() - logits[final_tid].item())


def run(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    action_tid = get_action_token_ids(tokenizer, "tool_call")[0]
    final_tid = get_action_token_ids(tokenizer, "finish")[0]
    print(f"Token IDs: Action={action_tid}, Final={final_tid}")

    # Load action-steering direction
    d = np.load(args.direction_path)
    key = [k for k in d.files if "direction" in k.lower() or "vec" in k.lower()]
    action_dir = d[key[0] if key else d.files[0]].astype(np.float32)
    action_dir = action_dir / (np.linalg.norm(action_dir) + 1e-12)
    print(f"Action direction shape: {action_dir.shape}")

    # Select ALL qualifying candidates (not just 50)
    all_samples = select_samples(args.baseline_trace, args.hotpotqa_data,
                                 n=9999, seed=args.seed)
    rescued = [s for s in all_samples if s["sample_id"] in A3_RESCUED]
    non_rescued = [s for s in all_samples if s["sample_id"] not in A3_RESCUED]

    # Use 4x matched controls for good power without excessive runtime
    rng = random.Random(args.seed + 1)
    rng.shuffle(non_rescued)
    n_ctrl = min(len(rescued) * 4, len(non_rescued))
    controls = non_rescued[:n_ctrl]
    print(f"Rescued: {len(rescued)}, Controls: {len(controls)}")

    results = {"rescued": [], "control": []}

    for label, group_samples in [("rescued", rescued), ("control", controls)]:
        print(f"\n=== {label.upper()} ({len(group_samples)} samples) ===", flush=True)
        grng = random.Random(args.seed)  # fresh RNG per group for consistency
        for i, sample in enumerate(group_samples):
            clean_obs, corrupted_obs = make_corrupted_obs(sample, "A", grng)

            p_clean = build_prompt(tokenizer, sample["question"],
                                   sample["step0_query"], clean_obs)
            p_corrupt = build_prompt(tokenizer, sample["question"],
                                     sample["step0_query"], corrupted_obs)

            # Activation shift
            h_clean = extract_l20_hidden(model, tokenizer, p_clean, layer_idx=20)
            h_corrupt = extract_l20_hidden(model, tokenizer, p_corrupt, layer_idx=20)
            delta_action = abs(float(np.dot(h_clean - h_corrupt, action_dir)))

            # Logit margin shift
            m_clean = get_first_token_margin(model, tokenizer, p_clean,
                                             action_tid, final_tid)
            m_corrupt = get_first_token_margin(model, tokenizer, p_corrupt,
                                               action_tid, final_tid)
            delta_margin = abs(m_clean - m_corrupt)

            results[label].append({
                "sample_id": sample["sample_id"],
                "delta_action": delta_action,
                "delta_margin": delta_margin,
                "margin_clean": m_clean,
                "margin_corrupt": m_corrupt,
            })
            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{len(group_samples)}]", flush=True)

    # ── Statistics ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESCUED CROSS-ANALYSIS RESULTS")
    print("=" * 70)

    stats = {}

    # --- 1. Raw absolute metrics (for reference) ---
    print("\n--- 1. Raw absolute metrics ---")
    for metric in ["delta_action", "delta_margin"]:
        vals_r = np.array([r[metric] for r in results["rescued"]])
        vals_c = np.array([r[metric] for r in results["control"]])
        u, p = mannwhitneyu(vals_r, vals_c, alternative="greater")
        ratio = vals_r.mean() / vals_c.mean() if vals_c.mean() > 0 else float("inf")
        print(f"\n  {metric}:")
        print(f"    Rescued:  {vals_r.mean():.4f} ± {vals_r.std():.4f}")
        print(f"    Control:  {vals_c.mean():.4f} ± {vals_c.std():.4f}")
        print(f"    Ratio:    {ratio:.2f}x")
        print(f"    MW U={u:.0f}, p={p:.4f}")
        stats[f"raw_{metric}"] = {
            "rescued_mean": float(vals_r.mean()), "rescued_std": float(vals_r.std()),
            "control_mean": float(vals_c.mean()), "control_std": float(vals_c.std()),
            "ratio": float(ratio), "MW_U": float(u), "MW_p": float(p),
        }

    # --- 2. Clean margin comparison (boundary proximity) ---
    print("\n--- 2. Clean margin (boundary proximity) ---")
    mc_r = np.array([r["margin_clean"] for r in results["rescued"]])
    mc_c = np.array([r["margin_clean"] for r in results["control"]])
    u_m, p_m = mannwhitneyu(mc_r, mc_c, alternative="greater")
    near_r = int((mc_r > -5).sum())
    near_c = int((mc_c > -5).sum())
    print(f"  Rescued:  {mc_r.mean():.2f} ± {mc_r.std():.2f}")
    print(f"  Control:  {mc_c.mean():.2f} ± {mc_c.std():.2f}")
    print(f"  MW (rescued > control): U={u_m:.0f}, p={p_m:.6f}")
    print(f"  Near boundary (>-5): rescued {near_r}/{len(mc_r)} "
          f"({near_r/len(mc_r):.0%}), control {near_c}/{len(mc_c)} "
          f"({near_c/len(mc_c):.0%})")
    stats["clean_margin"] = {
        "rescued_mean": float(mc_r.mean()), "rescued_std": float(mc_r.std()),
        "control_mean": float(mc_c.mean()), "control_std": float(mc_c.std()),
        "MW_U": float(u_m), "MW_p": float(p_m),
        "near_boundary_rescued": near_r, "near_boundary_control": near_c,
        "near_boundary_rescued_pct": float(near_r / len(mc_r)),
        "near_boundary_control_pct": float(near_c / len(mc_c)),
    }

    # --- 3. Relative sensitivity: |Δm| / (|m_clean| + 0.5) ---
    print("\n--- 3. Relative sensitivity (scale-corrected) ---")
    EPS = 0.5
    rel_dm_r = np.array([r["delta_margin"] / (abs(r["margin_clean"]) + EPS)
                         for r in results["rescued"]])
    rel_dm_c = np.array([r["delta_margin"] / (abs(r["margin_clean"]) + EPS)
                         for r in results["control"]])
    u_rel, p_rel = mannwhitneyu(rel_dm_r, rel_dm_c, alternative="greater")
    ratio_rel = float(rel_dm_r.mean() / rel_dm_c.mean()) if rel_dm_c.mean() > 0 else float("inf")
    print(f"  Rescued:  {rel_dm_r.mean():.3f} ± {rel_dm_r.std():.3f}  "
          f"median={np.median(rel_dm_r):.3f}")
    print(f"  Control:  {rel_dm_c.mean():.3f} ± {rel_dm_c.std():.3f}  "
          f"median={np.median(rel_dm_c):.3f}")
    print(f"  Ratio:    {ratio_rel:.2f}x")
    print(f"  MW U={u_rel:.0f}, p={p_rel:.4f}")
    stats["relative_delta_margin"] = {
        "rescued_mean": float(rel_dm_r.mean()), "rescued_std": float(rel_dm_r.std()),
        "rescued_median": float(np.median(rel_dm_r)),
        "control_mean": float(rel_dm_c.mean()), "control_std": float(rel_dm_c.std()),
        "control_median": float(np.median(rel_dm_c)),
        "ratio": ratio_rel, "MW_U": float(u_rel), "MW_p": float(p_rel),
    }

    # --- 4. Signed Δmargin directionality ---
    #   signed_dm = margin_clean - margin_corrupt
    #   Positive = evidence removal pushed margin DOWN (toward stop) = expected
    print("\n--- 4. Signed Δmargin directionality ---")
    sdm_r = np.array([r["margin_clean"] - r["margin_corrupt"]
                       for r in results["rescued"]])
    sdm_c = np.array([r["margin_clean"] - r["margin_corrupt"]
                       for r in results["control"]])
    consist_r = int((sdm_r > 0).sum())
    consist_c = int((sdm_c > 0).sum())
    print(f"  Rescued: mean={sdm_r.mean():+.3f}, "
          f"consistent (>0): {consist_r}/{len(sdm_r)} ({consist_r/len(sdm_r):.0%})")
    print(f"  Control: mean={sdm_c.mean():+.3f}, "
          f"consistent (>0): {consist_c}/{len(sdm_c)} ({consist_c/len(sdm_c):.0%})")
    stats["signed_directionality"] = {
        "rescued_mean": float(sdm_r.mean()), "rescued_std": float(sdm_r.std()),
        "control_mean": float(sdm_c.mean()), "control_std": float(sdm_c.std()),
        "rescued_consistent": consist_r, "rescued_n": len(sdm_r),
        "control_consistent": consist_c, "control_n": len(sdm_c),
    }

    # --- 5. Confound check: |m_clean| vs |Δm| correlation ---
    print("\n--- 5. Confound: |m_clean| vs |Δmargin| correlation (all) ---")
    from scipy.stats import spearmanr, pearsonr
    all_mc = np.concatenate([mc_r, mc_c])
    all_dm = np.concatenate([
        np.array([r["delta_margin"] for r in results["rescued"]]),
        np.array([r["delta_margin"] for r in results["control"]]),
    ])
    rho_s, p_s = spearmanr(np.abs(all_mc), all_dm)
    r_p, p_p = pearsonr(np.abs(all_mc), all_dm)
    print(f"  Spearman rho={rho_s:.3f}, p={p_s:.4f}")
    print(f"  Pearson  r  ={r_p:.3f}, p={p_p:.4f}")
    print("  → Positive = larger baseline margin ↔ larger |Δ| (scale confound)")
    stats["confound_correlation"] = {
        "spearman_rho": float(rho_s), "spearman_p": float(p_s),
        "pearson_r": float(r_p), "pearson_p": float(p_p),
    }

    # ── Save ────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out = {
        "timestamp": datetime.now().isoformat(),
        "config": {"n_rescued": len(rescued), "n_control": len(controls),
                    "seed": args.seed, "model": args.model},
        "stats": stats,
        "results": results,
    }
    out_path = os.path.join(args.output_dir, "rescued_cross_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--baseline-trace",
                    default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--hotpotqa-data",
                    default="data/hotpotqa/hotpot_dev_distractor_v1.json")
    ap.add_argument("--direction-path",
                    default="steering/directions/direction_search_v3_layer20.npz")
    ap.add_argument("--output-dir", default="results/paired_corruption")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()

