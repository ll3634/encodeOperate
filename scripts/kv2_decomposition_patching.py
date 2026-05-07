#!/usr/bin/env python3
"""
KV2 Clean-Corrupt Decomposition Patching
=========================================
For each paired (clean, corrupt) sample, compute KV2's d_model-space
clean-corrupt delta at L18 (pre-o_proj head slice → o_proj), and decompose
along evidence_dir:

    Δ_KV2     = W_o[:, s:e] @ (pre_oproj_clean[s:e] - pre_oproj_corrupt[s:e])
    Δ_par     = (Δ_KV2 · ê) · ê                     (1-D, evidence axis)
    Δ_orth    = Δ_KV2 - Δ_par                        (d-1 D complement)
    Δ_rand_p  = (Δ_KV2 · r̂) · r̂                   (1-D, random axis control)
    Δ_rand_o  = Δ_KV2 - Δ_rand_p

Five patched forwards per sample: FULL / PAR / ORTH / RAND_PAR / RAND_ORTH.
Each ADDS the respective delta to attn_L18 output at last token in the
corrupt run, then measures action_dir projection of residual_L20[last].

Recovery = (action_patched - action_corrupt) / (action_clean - action_corrupt)

Output → results/kv_decomposition/kv2_decomposition_patching.json
"""

import os, sys, json, argparse, random
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent.parent))
from steering.hook_utils import get_model_layers
from scripts.paired_corruption_analysis import (
    select_samples, make_corrupted_obs, build_prompt,
)
from scripts.activation_patching import capture_attn_pre_oproj


def extract_l20_last(model, tokenizer, prompt, measure_layer=20):
    layers = get_model_layers(model)
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    cap = {}
    def hk(m, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        cap['h'] = h[0, -1, :].detach().float().cpu().numpy()
    hdl = layers[measure_layer].register_forward_hook(hk)
    with torch.no_grad():
        model(ids)
    hdl.remove()
    return cap['h']


def patched_forward_measure(model, tokenizer, prompt_corrupt, patch_layer,
                            delta_vec, action_dir, measure_layer=20):
    """Add delta_vec (d_model) to self_attn output at last token of L{patch_layer}.
    Measure action_dir · residual_L{measure_layer}[last].
    """
    layers = get_model_layers(model)
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt_corrupt, return_tensors="pt").to(device)
    delta_t = torch.tensor(delta_vec, dtype=torch.float16, device=device)

    cap = {}
    handles = []

    def attn_add_hook(m, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h[0, -1, :] = h[0, -1, :] + delta_t
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h

    handles.append(
        layers[patch_layer].self_attn.register_forward_hook(attn_add_hook))

    def meas_hook(m, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        cap['h'] = h[0, -1, :].detach().float().cpu().numpy()
    handles.append(layers[measure_layer].register_forward_hook(meas_hook))

    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()
    return float(np.dot(cap['h'], action_dir))


def run(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    # Directions
    ev_d = np.load(args.evidence_dir)["decision_direction"].astype(np.float32)
    ev_hat = ev_d / (np.linalg.norm(ev_d) + 1e-12)
    act_d = np.load(args.action_dir)["decision_direction"].astype(np.float32)
    act_hat = act_d / (np.linalg.norm(act_d) + 1e-12)
    print(f"cos(evidence, action) = {np.dot(ev_hat, act_hat):+.4f}")

    # o_proj weight for L18
    layers = get_model_layers(model)
    W_o = layers[args.patch_layer].self_attn.o_proj.weight.detach().float().cpu().numpy()
    # shape: (d_model, n_heads*head_dim) = (3584, 3584)

    n_heads = 28
    head_dim = 128
    n_kv = 4
    heads_per_kv = n_heads // n_kv  # 7
    s = 2 * heads_per_kv * head_dim  # KV2: heads 14..20 → 14*128..21*128
    e = (2 + 1) * heads_per_kv * head_dim
    print(f"KV2 pre_oproj slice: [{s}:{e}] (heads 14-20, dim {e-s})")

    samples = select_samples(args.baseline_trace, args.hotpotqa_data,
                             n=args.n_samples, seed=args.seed)
    if args.n_samples > len(samples):
        print(f"WARNING: only {len(samples)} samples available")

    rows = []
    for i, sample in enumerate(samples):
        # Reproduce same rng state as F3
        rng_copy = random.Random(args.seed)
        for j in range(i):
            make_corrupted_obs(samples[j], "A", rng_copy)
        clean_obs, corrupted_obs = make_corrupted_obs(sample, "A", rng_copy)

        prompt_clean = build_prompt(
            tokenizer, sample["question"], sample["step0_query"], clean_obs)
        prompt_corrupt = build_prompt(
            tokenizer, sample["question"], sample["step0_query"], corrupted_obs)

        # Clean/corrupt baselines
        h_clean = extract_l20_last(model, tokenizer, prompt_clean, args.measure_layer)
        h_corrupt = extract_l20_last(model, tokenizer, prompt_corrupt, args.measure_layer)
        a_clean = float(np.dot(h_clean, act_hat))
        a_corrupt = float(np.dot(h_corrupt, act_hat))
        delta_full = a_clean - a_corrupt
        if abs(delta_full) < 1e-4:
            continue

        # Pre-o_proj at L18
        pre_clean = capture_attn_pre_oproj(model, tokenizer, prompt_clean,
                                            args.patch_layer)
        pre_corrupt = capture_attn_pre_oproj(model, tokenizer, prompt_corrupt,
                                              args.patch_layer)
        dpre = (pre_clean[s:e] - pre_corrupt[s:e]).astype(np.float32)
        # KV2's d_model contribution delta
        delta_kv2 = W_o[:, s:e] @ dpre  # (d_model,)

        # Evidence-axis decomposition
        c_ev = float(np.dot(delta_kv2, ev_hat))
        delta_par = c_ev * ev_hat
        delta_orth = delta_kv2 - delta_par

        # Random-axis control (seeded per sample)
        rng_np = np.random.default_rng(args.seed * 1000 + i)
        r = rng_np.standard_normal(delta_kv2.shape[0]).astype(np.float32)
        r_hat = r / (np.linalg.norm(r) + 1e-12)
        c_r = float(np.dot(delta_kv2, r_hat))
        delta_rand_par = c_r * r_hat
        delta_rand_orth = delta_kv2 - delta_rand_par

        # Five patched forwards
        conds = {
            "full":      delta_kv2,
            "par":       delta_par,
            "orth":      delta_orth,
            "rand_par":  delta_rand_par,
            "rand_orth": delta_rand_orth,
        }
        recov = {}
        for name, d in conds.items():
            a_p = patched_forward_measure(
                model, tokenizer, prompt_corrupt, args.patch_layer,
                d, act_hat, args.measure_layer)
            recov[name] = (a_p - a_corrupt) / delta_full

        # Alignment diagnostics
        norm_kv2 = float(np.linalg.norm(delta_kv2))
        frac_par = (c_ev ** 2) / (norm_kv2 ** 2 + 1e-12)
        frac_rand_par = (c_r ** 2) / (norm_kv2 ** 2 + 1e-12)

        rows.append({
            "sample_id": sample["sample_id"],
            "a_clean": a_clean, "a_corrupt": a_corrupt, "delta": delta_full,
            "norm_delta_kv2": norm_kv2,
            "coef_evidence": c_ev,
            "coef_random": c_r,
            "frac_energy_par": frac_par,
            "frac_energy_rand_par": frac_rand_par,
            "recov_full": recov["full"],
            "recov_par": recov["par"],
            "recov_orth": recov["orth"],
            "recov_rand_par": recov["rand_par"],
            "recov_rand_orth": recov["rand_orth"],
        })

        if (i + 1) % 5 == 0 or i < 3:
            r_full = recov["full"]; r_par = recov["par"]; r_orth = recov["orth"]
            r_rp = recov["rand_par"]; r_ro = recov["rand_orth"]
            print(f"  [{i+1:3d}/{len(samples)}] Δ={delta_full:+.3f} |kv2|={norm_kv2:.3f} "
                  f"full={r_full:+.3f} par={r_par:+.3f} orth={r_orth:+.3f} "
                  f"rp={r_rp:+.3f} ro={r_ro:+.3f}", flush=True)

    # Summary
    def summarize(key):
        vals = np.array([r[key] for r in rows])
        return {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "std": float(vals.std()),
            "n": len(vals),
        }

    summary = {k: summarize(f"recov_{k}")
               for k in ("full", "par", "orth", "rand_par", "rand_orth")}

    # Paired Wilcoxon tests
    par = np.array([r["recov_par"] for r in rows])
    orth = np.array([r["recov_orth"] for r in rows])
    rp = np.array([r["recov_rand_par"] for r in rows])
    ro = np.array([r["recov_rand_orth"] for r in rows])
    tests = {}
    try:
        w_par_vs_orth = wilcoxon(par, orth)
        tests["par_vs_orth"] = {"stat": float(w_par_vs_orth.statistic),
                                 "p": float(w_par_vs_orth.pvalue)}
        w_par_vs_rp = wilcoxon(par, rp)
        tests["par_vs_rand_par"] = {"stat": float(w_par_vs_rp.statistic),
                                     "p": float(w_par_vs_rp.pvalue)}
        w_orth_vs_ro = wilcoxon(orth, ro)
        tests["orth_vs_rand_orth"] = {"stat": float(w_orth_vs_ro.statistic),
                                       "p": float(w_orth_vs_ro.pvalue)}
    except Exception as exn:
        tests["error"] = str(exn)

    od = Path(args.output_dir)
    if od.suffix == ".json":
        out_path = od
        od.parent.mkdir(parents=True, exist_ok=True)
    else:
        od.mkdir(parents=True, exist_ok=True)
        out_path = od / "kv2_decomposition_patching.json"
    out = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": args.model,
            "patch_layer": args.patch_layer,
            "measure_layer": args.measure_layer,
            "n_samples": len(rows),
            "seed": args.seed,
            "kv_group": 2,
            "kv_slice": [s, e],
            "evidence_dir": args.evidence_dir,
            "action_dir": args.action_dir,
        },
        "summary": summary,
        "tests": tests,
        "per_sample": rows,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")

    print("\n── Summary (recovery stats) ──")
    for k in ("full", "par", "orth", "rand_par", "rand_orth"):
        s_k = summary[k]
        print(f"  {k:<10} mean={s_k['mean']:+.4f} median={s_k['median']:+.4f} "
              f"std={s_k['std']:.4f} n={s_k['n']}")
    print("\n── Wilcoxon tests ──")
    for k, v in tests.items():
        if isinstance(v, dict):
            print(f"  {k:<22} p={v['p']:.4g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--baseline-trace",
                    default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--hotpotqa-data",
                    default="data/hotpotqa/hotpot_dev_distractor_v1.json")
    ap.add_argument("--evidence-dir",
                    default="results/phase1_probe/probe_direction_l20.npz")
    ap.add_argument("--action-dir",
                    default="steering/directions/direction_search_v3_layer20.npz")
    ap.add_argument("--output-dir", default="results/kv_decomposition")
    ap.add_argument("--patch-layer", type=int, default=18)
    ap.add_argument("--measure-layer", type=int, default=20)
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
