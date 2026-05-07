#!/usr/bin/env python3
"""A1 Mistral residual audit: K=100 random null + no-renorm parallel injection.

Pending §02 §8.4 audit items.

Reuses the same prompts (N0, N=50), peak layer (28), rho (-0.20), and
direction sources as scripts/decomposition_cross_model.py for Mistral.

Three tests:
  T1  K=100 random RMS-normalized null   (same setup as existing run)
  T2  no-renorm parallel injection at the natural component norm
       + K=100 random null at matched natural norm
  T3  matched-norm sanity == T1 reused for the ±0.340 baseline check

Required summary fields:
  K, n_random, parallel_norm_natural, parallel_norm_rms,
  random_null_mean_shift, random_null_abs_shift,
  random_null_percentile_at_observed
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import percentileofscore
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS  # noqa: E402
from steering.hook_utils import get_model_layers, compute_rms  # noqa: E402
from scripts.cross_model_full import apply_chat_template_safe  # noqa: E402
from scripts.gemma_steering_sanity import (  # noqa: E402
    get_hidden_rms_at_layer, steered_margin,
)


def normalize_rms(v, target_rms=1.0):
    rms = float(np.sqrt(np.mean(v ** 2)))
    return v if rms < 1e-12 else v * (target_rms / rms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="unsloth/mistral-7b-instruct-v0.3")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--prior-dirs",
                    default="results/mistral_circuit_sanity/exp2_samelayer/directions.npz")
    ap.add_argument("--peak-layer", type=int, default=28)
    ap.add_argument("--steering-pairs",
                    default="results/extractability_support_toggle/pairs.jsonl")
    ap.add_argument("--steering-cond", default="N0")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--rho", type=float, default=-0.20)
    ap.add_argument("--K", type=int, default=100, help="random directions per scale")
    ap.add_argument("--observed-parallel-shift", type=float, default=0.8177343940734864,
                    help="reference value: existing Mistral evi_parallel mean shift")
    ap.add_argument("--seed", type=int, default=20260427)
    ap.add_argument("--out-dir", default="results/a1_mistral_audit_no_renorm")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ----- load prior directions (Mistral L28 same-layer) -----
    prior = np.load(args.prior_dirs)
    action_dir = prior["action_dir"].astype(np.float32)          # unit-L2
    if "evidence_dir_L37" in prior.files:
        evidence_dir = prior["evidence_dir_L37"].astype(np.float32)
    elif "evidence_dir" in prior.files:
        evidence_dir = prior["evidence_dir"].astype(np.float32)
    else:
        raise KeyError(f"no evidence direction in {args.prior_dirs}")

    a = action_dir
    e_unit = evidence_dir / np.linalg.norm(evidence_dir)
    a_unit = a / np.linalg.norm(a)
    cos_ae = float(np.dot(a_unit, e_unit))
    s_par_natural = (float(np.dot(a, e_unit)) * e_unit).astype(np.float32)
    s_par_rms_normed = normalize_rms(s_par_natural, 1.0)

    parallel_norm_natural_L2  = float(np.linalg.norm(s_par_natural))
    parallel_norm_natural_RMS = float(np.sqrt(np.mean(s_par_natural ** 2)))
    parallel_norm_rms_L2  = float(np.linalg.norm(s_par_rms_normed))
    parallel_norm_rms_RMS = float(np.sqrt(np.mean(s_par_rms_normed ** 2)))

    print(f"[geom] cos(action,evidence) = {cos_ae:+.6f}")
    print(f"[geom] s_par_natural  L2={parallel_norm_natural_L2:.6f}  "
          f"RMS={parallel_norm_natural_RMS:.6e}")
    print(f"[geom] s_par_rms_norm L2={parallel_norm_rms_L2:.4f}  "
          f"RMS={parallel_norm_rms_RMS:.4f}")

    # ----- load model -----
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"[info] loading {args.model_path} dtype={args.dtype}")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True)
    model.eval(); device = next(model.parameters()).device
    D = model.config.hidden_size
    sqrtD = float(np.sqrt(D))
    print(f"[info] n_layers={len(get_model_layers(model))} D={D}")

    # ----- prepare prompts -----
    records_all = [json.loads(l) for l in open(args.steering_pairs)]
    records = [r for r in records_all if r.get("condition") == args.steering_cond][:args.limit]
    print(f"[info] {len(records)} {args.steering_cond} prompts")
    builder = PromptBuilder()
    tool_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]

    # ----- random direction banks -----
    rng = np.random.default_rng(args.seed)
    # bank shared across scales: K unit-L2 random vectors, then RMS-normalize for T1/T3
    random_unit = []
    random_rms_norm = []
    for _ in range(args.K):
        r = rng.standard_normal(D).astype(np.float32)
        random_unit.append((r / np.linalg.norm(r)).astype(np.float32))
        random_rms_norm.append(normalize_rms(r, 1.0))

    rows_path = out_dir / "results.jsonl"
    n_rows = 0; t0 = time.time()
    fout = open(rows_path, "w")

    deltas_par_no_renorm = []     # T2 fixed: no-renorm parallel
    deltas_random_rms = [[] for _ in range(args.K)]      # T1/T3
    deltas_random_natural = [[] for _ in range(args.K)]  # T2 null

    for i, rec in enumerate(records):
        steps = [{"action": "search",
                  "action_input": f"about: {rec['question'][:80]}",
                  "observation": rec["obs"]}]
        msgs = builder.build_full_prompt(rec["question"], steps)
        prompt = apply_chat_template_safe(tok, msgs, add_generation_prompt=True)
        rms_h, _ = get_hidden_rms_at_layer(model, tok, prompt, args.peak_layer, device)
        m_base = steered_margin(model, tok, prompt, device, tool_ids, fin_ids,
                                None, 0.0, args.peak_layer)
        fout.write(json.dumps({"sample_id": rec["sample_id"], "cond": "baseline",
                               "margin": m_base, "delta": 0.0,
                               "hidden_rms": rms_h}) + "\n"); n_rows += 1

        # alpha conventions
        # current RMS-normalized injection: direction RMS=1, alpha = rho * rms_h
        alpha_rms = args.rho * rms_h
        # natural-norm injection on action_dir (unit-L2): alpha = rho * rms_h * sqrt(D)
        # so effective per-component scale matches the RMS-normalized version
        alpha_natural = args.rho * rms_h * sqrtD

        # ----- T2: no-renorm parallel injection -----
        # direction is s_par_natural (L2 = |cos|), apply alpha_natural
        m_par_nat = steered_margin(model, tok, prompt, device, tool_ids, fin_ids,
                                   s_par_natural, alpha_natural, args.peak_layer)
        d_par_nat = m_par_nat - m_base
        deltas_par_no_renorm.append(d_par_nat)
        fout.write(json.dumps({"sample_id": rec["sample_id"], "cond": "par_no_renorm",
                               "margin": m_par_nat, "delta": d_par_nat,
                               "alpha": alpha_natural,
                               "inj_L2": alpha_natural * parallel_norm_natural_L2,
                               "inj_RMS": alpha_natural * parallel_norm_natural_RMS}) + "\n"); n_rows += 1

        # ----- T1/T3: K random RMS-normalized -----
        for k in range(args.K):
            m_st = steered_margin(model, tok, prompt, device, tool_ids, fin_ids,
                                  random_rms_norm[k], alpha_rms, args.peak_layer)
            d = m_st - m_base
            deltas_random_rms[k].append(d)
            fout.write(json.dumps({"sample_id": rec["sample_id"],
                                   "cond": f"random_rms_{k:03d}",
                                   "margin": m_st, "delta": d,
                                   "alpha": alpha_rms}) + "\n"); n_rows += 1

        # ----- T2 null: K random at matched natural component norm -----
        # direction = unit-L2 random scaled to L2=|cos|; alpha = alpha_natural
        # equivalent: direction unit-L2, alpha = alpha_natural * |cos|
        alpha_random_natural = alpha_natural * parallel_norm_natural_L2
        for k in range(args.K):
            m_st = steered_margin(model, tok, prompt, device, tool_ids, fin_ids,
                                  random_unit[k], alpha_random_natural, args.peak_layer)
            d = m_st - m_base
            deltas_random_natural[k].append(d)
            fout.write(json.dumps({"sample_id": rec["sample_id"],
                                   "cond": f"random_natural_{k:03d}",
                                   "margin": m_st, "delta": d,
                                   "alpha": alpha_random_natural}) + "\n"); n_rows += 1

        fout.flush()
        if (i + 1) % 5 == 0 or i + 1 == len(records):
            print(f"  [{i+1}/{len(records)}] {time.time()-t0:.1f}s rows={n_rows}")

    fout.close()

    # ----- aggregate -----
    par_nat_arr = np.array(deltas_par_no_renorm, dtype=np.float64)
    par_nat_mean = float(par_nat_arr.mean())
    par_nat_abs_mean = float(np.mean(np.abs(par_nat_arr)))

    rand_rms_per_dir_mean = np.array([np.mean(d) for d in deltas_random_rms])
    rand_rms_per_dir_abs  = np.array([np.mean(np.abs(d)) for d in deltas_random_rms])
    rand_nat_per_dir_mean = np.array([np.mean(d) for d in deltas_random_natural])
    rand_nat_per_dir_abs  = np.array([np.mean(np.abs(d)) for d in deltas_random_natural])

    # T1: percentile of observed parallel shift (+0.818, RMS-normalized run) in K=100 RMS null
    pct_at_observed_signed = float(percentileofscore(rand_rms_per_dir_mean,
                                                     args.observed_parallel_shift))
    pct_at_observed_abs    = float(percentileofscore(rand_rms_per_dir_abs,
                                                     abs(args.observed_parallel_shift)))

    # T2: percentile of par_no_renorm in matched natural-norm null
    pct_par_nat_signed = float(percentileofscore(rand_nat_per_dir_mean, par_nat_mean))
    pct_par_nat_abs    = float(percentileofscore(rand_nat_per_dir_abs, par_nat_abs_mean))

    # null bands
    def band(x, qlo=2.5, qhi=97.5):
        return float(np.percentile(x, qlo)), float(np.percentile(x, qhi))

    rand_rms_p25_p975 = band(rand_rms_per_dir_mean)
    rand_nat_p25_p975 = band(rand_nat_per_dir_mean)
    rand_rms_abs_p975 = float(np.percentile(rand_rms_per_dir_abs, 97.5))
    rand_nat_abs_p975 = float(np.percentile(rand_nat_per_dir_abs, 97.5))

    # stopping rule per spec: |Δm| of no-renorm parallel inside K=100 null band?
    in_null_band = bool(abs(par_nat_mean) < rand_nat_abs_p975)
    verdict = ("residual-explained-by-normalization"
               if in_null_band else "orthogonal-dominant-with-residual")

    summary = {
        "model": args.model_path,
        "peak_layer": args.peak_layer,
        "rho": args.rho,
        "n_samples": int(len(par_nat_arr)),
        "K": int(args.K),
        "n_random": int(args.K),

        "geometry": {
            "d_model": int(D),
            "cos_action_evidence": cos_ae,
            "parallel_norm_natural": parallel_norm_natural_L2,
            "parallel_norm_natural_L2":  parallel_norm_natural_L2,
            "parallel_norm_natural_RMS": parallel_norm_natural_RMS,
            "parallel_norm_rms":         parallel_norm_rms_L2,
            "parallel_norm_rms_L2":      parallel_norm_rms_L2,
            "parallel_norm_rms_RMS":     parallel_norm_rms_RMS,
            "amplification_factor_rms_over_natural":
                float(parallel_norm_rms_L2 / parallel_norm_natural_L2),
        },

        "observed_parallel_shift_rms_normalized": args.observed_parallel_shift,

        "no_renorm_parallel": {
            "mean_shift": par_nat_mean,
            "abs_mean_shift": par_nat_abs_mean,
            "n": int(len(par_nat_arr)),
        },

        # required field block — references the K=100 RMS-norm null
        "random_null_mean_shift":  float(rand_rms_per_dir_mean.mean()),
        "random_null_abs_shift":   float(rand_rms_per_dir_abs.mean()),
        "random_null_percentile_at_observed": pct_at_observed_signed,

        "random_null_rms_normalized": {
            "K": int(args.K),
            "per_dir_mean_shift_mean": float(rand_rms_per_dir_mean.mean()),
            "per_dir_mean_shift_std":  float(rand_rms_per_dir_mean.std()),
            "per_dir_abs_shift_mean":  float(rand_rms_per_dir_abs.mean()),
            "per_dir_abs_shift_std":   float(rand_rms_per_dir_abs.std()),
            "per_dir_mean_shift_min":  float(rand_rms_per_dir_mean.min()),
            "per_dir_mean_shift_max":  float(rand_rms_per_dir_mean.max()),
            "p2_5":  rand_rms_p25_p975[0],
            "p97_5": rand_rms_p25_p975[1],
            "abs_p97_5": rand_rms_abs_p975,
            "percentile_signed_at_observed_parallel": pct_at_observed_signed,
            "percentile_abs_at_observed_parallel":    pct_at_observed_abs,
        },

        "random_null_natural_norm": {
            "K": int(args.K),
            "per_dir_mean_shift_mean": float(rand_nat_per_dir_mean.mean()),
            "per_dir_mean_shift_std":  float(rand_nat_per_dir_mean.std()),
            "per_dir_abs_shift_mean":  float(rand_nat_per_dir_abs.mean()),
            "per_dir_abs_shift_std":   float(rand_nat_per_dir_abs.std()),
            "per_dir_mean_shift_min":  float(rand_nat_per_dir_mean.min()),
            "per_dir_mean_shift_max":  float(rand_nat_per_dir_mean.max()),
            "p2_5":  rand_nat_p25_p975[0],
            "p97_5": rand_nat_p25_p975[1],
            "abs_p97_5": rand_nat_abs_p975,
            "percentile_signed_at_par_no_renorm": pct_par_nat_signed,
            "percentile_abs_at_par_no_renorm":    pct_par_nat_abs,
        },

        "stopping_rule": {
            "criterion": "|no_renorm_parallel.mean_shift| < random_null_natural_norm.abs_p97_5",
            "lhs_abs_par_no_renorm": float(abs(par_nat_mean)),
            "rhs_abs_p97_5_natural_null": rand_nat_abs_p975,
            "in_null_band": in_null_band,
            "verdict": verdict,
        },
    }
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)

    # ----- report.md -----
    md = []
    md.append("# A1 Mistral Residual Audit (no-renorm)\n")
    md.append("Pending §02 §8.4 audit items. Reuses Mistral L28 directions and N=50 N0 prompts "
              "from `decomposition_cross_model.py` (rho={:.2f}).\n".format(args.rho))
    md.append("## Geometry\n")
    md.append(f"- d_model = {D}\n"
              f"- cos(action_dir, evidence_dir) = {cos_ae:+.6f}\n"
              f"- parallel component natural L2-norm = {parallel_norm_natural_L2:.6f}\n"
              f"- parallel component natural RMS    = {parallel_norm_natural_RMS:.3e}\n"
              f"- parallel component after RMS-renorm L2-norm = {parallel_norm_rms_L2:.3f}\n"
              f"- amplification factor (rms-norm L2 / natural L2) = "
              f"{parallel_norm_rms_L2/parallel_norm_natural_L2:.1f}x\n")

    md.append("## T1/T3 — K=100 random RMS-normalized null\n")
    md.append(f"- K = {args.K}\n"
              f"- mean of per-dir mean shift = {rand_rms_per_dir_mean.mean():+.4f}\n"
              f"- std  of per-dir mean shift = {rand_rms_per_dir_mean.std():.4f}\n"
              f"- mean of per-dir |shift|    = {rand_rms_per_dir_abs.mean():.4f}\n"
              f"- 95% null band (signed)     = "
              f"[{rand_rms_p25_p975[0]:+.4f}, {rand_rms_p25_p975[1]:+.4f}]\n"
              f"- 97.5% null bound (|shift|) = {rand_rms_abs_p975:.4f}\n"
              f"- observed evi_parallel mean shift (existing run) = "
              f"{args.observed_parallel_shift:+.4f}\n"
              f"- percentile of observed parallel within K=100 RMS null (signed) = "
              f"{pct_at_observed_signed:.1f}%\n"
              f"- percentile of |observed| within K=100 RMS null (|.|)         = "
              f"{pct_at_observed_abs:.1f}%\n"
              f"- §8.4 reference: random ±0.340 baseline (K=30); reproduced here at K=100 with "
              f"std={rand_rms_per_dir_mean.std():.3f}\n")

    md.append("## T2 — no-renorm parallel injection\n")
    md.append(f"- direction: s_par_natural = (action·evidence_unit) · evidence_unit, L2={parallel_norm_natural_L2:.6f}\n"
              f"- alpha matches the natural-norm convention for action_dir "
              f"(alpha = rho · rms_h · sqrt(D))\n"
              f"- mean shift = {par_nat_mean:+.4f}\n"
              f"- |mean shift| = {par_nat_abs_mean:.4f}\n"
              f"- compared to K=100 random null at matched natural norm:\n"
              f"  - null mean = {rand_nat_per_dir_mean.mean():+.4f} ± {rand_nat_per_dir_mean.std():.4f}\n"
              f"  - null 95% band signed = [{rand_nat_p25_p975[0]:+.4f}, {rand_nat_p25_p975[1]:+.4f}]\n"
              f"  - null 97.5% bound on |shift| = {rand_nat_abs_p975:.4f}\n"
              f"  - percentile of par_no_renorm in matched null (signed) = {pct_par_nat_signed:.1f}%\n"
              f"  - percentile of |par_no_renorm| in matched null (|.|)  = {pct_par_nat_abs:.1f}%\n")

    md.append("## Stopping rule\n")
    md.append(f"- criterion: |Δm of no-renorm parallel| < null 97.5% bound on |Δm| at matched scale\n"
              f"- LHS = {abs(par_nat_mean):.4f}\n"
              f"- RHS = {rand_nat_abs_p975:.4f}\n"
              f"- in_null_band = **{in_null_band}**\n"
              f"- verdict: **{verdict}**\n")

    md.append("## Interpretation\n")
    if in_null_band:
        md.append("The no-renorm parallel injection falls inside the matched-scale random null band. "
                  "The +0.818 evidence-parallel residual reported in §8.4 is consistent with "
                  "RMS-renormalization amplifying a very small natural parallel component "
                  f"(amplification factor {parallel_norm_rms_L2/parallel_norm_natural_L2:.0f}x). "
                  "**Status: residual-explained-by-normalization.**\n")
    else:
        md.append("The no-renorm parallel injection remains outside the matched-scale random null band. "
                  "The evidence-parallel residual is not solely an RMS-renormalization artifact. "
                  "**Status: orthogonal-dominant-with-residual.**\n")

    open(out_dir / "report.md", "w").write("\n".join(md))

    print("\n[summary]")
    print(f"  no-renorm parallel mean shift = {par_nat_mean:+.4f}  "
          f"|.|={par_nat_abs_mean:.4f}")
    print(f"  K=100 RMS-norm random null     mean={rand_rms_per_dir_mean.mean():+.4f} "
          f"± {rand_rms_per_dir_mean.std():.4f}  |.|={rand_rms_per_dir_abs.mean():.4f}")
    print(f"  K=100 natural-norm random null mean={rand_nat_per_dir_mean.mean():+.4f} "
          f"± {rand_nat_per_dir_mean.std():.4f}  |.|={rand_nat_per_dir_abs.mean():.4f}")
    print(f"  percentile of observed +0.818 in RMS null = {pct_at_observed_signed:.1f}% (signed) / "
          f"{pct_at_observed_abs:.1f}% (|.|)")
    print(f"  stopping rule: in_null_band={in_null_band}  verdict={verdict}")
    print(f"\n[wrote] {out_dir/'summary.json'}")
    print(f"[wrote] {out_dir/'report.md'}")


if __name__ == "__main__":
    main()
