#!/usr/bin/env python3
"""Cross-model functional orthogonality decomposition.

Replicates the Qwen functional_orthogonality_control.py result for Mistral
and Gemma at each model's action peak layer.

Inputs (per model):
  --prior-dirs : directions.npz from <model>_circuit_sanity/exp2_samelayer
                 contains action_dir + same-layer evidence direction
  --peak-layer : action peak layer (Mistral=28, Gemma=37)

Decomposition (action_dir s, evidence_dir e, both unit-norm):
  s_par  = (s · ê) ê
  s_perp = s - s_par
All three (s, s_par, s_perp) and K random vectors are then RMS-normalized
to 1.0 so all interventions have identical effective magnitude.

For each of N=50 N0 prompts, we measure first-token margin shift at the
decision token under {full, evi_par, evi_perp, random_k} steering at the
peak action layer with rho=-0.20. Per-prompt alpha uses live hidden_rms.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import percentileofscore
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS  # noqa: E402
from steering.hook_utils import get_model_layers, SteeringHook, compute_rms  # noqa: E402
from scripts.cross_model_full import apply_chat_template_safe  # noqa: E402
from scripts.gemma_steering_sanity import (  # noqa: E402
    margin_first_token, get_hidden_rms_at_layer, steered_margin,
)


def normalize_rms(v, target_rms=1.0):
    rms = float(np.sqrt(np.mean(v ** 2)))
    return v if rms < 1e-12 else v * (target_rms / rms)


def decompose(action_dir, evidence_dir):
    a = action_dir.astype(np.float32)
    e = evidence_dir.astype(np.float32)
    e_unit = e / np.linalg.norm(e)
    s_par = float(np.dot(a, e_unit)) * e_unit
    s_perp = a - s_par
    return a, s_par.astype(np.float32), s_perp.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--prior-dirs", required=True,
                    help="directions.npz from <model>_circuit_sanity/exp2_samelayer")
    ap.add_argument("--peak-layer", type=int, required=True)
    ap.add_argument("--steering-pairs",
                    default="results/extractability_support_toggle/pairs.jsonl")
    ap.add_argument("--steering-cond", default="N0")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--rho", type=float, default=-0.20)
    ap.add_argument("--n-random", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260427)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load prior directions
    prior = np.load(args.prior_dirs)
    action_dir = prior["action_dir"].astype(np.float32)
    if "evidence_dir_L37" in prior.files:
        evidence_dir = prior["evidence_dir_L37"].astype(np.float32)  # same-layer
    elif "evidence_dir" in prior.files:
        evidence_dir = prior["evidence_dir"].astype(np.float32)
    else:
        raise KeyError(f"no evidence direction in {args.prior_dirs}")

    a_raw, s_par_raw, s_perp_raw = decompose(action_dir, evidence_dir)

    # Geometry pre-normalization
    geom = {
        "action_norm": float(np.linalg.norm(a_raw)),
        "parallel_norm": float(np.linalg.norm(s_par_raw)),
        "perp_norm": float(np.linalg.norm(s_perp_raw)),
        "var_parallel_fraction": float(
            np.linalg.norm(s_par_raw) ** 2 / np.linalg.norm(a_raw) ** 2),
        "cos_action_evidence": float(
            np.dot(a_raw / np.linalg.norm(a_raw),
                   evidence_dir / np.linalg.norm(evidence_dir))),
    }
    print(f"  ||action||={geom['action_norm']:.4f}  ||par||={geom['parallel_norm']:.4f}  "
          f"||perp||={geom['perp_norm']:.4f}")
    print(f"  var(par)/var(action)={geom['var_parallel_fraction']:.6f}  "
          f"cos(action,evidence)={geom['cos_action_evidence']:+.4f}")

    # RMS-normalize all to 1.0
    full_dir = normalize_rms(a_raw, 1.0)
    par_dir  = normalize_rms(s_par_raw, 1.0)
    perp_dir = normalize_rms(s_perp_raw, 1.0)
    rng = np.random.default_rng(args.seed)
    D = full_dir.shape[0]
    random_dirs = [normalize_rms(rng.standard_normal(D).astype(np.float32), 1.0)
                   for _ in range(args.n_random)]

    # Load model
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"[info] loading {args.model_path} dtype={args.dtype}")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True)
    model.eval(); device = next(model.parameters()).device
    print(f"[info] n_layers={len(get_model_layers(model))} D={model.config.hidden_size}")

    # Build prompts (same convention as gemma_steering_sanity)
    records_all = [json.loads(l) for l in open(args.steering_pairs)]
    records = [r for r in records_all if r.get("condition") == args.steering_cond][:args.limit]
    print(f"[info] {len(records)} {args.steering_cond} prompts")
    builder = PromptBuilder()
    tool_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]

    fixed_dirs = {"full": full_dir, "evi_parallel": par_dir, "evi_perp": perp_dir}
    rand_keys = [f"random_{i:03d}" for i in range(args.n_random)]
    all_dirs = list(fixed_dirs.items()) + list(zip(rand_keys, random_dirs))

    rows_path = out_dir / "results.jsonl"
    n_rows = 0; t0 = time.time()
    with open(rows_path, "w") as f:
        for i, rec in enumerate(records):
            steps = [{"action": "search",
                      "action_input": f"about: {rec['question'][:80]}",
                      "observation": rec["obs"]}]
            msgs = builder.build_full_prompt(rec["question"], steps)
            prompt = apply_chat_template_safe(tok, msgs, add_generation_prompt=True)
            rms_h, _ = get_hidden_rms_at_layer(model, tok, prompt, args.peak_layer, device)
            m_base = steered_margin(model, tok, prompt, device, tool_ids, fin_ids,
                                    None, 0.0, args.peak_layer)
            f.write(json.dumps({"sample_id": rec["sample_id"], "cond": "baseline",
                                "margin": m_base, "delta": 0.0,
                                "hidden_rms": rms_h}) + "\n"); n_rows += 1

            for name, vec in all_dirs:
                d_rms = compute_rms(vec)
                alpha = args.rho * (rms_h / d_rms)
                m_st = steered_margin(model, tok, prompt, device, tool_ids, fin_ids,
                                      vec, alpha, args.peak_layer)
                f.write(json.dumps({"sample_id": rec["sample_id"], "cond": name,
                                    "margin": m_st, "delta": m_st - m_base,
                                    "alpha": alpha}) + "\n"); n_rows += 1
            f.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(records):
                print(f"  [{i+1}/{len(records)}] {time.time()-t0:.1f}s rows={n_rows}")

    # Aggregate
    rows = [json.loads(l) for l in open(rows_path)]
    by_cond = {}
    for r in rows:
        by_cond.setdefault(r["cond"], []).append(r)
    baseline_margins = np.array([r["margin"] for r in by_cond["baseline"]])

    def stats(name):
        d = np.array([r["delta"] for r in by_cond[name]])
        return {"mean": float(d.mean()), "std": float(d.std()),
                "abs_mean": float(np.mean(np.abs(d))), "n": len(d)}

    full_s = stats("full"); par_s = stats("evi_parallel"); perp_s = stats("evi_perp")
    rand_means = np.array([np.mean([r["delta"] for r in by_cond[k]]) for k in rand_keys])
    rand_abs   = np.array([np.mean(np.abs([r["delta"] for r in by_cond[k]])) for k in rand_keys])
    pct_signed = float(percentileofscore(rand_means, par_s["mean"]))
    pct_abs    = float(percentileofscore(rand_abs, par_s["abs_mean"]))
    pct_full   = float(percentileofscore(rand_means, full_s["mean"]))

    summary = {
        "model": args.model_path, "peak_layer": args.peak_layer,
        "rho": args.rho, "n_samples": int(len(baseline_margins)),
        "n_random": args.n_random,
        "baseline_margin_mean": float(baseline_margins.mean()),
        "baseline_margin_std":  float(baseline_margins.std()),
        "geometry": geom,
        "full_mean_shift": full_s["mean"], "full_abs_mean_shift": full_s["abs_mean"],
        "evidence_parallel_mean_shift": par_s["mean"],
        "evidence_parallel_abs_mean_shift": par_s["abs_mean"],
        "evidence_perp_mean_shift": perp_s["mean"],
        "evidence_perp_abs_mean_shift": perp_s["abs_mean"],
        "random_mean_shifts_mean": float(rand_means.mean()),
        "random_mean_shifts_std":  float(rand_means.std()),
        "random_abs_mean_shifts_mean": float(rand_abs.mean()),
        "random_abs_mean_shifts_std":  float(rand_abs.std()),
        "random_mean_shifts_min": float(rand_means.min()),
        "random_mean_shifts_max": float(rand_means.max()),
        "evidence_percentile_signed": pct_signed,
        "evidence_percentile_abs": pct_abs,
        "full_percentile_signed": pct_full,
    }
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(f"\n  baseline margin = {summary['baseline_margin_mean']:+.3f} "
          f"± {summary['baseline_margin_std']:.3f}")
    print(f"  full           Δ={full_s['mean']:+.3f}  |Δ|={full_s['abs_mean']:.3f}")
    print(f"  evi_parallel   Δ={par_s['mean']:+.3f}  |Δ|={par_s['abs_mean']:.3f}  "
          f"(signed pct={pct_signed:.1f}%, |.|pct={pct_abs:.1f}%)")
    print(f"  evi_perp       Δ={perp_s['mean']:+.3f}  |Δ|={perp_s['abs_mean']:.3f}")
    print(f"  random (N={args.n_random}) mean Δ={rand_means.mean():+.3f}±{rand_means.std():.3f}  "
          f"|Δ|={rand_abs.mean():.3f}±{rand_abs.std():.3f}")
    print(f"  random Δ range: [{rand_means.min():+.3f}, {rand_means.max():+.3f}]")
    print(f"  full percentile in random = {pct_full:.1f}%")
    print(f"\n[wrote] {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
