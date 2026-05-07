#!/usr/bin/env python3
"""
Cross-Family CI-Hardened Functional Decomposition (Gemma-2-9B + Mistral-7B-v0.3)
================================================================================

Replicates the §8.3 / decomposition_ci_null protocol used on Qwen, but for
Gemma-2-9B-it (L37 same-layer) and Mistral-7B-Instruct-v0.3 (L28 same-layer),
WITH a no-renorm parallel injection added so the parallel/full ratio is
reported at both the RMS-normalized and natural component norms.

Per model, on N=50 N0 prompts (extractability_support_toggle), we measure:

    baseline      : no steering
    full_RMS      : action_dir RMS-normalized, alpha = rho * rms_h
    par_RMS       : (cos·evidence_unit) RMS-renormalized to RMS=1.0
    perp_RMS      : (action - parallel) RMS-renormalized to RMS=1.0
    par_natural   : (cos·evidence_unit) at natural L2=|cos|, alpha = rho*rms_h*sqrt(D)
    K=200 random  : RMS-normalized random unit-RMS directions (matches Qwen)
    K=200 random  : matched-natural-norm random (L2=|cos|) for the no-renorm test

Statistics:
  - Bootstrap 10k 95% CI for the mean of each named condition
  - Pairwise paired permutation tests (10k):
      a) par_RMS    vs random_per_prompt_RMS
      b) perp_RMS   vs full_RMS
      c) perp_RMS   vs par_RMS
      d) gap = perp_RMS - par_RMS exchangeability
      e) par_natural vs random_per_prompt_natural    (no-renorm sanity)
  - Stopping rule: |par_natural mean| < natural-norm null 97.5% bound on |Δm|

Usage:
  cd tmc/scripts/e2e_agent
  python scripts/decomposition_ci_hardened_cross_model.py --model gemma
  python scripts/decomposition_ci_hardened_cross_model.py --model mistral
  python scripts/decomposition_ci_hardened_cross_model.py --model both
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import percentileofscore
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS                  # noqa: E402
from steering.hook_utils import get_model_layers, SteeringHook, compute_rms  # noqa: E402
from scripts.cross_model_full import apply_chat_template_safe           # noqa: E402
from scripts.gemma_steering_sanity import (                              # noqa: E402
    get_hidden_rms_at_layer, steered_margin,
)


SEED  = 20260430
N_BOOT = 10_000
N_PERM = 10_000

MODEL_CFG = {
    "gemma": {
        "model_path": "unsloth/gemma-2-9b-it",
        "peak_layer": 37,
        "directions": "results/gemma_circuit_sanity/exp2_samelayer/directions.npz",
        "evidence_key": "evidence_dir_L37",
    },
    "mistral": {
        "model_path": "unsloth/mistral-7b-instruct-v0.3",
        "peak_layer": 28,
        "directions": "results/mistral_circuit_sanity/exp2_samelayer/directions.npz",
        "evidence_key": "evidence_dir_L37",  # filename quirk: actually L28 for mistral
    },
    # Qwen3-32B scale-check entry (§20.2 cross-family replication at 32B).
    # peak_layer and directions are overridable via --peak-layer / --directions CLI args;
    # they are populated from results/qwen3_32b_scale_check/full_results.json after Exp 1+2.
    "qwen3_32b": {
        "model_path": "/home/featurize/work/models/Qwen3-32B",
        "peak_layer": None,   # filled via --peak-layer at runtime
        "directions": "results/qwen3_32b_scale_check/directions.npz",
        "evidence_key": "evidence_dir",  # saved by cross_model_full.py with this key
    },
}


# ─── Statistics helpers (mirror decomposition_ci_null.py) ────────────────────

def bootstrap_ci(values, n_boot=N_BOOT, ci=95.0, seed=SEED):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if n > 1 else 0.0,
        "n": int(n),
        "ci_low":  float(np.percentile(boot_means, (100 - ci) / 2.0)),
        "ci_high": float(np.percentile(boot_means, 100 - (100 - ci) / 2.0)),
        "ci_level": ci,
        "boot_mean": float(boot_means.mean()),
        "boot_std":  float(boot_means.std(ddof=1)),
    }


def paired_permutation_test(a, b, n_perm=N_PERM, seed=SEED, two_sided=True):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    diff = a - b
    n = len(diff)
    obs = float(np.abs(diff.mean())) if two_sided else float(diff.mean())
    rng = np.random.RandomState(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_means = (signs * diff).mean(axis=1)
    if two_sided:
        ge = int(np.sum(np.abs(perm_means) >= obs))
    else:
        ge = int(np.sum(perm_means >= obs))
    p = (ge + 1) / (n_perm + 1)
    return {
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_diff": float((a - b).mean()),
        "abs_mean_diff": float(np.abs((a - b).mean())),
        "n_pairs": int(n), "n_perm": int(n_perm),
        "n_perm_ge_observed": int(ge),
        "p_value": float(p), "two_sided": bool(two_sided),
    }


def normalize_rms(v, target_rms=1.0):
    rms = float(np.sqrt(np.mean(v ** 2)))
    return v if rms < 1e-12 else (v * (target_rms / rms)).astype(np.float32)


# ─── Direction prep ──────────────────────────────────────────────────────────

def prepare_directions(npz_path, evidence_key):
    """Load action_dir and evidence_dir; build full / par / perp at RMS=1, plus
    par_natural at L2=|cos| (no renorm)."""
    d = np.load(npz_path)
    action  = d["action_dir"].astype(np.float32)
    if evidence_key not in d.files:
        for k in ["evidence_dir", "evidence_dir_L37", "evidence_dir_L28"]:
            if k in d.files:
                evidence_key = k; break
    evidence = d[evidence_key].astype(np.float32)
    a_unit = action  / np.linalg.norm(action)
    e_unit = evidence / np.linalg.norm(evidence)
    cos_ae = float(np.dot(a_unit, e_unit))

    par_natural = (float(np.dot(action, e_unit)) * e_unit).astype(np.float32)  # L2=|cos|
    perp_natural = (action - par_natural).astype(np.float32)                   # L2~1

    full_rms = normalize_rms(action,       1.0)
    par_rms  = normalize_rms(par_natural,  1.0)
    perp_rms = normalize_rms(perp_natural, 1.0)
    return {
        "action": action, "evidence": evidence,
        "cos_action_evidence": cos_ae,
        "par_natural": par_natural, "perp_natural": perp_natural,
        "full_rms": full_rms, "par_rms": par_rms, "perp_rms": perp_rms,
    }


# ─── Per-model run ───────────────────────────────────────────────────────────

def run_model(model_key, args):
    cfg = MODEL_CFG[model_key]
    out_dir = Path(args.output_dir) / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*72}\n[run] model={model_key}  layer=L{cfg['peak_layer']}  rho={args.rho}"
          f"  N={args.limit}  K_rms={args.k_random_rms}  K_nat={args.k_random_natural}\n{'='*72}")

    # ── Directions ──────────────────────────────────────────────────────────
    dirs = prepare_directions(cfg["directions"], cfg["evidence_key"])
    D = dirs["action"].shape[0]
    sqrtD = float(np.sqrt(D))
    cos_ae = dirs["cos_action_evidence"]

    par_nat_L2 = float(np.linalg.norm(dirs["par_natural"]))
    perp_nat_L2 = float(np.linalg.norm(dirs["perp_natural"]))
    print(f"[geom] D={D}  cos(a,e)={cos_ae:+.6f}")
    print(f"[geom] L2 par_natural={par_nat_L2:.6f}  perp_natural={perp_nat_L2:.6f}")
    print(f"[geom] amplification (rms-norm L2 / natural L2) = "
          f"{(np.linalg.norm(dirs['par_rms']) / max(par_nat_L2, 1e-12)):.1f}x")

    # ── K=200 random RMS-normalized directions (matches Qwen) ───────────────
    _seed_offset = {"gemma": 1, "mistral": 2, "qwen3_32b": 3}.get(model_key, 2)
    rng = np.random.RandomState(SEED + _seed_offset)
    rand_unit  = [(r / np.linalg.norm(r)).astype(np.float32)
                  for r in rng.standard_normal((args.k_random_rms, D)).astype(np.float32)]
    rand_rms   = [normalize_rms(r, 1.0) for r in rand_unit]
    # K=100 random unit-L2 for matched-natural-norm null
    rng2 = np.random.RandomState(SEED + 100 + _seed_offset)
    rand_unit_nat = [(r / np.linalg.norm(r)).astype(np.float32)
                     for r in rng2.standard_normal((args.k_random_natural, D)).astype(np.float32)]

    # ── Load model ──────────────────────────────────────────────────────────
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"[load] {cfg['model_path']} dtype={args.dtype}")
    tok = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"], torch_dtype=dtype, device_map="auto",
        trust_remote_code=True)
    model.eval(); device = next(model.parameters()).device
    tool_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]
    builder = PromptBuilder()

    # ── Prompts: N0 condition from extractability_support_toggle/pairs.jsonl
    records_all = [json.loads(l) for l in open(args.steering_pairs)]
    records = [r for r in records_all if r.get("condition") == args.steering_cond][:args.limit]
    print(f"[prompts] cond={args.steering_cond} N={len(records)}")

    n = len(records)
    K_rms = args.k_random_rms; K_nat = args.k_random_natural
    base       = np.zeros(n, dtype=np.float32)
    full_m     = np.zeros(n, dtype=np.float32)
    par_m_rms  = np.zeros(n, dtype=np.float32)
    perp_m_rms = np.zeros(n, dtype=np.float32)
    par_m_nat  = np.zeros(n, dtype=np.float32)
    rand_m_rms = np.zeros((K_rms, n), dtype=np.float32)
    rand_m_nat = np.zeros((K_nat, n), dtype=np.float32)
    rms_h_arr  = np.zeros(n, dtype=np.float32)

    rows_path = out_dir / "per_example_rows.jsonl"
    fout = open(rows_path, "w")
    t0 = time.time()
    n_total_fw = n * (4 + K_rms + K_nat)  # baseline+rms_h share a forward; full,par_rms,perp_rms,par_nat
    fw_done = 0

    for i, rec in enumerate(records):
        steps = [{"action": "search",
                  "action_input": f"about: {rec['question'][:80]}",
                  "observation": rec["obs"]}]
        msgs = builder.build_full_prompt(rec["question"], steps)
        prompt = apply_chat_template_safe(tok, msgs, add_generation_prompt=True)

        rms_h, _ = get_hidden_rms_at_layer(model, tok, prompt, cfg["peak_layer"], device)
        rms_h_arr[i] = rms_h
        # alpha conventions
        alpha_rms     = args.rho * rms_h                   # for direction with RMS=1
        alpha_natural = args.rho * rms_h * sqrtD           # for unit-L2 direction (full action)
        # equivalent random matched to par_natural L2 = |cos|
        alpha_random_natural = alpha_natural * par_nat_L2  # since rand is unit-L2

        m_base = steered_margin(model, tok, prompt, device, tool_ids, fin_ids,
                                None, 0.0, cfg["peak_layer"]); fw_done += 1
        base[i] = m_base
        m_full = steered_margin(model, tok, prompt, device, tool_ids, fin_ids,
                                dirs["full_rms"], alpha_rms, cfg["peak_layer"]); fw_done += 1
        m_par_r = steered_margin(model, tok, prompt, device, tool_ids, fin_ids,
                                 dirs["par_rms"],  alpha_rms, cfg["peak_layer"]); fw_done += 1
        m_perp_r = steered_margin(model, tok, prompt, device, tool_ids, fin_ids,
                                  dirs["perp_rms"], alpha_rms, cfg["peak_layer"]); fw_done += 1
        m_par_n = steered_margin(model, tok, prompt, device, tool_ids, fin_ids,
                                 dirs["par_natural"], alpha_natural, cfg["peak_layer"]); fw_done += 1
        full_m[i]     = m_full
        par_m_rms[i]  = m_par_r
        perp_m_rms[i] = m_perp_r
        par_m_nat[i]  = m_par_n
        fout.write(json.dumps({
            "sample_id": rec["sample_id"], "i": i,
            "baseline_margin": float(m_base), "rms_h": float(rms_h),
            "full_shift": float(m_full - m_base),
            "par_rms_shift":  float(m_par_r - m_base),
            "perp_rms_shift": float(m_perp_r - m_base),
            "par_natural_shift": float(m_par_n - m_base),
        }) + "\n")

        for k in range(K_rms):
            m = steered_margin(model, tok, prompt, device, tool_ids, fin_ids,
                               rand_rms[k], alpha_rms, cfg["peak_layer"])
            rand_m_rms[k, i] = m; fw_done += 1
        for k in range(K_nat):
            m = steered_margin(model, tok, prompt, device, tool_ids, fin_ids,
                               rand_unit_nat[k], alpha_random_natural, cfg["peak_layer"])
            rand_m_nat[k, i] = m; fw_done += 1
        fout.flush()

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (n - i - 1)
        if (i + 1) % 1 == 0:
            print(f"  [{i+1}/{n}] {elapsed:.0f}s elapsed ~{eta:.0f}s ETA "
                  f"({fw_done}/{n_total_fw} fw)  "
                  f"base={m_base:+.2f} full={m_full-m_base:+.2f} "
                  f"par_R={m_par_r-m_base:+.2f} perp_R={m_perp_r-m_base:+.2f} "
                  f"par_N={m_par_n-m_base:+.4f}")
    fout.close()

    # ── Per-example shifts ──────────────────────────────────────────────────
    full_sh   = full_m     - base
    par_sh_r  = par_m_rms  - base
    perp_sh_r = perp_m_rms - base
    par_sh_n  = par_m_nat  - base
    rand_sh_rms = rand_m_rms - base[None, :]
    rand_sh_nat = rand_m_nat - base[None, :]
    rand_per_prompt_rms = rand_sh_rms.mean(axis=0)
    rand_per_prompt_nat = rand_sh_nat.mean(axis=0)
    rand_per_dir_rms_means = rand_sh_rms.mean(axis=1)
    rand_per_dir_rms_abs   = np.abs(rand_sh_rms).mean(axis=1)
    rand_per_dir_nat_means = rand_sh_nat.mean(axis=1)
    rand_per_dir_nat_abs   = np.abs(rand_sh_nat).mean(axis=1)

    np.savez(out_dir / "per_example_shifts.npz",
             baseline=base, rms_h=rms_h_arr,
             full=full_sh, par_rms=par_sh_r, perp_rms=perp_sh_r, par_natural=par_sh_n,
             random_rms=rand_sh_rms, random_natural=rand_sh_nat)

    summary = _summarize(model_key, cfg, args, dirs, out_dir,
                         base, full_sh, par_sh_r, perp_sh_r, par_sh_n,
                         rand_per_prompt_rms, rand_per_prompt_nat,
                         rand_per_dir_rms_means, rand_per_dir_rms_abs,
                         rand_per_dir_nat_means, rand_per_dir_nat_abs)
    del model, tok
    import gc; gc.collect(); torch.cuda.empty_cache()
    return summary


# ─── Summary / stats ─────────────────────────────────────────────────────────

def _summarize(model_key, cfg, args, dirs, out_dir,
               base, full_sh, par_sh_r, perp_sh_r, par_sh_n,
               rand_per_prompt_rms, rand_per_prompt_nat,
               rand_per_dir_rms_means, rand_per_dir_rms_abs,
               rand_per_dir_nat_means, rand_per_dir_nat_abs):
    n = len(base)
    # Bootstrap CIs
    bootstrap = {
        "full":         bootstrap_ci(full_sh,  N_BOOT, 95.0, SEED + 11),
        "par_rms":      bootstrap_ci(par_sh_r, N_BOOT, 95.0, SEED + 12),
        "perp_rms":     bootstrap_ci(perp_sh_r,N_BOOT, 95.0, SEED + 13),
        "par_natural":  bootstrap_ci(par_sh_n, N_BOOT, 95.0, SEED + 14),
    }
    # Pairwise permutation tests (mirrors Qwen)
    test_a = paired_permutation_test(par_sh_r,  rand_per_prompt_rms, N_PERM, SEED + 21)
    test_b = paired_permutation_test(perp_sh_r, full_sh,             N_PERM, SEED + 22)
    test_c = paired_permutation_test(perp_sh_r, par_sh_r,            N_PERM, SEED + 23)
    # Dissociation gap exchangeability
    obs_gap = float(perp_sh_r.mean() - par_sh_r.mean())
    diff_pp = perp_sh_r.astype(np.float64) - par_sh_r.astype(np.float64)
    rng_d = np.random.RandomState(SEED + 24)
    signs = rng_d.choice([-1.0, 1.0], size=(N_PERM, n))
    perm_gaps = (signs * diff_pp).mean(axis=1)
    ge_d = int(np.sum(np.abs(perm_gaps) >= abs(obs_gap)))
    test_d = {"observed_gap_perp_minus_par": obs_gap,
              "abs_observed_gap": abs(obs_gap),
              "n_perm": int(N_PERM),
              "n_perm_ge_observed_abs": ge_d,
              "p_value": (ge_d + 1) / (N_PERM + 1)}
    # No-renorm parallel vs matched-natural-norm random
    test_e = paired_permutation_test(par_sh_n, rand_per_prompt_nat, N_PERM, SEED + 25)

    # Random null bands (RMS-normalized, K=200)
    null_rms = {
        "K": int(len(rand_per_dir_rms_means)),
        "n_samples_per_dir": int(n),
        "signed": {
            "mean": float(rand_per_dir_rms_means.mean()),
            "std":  float(rand_per_dir_rms_means.std(ddof=1)),
            "p2_5": float(np.percentile(rand_per_dir_rms_means, 2.5)),
            "p97_5":float(np.percentile(rand_per_dir_rms_means, 97.5)),
            "p50":  float(np.percentile(rand_per_dir_rms_means, 50.0)),
            "min":  float(rand_per_dir_rms_means.min()),
            "max":  float(rand_per_dir_rms_means.max()),
        },
        "abs": {
            "mean": float(rand_per_dir_rms_abs.mean()),
            "std":  float(rand_per_dir_rms_abs.std(ddof=1)),
            "p97_5":float(np.percentile(rand_per_dir_rms_abs, 97.5)),
        },
    }
    # Natural-norm null (matched to par_natural L2)
    null_nat = {
        "K": int(len(rand_per_dir_nat_means)),
        "n_samples_per_dir": int(n),
        "signed": {
            "mean": float(rand_per_dir_nat_means.mean()),
            "std":  float(rand_per_dir_nat_means.std(ddof=1)),
            "p2_5": float(np.percentile(rand_per_dir_nat_means, 2.5)),
            "p97_5":float(np.percentile(rand_per_dir_nat_means, 97.5)),
            "p50":  float(np.percentile(rand_per_dir_nat_means, 50.0)),
        },
        "abs": {
            "mean": float(rand_per_dir_nat_abs.mean()),
            "std":  float(rand_per_dir_nat_abs.std(ddof=1)),
            "p97_5":float(np.percentile(rand_per_dir_nat_abs, 97.5)),
        },
    }
    # Stopping rule on the no-renorm parallel
    pct_par_n_signed = float(percentileofscore(rand_per_dir_nat_means, par_sh_n.mean()))
    pct_par_n_abs    = float(percentileofscore(rand_per_dir_nat_abs,
                                               float(np.abs(par_sh_n).mean())))
    in_null_band = bool(abs(par_sh_n.mean()) < null_nat["abs"]["p97_5"])
    verdict = ("residual-explained-by-normalization"
               if in_null_band else "orthogonal-dominant-with-residual")

    full_mean = float(full_sh.mean())
    par_r_mean = float(par_sh_r.mean()); perp_r_mean = float(perp_sh_r.mean())
    par_n_mean = float(par_sh_n.mean())
    par_n_L2 = float(np.linalg.norm(dirs["par_natural"]))

    summary = {
        "model_key": model_key, "model": cfg["model_path"],
        "config": {
            "layer": cfg["peak_layer"], "rho": args.rho,
            "n_samples": int(n),
            "K_random_rms": int(args.k_random_rms),
            "K_random_natural": int(args.k_random_natural),
            "n_boot": N_BOOT, "n_perm": N_PERM, "seed": SEED,
            "directions": cfg["directions"],
        },
        "geometry": {
            "d_model": int(dirs["action"].shape[0]),
            "cos_action_evidence": dirs["cos_action_evidence"],
            "parallel_norm_natural_L2":  par_n_L2,
            "parallel_norm_rms_L2":      float(np.linalg.norm(dirs["par_rms"])),
            "amplification_factor_rms_over_natural":
                float(np.linalg.norm(dirs["par_rms"]) / max(par_n_L2, 1e-12)),
        },
        "point_estimates": {
            "baseline_margin_mean": float(base.mean()),
            "full_mean_shift": full_mean,
            "par_rms_mean_shift":  par_r_mean,
            "perp_rms_mean_shift": perp_r_mean,
            "par_natural_mean_shift": par_n_mean,
            "par_rms_abs_mean_shift":  float(np.abs(par_sh_r).mean()),
            "par_natural_abs_mean_shift": float(np.abs(par_sh_n).mean()),
            "random_per_prompt_rms_mean": float(rand_per_prompt_rms.mean()),
            "random_per_prompt_nat_mean": float(rand_per_prompt_nat.mean()),
        },
        "bootstrap_95ci": {k: {"mean": v["mean"], "ci_low": v["ci_low"],
                               "ci_high": v["ci_high"], "n": v["n"]}
                           for k, v in bootstrap.items()},
        "ratios_to_full": {
            "par_rms_over_full":      par_r_mean / full_mean if abs(full_mean) > 1e-9 else None,
            "par_natural_over_full":  par_n_mean / full_mean if abs(full_mean) > 1e-9 else None,
            "perp_rms_over_full":     perp_r_mean / full_mean if abs(full_mean) > 1e-9 else None,
        },
        "random_null_rms_K": null_rms,
        "random_null_natural_K": null_nat,
        "stopping_rule_no_renorm_parallel": {
            "criterion": "|par_natural mean| < natural-norm null abs_p97_5",
            "lhs": float(abs(par_n_mean)),
            "rhs": null_nat["abs"]["p97_5"],
            "in_null_band": in_null_band,
            "verdict": verdict,
            "percentile_signed_in_natural_null": pct_par_n_signed,
            "percentile_abs_in_natural_null":    pct_par_n_abs,
        },
        "pairwise_p_values": {
            "a_par_rms_vs_random_per_prompt_rms": test_a["p_value"],
            "b_perp_vs_full":                     test_b["p_value"],
            "c_perp_vs_par_rms":                  test_c["p_value"],
            "d_dissociation_gap":                 test_d["p_value"],
            "e_par_natural_vs_random_per_prompt_natural": test_e["p_value"],
        },
        "pairwise_tests_full": {
            "a_par_rms_vs_random_rms":    test_a,
            "b_perp_vs_full":             test_b,
            "c_perp_vs_par_rms":          test_c,
            "d_dissociation_gap":         test_d,
            "e_par_natural_vs_random_natural": test_e,
        },
    }
    json.dump(summary, open(out_dir / f"{model_key}_decomposition.json", "w"), indent=2)
    print(f"[save] {out_dir/(model_key+'_decomposition.json')}")
    return summary


# ─── Cross-family table + report ─────────────────────────────────────────────

QWEN_REF = {
    "model": "Qwen/Qwen2.5-7B-Instruct", "layer": 20,
    "n_samples": 100, "K_random_rms": 200,
    "cos_action_evidence": -0.0135,  # known constant from §8.3
    "full":      {"mean":  0.910, "ci_low":  0.841, "ci_high":  0.980},
    "par_rms":   {"mean": -0.157, "ci_low": -0.192, "ci_high": -0.122},
    "perp_rms":  {"mean":  0.909, "ci_low":  0.839, "ci_high":  0.979},
    "par_natural": None,  # not measured for Qwen in §8.3 (small cos)
    "ratio_par_rms_over_full":  -0.157 / 0.910,
    "ratio_perp_over_full":      0.909 / 0.910,
    "p_par_vs_random":  9.999e-05,
    "p_perp_vs_full":   0.9043,
    "p_perp_vs_par":    9.999e-05,
    "p_dissociation":   9.999e-05,
}


def write_cross_family_table(summaries, out_dir):
    table = {"qwen": QWEN_REF}
    for s in summaries:
        bk = s["bootstrap_95ci"]
        table[s["model_key"]] = {
            "model": s["model"], "layer": s["config"]["layer"],
            "n_samples": s["config"]["n_samples"],
            "K_random_rms": s["config"]["K_random_rms"],
            "cos_action_evidence": s["geometry"]["cos_action_evidence"],
            "full":         {"mean": bk["full"]["mean"],
                             "ci_low": bk["full"]["ci_low"],
                             "ci_high": bk["full"]["ci_high"]},
            "par_rms":      {"mean": bk["par_rms"]["mean"],
                             "ci_low": bk["par_rms"]["ci_low"],
                             "ci_high": bk["par_rms"]["ci_high"]},
            "perp_rms":     {"mean": bk["perp_rms"]["mean"],
                             "ci_low": bk["perp_rms"]["ci_low"],
                             "ci_high": bk["perp_rms"]["ci_high"]},
            "par_natural":  {"mean": bk["par_natural"]["mean"],
                             "ci_low": bk["par_natural"]["ci_low"],
                             "ci_high": bk["par_natural"]["ci_high"]},
            "ratio_par_rms_over_full":     s["ratios_to_full"]["par_rms_over_full"],
            "ratio_par_natural_over_full": s["ratios_to_full"]["par_natural_over_full"],
            "ratio_perp_over_full":        s["ratios_to_full"]["perp_rms_over_full"],
            "amplification_rms_over_natural":
                s["geometry"]["amplification_factor_rms_over_natural"],
            "p_par_vs_random":     s["pairwise_p_values"]["a_par_rms_vs_random_per_prompt_rms"],
            "p_perp_vs_full":      s["pairwise_p_values"]["b_perp_vs_full"],
            "p_perp_vs_par":       s["pairwise_p_values"]["c_perp_vs_par_rms"],
            "p_dissociation":      s["pairwise_p_values"]["d_dissociation_gap"],
            "p_par_natural_vs_random_natural":
                s["pairwise_p_values"]["e_par_natural_vs_random_per_prompt_natural"],
            "stopping_rule_no_renorm": s["stopping_rule_no_renorm_parallel"],
        }
    json.dump(table, open(out_dir / "crossfamily_table.json", "w"), indent=2)
    print(f"[save] {out_dir/'crossfamily_table.json'}")
    return table


def write_three_panel_figure(table, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ["qwen", "gemma", "mistral"]
    titles = ["Qwen2.5-7B (L20)", "Gemma-2-9B (L37)", "Mistral-7B-v0.3 (L28)"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=False)
    bar_labels = ["full", "par_rms", "par_nat", "perp_rms"]
    bar_colors = ["#1f77b4", "#d62728", "#ff9f9b", "#2ca02c"]
    for ax, k, ti in zip(axes, keys, titles):
        if k not in table:
            ax.set_title(f"{ti}\n(missing)"); continue
        t = table[k]
        means = [t["full"]["mean"], t["par_rms"]["mean"],
                 (t["par_natural"]["mean"] if t.get("par_natural") else 0.0),
                 t["perp_rms"]["mean"]]
        lows  = [t["full"]["ci_low"], t["par_rms"]["ci_low"],
                 (t["par_natural"]["ci_low"] if t.get("par_natural") else 0.0),
                 t["perp_rms"]["ci_low"]]
        highs = [t["full"]["ci_high"], t["par_rms"]["ci_high"],
                 (t["par_natural"]["ci_high"] if t.get("par_natural") else 0.0),
                 t["perp_rms"]["ci_high"]]
        err_lo = [m - lo for m, lo in zip(means, lows)]
        err_hi = [hi - m for m, hi in zip(means, highs)]
        x = np.arange(len(means))
        bars = ax.bar(x, means, yerr=[err_lo, err_hi], color=bar_colors,
                      capsize=3, edgecolor="black", linewidth=0.6)
        if k == "qwen":
            bars[2].set_alpha(0.25)
            ax.text(2, 0.0, "n/a", ha="center", va="center", fontsize=8, color="grey")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, rotation=20, ha="right", fontsize=9)
        ax.set_title(f"{ti}  cos={t['cos_action_evidence']:+.4f}", fontsize=10)
        ax.set_ylabel("Δ margin (logits)")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        # annotate ratios
        if t.get("par_natural"):
            r_n = t.get("ratio_par_natural_over_full")
            ax.text(2, max(highs[2], 0) + 0.05, f"{r_n*100:+.1f}% of full",
                    ha="center", fontsize=8)
        r_par_rms = t.get("ratio_par_rms_over_full")
        if r_par_rms is not None:
            ax.text(1, max(highs[1], 0) + 0.05, f"{r_par_rms*100:+.1f}% of full",
                    ha="center", fontsize=8)
    plt.suptitle("Cross-family functional decomposition at action peak layer\n"
                 "(N0 prompts, ρ=−0.20, 95% bootstrap CIs)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_dir / "figure_three_panel.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "figure_three_panel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_dir/'figure_three_panel.{pdf,png}'}")


def write_report(table, summaries, out_dir):
    md = ["# Cross-family CI-Hardened Functional Decomposition\n"]
    md.append("Replicates the §8.3 Qwen decomposition_ci_null protocol on Gemma-2-9B-it (L37) "
              "and Mistral-7B-Instruct-v0.3 (L28), with the addition of a no-renorm parallel "
              "injection so RMS-renormalization artefacts are visible.\n")
    md.append("## Summary table\n")
    md.append("| model | layer | cos(a,e) | full | par_RMS | par_natural | perp | par_RMS / full | par_nat / full | perp / full |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for k in ["qwen", "gemma", "mistral"]:
        if k not in table: continue
        t = table[k]
        full_s = f"{t['full']['mean']:+.3f} [{t['full']['ci_low']:+.3f},{t['full']['ci_high']:+.3f}]"
        par_s  = f"{t['par_rms']['mean']:+.3f} [{t['par_rms']['ci_low']:+.3f},{t['par_rms']['ci_high']:+.3f}]"
        if t.get("par_natural"):
            pn = t["par_natural"]
            par_n_s = f"{pn['mean']:+.4f} [{pn['ci_low']:+.4f},{pn['ci_high']:+.4f}]"
            r_n = f"{t['ratio_par_natural_over_full']*100:+.1f}%"
        else:
            par_n_s = "n/a"; r_n = "n/a"
        perp_s = f"{t['perp_rms']['mean']:+.3f} [{t['perp_rms']['ci_low']:+.3f},{t['perp_rms']['ci_high']:+.3f}]"
        r_par = f"{t['ratio_par_rms_over_full']*100:+.1f}%" if t.get('ratio_par_rms_over_full') else "n/a"
        r_perp = f"{t['ratio_perp_over_full']*100:+.1f}%"
        md.append(f"| {t['model']} | L{t['layer']} | {t['cos_action_evidence']:+.4f} | "
                  f"{full_s} | {par_s} | {par_n_s} | {perp_s} | {r_par} | {r_n} | {r_perp} |")
    md.append("\n## Stopping rule (par_natural inside matched-natural-norm null band)\n")
    md.append("| model | par_nat mean | natural-norm null abs_p97.5 | in_null_band | verdict |")
    md.append("|---|---|---|---|---|")
    for s in summaries:
        sr = s["stopping_rule_no_renorm_parallel"]
        md.append(f"| {s['model_key']} | {s['point_estimates']['par_natural_mean_shift']:+.4f} | "
                  f"{sr['rhs']:.4f} | **{sr['in_null_band']}** | **{sr['verdict']}** |")
    md.append("\n## Pairwise permutation p-values (10k)\n")
    md.append("| model | par_RMS vs random | perp vs full | perp vs par_RMS | dissociation gap | par_nat vs random_nat |")
    md.append("|---|---|---|---|---|---|")
    for k in ["qwen", "gemma", "mistral"]:
        if k not in table: continue
        t = table[k]
        e = t.get("p_par_natural_vs_random_natural", None)
        e_s = f"{e:.4f}" if e is not None else "n/a"
        md.append(f"| {t['model']} | {t['p_par_vs_random']:.4f} | {t['p_perp_vs_full']:.4f} | "
                  f"{t['p_perp_vs_par']:.4f} | {t['p_dissociation']:.4f} | {e_s} |")
    md.append("\n## Geometry / amplification factors\n")
    for s in summaries:
        g = s["geometry"]
        md.append(f"- **{s['model_key']}** ({s['model']}, L{s['config']['layer']}): "
                  f"D={g['d_model']}, |cos|={abs(g['cos_action_evidence']):.4f}, "
                  f"par_natural L2 = {g['parallel_norm_natural_L2']:.6f}, "
                  f"par_rms L2 = {g['parallel_norm_rms_L2']:.3f}, "
                  f"amplification factor = {g['amplification_factor_rms_over_natural']:.1f}x")
    md.append("\n## Pass / partial / fail criteria\n")
    md.append("- **PASS** for a model if par_natural inside matched null band AND perp recovers ≥85% of full.")
    md.append("- **FAIL** if par_natural carries >30% of full at natural norm.")
    for s in summaries:
        sr = s["stopping_rule_no_renorm_parallel"]
        rt = s["ratios_to_full"]
        perp_share = abs(rt["perp_rms_over_full"]) if rt["perp_rms_over_full"] is not None else 0.0
        par_n_share = abs(rt["par_natural_over_full"]) if rt["par_natural_over_full"] is not None else 0.0
        if sr["in_null_band"] and perp_share >= 0.85:
            verdict_overall = "PASS"
        elif par_n_share > 0.30:
            verdict_overall = "FAIL"
        else:
            verdict_overall = "PARTIAL"
        md.append(f"- **{s['model_key']}**: in_null_band={sr['in_null_band']}, "
                  f"perp_share={perp_share*100:.1f}%, par_nat_share={par_n_share*100:.1f}% → "
                  f"**{verdict_overall}**")
    open(out_dir / "report.md", "w").write("\n".join(md) + "\n")
    print(f"[save] {out_dir/'report.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gemma", "mistral", "both", "qwen3_32b"],
                    default="both")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--steering-pairs",
                    default="results/extractability_support_toggle/pairs.jsonl")
    ap.add_argument("--steering-cond", default="N0")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--rho", type=float, default=-0.20)
    ap.add_argument("--k-random-rms", type=int, default=200)
    ap.add_argument("--k-random-natural", type=int, default=100)
    ap.add_argument("--output-dir", default="results/crossfamily_ci_decomposition")
    ap.add_argument("--skip-figure", action="store_true")
    # CLI overrides for peak_layer / directions (required for qwen3_32b whose values
    # are determined at runtime from Exp 1+2 output):
    ap.add_argument("--peak-layer", type=int, default=None,
                    help="Override MODEL_CFG[model].peak_layer (used for qwen3_32b).")
    ap.add_argument("--directions", type=str, default=None,
                    help="Override MODEL_CFG[model].directions npz path.")
    args = ap.parse_args()

    # Apply CLI overrides into MODEL_CFG
    if args.peak_layer is not None:
        MODEL_CFG.get(args.model, {})["peak_layer"] = args.peak_layer
    if args.directions is not None:
        MODEL_CFG.get(args.model, {})["directions"] = args.directions

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    keys = ["gemma", "mistral"] if args.model == "both" else [args.model]
    for k in keys:
        s = run_model(k, args)
        summaries.append(s)
        # Free memory before loading next model
        torch.cuda.empty_cache()

    # If only one model run, try loading the other from existing JSON
    for k in ["gemma", "mistral"]:
        if k in [s["model_key"] for s in summaries]:
            continue
        cached = Path(args.output_dir) / k / f"{k}_decomposition.json"
        if cached.exists():
            summaries.append(json.load(open(cached)))
            print(f"[load] reused cached {cached}")

    table = write_cross_family_table(summaries, out_dir)
    json.dump({"qwen_reference": QWEN_REF,
               "models_run": [s["model_key"] for s in summaries],
               "config": {"limit": args.limit, "rho": args.rho,
                          "K_rms": args.k_random_rms,
                          "K_nat": args.k_random_natural,
                          "n_boot": N_BOOT, "n_perm": N_PERM, "seed": SEED}},
              open(out_dir / "summary.json", "w"), indent=2)
    print(f"[save] {out_dir/'summary.json'}")
    if not args.skip_figure:
        try:
            write_three_panel_figure(table, out_dir)
        except Exception as ex:
            print(f"[warn] figure failed: {ex}")
    write_report(table, summaries, out_dir)


if __name__ == "__main__":
    main()

