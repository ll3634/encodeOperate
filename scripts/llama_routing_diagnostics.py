#!/usr/bin/env python3
"""Phase-α diagnostics for the Llama paired-corruption null.

E5: evidence-direction sanity gate (positive control on corruption protocol)
E3: cos(action_dir_PopQA, action_dir_step1_own) per model
E1: Llama-own action_dir from sign(step1_margin) diff-of-means → AB ratio
E2: same protocol on Qwen / Mistral / Gemma controls (R1 excluded — degenerate)

Inputs:  results/cross_model_*_v2/per_sample.npz
Outputs: results/llama_routing_diagnostics/{summary.json, README.md}
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
from scipy.stats import mannwhitneyu

SEED = 20260503
N_BOOT = 10000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "llama_routing_diagnostics")

MODELS = [
    ("qwen25_7b",   "cross_model_qwen25_v2"),
    ("mistral_7b",  "cross_model_mistral_v2"),
    ("llama31_8b",  "cross_model_llama31_v2"),
    ("gemma2_9b",   "cross_model_gemma2_v2"),
    ("r1distill_7b","cross_model_r1distill_v2"),
]


def geom_median(x, n_iter=200, eps=1e-9):
    y = float(np.median(x))
    for _ in range(n_iter):
        d = np.maximum(np.abs(x - y), eps)
        w = 1.0 / d
        y_new = float(np.sum(w * x) / np.sum(w))
        if abs(y_new - y) < eps:
            break
        y = y_new
    return y


def lognormal_boot_ratio_ci(a, b, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    na, nb = len(a), len(b)
    lr = np.empty(n_boot)
    for i in range(n_boot):
        ai = a[rng.integers(0, na, na)]
        bi = b[rng.integers(0, nb, nb)]
        lr[i] = np.log(geom_median(ai)) - np.log(geom_median(bi))
    return float(np.exp(np.quantile(lr, 0.025))), float(np.exp(np.quantile(lr, 0.975)))


def project_pair_delta(pair_h_clean, pair_h_corrupted, direction):
    """Return (3, n_pairs) array of |Δ projection| per pair per group."""
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    proj_c = pair_h_clean @ direction
    proj_x = pair_h_corrupted @ direction
    return np.abs(proj_c - proj_x)


def ab_stats(d_proj, group_idx):
    a = d_proj[group_idx["A"]].astype(np.float64)
    b = d_proj[group_idx["B"]].astype(np.float64)
    c = d_proj[group_idx["C"]].astype(np.float64)
    gmA, gmB, gmC = geom_median(a), geom_median(b), geom_median(c)
    ratio = gmA / gmB if gmB > 0 else float("nan")
    lo, hi = lognormal_boot_ratio_ci(a, b)
    mw2 = mannwhitneyu(a, b, alternative="two-sided")
    mw1 = mannwhitneyu(a, b, alternative="greater")
    return {"gm_A": gmA, "gm_B": gmB, "gm_C": gmC,
            "AB_ratio": float(ratio), "CI95": [lo, hi],
            "MW_p_two": float(mw2.pvalue), "MW_p_one_greater": float(mw1.pvalue)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for short, sub in MODELS:
        z = np.load(os.path.join(ROOT, "results", sub, "per_sample.npz"), allow_pickle=False)
        groups = list(z["pair_groups"])
        gi = {g: i for i, g in enumerate(groups)}
        ph_c = z["pair_h_clean"]
        ph_x = z["pair_h_corrupted"]
        evi_dir = z["evidence_dir"]
        act_dir_pop = z["action_dir"]
        step1_h = z["step1_h"]
        step1_m = z["step1_margin"]

        # E5 — evidence-direction positive control
        d_evi = project_pair_delta(ph_c, ph_x, evi_dir)
        e5 = ab_stats(d_evi, gi)

        # echo: PopQA action_dir result (matches Phase-5b)
        d_act_pop = project_pair_delta(ph_c, ph_x, act_dir_pop)
        e_pop = ab_stats(d_act_pop, gi)

        # E1/E2 — own-direction from sign(step1_margin) diff-of-means
        n_pos = int((step1_m > 0).sum())
        n_neg = int((step1_m < 0).sum())
        own = {"feasible": (n_pos >= 10 and n_neg >= 10),
               "n_search": n_pos, "n_stop": n_neg}
        if own["feasible"]:
            mu_s = step1_h[step1_m > 0].mean(axis=0)
            mu_t = step1_h[step1_m < 0].mean(axis=0)
            own_dir = mu_s - mu_t
            own_dir = own_dir / (np.linalg.norm(own_dir) + 1e-12)
            cos_pop_own = float(np.dot(own_dir, act_dir_pop /
                                        (np.linalg.norm(act_dir_pop) + 1e-12)))
            d_act_own = project_pair_delta(ph_c, ph_x, own_dir)
            e_own = ab_stats(d_act_own, gi)
            own.update({"cos_with_PopQA_action_dir": cos_pop_own,
                        "ab_stats": e_own})
        else:
            own.update({"reason": "degenerate sign(step1_margin) — labels uniform"})

        rows.append({
            "model_short": short,
            "peak_evi_layer": int(z["peak_evi_layer"]),
            "peak_act_layer": int(z["peak_act_layer"]),
            "n_pairs_per_group": int(ph_c.shape[1]),
            "E5_evidence_direction": e5,
            "echo_PopQA_action_direction": e_pop,
            "E1E2_own_step1_direction": own,
        })

    summary = {
        "spec_version": "phase-alpha-llama-diagnostics-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "experiments": {
            "E5": "Project paired pair_h onto evidence_dir; sanity-gate that A>B and corruption protocol moves the residual.",
            "E3": "cos(action_dir_PopQA, action_dir_own_step1) per model; embedded inside E1 row when feasible.",
            "E1": "Re-extract action_dir via diff-of-means on sign(step1_margin) on step1_h; re-project pair_h; recompute AB ratio.",
            "E2": "Same as E1 applied to Qwen/Mistral/Gemma controls; R1 excluded (degenerate labels).",
        },
        "rows": rows,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    write_readme(rows, summary)
    print_table(rows)


def print_table(rows):
    print(f"{'model':<14} {'E5_evi_AB':>10} {'E5_p2':>10} {'pop_AB':>8} {'pop_p2':>10} {'cos':>7} {'own_AB':>8} {'own_p2':>10}")
    for r in rows:
        e5 = r["E5_evidence_direction"]; ep = r["echo_PopQA_action_direction"]
        own = r["E1E2_own_step1_direction"]
        if own["feasible"]:
            o = own["ab_stats"]; cs = own["cos_with_PopQA_action_dir"]
            print(f"{r['model_short']:<14} {e5['AB_ratio']:>10.4f} {e5['MW_p_two']:>10.4g} "
                  f"{ep['AB_ratio']:>8.4f} {ep['MW_p_two']:>10.4g} "
                  f"{cs:>+7.3f} {o['AB_ratio']:>8.4f} {o['MW_p_two']:>10.4g}")
        else:
            print(f"{r['model_short']:<14} {e5['AB_ratio']:>10.4f} {e5['MW_p_two']:>10.4g} "
                  f"{ep['AB_ratio']:>8.4f} {ep['MW_p_two']:>10.4g}      n/a (degenerate)")


def write_readme(rows, summary):
    md = ["# Phase-α — Llama paired-corruption null diagnostics\n"]
    md.append(f"spec_version: {summary['spec_version']}")
    md.append(f"seed: {summary['seed']}  n_bootstrap: {summary['n_bootstrap']}\n")
    md.append("## Experiments\n")
    for k, v in summary["experiments"].items():
        md.append(f"- **{k}**: {v}")
    md.append("\n## Combined results\n")
    md.append("| model | L_evi | L_act | E5 evi AB | E5 evi p₂ | PopQA act AB | PopQA p₂ | cos(PopQA, own) | own AB | own CI95 | own p₂ |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        e5 = r["E5_evidence_direction"]; ep = r["echo_PopQA_action_direction"]
        own = r["E1E2_own_step1_direction"]
        if own["feasible"]:
            o = own["ab_stats"]
            cs = f"{own['cos_with_PopQA_action_dir']:+.3f}"
            md.append(f"| {r['model_short']} | L{r['peak_evi_layer']} | L{r['peak_act_layer']} | "
                      f"{e5['AB_ratio']:.4f} | {e5['MW_p_two']:.4g} | "
                      f"{ep['AB_ratio']:.4f} | {ep['MW_p_two']:.4g} | "
                      f"{cs} | {o['AB_ratio']:.4f} | "
                      f"[{o['CI95'][0]:.4f}, {o['CI95'][1]:.4f}] | {o['MW_p_two']:.4g} |")
        else:
            md.append(f"| {r['model_short']} | L{r['peak_evi_layer']} | L{r['peak_act_layer']} | "
                      f"{e5['AB_ratio']:.4f} | {e5['MW_p_two']:.4g} | "
                      f"{ep['AB_ratio']:.4f} | {ep['MW_p_two']:.4g} | "
                      f"n/a | n/a | n/a | n/a |")
    md.append("")
    md.append("R1 own-direction excluded: sign(step1_margin) is degenerate (486/486 positive); see results/r1_own_direction_ab/ Finding 1.")
    md.append("")
    with open(os.path.join(OUT_DIR, "README.md"), "w") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    main()
