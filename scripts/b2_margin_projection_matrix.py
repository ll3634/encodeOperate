#!/usr/bin/env python3
"""B.2 — Margin-projection A/B matrix across 5 model families.

For each model, compute:
  Δm = |margin_clean - margin_corrupted|   per pair
  AB_ratio_margin = geom_median(Δm[A]) / geom_median(Δm[B])
  95% CI via log-normal bootstrap (10k resamples, seed 20260503)
  MW two-sided + one-sided p-values

The activation-projection AB_ratio_action and MW p (already computed by
cross_model_full.py) are echoed alongside for the side-by-side row.

Outputs (results/margin_projection_ab/):
  summary.json
  README.md
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
from scipy.stats import mannwhitneyu

SEED = 20260503
N_BOOT = 10000
SPEC_VERSION = "B.2-v1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = [
    ("qwen25_7b",   "cross_model_qwen25_v2",   "Qwen/Qwen2.5-7B-Instruct"),
    ("mistral_7b",  "cross_model_mistral_v2",  "mistralai/Mistral-7B-Instruct-v0.3"),
    ("llama31_8b",  "cross_model_llama31_v2",  "unsloth/Meta-Llama-3.1-8B-Instruct"),
    ("gemma2_9b",   "cross_model_gemma2_v2",   "unsloth/gemma-2-9b-it"),
    ("r1distill_7b","cross_model_r1distill_v2","deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
]
OUT_DIR = os.path.join(ROOT, "results", "margin_projection_ab")


def geom_median(x, n_iter=200, eps=1e-9):
    y = float(np.median(x))
    for _ in range(n_iter):
        d = np.abs(x - y)
        d = np.maximum(d, eps)
        w = 1.0 / d
        y_new = float(np.sum(w * x) / np.sum(w))
        if abs(y_new - y) < eps:
            break
        y = y_new
    return y


def lognormal_bootstrap_ratio_ci(a, b, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    na, nb = len(a), len(b)
    log_ratios = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        ai = a[rng.integers(0, na, na)]
        bi = b[rng.integers(0, nb, nb)]
        log_ratios[i] = np.log(geom_median(ai)) - np.log(geom_median(bi))
    lo = float(np.exp(np.quantile(log_ratios, alpha / 2)))
    hi = float(np.exp(np.quantile(log_ratios, 1 - alpha / 2)))
    return lo, hi


def classify(ab_ratio, ci_lo, ci_hi, mw_p_two):
    """Categorical classification for the matrix."""
    if ci_lo > 1.0 and mw_p_two < 0.05:
        return "evi-routing-detected"
    if ci_hi < 1.0 and mw_p_two < 0.05:
        return "anti-evi-routing"  # B > A, would be unusual
    if mw_p_two >= 0.05 and ci_lo <= 1.0 <= ci_hi:
        return "null-CI-brackets-1"
    return "ambiguous"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for short, subdir, hf_name in MODELS:
        npz_path = os.path.join(ROOT, "results", subdir, "per_sample.npz")
        full_json = os.path.join(ROOT, "results", subdir, "full_results.json")
        z = np.load(npz_path, allow_pickle=False)
        groups = list(z["pair_groups"])
        g_idx = {g: i for i, g in enumerate(groups)}

        m_c = z["pair_margin_clean"]      # (3, n_pairs) float32
        m_x = z["pair_margin_corrupted"]  # (3, n_pairs) float32
        d_m = np.abs(m_c - m_x)

        a_arr = d_m[g_idx["A"]]
        b_arr = d_m[g_idx["B"]]
        c_arr = d_m[g_idx["C"]]

        gm_A = geom_median(a_arr)
        gm_B = geom_median(b_arr)
        ratio = gm_A / gm_B
        ci_lo, ci_hi = lognormal_bootstrap_ratio_ci(a_arr, b_arr)
        mw_two = mannwhitneyu(a_arr, b_arr, alternative="two-sided")
        mw_one = mannwhitneyu(a_arr, b_arr, alternative="greater")

        # Echo activation-projection result from cross_model_full.json
        with open(full_json) as f:
            fr = json.load(f)
        pc = fr["paired_corruption"]
        action_AB = float(pc["AB_ratio_action"])
        action_p = float(pc["MW_action_p"])

        cls = classify(ratio, ci_lo, ci_hi, float(mw_two.pvalue))

        row = {
            "model_short": short,
            "model_hf": hf_name,
            "n_pairs_per_group": int(a_arr.shape[0]),
            "peak_evi_layer": int(z["peak_evi_layer"]),
            "peak_act_layer": int(z["peak_act_layer"]),
            "delta_margin_geom_median_A": gm_A,
            "delta_margin_geom_median_B": gm_B,
            "delta_margin_geom_median_C": geom_median(c_arr),
            "margin_AB_ratio": float(ratio),
            "margin_AB_ci95_lognormal_bootstrap": [ci_lo, ci_hi],
            "margin_MW_p_two_sided": float(mw_two.pvalue),
            "margin_MW_p_one_sided_greater": float(mw_one.pvalue),
            "margin_MW_U_two_sided": float(mw_two.statistic),
            "action_AB_ratio_from_cross_model_full": action_AB,
            "action_MW_p_from_cross_model_full": action_p,
            "classification": cls,
        }
        rows.append(row)

    # 2x2 block: classification x consistency-with-action
    block = {"action_sig_margin_sig": [], "action_sig_margin_null": [],
             "action_null_margin_sig": [], "action_null_margin_null": []}
    for r in rows:
        a_sig = r["action_MW_p_from_cross_model_full"] < 0.05
        m_sig = r["margin_MW_p_two_sided"] < 0.05
        key = ("action_" + ("sig" if a_sig else "null") + "_margin_" +
               ("sig" if m_sig else "null"))
        block[key].append(r["model_short"])

    summary = {
        "spec_version": SPEC_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "input_npz_paths": {r["model_short"]:
                            os.path.relpath(
                                os.path.join(ROOT, "results", subdir, "per_sample.npz"),
                                ROOT)
                            for (s, subdir, _), r in zip(MODELS, rows)
                            for _x in [None]},
        "input_full_results_paths": {r["model_short"]:
                                     os.path.relpath(
                                         os.path.join(ROOT, "results", subdir, "full_results.json"),
                                         ROOT)
                                     for (s, subdir, _), r in zip(MODELS, rows)
                                     for _x in [None]},
        "metric_definitions": {
            "delta_margin": "abs(margin_clean - margin_corrupted) per pair, where margin = "
                            "logit(Action_token) - logit(Final_token) at last input position",
            "margin_AB_ratio": "geom_median(Δm[A]) / geom_median(Δm[B])",
            "MW_alternative_two_sided": "scipy.stats.mannwhitneyu alternative='two-sided' "
                                         "(cross-model standard)",
            "MW_alternative_one_sided": "alternative='greater' (Qwen-standard)",
            "CI_method": "log-normal bootstrap, 10000 resamples, seed=20260503",
            "classification_rule":
                "evi-routing-detected: CI_lo>1 AND MW_p<0.05 ; "
                "null-CI-brackets-1: MW_p>=0.05 AND 1.0 in CI ; "
                "anti-evi-routing: CI_hi<1 AND MW_p<0.05 ; ambiguous: otherwise",
        },
        "rows": rows,
        "block_2x2_action_x_margin": block,
        "qwen_published_note": (
            "Published §9 Qwen action AB=1.83x p=2.16e-4 was produced by "
            "scripts/paired_corruption_analysis.py at N=50, evidence_dir from "
            "results/phase1_probe/probe_direction_l20.npz, action_dir from "
            "steering/directions/direction_search_v3_layer20.npz, both at L20. "
            "The qwen25_7b row here is a matched-protocol re-measurement at "
            "N=200, peak layers from cross_model_full.py independent layer sweep "
            "(see peak_evi_layer/peak_act_layer); not numerically identical."
        ),
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    write_readme(summary)
    # Brief stdout
    print(f"{'model':<14} {'mAB':>8} {'CI95':>22} {'mP2':>10} {'aAB':>8} {'aP':>10}  class")
    for r in rows:
        ci = r["margin_AB_ci95_lognormal_bootstrap"]
        print(f"{r['model_short']:<14} {r['margin_AB_ratio']:>8.4f} "
              f"[{ci[0]:>6.3f},{ci[1]:>6.3f}]   "
              f"{r['margin_MW_p_two_sided']:>10.4g} "
              f"{r['action_AB_ratio_from_cross_model_full']:>8.4f} "
              f"{r['action_MW_p_from_cross_model_full']:>10.4g}  {r['classification']}")
    print()
    print("2x2 block:", json.dumps(block))


def write_readme(s):
    md = [f"# B.2 — Margin-projection A/B matrix\n"]
    md.append(f"spec_version: {s['spec_version']}")
    md.append(f"seed: {s['seed']}")
    md.append(f"n_bootstrap: {s['n_bootstrap']}\n")
    md.append("## Inputs\n")
    for k, p in s["input_npz_paths"].items():
        md.append(f"- {k}: {p}")
    md.append("")
    md.append("## Margin-projection table (geom-median A/B, log-normal bootstrap CI)\n")
    md.append("| model | n_pairs | L_evi | L_act | mΔ_A | mΔ_B | mΔ_C | margin_AB | 95% CI | mMW_p_two | mMW_p_one | actAB | act_p | class |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in s["rows"]:
        ci = r["margin_AB_ci95_lognormal_bootstrap"]
        md.append(
            f"| {r['model_short']} | {r['n_pairs_per_group']} | "
            f"L{r['peak_evi_layer']} | L{r['peak_act_layer']} | "
            f"{r['delta_margin_geom_median_A']:.4f} | "
            f"{r['delta_margin_geom_median_B']:.4f} | "
            f"{r['delta_margin_geom_median_C']:.4f} | "
            f"{r['margin_AB_ratio']:.4f} | "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] | "
            f"{r['margin_MW_p_two_sided']:.4g} | "
            f"{r['margin_MW_p_one_sided_greater']:.4g} | "
            f"{r['action_AB_ratio_from_cross_model_full']:.4f} | "
            f"{r['action_MW_p_from_cross_model_full']:.4g} | "
            f"{r['classification']} |"
        )
    md.append("")
    md.append("## 2x2 block (action significance x margin significance)\n")
    block = s["block_2x2_action_x_margin"]
    md.append(f"- action_sig_margin_sig: {block['action_sig_margin_sig']}")
    md.append(f"- action_sig_margin_null: {block['action_sig_margin_null']}")
    md.append(f"- action_null_margin_sig: {block['action_null_margin_sig']}")
    md.append(f"- action_null_margin_null: {block['action_null_margin_null']}")
    md.append("")
    md.append(f"Qwen note: {s['qwen_published_note']}")
    md.append("")
    with open(os.path.join(OUT_DIR, "README.md"), "w") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    main()
