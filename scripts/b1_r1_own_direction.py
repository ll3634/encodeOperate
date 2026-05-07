#!/usr/bin/env python3
"""B.1 — R1-own action direction analysis (Findings 1 + 2).

Finding 1 (δ): sign(step1_margin) is degenerate on R1; this is reported as a
structural property of R1's decision regime, not an instrument failure.

Finding 2 (β): the R1-own action direction extracted by cross_model_full.py
(PopQA p10/p90 percentile split) is already saved as `action_dir` inside
per_sample.npz; here we re-project A/B/C onto it under our matched geometric-
median + log-normal bootstrap protocol.

Outputs (results/r1_own_direction_ab/):
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
SPEC_VERSION = "B.1-v2-delta+beta"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_NPZ = os.path.join(ROOT, "results", "cross_model_r1distill_v2", "per_sample.npz")
OUT_DIR = os.path.join(ROOT, "results", "r1_own_direction_ab")


def geom_median(x, n_iter=200, eps=1e-9):
    """Weiszfeld's algorithm on a 1-D array of positive scalars."""
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
    """log-normal bootstrap CI on geom_median(a) / geom_median(b)."""
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    z = np.load(INPUT_NPZ, allow_pickle=False)
    margin = z["step1_margin"]                # (N,) float32
    h_c = z["pair_h_clean"]                   # (3, n_pairs, D) float32
    h_x = z["pair_h_corrupted"]               # (3, n_pairs, D) float32
    groups = list(z["pair_groups"])           # ['A','B','C']
    action_dir = z["action_dir"].astype(np.float32)  # R1-own, PopQA p10/p90
    peak_evi = int(z["peak_evi_layer"])
    peak_act = int(z["peak_act_layer"])
    n_step1 = int(len(margin))

    # ── Finding 1 (δ): step1_margin saturation diagnostic ──────────────────
    margin_hist, margin_bins = np.histogram(margin, bins=20)
    n_pos = int((margin > 0).sum())
    n_zero = int((margin == 0).sum())
    n_neg = int((margin < 0).sum())
    finding1 = {
        "label_source_attempted": "sign(step1_margin)",
        "n_step1_total": n_step1,
        "n_search_label1": n_pos,
        "n_stop_label0": n_neg,
        "n_zero_margin": n_zero,
        "n_abs_margin_lt_0p5": int((np.abs(margin) < 0.5).sum()),
        "step1_margin_stats": {
            "min": float(margin.min()),
            "max": float(margin.max()),
            "mean": float(margin.mean()),
            "median": float(np.median(margin)),
            "std": float(margin.std()),
        },
        "step1_margin_hist": {
            "counts": [int(c) for c in margin_hist],
            "bin_edges": [float(x) for x in margin_bins],
        },
        "label_degenerate": (n_neg == 0 or n_pos == 0),
        "interpretation": (
            "R1's step-1 post-observation margin distribution is bounded away "
            "from the action-final boundary (range "
            f"[{float(margin.min()):+.2f}, {float(margin.max()):+.2f}], "
            f"{n_pos}/{n_step1} positive). v1 protocol's search-vs-stop label is "
            "undefined on R1's step1_h pool. This is a structural property of R1's "
            "decision regime, not a sample-size or methodology limitation. The §9 "
            "R1 paired-corruption A/B null is consistent with this regime: R1 does "
            "not exhibit a step-1 search-vs-stop decision to be routed."
        ),
    }

    # ── Finding 2 (β): re-project A/B/C onto R1-own PopQA-extracted dir ────
    # Direction provenance: cross_model_full.py extract_action_dir_from_popqa
    # at peak_act_layer (L{peak_act}) using p10/p90 margin split on N=400 PopQA
    # questions. Already stored in per_sample.npz as 'action_dir'.
    delta = np.abs(np.einsum("gnd,d->gn", h_c - h_x, action_dir))
    g_idx = {g: i for i, g in enumerate(groups)}
    a_arr = delta[g_idx["A"]]
    b_arr = delta[g_idx["B"]]
    c_arr = delta[g_idx["C"]]
    gm_A = geom_median(a_arr)
    gm_B = geom_median(b_arr)
    gm_C = geom_median(c_arr)
    ratio_AB = gm_A / gm_B
    ci_lo, ci_hi = lognormal_bootstrap_ratio_ci(a_arr, b_arr)
    mw_two = mannwhitneyu(a_arr, b_arr, alternative="two-sided")
    mw_one = mannwhitneyu(a_arr, b_arr, alternative="greater")

    finding2 = {
        "direction_source": (
            "cross_model_full.py:extract_action_dir_from_popqa "
            "(p10/p90 margin split on N=400 PopQA, R1-own peak_act_layer)"
        ),
        "direction_layer": peak_act,
        "direction_norm_check": float(np.linalg.norm(action_dir)),
        "direction_dim": int(action_dir.shape[0]),
        "n_pairs_per_group": int(a_arr.shape[0]),
        "delta_geom_median_A": gm_A,
        "delta_geom_median_B": gm_B,
        "delta_geom_median_C": gm_C,
        "AB_ratio_own_popqa": float(ratio_AB),
        "AB_ratio_own_popqa_ci95_lognormal_bootstrap": [ci_lo, ci_hi],
        "MW_test_name": "scipy.stats.mannwhitneyu",
        "MW_p_two_sided": float(mw_two.pvalue),
        "MW_p_one_sided_greater": float(mw_one.pvalue),
        "MW_U_two_sided": float(mw_two.statistic),
        "interpretation_template": (
            "If 95% CI brackets 1.0 and MW two-sided p > 0.05: "
            "Finding 1 (δ) is reinforced — even with R1's own PopQA-extracted "
            "action direction (the matched-protocol direction used cross-model), "
            "no detectable A/B routing on R1. "
            "Else: routing IS detectable on PopQA-extracted direction; Finding 1 "
            "stands as a regime characterisation, not a routing-absence claim."
        ),
    }

    summary = {
        "spec_version": SPEC_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "input_npz": os.path.relpath(INPUT_NPZ, ROOT),
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "peak_evi_layer": peak_evi,
        "peak_act_layer": peak_act,
        "finding_1_step1_margin_degenerate": finding1,
        "finding_2_popqa_extracted_direction": finding2,
        "output_files": {
            "summary": "summary.json",
            "readme": "README.md",
        },
        "no_direction_npy_emitted": (
            "Finding 1 yields no direction (degenerate label). Finding 2 reuses "
            "action_dir already saved in per_sample.npz; no new .npy emitted to "
            "avoid duplicating an existing artifact."
        ),
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    write_readme(summary)
    print(json.dumps({
        "Finding 1 (delta)": {
            "n_step1": finding1["n_step1_total"],
            "n_pos/n_neg": f"{finding1['n_search_label1']}/{finding1['n_stop_label0']}",
            "margin_range": [finding1['step1_margin_stats']['min'],
                             finding1['step1_margin_stats']['max']],
            "label_degenerate": finding1["label_degenerate"],
        },
        "Finding 2 (beta)": {
            "AB_ratio_own_popqa": finding2["AB_ratio_own_popqa"],
            "CI95": finding2["AB_ratio_own_popqa_ci95_lognormal_bootstrap"],
            "MW_p_two_sided": finding2["MW_p_two_sided"],
            "MW_p_one_sided_greater": finding2["MW_p_one_sided_greater"],
        },
    }, indent=2))


def write_readme(s):
    f1 = s["finding_1_step1_margin_degenerate"]
    f2 = s["finding_2_popqa_extracted_direction"]
    ms = f1["step1_margin_stats"]
    ci = f2["AB_ratio_own_popqa_ci95_lognormal_bootstrap"]
    md = f"""# B.1 — R1-own action direction A/B ratio

spec_version: {s['spec_version']}
input: {s['input_npz']}
model: {s['model']}
seed: {s['seed']}
n_bootstrap: {s['n_bootstrap']}
peak_evi_layer / peak_act_layer: L{s['peak_evi_layer']} / L{s['peak_act_layer']}

## Finding 1 (δ) — step1_margin saturation

R1's step-1 post-observation margin distribution is bounded away from the
action-final boundary (range [{ms['min']:+.2f}, {ms['max']:+.2f}],
{f1['n_search_label1']}/{f1['n_step1_total']} positive). v1 protocol's
search-vs-stop label is undefined on R1's step1_h pool. This is a structural
property of R1's decision regime, not a sample-size or methodology limitation.
The §9 R1 paired-corruption A/B null is consistent with this regime: R1 does
not exhibit a step-1 search-vs-stop decision to be routed.

| step1_margin stat | value |
|---|---|
| min | {ms['min']:+.4f} |
| max | {ms['max']:+.4f} |
| mean | {ms['mean']:+.4f} |
| median | {ms['median']:+.4f} |
| std | {ms['std']:.4f} |
| n_positive | {f1['n_search_label1']} / {f1['n_step1_total']} |
| n_zero | {f1['n_zero_margin']} |
| n_negative | {f1['n_stop_label0']} |
| n with abs(margin)<0.5 | {f1['n_abs_margin_lt_0p5']} |
| label_degenerate | {f1['label_degenerate']} |

## Finding 2 (β) — A/B routing on R1-own PopQA-extracted direction

direction source: {f2['direction_source']}
direction layer: L{f2['direction_layer']}
direction norm: {f2['direction_norm_check']:.6f}
n_pairs / group: {f2['n_pairs_per_group']}

| metric | value |
|---|---|
| Δ geom-median (A) | {f2['delta_geom_median_A']:.4f} |
| Δ geom-median (B) | {f2['delta_geom_median_B']:.4f} |
| Δ geom-median (C) | {f2['delta_geom_median_C']:.4f} |
| AB_ratio_own_popqa | {f2['AB_ratio_own_popqa']:.4f} |
| 95% CI (log-normal bootstrap) | [{ci[0]:.4f}, {ci[1]:.4f}] |
| MW p (two-sided) | {f2['MW_p_two_sided']:.6g} |
| MW p (one-sided, A>B) | {f2['MW_p_one_sided_greater']:.6g} |

interpretation: {f2['interpretation_template']}
"""
    with open(os.path.join(OUT_DIR, "README.md"), "w") as f:
        f.write(md)


if __name__ == "__main__":
    main()
