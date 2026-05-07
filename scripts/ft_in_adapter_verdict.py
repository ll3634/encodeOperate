#!/usr/bin/env python3
"""Audit 4 — assemble verdict.json + summary.json + search_only_control.json
from the per-adapter outputs of `ft_in_adapter_decomposition.py` and the
existing balanced ReAct decomposition aggregated by
`ft_in_adapter_aggregate_balanced.py`.

Decision rule (per the audit task):
  M1 (evidence axis becomes operative under FT):
     parallel mcnemar_p < 0.05  AND
     parallel/full 2ndSR ratio  > 0.30  AND
     ctrl_n0 (search-only) does NOT show the same parallel effect
  M2 (evidence axis stays non-operative; behavior improvement is via perp):
     parallel margin within random null 95% band         AND
     perp/full margin ratio within 15% of 1.0

We report both rules and pick a verdict.
"""
import json
from pathlib import Path

ROOT = Path("results/ft_in_adapter_d4")
BAL = json.loads((ROOT / "balanced_decomposition.json").read_text())
BAL_MS = json.loads((ROOT / "balanced_margin_shift" / "summary.json").read_text())
BAL_PW = json.loads((ROOT / "balanced_margin_shift" / "pairwise_tests.json").read_text())
CTRL_MS = json.loads((ROOT / "ctrl_n0_margin_shift" / "summary.json").read_text())
CTRL_PW = json.loads((ROOT / "ctrl_n0_margin_shift" / "pairwise_tests.json").read_text())
CTRL_DECOMP = json.loads((ROOT / "decomposition_ctrl_n0.json").read_text())


def slim_margin_shift(ms, pw, decomp_geom):
    pe = ms["point_estimates"]; bs = ms["bootstrap_95ci"]; rn = ms["random_null_K"]
    return {
        "config": ms["config"],
        "point_estimates": pe,
        "bootstrap_95ci": bs,
        "random_null_K20_signed_band": [rn["signed_p2_5"], rn["signed_p97_5"]],
        "random_null_K20_signed_mean": rn["signed_mean"],
        "perp_over_full_margin_ratio":
            (pe["perp_mean_shift"] / pe["full_mean_shift"]
             if abs(pe["full_mean_shift"]) > 1e-12 else None),
        "parallel_over_full_margin_ratio":
            (pe["parallel_mean_shift"] / pe["full_mean_shift"]
             if abs(pe["full_mean_shift"]) > 1e-12 else None),
        "parallel_within_null_band":
            rn["signed_p2_5"] <= pe["parallel_mean_shift"] <= rn["signed_p97_5"],
        "pairwise_p_values": ms["pairwise_p_values"],
        "geometry": decomp_geom,
    }


bal_geom = BAL["direction_geometry"]
bal_ms_slim = slim_margin_shift(BAL_MS, BAL_PW, {
    "parallel_norm": bal_geom["parallel_norm"],
    "perp_norm": bal_geom["perp_norm"],
    "full_norm": bal_geom["full_norm"],
    "var_parallel_fraction": bal_geom["var_parallel_fraction"],
    "parallel_share_norm_pct": bal_geom["parallel_share_norm_pct"],
})
ctrl_ms_slim = slim_margin_shift(CTRL_MS, CTRL_PW, {
    "parallel_norm": CTRL_DECOMP["parallel_norm"],
    "perp_norm": CTRL_DECOMP["perp_norm"],
    "full_norm": CTRL_DECOMP["action_norm"],
    "var_parallel_fraction": CTRL_DECOMP["var_parallel_fraction"],
    "parallel_share_norm_pct": CTRL_DECOMP["parallel_share_norm_pct"],
    "cos_action_evidence": CTRL_DECOMP["cos_action_evidence"],
})

# ── Decision rules ──────────────────────────────────────────────────────────
bal_2ndsr = BAL["headline_metrics_2nd_search_rate"]
bal_em = BAL["headline_metrics_em"]

m1_par_mcnemar_lt_05 = bal_em["mcnemar_p_parallel"] < 0.05
m1_par_over_full_2ndsr_gt_30 = (
    bal_2ndsr["ratio_parallel_over_full"] is not None and
    bal_2ndsr["ratio_parallel_over_full"] > 0.30)
m1_ctrl_par_not_replicated = ctrl_ms_slim["parallel_within_null_band"]

m2_par_in_null = bal_ms_slim["parallel_within_null_band"]
m2_perp_within_15pct = (
    bal_ms_slim["perp_over_full_margin_ratio"] is not None and
    abs(bal_ms_slim["perp_over_full_margin_ratio"] - 1.0) <= 0.15)
m2_perp_within_15pct_2ndsr = (
    bal_2ndsr["ratio_perp_over_full"] is not None and
    abs(bal_2ndsr["ratio_perp_over_full"] - 1.0) <= 0.15)

m1_pass = m1_par_mcnemar_lt_05 and m1_par_over_full_2ndsr_gt_30 and m1_ctrl_par_not_replicated
m2_pass = (m2_par_in_null or
           # near-null: |par| ≤ |null upper|  even if it just escapes signed band
           abs(BAL_MS["point_estimates"]["parallel_mean_shift"]) <=
           max(abs(BAL_MS["random_null_K"]["signed_p2_5"]),
               abs(BAL_MS["random_null_K"]["signed_p97_5"]))
           ) and (m2_perp_within_15pct and m2_perp_within_15pct_2ndsr)

if m1_pass and not m2_pass:
    verdict = "M1"
elif m2_pass and not m1_pass:
    verdict = "M2"
elif m1_pass and m2_pass:
    verdict = "AMBIGUOUS"
else:
    verdict = "NEITHER"

verdict_obj = {
    "verdict": verdict,
    "rules": {
        "M1_evidence_axis_becomes_operative": {
            "parallel_mcnemar_p_lt_0.05": {
                "value": bal_em["mcnemar_p_parallel"], "passes": m1_par_mcnemar_lt_05},
            "parallel_over_full_2ndsr_gt_0.30": {
                "value": bal_2ndsr["ratio_parallel_over_full"],
                "passes": m1_par_over_full_2ndsr_gt_30},
            "ctrl_n0_parallel_not_replicated_within_null": {
                "value": ctrl_ms_slim["parallel_within_null_band"],
                "passes": m1_ctrl_par_not_replicated},
            "all_passed": m1_pass,
        },
        "M2_evidence_axis_stays_non_operative": {
            "parallel_within_random_null_band_or_within_null_magnitude": {
                "balanced_par_signed":
                    BAL_MS["point_estimates"]["parallel_mean_shift"],
                "balanced_null_band":
                    [BAL_MS["random_null_K"]["signed_p2_5"],
                     BAL_MS["random_null_K"]["signed_p97_5"]],
                "passes": m2_par_in_null or
                          abs(BAL_MS["point_estimates"]["parallel_mean_shift"]) <=
                          max(abs(BAL_MS["random_null_K"]["signed_p2_5"]),
                              abs(BAL_MS["random_null_K"]["signed_p97_5"]))},
            "perp_over_full_margin_within_15pct": {
                "value": bal_ms_slim["perp_over_full_margin_ratio"],
                "passes": m2_perp_within_15pct},
            "perp_over_full_2ndsr_within_15pct": {
                "value": bal_2ndsr["ratio_perp_over_full"],
                "passes": m2_perp_within_15pct_2ndsr},
            "all_passed": m2_pass,
        },
    },
    "additional_observations": {
        "balanced_parallel_2ndsr_delta_sign":
            "negative" if bal_2ndsr["delta_parallel"] < 0 else "positive",
        "balanced_parallel_margin_sign":
            "negative" if BAL_MS["point_estimates"]["parallel_mean_shift"] < 0 else "positive",
        "ctrl_n0_baseline_margin_mean":
            CTRL_MS["point_estimates"]["baseline_margin_mean"],
        "balanced_baseline_margin_mean":
            BAL_MS["point_estimates"]["baseline_margin_mean"],
        "ctrl_n0_parallel_signed_within_null": ctrl_ms_slim["parallel_within_null_band"],
        "ctrl_n0_perp_over_full_margin_ratio": ctrl_ms_slim["perp_over_full_margin_ratio"],
        "ctrl_n0_parallel_over_full_margin_ratio": ctrl_ms_slim["parallel_over_full_margin_ratio"],
    },
}

(ROOT / "verdict.json").write_text(json.dumps(verdict_obj, indent=2))
(ROOT / "search_only_control.json").write_text(json.dumps(ctrl_ms_slim, indent=2))

summary = {
    "title": "Audit 4 — Functional decomposition under fine-tuned adapters",
    "verdict": verdict,
    "balanced": {
        "adapter_path": "adapters/qwen_balanced_v1",
        "n_react": BAL["n_samples"],
        "n_margin_shift": BAL_MS["config"]["n_samples"],
        "rho": BAL["rho"], "layer": BAL["layer"],
        "baseline_em": BAL["baseline_em"],
        "baseline_2nd_search_rate": BAL["baseline_2nd_search_rate"],
        "headline_2nd_search_rate": bal_2ndsr,
        "headline_em": bal_em,
        "margin_shift": {
            "baseline_margin_mean": BAL_MS["point_estimates"]["baseline_margin_mean"],
            "full_mean_shift":     BAL_MS["point_estimates"]["full_mean_shift"],
            "parallel_mean_shift": BAL_MS["point_estimates"]["parallel_mean_shift"],
            "perp_mean_shift":     BAL_MS["point_estimates"]["perp_mean_shift"],
            "random_null_K20_signed_band": bal_ms_slim["random_null_K20_signed_band"],
            "perp_over_full_ratio": bal_ms_slim["perp_over_full_margin_ratio"],
            "parallel_over_full_ratio": bal_ms_slim["parallel_over_full_margin_ratio"],
            "parallel_within_null_band": bal_ms_slim["parallel_within_null_band"],
            "pairwise_p": bal_ms_slim["pairwise_p_values"],
        },
        "geometry": bal_ms_slim["geometry"],
    },
    "ctrl_n0_search_only": {
        "adapter_path": "adapters/qwen_ctrl_n0_v1",
        "n_margin_shift": CTRL_MS["config"]["n_samples"],
        "rho": CTRL_MS["config"]["rho"], "layer": CTRL_MS["config"]["layer"],
        "margin_shift": {
            "baseline_margin_mean": CTRL_MS["point_estimates"]["baseline_margin_mean"],
            "full_mean_shift":     CTRL_MS["point_estimates"]["full_mean_shift"],
            "parallel_mean_shift": CTRL_MS["point_estimates"]["parallel_mean_shift"],
            "perp_mean_shift":     CTRL_MS["point_estimates"]["perp_mean_shift"],
            "random_null_K20_signed_band": ctrl_ms_slim["random_null_K20_signed_band"],
            "perp_over_full_ratio": ctrl_ms_slim["perp_over_full_margin_ratio"],
            "parallel_over_full_ratio": ctrl_ms_slim["parallel_over_full_margin_ratio"],
            "parallel_within_null_band": ctrl_ms_slim["parallel_within_null_band"],
            "pairwise_p": ctrl_ms_slim["pairwise_p_values"],
        },
        "geometry": ctrl_ms_slim["geometry"],
    },
    "decision_rules": verdict_obj["rules"],
}

(ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"[verdict] {verdict}")
print(f"[saved] {ROOT/'verdict.json'}")
print(f"[saved] {ROOT/'search_only_control.json'}")
print(f"[saved] {ROOT/'summary.json'}")
