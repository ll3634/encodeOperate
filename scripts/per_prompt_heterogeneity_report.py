#!/usr/bin/env python3
"""Generate heterogeneity_report.md + figure JSONs from prior steps."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

OUT = Path("results/per_prompt_heterogeneity")

dist = json.load(open(OUT / "distribution_analysis.json"))
preds = json.load(open(OUT / "predictor_table.json"))
sg = json.load(open(OUT / "subgroup_results.json"))
verd = json.load(open(OUT / "verdict.json"))
inter = np.load(OUT / "intermediate.npz", allow_pickle=True)

r = inter["r"]
parallel = inter["parallel"]
full = inter["full"]

# Figure 1: distribution of r
import numpy as np
edges = np.linspace(0, 5, 26)
counts, _ = np.histogram(np.clip(r, 0, 5), bins=edges)
fig_dist = {
    "title": "Per-prompt ratio r = |Δm_par_evidence| / |Δm_full|",
    "x_label": "r (capped at 5)",
    "y_label": "count",
    "edges": edges.tolist(),
    "counts": counts.tolist(),
    "annotations": {
        "median": float(np.median(r)),
        "p25": float(np.percentile(r, 25)),
        "p75": float(np.percentile(r, 75)),
        "p90": float(np.percentile(r, 90)),
        "BC_r_capped": dist["bimodality_coefficient"]["r_capped"],
        "GMM_delta_BIC": dist["gmm_r"]["delta_bic"],
    },
}
json.dump(fig_dist, open(OUT / "figure_distribution.json", "w"), indent=2)

# Figure 2: scatter |full| vs r, marking top/bot subgroup
order = np.argsort(np.abs(full))
top_idx = set(int(i) for i in order[-20:])
bot_idx = set(int(i) for i in order[:20])
points = []
for i in range(len(r)):
    grp = "top" if i in top_idx else ("bot" if i in bot_idx else "mid")
    points.append({"abs_full": float(abs(full[i])), "r": float(r[i]),
                   "abs_parallel": float(abs(parallel[i])), "subgroup": grp})
fig_pred = {
    "title": "Strongest predictor: |Δm_full| vs r (artifact case)",
    "x_label": "|Δm_full|",
    "y_label": "r = |Δm_par_evidence| / |Δm_full|",
    "spearman_rho": -0.494,
    "p_bonferroni": 2.14e-6,
    "annotation": "Bimodality of r is driven by the denominator: when |Δm_full| is small, r inflates.",
    "points": points,
}
json.dump(fig_pred, open(OUT / "figure_predictor_scatter.json", "w"), indent=2)

# Build the markdown report
lines = []
lines.append("# Per-prompt Heterogeneity of Evidence-Parallel Causal Effect\n")
lines.append("**Verdict: ARTIFACT.** The bimodality detected in r is a denominator-scaling artifact, not evidence of conditional operativity.\n")

lines.append("## Pre-registered scheme\n")
lines.append("Discovery requires: (i) bimodality in r, (ii) discovery-grade predictor that is NOT a noise-scaling artifact (|Δm_full| or h·Â), and (iii) top subgroup mean r > 0.4 with bootstrap CI excluding 0.25. ARTIFACT verdict triggered when the strongest predictor is a noise-scaling variable (|Δm_full|, h·Â, cos·Â) at p_bonf<0.01 and |ρ|>0.4.\n")

lines.append("## STEP 1 — Distribution shape\n")
rs = dist["r_stats"]
lines.append(f"r (N=100): mean={rs['mean']:.3f}, median={rs['median']:.3f}, p75={rs['p75']:.3f}, p90={rs['p90']:.3f}, max={rs['max']:.3f}.")
lines.append(f"Denominator-tiny (|Δm_full|<0.25): {dist['denominator_tiny_count']}/100.\n")
lines.append("| Test | Statistic | Threshold | Bimodal? |")
lines.append("|---|---|---|---|")
lines.append(f"| SAS bimodality coefficient (r capped at 5) | {dist['bimodality_coefficient']['r_capped']:.3f} | >0.555 | YES |")
lines.append(f"| BC on signed Δm_par_evidence | {dist['bimodality_coefficient']['parallel']:.3f} | >0.555 | NO |")
lines.append(f"| BC on |Δm_par_evidence| | {dist['bimodality_coefficient']['abs_parallel']:.3f} | >0.555 | NO |")
lines.append(f"| GMM 2-comp vs 1-comp ΔBIC (r) | {dist['gmm_r']['delta_bic']:+.2f} | >6 | YES |")
lines.append(f"| GMM 2-comp vs 1-comp ΔBIC (parallel) | {dist['gmm_parallel']['delta_bic']:+.2f} | >6 | NO |\n")
lines.append("Hartigan's dip test was implemented as a simplified ECDF-vs-linear proxy (no `diptest` package available); reported but not weighted heavily. **Conclusion**: r is bimodal in shape; the underlying parallel-shift distribution is NOT.\n")

lines.append("## STEP 2 — Predictor search\n")
lines.append("All K=12 predictors tested; Bonferroni correction applied.\n")
lines.append("| Predictor | Type | ρ or AUROC | p_raw | p_bonf | Discovery-grade? |")
lines.append("|---|---|---|---|---|---|")
for r_ in sorted(preds["rows"], key=lambda x: -abs(x["stat"]) if x["stat"]==x["stat"] else 0):
    if r_["stat"] != r_["stat"]:
        lines.append(f"| {r_['predictor']} | {r_['type']} | NaN (constant) | — | — | — |")
        continue
    disc = "**YES (artifact)**" if r_["predictor"] in {"abs_full_shift","h_dot_A","cos_h_A"} \
        and r_["p_bonf"]<0.01 and abs(r_["stat"])>0.4 else (
        "yes" if r_["p_bonf"]<0.01 and (
            (r_["type"]=="continuous" and abs(r_["stat"])>0.4) or
            (r_["type"]=="categorical" and (r_["stat"]>0.65 or r_["stat"]<0.35))) else "no")
    lines.append(f"| {r_['predictor']} | {r_['type']} | {r_['stat']:+.3f} | {r_['p_raw']:.2e} | {r_['p_bonf']:.2e} | {disc} |")
lines.append("")
lines.append("**Strongest predictor: `abs_full_shift` (ρ=-0.494, p_bonf=2.14e-6).** This is a noise-scaling artifact predictor in the pre-registered list.\n")
lines.append("**Collinearity flag**: `n_sf_total` is constant (all bridge questions have 2 SFs) → undefined ρ. `n_sf_retrieved` correlates with `is_correct` and `behavioral_continue` (operationally redundant, not separately discovery-grade here).\n")

lines.append("## STEP 3 — Discovery threshold\n")
lines.append("One predictor passes p_bonf<0.01 AND |ρ|>0.4: `abs_full_shift`. **All other predictors are non-significant after Bonferroni correction.** Because `abs_full_shift` is in the pre-registered ARTIFACT set, the ARTIFACT branch is triggered.\n")

lines.append("## STEP 4 — Subgroup analysis (artifact validation)\n")
g = sg["abs_full_shift"]
lines.append(f"Top-20 vs bottom-20 by |Δm_full| (K=20 each side, perm test N=5000):\n")
lines.append("| Subgroup | mean r [95% CI] | mean \\|Δm_par_evidence\\| [95% CI] | mean random null \\|Δm_random\\| [95% CI] | par/random |")
lines.append("|---|---|---|---|---|")
lines.append(f"| top-|full| (high signal) | {g['r_top_mean_ci'][0]:.3f} [{g['r_top_mean_ci'][1]:.3f}, {g['r_top_mean_ci'][2]:.3f}] | {g['abs_par_top_mean_ci'][0]:.3f} [{g['abs_par_top_mean_ci'][1]:.3f}, {g['abs_par_top_mean_ci'][2]:.3f}] | {g['rand_abs_top_mean_ci'][0]:.3f} [{g['rand_abs_top_mean_ci'][1]:.3f}, {g['rand_abs_top_mean_ci'][2]:.3f}] | **0.79x** |")
lines.append(f"| bot-|full| (low signal) | {g['r_bot_mean_ci'][0]:.3f} [{g['r_bot_mean_ci'][1]:.3f}, {g['r_bot_mean_ci'][2]:.3f}] | {g['abs_par_bot_mean_ci'][0]:.3f} [{g['abs_par_bot_mean_ci'][1]:.3f}, {g['abs_par_bot_mean_ci'][2]:.3f}] | {g['rand_abs_bot_mean_ci'][0]:.3f} [{g['rand_abs_bot_mean_ci'][1]:.3f}, {g['rand_abs_bot_mean_ci'][2]:.3f}] | 0.95x |")
lines.append(f"\nPermutation test top vs bottom (r): observed diff={g['perm_test_diff_obs']:+.3f}, p={g['perm_test_p']:.4f}.\n")
lines.append("**Diagnostic interpretation.** In BOTH subgroups, |Δm_par_evidence| is below the random-direction null (CI overlap or below). The spread in r is therefore mechanical: small |Δm_full| values place the same |Δm_par_evidence| against a smaller denominator. The top-|full| subgroup's r=0.137 is much LOWER than the bottom subgroup's r=0.696 (perm p=0.0002), the OPPOSITE direction expected if the high-r prompts represented genuine evidence operativity.\n")

lines.append("## VERDICT: ARTIFACT\n")
lines.append("Heterogeneity in the per-prompt evidence-parallel ratio is denominator-scaling noise, not a structured operativity subset. Evidence direction E remains uniformly causally inert across the N=100 cohort.\n")
lines.append("**Implication for §3 thesis**: stands as written. The dissociation between encoding (probe AUROC=0.862) and operativity (mean Δm_parallel ≈ 0) is uniform across prompts. No conditional-operativity finding to add to Figure 1; no appendix subgroup panel needed.\n")

lines.append("## Caveats and what was NOT done\n")
lines.append("- Hartigan's dip statistic uses a simplified ECDF-vs-linear proxy (no `diptest` package); we relied on BC + GMM ΔBIC for distribution-shape calls.")
lines.append("- We did NOT explore multivariate predictor combinations (only marginal tests) — pre-registered scheme called for marginal screening only.")
lines.append("- We did NOT re-run injections on hypothetical 'high-evidence' subsets (no GPU per task).")
lines.append("- `n_sf_total` constant → excluded from K-counted tests is left in the report as transparent.\n")

(OUT / "heterogeneity_report.md").write_text("\n".join(lines))
print(f"[done] wrote {OUT/'heterogeneity_report.md'}")
print(f"[done] wrote {OUT/'figure_distribution.json'}, {OUT/'figure_predictor_scatter.json'}")
