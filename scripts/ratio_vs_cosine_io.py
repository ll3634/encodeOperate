"""Plot + report writer for ratio_vs_cosine_analysis.py."""
from pathlib import Path

import numpy as np

OUT = Path("results/ocft/ratio_vs_cosine")

PRE_REG = """\
## Pre-registration (declared BEFORE inspecting results)

**Inputs (cached, no new GPU runs)**
- K=200 random unit-RMS direction vectors regenerated from
  `decomposition_ci_null.SEED = 20260429`, `np.random.RandomState(SEED)`,
  sequential `randn(3584)` → RMS-normalised. Per-direction shifts loaded
  from `results/decomposition_ci_null/per_example_shifts.npz['random']`
  (shape K=200 × N=100; cached at L20, ρ=−0.20, hidden_rms=0.65).
- A_L20 = `direction_decomp_full_layer20.npz['decision_direction']`.
- E = `results/phase1_probe/probe_direction_l20.npz` (the canonical §3
  evidence direction; cos(A, E) = -0.0135 matches §3/CLAUDE.md exactly,
  and `direction_decomp_parallel_layer20.npz` is the projection of A onto
  this E with cos = -1.0). D1–D4 = `per_candidate/<DK>/direction.npy`.
- Per-prompt parallel shifts for E and D1–D4 from cached .npz files.

**STEP 1** — `cos_k = dot(R_k, A) / (‖R_k‖ ‖A‖)` for each random; same for E, D1–D4.

**STEP 2** — per-direction aggregate ratio:
`ratio_k = |mean_i Δm_Rk_i| / |mean_i Δm_full_i|`. Signed version retained.

**STEP 3** — evidence cos-bin = [0.005, 0.025]. If `n<10`, widen to [0.001, 0.040]
and report widening. Compute pool stats, evidence percentile, one-sided
permutation p-value (10,000 draws from pool ∪ {value}).

**STEP 4** — D3 cos-bin = [0.025, 0.055]. Same metrics.

**STEP 5** — Spearman ρ(|cos|, |ratio|) on K=200 randoms; if p<0.05 fit
linear trend and locate E and D3 on the regression line.

**Pre-registered Outcomes**
- **α (E)** — `pct < 25` AND `p_below_median < 0.05`:
  evidence is MORE inert than geometric default; mechanistic suppression.
- **β (E)** — `25 ≤ pct ≤ 75` OR `p_below_median ≥ 0.05`:
  evidence inertness is the geometric default at its cosine.
- **γ (E)** — `pct > 75`: contradicts §3 inertness. (Very unlikely.)
- **δ (D3)** — `pct > 95` in D3 bin: D3 is mechanistically operative
  above geometric baseline.
"""


def _outcome_E(stats):
    pct = stats["percentile_of_value"]
    p = stats["p_value_below_median"]
    if pct is None: return "N/A", "evidence bin empty"
    if pct < 25 and p < 0.05:
        return "α", "Evidence is MORE inert than random at the same cosine — mechanistic suppression."
    if pct > 75:
        return "γ", "Evidence is MORE operative than random at the same cosine — contradicts §3 inertness."
    return "β", "Evidence inertness matches the geometric default at its cosine."


def _outcome_D3(stats):
    pct = stats["percentile_of_value"]
    if pct is None: return False, "D3 bin empty"
    return (pct > 95), f"D3 percentile in its cos-bin = {pct:.1f}"


def make_plot(abs_cos_R, rand_abs_ratio, named, lin, e_bin, d3_bin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.scatter(abs_cos_R, rand_abs_ratio, s=14, alpha=0.45, c="#999999",
               edgecolors="none", label=f"random (K={len(abs_cos_R)})")

    colors = {"E": "#2065c8", "D1": "#888800", "D2": "#cc6600",
              "D3": "#cc0000", "D4": "#5e9e58"}
    for n, d in named.items():
        ax.scatter([d["abs_cos"]], [d["abs_ratio"]], s=120, marker="*",
                   c=colors[n], edgecolors="black", linewidths=0.7, label=n, zorder=5)

    # Bin shading
    ax.axvspan(*e_bin, color="#2065c8", alpha=0.07, lw=0)
    ax.axvspan(*d3_bin, color="#cc0000", alpha=0.07, lw=0)

    # Trend line
    xs = np.linspace(0, max(abs_cos_R.max(), max(d["abs_cos"] for d in named.values())) * 1.05, 50)
    ys = lin.intercept + lin.slope * xs
    ax.plot(xs, ys, "-", color="#444444", lw=1, alpha=0.7,
            label=f"OLS slope={lin.slope:+.2f} R²={lin.rvalue**2:.3f}")

    ax.set_xlabel("|cos(D, A_L20)|")
    ax.set_ylabel("|ratio| = |mean Δm_par_D| / |mean Δm_full|")
    ax.set_title("Ratio vs |cos| — K=200 random + 6 named L20 directions")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT / "scatter_plot.png", dpi=160)
    plt.close(fig)
    print(f"[plot] {OUT/'scatter_plot.png'}")


def write_report(named, e_stats, d3_stats, rho, pval, lin):
    out_E, interp_E = _outcome_E(e_stats)
    out_D3, interp_D3 = _outcome_D3(d3_stats)
    lines = ["# Ratio-vs-Cosine — K=200 random + 6 named L20 directions", ""]
    lines.append(PRE_REG)
    lines.append("")
    lines.append("## Named directions (cos and aggregate ratio)")
    lines.append("")
    lines.append("| dir | cos(D, A) | |cos| | mean Δm_par | |ratio| |")
    lines.append("|-----|----------:|------:|------------:|-------:|")
    for n, d in named.items():
        lines.append(f"| {n} | {d['cos_to_A']:+.4f} | {d['abs_cos']:.4f} | "
                     f"{d['mean_par_shift']:+.4f} | {d['abs_ratio']:.4f} |")
    lines.append("")
    lines.append("## STEP 5 — Spearman & OLS trend (K=200 randoms)")
    lines.append("")
    lines.append(f"- Spearman ρ(|cos|, |ratio|) = **{rho:+.3f}**, p = {pval:.3g}")
    lines.append(f"- OLS slope = {lin.slope:+.3f}, intercept = {lin.intercept:+.3f}, "
                 f"R² = {lin.rvalue**2:.3f}, p = {lin.pvalue:.3g}")
    sig = "**significant**" if pval < 0.05 else "not significant"
    lines.append(f"- Geometric prediction: {sig}")
    lines.append("")
    lines.append("## STEP 3 — Evidence cos-bin")
    lines.append("")
    lines.append(f"Bin = [{e_stats['bin_lo']:.3f}, {e_stats['bin_hi']:.3f}]"
                 f"{' (WIDENED)' if e_stats.get('bin_widened') else ''}, "
                 f"n_in_bin = **{e_stats['n_in_bin']}**")
    if e_stats["n_in_bin"] > 0:
        lines.append("")
        lines.append("| stat | value |")
        lines.append("|------|------:|")
        for k in ("bin_min", "bin_p25", "bin_median", "bin_mean", "bin_p75", "bin_max"):
            lines.append(f"| {k} | {e_stats[k]:.4f} |")
        lines.append(f"| **E |ratio|** | **{e_stats['value']:.4f}** |")
        lines.append(f"| E percentile in bin | {e_stats['percentile_of_value']:.1f} |")
        lines.append(f"| p (E ≤ bin via permutation) | {e_stats['p_value_below_median']:.4f} |")
    lines.append("")
    lines.append("## STEP 4 — D3 cos-bin")
    lines.append("")
    lines.append(f"Bin = [{d3_stats['bin_lo']:.3f}, {d3_stats['bin_hi']:.3f}], "
                 f"n_in_bin = **{d3_stats['n_in_bin']}**")
    if d3_stats["n_in_bin"] > 0:
        lines.append("")
        lines.append("| stat | value |")
        lines.append("|------|------:|")
        for k in ("bin_min", "bin_p25", "bin_median", "bin_mean", "bin_p75", "bin_max"):
            lines.append(f"| {k} | {d3_stats[k]:.4f} |")
        lines.append(f"| **D3 |ratio|** | **{d3_stats['value']:.4f}** |")
        lines.append(f"| D3 percentile in bin | {d3_stats['percentile_of_value']:.1f} |")
        if "p_value_above_median" in d3_stats:
            lines.append(f"| p (D3 ≥ bin via permutation) | {d3_stats['p_value_above_median']:.4f} |")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- **Evidence (Outcome {out_E})** — {interp_E}")
    lines.append(f"- **D3 (Outcome δ {'YES' if out_D3 else 'NO'})** — {interp_D3}")
    lines.append("")
    lines.append("![scatter](scatter_plot.png)")
    (OUT / "report.md").write_text("\n".join(lines))
    print(f"\n[done] wrote {OUT}/report.md")
    print(f"[verdict] Evidence: Outcome {out_E}    |    D3: δ {'YES' if out_D3 else 'NO'}")
