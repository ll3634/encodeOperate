#!/usr/bin/env python3
"""Cross-family decomposition figure v4 — minimalist two-panel forest plot.

Design philosophy:  one panel = one claim.  No geometric tricks, no insets,
no double axes.  Just two publication-grade dot plots with explicit CIs.

  Panel A  —  "Orthogonal component captures the full effect"
              For each model, plot |full Δm| (square) and |⊥ component|
              (circle) at the same y-row.  The two markers VISUALLY OVERLAP,
              which is the entire claim.  Bootstrap 95% CI bars.

  Panel B  —  "Parallel component is indistinguishable from zero"
              Per-model horizontal null band (random-direction 95% envelope,
              natural-norm) drawn as background.  Marker + CI for the
              measured ∥ component.  Vertical line at 0 = perfect
              orthogonality.  Every marker falls inside its null band.

Output: results/fig_crossfamily_best/crossfamily_v4.{png,pdf}
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results/crossfamily_ci_decomposition/crossfamily_table.json"
OUT  = ROOT / "results/fig_crossfamily_best"
OUT.mkdir(parents=True, exist_ok=True)
tbl = json.load(open(DATA))

# ── rcParams ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10.0,
    "axes.linewidth":     0.9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.major.width":  0.7, "xtick.major.size": 3.5,
    "ytick.major.width":  0.7, "ytick.major.size": 0.0,
    "legend.framealpha":  0.96,
    "legend.edgecolor":   "#CCCCCC",
    "legend.fontsize":    8.6,
})

NICE   = {"qwen": "Qwen2.5-7B-Instruct", "gemma": "Gemma-2-9B-it",  "mistral": "Mistral-7B-v0.3"}
LAYER  = {"qwen": 20, "gemma": 37, "mistral": 28}
DIM    = {"qwen": 3584, "gemma": 3584, "mistral": 4096}
COLORS = {"qwen": "#1F77B4", "gemma": "#D9822B", "mistral": "#2CA02C"}
ROW_ORDER = ["mistral", "gemma", "qwen"]   # bottom-to-top

# ── Per-model derived quantities ──────────────────────────────────────────────
def get(key):
    m = tbl[key]
    full      = m["full"]["mean"];      full_lo  = m["full"]["ci_low"];      full_hi  = m["full"]["ci_high"]
    perp      = m["perp_rms"]["mean"];  perp_lo  = m["perp_rms"]["ci_low"];  perp_hi  = m["perp_rms"]["ci_high"]
    if m["par_natural"] is not None:
        par     = m["par_natural"]["mean"]
        par_lo  = m["par_natural"]["ci_low"]
        par_hi  = m["par_natural"]["ci_high"]
        null95  = m["stopping_rule_no_renorm"]["rhs"]
        in_null = m["stopping_rule_no_renorm"]["in_null_band"]
        ratio   = m["ratio_par_natural_over_full"] * 100
        has_par = True
    else:
        par = par_lo = par_hi = None
        null95 = None
        in_null = None
        ratio = None
        has_par = False
    return dict(
        full=full, full_lo=full_lo, full_hi=full_hi,
        perp=perp, perp_lo=perp_lo, perp_hi=perp_hi,
        par=par, par_lo=par_lo, par_hi=par_hi,
        null95=null95, in_null=in_null,
        ratio_perp=m["ratio_perp_over_full"]*100,
        ratio_par=ratio,
        cos_ae=m["cos_action_evidence"],
        n=m["n_samples"],
        layer=LAYER[key],
        d=DIM[key],
        has_par=has_par,
    )

DAT = {k: get(k) for k in ROW_ORDER}

# ── Figure skeleton ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11.0, 5.6))
gs = fig.add_gridspec(
    1, 2,
    width_ratios=[1.0, 1.0],
    left=0.075, right=0.985,
    top=0.84, bottom=0.26,
    wspace=0.22,
)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])

xpos = np.arange(len(ROW_ORDER))
xlabels = []
for k in ROW_ORDER:
    g = DAT[k]
    xlabels.append(f"{NICE[k]}\nL{g['layer']}  d={g['d']}  N={g['n']}")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — vertical bars: full Δm vs ⊥ component
# ══════════════════════════════════════════════════════════════════════════════
BAR_W = 0.36

for i, k in enumerate(ROW_ORDER):
    g = DAT[k]
    color = COLORS[k]
    # full Δm — open bar, left of group center
    axA.bar(
        i - BAR_W/2, g["full"], width=BAR_W,
        facecolor="white", edgecolor=color, linewidth=1.8,
        yerr=[[g["full"]-g["full_lo"]], [g["full_hi"]-g["full"]]],
        ecolor=color, capsize=3.5, error_kw=dict(elinewidth=1.3, capthick=1.0),
        zorder=3,
    )
    # ⊥ component — filled bar, right of group center
    axA.bar(
        i + BAR_W/2, g["perp"], width=BAR_W,
        facecolor=color, edgecolor=color, linewidth=0,
        yerr=[[g["perp"]-g["perp_lo"]], [g["perp_hi"]-g["perp"]]],
        ecolor=color, capsize=3.5, error_kw=dict(elinewidth=1.3, capthick=1.0),
        zorder=3, alpha=0.92,
    )
    # Ratio annotation above the group
    top = max(g["full_hi"], g["perp_hi"])
    axA.text(
        i, top * 1.045,
        f"{g['ratio_perp']:.2f}%",
        ha="center", va="bottom", fontsize=10.0,
        color=color, fontweight="bold",
    )

axA.set_xticks(xpos)
axA.set_xticklabels(xlabels, fontsize=9.2)
axA.set_xlim(-0.6, len(ROW_ORDER) - 0.4)
axA.set_ylabel(r"Projection magnitude on $\hat{\mathbf{a}}$" + "\n(rms-normalized)",
               fontsize=10.2)
axA.set_ylim(0, max(DAT[k]["full"] for k in ROW_ORDER) * 1.18)
axA.grid(axis="y", color="#EEEEEE", linewidth=0.6, zorder=0)

axA.set_title(
    r"$\bf{A}$    $\perp$-component captures the full effect",
    loc="left", fontsize=11.6, pad=10, color="#111",
)
# subtitle annotation
axA.text(0.5, 0.96, r"ratio  $\perp / \mathrm{full}$",
         transform=axA.transAxes, ha="center", va="top",
         fontsize=8.6, color="#555", style="italic")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — vertical bars: ∥ component vs random null band
# ══════════════════════════════════════════════════════════════════════════════
COL_HALF = 0.36

for i, k in enumerate(ROW_ORDER):
    g = DAT[k]
    color = COLORS[k]
    if g["has_par"]:
        # null-band background column (per-model width)
        axB.add_patch(plt.Rectangle(
            (i - COL_HALF, -g["null95"]),
            2 * COL_HALF, 2 * g["null95"],
            facecolor="#DDDDDD", alpha=0.55, edgecolor="none", zorder=0,
        ))
        # null-band edges
        for sign in (-1, +1):
            axB.plot([i - COL_HALF, i + COL_HALF],
                     [sign * g["null95"]] * 2,
                     color="#888", lw=0.7, zorder=0.5)
        # measured ∥ as a bar from 0 with CI
        axB.bar(
            i, g["par"], width=BAR_W,
            facecolor=color, edgecolor=color, linewidth=0,
            yerr=[[g["par"]-g["par_lo"]], [g["par_hi"]-g["par"]]],
            ecolor=color, capsize=3.5,
            error_kw=dict(elinewidth=1.3, capthick=1.0),
            zorder=3, alpha=0.92,
        )
        # ratio annotation above the column
        ymax_local = max(g["par_hi"], g["null95"])
        axB.text(
            i, ymax_local * 1.10,
            f"{g['ratio_par']:+.3f}%",
            ha="center", va="bottom", fontsize=10.0,
            color=color, fontweight="bold",
        )
    else:
        # Qwen — par_natural unmeasured: explicit white-box note
        axB.text(
            i, 0, "par_natural\nunmeasured",
            ha="center", va="center", fontsize=8.6, color=color,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=color, linewidth=1.0),
            zorder=4,
        )
        axB.text(
            i, max(DAT[m]["null95"] for m in ROW_ORDER if DAT[m]["has_par"]) * 1.10,
            "n/a",
            ha="center", va="bottom", fontsize=10.0,
            color=color, fontweight="bold",
        )

# zero line — perfect orthogonality
axB.axhline(0, color="#222", lw=1.1, zorder=1)

axB.set_xticks(xpos)
axB.set_xticklabels(xlabels, fontsize=9.2)
axB.set_xlim(-0.6, len(ROW_ORDER) - 0.4)
axB.set_ylabel(r"$\parallel$-component projection" + "\n(natural-norm)",
               fontsize=10.2)
ylim_B = max(DAT[k]["null95"] for k in ROW_ORDER if DAT[k]["has_par"]) * 1.55
axB.set_ylim(-ylim_B, ylim_B)
axB.grid(axis="y", color="#EEEEEE", linewidth=0.6, zorder=0)

axB.set_title(
    r"$\bf{B}$    $\parallel$-component is contained in the random null band",
    loc="left", fontsize=11.6, pad=10, color="#111",
)
axB.text(0.5, 0.96, r"ratio  $\parallel / \mathrm{full}$",
         transform=axB.transAxes, ha="center", va="top",
         fontsize=8.6, color="#555", style="italic")

# ── Shared legend below the figure ────────────────────────────────────────────
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#444",
                  linewidth=1.6, label=r"full $\Delta m$  (95% CI)"),
    plt.Rectangle((0, 0), 1, 1, facecolor="#444", edgecolor="#444",
                  linewidth=0,   label=r"$\perp$ component  (Panel A)  /  $\parallel$ component  (Panel B)"),
    plt.Rectangle((0, 0), 1, 1, facecolor="#DDDDDD", edgecolor="#888",
                  linewidth=0.7, label="random-direction 95% null band  (Panel B)"),
    Line2D([0], [0], color="#222", lw=1.1, label="perfect orthogonality (∥ = 0)"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center", ncol=2, frameon=False,
    fontsize=9.0, bbox_to_anchor=(0.5, 0.005),
    columnspacing=2.6, handletextpad=0.8, handlelength=2.2,
)

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    "Action steering decomposition is universal across model families",
    fontsize=12.8, fontweight="bold", color="#111",
    x=0.012, y=0.965, ha="left",
)

out_png = OUT / "crossfamily_v4.png"
out_pdf = OUT / "crossfamily_v4.pdf"
fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.18)
fig.savefig(out_pdf,            bbox_inches="tight", pad_inches=0.18)
print(f"[v4] wrote {out_png}")
print(f"[v4] wrote {out_pdf}")
