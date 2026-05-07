#!/usr/bin/env python3
"""Cross-family decomposition v5 — vertical bars (fig1_butterfly style) + forest.

  Panel A  —  vertical grouped bar chart per model: full Δm vs ⊥ component.
              Light-fill bars + colored edges + CI whiskers + filled-circle
              markers, matching the visual language of fig1_butterfly_v3.
              The two bars per model reach the same height -> ⊥ ≈ full.

  Panel B  —  horizontal forest plot of ∥ component vs random null band.
              Per-model null band (gray rectangle), measured ∥ marker + CI.

Legends moved to figure-level bottom (no overlap with axes).
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results/crossfamily_ci_decomposition/crossfamily_table.json"
OUT  = ROOT / "results/fig_crossfamily_best"
OUT.mkdir(parents=True, exist_ok=True)
tbl = json.load(open(DATA))

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10.0,
    "axes.linewidth":     0.9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.major.width":  0.7, "xtick.major.size": 3.5,
    "ytick.major.width":  0.7, "ytick.major.size": 3.5,
    "legend.framealpha":  0.96,
    "legend.edgecolor":   "#CCCCCC",
})

NICE   = {"qwen": "Qwen2.5-7B-Instruct", "gemma": "Gemma-2-9B-it",  "mistral": "Mistral-7B-v0.3"}
LAYER  = {"qwen": 20, "gemma": 37, "mistral": 28}
DIM    = {"qwen": 3584, "gemma": 3584, "mistral": 4096}
COLORS = {"qwen": "#1F77B4", "gemma": "#D9822B", "mistral": "#2CA02C"}
ORDER_X = ["qwen", "gemma", "mistral"]    # left-to-right in Panel A
ORDER_Y = ["mistral", "gemma", "qwen"]    # bottom-to-top in Panel B


def get(key):
    m = tbl[key]
    full = m["full"]["mean"];     full_lo = m["full"]["ci_low"];     full_hi = m["full"]["ci_high"]
    perp = m["perp_rms"]["mean"]; perp_lo = m["perp_rms"]["ci_low"]; perp_hi = m["perp_rms"]["ci_high"]
    if m["par_natural"] is not None:
        par   = m["par_natural"]["mean"]
        par_lo= m["par_natural"]["ci_low"]
        par_hi= m["par_natural"]["ci_high"]
        null95= m["stopping_rule_no_renorm"]["rhs"]
        ratio = m["ratio_par_natural_over_full"] * 100
        has_par = True
    else:
        par = par_lo = par_hi = None
        null95 = None;  ratio = None;  has_par = False
    return dict(
        full=full, full_lo=full_lo, full_hi=full_hi,
        perp=perp, perp_lo=perp_lo, perp_hi=perp_hi,
        par=par, par_lo=par_lo, par_hi=par_hi,
        null95=null95,
        ratio_perp=m["ratio_perp_over_full"]*100,
        ratio_par=ratio,
        cos_ae=m["cos_action_evidence"],
        n=m["n_samples"],
        layer=LAYER[key], d=DIM[key],
        has_par=has_par,
    )

DAT = {k: get(k) for k in ORDER_X}

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13.0, 5.4))
gs = fig.add_gridspec(
    1, 2,
    width_ratios=[1.0, 1.05],
    left=0.075, right=0.985,
    top=0.83, bottom=0.30,        # extra bottom for shared legend
    wspace=0.30,
)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])


# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — vertical grouped bars  (fig1_butterfly visual language)
# ══════════════════════════════════════════════════════════════════════════════
BAR_W = 0.34
GAP   = 0.04
xpos  = np.arange(len(ORDER_X))
x_full = xpos - (BAR_W/2 + GAP/2)
x_perp = xpos + (BAR_W/2 + GAP/2)

for i, k in enumerate(ORDER_X):
    g = DAT[k]; c = COLORS[k]

    # full Δm bar — light fill + colored edge + diagonal hatch (ref-style)
    axA.bar(
        x_full[i], g["full"], width=BAR_W,
        color=c, alpha=0.18,
        edgecolor=c, linewidth=1.4,
        hatch="///", zorder=2,
    )
    # CI whisker (vertical line) on full
    axA.plot([x_full[i], x_full[i]], [g["full_lo"], g["full_hi"]],
             color=c, lw=1.6, alpha=0.85, zorder=4)
    # marker — open square for full
    axA.plot(x_full[i], g["full"], "s",
             markerfacecolor="white", markeredgecolor=c, markeredgewidth=1.6,
             markersize=10, zorder=5)
    # numeric label above bar
    axA.text(x_full[i], g["full_hi"] + 0.10, f"{g['full']:.3f}",
             ha="center", va="bottom", fontsize=9.0, color=c, fontweight="bold",
             zorder=6)

    # perp bar — light fill + colored edge + filled circle
    axA.bar(
        x_perp[i], g["perp"], width=BAR_W,
        color=c, alpha=0.40,
        edgecolor=c, linewidth=1.4,
        zorder=2,
    )
    axA.plot([x_perp[i], x_perp[i]], [g["perp_lo"], g["perp_hi"]],
             color=c, lw=1.6, alpha=0.85, zorder=4)
    axA.plot(x_perp[i], g["perp"], "o",
             markerfacecolor=c, markeredgecolor="white", markeredgewidth=1.0,
             markersize=11, zorder=5)
    axA.text(x_perp[i], g["perp_hi"] + 0.10, f"{g['perp']:.3f}",
             ha="center", va="bottom", fontsize=9.0, color=c, fontweight="bold",
             zorder=6)

    # ratio annotation centered under each model group
    axA.text(xpos[i], -0.32, f"⊥ / full = {g['ratio_perp']:.2f}%",
             ha="center", va="top", fontsize=9.2, color=c,
             fontweight="bold", transform=axA.transData)

axA.set_xticks(xpos)
axA.set_xticklabels(
    [f"{NICE[k]}\nL{DAT[k]['layer']}  d={DAT[k]['d']}  N={DAT[k]['n']}"
     for k in ORDER_X],
    fontsize=9.4,
)
# nudge tick labels down so they don't collide with the ratio annotation
axA.tick_params(axis="x", pad=24)

axA.set_ylabel(r"Projection magnitude on $\hat{\mathbf{a}}$  (rms-normalized)",
               fontsize=10.2)
ymax = max(DAT[k]["full_hi"] for k in ORDER_X) * 1.18
axA.set_ylim(0, ymax)
axA.set_xlim(-0.6, len(ORDER_X) - 0.4)
axA.grid(axis="y", color="#EEEEEE", linewidth=0.6, zorder=0)

axA.set_title(
    r"$\bf{A}$    $\perp$-component captures the full effect",
    loc="left", fontsize=11.6, pad=10, color="#111",
)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — horizontal forest plot of ∥ vs random null band
# ══════════════════════════════════════════════════════════════════════════════
ROW_HALF = 0.32
ypos = np.arange(len(ORDER_Y))

for i, k in enumerate(ORDER_Y):
    g = DAT[k]; c = COLORS[k]
    if g["has_par"]:
        # null-band background per model
        axB.add_patch(Rectangle(
            (-g["null95"], i - ROW_HALF),
            2 * g["null95"], 2 * ROW_HALF,
            facecolor="#DDDDDD", alpha=0.55, edgecolor="none", zorder=0,
        ))
        for sign in (-1, +1):
            axB.plot([sign * g["null95"]] * 2,
                     [i - ROW_HALF, i + ROW_HALF],
                     color="#888", lw=0.7, zorder=0.5)
        # measured ∥ marker + CI
        axB.plot([g["par_lo"], g["par_hi"]], [i, i],
                 color=c, lw=1.6, alpha=0.85, zorder=4)
        axB.plot(g["par"], i, "o",
                 markerfacecolor=c, markeredgecolor="white", markeredgewidth=1.0,
                 markersize=11, zorder=5)
        axB.text(g["null95"] * 1.10, i,
                 f"∥ / full = {g['ratio_par']:+.3f}%",
                 va="center", ha="left", fontsize=8.8, color=c, fontweight="bold")
    else:
        axB.text(
            0, i, "par_natural unmeasured\n(direction-level cos(A,E) only)",
            ha="center", va="center", fontsize=8.6, color=c, style="italic",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=c, linewidth=1.0),
            zorder=4,
        )

axB.axvline(0, color="#222", lw=1.1, zorder=1)
axB.text(0, len(ORDER_Y) - 0.32, "perfect\northogonality",
         ha="center", va="bottom", fontsize=8.2, color="#222", style="italic")

axB.set_yticks(ypos)
axB.set_yticklabels(
    [f"{NICE[k]}\nL{DAT[k]['layer']}  d={DAT[k]['d']}  N={DAT[k]['n']}"
     for k in ORDER_Y],
    fontsize=9.4,
)
axB.set_ylim(-0.55, len(ORDER_Y) - 0.30)
axB.set_xlabel(r"$\parallel$-component projection  (natural-norm)",
               fontsize=10.2)
xlim_B = max(DAT[k]["null95"] for k in ORDER_Y if DAT[k]["has_par"]) * 2.4
axB.set_xlim(-xlim_B, xlim_B)
axB.grid(axis="x", color="#EEEEEE", linewidth=0.6, zorder=0)
axB.tick_params(axis="y", left=False)

axB.set_title(
    r"$\bf{B}$    $\parallel$-component is contained in the random null band",
    loc="left", fontsize=11.6, pad=10, color="#111",
)

# ══════════════════════════════════════════════════════════════════════════════
# Shared figure-level legend (bottom)
# ══════════════════════════════════════════════════════════════════════════════
legend_handles = [
    Rectangle((0, 0), 1, 1, facecolor="#888", alpha=0.18, edgecolor="#444",
              linewidth=1.2, hatch="///",
              label=r"full $\Delta m$  (open square + hatched bar)"),
    Rectangle((0, 0), 1, 1, facecolor="#888", alpha=0.40, edgecolor="#444",
              linewidth=1.2,
              label=r"$\perp$ component  (filled circle + solid bar)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#444",
           markeredgecolor="white", markersize=10,
           label=r"measured $\parallel$  (Panel B, 95% CI whisker)"),
    Rectangle((0, 0), 1, 1, facecolor="#DDDDDD", alpha=0.55, edgecolor="#888",
              linewidth=0.7,
              label="random-direction 95% null band  (per-model)"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center", bbox_to_anchor=(0.5, 0.015),
    ncol=2, fontsize=9.0, frameon=True, framealpha=0.94,
    edgecolor="#CCCCCC", columnspacing=2.4, handlelength=2.8,
)

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    "Action steering decomposition is universal across model families",
    fontsize=12.8, fontweight="bold", color="#111",
    x=0.012, y=0.965, ha="left",
)

out_png = OUT / "crossfamily_v5.png"
out_pdf = OUT / "crossfamily_v5.pdf"
fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.18)
fig.savefig(out_pdf,            bbox_inches="tight", pad_inches=0.18)
print(f"[v5] wrote {out_png}")
print(f"[v5] wrote {out_pdf}")

