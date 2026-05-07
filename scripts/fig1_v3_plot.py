#!/usr/bin/env python3
"""Figure 1 v3 — Double-Dissociation bar chart.

Two side-by-side horizontal bar panels, same row order:
  Left  : Evidence-sufficiency AUROC (→ who detects evidence?)
  Right : Causal operativity |Δm|_perp (→ who shifts the action margin?)

Reading rule: no row is long in BOTH panels.
Evidence probes (blue) : long left, short right.
Operative dirs (green) : short left, long right.
Inert controls (red)  : short on both sides.

Cross-layer trajectory inset (bottom-right): AUROC vs layer for
the evidence probe family → evidence concentrates at L20.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "results/fig1_v3/results.json"
EXT  = ROOT / "results/fig1_v3/extract_summary.json"
OUT  = ROOT / "results/fig1_geometry"; OUT.mkdir(parents=True, exist_ok=True)

data  = json.load(open(SRC));  DIRS  = data["directions"]
EDIRS = json.load(open(EXT))["directions"]
NULL_P95  = 0.123
A_FULL_DM = 0.801
CHANCE    = 0.5

# ── rows: name, display label, AUROC, |Δm|_perp, CI-lo, CI-hi, color, family
ROWS = []
SPECS = [
    # evidence family
    ("E1_LR_L20",    "E1  LR probe (L20)",      "#2166ac", "evidence"),
    ("E2_Ridge_L20", "E2  Ridge probe (L20)",    "#4393c3", "evidence"),
    # operative family
    ("O1_D3prime",   "O1  D3′  contrastive",     "#1b7837", "operative"),
    ("O2_D1_source", "O2  D1  source-token",     "#4dac26", "operative"),
    ("O3_joint",     "O3  joint(D3′+D1)",        "#74c476", "operative"),
    # inert controls
    ("I1_D2bal",     "I1  D2-bal  (inert ctrl)", "#d6604d", "inert"),
    ("I2_D4_obslen", "I2  D4 obs-len (inert)",   "#f4a582", "inert"),
]
for key, lbl, col, fam in SPECS:
    s = DIRS[key]
    ROWS.append(dict(key=key, lbl=lbl, col=col, fam=fam,
                     auroc=s["oof_auroc"],
                     dm=s["abs_signed_mean_dm"],
                     ci_lo=s["abs_signed_mean_dm_ci"][0],
                     ci_hi=s["abs_signed_mean_dm_ci"][1]))

# Cross-layer chain
CHAIN = ["ExL12_LR", "ExL16_LR", "E1_LR_L20", "ExL24_LR"]
CLBLS = ["L12", "L16", "L20", "L24"]
CHAIN_LAYER = [12, 16, 20, 24]

# ── layout ────────────────────────────────────────────────────────────────
N = len(ROWS)
ys = np.arange(N)  # row positions (top=0 → bottom=N-1)

fig = plt.figure(figsize=(13, 7))
gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 0.06, 1],
                       left=0.22, right=0.97, top=0.88, bottom=0.10,
                       wspace=0.04)
axL  = fig.add_subplot(gs[0])   # left: AUROC
axM  = fig.add_subplot(gs[1])   # middle: divider
axR  = fig.add_subplot(gs[2])   # right: |Δm|_perp

# Inset: cross-layer trajectory (bottom-right corner)
ax_ins = fig.add_axes([0.76, 0.11, 0.20, 0.28])

# ── helper ───────────────────────────────────────────────────────────────
def draw_panel(ax, value_fn, xlim, xlabel, threshold, thresh_label, shade_high=True):
    for i, row in enumerate(ROWS):
        val, lo, hi = value_fn(row)
        y = N - 1 - i
        ax.barh(y, val, height=0.55, color=row["col"], alpha=0.82, left=0, zorder=3)
        if lo != val:
            ax.plot([lo, hi], [y, y], color=row["col"], lw=2.0,
                    alpha=0.6, zorder=4, solid_capstyle="round")
        lx = val + (xlim[1] - xlim[0]) * 0.02
        ax.text(lx, y, f"{val:.2f}", va="center", fontsize=8.5, color=row["col"])
    ax.axvline(threshold, color="#555", lw=1.1, ls="--", alpha=0.55, zorder=2)
    ax.text(threshold, -0.7, thresh_label, ha="center", fontsize=7.5,
            color="#555", style="italic")
    if shade_high:
        ax.axvspan(threshold, xlim[1], facecolor="#fffde7", alpha=0.45, zorder=0)
    ax.set_xlim(*xlim); ax.set_ylim(-1, N)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=10, labelpad=6)
    ax.grid(axis="x", alpha=0.20, zorder=0)
    ax.spines[["top","right","left"]].set_visible(False)

# separator lines between families
def add_separators(ax):
    for gap_after in [1, 4]:
        ax.axhline(N - 1 - gap_after - 0.5, color="#bbb", lw=1.0, alpha=0.5)

# ── LEFT panel: AUROC ─────────────────────────────────────────────────────
draw_panel(axL,
           lambda r: (r["auroc"], r["auroc"], r["auroc"]),
           xlim=(0.44, 1.02),
           xlabel="Evidence-sufficiency AUROC\n(5-fold OOF, N=486)",
           threshold=0.73, thresh_label="AUROC=0.73\n(strong evidence)")
axL.set_title("Does it detect\nevidence?", fontsize=12, fontweight="bold",
              pad=8, color="#333")
axL.axvline(CHANCE, color="#bbb", lw=0.8, ls=":", alpha=0.7)
axL.text(CHANCE - 0.005, N - 0.3, "chance", fontsize=7, color="#aaa",
         ha="right", style="italic")
for i, row in enumerate(ROWS):
    axL.text(-0.005, N - 1 - i, row["lbl"],
             ha="right", va="center", fontsize=9.2, color=row["col"],
             fontweight="bold", transform=axL.get_yaxis_transform())
add_separators(axL)

# ── middle divider ─────────────────────────────────────────────────────────
axM.axis("off")
axM.text(0.5, 0.5, "↔", ha="center", va="center",
         fontsize=20, color="#aaa", transform=axM.transAxes)

# ── RIGHT panel: |Δm|_perp ────────────────────────────────────────────────
draw_panel(axR,
           lambda r: (r["dm"], r["ci_lo"], r["ci_hi"]),
           xlim=(0.0, 2.0),
           xlabel="|Δm|_perp   (factor=2.0 flip, N=100)\ncausal operativity in null(A)",
           threshold=NULL_P95, thresh_label=f"p95 null\n={NULL_P95:.3f}",
           shade_high=False)
axR.set_title("Does it shift the\naction margin?", fontsize=12, fontweight="bold",
              pad=8, color="#333")
add_separators(axR)

# ── figure-level title + subtitle ─────────────────────────────────────────
fig.text(0.50, 0.96,
         "Evidence encoding and causal operativity are dissociated at L20",
         ha="center", fontsize=13, fontweight="bold", color="#222")
fig.text(0.50, 0.925,
         "No direction is long in BOTH panels  ·  "
         "All directions: cos(d, A) ≈ 0  (range −0.19 to +0.19)",
         ha="center", fontsize=9, color="#666", style="italic")

# ── legend ────────────────────────────────────────────────────────────────
handles = [
    Line2D([0],[0], lw=8, color="#2166ac", alpha=0.82, label="evidence probe"),
    Line2D([0],[0], lw=8, color="#1b7837", alpha=0.82, label="operative direction"),
    Line2D([0],[0], lw=8, color="#d6604d", alpha=0.82, label="inert control (matched cos·A≈0)"),
]
axL.legend(handles=handles, loc="lower right", fontsize=8.5,
           frameon=True, framealpha=0.95, edgecolor="#ccc")

# ── inset: cross-layer AUROC trajectory ───────────────────────────────────
cauroc = [DIRS[c]["oof_auroc"] for c in CHAIN]
ax_ins.plot(CHAIN_LAYER, cauroc, "o-", color="#2166ac", lw=2, ms=7, zorder=4)
ax_ins.axhline(NULL_P95, color="#4caf50", lw=0.9, ls="--", alpha=0.6)
ax_ins.set_xlabel("Layer", fontsize=8.5)
ax_ins.set_ylabel("Evidence AUROC", fontsize=8.5)
ax_ins.set_xticks(CHAIN_LAYER); ax_ins.set_ylim(0.48, 0.90)
ax_ins.grid(alpha=0.20)
ax_ins.set_title("Cross-layer evidence\ntrajectory", fontsize=8.5, pad=4)
for lx, ly, lb in zip(CHAIN_LAYER, cauroc, CLBLS):
    ax_ins.text(lx, ly + 0.025, lb, ha="center", fontsize=7.5, color="#2166ac",
                fontweight="bold" if lb == "L20" else "normal")
ax_ins.spines[["top","right"]].set_visible(False)

plt.savefig(OUT / "fig1_v3.png", dpi=170, bbox_inches="tight")
print(f"Saved: {OUT/'fig1_v3.png'}")


