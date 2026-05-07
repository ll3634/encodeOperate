#!/usr/bin/env python3
"""Best-practice cross-family decomposition figure (v2).

Two-panel design chosen for maximum information density:

  Panel A  –  perp vs. full scatter  (square axes, y = x reference)
              Three model families form a tight cluster on the y=x diagonal,
              proving   ⊥-component ≈ full Δm   universally.

  Panel B  –  |par_natural| / null_upper  (normalized ratio, threshold = 1)
              Gemma and Mistral both sit at ≈0.22 of their null ceiling,
              proving  ∥-component ≤ noise floor  universally.
              Qwen shown as N/A (par_rms is opposite-sign normalization artifact).

Output: results/fig_crossfamily_best/crossfamily_v2.{png,pdf}
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results/crossfamily_ci_decomposition/crossfamily_table.json"
OUT  = ROOT / "results/fig_crossfamily_best"
OUT.mkdir(parents=True, exist_ok=True)

tbl = json.load(open(DATA))

# ── Publication rcParams ───────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          9.5,
    "axes.linewidth":     0.9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.major.width":  0.7, "xtick.major.size": 3.5,
    "ytick.major.width":  0.7, "ytick.major.size": 3.5,
    "legend.framealpha":  0.93,
    "legend.edgecolor":   "#CCCCCC",
    "legend.fontsize":    8.4,
})

# ── Model descriptors ──────────────────────────────────────────────────────────
# ── Figure skeleton ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12.5, 5.0))
gs = fig.add_gridspec(
    1, 2,
    width_ratios=[1.05, 0.95],
    left=0.07, right=0.97,
    top=0.87, bottom=0.14,
    wspace=0.32,
)
axA = fig.add_subplot(gs[0, 0])   # Panel A: scatter perp vs full
axB = fig.add_subplot(gs[0, 1])   # Panel B: |par| / null_upper

MODELS = [
    dict(key="qwen",
         label="Qwen2.5-7B\n(L20, d=3584)", short="Qwen",
         color="#2563EB", marker="o",
         cos=-0.0135,
         full_m=0.910,  full_lo=0.841,  full_hi=0.980,
         perp_m=0.909,  perp_lo=0.839,  perp_hi=0.979,
         par_m=None, par_lo=None, par_hi=None, null_hi=None),
    dict(key="gemma",
         label="Gemma-2-9B-it\n(L37, d=3584)", short="Gemma",
         color="#D97706", marker="s",
         cos=+0.0110,
         full_m=tbl["gemma"]["full"]["mean"],
         full_lo=tbl["gemma"]["full"]["ci_low"],
         full_hi=tbl["gemma"]["full"]["ci_high"],
         perp_m=tbl["gemma"]["perp_rms"]["mean"],
         perp_lo=tbl["gemma"]["perp_rms"]["ci_low"],
         perp_hi=tbl["gemma"]["perp_rms"]["ci_high"],
         par_m=tbl["gemma"]["par_natural"]["mean"],
         par_lo=tbl["gemma"]["par_natural"]["ci_low"],
         par_hi=tbl["gemma"]["par_natural"]["ci_high"],
         null_hi=0.03381),
    dict(key="mistral",
         label="Mistral-7B-v0.3\n(L28, d=4096)", short="Mistral",
         color="#059669", marker="^",
         cos=-0.0090,
         full_m=tbl["mistral"]["full"]["mean"],
         full_lo=tbl["mistral"]["full"]["ci_low"],
         full_hi=tbl["mistral"]["full"]["ci_high"],
         perp_m=tbl["mistral"]["perp_rms"]["mean"],
         perp_lo=tbl["mistral"]["perp_rms"]["ci_low"],
         perp_hi=tbl["mistral"]["perp_rms"]["ci_high"],
         par_m=tbl["mistral"]["par_natural"]["mean"],
         par_lo=tbl["mistral"]["par_natural"]["ci_low"],
         par_hi=tbl["mistral"]["par_natural"]["ci_high"],
         null_hi=0.01966),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Panel A  –  perp vs. full  (scatter on y=x)
# ═══════════════════════════════════════════════════════════════════════════════
AX_LO, AX_HI = 0.65, 3.45

# --- perfect equality reference (y = x) ---
ref_x = np.array([AX_LO, AX_HI])
axA.plot(ref_x, ref_x, color="#9CA3AF", lw=1.4, ls="--", zorder=1,
         label="$y = x$  (perfect equality)")

# --- 1-1 band: ± 1 % of x ---
axA.fill_between(ref_x, ref_x * 0.99, ref_x * 1.01,
                 color="#E5E7EB", alpha=0.55, zorder=0, label="±1% band")

for m in MODELS:
    c, mk = m["color"], m["marker"]
    xerr = [[m["full_m"] - m["full_lo"]], [m["full_hi"] - m["full_m"]]]
    yerr = [[m["perp_m"] - m["perp_lo"]], [m["perp_hi"] - m["perp_m"]]]
    axA.errorbar(m["full_m"], m["perp_m"],
                 xerr=xerr, yerr=yerr,
                 fmt="none", ecolor=c, elinewidth=1.8, capsize=4,
                 capthick=1.5, zorder=3, alpha=0.7)
    axA.plot(m["full_m"], m["perp_m"],
             marker=mk, ms=13, color=c,
             mec="white", mew=1.8, zorder=5, label=m["short"])

    # Annotate perp/full %
    pct = m["perp_m"] / m["full_m"] * 100
    dx = 0.10 if m["key"] != "qwen" else 0.10
    dy = -0.12 if m["key"] == "gemma" else 0.09
    axA.text(m["full_m"] + dx, m["perp_m"] + dy,
             f"{pct:.1f}%", fontsize=8.2, color=c,
             fontweight="bold", va="center", zorder=6)

axA.set_xlim(AX_LO, AX_HI)
axA.set_ylim(AX_LO, AX_HI)
axA.set_aspect("equal", adjustable="box")
axA.set_xlabel("Full  $|\\Delta m|$  (natural-norm)", fontsize=10.5)
axA.set_ylabel("Orthogonal  $|\\Delta m_\\perp|$  (natural-norm)", fontsize=10.5)
axA.set_xticks([1.0, 1.5, 2.0, 2.5, 3.0])
axA.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0])
axA.set_title("(A)  $\\perp$ component $\\approx$ full effect  —  all three families",
              fontsize=10.5, loc="left", pad=6, fontweight="bold")
axA.legend(loc="upper left", fontsize=8.5,
           handlelength=1.2, handletextpad=0.5, borderpad=0.7,
           title="model family  ·  95% CI whiskers",
           title_fontsize=7.8)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel B  –  |par_natural| normalized to model-specific null_upper
# ═══════════════════════════════════════════════════════════════════════════════
models_with_par = [m for m in MODELS if m["par_m"] is not None]
ys_b = np.arange(len(models_with_par))[::-1]   # top-to-bottom

# null band: 0 → 1 (normalized)
axB.axvspan(0, 1.0, color="#E5E7EB", alpha=0.65, zorder=0)
axB.axvline(1.0, color="#6B7280", lw=1.4, ls="--", zorder=1)
axB.text(1.02, len(models_with_par) - 0.45,
         "null\nthreshold", fontsize=7.8, color="#6B7280",
         style="italic", va="top")

for y_b, m in zip(ys_b, models_with_par):
    c = m["color"]
    ratio      = m["par_m"]   / m["null_hi"]
    ratio_lo   = m["par_lo"]  / m["null_hi"]
    ratio_hi   = m["par_hi"]  / m["null_hi"]

    # CI bar
    axB.plot([ratio_lo, ratio_hi], [y_b, y_b],
             color=c, lw=2.2, alpha=0.80, zorder=3,
             solid_capstyle="round")
    # Point
    axB.plot(ratio, y_b, marker=m["marker"], ms=13, color=c,
             mec="white", mew=1.8, zorder=5)
    # Ratio label: "0.22× (0.0075 / 0.034)"
    axB.text(ratio_hi + 0.05, y_b + 0.14,
             f"{ratio:.2f}\u00d7  ({m['par_m']:.4f} / {m['null_hi']:.4f})",
             va="center", fontsize=8.2, color=c, fontweight="bold", zorder=6)
    # Model name label (on y)
    axB.text(-0.10, y_b, m["short"], ha="right", va="center",
             fontsize=10.5, fontweight="bold", color=c)

# Qwen N/A annotation (no par_natural)
qwen = next(m for m in MODELS if m["key"] == "qwen")
y_qwen = len(models_with_par)   # above the range
axB.text(0.50, -0.62,
         f"Qwen: par_natural N/A  (par_rms = −0.157, opposite-sign normalization artifact)",
         ha="center", va="top", fontsize=7.8, color=qwen["color"],
         style="italic", zorder=6)

axB.set_xlim(-0.05, 1.55)
axB.set_ylim(-0.75, len(models_with_par) - 0.28)
axB.set_yticks([])
axB.set_xticks([0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50])
axB.set_xticklabels(["0", "0.25×", "0.50×", "0.75×", "1.0×", "1.25×", "1.50×"],
                    fontsize=8.2)
axB.set_xlabel("$|\\mathrm{par_{natural}}|$ / null-band ceiling  (model-specific)",
               fontsize=10.5)
axB.set_title("(B)  $\\parallel$ component $\\leq$ noise floor  —  in null band",
              fontsize=10.5, loc="left", pad=6, fontweight="bold")
axB.spines["left"].set_visible(False)

# null-band legend
nb_patch = mpatches.Patch(fc="#E5E7EB", alpha=0.65,
                           label="Null band  [0, 1.0×]  (abs-p97.5, K=100 random dirs)")
axB.legend(handles=[nb_patch], loc="upper right", fontsize=7.8,
           handlelength=1.2, borderpad=0.7)

# ── suptitle + caption ─────────────────────────────────────────────────────────
fig.suptitle(
    "Functional Orthogonality: Steering Effect Lives Entirely in the "
    "Evidence-Orthogonal Subspace  —  Cross-Family Replication",
    fontsize=11.5, fontweight="bold", y=0.972,
)
CAPTION = (
    "(A) All three model families lie on $y=x$: the $\\perp$-component captures"
    " $\\geq$99.9% of the total effect (bootstrap 95% CI).  "
    "(B) The $\\parallel$-component (natural-norm) sits at $\\leq$22% of the"
    " model-specific null ceiling, indistinguishable from a random direction"
    " ($p_e$ = 0.96 / 0.21 for Gemma / Mistral, 10k permutations).  "
    "N = 100 (Qwen), 50 (Gemma, Mistral).  Error bars: bootstrap 95% CI."
)
fig.text(0.5, 0.01, CAPTION, ha="center", va="bottom", fontsize=7.8,
         color="#4B5563", linespacing=1.35)

# ── save ──────────────────────────────────────────────────────────────────────
for suffix, kw in [(".png", dict(dpi=200)), (".pdf", {})]:
    p = OUT / f"crossfamily_v2{suffix}"
    fig.savefig(p, bbox_inches="tight", **kw)
    print(f"✓  {p}")

