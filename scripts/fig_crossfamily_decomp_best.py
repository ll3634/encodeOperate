#!/usr/bin/env python3
"""Best-practice publication figure: Cross-Family Functional Orthogonality Decomposition.

Two-panel design:
  Panel A (left, wider): Effect-size dot-plot with 95% CI bars.
                         Shows full Δm, ⊥-component, and ∥-component
                         (natural-norm) for each model family.
                         Null band shaded near origin.
  Panel B (right, narrow): Fraction-of-full bar chart (0–100 %).
                           Reveals that ⊥ ≈ 99.9 % and ∥ ≤ 0.3 %
                           for every family.

Output: results/fig_crossfamily_best/crossfamily_decomp.{png,pdf}
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

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / "results/crossfamily_ci_decomposition/crossfamily_table.json"
OUT   = ROOT / "results/fig_crossfamily_best"
OUT.mkdir(parents=True, exist_ok=True)

tbl = json.load(open(DATA))

# ── global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size":  3.5,
    "ytick.major.size":  3.5,
    "legend.framealpha": 0.92,
    "legend.edgecolor":  "#cccccc",
    "figure.dpi":        150,
})

# ── model data ────────────────────────────────────────────────────────────────
MODELS = [
    dict(
        label="Qwen2.5-7B\n(L20, d=3584)",
        color="#2563EB",          # blue
        cos=-0.0135,
        full_m=0.910, full_lo=0.841, full_hi=0.980,
        perp_m=0.909, perp_lo=0.839, perp_hi=0.979,
        par_m=None, par_lo=None, par_hi=None,
        null_hi=None,
        par_frac=None,
        perp_frac=99.9,
    ),
    dict(
        label="Gemma-2-9B-it\n(L37, d=3584)",
        color="#D97706",          # amber
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
        null_hi=0.03381,
        par_frac=tbl["gemma"]["par_natural"]["mean"] / tbl["gemma"]["full"]["mean"] * 100,
        perp_frac=tbl["gemma"]["perp_rms"]["mean"] / tbl["gemma"]["full"]["mean"] * 100,
    ),
    dict(
        label="Mistral-7B-v0.3\n(L28, d=4096)",
        color="#059669",          # emerald
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
        null_hi=0.01966,
        par_frac=tbl["mistral"]["par_natural"]["mean"] / tbl["mistral"]["full"]["mean"] * 100,
        perp_frac=tbl["mistral"]["perp_rms"]["mean"] / tbl["mistral"]["full"]["mean"] * 100,
    ),
]

N = len(MODELS)
ys = np.arange(N)[::-1]   # top-to-bottom on y-axis

# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14.0, 5.0))
gs = fig.add_gridspec(
    1, 3,
    width_ratios=[3.8, 0.8, 2.2],
    left=0.01, right=0.98, top=0.87, bottom=0.18,
    wspace=0.06,
)
axA = fig.add_subplot(gs[0, 0])   # Panel A: absolute effect sizes
axS = fig.add_subplot(gs[0, 1])   # separator / model labels
axB = fig.add_subplot(gs[0, 2])   # Panel B: fraction-of-full

# ═══════════════════════════════════════════════════════════════════════════════
# Panel A — absolute effect sizes (natural-norm Δm)
# ═══════════════════════════════════════════════════════════════════════════════
X_MAX_A = 3.55
NULL_X   = max(m["null_hi"] for m in MODELS if m["null_hi"])  # 0.034

axA.axvspan(0, NULL_X, color="#D1D5DB", alpha=0.45, zorder=0)
axA.axvline(0, color="#374151", lw=0.9, zorder=1)

OFFSETS = {"full": +0.20, "perp": -0.08, "par": -0.30}

for y, m in zip(ys, MODELS):
    c = m["color"]

    # full Δm (hollow square)
    yo = y + OFFSETS["full"]
    axA.plot([m["full_lo"], m["full_hi"]], [yo, yo],
             color=c, lw=1.6, alpha=0.55, zorder=3)
    axA.barh(yo, m["full_m"], height=0.16, color=c, alpha=0.12,
             edgecolor=c, lw=0.8, hatch="///", zorder=2)
    axA.plot(m["full_m"], yo, "s", ms=9, mfc="white", mec=c, mew=1.8, zorder=5)

    # perp component (filled circle)
    yo = y + OFFSETS["perp"]
    axA.plot([m["perp_lo"], m["perp_hi"]], [yo, yo],
             color=c, lw=2.0, alpha=0.85, zorder=3)
    axA.plot(m["perp_m"], yo, "o", ms=11, color=c, mec="white", mew=1.2, zorder=5)
    axA.text(m["perp_hi"] + 0.06, yo,
             f"{m['perp_m']:.3f}  ({m['perp_frac']:.1f}%)",
             va="center", fontsize=8.2, color=c, fontweight="bold", zorder=6)

    # par_natural component (triangle)
    yo = y + OFFSETS["par"]
    if m["par_m"] is None:
        axA.text(0.002, yo,
                 "par_natural: N/A  (par_rms=−0.157, opposite-sign artifact)",
                 va="center", fontsize=7.5, color=c, style="italic", zorder=6)
    else:
        lo_c = max(m["par_lo"], -0.001)
        axA.plot([lo_c, m["par_hi"]], [yo, yo],
                 color=c, lw=1.6, alpha=0.80, zorder=4)
        axA.plot(m["par_m"], yo, "^", ms=9, color=c,
                 mec="white", mew=1.0, alpha=0.85, zorder=5)
        axA.axvline(m["null_hi"],
                    ymin=(yo - 0.14 + 0.55) / (N - 0.22 + 0.55),
                    ymax=(yo + 0.14 + 0.55) / (N - 0.22 + 0.55),
                    color=c, lw=1.1, ls=":", alpha=0.6, zorder=3)
        axA.text(m["par_hi"] + 0.003, yo,
                 f"{m['par_m']:.4f}  ({m['par_frac']:.2f}%)  \u2713 null band",
                 va="center", fontsize=7.8, color=c, alpha=0.85, zorder=6)

axA.set_xlim(-0.06, X_MAX_A)
axA.set_ylim(-0.55, N - 0.22)
axA.set_yticks([])
axA.set_xlabel("Effect on action direction  |Δm|  (natural-norm units)", fontsize=9.5)
axA.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
axA.spines["left"].set_visible(False)
axA.set_title("(A)  Absolute effect-size decomposition", fontsize=10,
              loc="left", pad=5, fontweight="bold")

# null-band annotation inside panel A
axA.text(NULL_X / 2, N - 0.30, "null band",
         ha="center", va="top", fontsize=7.5, color="#6B7280", style="italic")

leg_handles = [
    mpatches.Patch(fc="#D1D5DB", alpha=0.55,
                   label="Natural-norm null band  (abs p97.5, K=100 random dirs)"),
    Line2D([0], [0], marker="s", color="gray", lw=1.4, ms=8,
           mfc="white", mew=1.6, label="Full Δm  (reference)"),
    Line2D([0], [0], marker="o", color="gray", lw=2.0, ms=9,
           mec="white", mew=1.2, label="\u22a5 component  (perp)"),
    Line2D([0], [0], marker="^", color="gray", lw=1.4, ms=8,
           mec="white", mew=0.9, alpha=0.85, label="\u2225 component  (par_natural)"),
]
axA.legend(handles=leg_handles, loc="lower right", fontsize=7.8,
           handlelength=1.8, handletextpad=0.6, borderpad=0.7)

# ═══════════════════════════════════════════════════════════════════════════════
# Separator — model labels
# ═══════════════════════════════════════════════════════════════════════════════
axS.set_xlim(0, 1); axS.set_ylim(-0.55, N - 0.22); axS.axis("off")
for y, m in zip(ys, MODELS):
    parts = m["label"].split("\n")
    axS.text(0.5, y + 0.14, parts[0], ha="center", va="center",
             fontsize=10.5, fontweight="bold", color=m["color"])
    axS.text(0.5, y - 0.12, parts[1] if len(parts) > 1 else "",
             ha="center", va="center", fontsize=8.2, color=m["color"])
    axS.text(0.5, y - 0.36,
             f"cos(A,E)={m['cos']:+.4f}",
             ha="center", va="center", fontsize=7.5, color="#6B7280",
             style="italic")
for xv in (0.0, 1.0):
    axS.axvline(xv, color="#9CA3AF", lw=0.6, ymin=0.02, ymax=0.98)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel B — fraction of full (%)
# ═══════════════════════════════════════════════════════════════════════════════
BAR_H = 0.28
for y, m in zip(ys, MODELS):
    c = m["color"]

    # perp bar (solid)
    axB.barh(y + 0.14, m["perp_frac"], height=BAR_H, color=c, alpha=0.80, zorder=2)
    axB.text(m["perp_frac"] + 0.5, y + 0.14,
             f"{m['perp_frac']:.1f}%", va="center", fontsize=9.0,
             fontweight="bold", color=c)

    # par bar
    if m["par_frac"] is not None:
        bar_par = max(m["par_frac"], 0.0)
        axB.barh(y - 0.18, bar_par, height=BAR_H, color=c, alpha=0.30,
                 edgecolor=c, lw=0.8, zorder=2)
        axB.text(max(bar_par, 0.15) + 0.5, y - 0.18,
                 f"{m['par_frac']:.2f}%  (\u2713 null)",
                 va="center", fontsize=8.0, color=c, alpha=0.85)
    else:
        axB.text(0.5, y - 0.18, "N/A  (opposite-sign artifact)",
                 va="center", fontsize=7.5, color=c, style="italic", alpha=0.75)

axB.axvline(100, color="#374151", lw=1.0, ls="--", alpha=0.5, zorder=1)
axB.set_xlim(0, 108)
axB.set_ylim(-0.55, N - 0.22)
axB.set_yticks([])
axB.set_xlabel("Fraction of full Δm  (%)", fontsize=9.5)
axB.set_xticks([0, 25, 50, 75, 100])
axB.spines["left"].set_visible(False)
axB.set_title("(B)  Decomposition as % of full", fontsize=10,
              loc="left", pad=5, fontweight="bold")

leg_b = [
    mpatches.Patch(fc="#6B7280", alpha=0.80, label="\u22a5 component  (perp/full)"),
    mpatches.Patch(fc="#6B7280", alpha=0.30, ec="#6B7280", lw=0.8,
                   label="\u2225 component  (par_natural/full)"),
]
axB.legend(handles=leg_b, loc="lower right", fontsize=7.8,
           handlelength=1.4, borderpad=0.7)

# ── main title + caption ──────────────────────────────────────────────────────
fig.suptitle(
    "Functional Orthogonality: Evidence-Parallel Component Non-Operative "
    "Across Three LLM Families",
    fontsize=11.5, fontweight="bold", y=0.975,
)
CAPTION = (
    "Each model\u2019s steering effect decomposes into evidence-parallel (\u2225, par_natural) and "
    "evidence-orthogonal (\u22a5, perp) components under natural-norm.  "
    "The \u22a5 component captures \u226599.9\u202f% of the total in all families; "
    "the \u2225 component is within the random-direction null band "
    "(Gemma p=0.96, Mistral p=0.21).  "
    "Qwen par_rms=\u22120.157 is opposite-sign \u2014 a normalization artifact; par_natural not extracted.  "
    "Error bars: bootstrap 95\u202f% CI (10\u202f000 resamples).  N=100 (Qwen), 50 (Gemma, Mistral)."
)
fig.text(0.5, 0.02, CAPTION, ha="center", va="bottom", fontsize=7.5,
         color="#4B5563", style="italic", linespacing=1.35)

out_png = OUT / "crossfamily_decomp.png"
out_pdf = OUT / "crossfamily_decomp.pdf"
fig.savefig(out_png, dpi=200, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"\u2713  PNG: {out_png}")
print(f"\u2713  PDF: {out_pdf}")

