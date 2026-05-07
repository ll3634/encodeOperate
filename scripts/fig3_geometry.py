#!/usr/bin/env python3
"""Figure 3: Geometry of operative subspace in null(A).

Two-panel figure:
  Left  — 2D PCA scatter of directions in null(A) plane, with 20 random null
           dots and A shown as an out-of-plane axis label.
  Right — Horizontal bars: abs_signed_mean_dm under perp-protocol (flip x2),
           plus a "joint" bar proving multi-dimensionality.

Theme: "Same cos·A ≈ 0, yet completely different operativity.
         Operativity is subspace membership, not orthogonality to A."
"""

from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

ROOT = Path("tmc/scripts/e2e_agent")
EMB  = json.load(open(ROOT / "results/fig1_pca_embedding/embedding.json"))
OUT  = ROOT / "results/fig3_geometry"
OUT.mkdir(parents=True, exist_ok=True)

# ── Data ────────────────────────────────────────────────────────────────────
coords = EMB["coordinates"]

OPERATIVE  = {"D3p": "D3′", "D1": "D1"}
INERT      = {"E": "E", "D4": "D4", "D2bal": "D2"}
JOINT      = {"joint": "D3′+D1"}

# Perp-protocol abs_signed_mean_dm (corrected model, N=100)
DM = {
    "D3p":   0.8562,
    "D1":    0.4238,
    "D4":    0.0450,
    "D2bal": 0.0675,
    "E":     0.0413,   # from figure_spectrum full-dir (full≈perp for E, cos·A=-0.013)
    "joint": 1.7000,
}
NULL_MEAN = 0.0432
NULL_P95  = 0.1226

COLORS = {
    "D3p": "#2F6FB5", "D1": "#5BA3D9",
    "E": "#C0392B", "D4": "#E67E22", "D2bal": "#8E44AD",
    "joint": "#27AE60",
}
RANDOM_COLOR = "#AAAAAA"

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4),
                         gridspec_kw={"width_ratios": [1.0, 0.85]})
ax_geo, ax_bar = axes

# ══════════════════════════════════════════════════════════════════════════════
# LEFT: 2D PCA scatter
# ══════════════════════════════════════════════════════════════════════════════
# Random null dots
for k in range(20):
    x, y = coords[f"r{k:02d}"]
    ax_geo.plot(x, y, ".", ms=5, color=RANDOM_COLOR, alpha=0.55, zorder=1)

# Named directions as arrows from origin
for key, label in {**OPERATIVE, **INERT}.items():
    x, y = coords[key]
    ax_geo.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=COLORS[key],
                                   lw=1.8, shrinkA=0, shrinkB=4), zorder=4)
    dm_str = f"|Δm|={DM[key]:.3f}"
    # offset label to avoid overlap
    off = 0.06
    ax_geo.text(x + np.sign(x) * off, y + np.sign(y) * off + (0.04 if key == "D4" else 0),
                f"{label}\n{dm_str}", ha="center", va="center",
                fontsize=9.5, color=COLORS[key], fontweight="bold", zorder=5)

# Joint arrow (dashed green)
jx, jy = coords["joint"]
ax_geo.annotate("", xy=(jx * 0.92, jy * 0.92), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=COLORS["joint"],
                                lw=2.2, shrinkA=0, shrinkB=2,
                                linestyle="dashed"), zorder=4)
ax_geo.text(jx + 0.08, jy + 0.06, f"Joint\n|Δm|=1.700",
            ha="center", va="center", fontsize=9.5,
            color=COLORS["joint"], fontweight="bold", zorder=5)

# A axis label (out-of-plane reference)
ax_geo.text(0.02, 0.97,
            "⊗  A (action axis, perpendicular to this plane)",
            transform=ax_geo.transAxes, ha="left", va="top",
            fontsize=9, color="#2E8B57",
            bbox=dict(boxstyle="round,pad=0.3", fc="#F0FFF4", ec="#2E8B57", alpha=0.85))

ax_geo.set_xlabel("PC1 of null(A)  (7.5% var)", fontsize=10)
ax_geo.set_ylabel("PC2 of null(A)  (4.6% var)", fontsize=10)
ax_geo.set_title("Directions projected onto null(A) — 2D PCA embedding\n"
                 "All arrows: cos(dir, A) ≈ 0  (same orthogonality to action axis)",
                 fontsize=10)
ax_geo.axhline(0, lw=0.5, color="0.7"); ax_geo.axvline(0, lw=0.5, color="0.7")
ax_geo.set_aspect("equal")
ax_geo.grid(False)

legend_handles = (
    [Line2D([0],[0], marker=".", color=RANDOM_COLOR, lw=0, ms=7,
             label="K=20 random null directions in null(A)")] +
    [mpatches.Patch(color=COLORS[k], label=f"{v}") for k, v in OPERATIVE.items()] +
    [mpatches.Patch(color=COLORS["joint"], label="Joint D3′+D1")] +
    [mpatches.Patch(color=COLORS[k], label=f"{v} (inert)") for k, v in INERT.items()]
)
ax_geo.legend(handles=legend_handles, fontsize=8, loc="lower left", framealpha=0.9)

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT: Horizontal bar chart of perp-protocol |Δm|
# ══════════════════════════════════════════════════════════════════════════════
bar_order = ["joint", "D3p", "D1", "D4", "D2bal", "E"]
bar_labels = ["Joint\n(D3′+D1)", "D3′", "D1", "D4", "D2", "E"]
bar_vals   = [DM[k] for k in bar_order]
bar_colors = [COLORS[k] for k in bar_order]

y_pos = np.arange(len(bar_order))
ax_bar.barh(y_pos, bar_vals, color=bar_colors, edgecolor="black", linewidth=0.6, height=0.6)
ax_bar.axvline(NULL_MEAN, color="0.4", ls="--", lw=1.2, label=f"null mean={NULL_MEAN:.3f}")
ax_bar.axvline(NULL_P95,  color="0.6", ls=":",  lw=1.0, label=f"null p95={NULL_P95:.3f}")

for i, (v, k) in enumerate(zip(bar_vals, bar_order)):
    ratio = v / NULL_MEAN
    ax_bar.text(v + 0.03, i, f"{v:.3f}  ({ratio:.0f}x null)",
                va="center", fontsize=9, color=COLORS[k], fontweight="bold")

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(bar_labels, fontsize=10)
ax_bar.set_xlabel("mean |signed Δ margin|  (perp-in-null(A) flip, N=100)", fontsize=10)
ax_bar.set_title("Causal effect in null(A) plane\n"
                 "Joint > D3′ + D1  →  subspace ≥ 2D", fontsize=10)
ax_bar.set_xlim(0, 2.10)
ax_bar.legend(fontsize=8, loc="lower right")
ax_bar.grid(axis="x", ls=":", alpha=0.4)
ax_bar.axvline(0, color="black", lw=0.5)

fig.suptitle(
    "Figure 3. Operative subspace geometry at L20: position in null(A) predicts operativity\n"
    "cos(dir, A) ≈ 0 for ALL directions — but only D3′ and D1 causally shift the action margin",
    fontsize=11, y=1.01
)
fig.tight_layout()
out_file = OUT / "fig3_geometry.png"
fig.savefig(out_file, dpi=200, bbox_inches="tight")
print(f"Saved: {out_file}")
