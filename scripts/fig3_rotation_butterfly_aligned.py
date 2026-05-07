#!/usr/bin/env python3
"""Paper-ready rotation figure aligned with the Butterfly story.

Main claim: operativity is angularly localized inside null(A).
We intentionally exclude D1/Joint from the main figure because D1 is a
source-domain direction with signed-mean fragility; those belong in supplement.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
DATA = json.load(open(ROOT / "results/fig3_geometry/fig3_data.json"))
OUT = ROOT / "results/fig3_geometry"

PATHS = DATA["rotation_sweep"]["paths"]
NULL = DATA["random_null_in_nullA_K20"]
A_REF = DATA["A_positive_control"]["abs_signed_mean_dm"]
A_CI = [0.6312187731266021, 0.9800312682986257]
THETAS = np.array(DATA["rotation_sweep"]["_meta"]["thetas"])

STYLE = {
    "D3p_to_E":      dict(label="D3′ → E",       sub="exit to evidence",  color="#2F6FB5", lw=3.2, ls="-",  marker="o"),
    "D3p_to_D4":     dict(label="D3′ → D4",      sub="exit to nuisance",  color="#D7832F", lw=3.0, ls="-",  marker="s"),
    "D3p_to_random": dict(label="D3′ → random",  sub="exit to null",      color="#8A8A8A", lw=2.6, ls="--", marker="X"),
    "E_to_D3p":      dict(label="E → D3′",       sub="enter operative",   color="#1F8F4E", lw=3.0, ls="-.", marker="^"),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 16,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def series(name):
    pts = PATHS[name]
    return (np.array([p["theta_deg"] for p in pts]),
            np.array([p["abs_signed_mean_dm"] for p in pts]),
            np.array([p["ci_low"] for p in pts]),
            np.array([p["ci_high"] for p in pts]))


def txt(ax, x, y, s, **kw):
    t = ax.text(x, y, s, **kw)
    t.set_path_effects([pe.withStroke(linewidth=3, foreground="white", alpha=0.92)])
    return t


fig = plt.figure(figsize=(8.8, 6.6), facecolor="#FAFAF7")
ax = fig.add_axes([0.10, 0.25, 0.80, 0.56], facecolor="#FAFAF7")

# ── Title block ─────────────────────────────────────────────────────────────
fig.text(0.095, 0.94, "Angular localization of action operativity in null(A)",
         fontsize=17, fontweight="bold", color="#222222", ha="left")
fig.text(0.095, 0.902,
         "Rotating away from D3′ toward evidence, nuisance, or random directions erases action control; rotating back restores it.",
         fontsize=10.0, style="italic", color="#5B5B55", ha="left")

# ── Reference regions ──────────────────────────────────────────────────────
ax.axhspan(0, NULL["p95"], color="#DADADA", alpha=0.45, zorder=0)
ax.axhline(NULL["mean"], color="#8F8F8F", lw=1.0, ls="-", alpha=0.65, zorder=1)
ax.axhline(NULL["p95"], color="#6F6F6F", lw=1.15, ls=":", alpha=0.95, zorder=2)
ax.axhline(A_REF, color="#303030", lw=1.25, ls=(0, (6, 3)), alpha=0.70, zorder=1)
ax.fill_between([-2, 92], A_CI[0], A_CI[1], color="#303030", alpha=0.045, zorder=0)

txt(ax, 90.5, NULL["p95"] + 0.012, f"null p95 = {NULL['p95']:.3f}",
    ha="left", va="bottom", fontsize=9, color="#606060", style="italic")
txt(ax, 90.5, A_REF + 0.015, f"A full-flip reference = {A_REF:.3f}",
    ha="left", va="bottom", fontsize=9.5, color="#303030", fontweight="bold")
txt(ax, 90.5, A_REF - 0.045, "scale only · not in null(A)",
    ha="left", va="top", fontsize=8.4, color="#555555", style="italic")

# ── Curves ─────────────────────────────────────────────────────────────────
for name, st in STYLE.items():
    x, y, lo, hi = series(name)
    ax.fill_between(x, lo, hi, color=st["color"], alpha=0.12, zorder=3)
    ax.plot(x, y, color=st["color"], lw=st["lw"], ls=st["ls"], zorder=5,
            marker=st["marker"], ms=7.5, mec="white", mew=1.25,
            solid_capstyle="round", dash_capstyle="round")

# ── Endpoint callouts ──────────────────────────────────────────────────────
txt(ax, 0, 0.935, "D3′ anchor\n|Δm| = 0.875", ha="center", va="bottom",
    fontsize=10.5, color="#1F8F4E", fontweight="bold")
ax.scatter([0], [0.875], s=210, facecolor="#1F8F4E", edgecolor="white", lw=2.0, zorder=8)
ax.scatter([0], [0.875], s=520, facecolor="none", edgecolor="#1F8F4E", lw=1.6, alpha=0.18, zorder=7)

end_labels = [
    ("D3p_to_E",      "E evidence\n0.069",       90, 0.069,  0.06),
    ("D3p_to_D4",     "D4 nuisance\n0.083",       90, 0.083,  0.16),
    ("D3p_to_random", "random null\n0.013",       90, 0.013, -0.035),
]
for name, lab, x0, y0, dy in end_labels:
    c = STYLE[name]["color"]
    ax.annotate(lab, xy=(x0, y0), xytext=(78.5, y0 + dy),
                fontsize=9.5, color=c, ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=c, lw=1.0, alpha=0.92),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.0, shrinkA=2, shrinkB=4,
                                connectionstyle="arc3,rad=-0.18"), zorder=9)

ax.annotate("entry path restores\noperative control", xy=(60, 0.869), xytext=(35, 1.08),
            fontsize=10.0, color=STYLE["E_to_D3p"]["color"], fontweight="bold",
            ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.40", fc="#ECF7EF", ec="#1F8F4E", lw=1.1),
            arrowprops=dict(arrowstyle="->", color="#1F8F4E", lw=1.2,
                            connectionstyle="arc3,rad=-0.22"), zorder=10)

ax.annotate("three exits\nfall into null", xy=(76, NULL["p95"]), xytext=(49, 0.31),
            fontsize=9.6, color="#555555", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.38", fc="white", ec="#B0B0B0", lw=0.9, alpha=0.95),
            arrowprops=dict(arrowstyle="->", color="#777777", lw=1.0,
                            connectionstyle="arc3,rad=0.20"), zorder=10)

# ── Axes cosmetics ─────────────────────────────────────────────────────────
ax.set_xlim(-3, 93)
ax.set_ylim(-0.035, 1.15)
ax.set_xticks(THETAS)
ax.set_xlabel("rotation angle θ in null(A)  (degrees)")
ax.set_ylabel("action-margin shift  |Δm|$_{perp}$  (flip ×2, N=100)")
ax.grid(axis="y", color="#BDBDBD", alpha=0.24, ls="--", lw=0.8)
ax.grid(axis="x", color="#D0D0D0", alpha=0.16, ls=":", lw=0.7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#777777")
ax.spines["bottom"].set_color("#777777")

handles = [Line2D([0], [0], color=st["color"], lw=st["lw"], ls=st["ls"],
                  marker=st["marker"], ms=7, mec="white", mew=1,
                  label=f"{st['label']}  ·  {st['sub']}")
           for st in STYLE.values()]
handles += [Line2D([0], [0], color="#6F6F6F", lw=8, alpha=0.25,
                   label=f"random-null band: 0..p95={NULL['p95']:.3f}"),
            Line2D([0], [0], color="#303030", lw=1.3, ls=(0, (6, 3)),
                   label=f"A full-dir reference={A_REF:.3f}")]
leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.50, -0.16),
                ncol=2, frameon=True, framealpha=0.96, fontsize=8.2,
                columnspacing=1.1, handlelength=2.3, borderpad=0.75)
leg.get_frame().set_edgecolor("#D0D0D0")

# ── Bottom methodological note ─────────────────────────────────────────────
fig.text(0.095, 0.065,
         "All colored directions are projected into null(A); D1/Joint active-sector analysis is reserved for supplement.",
         fontsize=8.2, color="#6A6A64", style="italic", ha="left")

for ext in ("png", "pdf"):
    out = OUT / f"fig3_rotation_butterfly_aligned.{ext}"
    fig.savefig(out, dpi=300 if ext == "png" else 600)
    print(f"Saved: {out}")
