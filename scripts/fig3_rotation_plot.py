#!/usr/bin/env python3
"""Figure 3 — angular rotation plot in null(A).

Six paths sweeping θ ∈ [0, 90°] through unit perp directions in null(A):
  IN-SUBSPACE     :  D3'→D1     (operative→operative; KILLER)
  EXIT to evidence:  D3'→E      (operative→inert)
  EXIT to length  :  D3'→D4     (operative→inert)
  EXIT to null    :  D3'→random (operative→null control)
  ENTER via D3'   :  E→D3'      (inert→operative)
  ENTER via D1    :  E→D1       (inert→operative)
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
DATA = json.load(open(ROOT / "results/fig3_geometry/fig3_data.json"))
OUT  = ROOT / "results/fig3_geometry"

paths   = DATA["rotation_sweep"]["paths"]
nullbk  = DATA["random_null_in_nullA_K20"]
A_ref   = DATA["A_positive_control"]["abs_signed_mean_dm"]
THETAS  = DATA["rotation_sweep"]["_meta"]["thetas"]

# --- visual config ---------------------------------------------------------
STYLE = {
    "D3p_to_D1":     dict(color="#1F8F4E", lw=3.2, ls="-",  marker="o",
                          ms=8, label="D3′ → D1   (operative → operative)", z=10),
    "D3p_to_E":      dict(color="#C0392B", lw=1.6, ls="-",  marker="s",
                          ms=5, label="D3′ → E    (exit to inert)",     z=5),
    "D3p_to_D4":     dict(color="#E67E22", lw=1.6, ls="-",  marker="s",
                          ms=5, label="D3′ → D4   (exit to inert)",     z=5),
    "D3p_to_random": dict(color="#7F8C8D", lw=1.6, ls="--", marker="x",
                          ms=6, label="D3′ → random  (exit to null)",   z=5),
    "E_to_D3p":      dict(color="#2F6FB5", lw=1.6, ls=":",  marker="^",
                          ms=6, label="E → D3′    (enter operative)",   z=4),
    "E_to_D1":       dict(color="#5BA3D9", lw=1.6, ls=":",  marker="^",
                          ms=6, label="E → D1     (enter operative)",   z=4),
}

fig, ax = plt.subplots(figsize=(9.6, 6.0))

# --- null band -------------------------------------------------------------
ax.axhspan(0, nullbk["p95"], color="#CCCCCC", alpha=0.30, zorder=0,
           label=f"null in null(A): K=20 random  (mean={nullbk['mean']:.3f}, p95={nullbk['p95']:.3f})")
ax.axhline(nullbk["mean"], color="#888888", lw=1.0, ls="-",  zorder=1)
ax.axhline(nullbk["p95"],  color="#888888", lw=1.0, ls=":",  zorder=1)

# --- A positive control (reference) ----------------------------------------
ax.axhline(A_ref, color="#1F8F4E", lw=1.0, ls="-.", alpha=0.7, zorder=1)
ax.text(91, A_ref, f" A axis (full-dir flip) = {A_ref:.3f}",
        va="center", ha="left", color="#1F8F4E", fontsize=8.5, alpha=0.85)

# --- curves ---------------------------------------------------------------
for name, st in STYLE.items():
    pts = paths[name]
    x   = np.array([p["theta_deg"] for p in pts])
    y   = np.array([p["abs_signed_mean_dm"] for p in pts])
    lo  = np.array([p["ci_low"]  for p in pts])
    hi  = np.array([p["ci_high"] for p in pts])
    ax.fill_between(x, lo, hi, color=st["color"], alpha=0.13, zorder=st["z"]-1)
    ax.plot(x, y, color=st["color"], lw=st["lw"], ls=st["ls"],
            marker=st["marker"], ms=st["ms"], mec="white", mew=0.6,
            label=st["label"], zorder=st["z"])

# --- annotate joint peak ---------------------------------------------------
peak = paths["D3p_to_D1"][3]   # θ=45
ax.annotate(f"Joint(D3′+D1)\n|Δm|={peak['abs_signed_mean_dm']:.3f}\n= 1.94× max endpoint\n→ subspace ≥ 2D",
            xy=(45, peak["abs_signed_mean_dm"]),
            xytext=(56, 1.78),
            fontsize=10, color="#1F8F4E", fontweight="bold",
            ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.45", fc="#EAF6EE",
                      ec="#1F8F4E", lw=1.2),
            arrowprops=dict(arrowstyle="->", color="#1F8F4E", lw=1.2))

# --- annotate D3' → D1 floor (never drops to null) ------------------------
floor = min(p["abs_signed_mean_dm"] for p in paths["D3p_to_D1"])
ax.annotate(f"min along D3′→D1 = {floor:.3f}\n(12.6× null mean; never enters null)",
            xy=(90, floor),
            xytext=(58, 0.27),
            fontsize=9, color="#1F8F4E",
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color="#1F8F4E", lw=0.9, ls="--"))

# --- axes -----------------------------------------------------------------
ax.set_xticks(THETAS)
ax.set_xlabel("rotation angle  θ  (degrees, in null(A) plane)", fontsize=11)
ax.set_ylabel("mean |signed Δ margin|  (perp flip ×2,  N=100)", fontsize=11)
ax.set_xlim(-3, 93)
ax.set_ylim(-0.04, 1.92)
ax.grid(axis="y", ls=":", alpha=0.35, zorder=0)
ax.axhline(0, color="black", lw=0.5)

ax.set_title(
    "Figure 3.  Angular sweep through null(A): operativity is a continuous function of subspace membership\n"
    "Rotating between two operative directions stays high; rotating out decays smoothly to null",
    fontsize=11)

# Legend grouped: subspace-membership first, then exits, then enters, then null
order = ["D3p_to_D1", "D3p_to_E", "D3p_to_D4", "D3p_to_random",
         "E_to_D3p", "E_to_D1"]
handles = [Line2D([0],[0], color=STYLE[k]["color"], lw=STYLE[k]["lw"],
                  ls=STYLE[k]["ls"], marker=STYLE[k]["marker"], ms=STYLE[k]["ms"],
                  label=STYLE[k]["label"]) for k in order]
handles.append(Line2D([0],[0], color="#888888", lw=8, alpha=0.30,
                      label=f"null band (0 .. p95={nullbk['p95']:.3f})"))
ax.legend(handles=handles, fontsize=8.5, loc="upper right",
          framealpha=0.95, ncol=1)

fig.tight_layout()
out_file = OUT / "fig3_rotation.png"
fig.savefig(out_file, dpi=200, bbox_inches="tight")
print(f"Saved: {out_file}")
