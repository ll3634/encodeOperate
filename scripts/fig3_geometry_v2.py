#!/usr/bin/env python3
"""Figure 3 v2 — Operative-subspace landscape at L20.

Single 3D panel:
  XY plane = PCA embedding of directions in null(A)
  Z        = |signed mean Δ margin| under perp-protocol (flip x2)
  Stems    = lollipops from floor to marker (height = causal effect)
  Two faint horizontal reference planes mark random-null mean and p95.

A single visual element (3D position) encodes both geometry (XY)
and operativity (Z). No bar-chart side panel.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path("tmc/scripts/e2e_agent")
DATA = json.load(open(ROOT / "results/fig3_geometry/fig3_data.json"))
OUT  = ROOT / "results/fig3_geometry/fig3_geometry_v2.png"

NAMED = {
    "D3p":   ("D3'",   "#1F4E79", "operative"),
    "D1":    ("D1",    "#2F8FD9", "operative"),
    "joint": ("D3'+D1\n(joint)", "#27AE60", "joint"),
    "D4":    ("D4",    "#E67E22", "inert"),
    "D2bal": ("D2",    "#8E44AD", "inert"),
    "E":     ("E",     "#C0392B", "inert"),
}
PERP = DATA["perp_protocol_results"]
DM = {
    "D3p":   PERP["D3p_perp"]["abs_signed_mean_dm"],
    "D1":    PERP["D1_perp"]["abs_signed_mean_dm"],
    "joint": PERP["joint_D3pD1_perp"]["abs_signed_mean_dm"],
    "D4":    PERP["D4_perp"]["abs_signed_mean_dm"],
    "D2bal": PERP["D2_perp"]["abs_signed_mean_dm"],
    "E":     DATA["E_perp_proxy"]["abs_signed_mean_dm"],
}
COORDS = DATA["pca_embedding_2D"]["named_directions"]
RANDOM = DATA["pca_embedding_2D"]["random_nulls"]
NULL = DATA["random_null_in_nullA_K20"]
NULL_MEAN, NULL_P95 = NULL["mean"], NULL["p95"]

# Random null per-direction |Δm| values (matched index to coords)
NULL_VALS = NULL["individual_values"]

# ── Figure ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11.5, 8.0))
ax = fig.add_subplot(111, projection="3d")

# ---- Reference planes (null mean & p95) ----
xlim = (-0.55, 0.95); ylim = (-0.55, 0.85)
xx, yy = np.meshgrid([xlim[0], xlim[1]], [ylim[0], ylim[1]])
for z, color, alpha, label in [
    (NULL_MEAN, "#888888", 0.10, f"null mean = {NULL_MEAN:.3f}"),
    (NULL_P95,  "#666666", 0.07, f"null p95 = {NULL_P95:.3f}"),
]:
    surf = Poly3DCollection(
        [[(xlim[0], ylim[0], z), (xlim[1], ylim[0], z),
          (xlim[1], ylim[1], z), (xlim[0], ylim[1], z)]],
        facecolor=color, alpha=alpha, edgecolor="none")
    ax.add_collection3d(surf)
    ax.text(xlim[1], ylim[1], z, f"  {label}", fontsize=7.5,
            color="#444444", ha="left", va="bottom")

# ---- Random null lollipops ----
for k, (key, (x, y)) in enumerate(RANDOM.items()):
    z = NULL_VALS[k]
    ax.plot([x, x], [y, y], [0, z], color="#BBBBBB", lw=0.7, alpha=0.7, zorder=1)
    ax.scatter([x], [y], [z], s=18, c="#999999", alpha=0.7,
               edgecolor="none", zorder=2)

# ---- Named directions ----
for key, (label, color, role) in NAMED.items():
    x, y = COORDS[key]
    z = DM[key]
    # Stem
    lw = 3.0 if role in ("operative", "joint") else 1.6
    ls = "--" if role == "joint" else "-"
    ax.plot([x, x], [y, y], [0, z], color=color, lw=lw, ls=ls, zorder=4)
    # Marker
    ms = 220 if role == "joint" else (170 if role == "operative" else 110)
    ax.scatter([x], [y], [z], s=ms, c=color, edgecolor="black",
               linewidth=0.8, depthshade=False, zorder=5)
    # Floor footprint
    ax.scatter([x], [y], [0], s=50, c=color, marker="o",
               edgecolor="black", linewidth=0.4, alpha=0.45, zorder=3)
    # Label slightly above marker
    z_off = z * 0.04 + 0.07
    ax.text(x, y, z + z_off, f"{label}\n|Δm|={z:.3f}",
            fontsize=9, color=color, fontweight="bold",
            ha="center", va="bottom", zorder=6)

# ---- Origin & axes annotation ----
ax.scatter([0], [0], [0], s=40, c="black", marker="x", zorder=4)
ax.text(0, 0, -0.08, "origin\n(in null(A))", fontsize=7.5,
        color="0.3", ha="center", va="top")

# A-axis reference (out-of-plane, conceptual)
A_ref = DATA["A_positive_control"]["abs_signed_mean_dm"]
ax.plot([0, 0], [0, 0], [0, A_ref + 0.05], color="#2E8B57",
        lw=2.2, ls=":", zorder=4)
ax.scatter([0], [0], [A_ref], s=140, c="#2E8B57", marker="^",
           edgecolor="black", linewidth=0.8, depthshade=False, zorder=5)
ax.text(0, 0, A_ref + 0.10,
        f"A (positive control,\nfull-dir flip on action axis)\n|Δm|={A_ref:.3f}",
        fontsize=8.5, color="#2E8B57", fontweight="bold",
        ha="center", va="bottom", zorder=6)

# ---- Joint super-additivity annotation: dashed line from D3p → joint and D1 → joint ----
for src in ("D3p", "D1"):
    sx, sy = COORDS[src]; sz = DM[src]
    jx, jy = COORDS["joint"]; jz = DM["joint"]
    ax.plot([sx, jx], [sy, jy], [sz, jz],
            color="#27AE60", lw=0.8, ls=":", alpha=0.6, zorder=2)

# ---- Axes cosmetics ----
ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(0, 1.95)
ax.set_xlabel("PC1 of null(A)  (7.5% var)", fontsize=10, labelpad=6)
ax.set_ylabel("PC2 of null(A)  (4.6% var)", fontsize=10, labelpad=6)
ax.set_zlabel("|signed mean Δ margin|  (perp-flip ×2, N=100)", fontsize=10, labelpad=6)
ax.view_init(elev=22, azim=-58)
ax.xaxis.pane.set_alpha(0.04); ax.yaxis.pane.set_alpha(0.04)
ax.zaxis.pane.set_alpha(0.04)
ax.grid(True, alpha=0.25)

ax.set_title(
    "L20 operative-subspace landscape\n"
    "All named directions have cos(·,A)≈0 (sit in null(A)) — "
    "yet causal effect varies 40× by position",
    fontsize=11, pad=12)

# ---- Caption / legend text in figure ----
caption = (
    "• Floor: PCA plane of null(A). 20 grey dots = random-null directions in null(A).\n"
    "• Stem height = |signed mean Δ margin| under the same perp-protocol flip ×2 (N=100).\n"
    "• Two faint horizontal planes = K=20 random-null mean (0.043) and p95 (0.123).\n"
    "• Green ▲ at origin = A positive control (full-dir flip on action axis itself, 0.801) for height-scale reference.\n"
    "• Joint = (D3'+D1)/‖·‖ in null(A): observed 1.70 > linear sum 1.28 → operative subspace ≥ 2D."
)
fig.text(0.02, 0.02, caption, fontsize=8.0, color="0.20",
         family="DejaVu Sans", linespacing=1.4)

fig.tight_layout(rect=[0, 0.10, 1, 1])
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Saved: {OUT}")
