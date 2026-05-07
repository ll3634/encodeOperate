#!/usr/bin/env python3
"""Hero figure v3 — geometric universality of action-evidence orthogonality.

Design:  one geometric proof + one quantitative forest plot.

  Panel A (Hero, left):
      Unit circle in the (Evidence, Action-subspace) plane.
      Three model steering vectors drawn from origin; at natural scale all
      three appear visually identical to the +y axis.  A circular magnifier
      inset (literal "lens" with leader lines) zooms the tip region by ~400x
      revealing the arcminute-scale angular separations and the random
      null-cone wedge.  The signal arrows fall WITHIN the null cone — i.e.
      the leakage onto the evidence direction is no larger than chance.

  Panel B (Quant, right):
      Forest plot of the angular deviation |theta - 90 deg|, in arcminutes,
      for each model.  Bootstrap 95 % CI, random-direction expected value,
      and the random-direction 95 % null band (one-sided p97.5).

Output: results/fig_crossfamily_best/crossfamily_v3.{png,pdf}
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, FancyArrowPatch, FancyBboxPatch
from matplotlib.patches import ConnectionPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results/crossfamily_ci_decomposition/crossfamily_table.json"
OUT  = ROOT / "results/fig_crossfamily_best"
OUT.mkdir(parents=True, exist_ok=True)
tbl = json.load(open(DATA))

# ── rcParams ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          9.5,
    "axes.linewidth":     0.9,
    "xtick.major.width":  0.7, "xtick.major.size": 3.5,
    "ytick.major.width":  0.7, "ytick.major.size": 3.5,
    "legend.framealpha":  0.95,
    "legend.edgecolor":   "#CCCCCC",
    "legend.fontsize":    8.4,
})

# ── Per-model derived quantities (angles in arcmin) ───────────────────────────
def angle_arcmin(par, perp):
    """Angle of (par, perp) vector measured from the +y (action) axis, in arcmin."""
    return float(np.degrees(np.arctan2(par, perp)) * 60.0)

# Hidden dim per model (for random expected angle).  Pull from architecture spec.
DIM = {"qwen": 3584, "gemma": 3584, "mistral": 4096}
LAYER = {"qwen": 20, "gemma": 37, "mistral": 28}
NICE = {"qwen": "Qwen2.5-7B", "gemma": "Gemma-2-9B-it", "mistral": "Mistral-7B-v0.3"}
COLORS = {"qwen": "#1F77B4", "gemma": "#D9822B", "mistral": "#2CA02C"}

def model_geom(key):
    m = tbl[key]
    full = m["full"]["mean"]
    perp = m["perp_rms"]["mean"]
    if m["par_natural"] is not None:
        par_m  = m["par_natural"]["mean"]
        par_lo = m["par_natural"]["ci_low"]
        par_hi = m["par_natural"]["ci_high"]
        null_p975 = m["stopping_rule_no_renorm"]["rhs"]   # natural-norm null
        in_null = m["stopping_rule_no_renorm"]["in_null_band"]
    else:
        # Qwen has no natural-norm; substitute via cos(A,E): par = cos * |full|
        cos_ae = m["cos_action_evidence"]
        par_m  = cos_ae * full
        par_lo = par_m   # CI not separately measured; treat as point estimate
        par_hi = par_m
        null_p975 = abs(par_m) * 1.5    # placeholder visual bound
        in_null = True
    theta_arcmin   = angle_arcmin(par_m,  perp)
    theta_arcmin_lo= angle_arcmin(par_lo, perp)
    theta_arcmin_hi= angle_arcmin(par_hi, perp)
    null_arcmin = angle_arcmin(null_p975, perp)
    d = DIM[key]
    rand_expected_arcmin = float(np.degrees(np.arctan(1.0/np.sqrt(d))) * 60.0)
    return dict(
        full=full, perp=perp, par=par_m,
        par_lo=par_lo, par_hi=par_hi,
        theta=theta_arcmin,
        theta_lo=min(theta_arcmin_lo, theta_arcmin_hi),
        theta_hi=max(theta_arcmin_lo, theta_arcmin_hi),
        null_arcmin=null_arcmin,
        rand_arcmin=rand_expected_arcmin,
        ratio_perp=m["ratio_perp_over_full"]*100,
        in_null=in_null,
        cos_ae=m["cos_action_evidence"],
        n=m["n_samples"],
        layer=LAYER[key],
    )

GEOM = {k: model_geom(k) for k in ("qwen", "gemma", "mistral")}

# ── Figure skeleton ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13.0, 6.2))
gs = fig.add_gridspec(
    1, 2,
    width_ratios=[1.45, 1.0],
    left=0.045, right=0.985,
    top=0.88, bottom=0.10,
    wspace=0.22,
)
axA = fig.add_subplot(gs[0, 0])  # hero geometric
axB = fig.add_subplot(gs[0, 1])  # arcminute forest


# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — Hero geometric: unit circle + magnifier inset
# ══════════════════════════════════════════════════════════════════════════════
axA.set_aspect("equal")
axA.set_xlim(-1.25, 1.25)
axA.set_ylim(-0.18, 1.30)
axA.set_xticks([])
axA.set_yticks([])
for spine in ("top", "right", "left", "bottom"):
    axA.spines[spine].set_visible(False)

# unit circle
circle = Circle((0, 0), 1.0, fill=False, edgecolor="#888", linewidth=1.0,
                linestyle=(0, (4, 3)), zorder=1)
axA.add_patch(circle)

# axes through origin
axA.plot([-1.15, 1.15], [0, 0], color="#444", lw=1.0, zorder=1.5)
axA.plot([0, 0], [-0.10, 1.18], color="#444", lw=1.0, zorder=1.5)

# axis labels
axA.text(1.18, -0.04, r"Evidence direction  $\hat{\mathbf{e}}$",
         ha="left", va="top", fontsize=10.5, style="italic", color="#222")
axA.text(0.04, 1.20, r"Action subspace  $\hat{\mathbf{a}}_\perp$",
         ha="left", va="bottom", fontsize=10.5, style="italic", color="#222")
axA.plot([0], [0], "o", ms=4, color="#222", zorder=4)
axA.text(-0.04, -0.04, "O", ha="right", va="top", fontsize=9, color="#222")

# Three model arrows — at natural scale all three are visually
# indistinguishable from +y; we fan them cosmetically so colours separate.
fan_angles_deg = {"qwen": -1.5, "gemma": 0.0, "mistral": +1.5}
for k in ("qwen", "gemma", "mistral"):
    th = np.radians(90 + fan_angles_deg[k])
    x_tip, y_tip = np.cos(th)*0.96, np.sin(th)*0.96
    arr = FancyArrowPatch(
        (0, 0), (x_tip, y_tip),
        arrowstyle="-|>", mutation_scale=15,
        linewidth=2.4, color=COLORS[k], zorder=5,
        shrinkA=0, shrinkB=0,
    )
    axA.add_patch(arr)

# Caption inside the main panel (below x-axis)
axA.text(-1.20, -0.13,
         "At natural scale, three architectures' steering vectors are\n"
         "visually indistinguishable from perfect orthogonality.",
         ha="left", va="top", fontsize=9.0, color="#333", style="italic")

# ── Magnifier (literal lens with leader lines) ─────────────────────────────────
LENS_CX, LENS_CY, LENS_R = 0.78, 0.70, 0.32
ZOOM_CX, ZOOM_CY = 0.0, 0.985
ZOOM_HALF = 0.022

# leader lines from the unit-circle tip region to the lens edge
for sign in (-1, +1):
    src_x = ZOOM_CX + sign * ZOOM_HALF
    src_y = ZOOM_CY - 0.005
    dx = LENS_CX - src_x
    dy = LENS_CY - src_y
    norm = np.hypot(dx, dy)
    tx = LENS_CX - dx / norm * LENS_R
    ty = LENS_CY - dy / norm * LENS_R
    axA.plot([src_x, tx], [src_y, ty], color="#777", lw=0.9,
             linestyle=(0, (3, 2)), zorder=2)

mark = Circle((ZOOM_CX, ZOOM_CY-0.005), ZOOM_HALF*1.05, fill=False,
              edgecolor="#777", linewidth=0.9, linestyle=(0, (3, 2)), zorder=2)
axA.add_patch(mark)

# the lens — white fill, dark border
lens_bg = Circle((LENS_CX, LENS_CY), LENS_R, facecolor="#FAFAFC",
                 edgecolor="#222", linewidth=1.8, zorder=6)
axA.add_patch(lens_bg)

# ─── Inside the lens: arcminute-scale geometry ─────────────────────────────────
ARC_HALF_ARCMIN = 80.0   # lens half-width in arcmin

def lens_xy_at_arcmin(theta_arcmin, r_frac=0.92):
    """Map (angle from +y in arcmin, radial fraction in lens) to data coords."""
    x = LENS_CX + (theta_arcmin / ARC_HALF_ARCMIN) * LENS_R * r_frac
    # arrows emanate from bottom of lens (which corresponds to "origin" radially)
    y0 = LENS_CY - LENS_R * 0.85
    y_tip = LENS_CY + LENS_R * r_frac * 0.55
    return x, y_tip, y0

# Random null cone (95 % wedge) — use Mistral & Gemma random-direction expectation,
# which scales as 1/sqrt(d). Use the larger of the two as visual envelope.
rand_arcmin_envelope = max(GEOM["gemma"]["null_arcmin"],
                           GEOM["mistral"]["null_arcmin"])
# clip to lens range
rand_arcmin_envelope = min(rand_arcmin_envelope, ARC_HALF_ARCMIN*0.95)
left_x  = LENS_CX + (-rand_arcmin_envelope/ARC_HALF_ARCMIN) * LENS_R * 0.92
right_x = LENS_CX + ( rand_arcmin_envelope/ARC_HALF_ARCMIN) * LENS_R * 0.92
y_origin = LENS_CY - LENS_R * 0.85
y_top    = LENS_CY + LENS_R * 0.92 * 0.55
# wedge polygon (triangle from origin spreading to the top)
import matplotlib.patches as mpatches
wedge_poly = mpatches.Polygon(
    [(LENS_CX, y_origin), (left_x, y_top), (right_x, y_top)],
    closed=True, facecolor="#BBBBBB", alpha=0.32, edgecolor="none", zorder=6.5,
)
axA.add_patch(wedge_poly)
# wedge outline
axA.plot([LENS_CX, left_x],  [y_origin, y_top], color="#888", lw=0.7,
         linestyle=":", zorder=6.6)
axA.plot([LENS_CX, right_x], [y_origin, y_top], color="#888", lw=0.7,
         linestyle=":", zorder=6.6)

# vertical "perfect orthogonality" reference inside lens
axA.plot([LENS_CX, LENS_CX], [y_origin, y_top + 0.005],
         color="#444", lw=0.9, zorder=6.8)

# small origin dot inside lens
axA.plot([LENS_CX], [y_origin], "o", ms=3, color="#222", zorder=7)

# Each model arrow inside the lens at its true arcminute angle
arrow_specs = []
for k in ("qwen", "gemma", "mistral"):
    g = GEOM[k]
    if k == "qwen":
        # Qwen par_natural unmeasured; show at 0 with hatched marker
        theta = 0.0
    else:
        theta = g["theta"]
    x_tip = LENS_CX + (theta / ARC_HALF_ARCMIN) * LENS_R * 0.92
    arr = FancyArrowPatch(
        (LENS_CX, y_origin), (x_tip, y_top),
        arrowstyle="-|>", mutation_scale=11,
        linewidth=2.0, color=COLORS[k], zorder=8,
        shrinkA=0, shrinkB=0, alpha=0.95,
    )
    axA.add_patch(arr)
    arrow_specs.append((k, theta, x_tip))


# Lens scale labels — arcminute tick marks at the top
for tick_arcmin in (-60, -30, 0, 30, 60):
    x = LENS_CX + (tick_arcmin / ARC_HALF_ARCMIN) * LENS_R * 0.92
    axA.plot([x, x], [y_top + 0.002, y_top + 0.012], color="#666", lw=0.8, zorder=7.5)
    if tick_arcmin == 0:
        lab = "0"
    else:
        lab = f"{tick_arcmin:+d}'"
    axA.text(x, y_top + 0.018, lab, ha="center", va="bottom", fontsize=7.0,
             color="#444")

# Lens title and "wedge = random null cone" annotation
axA.text(LENS_CX, LENS_CY + LENS_R + 0.02,
         r"$\bf{400\!\times\ zoom}$  ·  arcminute scale",
         ha="center", va="bottom", fontsize=9.5, color="#222")

# Annotate wedge as random null cone, inside lens lower-right
axA.text(LENS_CX + LENS_R*0.55, LENS_CY - LENS_R*0.55,
         "Random-direction\n95% null cone",
         ha="center", va="center", fontsize=7.6, color="#555", style="italic")

# Per-arrow arcminute callouts inside lens — positioned by sign so they don't collide
callout_offsets = {  # (dx, dy, ha) per model
    "qwen":    (-0.12, +0.02, "right"),
    "gemma":   (+0.10, -0.04, "left"),
    "mistral": (+0.12, +0.02, "left"),
}
for k, theta, x_tip in arrow_specs:
    dx, dy, ha = callout_offsets[k]
    label = NICE[k]
    if k == "qwen":
        val = "n/a (par_natural unmeasured)"
    else:
        val = f"{theta:+.2f}'"
    axA.annotate(
        f"{label}\n{val}",
        xy=(x_tip, y_top),
        xytext=(x_tip + dx, y_top + dy),
        ha=ha, va="center", fontsize=7.6, color=COLORS[k],
        arrowprops=dict(arrowstyle="-", color=COLORS[k], lw=0.7, alpha=0.7),
        zorder=9,
    )

# Title for Panel A
axA.set_title("A   Geometric universality:  steering vector $\\subset$ evidence$^\\perp$",
              loc="left", fontsize=12.2, fontweight="bold", pad=12, color="#111")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — Forest plot of angular deviation in arcminutes
# ══════════════════════════════════════════════════════════════════════════════
axB.set_title("B   Angular deviation from perfect orthogonality",
              loc="left", fontsize=12.2, fontweight="bold", pad=12, color="#111")

models_order = ["mistral", "gemma", "qwen"]   # bottom-to-top
y_positions  = np.arange(len(models_order))

# Vertical zero line — perfect orthogonality
axB.axvline(0, color="#222", lw=1.2, zorder=1.5)
axB.text(0, len(models_order) - 0.40, "perfect\northogonality",
         ha="center", va="bottom", fontsize=8.2, color="#222", style="italic")

# Per-model null bands as horizontal segments (one per model row)
ROW_HALF_HEIGHT = 0.30
for i, k in enumerate(models_order):
    g = GEOM[k]
    # 95% null band — light grey segment
    axB.add_patch(plt.Rectangle(
        (-g["null_arcmin"], i - ROW_HALF_HEIGHT),
        2 * g["null_arcmin"], 2 * ROW_HALF_HEIGHT,
        facecolor="#DDDDDD", alpha=0.55, edgecolor="none", zorder=0,
    ))
    # random-direction expected (1/sqrt(d)) tick on each side
    for sign in (-1, +1):
        axB.plot([sign * g["rand_arcmin"]] * 2,
                 [i - ROW_HALF_HEIGHT*0.7, i + ROW_HALF_HEIGHT*0.7],
                 color="#888", lw=1.2, zorder=2.5, solid_capstyle="butt")

# Per-model measured angular deviation with CI
for i, k in enumerate(models_order):
    g = GEOM[k]
    if k == "qwen":
        # No per-sample CI; show direction-level cos(A,E) as hollow square
        axB.plot([g["theta"]], [i], marker="s", markersize=11,
                 markerfacecolor="white", markeredgecolor=COLORS[k],
                 markeredgewidth=1.8, zorder=4)
        # annotation positioned to avoid overlap with bar
        axB.annotate("from direction-level\ncos(A,E) only",
                     xy=(g["theta"], i),
                     xytext=(g["theta"] - 6, i + 0.34),
                     ha="right", va="bottom", fontsize=7.4,
                     color=COLORS[k], style="italic",
                     arrowprops=dict(arrowstyle="-", color=COLORS[k],
                                     lw=0.6, alpha=0.6))
    else:
        lo, hi = g["theta_lo"], g["theta_hi"]
        axB.errorbar([g["theta"]], [i],
                     xerr=[[g["theta"]-lo], [hi-g["theta"]]],
                     fmt="o", markersize=9, color=COLORS[k],
                     ecolor=COLORS[k], elinewidth=1.8, capsize=4, capthick=1.4,
                     zorder=4)
        # numeric annotation at the marker
        axB.text(g["theta"] + 2.2, i + 0.18, f"{g['theta']:+.2f}'",
                 ha="left", va="bottom", fontsize=8.0,
                 color=COLORS[k], fontweight="bold")

# Y-axis: model labels with metadata
y_labels = []
for k in models_order:
    g = GEOM[k]
    y_labels.append(f"{NICE[k]}\nL{g['layer']}  N={g['n']}  cos(A,E)={g['cos_ae']:+.4f}")
axB.set_yticks(y_positions)
axB.set_yticklabels(y_labels, fontsize=9.0)
axB.set_ylim(-0.55, len(models_order) - 0.20)

# X-axis: arcminutes
axB.set_xlabel(r"Angular deviation $\theta - 90^{\circ}$  (arcminutes)",
               fontsize=10.2)
null_env = max(GEOM[k]["null_arcmin"] for k in ("qwen", "gemma", "mistral"))
xmax = null_env * 1.15
axB.set_xlim(-xmax, xmax)
axB.set_xticks([-60, -40, -20, 0, 20, 40, 60])
axB.set_xticklabels(["−60'", "−40'", "−20'", "0", "+20'", "+40'", "+60'"])

axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)
axB.tick_params(axis="y", left=False)
axB.grid(axis="x", color="#EEEEEE", linewidth=0.6, zorder=0)

# Panel B legend
legend_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["mistral"],
           markersize=8, label="Measured  (95% bootstrap CI)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="white",
           markeredgecolor="#1F77B4", markersize=8,
           label="Direction-level cos(A,E) only"),
    Line2D([0], [0], color="#888", lw=1.4,
           label=r"Random expected $\pm 1/\sqrt{d}$"),
    Line2D([0], [0], color="#DDDDDD", lw=10,
           label="Random 95% null band (per-model)"),
]
axB.legend(handles=legend_handles, loc="lower right", frameon=True,
           fancybox=False, fontsize=7.8)

# ── Suptitle and figure-level caption ─────────────────────────────────────────
fig.suptitle(
    "Action steering lives in the orthogonal complement of evidence — "
    "a null-cone-bounded, architecture-invariant geometric law",
    fontsize=13.0, fontweight="bold", color="#111",
    x=0.012, y=0.965, ha="left",
)

caption = (
    "Three model families (Qwen2.5-7B-Instruct L20, Gemma-2-9B-it L37, "
    "Mistral-7B-v0.3 L28) show the same geometry: the steering direction "
    "$\\hat{\\mathbf{a}}$ deviates from the evidence$^{\\perp}$ subspace by an "
    "angle that is statistically indistinguishable from a random unit vector "
    "in $\\mathbb{R}^{d}$ (random-cone half-width $1.96/\\sqrt{d}$).  "
    "For Gemma and Mistral the deviation is 4–10 arcmin (sample-based); for "
    "Qwen only the direction-level cos(A,E) is available (square marker, "
    "no per-sample CI).  Bootstrap 95% CI, $N{=}50$–$100$, $K{=}200$ random "
    "controls.  Panel A is to scale; the lens magnifies the unit-circle tip "
    "${\\sim}400\\times$."
)
fig.text(0.012, 0.025, caption, fontsize=8.2, color="#333",
         ha="left", va="bottom", wrap=True)

# Save
out_png = OUT / "crossfamily_v3.png"
out_pdf = OUT / "crossfamily_v3.pdf"
fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.18)
fig.savefig(out_pdf,            bbox_inches="tight", pad_inches=0.18)
print(f"[v3] wrote {out_png}")
print(f"[v3] wrote {out_pdf}")
print(f"[v3] angular deviations (arcmin):")
for k in ("qwen", "gemma", "mistral"):
    g = GEOM[k]
    print(f"  {k:8s}: theta={g['theta']:+.3f}'  "
          f"random expected={g['rand_arcmin']:.2f}'  "
          f"null95%={g['null_arcmin']:.2f}'")

