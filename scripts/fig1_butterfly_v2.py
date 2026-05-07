#!/usr/bin/env python3
"""Figure 1 — butterfly (v3: verified data, clean rows only).

Layout: 3 columns, shared y-axis (one row per direction)
  LEFT wing  : AUROC vs evidence-sufficiency (5-fold OOF, N=486)
  CENTER     : direction names + cos·A
  RIGHT wing : |Δm|_perp  (perp-protocol only, factor=2.0, N=100)

Rows (top→bot):
  E1, E2            — pure evidence probes (regularised LR / Ridge)
  I2                — inert nuisance control (observation length)
  O1, O1⊥(amnesic)  — operative directions (OCFT + amnesic complement)
  A                 — action-axis reference
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent
OUT   = ROOT / "results/fig1_geometry"; OUT.mkdir(parents=True, exist_ok=True)

v3      = json.load(open(ROOT / "results/fig1_v3/results.json"))["directions"]
new_O   = json.load(open(ROOT / "results/fig1_new_O/results.json"))["directions"]
nullp   = json.load(open(ROOT / "results/d3_perp_vs_random_null/results.json")
                    )["random_null"]["abs_signed_mean_dm"]

# ── colours ────────────────────────────────────────────────────────────────
C_OP     = "#1F8F4E"   # operative  — green
C_INERT  = "#C0392B"   # inert      — red
C_E      = "#2F6FB5"   # pure evidence probe — dark blue
C_E2     = "#5B9BD5"   # pure evidence probe — lighter blue
C_ENAV   = "#C5622A"   # naive/contaminated evidence probe — orange
C_A      = "#555555"   # action-axis reference — grey


def _r(key, lbl1, lbl2, color, src=None):
    """Row entry.  lbl1/lbl2 = two display lines shown in the center column."""
    d = (src or v3)[key]
    return dict(lbl1=lbl1, lbl2=lbl2,
                cos_A=d["cos_A"], auroc=d["oof_auroc"],
                perp=d["abs_signed_mean_dm"],
                perp_ci=d["abs_signed_mean_dm_ci"],
                color=color, ref=False)


rows = [
    # ── evidence probes (regularised) ──────────────────────────────────
    _r("E1_LR_L20",    "LR Probe",      "evidence",          C_E),
    _r("E2_Ridge_L20", "Ridge Probe",   "evidence",          C_E2),
    # ── inert nuisance control (between evidence and operative) ─────────
    _r("I2_D4_obslen", "Obs-Length",    "Control",           C_INERT),
    # ── operative directions ────────────────────────────────────────────
    _r("O1_D3prime",   "OCFT Paired",   "operative",         C_OP),
    _r("O1_amnesic",   "Amnesic",       "Complement ⊥E1",    C_OP, src=new_O),
    # ── action-axis reference ───────────────────────────────────────────
    dict(lbl1="Action", lbl2="Direction",
         cos_A=1.000, auroc=0.589,
         perp=None, perp_ci=None,
         full=0.801, full_ci=[0.631, 0.980],
         color=C_A, ref=True),
]

# ── rows are ordered top→bot; n=6 rows, y ∈ {5..0} ──────────────────────────
# Group positions:  Evidence(5,4) | Inert(3) | Operative(2,1) | A(0)
n = len(rows); ys = np.arange(n)[::-1]

# separator y-values between groups
SEP = (3.5,   # between Evidence and Inert
       2.5,   # between Inert and Operative
       0.5)   # between Operative and A

fig = plt.figure(figsize=(15.0, 7.2))
gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 1.05, 3.4], wspace=0.04)
axL = fig.add_subplot(gs[0, 0])
axM = fig.add_subplot(gs[0, 1])
axR = fig.add_subplot(gs[0, 2])

Y_LO, Y_HI = -1.05, n - 0.4

# ── LEFT wing : AUROC (bars extend LEFTWARD) ──────────────────────────────
axL.set_xlim(1.02, 0.45)
axL.axvspan(0.50, 0.65, color="#CCCCCC", alpha=0.35, zorder=0)
axL.axvline(0.5, color="black", lw=0.6, zorder=1)
for y, r in zip(ys, rows):
    edge = "black" if r["ref"] else "white"
    hatch = "///" if r["ref"] else None
    axL.barh(y, r["auroc"] - 0.5, left=0.5, height=0.58,
             color=r["color"], alpha=0.50 if r["ref"] else 0.88,
             edgecolor=edge, lw=1.0 if r["ref"] else 0.6,
             hatch=hatch, zorder=3)
    # inverted axis: +offset shifts anchor leftward in pixels; ha="right" keeps
    # text fully left of that anchor, so numbers sit outside (left of) bar tip
    axL.text(r["auroc"] + 0.030, y, f"{r['auroc']:.3f}",
             va="center", ha="right", fontsize=10, fontweight="bold",
             color=r["color"], zorder=4)
axL.set_xticks([0.5, 0.7, 0.85, 1.0])
axL.set_xlabel("AUROC  vs  evidence-sufficiency labels   (5-fold OOF, N=486)",
               fontsize=10, labelpad=4)
axL.set_yticks([]); axL.tick_params(axis="y", length=0)
for s in ("top", "right", "left"): axL.spines[s].set_visible(False)
for y_div in SEP:
    axL.axhline(y_div, color="#888", lw=0.8, ls="--", zorder=2)
axL.set_ylim(Y_LO, Y_HI)

# ── CENTER : two-line label + cos·A ──────────────────────────────────────
axM.set_xlim(0, 1); axM.set_ylim(Y_LO, Y_HI); axM.axis("off")
axM.text(0.5, n - 0.42,
         "Direction  (cos·A ≈ 0\nat L20 decision token)",
         ha="center", va="bottom", fontsize=8.4, color="#333", style="italic",
         linespacing=1.15)
for y, r in zip(ys, rows):
    axM.text(0.5, y + 0.17, r["lbl1"], ha="center", va="center",
             fontsize=11.5, fontweight="bold", color=r["color"])
    axM.text(0.5, y - 0.05, r["lbl2"], ha="center", va="center",
             fontsize=9.5, color=r["color"])
    axM.text(0.5, y - 0.27, f"cos·A = {r['cos_A']:+.3f}",
             ha="center", va="center", fontsize=7.8, color="#666", style="italic")
for x_sep in (0.0, 1.0):
    axM.axvline(x_sep, color="black", lw=0.6, ymin=0.05, ymax=0.95)
for y_div in SEP:
    axM.axhline(y_div, color="#888", lw=0.8, ls="--")

# ── RIGHT wing : operativity  ●(perp) only ──────────────────────────────────
axR.axvspan(0, nullp["p95"], color="#CCCCCC", alpha=0.35, zorder=0)
axR.axvline(0, color="black", lw=0.6, zorder=1)
for y, r in zip(ys, rows):
    bar_w = r["perp"] if r["perp"] is not None else 0
    if bar_w:
        axR.barh(y, bar_w, height=0.62, color=r["color"], alpha=0.16,
                 edgecolor="none", zorder=1)
    if r["perp"] is not None:
        lo, hi = r["perp_ci"]
        axR.plot([lo, hi], [y, y], color=r["color"], lw=1.6, alpha=0.85, zorder=4)
        axR.plot(r["perp"], y, "o", color=r["color"], ms=11, mec="white", mew=1.0, zorder=5)
        x_lbl = max(r["perp"] + 0.04, hi + 0.02)
        axR.text(x_lbl, y, f"{r['perp']:.3f}",
                 va="center", fontsize=9.5, color=r["color"], fontweight="bold", zorder=6)
    else:
        # A row: draw hatched full-dir reference bar + label
        if r.get("full") is not None:
            full_w = r["full"]
            lo_f, hi_f = r["full_ci"]
            axR.barh(y, full_w, height=0.55, color=r["color"], alpha=0.18,
                     edgecolor=r["color"], linewidth=0.8, hatch="///", zorder=1)
            axR.plot([lo_f, hi_f], [y, y], color=r["color"], lw=1.4, alpha=0.7, zorder=4)
            axR.plot(full_w, y, "s", mfc="white", mec=r["color"], mew=1.8, ms=10, zorder=5)
            axR.text(hi_f + 0.04, y, f"full-dir = {full_w:.3f}",
                     va="center", fontsize=8.8, color=r["color"], alpha=0.90, zorder=6)
axR.set_xlim(-0.05, 2.10)
axR.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
axR.set_xlabel("|Δm|_perp   (flip ×2, projected into null(A),  N=100)",
               fontsize=10, labelpad=4)
axR.set_yticks([]); axR.tick_params(axis="y", length=0)
for s in ("top", "right", "left"): axR.spines[s].set_visible(False)
axR.set_ylim(Y_LO, Y_HI)
for y_div in SEP:
    axR.axhline(y_div, color="#888", lw=0.8, ls="--", zorder=2)

# ── right-margin brackets ─────────────────────────────────────────────────
# Layout (top→bot): E-pure(7,6) | E-naive(5,4) | Operative(3,2) | Inert(1) | A(0)
def _bracket(ax, y_top, y_bot, color, x_stem, x_tip, label_lines):
    ax.plot([x_stem, x_stem], [y_bot, y_top], color=color, lw=2.2,
            solid_capstyle="butt", zorder=6)
    for y_end in (y_top, y_bot):
        ax.plot([x_stem, x_tip], [y_end, y_end], color=color, lw=2.2,
                solid_capstyle="butt", zorder=6)
    y_mid = 0.5 * (y_top + y_bot)
    head, *tail = label_lines
    ax.text(x_stem + 0.04, y_mid + 0.18, head, color=color,
            fontsize=10.0, fontweight="bold", ha="left", va="center", zorder=7)
    if tail:
        ax.text(x_stem + 0.04, y_mid - 0.14, "\n".join(tail), color=color,
                fontsize=8.2, ha="left", va="top", linespacing=1.20, zorder=7)

_X_STEM, _X_TIP = 1.76, 1.70   # moved left so labels have room

# Evidence bracket: E1(y=5), E2(y=4)
_bracket(axR, y_top=5.32, y_bot=3.68, color=C_E, x_stem=_X_STEM, x_tip=_X_TIP,
         label_lines=["evidence",
                      "regularised LR / Ridge",
                      "high AUROC, causally inert"])

    # Inert bracket: D4(y=3) — single row
_bracket(axR, y_top=3.32, y_bot=2.68, color=C_INERT, x_stem=_X_STEM, x_tip=_X_TIP,
         label_lines=["inert control",
                          "obs-length nuisance",
                          "inside null band"])

# Operative bracket: O1(y=2), O1⊥(y=1)
_bracket(axR, y_top=2.32, y_bot=0.68, color=C_OP, x_stem=_X_STEM, x_tip=_X_TIP,
         label_lines=["operative",
                      "OCFT paired + amnesic",
                      "complement — above null"])

# Action reference bracket: A(y=0) — single row
_bracket(axR, y_top=0.32, y_bot=-0.32, color=C_A, x_stem=_X_STEM, x_tip=_X_TIP,
         label_lines=["reference",
                      "scale axis only",
                      "(perp ≡ 0 by construction)"])

# ── shared legend ─────────────────────────────────────────────────────────
all_handles = [
    Rectangle((0, 0), 1, 1, fc=C_E,     ec="white",
              label="evidence (LR / Ridge)"),
    Rectangle((0, 0), 1, 1, fc=C_INERT, ec="white",
              label="inert control"),
    Rectangle((0, 0), 1, 1, fc=C_OP,    ec="white",
              label="operative (OCFT + amnesic)"),
    Rectangle((0, 0), 1, 1, fc=C_A,     ec="black", hatch="///", alpha=0.5,
              label="action reference (search−stop, L20)"),
    Rectangle((0, 0), 1, 1, fc="#CCCCCC", alpha=0.45,
              label=f"near-chance / null  (AUROC ≤ 0.65,  p95 = {nullp['p95']:.3f})"),
]
fig.legend(handles=all_handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
           ncol=3, fontsize=8.6, frameon=True, framealpha=0.92,
           edgecolor="#ccc", columnspacing=1.4,
           title="direction class  ·  shading:", title_fontsize=8.8)

fig.suptitle(
    "Evidence–Operativity Double Dissociation at L20"
    "  (Qwen2.5-7B-Instruct, decision token)",
    fontsize=12, y=0.998)

fig.subplots_adjust(left=0.03, right=0.985, top=0.91, bottom=0.17, wspace=0.04)
out = OUT / "fig1_butterfly_v3.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
