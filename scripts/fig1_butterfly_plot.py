#!/usr/bin/env python3
"""Figure 1 — butterfly (diverging-bar) layout for evidence ↔ operativity dissociation.

Center axis = direction names with cos·A annotation.
Left wing  = AUROC against evidence-sufficiency labels (phase1, N=486).
Right wing = |Δm| under perp flip×2 in null(A) (N=100 paired).

Each row has a bar on BOTH sides (no n/a). A is shown as a right-wing reference
line only (perp(A) = 0 by construction).
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent
OUT   = ROOT / "results/fig1_geometry"; OUT.mkdir(parents=True, exist_ok=True)

extra  = json.load(open(ROOT / "results/fig1_extra_perp/results.json"))["directions"]
d3perp = json.load(open(ROOT / "results/d3_perp_vs_random_null/results.json"))
nullp  = d3perp["random_null"]["abs_signed_mean_dm"]
A_full_flip = 0.8012

C_OP, C_INERT, C_E, C_J = "#1F8F4E", "#C0392B", "#2F6FB5", "#7B3294"

# rows ordered by |Δm|_perp descending; AUROC computed on phase1 cohort (N=486)
rows = [
    dict(name="Joint(D3′+D1)", cos_A=0.000, auroc=0.548,
         perp=extra["joint_D3pD1_perp"]["abs_signed_mean_dm"],
         perp_ci=extra["joint_D3pD1_perp"]["abs_signed_mean_dm_ci"], color=C_J),
    dict(name="D3′", cos_A=-0.038, auroc=0.511,
         perp=d3perp["directions"]["D3prime_no_S0_perp"]["abs_signed_mean_dm"],
         perp_ci=d3perp["directions"]["D3prime_no_S0_perp"]["abs_signed_mean_dm_ci"], color=C_OP),
    dict(name="D1", cos_A=+0.006, auroc=0.571,
         perp=d3perp["directions"]["D1_source_perp"]["abs_signed_mean_dm"],
         perp_ci=d3perp["directions"]["D1_source_perp"]["abs_signed_mean_dm_ci"], color=C_OP),
    dict(name="D2_bal", cos_A=+0.006, auroc=0.519,
         perp=extra["D2bal_perp"]["abs_signed_mean_dm"],
         perp_ci=extra["D2bal_perp"]["abs_signed_mean_dm_ci"], color=C_INERT),
    dict(name="D4", cos_A=-0.100, auroc=0.525,
         perp=extra["D4_perp"]["abs_signed_mean_dm"],
         perp_ci=extra["D4_perp"]["abs_signed_mean_dm_ci"], color=C_INERT),
    dict(name="E   (evidence probe)", cos_A=-0.014, auroc=0.862,
         perp=0.0512, perp_ci=[0.0262, 0.0763], color=C_E),
]

n = len(rows); ys = np.arange(n)[::-1]
fig = plt.figure(figsize=(13.6, 5.6))
gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 1.0, 3.0], wspace=0.05)
axL = fig.add_subplot(gs[0, 0]); axM = fig.add_subplot(gs[0, 1]); axR = fig.add_subplot(gs[0, 2])

# ─── LEFT wing: AUROC (extending leftward from center 0.5) ──────────────────
axL.axvspan(0.85, 1.05, color="#D5E8D4", alpha=0.55, zorder=0,
            label="strong evidence prediction (AUROC ≥ 0.85)")
axL.axvline(0.5, color="black", lw=0.6, zorder=1)
for y, r in zip(ys, rows):
    axL.barh(y, r["auroc"] - 0.5, left=0.5, height=0.62,
             color=r["color"], alpha=0.88, edgecolor="white", lw=0.6, zorder=3)
    axL.text(r["auroc"] + 0.012, y, f"{r['auroc']:.3f}", va="center", ha="left",
             fontsize=9.5, fontweight="bold", color=r["color"], zorder=4)
axL.set_xlim(1.05, 0.45); axL.invert_xaxis()
axL.set_xticks([0.5, 0.7, 0.85, 1.0])
axL.set_yticks(ys); axL.set_yticklabels([])
axL.tick_params(axis="y", length=0)
axL.set_xlabel("AUROC  vs  evidence-sufficiency labels  (N=486 phase-1 cohort)",
               fontsize=10)
axL.set_title("← evidence-related descriptor", fontsize=10.5, fontweight="bold", color="#444")
axL.grid(axis="x", ls=":", alpha=0.3)
for s in ("top", "right", "left"): axL.spines[s].set_visible(False)

# ─── CENTER: direction names + cos·A annotation ─────────────────────────────
axM.set_xlim(0, 1); axM.set_ylim(-0.6, n - 0.4); axM.axis("off")
for y, r in zip(ys, rows):
    axM.text(0.5, y + 0.12, r["name"], ha="center", va="center",
             fontsize=11.5, fontweight="bold", color=r["color"])
    axM.text(0.5, y - 0.22, f"cos·A = {r['cos_A']:+.3f}",
             ha="center", va="center", fontsize=8.2, color="#666", style="italic")
# vertical separator
for x_sep in (0.0, 1.0):
    axM.axvline(x_sep, color="black", lw=0.6, ymin=0.05, ymax=0.95)

# ─── RIGHT wing: |Δm|_perp (extending rightward from 0) ─────────────────────
axR.axvspan(0, nullp["p95"], color="#F0E0E0", alpha=0.55, zorder=0,
            label=f"perp null  (K=20 random in null(A), p95={nullp['p95']:.3f})")
axR.axvline(0, color="black", lw=0.6, zorder=1)
axR.axvline(A_full_flip, color="#444444", lw=0.9, ls="-.", alpha=0.65, zorder=2)
axR.text(A_full_flip, n - 0.30, " A full-flip\n benchmark = 0.801",
         fontsize=8, color="#444", alpha=0.8, ha="left", va="top")
for y, r in zip(ys, rows):
    lo, hi = r["perp_ci"]
    axR.barh(y, r["perp"], height=0.62, color=r["color"], alpha=0.88,
             edgecolor="white", lw=0.6, zorder=3)
    axR.errorbar(r["perp"], y, xerr=[[r["perp"] - lo], [hi - r["perp"]]],
                 fmt="none", ecolor="black", elinewidth=1.1, capsize=3, zorder=4)
    axR.text(r["perp"] + 0.035, y, f"{r['perp']:.3f}", va="center",
             fontsize=9.5, fontweight="bold", color=r["color"], zorder=5)
# super-additivity callout for joint
axR.annotate("super-additive: 1.33× linear sum\n(D3′ + D1 = 1.280)",
             xy=(rows[0]["perp"], ys[0]), xytext=(0.92, ys[0] - 0.55),
             fontsize=8.6, color=C_J, fontweight="bold", ha="left", va="center",
             bbox=dict(boxstyle="round,pad=0.35", fc="#F4ECF7", ec=C_J, lw=1),
             arrowprops=dict(arrowstyle="->", color=C_J, lw=1.0))
axR.set_xlim(-0.05, 2.05); axR.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
axR.set_yticks(ys); axR.set_yticklabels([])
axR.tick_params(axis="y", length=0)
axR.set_xlabel("|Δm| under perp-flip ×2 in null(A)  (N=100 paired)", fontsize=10)
axR.set_title("operative on action margin →", fontsize=10.5, fontweight="bold", color="#444")
axR.grid(axis="x", ls=":", alpha=0.3)
for s in ("top", "right", "left"): axR.spines[s].set_visible(False)

# ─── shared legend ─────────────────────────────────────────────────────────
handles = [
    Rectangle((0, 0), 1, 1, fc="#D5E8D4", alpha=0.55,
              label="strong evidence prediction (AUROC ≥ 0.85)"),
    Rectangle((0, 0), 1, 1, fc="#F0E0E0", alpha=0.55,
              label=f"perp null band (p95 = {nullp['p95']:.3f})"),
    Line2D([0], [0], color="#444444", lw=1.2, ls="-.",
           label=f"A full-flip benchmark = {A_full_flip:.3f}"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, -0.02), frameon=False)

fig.suptitle(
    "Figure 1.  Evidence ↔ operativity dissociation at L20 (decision token, Qwen2.5-7B-Instruct)\n"
    "E is the only strong evidence-sufficiency predictor (left), yet has near-zero causal effect on the action margin (right);\n"
    "D3′ and D1 are the strong operators (right) but predict evidence sufficiency at chance (left).  Same matched cos·A ≈ 0.",
    fontsize=11, y=1.04)

fig.tight_layout(rect=(0, 0.02, 1, 0.98))
out = OUT / "fig1_butterfly.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
