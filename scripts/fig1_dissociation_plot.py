#!/usr/bin/env python3
"""Figure 1 — dissociation matrix.

Three-panel horizontal layout with shared y-axis (one row per direction):
  (a) cos·A             — geometric alignment with action axis
  (b) AUROC             — probe separability for evidence sufficiency
  (c) |Δm| operativity  — causal effect on action margin (perp ● and full-dir ◻)

Demonstrates that 5 OCFT candidates + E match on (a) and (b) yet dissociate on (c).
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
ROOT = _HERE.parent
OUT  = ROOT / "results/fig1_geometry"; OUT.mkdir(parents=True, exist_ok=True)

spec   = {x["name"]: x for x in json.load(open(ROOT / "results/evidence_erasure_test/figure_spectrum.json"))["directions"] if x["type"] != "random"}
extra  = json.load(open(ROOT / "results/fig1_extra_perp/results.json"))["directions"]
d3perp = json.load(open(ROOT / "results/d3_perp_vs_random_null/results.json"))
null_perp = d3perp["random_null"]["abs_signed_mean_dm"]

# colour by operativity tier
C_OP, C_INERT, C_E, C_A, C_J = "#1F8F4E", "#C0392B", "#2F6FB5", "#444444", "#7B3294"

rows = [
    dict(name="Joint(D3′+D1)", cos_A=None, auroc=None,
         full=None, full_ci=None,
         perp=extra["joint_D3pD1_perp"]["abs_signed_mean_dm"],
         perp_ci=extra["joint_D3pD1_perp"]["abs_signed_mean_dm_ci"], color=C_J),
    dict(name="A   (action axis)", cos_A=spec["A"]["cos_with_A"], auroc=None,
         full=spec["A"]["dm_flip"], full_ci=spec["A"]["dm_flip_ci"],
         perp=None, perp_ci=None, color=C_A),
    dict(name="D3 / D3′", cos_A=spec["D3"]["cos_with_A"], auroc=spec["D3"]["auroc"],
         full=spec["D3"]["dm_flip"], full_ci=spec["D3"]["dm_flip_ci"],
         perp=d3perp["directions"]["D3prime_no_S0_perp"]["abs_signed_mean_dm"],
         perp_ci=d3perp["directions"]["D3prime_no_S0_perp"]["abs_signed_mean_dm_ci"], color=C_OP),
    dict(name="D1", cos_A=spec["D1"]["cos_with_A"], auroc=spec["D1"]["auroc"],
         full=spec["D1"]["dm_flip"], full_ci=spec["D1"]["dm_flip_ci"],
         perp=d3perp["directions"]["D1_source_perp"]["abs_signed_mean_dm"],
         perp_ci=d3perp["directions"]["D1_source_perp"]["abs_signed_mean_dm_ci"], color=C_OP),
    dict(name="D2 / D2_bal", cos_A=spec["D2"]["cos_with_A"], auroc=spec["D2"]["auroc"],
         full=spec["D2"]["dm_flip"], full_ci=spec["D2"]["dm_flip_ci"],
         perp=extra["D2bal_perp"]["abs_signed_mean_dm"],
         perp_ci=extra["D2bal_perp"]["abs_signed_mean_dm_ci"], color=C_INERT),
    dict(name="D4", cos_A=spec["D4"]["cos_with_A"], auroc=spec["D4"]["auroc"],
         full=spec["D4"]["dm_flip"], full_ci=spec["D4"]["dm_flip_ci"],
         perp=extra["D4_perp"]["abs_signed_mean_dm"],
         perp_ci=extra["D4_perp"]["abs_signed_mean_dm_ci"], color=C_INERT),
    dict(name="E   (evidence probe)", cos_A=spec["E"]["cos_with_A"], auroc=spec["E"]["auroc"],
         full=spec["E"]["dm_flip"], full_ci=spec["E"]["dm_flip_ci"],
         perp=0.0512, perp_ci=[0.0262, 0.0763], color=C_E),
]

n = len(rows); ys = np.arange(n)[::-1]
fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.4), sharey=True,
                         gridspec_kw={"width_ratios": [1.25, 1.25, 3.0], "wspace": 0.10})

# (a) cos·A
ax = axes[0]
ax.axvspan(-0.10, 0.10, color="#CCCCCC", alpha=0.30)
ax.axvline(0, color="black", lw=0.4)
for y, r in zip(ys, rows):
    if r["cos_A"] is None:
        ax.text(0.5, y, "n/a", va="center", ha="center", fontsize=8.5, style="italic", color="#999"); continue
    ax.barh(y, r["cos_A"], height=0.62, color=r["color"], alpha=0.85, edgecolor="white", lw=0.6)
    off = 0.04 * (1 if r["cos_A"] >= 0 else -1)
    ax.text(r["cos_A"] + off, y, f"{r['cos_A']:+.3f}", va="center",
            ha="left" if r["cos_A"] >= 0 else "right", fontsize=8.5)
ax.set_yticks(ys); ax.set_yticklabels([r["name"] for r in rows], fontsize=10)
ax.set_xlim(-0.25, 1.18); ax.set_xticks([-0.2, 0, 0.5, 1.0])
ax.set_xlabel("cos(direction, A)", fontsize=10)
ax.set_title("(a) Geometric alignment with A", fontsize=10.5)
ax.grid(axis="x", ls=":", alpha=0.3); ax.tick_params(axis="y", length=0)

# (b) AUROC
ax = axes[1]
ax.axvspan(0.85, 1.05, color="#D5E8D4", alpha=0.45)
ax.axvline(0.5, color="#888", lw=0.4, ls=":")
for y, r in zip(ys, rows):
    if r["auroc"] is None:
        ax.text(0.78, y, "n/a", va="center", ha="center", fontsize=8.5, style="italic", color="#999"); continue
    ax.barh(y, r["auroc"] - 0.5, left=0.5, height=0.62, color=r["color"], alpha=0.85, edgecolor="white", lw=0.6)
    ax.text(r["auroc"] + 0.005, y, f"{r['auroc']:.3f}", va="center", ha="left", fontsize=8.5)
ax.set_xlim(0.48, 1.10); ax.set_xticks([0.5, 0.7, 0.85, 1.0])
ax.set_xlabel("AUROC  (sufficient vs insufficient)", fontsize=10)
ax.set_title("(b) Evidence-probe separability", fontsize=10.5)
ax.grid(axis="x", ls=":", alpha=0.3); ax.tick_params(axis="y", length=0)

# (c) operativity
ax = axes[2]
ax.axvspan(0, null_perp["p95"], color="#F0E0E0", alpha=0.50)
ax.axvline(null_perp["p95"], color="#999", lw=0.7, ls=":")
ax.axvline(spec["A"]["dm_flip"], color=C_A, lw=0.8, ls="-.", alpha=0.55)
ax.text(spec["A"]["dm_flip"], -0.55, " A full-flip = 0.801", fontsize=8, color=C_A, alpha=0.7, ha="center")
for y, r in zip(ys, rows):
    if r["perp"] is not None:
        lo, hi = r["perp_ci"]
        ax.plot([lo, hi], [y + 0.14, y + 0.14], color=r["color"], lw=1.4, alpha=0.85)
        ax.plot(r["perp"], y + 0.14, "o", color=r["color"], ms=10, mec="white", mew=1, zorder=5)
        ax.text(r["perp"] + 0.025, y + 0.32, f"{r['perp']:.3f}", va="center",
                fontsize=8.5, color=r["color"], fontweight="bold")
    if r["full"] is not None:
        lo, hi = r["full_ci"]
        ax.plot([lo, hi], [y - 0.14, y - 0.14], color=r["color"], lw=1.0, alpha=0.50)
        ax.plot(r["full"], y - 0.14, "s", mfc="white", mec=r["color"], mew=1.5, ms=8, zorder=5)
        ax.text(r["full"] + 0.025, y - 0.32, f"{r['full']:.3f}", va="center",
                fontsize=8, color=r["color"], alpha=0.85)
# joint super-additivity
ax.annotate("super-additive\n1.33× linear sum\n(D3′+D1=1.280)",
            xy=(rows[0]["perp"], ys[0] + 0.14), xytext=(1.05, ys[0] - 0.40),
            fontsize=8.8, color=C_J, fontweight="bold", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc="#F4ECF7", ec=C_J, lw=1),
            arrowprops=dict(arrowstyle="->", color=C_J, lw=1.0))
ax.set_xlim(-0.05, 2.0); ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
ax.set_xlabel("|Δm| under projection-flip ×2  (N=100 paired)", fontsize=10)
ax.set_title("(c) Causal operativity at L20", fontsize=10.5)
ax.grid(axis="x", ls=":", alpha=0.3); ax.tick_params(axis="y", length=0)

handles = [
    Line2D([0], [0], marker="o", color="gray", lw=0, ms=10, mec="white",
           label="perp protocol  (project into null(A) then flip×2)"),
    Line2D([0], [0], marker="s", color="white", mec="gray", mew=1.5, ms=8, lw=0,
           label="full-direction flip×2  (no projection)"),
    Rectangle((0, 0), 1, 1, fc="#F0E0E0", alpha=0.55,
              label=f"perp null band  (K=20 random in null(A), p95={null_perp['p95']:.3f})"),
]
ax.legend(handles=handles, fontsize=8.2, loc="lower right", framealpha=0.95)

fig.suptitle(
    "Figure 1.  Operativity dissociates from geometric & probe alignment at L20\n"
    "Five OCFT candidates and E are matched on cos·A ≈ 0 and AUROC ≥ 0.86, yet causal operativity diverges by ≥ 16×; "
    "Joint(D3′+D1) is super-additive → the operative subspace is multi-dimensional",
    fontsize=11, y=1.02)

fig.tight_layout()
out = OUT / "fig1_dissociation.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
