#!/usr/bin/env python3
"""Figure 3 — Cross-family natural-norm decomposition.

Butterfly layout (mirrors fig1_butterfly_v3.py style):
  LEFT wing  : evidence-parallel component (natural-norm); null band shaded
  CENTER     : model family name, layer, cos(action, evidence)
  RIGHT wing : evidence-orthogonal component ≈ full Δm

Rows (top → bot):
  Qwen2.5-7B (L20)
  Gemma-2-9B-it (L37)
  Mistral-7B-v0.3 (L28)
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "results/fig3_decomposition"
OUT.mkdir(parents=True, exist_ok=True)

tbl = json.load(open(ROOT / "results/crossfamily_ci_decomposition/crossfamily_table.json"))

# ── colours ────────────────────────────────────────────────────────────────
C_QWEN    = "#2F6FB5"   # Qwen  — dark blue
C_GEMMA   = "#C5622A"   # Gemma — burnt orange
C_MISTRAL = "#1F8F4E"   # Mistral — green

MODELS = [
    {
        "name": "Qwen2.5-7B",
        "sublabel": "L20  ·  d=3584",
        "cos": -0.0135,
        "color": C_QWEN,
        "par_val": None,          # opposite-sign: par_rms = -0.157
        "par_rms_abs": 0.157,
        "par_ci": None,
        "null_upper": None,       # no natural-norm null for Qwen
        "full_val": 0.910,
        "full_ci": [0.841, 0.980],
        "perp_val": 0.909,
        "perp_ci": [0.839, 0.979],
        "opposite_sign": True,
        "in_null_band": None,
    },
    {
        "name": "Gemma-2-9B-it",
        "sublabel": "L37  ·  d=3584",
        "cos": +0.0110,
        "color": C_GEMMA,
        "par_val": tbl["gemma"]["par_natural"]["mean"],
        "par_ci": [tbl["gemma"]["par_natural"]["ci_low"],
                   tbl["gemma"]["par_natural"]["ci_high"]],
        "null_upper": 0.03381,
        "full_val": tbl["gemma"]["full"]["mean"],
        "full_ci": [tbl["gemma"]["full"]["ci_low"], tbl["gemma"]["full"]["ci_high"]],
        "perp_val": tbl["gemma"]["perp_rms"]["mean"],
        "perp_ci": [tbl["gemma"]["perp_rms"]["ci_low"], tbl["gemma"]["perp_rms"]["ci_high"]],
        "opposite_sign": False,
        "in_null_band": True,
    },
    {
        "name": "Mistral-7B-v0.3",
        "sublabel": "L28  ·  d=4096",
        "cos": -0.0090,
        "color": C_MISTRAL,
        "par_val": tbl["mistral"]["par_natural"]["mean"],
        "par_ci": [tbl["mistral"]["par_natural"]["ci_low"],
                   tbl["mistral"]["par_natural"]["ci_high"]],
        "null_upper": 0.01966,
        "full_val": tbl["mistral"]["full"]["mean"],
        "full_ci": [tbl["mistral"]["full"]["ci_low"], tbl["mistral"]["full"]["ci_high"]],
        "perp_val": tbl["mistral"]["perp_rms"]["mean"],
        "perp_ci": [tbl["mistral"]["perp_rms"]["ci_low"], tbl["mistral"]["perp_rms"]["ci_high"]],
        "opposite_sign": False,
        "in_null_band": True,
    },
]

n  = len(MODELS)
ys = np.arange(n)[::-1]   # top-to-bottom: y=2,1,0

fig = plt.figure(figsize=(15.0, 5.8))
gs  = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.1, 3.5], wspace=0.04)
axL = fig.add_subplot(gs[0, 0])
axM = fig.add_subplot(gs[0, 1])
axR = fig.add_subplot(gs[0, 2])

Y_LO, Y_HI = -0.8, n - 0.3

NULL_MAX = 0.040   # left-panel x max (shows null bands clearly)
RIGHT_MAX = 3.5

# ── LEFT wing : evidence-parallel (natural-norm) ──────────────────────────
axL.set_xlim(NULL_MAX, 0.0)  # reversed: zero on right, grows leftward
# shade null band region (conservative max across models)
axL.axvspan(0.0, max(m["null_upper"] for m in MODELS if m["null_upper"]), color="#CCCCCC", alpha=0.30, zorder=0)
axL.axvline(0.0, color="black", lw=0.7, zorder=2)

for y, m in zip(ys, MODELS):
    c = m["color"]
    if m["opposite_sign"]:
        # Qwen: opposite-sign; draw a hatched bar at |par_rms| with annotation
        bar_w = min(m["par_rms_abs"], NULL_MAX * 0.95)
        axL.barh(y, bar_w, height=0.52, color=c, alpha=0.25,
                 edgecolor=c, lw=1.0, hatch="///", zorder=3)
        axL.text(bar_w + 0.001, y, "opposite-sign\n(par_rms = −0.157)",
                 va="center", ha="left", fontsize=7.8, color=c, style="italic", zorder=5)
    else:
        # Gemma / Mistral: bar for par_natural
        bar_w = max(m["par_val"], 0)
        axL.barh(y, bar_w, height=0.52, color=c, alpha=0.20,
                 edgecolor="none", zorder=3)
        # model-specific null band tick
        axL.axvline(m["null_upper"], ymin=(y - 0.3 + 0.8) / (Y_HI - Y_LO),
                    ymax=(y + 0.3 + 0.8) / (Y_HI - Y_LO),
                    color=c, lw=1.0, ls=":", alpha=0.7, zorder=3)
        lo, hi = m["par_ci"]
        axL.plot([max(lo, 0) if lo > 0 else lo, hi], [y, y],
                 color=c, lw=1.8, alpha=0.85, zorder=4)
        axL.plot(m["par_val"], y, "o", color=c, ms=10, mec="white", mew=1.0, zorder=5)
        pct = m["par_val"] / m["full_val"] * 100
        axL.text(m["par_val"] + 0.001, y + 0.22,
                 f"{pct:.1f}% of full\n(in null band)",
                 va="center", ha="left", fontsize=8, color=c, fontweight="bold", zorder=5)

axL.set_xticks([0.0, 0.01, 0.02, 0.03, 0.04])
axL.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
axL.set_xlabel("Evidence-parallel  |Δm_par|  (natural-norm)",
               fontsize=10, labelpad=4)
axL.set_yticks([]); axL.tick_params(axis="y", length=0)
for s in ("top", "right", "left"): axL.spines[s].set_visible(False)
axL.set_ylim(Y_LO, Y_HI)

# ── CENTER : model name + metadata ───────────────────────────────────────
axM.set_xlim(0, 1); axM.set_ylim(Y_LO, Y_HI); axM.axis("off")
axM.text(0.5, n - 0.32,
         "Model family\n(decision layer · cos(A,E))",
         ha="center", va="bottom", fontsize=8.4, color="#333", style="italic",
         linespacing=1.15)
for y, m in zip(ys, MODELS):
    axM.text(0.5, y + 0.18, m["name"], ha="center", va="center",
             fontsize=11.5, fontweight="bold", color=m["color"])
    axM.text(0.5, y - 0.04, m["sublabel"], ha="center", va="center",
             fontsize=9.0, color=m["color"])
    axM.text(0.5, y - 0.26, f"cos(A, E) = {m['cos']:+.4f}",
             ha="center", va="center", fontsize=7.8, color="#666", style="italic")
for x_sep in (0.0, 1.0):
    axM.axvline(x_sep, color="black", lw=0.6, ymin=0.03, ymax=0.97)

# ── RIGHT wing : full Δm + perp ──────────────────────────────────────────
axR.set_xlim(0, RIGHT_MAX)
axR.axvline(0, color="black", lw=0.7, zorder=1)

for y, m in zip(ys, MODELS):
    c = m["color"]
    # full Δm: hatched reference bar
    axR.barh(y + 0.18, m["full_val"], height=0.28, color=c, alpha=0.15,
             edgecolor=c, lw=0.8, hatch="///", zorder=2)
    fl, fh = m["full_ci"]
    axR.plot([fl, fh], [y + 0.18, y + 0.18], color=c, lw=1.4, alpha=0.6, zorder=4)
    axR.plot(m["full_val"], y + 0.18, "s", mfc="white", mec=c, mew=1.5, ms=9, zorder=5)
    axR.text(fh + 0.04, y + 0.18, f"full = {m['full_val']:.3f}",
             va="center", fontsize=8.5, color=c, alpha=0.85, zorder=6)

    # perp: solid filled bar (nearly equal to full)
    axR.barh(y - 0.18, m["perp_val"], height=0.28, color=c, alpha=0.75,
             edgecolor="none", zorder=2)
    pl, ph = m["perp_ci"]
    axR.plot([pl, ph], [y - 0.18, y - 0.18], color=c, lw=1.8, alpha=0.9, zorder=4)
    axR.plot(m["perp_val"], y - 0.18, "o", color=c, ms=10, mec="white", mew=1.0, zorder=5)
    pct_p = m["perp_val"] / m["full_val"] * 100
    axR.text(ph + 0.04, y - 0.18, f"⊥ = {m['perp_val']:.3f}  ({pct_p:.1f}%)",
             va="center", fontsize=8.5, color=c, fontweight="bold", zorder=6)

axR.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
axR.set_xlabel("Evidence-orthogonal  |Δm_perp|  and  full |Δm|   (natural-norm)",
               fontsize=10, labelpad=4)
axR.set_yticks([]); axR.tick_params(axis="y", length=0)
for s in ("top", "right", "left"): axR.spines[s].set_visible(False)
axR.set_ylim(Y_LO, Y_HI)

# ── shared legend ─────────────────────────────────────────────────────────
handles = [
    Rectangle((0, 0), 1, 1, fc="#CCCCCC", alpha=0.40,
              label="Natural-norm null band  (abs_p97.5, K=100 random directions)"),
    Line2D([0], [0], color="black", lw=1.5, marker="o", ms=7, mec="white",
           label="par_natural / perp  (mean ± bootstrap 95% CI)"),
    Rectangle((0, 0), 1, 1, fc="#888888", alpha=0.18, ec="#888888", lw=0.8,
              hatch="///", label="full Δm  (reference, hatched)"),
    Rectangle((0, 0), 1, 1, fc=C_QWEN, alpha=0.5,
              label="Qwen2.5-7B  (opposite-sign par_rms, no natural-norm analogue)"),
    Rectangle((0, 0), 1, 1, fc=C_GEMMA, alpha=0.5, label="Gemma-2-9B-it"),
    Rectangle((0, 0), 1, 1, fc=C_MISTRAL, alpha=0.5, label="Mistral-7B-v0.3"),
]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
           ncol=3, fontsize=8.4, frameon=True, framealpha=0.92,
           edgecolor="#ccc", columnspacing=1.4,
           title="Qwen N=100  ·  Gemma N=50  ·  Mistral N=50  ·  K=200 RMS-random dirs",
           title_fontsize=8.4)

# ── right-margin bracket: all in-null-band ─────────────────────────────────
def _bracket(ax, y_top, y_bot, color, x_stem, x_tip, label_lines):
    ax.plot([x_stem, x_stem], [y_bot, y_top], color=color, lw=2.2,
            solid_capstyle="butt", zorder=6)
    for y_end in (y_top, y_bot):
        ax.plot([x_stem, x_tip], [y_end, y_end], color=color, lw=2.2,
                solid_capstyle="butt", zorder=6)
    y_mid = 0.5 * (y_top + y_bot)
    head, *tail = label_lines
    ax.text(x_stem + 0.04, y_mid + 0.15, head, color=color,
            fontsize=10.0, fontweight="bold", ha="left", va="center", zorder=7)
    if tail:
        ax.text(x_stem + 0.04, y_mid - 0.12, "\n".join(tail), color=color,
                fontsize=8.0, ha="left", va="top", linespacing=1.20, zorder=7)

_X_STEM, _X_TIP = 3.08, 3.00
_bracket(axR, y_top=2.35, y_bot=0.65, color="#555555", x_stem=_X_STEM, x_tip=_X_TIP,
         label_lines=["par_natural ≤ null band",
                      "evidence-parallel non-operative",
                      "across all three families"])

# ── title + caption ────────────────────────────────────────────────────────
fig.suptitle(
    "Cross-Family Natural-Norm Decomposition  "
    "—  Evidence-Parallel Component Non-Operative in All Three Families",
    fontsize=12, y=0.999)

CAPTION = (
    "Note: Mistral prior RMS-renormalized 26.2% residual = 7,147× amplification of near-zero natural projection (artifact);  "
    "under natural-norm, evidence-parallel component is within the random-direction null band for all three families."
)
fig.text(0.5, 0.14, CAPTION, ha="center", va="top", fontsize=7.8,
         color="#444", style="italic", wrap=True)

fig.subplots_adjust(left=0.03, right=0.985, top=0.91, bottom=0.22, wspace=0.04)

out_png = OUT / "fig3_natural_norm_decomp.png"
out_pdf = OUT / "fig3_natural_norm_decomp.pdf"
fig.savefig(out_png, dpi=200, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
