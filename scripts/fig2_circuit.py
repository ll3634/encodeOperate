#!/usr/bin/env python3
"""Figure 2 — Two-Stage Evidence-to-Action Circuit.

Three panels:
  A: Circuit schematic  (attn_L18 KV2 → mlp_L20 → action) + cross-model table
  B: Causal recovery bars (behavioral_patching, N=50)
  C: Sufficiency & Necessity curves (path_patching, N=50)
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent
OUT   = ROOT / "results/fig2_circuit"; OUT.mkdir(parents=True, exist_ok=True)

bp = json.load(open(ROOT / "results/behavioral_patching/behavioral_patching_results.json"))
pp = json.load(open(ROOT / "results/path_patching/path_patching_results.json"))

C_MLP  = "#D4660A"   # mlp_L20 — burnt orange
C_ATTN = "#5B3FA0"   # attn_L18 — indigo
C_KV2  = "#9B78D4"   # KV2 subgroup — light purple
C_CTRL = "#AAAAAA"   # control — grey
C_OBS  = "#2357A0"   # observation tokens — dark blue
C_ACT  = "#1A7A40"   # action direction — green
C_STRM = "#888888"   # residual stream — mid-grey
C_EVID = "#1F78B4"   # evidence subspace — blue
C_BROKEN = "#C0392B" # broken circuit — red

fig = plt.figure(figsize=(18.0, 10.5))
gs  = fig.add_gridspec(2, 3, width_ratios=[1.85, 1.0, 1.05],
                        height_ratios=[1.3, 0.70],
                        left=0.02, right=0.98, top=0.93, bottom=0.06,
                        wspace=0.28, hspace=0.38)
axA  = fig.add_subplot(gs[0, 0])   # circuit schematic (top-left)
axAt = fig.add_subplot(gs[1, 0])   # cross-model table (bottom-left)
axB  = fig.add_subplot(gs[0, 1])
axC  = fig.add_subplot(gs[0, 2])
# merge B/C bottom rows for shared use
axB2 = fig.add_subplot(gs[1, 1])
axC2 = fig.add_subplot(gs[1, 2])
axB2.axis("off"); axC2.axis("off")

# ── Panel A : schematic ───────────────────────────────────────────────────────
axA.set_xlim(0, 12); axA.set_ylim(0, 7.8); axA.axis("off")
axA.set_title("(a)  Two-Stage Routing Circuit  (Qwen2.5-7B-Instruct)",
              fontsize=11, fontweight="bold", pad=6)

def _box(ax, cx, cy, w, h, lines, fc, ec=None, fs=8.5, alpha=0.90, tc="white"):
    ec = ec or fc
    p = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                       boxstyle="round,pad=0.10", lw=1.5,
                       facecolor=fc, edgecolor=ec, alpha=alpha, zorder=3)
    ax.add_patch(p)
    for i, ln in enumerate(lines):
        dy = (len(lines)-1)*0.19 - i*0.38
        ax.text(cx, cy+dy, ln, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=4)

def _arr(ax, x1,y1,x2,y2, color, lw=2.0, rad=0.0, label="", label_side="top", fs=7.2):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="->, head_length=0.35, head_width=0.18",
                                color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"), zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        dy = 0.23 if label_side=="top" else -0.23
        ax.text(mx, my+dy, label, ha="center", va="center",
                fontsize=fs, color=color, style="italic", zorder=6)

# ── observation tokens row ─────────────────────────────────────────────────────
for i, ox in enumerate([1.0, 2.1, 3.2, 4.1]):
    lbl = [f"obs {i+1}"] if i < 3 else ["obs n"]
    _box(axA, ox, 7.0, 0.88, 0.60, lbl, C_OBS, fs=8.0, alpha=0.85)
axA.text(4.75, 7.0, "· · ·", ha="center", va="center", fontsize=12, color=C_OBS)

# subspace badge on obs tokens
axA.text(1.0, 7.55, "evidence subspace", ha="center", va="center",
         fontsize=7.5, color=C_EVID,
         bbox=dict(boxstyle="round,pad=0.2", fc=C_EVID+"18", ec=C_EVID, lw=0.8))

# ── decision token ─────────────────────────────────────────────────────────────
_box(axA, 9.5, 7.0, 1.7, 0.70, ["decision token", "(p0)"], "#444444", fs=8.2)
axA.text(9.5, 7.58, "action subspace\n(cos(ev,ac)=−0.013)", ha="center", va="center",
         fontsize=7.0, color=C_ACT,
         bbox=dict(boxstyle="round,pad=0.2", fc=C_ACT+"18", ec=C_ACT, lw=0.8))

# ── attn L18 box — Stage 1: routing ──────────────────────────────────────────
_box(axA, 5.3, 5.05, 6.0, 1.20,
     ["Attn  L18  ·  KV Group 2  (H14–H20)",
      "cross-position · cross-subspace routing",
      "H15: obs-attn=35.7%  |  KV2 obs-attn=16.2% (vs 8% other)"],
     C_ATTN, fs=8.0)

# KV group legend inset on attn box
for gi, (label, rec, col) in enumerate([("KV0", "2.0%", "#7bafd4"),
                                          ("KV1", "4.6%", "#a8d5a2"),
                                          ("KV2", "11.5%", C_KV2),
                                          ("KV3", "4.5%", "#b39ddb")]):
    bx = 8.2 + gi * 0.78
    by = 5.05
    axA.add_patch(FancyBboxPatch((bx-0.33, by-0.52), 0.66, 0.38,
                                  boxstyle="round,pad=0.04", lw=1.2 if gi==2 else 0.6,
                                  facecolor=col, edgecolor="white", alpha=0.88, zorder=6))
    axA.text(bx, by-0.33, f"{label}\n{rec}", ha="center", va="center",
             fontsize=6.5, color="white", fontweight="bold" if gi==2 else "normal", zorder=7)

# ── residual-stream vertical bar at decision token ───────────────────────────
axA.plot([9.5, 9.5], [1.1, 6.64], color=C_STRM, lw=7, solid_capstyle="round",
         alpha=0.22, zorder=1)
axA.text(10.5, 4.1, "residual\nstream\n(dec. tok)", ha="center", va="center",
         fontsize=7.0, color=C_STRM, style="italic")
# signal accumulation markers
for ly, label, col, val in [(6.64,"→L18",C_STRM,"baseline"),
                              (5.05,"→L20\nin: act=1.785\n   ev=0.620",C_ATTN,""),
                              (2.8,"→L20\nout: act=2.293(+28%)\n   ev=0.344(−45%)",C_MLP,"")]:
    axA.text(10.55, ly, label, ha="left", va="center", fontsize=6.0, color=col)

# ── mlp L20 box — Stage 2: amplification ─────────────────────────────────────
_box(axA, 9.5, 3.0, 2.8, 1.20,
     ["MLP  L20  ·  action amplifier",
      "Jacobian: action→action = 12.1× random",
      "Jacobian: evidence→action = 0.66× random",
      "same-subspace gain ×1.28"],
     C_MLP, fs=7.8)

# ── action direction box ──────────────────────────────────────────────────────
_box(axA, 9.5, 1.18, 2.9, 0.80,
     ["action direction  (L20)",
      "search  ↔  stop  |  ⊥/full = 0.999"],
     C_ACT, fs=8.0)

# ── arrows ────────────────────────────────────────────────────────────────────
# obs → attn_L18 (K/V read)
for ox in [1.0, 2.1, 3.2]:
    _arr(axA, ox, 6.69, 3.4, 5.67, C_OBS, lw=1.2, rad=0.08)

# attn_L18 → decision token (value output → cross-position)
_arr(axA, 8.3, 5.05, 9.2, 6.64, C_ATTN, lw=2.6, rad=-0.25,
     label="cross-pos routing\nattn_L18: median recovery 18.7%\nL19/L20: ~2% (routing done)", label_side="top", fs=7.0)

# decision token → mlp_L20 (through residual stream)
_arr(axA, 9.5, 6.64, 9.5, 3.62, C_STRM, lw=2.2)

# mlp_L20 → action direction
_arr(axA, 9.5, 2.38, 9.5, 1.60, C_MLP, lw=2.6,
     label="mlp_L20: 51.4% causal recovery\n(mean=0.514, Wilcoxon p=1.5e-6)", label_side="top", fs=7.0)

# Q from dec tok to attn
axA.text(7.1, 4.50, "Q ← dec. tok", ha="center", va="center",
         fontsize=7.0, color=C_ATTN, style="italic")

# decomposition annotation
axA.annotate("Decomposition:\nfull → +15 EM\nperp → +14 EM\nparallel → −1 EM\n⇒ 100% via ⊥ component",
             xy=(9.5, 1.18), xytext=(6.3, 1.5),
             fontsize=7.0, color=C_ACT, ha="left",
             arrowprops=dict(arrowstyle="->", color=C_ACT, lw=1.0))

# ── Panel B : causal recovery bars ───────────────────────────────────────────
axB.set_title("(b)  Causal Recovery\nper Component (N=50)", fontsize=10, fontweight="bold", pad=6)
sm  = bp["summary"]
ts  = bp["tests"]

comp_keys  = ["mlp_L20", "attn_L18", "kv2_L18", "kv0_L18"]
comp_labels= ["mlp_L20", "attn_L18\n(full)", "attn_L18\n(KV2)", "attn_L18\n(KV0 ctrl)"]
comp_means  = [sm[k]["mean_recovery"] for k in comp_keys]
comp_sems   = [sm[k]["std_recovery"] / np.sqrt(sm[k]["n"]) for k in comp_keys]
comp_medians= [sm[k]["median_recovery"] for k in comp_keys]
comp_p      = [ts[k]["p"] for k in comp_keys]
comp_colors = [C_MLP, C_ATTN, C_KV2, C_CTRL]

ys_B = np.arange(len(comp_keys))
for i, (y, m, med, sem, c, p) in enumerate(
        zip(ys_B, comp_means, comp_medians, comp_sems, comp_colors, comp_p)):
    axB.barh(y, m, height=0.52, color=c, alpha=0.88, zorder=3)
    axB.plot([max(m-sem, 0), m+sem], [y, y], color="#333", lw=1.5, zorder=5)
    axB.plot(med, y, "|", color="white", ms=10, mew=2.5, zorder=6)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else "*" if p < 0.05 else "ns")
    axB.text(m + sem + 0.018, y, f"μ={m:.3f} {sig}\nmed={med:.3f}",
             va="center", fontsize=7.5, color=c)

axB.set_yticks(ys_B); axB.set_yticklabels(comp_labels, fontsize=9.0)
axB.set_xlim(-0.02, 0.90)
axB.set_xlabel("Mean causal recovery\n(Wilcoxon vs 0, N=50)", fontsize=9, labelpad=4)
axB.axvline(0, color="black", lw=0.8)
axB.invert_yaxis()
for s in ("top", "right"): axB.spines[s].set_visible(False)
axB.text(0.04, 3.7, "│ = median", fontsize=7, color="#555", style="italic")

# ── Panel C : sufficiency / necessity curves ──────────────────────────────────
axC.set_title("(c)  Sufficiency & Necessity\n(path patching, N=50)", fontsize=10, fontweight="bold", pad=6)

k_vals = [1, 2, 3, 5, 7]
suff_m = [pp[f"suff_top{k}"]["median"] for k in k_vals]
nec_7  = pp["nec_top7"]["median"]
ctrl_m = pp["suff_bottom7"]["median"]

C_SUFF = "#2F6FB5"
axC.plot(k_vals, suff_m, "o-", color=C_SUFF, lw=2.4, ms=7.5, zorder=4,
         label="Sufficiency (patch top-k in)")
axC.fill_between(k_vals, suff_m, ctrl_m, alpha=0.12, color=C_SUFF)
axC.axhline(ctrl_m, color=C_CTRL, lw=1.5, ls="--", zorder=2,
            label=f"Bottom-7 ctrl ({ctrl_m:.2f})")
axC.plot(7, nec_7, "s", color=C_BROKEN, ms=10, mec="white", mew=1.8, zorder=6)
axC.annotate(f"Necessity@k=7\n(ablate): {nec_7:.2f}",
             xy=(7, nec_7), xytext=(5.2, 0.65),
             fontsize=8, color=C_BROKEN,
             arrowprops=dict(arrowstyle="->", color=C_BROKEN, lw=1.2))
for k, s in zip(k_vals, suff_m):
    axC.text(k, s+0.04, f"{s:.2f}", ha="center", va="bottom", fontsize=7.5, color=C_SUFF)

axC.set_xlabel("Top-k components (by median attr.)", fontsize=9, labelpad=4)
axC.set_ylabel("Median recovery fraction", fontsize=9)
axC.set_xlim(0.5, 7.8); axC.set_ylim(0, 1.10)
axC.set_xticks(k_vals)
axC.legend(fontsize=8.0, frameon=True, framealpha=0.92, edgecolor="#ccc", loc="lower right")
for s in ("top", "right"): axC.spines[s].set_visible(False)

# ── Panel At : Cross-model comparison table ────────────────────────────────────
axAt.set_xlim(0, 12); axAt.set_ylim(0, 5.5); axAt.axis("off")
axAt.set_title("(d)  Cross-Model Circuit Evidence", fontsize=10, fontweight="bold", pad=4)

COLS = ["Model", "Arch", "Ev.AUROC\n(peak L)", "cos(ev,ac)",
        "Formation (routing)", "Locality",
        "Gain node", "A/B ratio (p)", "⊥/full", "Circuit"]
COL_X = [0.52, 1.50, 2.55, 3.52, 4.85, 6.00, 7.15, 8.45, 9.60, 10.75]
HDR_Y = 5.0
for cx, col in zip(COL_X, COLS):
    axAt.text(cx, HDR_Y, col, ha="center", va="center", fontsize=6.5, fontweight="bold",
              color="white",
              bbox=dict(boxstyle="round,pad=0.15", fc="#444444", ec="none"))

ROWS = [
    ("Qwen2.5-7B", "28L 3584h\nGQA 4×7",
     "0.862 (L18)", "−0.013",
     "attn_L18 KV2\nH14–H20\nH15:35.7% obs", "sample-\nspecific",
     "mlp_L20\nJ=12.1×\n×1.28 same-SS", "1.75*** (7e-11)", "0.999", "✓ full", "#E8F4FD"),
    ("Mistral-7B", "32L 4096h\nMHA",
     "0.773 (L16)", "−0.009",
     "attn_L16\nrecov=+0.58\n(matched)", "sample-spec\n→0.14 mism.",
     "mlp_L28\n~4× ev.\nsame-layer", "1.78*** (7e-10)", "0.999", "✓ full", "#E8F8EA"),
    ("Gemma-2-9B", "42L 3584h\nMHA",
     "0.842 (L23)", "−0.017",
     "attn_L25\nrecov=+0.70\n(class-level)", "class-level\n(≈mismatched)",
     "mlp_L37\n22× rand.\n3.1× ev.", "1.60*** (3e-8)", "0.999", "✓ part.", "#E8F8EA"),
    ("Llama-3.1-8B", "32L 4096h\nGQA",
     "0.861 (L24)", "+0.021",
     "✗ E5 AB=0.97\np=0.91 (null)", "—",
     "— (no routing)", "1.06 ns (0.42)", "—", "✗ broken", "#FEF0F0"),
    ("R1-Distill-7B", "28L 3584h\nGQA",
     "0.777 (L22)", "−0.031",
     "✗ degenerate\nlabels 486/486\n(CoT bypass)", "—",
     "—", "1.05 ns (0.48)", "—", "✗ none", "#FEF0F0"),
]

for ri, row in enumerate(ROWS):
    ry = HDR_Y - 0.95 * (ri + 1)
    model, arch, ev, cos_s, form, loc, gain, ab, perp, verdict, rc = row
    axAt.fill_between([0, 12], [ry-0.43, ry-0.43], [ry+0.43, ry+0.43],
                      color=rc, alpha=0.75, zorder=0)
    vals = [model, arch, ev, cos_s, form, loc, gain, ab, perp, verdict]
    for cx, val in zip(COL_X, vals):
        vc = C_BROKEN if "✗" in val else (C_ACT if "✓" in val else "#222222")
        fw = "bold" if val in [model, verdict] else "normal"
        axAt.text(cx, ry, val, ha="center", va="center", fontsize=6.3,
                  color=vc, fontweight=fw)

axAt.axhline(HDR_Y - 0.50, color="#999999", lw=0.8)

# ── shared title & save ───────────────────────────────────────────────────────
fig.suptitle(
    "Evidence-to-Action Circuit: attn_L18 (KV2) → mlp_L20  |  "
    "Cross-model: Qwen2.5 / Mistral / Gemma ✓   Llama-3.1 / R1-Distill ✗",
    fontsize=11.5, y=0.997)

out = OUT / "fig2_circuit.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
out_pdf = OUT / "fig2_circuit.pdf"
fig.savefig(out_pdf, bbox_inches="tight")
print(f"Saved: {out_pdf}")
