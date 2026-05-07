#!/usr/bin/env python3
"""Figure 3 — Mechanistic Interpretability: Evidence→Action Routing Circuit.

Two panels:
  A (left ~65%): Circuit diagram with head-level attention heatmap nodes
     - Residual streams as vertical rails (obs / decision token positions)
     - attn_L18: 28-head heatmap (obs-attn × action-proj) embedded in node
     - mlp_L20: I/O bar node (action-specific gain)
     - Thought erosion decay shown as fading strip
  B (right ~35%): Cross-model replication table (5 models)
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import Normalize
import numpy as np

# ── Head-level data (Phase F4 + F1) ──────────────────────────────────────────
OBS_CLEAN = [0.0428,0.0086,0.0870,0.0241,0.2122,0.1234,0.0356,
             0.0018,0.0783,0.0387,0.0609,0.0334,0.1105,0.2152,
             0.1066,0.3574,0.1125,0.0326,0.1801,0.1125,0.2315,
             0.1096,0.1492,0.0366,0.0922,0.0730,0.0413,0.1047]
ACTION_PROJ = [0.0270,0.0133,0.0085,0.0116,0.0476,0.0408,0.0086,
               0.0555,0.0190,0.0446,0.0160,0.0145,0.0232,0.0878,
               0.0355,0.1031,0.0235,0.0393,0.0745,0.0301,0.0785,
               0.0712,0.0167,0.0087,0.0316,0.0233,0.0466,0.0136]
KV_GROUPS = [0]*7 + [1]*7 + [2]*7 + [3]*7
# KV group causal recovery (Phase F3)
KV_RECOVERY = [0.020, 0.046, 0.115, 0.045]
# MLP I/O (Phase C)
MLP_IN_ACTION  = 1.785; MLP_OUT_ACTION = 2.293
MLP_IN_EV      = 0.620; MLP_OUT_EV     = 0.344

_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent
OUT   = ROOT / "results/fig3_circuit"; OUT.mkdir(parents=True, exist_ok=True)

# ── Colours ──────────────────────────────────────────────────────────────────
C_EV   = "#2563EB"   # evidence subspace  (blue)
C_AC   = "#DC2626"   # action subspace    (red)
C_ATTN = "#7C3AED"   # attn_L18 KV2      (purple)
C_MLP  = "#D97706"   # mlp_L20           (amber)
C_STRM = "#374151"   # residual stream    (dark grey)
C_OBS  = "#0F766E"   # observation tokens (teal)
C_DEC  = "#1D4ED8"   # decision token     (deep blue)
C_FADE = "#94A3B8"   # faded/eroded       (slate)
C_OK   = "#16A34A"   # confirmed (green)
C_FAIL = "#B91C1C"   # not replicated (red)
C_PART = "#CA8A04"   # partial            (yellow)
BG     = "#F8FAFC"

# ── Cross-model data (all verified from experiment files) ─────────────────────
MODELS = [
    # (short_name, n_layers, peak_ev_L, peak_ac_L, ev_auroc, cos_ev_ac,
    #  AB_ratio_action, AB_p, decomp_perp_ratio, decomp_confirmed)
    ("Qwen2.5-7B\n(primary)",    28, 18, 20, 0.862, -0.013,  1.75, "7e-11", 0.999, True),
    ("Mistral-7B\n-Instruct",    32, 16, 28, 0.773, -0.009,  1.78, "6e-10", 0.999, True),
    ("Gemma-2-9B\n-Instruct",    42, 23, 37, 0.842, -0.017,  1.60, "3e-8",  0.999, True),
    ("Llama-3.1-8B\n-Instruct",  32, 24, 28, 0.861, +0.021,  1.04, "0.42",  None,  False),
    ("R1-Distill\nQwen-7B",      28, 22, 22, 0.777, -0.031,  1.09, "0.48",  None,  False),
]

# ── Layout ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 11), facecolor=BG)
fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.04)

axA = fig.add_axes([0.01, 0.04, 0.60, 0.86])
axB = fig.add_axes([0.64, 0.04, 0.35, 0.86])
for ax in [axA, axB]:
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

# ═══════════════════════════════════════════════════════════════════════════
# PANEL A — Circuit Diagram with head-level nodes
# ═══════════════════════════════════════════════════════════════════════════
# Layout (axes-fraction coords, y=0 bottom, y=1 top):
#   obs column: x=0.16   dec column: x=0.62
#   Y_EMBED=0.06  Y_L18_BOT=0.18  Y_L18_TOP=0.54  Y_L20=0.68  Y_OUT=0.87
OBS_X, DEC_X = 0.16, 0.62
Y_EMBED = 0.065
Y_L18_BOT, Y_L18_TOP = 0.185, 0.545   # attn node vertical span
Y_L18_MID = (Y_L18_BOT + Y_L18_TOP) / 2
Y_L20     = 0.685
Y_OUT     = 0.875

KV_COLORS_HEX = ["#7bafd4", "#a8d5a2", "#e07b54", "#b39ddb"]  # KV0-3

def rail(cx, y_lo, y_hi, color, lw=4, alpha=0.85):
    axA.plot([cx, cx], [y_lo, y_hi], color=color, lw=lw, alpha=alpha,
             solid_capstyle="round", zorder=2, transform=axA.transAxes)

def tok_box(cx, cy, label, color, w=0.18, h=0.055):
    r = FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.01",
                       linewidth=1.6, edgecolor=color, facecolor=color+"1A", zorder=4,
                       transform=axA.transAxes)
    axA.add_patch(r)
    axA.text(cx, cy, label, ha="center", va="center", fontsize=7.8,
             fontweight="bold", color=color, zorder=5, transform=axA.transAxes)

def arc_arrow(x0, y0, x1, y1, color, rad=-0.38, lw=2.8, hw=0.020, hl=0.028):
    axA.annotate("", xy=(x1, y1), xytext=(x0, y0),
                 xycoords="axes fraction", textcoords="axes fraction",
                 arrowprops=dict(arrowstyle=f"->, head_width={hw}, head_length={hl}",
                                 color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"), zorder=7)

def lbl(x, y, txt, color, fs=8.0, ha="center", va="center", bold=False, bg=None):
    kw = dict(ha=ha, va=va, fontsize=fs, color=color, zorder=9,
              fontweight="bold" if bold else "normal", transform=axA.transAxes)
    if bg:
        kw["bbox"] = dict(boxstyle="round,pad=0.22", facecolor=bg, edgecolor="none", alpha=0.88)
    axA.text(x, y, txt, **kw)

# ── Residual stream rails ─────────────────────────────────────────────────
# obs: carries evidence signal (blue) all the way to attn node
rail(OBS_X, Y_EMBED+0.03, Y_L18_BOT, C_EV, lw=4)
# dec: before attn → neutral grey; after attn (L18→L20) → action red; above L20 → red
rail(DEC_X, Y_EMBED+0.03, Y_L18_BOT, C_STRM, lw=4, alpha=0.45)
rail(DEC_X, Y_L18_TOP,    Y_L20-0.035, C_AC,   lw=4, alpha=0.85)
rail(DEC_X, Y_L20+0.038,  Y_OUT-0.028, C_AC,   lw=4.5, alpha=0.92)

# ── Input token boxes ─────────────────────────────────────────────────────
tok_box(OBS_X, Y_EMBED, "Observation tokens\n(retrieved evidence)", C_OBS)
tok_box(DEC_X, Y_EMBED, "Decision token\n(last context token, p0)", C_DEC)

# evidence signal arrow into obs stream
arc_arrow(OBS_X, Y_EMBED+0.032, OBS_X, Y_EMBED+0.072, C_EV, rad=0, lw=2.2, hw=0.014, hl=0.022)
lbl(OBS_X-0.09, Y_EMBED+0.052, "evidence\nsignal", C_EV, fs=7.5, ha="right")

# ── attn_L18 NODE: 28-head heatmap ───────────────────────────────────────
# Outer border for the entire attn block (obs column side)
NODE_W = 0.26; NODE_H = Y_L18_TOP - Y_L18_BOT
attn_border = FancyBboxPatch((OBS_X - NODE_W/2, Y_L18_BOT), NODE_W, NODE_H,
                              boxstyle="round,pad=0.006", linewidth=2.0,
                              edgecolor=C_ATTN, facecolor=C_ATTN+"0C", zorder=3,
                              transform=axA.transAxes)
axA.add_patch(attn_border)
axA.text(OBS_X, Y_L18_TOP+0.012, "attn  L18  —  all 28 heads", ha="center", va="bottom",
         fontsize=9.2, fontweight="bold", color=C_ATTN, transform=axA.transAxes, zorder=8)

# Draw 28 head cells: 2 columns (obs-attn, action-proj), 28 rows
CELL_W = 0.090; CELL_H = (NODE_H - 0.018) / 28
OBS_COL_X = OBS_X - CELL_W*0.56     # center of col-1
ACT_COL_X = OBS_X + CELL_W*0.56     # center of col-2
OBS_NORM  = Normalize(0.0, 0.40)
ACT_NORM  = Normalize(0.0, 0.11)
OBS_CMAP  = plt.get_cmap("Blues")
ACT_CMAP  = plt.get_cmap("Oranges")

# column headers
axA.text(OBS_COL_X, Y_L18_TOP-0.004, "Obs-attn", ha="center", va="top",
         fontsize=6.5, color="#2563EB", transform=axA.transAxes)
axA.text(ACT_COL_X, Y_L18_TOP-0.004, "Act-proj", ha="center", va="top",
         fontsize=6.5, color="#D97706", transform=axA.transAxes)

KV_DIVS = {7: "KV1", 14: "KV2", 21: "KV3"}   # first head of each new group
for head in range(28):
    kv = KV_GROUPS[head]
    y_cell = Y_L18_TOP - 0.010 - (head + 0.5) * CELL_H  # top→bottom
    is_kv2 = (kv == 2)
    ew = 1.4 if is_kv2 else 0.3
    ec_obs = "#cc4400" if is_kv2 else "#aaaaaa"
    ec_act = "#cc4400" if is_kv2 else "#aaaaaa"

    # obs-attn cell
    fc_obs = OBS_CMAP(OBS_NORM(OBS_CLEAN[head]))
    r1 = FancyBboxPatch((OBS_COL_X - CELL_W/2, y_cell - CELL_H/2 + 0.001),
                         CELL_W, CELL_H - 0.002, boxstyle="square,pad=0",
                         linewidth=ew, edgecolor=ec_obs, facecolor=fc_obs,
                         zorder=5, transform=axA.transAxes)
    axA.add_patch(r1)

    # action-proj cell
    fc_act = ACT_CMAP(ACT_NORM(ACTION_PROJ[head]))
    r2 = FancyBboxPatch((ACT_COL_X - CELL_W/2, y_cell - CELL_H/2 + 0.001),
                         CELL_W, CELL_H - 0.002, boxstyle="square,pad=0",
                         linewidth=ew, edgecolor=ec_act, facecolor=fc_act,
                         zorder=5, transform=axA.transAxes)
    axA.add_patch(r2)

    # head label (left of cell block)
    fw = "bold" if is_kv2 else "normal"
    fc2 = C_ATTN if is_kv2 else "#555555"
    axA.text(OBS_X - NODE_W/2 - 0.005, y_cell, f"H{head:02d}",
             ha="right", va="center", fontsize=5.8, fontweight=fw, color=fc2,
             transform=axA.transAxes, zorder=8)

    # value text inside cell (only if large enough)
    tc_obs = "white" if OBS_NORM(OBS_CLEAN[head]) > 0.55 else "#222"
    tc_act = "white" if ACT_NORM(ACTION_PROJ[head]) > 0.55 else "#222"
    axA.text(OBS_COL_X, y_cell, f"{OBS_CLEAN[head]:.2f}", ha="center", va="center",
             fontsize=4.8, color=tc_obs, zorder=6, transform=axA.transAxes)
    axA.text(ACT_COL_X, y_cell, f"{ACTION_PROJ[head]:.3f}", ha="center", va="center",
             fontsize=4.8, color=tc_act, zorder=6, transform=axA.transAxes)

    # KV group divider lines
    if head in KV_DIVS:
        div_y = Y_L18_TOP - 0.010 - head * CELL_H
        axA.plot([OBS_X - NODE_W/2, OBS_X + NODE_W/2], [div_y, div_y],
                 color="#888888", lw=0.8, transform=axA.transAxes, zorder=6)
        axA.text(OBS_X + NODE_W/2 + 0.004, div_y, KV_DIVS[head],
                 ha="left", va="center", fontsize=6.2, color="#888888",
                 transform=axA.transAxes, zorder=8)

# KV group causal recovery bars on right side of attn node
BAR_X0 = OBS_X + NODE_W/2 + 0.035
BAR_MAX_W = 0.09
kv_ys = [Y_L18_TOP - 0.010 - 3.5*CELL_H,
         Y_L18_TOP - 0.010 - 10.5*CELL_H,
         Y_L18_TOP - 0.010 - 17.5*CELL_H,
         Y_L18_TOP - 0.010 - 24.5*CELL_H]
kv_labels_short = ["KV0\nrec=0.020", "KV1\nrec=0.046", "KV2\nrec=0.115\n×5.8×", "KV3\nrec=0.045"]
for ki, (ky, kr, klab) in enumerate(zip(kv_ys, KV_RECOVERY, kv_labels_short)):
    bw = BAR_MAX_W * (kr / 0.115)
    fc_bar = KV_COLORS_HEX[ki] if ki != 2 else C_ATTN
    alpha_b = 0.9 if ki == 2 else 0.6
    rb = FancyBboxPatch((BAR_X0, ky - CELL_H*2.5), bw, CELL_H*5,
                        boxstyle="round,pad=0.003", facecolor=fc_bar, edgecolor="none",
                        alpha=alpha_b, zorder=5, transform=axA.transAxes)
    axA.add_patch(rb)
    axA.text(BAR_X0 + bw + 0.005, ky, klab, ha="left", va="center",
             fontsize=5.8, color=fc_bar if ki != 2 else C_ATTN,
             fontweight="bold" if ki == 2 else "normal",
             transform=axA.transAxes, zorder=8)

axA.text(BAR_X0 + BAR_MAX_W/2, Y_L18_TOP + 0.012, "F3: causal\nrecovery",
         ha="center", va="bottom", fontsize=6.5, color="#555555",
         transform=axA.transAxes, zorder=8)

# ── Cross-position routing arc (attn_L18 obs→dec) ────────────────────────
arc_arrow(OBS_X + NODE_W/2, Y_L18_MID, DEC_X - 0.04, Y_L18_MID + 0.04,
          C_ATTN, rad=-0.28, lw=3.0, hw=0.022, hl=0.030)
lbl(0.40, Y_L18_MID + 0.09,
    "cross-position · cross-subspace\nrouting  (obs → dec)   median rec=18.7%",
    C_ATTN, fs=7.8, bg=C_ATTN+"18")

# ── mlp_L20 NODE: I/O bar node ───────────────────────────────────────────
MLP_W = 0.24; MLP_H = 0.10
mlp_border = FancyBboxPatch((DEC_X - MLP_W/2, Y_L20 - MLP_H/2), MLP_W, MLP_H,
                             boxstyle="round,pad=0.008", linewidth=2.0,
                             edgecolor=C_MLP, facecolor=C_MLP+"18", zorder=4,
                             transform=axA.transAxes)
axA.add_patch(mlp_border)
axA.text(DEC_X, Y_L20 + 0.022, "mlp  L20", ha="center", va="center",
         fontsize=9.5, fontweight="bold", color=C_MLP, transform=axA.transAxes, zorder=5)

# Mini I/O bar inside the MLP node
for bar_i, (label_txt, val_in, val_out, bcolor) in enumerate([
        ("action", MLP_IN_ACTION, MLP_OUT_ACTION, C_AC),
        ("evidence", MLP_IN_EV, MLP_OUT_EV, C_EV)]):
    by = Y_L20 - 0.004 - bar_i * 0.022
    bscale = 0.08 / 2.293   # scale: max bar = 2.293 → width 0.08
    # input bar (left half)
    axA.barh(by, val_in * bscale, left=DEC_X - 0.13, height=0.013,
             color=bcolor, alpha=0.5, transform=axA.transAxes, zorder=6)
    # output bar (right half, reversed for clarity)
    axA.barh(by, val_out * bscale, left=DEC_X + 0.02, height=0.013,
             color=bcolor, alpha=0.9, transform=axA.transAxes, zorder=6)
    axA.text(DEC_X - 0.135, by, label_txt, ha="right", va="center",
             fontsize=5.8, color=bcolor, transform=axA.transAxes, zorder=7)

axA.text(DEC_X - 0.095, Y_L20 - 0.055, "in", ha="center", va="top",
         fontsize=6.0, color="#888", transform=axA.transAxes)
axA.text(DEC_X + 0.075, Y_L20 - 0.055, "out", ha="center", va="top",
         fontsize=6.0, color="#888", transform=axA.transAxes)

# Jacobian annotation to the right of MLP node
lbl(DEC_X + 0.16, Y_L20 + 0.018,
    "J(action→action): 12.1×\nJ(evidence→action): 0.66×\ngain ×1.28 on action dir",
    C_MLP, fs=7.5, ha="left")

# ── Output node ──────────────────────────────────────────────────────────
out_box = FancyBboxPatch((DEC_X-0.11, Y_OUT-0.038), 0.22, 0.062,
                         boxstyle="round,pad=0.01", linewidth=2.0,
                         edgecolor="#16A34A", facecolor="#16A34A22", zorder=4,
                         transform=axA.transAxes)
axA.add_patch(out_box)
lbl(DEC_X, Y_OUT+0.004, "Action decision", "#16A34A", fs=9.5, bold=True)
lbl(DEC_X, Y_OUT-0.023, "search  ↔  stop", "#16A34A", fs=8.0)

# Orthogonality annotation between rails
lbl(DEC_X + 0.15, (Y_L18_TOP + Y_L20) / 2,
    "action subspace\ncos(ev, ac) = −0.013\n⊥/full = 0.999",
    C_AC, fs=7.8, ha="left", bg=C_AC+"12")

# ── Layer markers on left ─────────────────────────────────────────────────
for y_pos, lbl_txt in [(Y_EMBED, "Input"), (Y_L18_BOT, "L18 ▼"),
                        (Y_L18_TOP, "L18 ▲"), (Y_L20, "L20"), (Y_OUT, "Output")]:
    axA.plot([0.00, 0.035], [y_pos, y_pos], color="#CBD5E1", lw=0.8,
             transform=axA.transAxes, zorder=1)
    axA.text(0.016, y_pos, lbl_txt, transform=axA.transAxes,
             ha="center", va="center", fontsize=7.0, color="#64748B",
             bbox=dict(boxstyle="round,pad=0.16", facecolor=BG, edgecolor="none"))

# ── Thought erosion strip ─────────────────────────────────────────────────
lbl(0.47, 0.925, "Thought generation → evidence signal erodes", C_FADE, fs=8.0, bold=False)
ev_xs_e = np.linspace(0.50, 0.93, 5)
AUROC_E = [0.949, 0.633, 0.602, 0.586, 0.586]
TIMING_E = ["+15", "—", "−4", "—", "−23"]
for ei, (xi, auroc, tim) in enumerate(zip(ev_xs_e, AUROC_E, TIMING_E)):
    col = C_EV if ei == 0 else C_FADE
    alpha_c = max(0.25, 0.9 - ei * 0.17)
    circ = plt.Circle((xi, 0.950), 0.019, transform=axA.transAxes,
                       color=col, alpha=alpha_c, zorder=6, clip_on=False)
    axA.add_patch(circ)
    axA.text(xi, 0.950, f"p{ei}", transform=axA.transAxes,
             ha="center", va="center", fontsize=6.0, color="white", fontweight="bold", zorder=7)
    axA.text(xi, 0.925, f"{auroc:.3f}", transform=axA.transAxes,
             ha="center", va="top", fontsize=6.2, color=col, zorder=7)
    if tim != "—":
        tc = "#16A34A" if "+" in tim else "#DC2626"
        axA.text(xi, 0.910, tim+" EM", transform=axA.transAxes,
                 ha="center", va="top", fontsize=5.8, color=tc, zorder=7, fontweight="bold")
for i in range(4):
    axA.annotate("", xy=(ev_xs_e[i+1]-0.020, 0.950), xytext=(ev_xs_e[i]+0.020, 0.950),
                 xycoords="axes fraction", textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="->", color=C_FADE, lw=1.1, alpha=0.55), zorder=6)

# ── Title + legend ────────────────────────────────────────────────────────
axA.text(0.50, 0.985, "Evidence-to-Action Routing Circuit", ha="center", va="top",
         fontsize=12.5, fontweight="bold", color="#111827", transform=axA.transAxes)
legend_items = [
    mpatches.Patch(facecolor=C_EV+"44",   edgecolor=C_EV,   label="Evidence subspace"),
    mpatches.Patch(facecolor=C_AC+"44",   edgecolor=C_AC,   label="Action subspace"),
    mpatches.Patch(facecolor=C_ATTN+"44", edgecolor=C_ATTN, label="attn L18  (routing)"),
    mpatches.Patch(facecolor=C_MLP+"44",  edgecolor=C_MLP,  label="mlp L20  (amplifier)"),
]
axA.legend(handles=legend_items, loc="lower left", fontsize=8.0,
           framealpha=0.92, edgecolor="#CBD5E1", fancybox=True,
           bbox_to_anchor=(0.00, 0.00), ncol=2)

# ═══════════════════════════════════════════════════════════════════════════
# PANEL B — Cross-model replication table
# ═══════════════════════════════════════════════════════════════════════════
axB.text(0.5, 0.975, "Cross-Model Replication", ha="center", va="top",
         fontsize=12, fontweight="bold", color="#111827", transform=axB.transAxes)
axB.text(0.5, 0.945, "(5 model families, same HotpotQA task)", ha="center", va="top",
         fontsize=8.5, color="#6B7280", transform=axB.transAxes, style="italic")

# Column headers
COL_X = [0.09, 0.32, 0.52, 0.70, 0.88]
COL_H = ["Model", "Ev\nAUROC", "|cos|\n(ev,ac)", "A/B\nratio", "Decom-\nposition"]
ROW_H = 0.87
for cx, ch in zip(COL_X, COL_H):
    axB.text(cx, ROW_H, ch, ha="center", va="center", fontsize=8.2,
             fontweight="bold", color="#374151", transform=axB.transAxes)

axB.plot([0.01, 0.99], [ROW_H - 0.035, ROW_H - 0.035], color="#9CA3AF", lw=1.0,
         transform=axB.transAxes)

# Rows
ROW_START = 0.80
ROW_STEP  = 0.14

for i, (name, nlayers, ev_l, ac_l, auroc, cos_ea, ab_ratio, ab_p, perp, confirmed) in enumerate(MODELS):
    ry = ROW_START - i * ROW_STEP
    # Row background
    bg_c = "#EFF6FF" if confirmed else "#FEF9EC"
    rect = FancyBboxPatch((0.01, ry - 0.058), 0.98, ROW_STEP - 0.01,
                          boxstyle="round,pad=0.005", linewidth=0,
                          facecolor=bg_c, zorder=1)
    axB.add_patch(rect)

    # Model name
    axB.text(COL_X[0], ry, name, ha="center", va="center", fontsize=7.8,
             fontweight="bold", color="#1E3A5F", transform=axB.transAxes)

    # Ev AUROC — colored bar behind number
    bar_w = 0.12 * auroc
    bar = FancyBboxPatch((COL_X[1] - 0.06, ry - 0.022), bar_w*2, 0.044,
                         boxstyle="round,pad=0.003", facecolor=C_EV+"33",
                         edgecolor="none", zorder=2)
    axB.add_patch(bar)
    axB.text(COL_X[1], ry, f"{auroc:.3f}", ha="center", va="center",
             fontsize=8.5, fontweight="bold", color=C_EV, transform=axB.transAxes)

    # |cos| (orthogonality)
    cos_str = f"{abs(cos_ea):.3f}"
    cos_col = C_OK if abs(cos_ea) < 0.04 else C_PART
    axB.text(COL_X[2], ry, cos_str, ha="center", va="center",
             fontsize=8.5, color=cos_col, transform=axB.transAxes)

    # A/B ratio + p
    ab_col = C_OK if confirmed else C_FAIL
    ab_str = f"×{ab_ratio:.2f}"
    axB.text(COL_X[3], ry + 0.018, ab_str, ha="center", va="center",
             fontsize=8.5, fontweight="bold", color=ab_col, transform=axB.transAxes)
    axB.text(COL_X[3], ry - 0.018, f"p={ab_p}", ha="center", va="center",
             fontsize=7.0, color=ab_col, alpha=0.85, transform=axB.transAxes)

    # Decomposition
    if perp is not None:
        d_txt = f"⊥/full\n={perp:.3f}"
        d_col = C_OK
    else:
        d_txt = "not\ntested"
        d_col = "#9CA3AF"
    axB.text(COL_X[4], ry, d_txt, ha="center", va="center",
             fontsize=7.5, color=d_col, transform=axB.transAxes,
             fontweight="bold" if perp is not None else "normal")

    # Confirmed badge
    badge_x = 0.96
    badge_c = C_OK if confirmed else C_FAIL
    badge_t = "✓" if confirmed else "✗"
    axB.text(badge_x, ry, badge_t, ha="center", va="center",
             fontsize=11, color=badge_c, fontweight="bold", transform=axB.transAxes)

# Divider line between models
for i in range(1, len(MODELS)):
    ry_div = ROW_START - i * ROW_STEP + ROW_STEP/2 - 0.008
    axB.plot([0.02, 0.97], [ry_div, ry_div], color="#E2E8F0", lw=0.8,
             transform=axB.transAxes)

# Legend for status
axB.add_patch(FancyBboxPatch((0.01, 0.01), 0.97, 0.092,
                             boxstyle="round,pad=0.01", facecolor="#F1F5F9",
                             edgecolor="#CBD5E1", linewidth=0.8, zorder=3))
axB.text(0.50, 0.077, "Panel columns", ha="center", va="center", fontsize=7.5,
         fontweight="bold", color="#475569", transform=axB.transAxes)
foot_items = [
    ("Ev AUROC", "Evidence probe at peak layer  (0-doc vs 1+-doc)"),
    ("|cos|", "Geometric orthogonality of evidence & action dirs"),
    ("A/B ratio", "Paired corruption: evidence swap / distractor swap shift"),
    ("⊥/full", "Decomp: causal effect in null(evidence) subspace  [Qwen/Gem/Mis]"),
]
for fi, (abbr, desc) in enumerate(foot_items):
    axB.text(0.04, 0.058 - fi * 0.014, f"• {abbr}:", ha="left", va="center",
             fontsize=6.8, fontweight="bold", color="#334155", transform=axB.transAxes)
    axB.text(0.22, 0.058 - fi * 0.014, desc, ha="left", va="center",
             fontsize=6.8, color="#475569", transform=axB.transAxes)

# ── Suptitle ──────────────────────────────────────────────────────────────
fig.text(0.50, 0.985,
         "Two-Stage Evidence→Action Circuit  ·  Qwen2.5-7B-Instruct  ·  HotpotQA bridge questions",
         ha="center", va="top", fontsize=11, color="#1E293B")

out = OUT / "fig3_routing_circuit.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")

