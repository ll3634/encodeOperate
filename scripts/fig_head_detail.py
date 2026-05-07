"""
fig_head_detail.py — Detailed head-level circuit node diagram (attention-style)

Layout:
  A (left half):  28-head heatmap matrix (obs attn × action proj)
  B (top-right):  Cross-position layer recovery bar (Phase E)
  C (mid-right):  KV-group causal recovery (Phase F3)
  D (bot-right):  Thought erosion AUROC curve (p0→p4)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import json, os

# ── Data ────────────────────────────────────────────────────────────────────
# F4: obs attention mass per head (clean)
OBS_CLEAN = [0.0428,0.0086,0.0870,0.0241,0.2122,0.1234,0.0356,  # KV0 H0-6
             0.0018,0.0783,0.0387,0.0609,0.0334,0.1105,0.2152,  # KV1 H7-13
             0.1066,0.3574,0.1125,0.0326,0.1801,0.1125,0.2315,  # KV2 H14-20
             0.1096,0.1492,0.0366,0.0922,0.0730,0.0413,0.1047]  # KV3 H21-27

OBS_CORRUPT = [0.0400,0.0052,0.0716,0.0198,0.2400,0.1197,0.0295,
               0.0004,0.0642,0.0353,0.0516,0.0234,0.1236,0.2136,
               0.1020,0.4202,0.1025,0.0231,0.2139,0.0988,0.2488,
               0.1292,0.1489,0.0444,0.0990,0.0902,0.0415,0.1063]

# F1: abs action_dir projection per head
ACTION_PROJ = [0.0270,0.0133,0.0085,0.0116,0.0476,0.0408,0.0086,
               0.0555,0.0190,0.0446,0.0160,0.0145,0.0232,0.0878,
               0.0355,0.1031,0.0235,0.0393,0.0745,0.0301,0.0785,
               0.0712,0.0167,0.0087,0.0316,0.0233,0.0466,0.0136]

OBS_DELTA = [c - o for c, o in zip(OBS_CORRUPT, OBS_CLEAN)]

N_HEADS = 28
KV_GROUPS = [0]*7 + [1]*7 + [2]*7 + [3]*7

# Phase E: cross-position recovery by layer
LAYER_E = [16, 18, 19, 20]
MEDIAN_E = [0.0071, 0.1873, 0.0234, 0.0146]

# Phase F3: KV group causal recovery (median)
KV_RECOVERY = [0.020, 0.046, 0.115, 0.045]
KV_LABELS = ['KV0\n(H0–6)', 'KV1\n(H7–13)', 'KV2\n(H14–20)', 'KV3\n(H21–27)']

# Thought erosion
POSITIONS = ['p0\n(input)', 'p1\n(25%)', 'p2\n(50%)', 'p3\n(75%)', 'p4\n(100%)']
FIXED_AUROC = [0.9488, 0.6327, 0.6017, 0.5861, 0.5860]
PROBE_AUROC = [0.8981, 0.7814, 0.6603, 0.6090, 0.7365]

KV_COLORS = ['#7bafd4','#a8d5a2','#e07b54','#b39ddb']  # KV0-3
HIGHLIGHT = '#e07b54'  # KV2

# ── Figure Layout ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#fafafa')
gs = gridspec.GridSpec(3, 2, figure=fig, left=0.06, right=0.97,
                       top=0.94, bottom=0.06, hspace=0.45, wspace=0.32,
                       height_ratios=[1.1, 1.0, 1.0])
ax_heat = fig.add_subplot(gs[:, 0])       # Panel A: head heatmap (all rows)
ax_layer = fig.add_subplot(gs[0, 1])      # Panel B: layer recovery
ax_kv    = fig.add_subplot(gs[1, 1])      # Panel C: KV group recovery
ax_eros  = fig.add_subplot(gs[2, 1])      # Panel D: erosion curve

# ── Panel A: Head Heatmap ────────────────────────────────────────────────
ax = ax_heat
ax.set_facecolor('#f0f0f0')

COLS = ['Obs-attn\n(clean)', 'Obs-attn\n(corrupt→ clean Δ)', 'Action-dir\nprojection']
DATA = [OBS_CLEAN, OBS_DELTA, ACTION_PROJ]
CMAPS = ['Blues', 'RdBu_r', 'Oranges']
VMINS = [0, -0.05, 0]
VMAXS = [0.40, 0.10, 0.11]

n_cols = len(COLS)
cell_w = 0.7
cell_h = 0.9
xs = [1.0, 2.2, 3.4]  # x center for each column

# KV group background bands
kv_spans = [(0,6,'KV0'), (7,13,'KV1'), (14,20,'KV2'), (21,27,'KV3')]
for kv_idx, (start, end, label) in enumerate(kv_spans):
    y0 = -(end+0.5)*cell_h
    y1 = -(start-0.5)*cell_h
    fc = KV_COLORS[kv_idx] + '28'  # semi-transparent
    ax.fill_between([0.4, 4.2], [y0,y0], [y1,y1], color=KV_COLORS[kv_idx], alpha=0.12, zorder=0)
    ax.text(0.2, (y0+y1)/2, label, ha='center', va='center', fontsize=9,
            fontweight='bold', color=KV_COLORS[kv_idx], rotation=90)

# Draw cells
for col_idx, (col_data, cmap_name, vmin, vmax) in enumerate(zip(DATA, CMAPS, VMINS, VMAXS)):
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)
    xcen = xs[col_idx]
    for head_idx, val in enumerate(col_data):
        ycen = -head_idx * cell_h
        kv = KV_GROUPS[head_idx]
        color = cmap(norm(val))
        rect = mpatches.FancyBboxPatch(
            (xcen - cell_w/2, ycen - cell_h/2), cell_w, cell_h,
            boxstyle="round,pad=0.03", linewidth=0.4,
            edgecolor='#cccccc' if kv != 2 else '#cc4400',
            facecolor=color, zorder=2)
        ax.add_patch(rect)
        txt_color = 'white' if norm(val) > 0.6 else '#222222'
        ax.text(xcen, ycen, f'{val:.3f}', ha='center', va='center',
                fontsize=6.5, color=txt_color, zorder=3)

# Head labels on left
for head_idx in range(N_HEADS):
    ycen = -head_idx * cell_h
    kv = KV_GROUPS[head_idx]
    fw = 'bold' if kv == 2 else 'normal'
    fc = HIGHLIGHT if kv == 2 else '#444444'
    ax.text(0.55, ycen, f'H{head_idx:02d}', ha='right', va='center',
            fontsize=7.5, fontweight=fw, color=fc)

# Column headers
for col_idx, col_name in enumerate(COLS):
    ax.text(xs[col_idx], 1.0, col_name, ha='center', va='bottom',
            fontsize=9.5, fontweight='bold', color='#333333')

# KV2 bracket
kv2_y0 = -20.5*cell_h; kv2_y1 = -14.0*cell_h + cell_h*0.5  # wrong, fix below
kv2_ystart = -14*cell_h - cell_h/2; kv2_yend = -20*cell_h + cell_h/2
ax.annotate('', xy=(4.35, kv2_yend), xytext=(4.35, kv2_ystart),
            arrowprops=dict(arrowstyle='<->', color=HIGHLIGHT, lw=2))
ax.text(4.5, (kv2_ystart+kv2_yend)/2, 'KV Group 2\n(H14–H20)', 
        ha='left', va='center', fontsize=9, color=HIGHLIGHT, fontweight='bold')

# Special annotation for H15
h15y = -15*cell_h
ax.annotate('H15: obs=35.7%\nmax_proj=0.103',
            xy=(xs[0]+cell_w/2, h15y), xytext=(4.9, h15y+0.5),
            fontsize=8, color='#882200', ha='left',
            arrowprops=dict(arrowstyle='->', color='#882200', lw=1.2))

ax.set_xlim(0.1, 5.2)
ax.set_ylim(-27.5*cell_h, 1.8)
ax.axis('off')
ax.set_title('Layer 18 — All 28 Attention Heads\n(Qwen2.5-7B-Instruct, GQA 4 KV-groups × 7 heads)',
             fontsize=11, fontweight='bold', pad=8)

# ── Panel B: Cross-position layer recovery (Phase E) ─────────────────────
ax = ax_layer
bars = ax.bar(range(len(LAYER_E)), MEDIAN_E,
              color=['#7bafd4','#e07b54','#aaaaaa','#aaaaaa'],
              width=0.5, edgecolor='white', linewidth=0.8)
ax.axhline(0.02, color='#999999', ls='--', lw=1.0, label='KV0 baseline (0.020)')
for b, v in zip(bars, MEDIAN_E):
    ax.text(b.get_x()+b.get_width()/2, v+0.004, f'{v:.3f}',
            ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax.set_xticks(range(len(LAYER_E)))
ax.set_xticklabels([f'L{l}' for l in LAYER_E], fontsize=10)
ax.set_ylabel('Median causal recovery', fontsize=9)
ax.set_title('Phase E — Cross-position patching\n(obs→dec, per layer)', fontsize=10, fontweight='bold')
ax.set_ylim(0, 0.28)
ax.legend(fontsize=8)
ax.spines[['top','right']].set_visible(False)
ax.set_facecolor('#fafafa')
# annotate: routing complete by L19
ax.annotate('routing complete\nbefore L19', xy=(2, 0.0234), xytext=(2.5, 0.12),
            arrowprops=dict(arrowstyle='->', color='#666666', lw=1.0),
            fontsize=8, color='#666666', ha='center')

# ── Panel C: KV group causal recovery (Phase F3) ─────────────────────────
ax = ax_kv
bars = ax.bar(range(4), KV_RECOVERY,
              color=[KV_COLORS[i] for i in range(4)],
              width=0.5, edgecolor='white', linewidth=0.8)
for b, v in zip(bars, KV_RECOVERY):
    ax.text(b.get_x()+b.get_width()/2, v+0.003, f'{v:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(range(4))
ax.set_xticklabels(KV_LABELS, fontsize=9)
ax.set_ylabel('Median causal recovery', fontsize=9)
ax.set_title('Phase F3 — KV-group cross-pos patching\n(attn_L18, obs→dec)', fontsize=10, fontweight='bold')
ax.set_ylim(0, 0.18)
ax.spines[['top','right']].set_visible(False)
ax.set_facecolor('#fafafa')
# add multiplier labels
for i, v in enumerate(KV_RECOVERY):
    mult = v / KV_RECOVERY[0]
    if i > 0:
        ax.text(i, v + 0.013, f'×{mult:.1f}',
                ha='center', va='bottom', fontsize=8.5, color='#444444')
ax.annotate('', xy=(2, KV_RECOVERY[2]-0.005), xytext=(2, KV_RECOVERY[2]+0.018),
            arrowprops=dict(arrowstyle='->', color=HIGHLIGHT, lw=1.5))

# ── Panel D: Thought erosion (AUROC p0→p4) ───────────────────────────────
ax = ax_eros
xs_eros = range(5)
ax.plot(xs_eros, FIXED_AUROC, 'o-', color='#2171b5', lw=2, ms=8, label='Fixed-dir AUROC')
ax.plot(xs_eros, PROBE_AUROC, 's--', color='#6baed6', lw=1.5, ms=7, label='Retrained-probe AUROC')
ax.axhline(0.5, color='#bbbbbb', ls=':', lw=1.2, label='Random (0.50)')

for i, (y1, y2) in enumerate(zip(FIXED_AUROC, PROBE_AUROC)):
    ax.text(i, y1+0.015, f'{y1:.3f}', ha='center', va='bottom', fontsize=8, color='#2171b5')

# Timing intervention net gain annotations
TIMING = [(0,'+15'),(2,'−4'),(4,'−23')]
for xi, label in TIMING:
    yp = FIXED_AUROC[xi]
    color = '#2ca02c' if '+' in label else '#d62728'
    ax.annotate(f'steer: {label}', xy=(xi, yp), xytext=(xi+0.35, yp-0.05),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.0),
                fontsize=8, color=color, ha='left')

ax.set_xticks(xs_eros)
ax.set_xticklabels(POSITIONS, fontsize=9)
ax.set_ylabel('AUROC (evidence readability)', fontsize=9)
ax.set_ylim(0.40, 1.05)
ax.set_title('Thought erosion — Evidence readability decay\nduring Thought generation (p0=before, p4=end)',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.spines[['top','right']].set_visible(False)
ax.set_facecolor('#fafafa')

# ── Overall title + save ─────────────────────────────────────────────────
fig.suptitle('Circuit Mechanism: L18 Head-Level Routing Analysis\n'
             'Qwen2.5-7B-Instruct · Phase D/E/F (activation patching) · Phase F3/F4 (head-level)',
             fontsize=13, fontweight='bold', y=0.98)

out = 'results/fig_head_detail.pdf'
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
out_png = out.replace('.pdf', '.png')
fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved: {out}")
print(f"Saved: {out_png}")
plt.close(fig)
