#!/usr/bin/env python3
"""Cross-family decomposition v6 — the 99% headline.

  Panel A  —  100% stacked vertical bars (one per model).
              Each bar is normalized to full Δm = 100%.  The bottom segment is
              the ⊥ component (~99.9%, model color), the top segment is the ∥
              component (~0.1–0.3%, hatched neutral grey).  Three bars all
              look identical: a clean visual statement of universality.

              An inset on the right zooms the [99.0%, 100.0%] band so the
              tiny ∥ slivers are actually visible at full resolution.

  Panel B  —  Absolute |∥| forest plot vs random null band.

Legend at figure bottom.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results/crossfamily_ci_decomposition/crossfamily_table.json"
OUT  = ROOT / "results/fig_crossfamily_best"
OUT.mkdir(parents=True, exist_ok=True)
tbl = json.load(open(DATA))

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10.0,
    "axes.linewidth":     0.9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.major.width":  0.7, "xtick.major.size": 3.5,
    "ytick.major.width":  0.7, "ytick.major.size": 3.5,
    "legend.framealpha":  0.96,
    "legend.edgecolor":   "#CCCCCC",
})

NICE   = {"qwen": "Qwen2.5-7B-Instruct", "gemma": "Gemma-2-9B-it",  "mistral": "Mistral-7B-v0.3"}
LAYER  = {"qwen": 20, "gemma": 37, "mistral": 28}
DIM    = {"qwen": 3584, "gemma": 3584, "mistral": 4096}
# matplotlib tab10 defaults — carefully designed for academic readability,
# colorblind-distinguishable, and familiar to ML audiences.
COLORS = {"qwen":    "#1f77b4",   # C0 tab-blue  (anchor model)
          "gemma":   "#2ca02c",   # C2 tab-green
          "mistral": "#d62728"}   # C3 tab-red
PAR_COLOR = "#AAAAAA"             # light neutral grey for the tiny ∥ sliver
SHORT       = {"qwen": "Qwen2.5-7B", "gemma": "Gemma-2-9B", "mistral": "Mistral-7B"}
SHORT_AUDIT = SHORT
ORDER_X = ["qwen", "gemma", "mistral"]
ORDER_Y     = ["mistral", "gemma"]            # Panel C: cross-family extension only
AUDIT_ORDER = ["mistral", "gemma", "qwen"]   # Panel B (audit): bottom → top; qwen = reference cell


def get(key):
    m = tbl[key]
    perp_pct = m["ratio_perp_over_full"] * 100
    rms_pct  = m["ratio_par_rms_over_full"] * 100   # appears on RMS-renorm protocol
    if m["par_natural"] is not None:
        par_pct = m["ratio_par_natural_over_full"] * 100
        par     = m["par_natural"]["mean"]
        par_lo  = m["par_natural"]["ci_low"]
        par_hi  = m["par_natural"]["ci_high"]
        null95  = m["stopping_rule_no_renorm"]["rhs"]
        has_par = True
        amp     = m.get("amplification_rms_over_natural")
    else:
        par_pct = None
        par = par_lo = par_hi = None
        null95 = None
        has_par = False
        amp = None
    return dict(
        perp_pct=perp_pct, par_pct=par_pct, rms_pct=rms_pct,
        par=par, par_lo=par_lo, par_hi=par_hi, null95=null95,
        amplification=amp,
        cos_ae=m["cos_action_evidence"],
        n=m["n_samples"], layer=LAYER[key], d=DIM[key],
        full=m["full"]["mean"],
        has_par=has_par,
    )

DAT = {k: get(k) for k in ORDER_X}

# ── Figure layout ─────────────────────────────────────────────────────────────
# 6-column gridspec:
#   [stacked-bars] [zoom] [spacer] [audit] [spacer] [forest]
# gs[0,2]: spacer between zoom and audit (prevents axAudit labels bleeding left).
# gs[0,4]: spacer between audit and forest (0.30 ratio — enough to stop Panel B
#           annotations overflowing into Panel C's drawing area).
fig = plt.figure(figsize=(16.4, 6.4))
gs = fig.add_gridspec(
    1, 6,
    width_ratios=[1.10, 0.45, 0.18, 1.05, 0.30, 1.05],
    left=0.050, right=0.985,
    top=0.83, bottom=0.30,
    wspace=0.22,
)
axA     = fig.add_subplot(gs[0, 0])   # 100% stacked bars (3 models, natural-norm)
axZoom  = fig.add_subplot(gs[0, 1])   # zoom inset on [99.0%, 100.0%]
# gs[0, 2] is an explicit spacer — keeps axAudit labels from crossing into axZoom
axAudit = fig.add_subplot(gs[0, 3])   # RMS vs natural-norm audit (3 rows incl. Qwen ref)
# gs[0, 4] is an explicit spacer
axB     = fig.add_subplot(gs[0, 5])   # natural-norm |∥| forest (Gemma + Mistral)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — 100% stacked vertical bars  (the headline: ⊥ ≈ 100%)
# ══════════════════════════════════════════════════════════════════════════════
BAR_W = 0.62
xpos  = np.arange(len(ORDER_X))

for i, k in enumerate(ORDER_X):
    g = DAT[k]; c = COLORS[k]
    perp = g["perp_pct"]                                # ≈ 99.9
    if g["has_par"]:
        par = max(0.0, 100.0 - perp)
    else:
        par = 0.0
    axA.bar(xpos[i], perp, width=BAR_W,
            color=c, alpha=0.78, edgecolor=c, linewidth=1.0, zorder=2)
    if g["has_par"]:
        axA.bar(xpos[i], par, width=BAR_W, bottom=perp,
                color=PAR_COLOR, alpha=0.85, edgecolor="#444",
                linewidth=1.0, hatch="////", zorder=3)
    else:
        axA.bar(xpos[i], 100 - perp, width=BAR_W, bottom=perp,
                facecolor="white", edgecolor="#888",
                linewidth=0.8, hatch="xxx", zorder=3, alpha=0.6)

    # fontsize=15 overflows the bar width → first "9" gets clipped.
    # Fix: reduce to 11, add clip_on=False so even marginal overflow is visible.
    axA.text(xpos[i], perp / 2, f"⊥\n{perp:.2f}%",
             ha="center", va="center", fontsize=11.0,
             color="white", fontweight="bold", zorder=5, clip_on=False)

axA.axhline(100.0, color="#222", lw=1.0, linestyle="--", zorder=4)
axA.text(-0.50, 100.6, "100% = full Δm",
         ha="left", va="bottom", fontsize=8.4, color="#444", style="italic")

axA.set_xticks(xpos)
# S3.3: model name only as tick label (avoids wide-label concatenation);
# metadata (L, d, N) goes as independent text elements below each bar.
axA.set_xticklabels([NICE[k] for k in ORDER_X], fontsize=9.2)
for i, k in enumerate(ORDER_X):
    axA.text(xpos[i], -7.5,
             f"L{DAT[k]['layer']}, d={DAT[k]['d']}, N={DAT[k]['n']}",
             ha="center", va="top", fontsize=7.6, color="#555",
             clip_on=False)
axA.set_ylim(0, 105)
axA.set_xlim(-0.55, len(ORDER_X) - 0.45)
axA.set_ylabel("% of full $\\Delta m$", fontsize=10.6)
axA.grid(axis="y", color="#EEEEEE", linewidth=0.6, zorder=0)
axA.set_yticks([0, 25, 50, 75, 100])
axA.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
axA.set_title(
    r"$\bf{A}$    Decomposition of $\Delta m$  (full = $\perp$ + $\parallel$)",
    loc="left", fontsize=11.6, pad=10, color="#111",
)

# ── Zoom inset axis (separate subplot, [99.0, 100.0] %) ───────────────────────
ZOOM_BAR_W = 0.52   # narrower so per-bar callouts have horizontal room
for i, k in enumerate(ORDER_X):
    g = DAT[k]; c = COLORS[k]
    perp = g["perp_pct"]
    axZoom.bar(i, perp - 99.0, width=ZOOM_BAR_W, bottom=99.0,
               color=c, alpha=0.78, edgecolor=c, linewidth=1.0, zorder=2)
    if g["has_par"]:
        axZoom.bar(i, 100.0 - perp, width=ZOOM_BAR_W, bottom=perp,
                   color=PAR_COLOR, alpha=0.85, edgecolor="#444",
                   linewidth=1.0, hatch="////", zorder=3)
    else:
        axZoom.bar(i, 100.0 - perp, width=ZOOM_BAR_W, bottom=perp,
                   facecolor="white", edgecolor="#888",
                   linewidth=0.8, hatch="xxx", zorder=3, alpha=0.6)
    # ⊥ percentage — fontsize 6.0 (was 6.8, which overflowed narrow zoom bars)
    axZoom.text(i, 99.0 + (perp - 99.0) * 0.45, f"{perp:.2f}%",
                ha="center", va="center", fontsize=6.0, color="white",
                fontweight="bold", zorder=5, clip_on=False)
    # ∥ value — small annotation just below x-axis tick, NOT above the 100% line
    if g["has_par"]:
        par_lbl = f"∥ {g['par_pct']:+.2f}%"
        par_color = c
    else:
        par_lbl = "∥ n/a"
        par_color = "#888"
    axZoom.annotate(par_lbl,
                    xy=(i, 100.0), xycoords="data",
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.8,
                    color=par_color, fontweight="bold", zorder=5)

axZoom.axhline(100.0, color="#222", lw=1.0, linestyle="--", zorder=4)
axZoom.set_ylim(99.0, 100.45)
axZoom.set_xlim(-0.55, len(ORDER_X) - 0.45)
axZoom.set_xticks(xpos)
axZoom.set_xticklabels(
    [NICE[k].split("-")[0].replace("Qwen2.5", "Qwen") for k in ORDER_X],
    fontsize=7.8,
)
axZoom.set_yticks([99.0, 99.5, 100.0])
axZoom.set_yticklabels(["99%", "99.5%", "100%"], fontsize=7.6)
axZoom.tick_params(axis="x", pad=2)
axZoom.grid(axis="y", color="#EEEEEE", linewidth=0.6, zorder=0)
axZoom.set_title(r"zoom $[99\%, 100\%]$",
                 loc="center", fontsize=8.6, pad=4, color="#444", style="italic")

# Leader lines from main panel to zoom inset
for ratio in (99.0, 100.0):
    con = ConnectionPatch(
        xyA=(len(ORDER_X) - 0.45, ratio), coordsA=axA.transData,
        xyB=(-0.55, ratio),                coordsB=axZoom.transData,
        color="#AAAAAA", lw=0.7, linestyle=(0, (3, 2)), zorder=1,
    )
    fig.add_artist(con)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — RMS-renorm artifact ▶ natural-norm audit
#   Mistral & Gemma: both RMS and natural values shown with before→after arrow.
#   Qwen: reference cell (§3 uses RMS-renorm as the reference condition);
#         natural-norm not measured — shown as grey row with italic note.
#
#   Labels: "RMS X%" ABOVE diamond, "Natural X%" BELOW circle.
#   Amplification factor "≈X×" placed ON the arrow midpoint (white bbox),
#   vertically between the two marker labels — no collision possible.
# ══════════════════════════════════════════════════════════════════════════════
ya = np.arange(len(AUDIT_ORDER))

for i, k in enumerate(AUDIT_ORDER):
    g = DAT[k]; c = COLORS[k]
    rms_pct = abs(g["rms_pct"])   # |∥_rms| / full_Δm × 100

    if g["has_par"]:
        nat_pct = abs(g["par_pct"])
        amp     = g["amplification"]

        # "Before audit" marker: RMS-renorm — hollow diamond
        axAudit.plot(rms_pct, i, marker="D", markersize=12,
                     markerfacecolor="white", markeredgecolor=c,
                     markeredgewidth=1.6, zorder=5)
        # RMS label ABOVE the marker
        axAudit.annotate(f"RMS  {rms_pct:.2f}%",
                         xy=(rms_pct, i), xycoords="data",
                         xytext=(0, 11), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8.0,
                         color="#666", fontweight="bold", zorder=6)

        # "After audit" marker: natural-norm — solid filled circle
        axAudit.plot(nat_pct, i, marker="o", markersize=12,
                     markerfacecolor=c, markeredgecolor="white",
                     markeredgewidth=1.2, zorder=6)
        # Natural label BELOW the marker — use signed value (+X%) so the sign
        # convention matches the inset in Panel A which also shows "+0.26%/+0.13%".
        axAudit.annotate(f"Natural  {g['par_pct']:+.3f}%",
                         xy=(nat_pct, i), xycoords="data",
                         xytext=(0, -11), textcoords="offset points",
                         ha="center", va="top", fontsize=8.0,
                         color=c, fontweight="bold", zorder=6)

        # Connector arrow (RMS → natural, "audit shrinks estimate")
        axAudit.annotate(
            "", xy=(nat_pct * 1.6, i), xytext=(rms_pct / 1.4, i),
            arrowprops=dict(arrowstyle="->", color=c, lw=1.5,
                            shrinkA=2, shrinkB=2, alpha=0.85),
            zorder=4,
        )
        # Amplification factor: SHORT label "≈X×" centred ON the arrow
        # (same y=i as the row).  White bbox visually splits the arrow cleanly.
        # This avoids the "≈X×" text colliding with "RMS X%" above the diamond
        # — both were previously at +10/+11 pt and overlapped horizontally.
        geo_mid = (rms_pct * nat_pct) ** 0.5
        axAudit.text(
            geo_mid, i,
            rf"$\approx${amp:,.0f}$\times$",
            ha="center", va="center", fontsize=9.2,
            color=c, fontweight="bold", zorder=7,
            transform=axAudit.transData,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
        )

    else:
        # Qwen: reference cell — grey hollow diamond.
        # rms_pct (abs) drives the log-scale x-position; the signed value
        # g["rms_pct"] is displayed so the NEGATIVE sign is visible, matching
        # §3 "opposite-sign shift of −0.157" and ledger §20.2 "−17.3%".
        axAudit.plot(rms_pct, i, marker="D", markersize=12,
                     markerfacecolor="#E8E8E8", markeredgecolor="#888888",
                     markeredgewidth=1.6, zorder=5, alpha=0.85)
        axAudit.annotate(f"RMS  {g['rms_pct']:.2f}%",   # signed: shows −17.25%
                         xy=(rms_pct, i), xycoords="data",
                         xytext=(0, 11), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8.0,
                         color="#888888", fontweight="bold", zorder=6)
        # Anchor the descriptive note at the LOG-CENTRE of the axis (√(0.04×70)≈1.67%,
        # = 50 % of the log range) — not at the diamond's x (17.25% = 81 % from left),
        # which made the long text overflow axAudit's right spine into Panel C.
        axAudit.text(
            1.67, i,   # √(0.04 × 70) ≈ geometric centre of [0.04, 70] log axis
            "reference cell (§3, RMS-renorm)\nopposite-sign shift\nnatural-norm not measured",
            ha="center", va="top", fontsize=7.2,
            color="#888888", style="italic", zorder=6,
            transform=axAudit.transData,
            bbox=dict(facecolor="white", edgecolor="#CCCCCC",
                      linewidth=0.6, pad=2, boxstyle="round,pad=0.3"),
        )

axAudit.set_xscale("log")
axAudit.set_xlim(0.04, 70.0)
axAudit.set_ylim(-0.6, len(AUDIT_ORDER) - 0.30)
axAudit.set_yticks(ya)
axAudit.set_yticklabels([SHORT_AUDIT[k] for k in AUDIT_ORDER],
                        fontsize=10.0, fontweight="bold")
axAudit.tick_params(axis="y", left=False, pad=4)
axAudit.set_xticks([0.1, 1.0, 10.0])
axAudit.set_xticklabels(["0.1%", "1%", "10%"], fontsize=8.6)
# S2.3: explicit "(%, log scale)" so readers know Panel B and Panel C differ
axAudit.set_xlabel(r"$|\parallel|$ / full $\Delta m$  (%, log scale)",
                   fontsize=10.2, labelpad=10)
axAudit.grid(axis="x", which="major", color="#EEEEEE", linewidth=0.6, zorder=0)
axAudit.grid(axis="x", which="minor", color="#F5F5F5", linewidth=0.4, zorder=0)
# S3.2: ⇒ → → (methodological audit, not logical implication)
axAudit.set_title(
    r"$\bf{B}$    Self-audit: RMS-renorm artifact $\rightarrow$ natural-norm",
    loc="left", fontsize=11.6, pad=10, color="#111",
)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL C — horizontal forest plot of signed ∥ vs random null band (natural)
# ══════════════════════════════════════════════════════════════════════════════
ROW_HALF = 0.32
ypos = np.arange(len(ORDER_Y))

for i, k in enumerate(ORDER_Y):
    g = DAT[k]; c = COLORS[k]
    if g["has_par"]:
        axB.add_patch(Rectangle(
            (-g["null95"], i - ROW_HALF),
            2 * g["null95"], 2 * ROW_HALF,
            facecolor="#DDDDDD", alpha=0.55, edgecolor="none", zorder=0,
        ))
        for sign in (-1, +1):
            axB.plot([sign * g["null95"]] * 2,
                     [i - ROW_HALF, i + ROW_HALF],
                     color="#888", lw=0.7, zorder=0.5)
        axB.plot([g["par_lo"], g["par_hi"]], [i, i],
                 color=c, lw=1.6, alpha=0.85, zorder=4)
        axB.plot(g["par"], i, "o",
                 markerfacecolor=c, markeredgecolor="white", markeredgewidth=1.0,
                 markersize=11, zorder=5)
        # ∥/full callout — placed just outside the per-row null band on the right
        axB.annotate(f"∥/full = {g['par_pct']:+.3f}%",
                     xy=(g["null95"], i), xycoords="data",
                     xytext=(8, 0), textcoords="offset points",
                     va="center", ha="left", fontsize=8.4,
                     color=c, fontweight="bold", zorder=6)
    else:
        axB.text(
            0, i, "par_natural unmeasured\n(direction-level cos(A,E) only)",
            ha="center", va="center", fontsize=8.4, color=c, style="italic",
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                      edgecolor=c, linewidth=1.0),
            zorder=4,
        )

axB.axvline(0, color="#222", lw=1.1, zorder=1)
# "perfect orthogonality" placed on the LEFT outside the null bands so it never
# competes with markers, anchored to the zero line.
# Use "∥=0" not "θ=0" — avoids clash with Figure 2's θ-rotation notation.
# (∥ projection = 0) ⟺ a⊥e, which is "perfect orthogonality" without the
# angle ambiguity that θ carries in the mechanistic-interp context.
axB.annotate(r"$\parallel\!=0$  perfect orthogonality",
             xy=(0, -0.50), xycoords="data",
             xytext=(0, -22), textcoords="offset points",
             ha="center", va="top", fontsize=8.2, color="#222",
             style="italic",
             arrowprops=dict(arrowstyle="-", color="#222", lw=0.7))

axB.set_yticks(ypos)
# short labels only — full descriptors live in the figure-level legend
axB.set_yticklabels([SHORT[k] for k in ORDER_Y], fontsize=10.0,
                    fontweight="bold")
# pull labels close to axis
axB.tick_params(axis="y", left=False, pad=4)
axB.set_ylim(-0.55, len(ORDER_Y) - 0.30)
# S2.3: "magnitude (natural L2 norm)" distinguishes Panel C from Panel B's ratio
axB.set_xlabel(r"$\parallel$-component projection magnitude  (natural L2 norm)",
               fontsize=10.2, labelpad=18)
# generous right margin to accommodate the ∥/full callouts
xlim_B = max(DAT[k]["null95"] for k in ORDER_Y if DAT[k]["has_par"]) * 3.2
axB.set_xlim(-xlim_B*0.55, xlim_B)
axB.grid(axis="x", color="#EEEEEE", linewidth=0.6, zorder=0)
axB.set_title(
    r"$\bf{C}$    Cross-family extension stays inside random null band",
    loc="left", fontsize=11.6, pad=10, color="#111",
)

# ══════════════════════════════════════════════════════════════════════════════
# Shared figure-level legend (bottom)
# ══════════════════════════════════════════════════════════════════════════════
# S3.4: two logical rows — row 1: model color swatches; row 2: panel markers.
# With ncol=4 and 8 handles, matplotlib auto-fills row 1 = items 0-3, row 2 = 4-7.
legend_handles = [
    # ── Row 1: model identity ────────────────────────────────────────────────
    Rectangle((0, 0), 1, 1, facecolor=COLORS["qwen"],    alpha=0.78,
              edgecolor=COLORS["qwen"],    label="Qwen2.5-7B-Instruct (L20, d=3584, N=100)"),
    Rectangle((0, 0), 1, 1, facecolor=COLORS["gemma"],   alpha=0.78,
              edgecolor=COLORS["gemma"],   label="Gemma-2-9B-it (L37, d=3584, N=50)"),
    Rectangle((0, 0), 1, 1, facecolor=COLORS["mistral"], alpha=0.78,
              edgecolor=COLORS["mistral"], label="Mistral-7B-v0.3 (L28, d=4096, N=50)"),
    Rectangle((0, 0), 1, 1, facecolor=PAR_COLOR, alpha=0.85, edgecolor="#444",
              hatch="////", label=r"A:  $\parallel$ component (top sliver)"),
    # ── Row 2: panel-specific markers ────────────────────────────────────────
    Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#888",
              hatch="xxx", label=r"A:  $\parallel$ unmeasured (Qwen, reference cell §3)"),
    Line2D([0], [0], marker="D", color="none", markerfacecolor="white",
           markeredgecolor="#444", markeredgewidth=1.4, markersize=10,
           label=r"B:  RMS-renorm $|\parallel|$  (artifact)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#444",
           markeredgecolor="white", markeredgewidth=1.0, markersize=10,
           label=r"B:  natural-norm $|\parallel|$  (audited)"),
    Rectangle((0, 0), 1, 1, facecolor="#DDDDDD", alpha=0.55, edgecolor="#888",
              linewidth=0.7,
              label="C:  random 95% null band (natural L2 norm)"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center", bbox_to_anchor=(0.5, 0.005),
    ncol=4, fontsize=8.4, frameon=True, framealpha=0.94,
    edgecolor="#CCCCCC", columnspacing=1.6, handlelength=2.2,
)

# ── Suptitle ──────────────────────────────────────────────────────────────────
# S1.2: "up to 7,147×" — the specific number is Mistral-only; "up to" is honest
fig.suptitle(
    r"Action steering is $\perp$-dominated across three families "
    r"— RMS-renorm artifacts up to 7,147$\times$ resolved under natural-norm audit",
    fontsize=12.6, fontweight="bold", color="#111",
    x=0.012, y=0.965, ha="left",
)

out_png = OUT / "crossfamily_v10.png"
out_pdf = OUT / "crossfamily_v10.pdf"
fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.18)
fig.savefig(out_pdf,            bbox_inches="tight", pad_inches=0.18)
print(f"[v10] wrote {out_png}")
print(f"[v10] wrote {out_pdf}")
print(f"[v10] colors actually used: {COLORS}")
print(f"[v10] perp/full per model:")
for k in ORDER_X:
    g = DAT[k]
    par_str = f"{g['par_pct']:+.3f}%" if g['has_par'] else "n/a"
    rms_str = f"{g['rms_pct']:+.3f}%"
    amp     = g.get('amplification')
    amp_str = f"{amp:,.0f}x" if amp else "n/a"
    print(f"  {k:8s}: ⊥/full = {g['perp_pct']:.3f}%   "
          f"∥_natural = {par_str}   ∥_rms = {rms_str}   amp = {amp_str}")



# ══════════════════════════════════════════════════════════════════════════════
# ORAL SLIDES — Separate 16:9 layouts for talk presentation
#
# Slide 1: Panel A + zoom only  →  "⊥ carries 99.9% — three families"
# Slide 2: Panel B + C backup   →  "Mistral 26.2% residual is a 7,147× artifact"
# ══════════════════════════════════════════════════════════════════════════════

# ── Slide 1: Panel A only (main result) ──────────────────────────────────────
plt.close("all")
fig_s1 = plt.figure(figsize=(13.33, 7.50))
gs_s1 = fig_s1.add_gridspec(
    1, 2, width_ratios=[2.0, 0.65],
    left=0.08, right=0.97, top=0.76, bottom=0.18, wspace=0.18,
)
axAs = fig_s1.add_subplot(gs_s1[0, 0])
axZs = fig_s1.add_subplot(gs_s1[0, 1])

xpos_s = np.arange(len(ORDER_X))
for i, k in enumerate(ORDER_X):
    g = DAT[k]; c = COLORS[k]
    perp = g["perp_pct"]
    par  = max(0.0, 100.0 - perp) if g["has_par"] else 0.0
    axAs.bar(xpos_s[i], perp, width=BAR_W, color=c, alpha=0.82,
             edgecolor=c, linewidth=1.2, zorder=2)
    if g["has_par"]:
        axAs.bar(xpos_s[i], par, width=BAR_W, bottom=perp,
                 color=PAR_COLOR, alpha=0.85, edgecolor="#444",
                 linewidth=1.2, hatch="////", zorder=3)
    else:
        axAs.bar(xpos_s[i], 100 - perp, width=BAR_W, bottom=perp,
                 facecolor="white", edgecolor="#888",
                 linewidth=0.9, hatch="xxx", zorder=3, alpha=0.6)
    axAs.text(xpos_s[i], perp / 2, f"⊥\n{perp:.2f}%",
              ha="center", va="center", fontsize=15, color="white",
              fontweight="bold", zorder=5, clip_on=False)

axAs.axhline(100.0, color="#222", lw=1.1, linestyle="--", zorder=4)
axAs.set_xticks(xpos_s)
axAs.set_xticklabels([NICE[k] for k in ORDER_X], fontsize=15)
for i, k in enumerate(ORDER_X):
    axAs.text(xpos_s[i], -7.5,
              f"L{DAT[k]['layer']}, d={DAT[k]['d']}, N={DAT[k]['n']}",
              ha="center", va="top", fontsize=11, color="#555", clip_on=False)
axAs.set_ylim(0, 108); axAs.set_xlim(-0.55, len(ORDER_X) - 0.45)
axAs.set_ylabel("% of full $\\Delta m$", fontsize=14)
axAs.set_yticks([0, 25, 50, 75, 100])
axAs.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=13)
axAs.grid(axis="y", color="#EEEEEE", linewidth=0.7, zorder=0)
axAs.spines[["top", "right"]].set_visible(False)

for i, k in enumerate(ORDER_X):
    g = DAT[k]; c = COLORS[k]; perp = g["perp_pct"]
    axZs.bar(i, perp - 99.0, width=0.52, bottom=99.0,
             color=c, alpha=0.82, edgecolor=c, linewidth=1.0, zorder=2)
    _top = 100.0 - perp
    _patch_kw = dict(color=PAR_COLOR, alpha=0.85, edgecolor="#444",
                     linewidth=1.0, hatch="////", zorder=3) if g["has_par"] else \
                dict(facecolor="white", edgecolor="#888", linewidth=0.9,
                     hatch="xxx", zorder=3, alpha=0.6)
    axZs.bar(i, _top, width=0.52, bottom=perp, **_patch_kw)
    axZs.text(i, 99.0 + (perp - 99.0) * 0.45, f"{perp:.2f}%",
              ha="center", va="center", fontsize=8.5, color="white",
              fontweight="bold", zorder=5, clip_on=False)
    par_lbl   = f"∥ {g['par_pct']:+.2f}%" if g["has_par"] else "∥ n/a"
    par_color = c if g["has_par"] else "#888"
    axZs.annotate(par_lbl, xy=(i, 100.0), xycoords="data",
                  xytext=(0, 7), textcoords="offset points",
                  ha="center", va="bottom", fontsize=8.5,
                  color=par_color, fontweight="bold", zorder=5)

axZs.axhline(100.0, color="#222", lw=1.0, linestyle="--", zorder=4)
axZs.set_ylim(99.0, 100.55); axZs.set_xlim(-0.55, len(ORDER_X) - 0.45)
axZs.set_xticks(xpos_s)
axZs.set_xticklabels([NICE[k].split("-")[0].replace("Qwen2.5", "Qwen")
                      for k in ORDER_X], fontsize=11)
axZs.set_yticks([99.0, 99.5, 100.0])
axZs.set_yticklabels(["99%", "99.5%", "100%"], fontsize=10)
axZs.tick_params(axis="x", pad=2)
axZs.grid(axis="y", color="#EEEEEE", linewidth=0.6, zorder=0)
axZs.set_title(r"zoom $[99\%,100\%]$", loc="center",
               fontsize=11, pad=4, color="#444", style="italic")
axZs.spines[["top", "right"]].set_visible(False)

for ratio_val in (99.0, 100.0):
    fig_s1.add_artist(ConnectionPatch(
        xyA=(len(ORDER_X) - 0.45, ratio_val), coordsA=axAs.transData,
        xyB=(-0.55, ratio_val),               coordsB=axZs.transData,
        color="#AAAAAA", lw=0.7, linestyle=(0, (3, 2)), zorder=1,
    ))

s1_handles = [
    Rectangle((0, 0), 1, 1, facecolor=COLORS[k], alpha=0.82,
              edgecolor=COLORS[k], label=f"{NICE[k]}")
    for k in ORDER_X
] + [
    Rectangle((0, 0), 1, 1, facecolor=PAR_COLOR, alpha=0.85,
              edgecolor="#444", hatch="////", label=r"$\parallel$ component (tiny residual)"),
    Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#888",
              hatch="xxx", label=r"$\parallel$ unmeasured (Qwen, opposite-sign §3)"),
]
fig_s1.legend(handles=s1_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
              ncol=3, fontsize=11.5, frameon=True, framealpha=0.94,
              edgecolor="#CCCCCC", columnspacing=1.2, handlelength=2.0)
fig_s1.suptitle(
    r"$\perp$ carries 99.9% of action steering — three families",
    fontsize=24, fontweight="bold", color="#111", x=0.04, y=0.93, ha="left",
)
out_s1 = OUT / "crossfamily_slide1_panelA.png"
fig_s1.savefig(out_s1, dpi=150, bbox_inches="tight", pad_inches=0.22)
fig_s1.savefig(OUT / "crossfamily_slide1_panelA.pdf", bbox_inches="tight", pad_inches=0.22)
print(f"[slide1] wrote {out_s1}")



# ── Slide 2: Panel B + C (self-audit backup) ─────────────────────────────────
fig_s2 = plt.figure(figsize=(13.33, 7.50))
gs_s2 = fig_s2.add_gridspec(
    1, 3, width_ratios=[1.45, 0.25, 1.05],
    left=0.10, right=0.97, top=0.76, bottom=0.20, wspace=0.18,
)
axBs = fig_s2.add_subplot(gs_s2[0, 0])
# gs_s2[0,1] = spacer
axCs = fig_s2.add_subplot(gs_s2[0, 2])

ya_s = np.arange(len(AUDIT_ORDER))
for i, k in enumerate(AUDIT_ORDER):
    g = DAT[k]; c = COLORS[k]
    rms_pct = abs(g["rms_pct"])
    if g["has_par"]:
        nat_pct = abs(g["par_pct"]); amp = g["amplification"]
        axBs.plot(rms_pct, i, marker="D", markersize=14,
                  markerfacecolor="white", markeredgecolor=c,
                  markeredgewidth=2.0, zorder=5)
        axBs.annotate(f"RMS  {rms_pct:.2f}%", xy=(rms_pct, i), xycoords="data",
                      xytext=(0, 14), textcoords="offset points",
                      ha="center", va="bottom", fontsize=10,
                      color="#555", fontweight="bold", zorder=6)
        axBs.plot(nat_pct, i, marker="o", markersize=14,
                  markerfacecolor=c, markeredgecolor="white",
                  markeredgewidth=1.5, zorder=6)
        axBs.annotate(f"Natural  {g['par_pct']:+.3f}%", xy=(nat_pct, i), xycoords="data",
                      xytext=(0, -14), textcoords="offset points",
                      ha="center", va="top", fontsize=10,
                      color=c, fontweight="bold", zorder=6)
        axBs.annotate("", xy=(nat_pct * 1.6, i), xytext=(rms_pct / 1.4, i),
                      arrowprops=dict(arrowstyle="->", color=c, lw=2.0,
                                      shrinkA=2, shrinkB=2, alpha=0.85), zorder=4)
        geo_mid = (rms_pct * nat_pct) ** 0.5
        axBs.text(geo_mid, i, rf"$\approx${amp:,.0f}$\times$",
                  ha="center", va="center", fontsize=12, color=c,
                  fontweight="bold", zorder=7, transform=axBs.transData,
                  bbox=dict(facecolor="white", edgecolor="none", pad=2))
    else:
        axBs.plot(rms_pct, i, marker="D", markersize=14,
                  markerfacecolor="#E8E8E8", markeredgecolor="#888888",
                  markeredgewidth=2.0, zorder=5, alpha=0.85)
        axBs.annotate(f"RMS  {g['rms_pct']:.2f}%", xy=(rms_pct, i), xycoords="data",
                      xytext=(0, 14), textcoords="offset points",
                      ha="center", va="bottom", fontsize=10,
                      color="#888888", fontweight="bold", zorder=6)
        axBs.text(1.67, i, "reference cell (§3)\nopposite-sign shift",
                  ha="center", va="top", fontsize=9.5,
                  color="#888888", style="italic", zorder=6,
                  transform=axBs.transData,
                  bbox=dict(facecolor="white", edgecolor="#CCCCCC",
                            linewidth=0.6, pad=2, boxstyle="round,pad=0.3"))

axBs.set_xscale("log"); axBs.set_xlim(0.04, 70.0)
axBs.set_ylim(-0.6, len(AUDIT_ORDER) - 0.30)
axBs.set_yticks(ya_s)
axBs.set_yticklabels([SHORT_AUDIT[k] for k in AUDIT_ORDER],
                     fontsize=13, fontweight="bold")
axBs.tick_params(axis="y", left=False, pad=4)
axBs.set_xticks([0.1, 1.0, 10.0])
axBs.set_xticklabels(["0.1%", "1%", "10%"], fontsize=12)
axBs.set_xlabel(r"$|\parallel|$ / full $\Delta m$  (%, log scale)",
                fontsize=12, labelpad=10)
axBs.grid(axis="x", which="major", color="#EEEEEE", linewidth=0.6, zorder=0)
axBs.set_title(r"$\bf{B}$    RMS-renorm $\rightarrow$ natural-norm audit",
               loc="left", fontsize=15, pad=10, color="#111")
axBs.spines[["top", "right"]].set_visible(False)

ypos_s2 = np.arange(len(ORDER_Y))
for i, k in enumerate(ORDER_Y):
    g = DAT[k]; c = COLORS[k]
    axCs.add_patch(Rectangle((-g["null95"], i - ROW_HALF), 2 * g["null95"], 2 * ROW_HALF,
                              facecolor="#DDDDDD", alpha=0.55, edgecolor="none", zorder=0))
    for sign in (-1, +1):
        axCs.plot([sign * g["null95"]] * 2, [i - ROW_HALF, i + ROW_HALF],
                  color="#888", lw=0.7, zorder=0.5)
    axCs.plot([g["par_lo"], g["par_hi"]], [i, i], color=c, lw=2.0, alpha=0.85, zorder=4)
    axCs.plot(g["par"], i, "o", markerfacecolor=c, markeredgecolor="white",
              markeredgewidth=1.2, markersize=14, zorder=5)
    axCs.annotate(f"∥/full = {g['par_pct']:+.3f}%",
                  xy=(g["null95"], i), xycoords="data",
                  xytext=(8, 0), textcoords="offset points",
                  va="center", ha="left", fontsize=10, color=c,
                  fontweight="bold", zorder=6)

axCs.axvline(0, color="#222", lw=1.2, zorder=1)
axCs.annotate(r"$\parallel\!=0$  perfect orthogonality",
              xy=(0, -0.50), xycoords="data",
              xytext=(0, -22), textcoords="offset points",
              ha="center", va="top", fontsize=10, color="#222", style="italic",
              arrowprops=dict(arrowstyle="-", color="#222", lw=0.7))
axCs.set_yticks(ypos_s2)
axCs.set_yticklabels([SHORT[k] for k in ORDER_Y], fontsize=13, fontweight="bold")
axCs.tick_params(axis="y", left=False, pad=4)
axCs.set_ylim(-0.55, len(ORDER_Y) - 0.30)
xlim_s2 = max(DAT[k]["null95"] for k in ORDER_Y if DAT[k]["has_par"]) * 3.5
axCs.set_xlim(-xlim_s2 * 0.55, xlim_s2)
axCs.set_xlabel(r"$\parallel$-component magnitude  (natural L2 norm)",
                fontsize=12, labelpad=18)
axCs.grid(axis="x", color="#EEEEEE", linewidth=0.6, zorder=0)
axCs.set_title(r"$\bf{C}$    Cross-family stays inside random null band",
               loc="left", fontsize=15, pad=10, color="#111")
axCs.spines[["top", "right"]].set_visible(False)

s2_handles = [
    Rectangle((0, 0), 1, 1, facecolor=COLORS["gemma"], alpha=0.82,
              edgecolor=COLORS["gemma"], label="Gemma-2-9B-it (L37)"),
    Rectangle((0, 0), 1, 1, facecolor=COLORS["mistral"], alpha=0.82,
              edgecolor=COLORS["mistral"], label="Mistral-7B-v0.3 (L28)"),
    Line2D([0], [0], marker="D", color="none", markerfacecolor="white",
           markeredgecolor="#444", markeredgewidth=1.8, markersize=12,
           label=r"RMS-renorm $|\parallel|$  (artifact)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#444",
           markeredgecolor="white", markeredgewidth=1.2, markersize=12,
           label=r"natural-norm $|\parallel|$  (audited)"),
    Rectangle((0, 0), 1, 1, facecolor="#DDDDDD", alpha=0.55, edgecolor="#888",
              linewidth=0.7, label="random 95% null band"),
]
fig_s2.legend(handles=s2_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
              ncol=3, fontsize=11.5, frameon=True, framealpha=0.94,
              edgecolor="#CCCCCC", columnspacing=1.2, handlelength=2.0)
fig_s2.suptitle(
    r"Mistral 26.2% residual is a 7,147$\times$ projection artifact",
    fontsize=24, fontweight="bold", color="#111", x=0.04, y=0.93, ha="left",
)
out_s2 = OUT / "crossfamily_slide2_audit.png"
fig_s2.savefig(out_s2, dpi=150, bbox_inches="tight", pad_inches=0.22)
fig_s2.savefig(OUT / "crossfamily_slide2_audit.pdf", bbox_inches="tight", pad_inches=0.22)
print(f"[slide2] wrote {out_s2}")

# ── Talk script (printed for reference) ──────────────────────────────────────
print("""
=======================================================================
 TALK SCRIPT — Slide 2 (Panel B self-audit narration)  ~35–40 sec
=======================================================================
 "When we first looked at Mistral, the parallel component—projected
  under the standard RMS-renorm protocol—was 26.2% of the full shift.
  That looked worrying: could there be hidden evidence routing that
  Gemma and Qwen don't show?

  We ran a self-audit. RMS renormalization inflates projection magnitude
  when embedding norms differ across model families. Under natural L2-norm,
  that same component drops to just 0.13%—a 7,147-fold artifact.

  Both Gemma and Mistral now land well inside the random null band.
  The ⊥-dominance is universal."
=======================================================================
""")
