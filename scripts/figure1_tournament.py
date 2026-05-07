#!/usr/bin/env python3
"""
Figure 1 — Evidence axis is non-operative.

Single-panel bar chart contrasting four steering conditions on a common
margin-shift axis. The visual statement: the two operative bars (full action
direction, evidence-perpendicular component) sit at the top; the two inert
bars (evidence-parallel component, random directions) sit inside the random
null band. Renders to PDF + PNG at 300 DPI.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ── Data (paper headline numbers) ─────────────────────────────────────────────
LABELS = ["Full\naction\ndirection",
          "Evidence-\nparallel\ncomponent",
          "Evidence-\nperpendicular\ncomponent",
          "Random\ndirections"]
VALUES = [0.934, -0.181, 0.941, -0.021]
NULL_MEAN = -0.021
NULL_SD = 0.214  # mean |shift| under random directions, used as a robust SD proxy
BAND_LOW = NULL_MEAN - 2 * NULL_SD
BAND_HIGH = NULL_MEAN + 2 * NULL_SD

# ── Color story (Okabe-Ito, colorblind-safe) ─────────────────────────────────
C_OPERATIVE = "#0072B2"   # blue — full + evidence-perpendicular
C_INERT = "#BBBBBB"       # neutral grey — evidence-parallel + random
C_BAND = "#EEEEEE"        # null band fill
C_AXIS = "#444444"
C_TEXT = "#222222"
C_MUTED = "#777777"
COLORS = [C_OPERATIVE, C_INERT, C_OPERATIVE, C_INERT]


def build_figure():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "pdf.fonttype": 42,    # editable text in PDF (TrueType, not Type-3)
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    x = np.arange(len(LABELS))

    # Null band (drawn behind everything)
    ax.axhspan(BAND_LOW, BAND_HIGH, color=C_BAND, zorder=0)
    ax.axhline(0, color=C_AXIS, linewidth=0.6, zorder=1)

    # Bars
    bars = ax.bar(x, VALUES, width=0.62, color=COLORS,
                  edgecolor="white", linewidth=1.2, zorder=3)

    # Value labels above / below each bar
    for bar, v in zip(bars, VALUES):
        cx = bar.get_x() + bar.get_width() / 2
        if v >= 0:
            ax.text(cx, v + 0.035, f"{v:+.2f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color=C_TEXT, zorder=4)
        else:
            ax.text(cx, v - 0.035, f"{v:+.2f}", ha="center", va="top",
                    fontsize=11, fontweight="bold", color=C_TEXT, zorder=4)

    # Null-band label (inside the band, right-aligned)
    ax.text(3.42, NULL_MEAN, "random null\n(\u00b12\u03c3)",
            ha="right", va="center", fontsize=9,
            color=C_MUTED, style="italic", zorder=2)

    # "Behaviorally identical" tie between bar 0 and bar 2
    tie_y = max(VALUES) + 0.18
    ax.annotate("", xy=(2, tie_y), xytext=(0, tie_y),
                arrowprops=dict(arrowstyle="-", color=C_OPERATIVE,
                                lw=1.2, shrinkA=4, shrinkB=4),
                zorder=4)
    ax.plot([0, 0], [max(VALUES[0], 0) + 0.06, tie_y],
            color=C_OPERATIVE, lw=0.8, zorder=4)
    ax.plot([2, 2], [max(VALUES[2], 0) + 0.06, tie_y],
            color=C_OPERATIVE, lw=0.8, zorder=4)
    ax.text(1, tie_y + 0.025, "behaviorally identical",
            ha="center", va="bottom", fontsize=10,
            color=C_OPERATIVE, style="italic", zorder=4)

    # Axes cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=10, color=C_TEXT)
    ax.set_ylabel("mean margin shift  (toward search)", fontsize=11,
                  color=C_TEXT)
    ax.set_ylim(-0.65, 1.40)
    ax.set_yticks(np.arange(-0.5, 1.51, 0.5))
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", colors=C_AXIS, length=4, width=0.8)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_AXIS)
        ax.spines[s].set_linewidth(0.8)

    fig.tight_layout()
    return fig


def main():
    out = Path("results/figure1_tournament")
    out.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    pdf_path = out / "figure1.pdf"
    png_path = out / "figure1.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"[save] {pdf_path}")
    print(f"[save] {png_path}")
    print(f"[size] PDF={pdf_path.stat().st_size} bytes  "
          f"PNG={png_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
