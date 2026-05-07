#!/usr/bin/env python3
"""
Decomposition figure for the Arm B (rank-1 rotation) result.

Four side-by-side bar pairs comparing canonical action steering against
rank-1 evidence->action rotation on the same paired N=483 sample:
  (1) Triggering           — P(2nd search | injection)
  (2) Conversion           — P(rescue | newly triggered 2nd search)
  (3) Selectivity          — rescued / (rescued + regressed)
  (4) Regression rate      — P(regression | newly triggered 2nd search)

Renders to PDF + PNG at 300 DPI, single-column width (~3.5 in).
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "results/reconnection_arm_b_decomposition/decomposition.json"
OUT_DIR   = REPO_ROOT / "results/reconnection_arm_b_decomposition"

# ── Okabe-Ito, colorblind-safe ───────────────────────────────────────────────
C_CANON = "#0072B2"   # blue  — canonical action axis
C_ROT   = "#D55E00"   # vermilion — rank-1 rotation
C_AXIS  = "#444444"
C_TEXT  = "#222222"
C_MUTED = "#777777"
C_HIGHLIGHT = "#D55E00"


def load_metrics():
    payload = json.load(open(DATA_PATH))
    canon = payload["canonical_action"]
    rot   = payload["arm_b_rotation"]
    # (panel_title, subtitle, v_canon%, v_rot%, ymax%, dramatic_gap?)
    return [
        ("(1) Triggering",  "P(2nd search | injection)",
         canon["n2_steered"] / canon["n"] * 100,
         rot["n2_steered"]   / rot["n"]   * 100, 28.0, False),
        ("(2) Conversion",  "P(rescue | triggered)",
         canon["p_rescue_per_newly_triggered"] * 100,
         rot["p_rescue_per_newly_triggered"]   * 100, 25.0, False),
        ("(3) Selectivity", "rescued / (R + G)",
         canon["selectivity"] * 100,
         rot["selectivity"]   * 100, 100.0, True),
        ("(4) Regression",  "P(regression | triggered)",
         canon["p_regression_per_newly_triggered"] * 100,
         rot["p_regression_per_newly_triggered"]   * 100, 5.5, True),
    ]


def build_figure():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "hatch.linewidth": 0.6,
    })
    panels = load_metrics()
    fig, axes = plt.subplots(
        1, 4, figsize=(3.5, 2.6),
        gridspec_kw={"wspace": 0.45, "left": 0.04, "right": 0.99,
                     "top": 0.78, "bottom": 0.20},
    )

    for ax, (title, sub, v_canon, v_rot, ymax, dramatic) in zip(axes, panels):
        x = np.arange(2)
        bars = ax.bar(
            x, [v_canon, v_rot], width=0.66,
            color=[C_CANON, C_ROT],
            hatch=["", "////"],
            edgecolor="white", linewidth=0.8, zorder=3,
        )

        # Value labels on top of each bar
        for b, v in zip(bars, [v_canon, v_rot]):
            ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.03,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color=C_TEXT, zorder=4)

        # Ratio annotation centered above bars (color-coded)
        ratio = v_rot / v_canon if v_canon > 0 else float("inf")
        ratio_color  = C_HIGHLIGHT if dramatic else C_MUTED
        ratio_weight = "bold"      if dramatic else "normal"
        ax.text(0.5, 1.20, f"\u00d7{ratio:.2f}",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9, color=ratio_color, fontweight=ratio_weight)

        # Panel title (number + name) and subtitle (metric)
        ax.text(0.5, 1.36, title, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", color=C_TEXT)
        ax.text(0.5, -0.30, sub, transform=ax.transAxes,
                ha="center", va="top", fontsize=7,
                color=C_MUTED, style="italic")

        ax.set_ylim(0, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels(["action", "rotation"], fontsize=7, color=C_TEXT)
        ax.tick_params(axis="x", length=0, pad=2)
        # Hide y-axis (value labels carry the info, scales differ per panel)
        ax.set_yticks([])
        ax.tick_params(axis="y", length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(C_AXIS)
        ax.spines["bottom"].set_linewidth(0.6)
        # Common unit hint: y-axis values are percentages
        # (subtle "%"-style mark on first panel only)

    # Subtle "% scale" note on left of first panel
    axes[0].text(-0.15, 1.02, "%", transform=axes[0].transAxes,
                 ha="right", va="top", fontsize=7, color=C_MUTED,
                 style="italic")

    # Single shared legend at bottom
    handles = [
        mpl.patches.Patch(facecolor=C_CANON, edgecolor="white",
                          label="Action axis (canonical)"),
        mpl.patches.Patch(facecolor=C_ROT,   edgecolor="white",
                          hatch="////", label="Rank-1 rotation (Arm B)"),
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=2, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.00))
    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    pdf_path = OUT_DIR / "figure_decomposition.pdf"
    png_path = OUT_DIR / "figure_decomposition.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"[save] {pdf_path}")
    print(f"[save] {png_path}")
    print(f"[size] PDF={pdf_path.stat().st_size} bytes  "
          f"PNG={png_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
