#!/usr/bin/env python3
"""
Pareto efficiency scatter: Success% vs Cost (tokens / tool_calls).

Input:  metrics.json from analyze_runs.py
Output: fig_pareto.png + pdf

Usage:
  python scripts/plot_pareto.py --metrics results/popqa/analysis/metrics.json \
                                --out results/popqa/analysis/fig_pareto.png
"""

import json, argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = {
    "baseline": "#2196F3", "force_adopt": "#FF9800",
    "force_reject": "#F44336", "jes": "#4CAF50", "fixed_rho": "#9C27B0",
}
LABELS = {
    "baseline": "Baseline", "force_adopt": "Force Adopt",
    "force_reject": "Force Reject", "jes": "JES (ours)", "fixed_rho": "Fixed ρ",
}
MARKERS = {
    "baseline": "o", "force_adopt": "s",
    "force_reject": "D", "jes": "^", "fixed_rho": "v",
}


def main():
    parser = argparse.ArgumentParser(description="Pareto efficiency scatter")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.metrics) as f:
        metrics = json.load(f)
    macro = metrics.get("macro", {})
    if not macro:
        print("No macro metrics — skipping plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, cost_key, xlabel in [
        (ax1, "avg_tokens", "Mean Tokens"),
        (ax2, "avg_tool_calls", "Mean Tool Calls"),
    ]:
        for pname, m in macro.items():
            c = COLORS.get(pname, "#9E9E9E")
            mk = MARKERS.get(pname, "o")
            lbl = LABELS.get(pname, pname)
            x = m.get(cost_key, 0)
            y = m.get("success_rate", 0) * 100
            ax.scatter(x, y, c=c, s=160, marker=mk, edgecolors="black",
                       linewidth=1, zorder=5, label=lbl)
            ax.annotate(lbl, (x, y), textcoords="offset points",
                        xytext=(8, 8), fontsize=8, alpha=0.85)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Success Rate (%)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.annotate("← better", xy=(0.02, 0.98), xycoords="axes fraction",
                    fontsize=8, alpha=0.5, va="top")

    handles = [Line2D([0], [0], marker=MARKERS.get(p, 'o'), color='w',
                       markerfacecolor=COLORS.get(p, "#9E9E9E"),
                       markersize=10, label=LABELS.get(p, p))
               for p in macro]
    fig.legend(handles=handles, loc="lower center", ncol=len(macro),
               fontsize=9, frameon=True)
    fig.suptitle("Pareto Efficiency: Success vs Cost", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.07, 1, 0.94])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}  (+pdf)")


if __name__ == "__main__":
    main()

