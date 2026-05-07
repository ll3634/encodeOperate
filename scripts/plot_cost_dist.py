#!/usr/bin/env python3
"""
Cost distribution plots: tokens, tool_calls, steps per policy.

Input:  directory with {policy}.jsonl files
Output: fig_cost_dist.png + pdf

Usage:
  python scripts/plot_cost_dist.py --run-dir results/popqa_500 \
                                   --out results/popqa_500/fig_cost_dist.png
"""

import json, argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.unified_output import load_records

COLORS = {
    "baseline": "#2196F3", "force_adopt": "#FF9800",
    "force_reject": "#F44336", "jes": "#4CAF50", "fixed_rho": "#9C27B0",
}
LABELS = {
    "baseline": "Baseline", "force_adopt": "Force Adopt",
    "force_reject": "Force Reject", "jes": "JES (ours)", "fixed_rho": "Fixed ρ",
}
POLICY_ORDER = ["baseline", "force_adopt", "force_reject", "jes", "fixed_rho"]


def main():
    parser = argparse.ArgumentParser(description="Cost distribution plots")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    all_recs = {}
    for p in sorted(run_dir.glob("*.jsonl")):
        if p.stem == "manifest":
            continue
        recs = load_records(str(p))
        if recs:
            pname = recs[0].get("policy_name", p.stem)
            all_recs[pname] = recs

    if not all_recs:
        print("No JSONL files found — skipping.")
        return

    policies = [p for p in POLICY_ORDER if p in all_recs]
    metrics_list = [
        ("tokens_total", "Total Tokens"),
        ("tool_calls", "Tool Calls"),
        ("steps", "Steps"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (key, title) in zip(axes, metrics_list):
        data = []
        labels = []
        colors = []
        for pname in policies:
            vals = [r.get(key, 0) for r in all_recs[pname]]
            data.append(vals)
            labels.append(LABELS.get(pname, pname))
            colors.append(COLORS.get(pname, "#9E9E9E"))

        bplot = ax.boxplot(data, labels=labels, patch_artist=True,
                           showfliers=False, widths=0.5)
        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)

        # Add median labels
        for i, med in enumerate(bplot["medians"]):
            median_val = med.get_ydata()[0]
            ax.text(i + 1, median_val, f"{median_val:.0f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")

    fig.suptitle("Cost Distribution by Policy", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}  (+pdf)")


if __name__ == "__main__":
    main()

