#!/usr/bin/env python3
"""
Micro-level bar chart: stealth_choice recovery & tool_harmful protection
with bootstrap 95% CI error bars.

Input:  metrics.json from analyze_runs.py
Output: fig_micro.png + fig_micro.pdf

Usage:
  python scripts/plot_micro.py --metrics results/popqa/analysis/metrics.json \
                               --out results/popqa/analysis/fig_micro.png
"""

import json, argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "baseline": "#2196F3",
    "force_adopt": "#FF9800",
    "force_reject": "#F44336",
    "jes": "#4CAF50",
    "fixed_rho": "#9C27B0",
}
LABELS = {
    "baseline": "Baseline",
    "force_adopt": "Force Adopt",
    "force_reject": "Force Reject",
    "jes": "JES (ours)",
    "fixed_rho": "Fixed ρ",
}
POLICY_ORDER = ["baseline", "force_adopt", "force_reject", "jes", "fixed_rho"]


def plot_subset_bars(micro: dict, subset_name: str, ax, title: str):
    """Plot one subset panel with CI error bars."""
    policies = [p for p in POLICY_ORDER if p in micro and subset_name in micro[p]]
    if not policies:
        ax.set_title(f"{title}\n(no data)")
        return
    x = np.arange(len(policies))
    rates = [micro[p][subset_name]["success_rate"] * 100 for p in policies]
    ci_lo = [micro[p][subset_name]["ci_lower"] * 100 for p in policies]
    ci_hi = [micro[p][subset_name]["ci_upper"] * 100 for p in policies]
    errs = [[r - lo for r, lo in zip(rates, ci_lo)],
            [hi - r for r, hi in zip(rates, ci_hi)]]
    colors = [COLORS.get(p, "#9E9E9E") for p in policies]
    labels = [LABELS.get(p, p) for p in policies]
    counts = [micro[p][subset_name]["n"] for p in policies]

    bars = ax.bar(x, rates, color=colors, edgecolor="black", linewidth=0.8,
                  width=0.55, yerr=errs, capsize=4, error_kw={"linewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Success Rate (%)", fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylim(0, min(max(ci_hi) + 20, 110))
    ax.grid(axis="y", alpha=0.3)
    for bar, rate, n in zip(bars, rates, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f"{rate:.0f}%\nn={n}", ha="center", va="bottom", fontsize=7)


def main():
    parser = argparse.ArgumentParser(description="Plot micro subset bars with CI")
    parser.add_argument("--metrics", required=True, help="metrics.json from analyze_runs")
    parser.add_argument("--out", required=True, help="Output figure path")
    args = parser.parse_args()

    with open(args.metrics) as f:
        metrics = json.load(f)
    micro = metrics.get("micro", {})

    subsets = [
        ("stealth_choice", "Stealth-Choice Recovery\n(BL skipped tool → FA succeeds)"),
        ("tool_critical", "Tool-Critical Recovery\n(BL fails → FA succeeds)"),
        ("tool_harmful", "Tool-Harmful Protection\n(BL succeeds → FA fails)"),
    ]
    n_panels = sum(1 for s, _ in subsets
                   if any(s in micro.get(p, {}) for p in micro))
    if n_panels == 0:
        print("No data for any micro subset — skipping plot.")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), squeeze=False)
    ax_idx = 0
    for sname, stitle in subsets:
        if not any(sname in micro.get(p, {}) for p in micro):
            continue
        plot_subset_bars(micro, sname, axes[0, ax_idx], stitle)
        ax_idx += 1

    fig.suptitle("Micro-Level Control: Subset-Stratified Results",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}  (+pdf)")


if __name__ == "__main__":
    main()

