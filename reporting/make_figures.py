#!/usr/bin/env python3
"""
Publication-grade figures for E2E agent evaluation.

Generates:
  FigA  – Stealth/RedFlag recovery & protection bar chart (+ bootstrap CI)
  Table1 – Main results table (success, tokens, tool_calls, regression) for all policies
  FigC  – Pareto scatter: success vs tokens; success vs tool_calls

Usage:
  python -m reporting.make_figures --results-dir results/popqa_500_unified --out figures/
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Styling constants
POLICY_COLORS = {
    "baseline": "#2196F3",
    "force_adopt": "#FF9800",
    "force_reject": "#F44336",
    "jes": "#4CAF50",
}
POLICY_LABELS = {
    "baseline": "Baseline",
    "force_adopt": "Force Adopt",
    "force_reject": "Force Reject",
    "jes": "JES (ours)",
}


def _load_summary(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# FigA: Stealth / RedFlag subset recovery / protection bar chart
# ---------------------------------------------------------------------------

def fig_a_subset_bars(
    subset_metrics: Dict,
    output_path: str,
    title: str = "Micro-Level Control: Stealth Recovery & Red-Flag Protection",
):
    """
    Bar chart showing per-policy success on Stealth and RedFlag subsets.
    subset_metrics: output of subset_labeling.compute_subset_metrics()
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, sname, subtitle in zip(
        axes,
        ["stealth", "red_flag"],
        ["Stealth Subset (BL fails, FA succeeds)\nRecovery Rate",
         "Red-Flag Subset (BL succeeds, FA fails)\nProtection Rate"],
    ):
        if sname not in subset_metrics:
            continue
        data = subset_metrics[sname]
        policies = sorted(data.keys(), key=lambda p: list(POLICY_COLORS).index(p) if p in POLICY_COLORS else 99)
        x = np.arange(len(policies))
        rates = [data[p]["success_rate"] * 100 for p in policies]
        colors = [POLICY_COLORS.get(p, "#9E9E9E") for p in policies]
        labels = [POLICY_LABELS.get(p, p) for p in policies]
        counts = [f"n={data[p]['count']}" for p in policies]

        bars = ax.bar(x, rates, color=colors, edgecolor="black", linewidth=0.8, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Success Rate (%)", fontsize=11)
        ax.set_title(subtitle, fontsize=11)
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=0.3)

        for bar, rate, cnt in zip(bars, rates, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{rate:.0f}%\n{cnt}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Table 1: Main results markdown/CSV
# ---------------------------------------------------------------------------

def table1_main_results(
    summaries: Dict[str, Dict],
    output_path: str,
):
    """
    Generate Table 1 as CSV and markdown.
    summaries: {policy_name: run_summary_dict}
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    header = ["Policy", "N", "Success%", "AvgTokens", "AvgToolCalls",
              "AvgSteps", "Regress%", "Rescue%", "NetGain", "Saturate%"]
    rows = []
    order = ["baseline", "force_adopt", "force_reject", "jes"]
    for pname in order:
        if pname not in summaries:
            continue
        s = summaries[pname]
        rows.append([
            POLICY_LABELS.get(pname, pname),
            s.get("n", ""),
            f"{s.get('success_rate', 0) * 100:.1f}",
            f"{s.get('avg_tokens_total', 0):.0f}",
            f"{s.get('avg_tool_calls', 0):.2f}",
            f"{s.get('avg_steps', 0):.1f}",
            f"{s.get('regression_rate', 0) * 100:.1f}",
            f"{s.get('rescue_rate', 0) * 100:.1f}",
            str(s.get("net_gain", 0)),
            f"{s.get('saturation_rate', 0) * 100:.1f}",
        ])

    # CSV
    csv_path = p.with_suffix(".csv")
    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(c) for c in row) + "\n")

    # Markdown
    md_path = p.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("| " + " | ".join("---" for _ in header) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(c) for c in row) + " |\n")

    print(f"Table1 written to {csv_path} and {md_path}")


# ---------------------------------------------------------------------------
# FigC: Pareto scatter (success vs tokens & success vs tool_calls)
# ---------------------------------------------------------------------------

def fig_c_pareto(
    summaries: Dict[str, Dict],
    output_path: str,
    title: str = "Pareto Efficiency: Success vs Cost",
):
    """Dual-panel Pareto scatter."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, cost_key, xlabel in [
        (ax1, "avg_tokens_total", "Mean Tokens"),
        (ax2, "avg_tool_calls", "Mean Tool Calls"),
    ]:
        for pname, s in summaries.items():
            color = POLICY_COLORS.get(pname, "#9E9E9E")
            label = POLICY_LABELS.get(pname, pname)
            x = s.get(cost_key, 0)
            y = s.get("success_rate", 0) * 100
            ax.scatter(x, y, c=color, s=140, edgecolors="black", linewidth=1,
                       zorder=5, label=label)
            ax.annotate(label, (x, y), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, alpha=0.85)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Success Rate (%)", fontsize=11)
        ax.grid(True, alpha=0.3)
        # Ideal corner annotation
        ax.annotate("← better", xy=(0.02, 0.98), xycoords="axes fraction",
                    fontsize=8, alpha=0.5, va="top")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    handles = [Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=POLICY_COLORS.get(p, "#9E9E9E"),
                       markersize=10, label=POLICY_LABELS.get(p, p))
               for p in summaries]
    fig.legend(handles=handles, loc="lower center", ncol=len(summaries),
               fontsize=9, frameon=True)
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _save_fig(fig, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {p}  (+pdf)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate publication-grade figures")
    parser.add_argument("--results-dir", required=True,
                        help="Dir containing unified summary JSONs (per-policy)")
    parser.add_argument("--subset-report", default=None,
                        help="Path to subset_report.json from subset_labeling")
    parser.add_argument("--out", default="figures", help="Output directory")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rdir = Path(args.results_dir)

    # Load summaries
    summaries = {}
    for p in sorted(rdir.glob("*_summary.json")):
        s = _load_summary(str(p))
        pname = s.get("policy_name", p.stem.replace("_summary", ""))
        summaries[pname] = s

    if not summaries:
        print(f"No *_summary.json found in {rdir}")
        return

    # Table 1
    table1_main_results(summaries, str(out / "table1"))

    # FigC: Pareto
    fig_c_pareto(summaries, str(out / "fig_c_pareto.png"))

    # FigA: Subset bars (needs subset report)
    if args.subset_report:
        with open(args.subset_report) as f:
            subset_data = json.load(f)
        fig_a_subset_bars(subset_data.get("metrics", {}),
                          str(out / "fig_a_subsets.png"))
    else:
        # Try auto-detect
        sr = rdir / "subset_report.json"
        if sr.exists():
            with open(sr) as f:
                subset_data = json.load(f)
            fig_a_subset_bars(subset_data.get("metrics", {}),
                              str(out / "fig_a_subsets.png"))
        else:
            print("Skipping FigA (no subset report found)")

    print(f"\nAll figures saved to {out}/")


if __name__ == "__main__":
    main()

