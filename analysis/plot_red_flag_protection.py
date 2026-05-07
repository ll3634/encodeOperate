#!/usr/bin/env python3
"""
Plot Red Flag protection visualization.
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_subset_distribution(subset_analysis_path: str, output_path: str):
    """Plot subset distribution pie chart."""
    with open(subset_analysis_path) as f:
        data = json.load(f)
    
    sizes = [
        data["subset_sizes"]["Stealth"],
        data["subset_sizes"]["Red Flag"],
        data["subset_sizes"]["Indifferent"],
    ]
    labels = ["Stealth\n(Under-reliance)", "Red Flag\n(Over-reliance)", "Indifferent"]
    colors = ["#3498db", "#e74c3c", "#95a5a6"]
    explode = (0, 0.1, 0)  # Explode Red Flag slice
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(sizes))} samples)',
           shadow=True, startangle=90)
    ax.set_title("Sample Distribution by Tool Utility\n(PopQA Hard, n=100)", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved subset distribution to {output_path}")


def plot_red_flag_protection(red_flag_report_path: str, output_path: str):
    """Plot Red Flag protection bar chart."""
    with open(red_flag_report_path) as f:
        data = json.load(f)
    
    policies = ["Baseline\n(No Tool)", "Force Adopt\n(Always Tool)", "JES\n(Adaptive)", "Force Reject\n(Never Tool)"]
    success_rates = [
        data["baseline_success_rate_on_red_flags"],
        data["force_adopt_success_rate_on_red_flags"],
        data["jes_success_rate_on_red_flags"],
        0.727,  # From subset analysis
    ]
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(policies, [r * 100 for r in success_rates], color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate*100:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title(f"Red Flag Protection: Success Rate on {data['red_flag_count']} Red Flag Samples\n" +
                 "(Baseline succeeds without tool, Force Adopt fails with tool)",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect Protection')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved Red Flag protection plot to {output_path}")


def plot_overall_comparison(subset_analysis_path: str, output_path: str):
    """Plot overall performance comparison across subsets."""
    with open(subset_analysis_path) as f:
        data = json.load(f)
    
    subsets = ["Stealth\n(n=4)", "Red Flag\n(n=11)", "Indifferent\n(n=85)"]
    baseline_rates = [
        data["success_rates"]["Stealth"]["baseline"]["success_rate"],
        data["success_rates"]["Red Flag"]["baseline"]["success_rate"],
        data["success_rates"]["Indifferent"]["baseline"]["success_rate"],
    ]
    jes_rates = [
        data["success_rates"]["Stealth"]["jes"]["success_rate"],
        data["success_rates"]["Red Flag"]["jes"]["success_rate"],
        data["success_rates"]["Indifferent"]["jes"]["success_rate"],
    ]
    
    x = np.arange(len(subsets))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, [r * 100 for r in baseline_rates], width, label='Baseline', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, [r * 100 for r in jes_rates], width, label='JES', color='#2ecc71', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("Performance Comparison Across Subsets", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(subsets)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved overall comparison to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Red Flag protection visualizations")
    parser.add_argument("--subset-analysis", required=True, help="Subset analysis JSON")
    parser.add_argument("--red-flag-report", required=True, help="Red Flag report JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory for plots")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating visualizations...")
    
    # Plot 1: Subset distribution
    plot_subset_distribution(
        args.subset_analysis,
        str(output_dir / "subset_distribution.png")
    )
    
    # Plot 2: Red Flag protection
    plot_red_flag_protection(
        args.red_flag_report,
        str(output_dir / "red_flag_protection.png")
    )
    
    # Plot 3: Overall comparison
    plot_overall_comparison(
        args.subset_analysis,
        str(output_dir / "overall_comparison.png")
    )
    
    print("\nAll visualizations generated successfully!")


if __name__ == "__main__":
    main()

