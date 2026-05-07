#!/usr/bin/env python3
"""
Plot success vs cost Pareto frontier.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_summaries(summary_path: str) -> Dict:
    """Load experiment summaries from JSON file."""
    with open(summary_path) as f:
        return json.load(f)


def extract_pareto_points(summaries: Dict) -> List[Tuple[str, float, float]]:
    """
    Extract (name, success_rate, cost) points.
    Cost can be tokens or tool_calls.
    """
    points = []
    for name, summary in summaries.items():
        success = summary.get("success_rate", 0)
        cost = summary.get("mean_tokens", 0)
        points.append((name, success, cost))
    return points


def compute_pareto_frontier(points: List[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
    """
    Compute Pareto frontier (maximize success, minimize cost).
    """
    # Sort by success descending, then cost ascending
    sorted_points = sorted(points, key=lambda x: (-x[1], x[2]))
    
    frontier = []
    min_cost = float('inf')
    
    for point in sorted_points:
        if point[2] < min_cost:
            frontier.append(point)
            min_cost = point[2]
    
    return frontier


def plot_pareto(summaries: Dict, output_path: str = None, title: str = "Success vs Cost Pareto"):
    """
    Create Pareto frontier plot.
    """
    points = extract_pareto_points(summaries)
    frontier = compute_pareto_frontier(points)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot all points
    names = [p[0] for p in points]
    successes = [p[1] * 100 for p in points]  # Convert to percentage
    costs = [p[2] for p in points]
    
    # Color by policy type
    colors = []
    for name in names:
        if "baseline" in name:
            colors.append("#2196F3")  # Blue
        elif "jes" in name:
            colors.append("#4CAF50")  # Green
        elif "force" in name:
            colors.append("#FF9800")  # Orange
        else:
            colors.append("#9E9E9E")  # Gray
    
    scatter = ax.scatter(costs, successes, c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=1)
    
    # Plot Pareto frontier line
    if frontier:
        frontier_costs = [p[2] for p in frontier]
        frontier_successes = [p[1] * 100 for p in frontier]
        ax.plot(frontier_costs, frontier_successes, 'r--', linewidth=2, label='Pareto Frontier', alpha=0.7)
    
    # Add labels
    for name, success, cost in points:
        short_name = name.split("_")[-1] if "_" in name else name
        ax.annotate(short_name, (cost, success * 100), textcoords="offset points",
                   xytext=(5, 5), fontsize=8, alpha=0.8)
    
    ax.set_xlabel('Cost (Mean Tokens)', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add legend for colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3', markersize=10, label='Baseline'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50', markersize=10, label='JES'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF9800', markersize=10, label='Forced'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Pareto frontier")
    parser.add_argument("--input", required=True, help="Summary JSON file")
    parser.add_argument("--output", help="Output plot path (PNG)")
    parser.add_argument("--title", default="Success vs Cost Pareto Frontier")
    
    args = parser.parse_args()
    
    summaries = load_summaries(args.input)
    plot_pareto(summaries, args.output, args.title)


if __name__ == "__main__":
    main()

