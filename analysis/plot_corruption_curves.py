#!/usr/bin/env python3
"""
Plot corruption sweep curves showing success rate vs corruption probability.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_corruption_results(path: str) -> Dict:
    """Load corruption sweep results from JSON file."""
    with open(path) as f:
        return json.load(f)


def extract_corruption_curve(results: Dict) -> tuple:
    """
    Extract (probabilities, success_rates) from corruption sweep results.
    """
    probs = []
    success_rates = []
    
    for key, summary in sorted(results.items()):
        # Parse probability from key like "p=0.1"
        if key.startswith("p="):
            prob = float(key.split("=")[1])
            probs.append(prob)
            success_rates.append(summary.get("success_rate", 0) * 100)
    
    return np.array(probs), np.array(success_rates)


def plot_corruption_curves(
    results_dict: Dict[str, Dict],
    output_path: str = None,
    title: str = "Success Rate vs Tool Corruption"
):
    """
    Plot corruption curves for multiple experiments.
    
    Args:
        results_dict: Dict mapping experiment name to corruption sweep results
        output_path: Output plot path
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (name, results) in enumerate(results_dict.items()):
        probs, success_rates = extract_corruption_curve(results)
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        ax.plot(probs, success_rates, 
               color=color, marker=marker, markersize=8,
               linewidth=2, label=name, alpha=0.8)
        
        # Fill area under curve
        ax.fill_between(probs, success_rates, alpha=0.1, color=color)
    
    ax.set_xlabel('Corruption Probability', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-0.02, 0.42)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Add annotation
    ax.annotate('Higher curve = more robust',
               xy=(0.3, 80), fontsize=10, alpha=0.7,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_single_corruption_curve(
    results: Dict,
    output_path: str = None,
    title: str = "Success Rate vs Tool Corruption"
):
    """Plot a single corruption curve."""
    probs, success_rates = extract_corruption_curve(results)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(probs, success_rates, 
           color='#4CAF50', marker='o', markersize=10,
           linewidth=2.5, label='JES')
    ax.fill_between(probs, success_rates, alpha=0.2, color='#4CAF50')
    
    # Add reference line for random baseline
    ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    
    ax.set_xlabel('Corruption Probability', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-0.02, 0.42)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
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
    parser = argparse.ArgumentParser(description="Plot corruption curves")
    parser.add_argument("--input", required=True, nargs="+", help="Corruption sweep JSON file(s)")
    parser.add_argument("--names", nargs="+", help="Names for each input file")
    parser.add_argument("--output", help="Output plot path (PNG)")
    parser.add_argument("--title", default="Success Rate vs Tool Corruption")
    
    args = parser.parse_args()
    
    if len(args.input) == 1:
        results = load_corruption_results(args.input[0])
        plot_single_corruption_curve(results, args.output, args.title)
    else:
        names = args.names or [Path(p).stem for p in args.input]
        results_dict = {}
        for name, path in zip(names, args.input):
            results_dict[name] = load_corruption_results(path)
        plot_corruption_curves(results_dict, args.output, args.title)


if __name__ == "__main__":
    main()

