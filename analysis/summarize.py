#!/usr/bin/env python3
"""
Summarize experiment results from JSONL files.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.metrics import compute_metrics, MetricsSummary


def summarize_experiments(input_dir: str, output_path: str = None):
    """
    Summarize all experiment results in a directory.
    
    Args:
        input_dir: Directory containing JSONL result files
        output_path: Output JSON file path
    """
    input_dir = Path(input_dir)
    
    # Find all JSONL files
    jsonl_files = list(input_dir.glob("**/*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}")
        return
    
    summaries = {}
    
    for jsonl_path in sorted(jsonl_files):
        print(f"Processing: {jsonl_path.name}")
        
        # Load results
        results = []
        with open(jsonl_path) as f:
            for line in f:
                results.append(json.loads(line))
        
        if not results:
            continue
        
        # Compute metrics
        summary = compute_metrics(results)
        
        # Extract experiment info from filename
        exp_name = jsonl_path.stem
        summaries[exp_name] = summary.to_dict()
    
    # Print summary table
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<40} {'Success':<10} {'Tool Calls':<12} {'Tokens':<10}")
    print("-" * 80)
    
    for exp_name, summary in summaries.items():
        success = f"{summary['success_rate']:.1%}"
        tools = f"{summary['mean_tool_calls']:.2f}"
        tokens = f"{summary['mean_tokens']:.0f}"
        print(f"{exp_name:<40} {success:<10} {tools:<12} {tokens:<10}")
    
    print("=" * 80)
    
    # Save to file
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nSaved summary to {output_path}")
    
    return summaries


def compare_policies(summaries: Dict) -> Dict:
    """
    Compare baseline vs JES policies.
    """
    comparisons = {}
    
    # Group by dataset
    datasets = defaultdict(dict)
    for exp_name, summary in summaries.items():
        if "baseline" in exp_name:
            dataset = exp_name.replace("_baseline", "")
            datasets[dataset]["baseline"] = summary
        elif "jes" in exp_name:
            dataset = exp_name.replace("_jes", "")
            datasets[dataset]["jes"] = summary
    
    for dataset, policies in datasets.items():
        if "baseline" in policies and "jes" in policies:
            baseline = policies["baseline"]
            jes = policies["jes"]
            
            comparisons[dataset] = {
                "baseline_success": baseline["success_rate"],
                "jes_success": jes["success_rate"],
                "success_delta": jes["success_rate"] - baseline["success_rate"],
                "baseline_tools": baseline["mean_tool_calls"],
                "jes_tools": jes["mean_tool_calls"],
                "jes_mean_rho": jes.get("mean_abs_rho", 0),
            }
    
    return comparisons


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment results")
    parser.add_argument("--input-dir", required=True, help="Directory with JSONL files")
    parser.add_argument("--output", help="Output summary JSON file")
    parser.add_argument("--compare", action="store_true", help="Compare baseline vs JES")
    
    args = parser.parse_args()
    
    summaries = summarize_experiments(args.input_dir, args.output)
    
    if args.compare and summaries:
        print("\n" + "=" * 80)
        print("POLICY COMPARISON: Baseline vs JES")
        print("=" * 80)
        
        comparisons = compare_policies(summaries)
        for dataset, comp in comparisons.items():
            print(f"\n{dataset}:")
            print(f"  Baseline success: {comp['baseline_success']:.1%}")
            print(f"  JES success:      {comp['jes_success']:.1%}")
            print(f"  Delta:            {comp['success_delta']:+.1%}")
            if comp['jes_mean_rho'] > 0:
                print(f"  JES mean |ρ|:     {comp['jes_mean_rho']:.4f}")


if __name__ == "__main__":
    main()

