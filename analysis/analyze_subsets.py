#!/usr/bin/env python3
"""
Analyze experiment results to identify Stealth, Red Flag, and Indifferent subsets.

Classification logic:
- Stealth: baseline fails (no tool use) AND force_adopt succeeds (tool helps)
- Red Flag: baseline succeeds (no tool use) AND force_adopt fails (tool hurts)
- Indifferent: all other cases (tool doesn't matter much)
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


def load_results(path: str) -> List[dict]:
    """Load JSONL results."""
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def classify_samples(baseline_results: List[dict],
                     force_adopt_results: List[dict],
                     force_reject_results: List[dict]) -> Dict[str, List[str]]:
    """Classify samples into Stealth, Red Flag, and Indifferent."""
    
    # Build lookup by sample ID
    baseline_by_id = {r["id"]: r for r in baseline_results}
    force_adopt_by_id = {r["id"]: r for r in force_adopt_results}
    force_reject_by_id = {r["id"]: r for r in force_reject_results}
    
    # Ensure all have same samples
    sample_ids = set(baseline_by_id.keys())
    assert sample_ids == set(force_adopt_by_id.keys()), "Baseline and force_adopt have different samples"
    assert sample_ids == set(force_reject_by_id.keys()), "Baseline and force_reject have different samples"
    
    stealth = []
    red_flag = []
    indifferent = []
    
    for sample_id in sample_ids:
        baseline_success = baseline_by_id[sample_id]["success"]
        force_adopt_success = force_adopt_by_id[sample_id]["success"]
        force_reject_success = force_reject_by_id[sample_id]["success"]
        
        # Stealth: baseline fails, force_adopt succeeds
        # (model should use tool but doesn't)
        if not baseline_success and force_adopt_success:
            stealth.append(sample_id)
        
        # Red Flag: baseline succeeds, force_adopt fails
        # (model shouldn't use tool but does)
        elif baseline_success and not force_adopt_success:
            red_flag.append(sample_id)
        
        # Indifferent: all other cases
        else:
            indifferent.append(sample_id)
    
    return {
        "Stealth": stealth,
        "Red Flag": red_flag,
        "Indifferent": indifferent,
    }


def compute_subset_metrics(subset_ids: List[str],
                           results_by_policy: Dict[str, Dict[str, dict]]) -> dict:
    """Compute metrics for a subset across all policies."""
    metrics = {}
    
    for policy, results_by_id in results_by_policy.items():
        subset_results = [results_by_id[sid] for sid in subset_ids if sid in results_by_id]
        
        if not subset_results:
            continue
        
        success_count = sum(1 for r in subset_results if r["success"])
        total_tokens = sum(r["totals"]["total_tokens"] for r in subset_results)
        
        metrics[policy] = {
            "success_rate": success_count / len(subset_results),
            "count": len(subset_results),
            "success_count": success_count,
            "mean_tokens": total_tokens / len(subset_results),
        }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Analyze Stealth/Red Flag/Indifferent subsets")
    parser.add_argument("--baseline", required=True, help="Baseline results JSONL")
    parser.add_argument("--jes", required=True, help="JES results JSONL")
    parser.add_argument("--force-adopt", required=True, help="Force adopt results JSONL")
    parser.add_argument("--force-reject", required=True, help="Force reject results JSONL")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    
    print("Loading results...")
    baseline_results = load_results(args.baseline)
    jes_results = load_results(args.jes)
    force_adopt_results = load_results(args.force_adopt)
    force_reject_results = load_results(args.force_reject)
    
    print(f"Loaded {len(baseline_results)} samples")
    
    # Classify samples
    print("\nClassifying samples...")
    subsets = classify_samples(baseline_results, force_adopt_results, force_reject_results)
    
    print(f"  Stealth: {len(subsets['Stealth'])} samples")
    print(f"  Red Flag: {len(subsets['Red Flag'])} samples")
    print(f"  Indifferent: {len(subsets['Indifferent'])} samples")
    
    # Build results lookup by policy
    results_by_policy = {
        "baseline": {r["id"]: r for r in baseline_results},
        "jes": {r["id"]: r for r in jes_results},
        "force_adopt": {r["id"]: r for r in force_adopt_results},
        "force_reject": {r["id"]: r for r in force_reject_results},
    }
    
    # Compute metrics for each subset
    print("\nComputing subset metrics...")
    subset_metrics = {}
    for subset_name, subset_ids in subsets.items():
        subset_metrics[subset_name] = compute_subset_metrics(subset_ids, results_by_policy)
    
    # Build output
    output = {
        "subsets": list(subsets.keys()),
        "subset_sizes": {k: len(v) for k, v in subsets.items()},
        "success_rates": subset_metrics,
        "sample_ids": subsets,  # Include sample IDs for further analysis
    }
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved analysis to {output_path}")


if __name__ == "__main__":
    main()

