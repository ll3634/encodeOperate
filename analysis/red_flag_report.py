#!/usr/bin/env python3
"""
Generate detailed Red Flag analysis report.

This script analyzes Red Flag samples (where baseline succeeds without tool,
but force_adopt fails with tool) to demonstrate JES's safety value.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: str) -> List[dict]:
    """Load JSONL file."""
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def analyze_red_flag_protection(subset_analysis_path: str,
                                 baseline_path: str,
                                 jes_path: str,
                                 force_adopt_path: str) -> dict:
    """Analyze JES protection rate on Red Flag samples."""
    
    # Load subset analysis
    with open(subset_analysis_path, encoding="utf-8") as f:
        subset_analysis = json.load(f)
    
    red_flag_ids = set(subset_analysis["sample_ids"]["Red Flag"])
    
    # Load results
    baseline_results = {r["id"]: r for r in load_jsonl(baseline_path)}
    jes_results = {r["id"]: r for r in load_jsonl(jes_path)}
    force_adopt_results = {r["id"]: r for r in load_jsonl(force_adopt_path)}
    
    # Analyze Red Flag samples
    red_flag_details = []
    
    for sample_id in red_flag_ids:
        baseline = baseline_results[sample_id]
        jes = jes_results[sample_id]
        force_adopt = force_adopt_results[sample_id]
        
        # Red Flag definition: baseline succeeds, force_adopt fails
        assert baseline["success"], f"Red Flag sample {sample_id} should succeed in baseline"
        assert not force_adopt["success"], f"Red Flag sample {sample_id} should fail in force_adopt"
        
        # Check if JES protected (avoided the tool)
        jes_protected = jes["success"]
        
        # Get tool usage
        baseline_tool_calls = baseline["totals"]["tool_calls"]
        jes_tool_calls = jes["totals"]["tool_calls"]
        force_adopt_tool_calls = force_adopt["totals"]["tool_calls"]
        
        red_flag_details.append({
            "id": sample_id,
            "question": baseline["question"],
            "gold_answer": baseline["gold_answer"],
            "baseline_success": baseline["success"],
            "baseline_tool_calls": baseline_tool_calls,
            "jes_success": jes["success"],
            "jes_tool_calls": jes_tool_calls,
            "jes_protected": jes_protected,
            "force_adopt_success": force_adopt["success"],
            "force_adopt_tool_calls": force_adopt_tool_calls,
        })
    
    # Compute protection rate
    total_red_flags = len(red_flag_details)
    protected_count = sum(1 for d in red_flag_details if d["jes_protected"])
    protection_rate = protected_count / total_red_flags if total_red_flags > 0 else 0
    
    return {
        "total_samples": subset_analysis["subset_sizes"]["Stealth"] + 
                        subset_analysis["subset_sizes"]["Red Flag"] + 
                        subset_analysis["subset_sizes"]["Indifferent"],
        "red_flag_count": total_red_flags,
        "red_flag_percentage": total_red_flags / (subset_analysis["subset_sizes"]["Stealth"] + 
                                                   subset_analysis["subset_sizes"]["Red Flag"] + 
                                                   subset_analysis["subset_sizes"]["Indifferent"]),
        "jes_protected_count": protected_count,
        "jes_protection_rate": protection_rate,
        "baseline_success_rate_on_red_flags": 1.0,  # By definition
        "force_adopt_success_rate_on_red_flags": 0.0,  # By definition
        "jes_success_rate_on_red_flags": protection_rate,
        "red_flag_samples": red_flag_details,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate Red Flag analysis report")
    parser.add_argument("--subset-analysis", required=True, help="Subset analysis JSON")
    parser.add_argument("--baseline", required=True, help="Baseline results JSONL")
    parser.add_argument("--jes", required=True, help="JES results JSONL")
    parser.add_argument("--force-adopt", required=True, help="Force adopt results JSONL")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    
    print("Analyzing Red Flag protection...")
    report = analyze_red_flag_protection(
        args.subset_analysis,
        args.baseline,
        args.jes,
        args.force_adopt,
    )
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Red Flag Analysis Report ===")
    print(f"Total samples: {report['total_samples']}")
    print(f"Red Flag samples: {report['red_flag_count']} ({report['red_flag_percentage']:.1%})")
    print(f"\nRed Flag Protection:")
    print(f"  Baseline (no tool): {report['baseline_success_rate_on_red_flags']:.1%} success")
    print(f"  Force Adopt (always tool): {report['force_adopt_success_rate_on_red_flags']:.1%} success")
    print(f"  JES (adaptive): {report['jes_success_rate_on_red_flags']:.1%} success")
    print(f"\nJES Protection Rate: {report['jes_protection_rate']:.1%}")
    print(f"  Protected: {report['jes_protected_count']}/{report['red_flag_count']} samples")
    print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()

