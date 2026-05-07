#!/usr/bin/env python3
"""
Metrics computation for E2E agent evaluation.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from .scorers import answer_scorer


@dataclass
class MetricsSummary:
    """Summary of evaluation metrics."""
    n_samples: int = 0
    n_success: int = 0
    success_rate: float = 0.0
    
    # Tool usage
    mean_tool_calls: float = 0.0
    tool_precision: float = 0.0  # Correct tool calls / total tool calls
    tool_recall: float = 0.0     # Correct tool calls / needed tool calls
    
    # Cost metrics
    mean_tokens: float = 0.0
    mean_wall_time_ms: float = 0.0
    
    # JES-specific metrics
    mean_abs_rho: float = 0.0
    achieved_rate: float = 0.0  # % where margin crossed threshold
    
    # Failure attribution
    failure_counts: Dict[str, int] = field(default_factory=dict)
    
    # Per-policy breakdown
    policy: str = "unknown"
    
    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "n_success": self.n_success,
            "success_rate": self.success_rate,
            "mean_tool_calls": self.mean_tool_calls,
            "tool_precision": self.tool_precision,
            "tool_recall": self.tool_recall,
            "mean_tokens": self.mean_tokens,
            "mean_wall_time_ms": self.mean_wall_time_ms,
            "mean_abs_rho": self.mean_abs_rho,
            "achieved_rate": self.achieved_rate,
            "failure_counts": self.failure_counts,
            "policy": self.policy,
        }


def compute_metrics(
    results: List[Dict],
    score_mode: str = "any",
) -> MetricsSummary:
    """
    Compute metrics from a list of episode results.
    
    Args:
        results: List of EpisodeResult.to_dict() outputs
        score_mode: Scoring mode for answer matching
        
    Returns:
        MetricsSummary with aggregated metrics
    """
    if not results:
        return MetricsSummary()
    
    summary = MetricsSummary(n_samples=len(results))
    
    # Aggregate values
    total_tool_calls = 0
    total_tokens = 0
    total_wall_time = 0
    total_rho = 0
    n_achieved = 0
    n_with_steering = 0
    failure_counts = defaultdict(int)
    
    # Tool usage tracking
    correct_tool_calls = 0
    needed_tool_calls = 0
    
    for r in results:
        # Success scoring
        score_result = answer_scorer(
            r.get("final_answer", ""),
            r.get("gold_answer", ""),
            mode=score_mode
        )
        if score_result["matched"]:
            summary.n_success += 1
        
        # Tool calls
        totals = r.get("totals", {})
        tool_calls = totals.get("tool_calls", 0)
        total_tool_calls += tool_calls
        
        # Tokens and time
        total_tokens += totals.get("total_tokens", 0)
        total_wall_time += totals.get("total_wall_time_ms", 0)
        
        # Failure attribution
        if r.get("failure_reason"):
            failure_counts[r["failure_reason"]] += 1
        
        # JES metrics from steps
        for step in r.get("steps", []):
            steering = step.get("steering", {})
            if steering and "rho_used" in steering:
                n_with_steering += 1
                total_rho += abs(steering.get("rho_used", 0))
                if steering.get("achieved", False):
                    n_achieved += 1
    
    # Compute averages
    n = summary.n_samples
    summary.success_rate = summary.n_success / n if n > 0 else 0
    summary.mean_tool_calls = total_tool_calls / n if n > 0 else 0
    summary.mean_tokens = total_tokens / n if n > 0 else 0
    summary.mean_wall_time_ms = total_wall_time / n if n > 0 else 0
    summary.mean_abs_rho = total_rho / n_with_steering if n_with_steering > 0 else 0
    summary.achieved_rate = n_achieved / n_with_steering if n_with_steering > 0 else 0
    summary.failure_counts = dict(failure_counts)
    
    # Policy from first result
    if results:
        summary.policy = results[0].get("policy", "unknown")
    
    return summary


def load_results(path: str) -> List[Dict]:
    """Load results from JSONL file."""
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def save_summary(summary: MetricsSummary, path: str):
    """Save summary to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary.to_dict(), f, indent=2)
    print(f"Saved summary to {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", help="Output summary JSON file")
    args = parser.parse_args()
    
    results = load_results(args.input)
    summary = compute_metrics(results)
    
    print(f"Metrics for {args.input}:")
    print(f"  Success rate: {summary.success_rate:.1%}")
    print(f"  Mean tool calls: {summary.mean_tool_calls:.2f}")
    print(f"  Mean tokens: {summary.mean_tokens:.0f}")
    
    if args.output:
        save_summary(summary, args.output)

