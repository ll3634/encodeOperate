#!/usr/bin/env python3
"""
Run JES on verify-critical manifest.

Evaluates JESStep2OnlyPolicy on:
1. verify_critical subset (headline: rescue rate)
2. Full dataset (do-no-harm: regression rate)

Usage:
    python scripts/run_jes_on_manifest.py \
        --manifest-dir results/verify_mining \
        --direction-path steering/directions/direction_search_v3.npz \
        --out results/jes_eval
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.react_loop import ReActAgent, AgentConfig
from agent.policies_verify import JESStep2OnlyPolicy, JESStep2ForcePolicy, Baseline1HopPolicy
from agent.policies import BaselinePolicy
from datasets.popqa import PopQADataset, PopQASample
from tools.search_tool import SearchTool
from steering.directions import load_direction
from steering.jes import JESConfig
from eval.paired_stats import mcnemar_test, bootstrap_ci, do_no_harm_metrics


def load_manifest(manifest_dir: Path) -> tuple:
    """Load manifest and baseline results."""
    manifest = []
    with open(manifest_dir / "manifest.jsonl") as f:
        for line in f:
            manifest.append(json.loads(line))
    
    baseline_results = []
    with open(manifest_dir / "baseline_results.jsonl") as f:
        for line in f:
            baseline_results.append(json.loads(line))
    
    return manifest, baseline_results


def load_model_and_tokenizer(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def run_jes_episode(agent, sample_dict, policy, score_mode="any") -> Dict:
    """Run JES on a single sample from manifest."""
    # Create pseudo-sample
    class PseudoSample:
        def __init__(self, d):
            self.id = d["sample_id"]
            self.question = d["question"]
            self.answer = d["gold_answer"]
            self.answers = d.get("gold_answers", [d["gold_answer"]])
    
    sample = PseudoSample(sample_dict)
    policy.reset_episode()
    
    result = agent.run(
        question=sample.question,
        policy=policy,
        gold_answer=sample.answers,
        episode_id=sample.id,
        target_side="positive"
    )
    
    steps = []
    for s in result.steps:
        steps.append({
            "step_idx": s.step_idx,
            "action": s.action,
            "action_input": s.action_input,
            "observation": s.observation[:200] if s.observation else None,
            "margin_before": s.margin_before,
            "steering": s.steering,
            "raw_model_text": s.raw_model_text,
            "parse_failure_reason": s.parse_failure_reason,
        })
    
    return {
        "sample_id": sample.id,
        "policy": policy.name,
        "final_answer": result.final_answer,
        "is_correct": result.success,
        "steps": steps,
        "n_steps": len(result.steps),
        "tool_calls": result.total_tool_calls,
        "total_tokens": result.total_tokens,
    }


def compute_paired_stats(baseline_results: List[Dict], jes_results: List[Dict],
                         subset_ids: Optional[set] = None) -> Dict:
    """Compute McNemar, bootstrap CI, rescue/regression."""
    bl_by_id = {r["sample_id"]: r for r in baseline_results}
    jes_by_id = {r["sample_id"]: r for r in jes_results}
    
    if subset_ids:
        common = sorted(subset_ids & set(bl_by_id.keys()) & set(jes_by_id.keys()))
    else:
        common = sorted(set(bl_by_id.keys()) & set(jes_by_id.keys()))
    
    bl_correct = [bl_by_id[sid]["is_correct"] for sid in common]
    jes_correct = [jes_by_id[sid]["is_correct"] for sid in common]
    
    n = len(common)
    bl_success = sum(bl_correct)
    jes_success = sum(jes_correct)
    
    rescued = sum(1 for bl, jes in zip(bl_correct, jes_correct) if not bl and jes)
    regressed = sum(1 for bl, jes in zip(bl_correct, jes_correct) if bl and not jes)
    
    stats = {
        "n": n,
        "baseline_success": bl_success,
        "baseline_rate": bl_success / n if n else 0,
        "jes_success": jes_success,
        "jes_rate": jes_success / n if n else 0,
        "rescued": rescued,
        "regressed": regressed,
        "net_gain": rescued - regressed,
        "rescue_rate": rescued / n if n else 0,
        "regression_rate": regressed / n if n else 0,
    }
    
    # McNemar
    if n > 0:
        mcnemar = mcnemar_test(bl_correct, jes_correct)
        stats["mcnemar_p"] = mcnemar["mcnemar_p"]
    
    # Bootstrap CI for success_diff
    if n > 0:
        boot = bootstrap_ci(bl_correct, jes_correct, metric="success_diff", n_bootstrap=10000)
        stats["success_diff"] = boot["observed"]
        stats["success_diff_ci_lo"] = boot["ci_lower"]
        stats["success_diff_ci_hi"] = boot["ci_upper"]

    return stats


def compute_cost_stats(results: List[Dict]) -> Dict:
    """Compute cost metrics: tokens, tool_calls, steps."""
    tokens = [r["total_tokens"] for r in results]
    tool_calls = [r["tool_calls"] for r in results]
    steps = [r["n_steps"] for r in results]

    def percentile_stats(arr):
        arr = np.array(arr)
        return {
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
        }

    return {
        "tokens": percentile_stats(tokens),
        "tool_calls": percentile_stats(tool_calls),
        "steps": percentile_stats(steps),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=str, required=True,
                        help="Directory with manifest.jsonl and baseline_results.jsonl")
    parser.add_argument("--corpus-path", type=str, required=True)
    parser.add_argument("--direction-path", type=str, required=True)
    parser.add_argument("--out", type=str, default="results/jes_eval")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--tau", type=float, default=1.5, help="JES tau threshold")
    parser.add_argument("--score-mode", type=str, default="any")
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    print("[1/4] Loading manifest...")
    manifest, baseline_results = load_manifest(manifest_dir)
    print(f"  Loaded {len(manifest)} samples")

    verify_critical = [m for m in manifest if m["label"] == "verify_critical"]
    verify_harmful = [m for m in manifest if m["label"] == "verify_harmful"]
    indifferent = [m for m in manifest if m["label"] == "indifferent"]
    print(f"  verify_critical: {len(verify_critical)}")
    print(f"  verify_harmful: {len(verify_harmful)}")
    print(f"  indifferent: {len(indifferent)}")

    # Load model
    print("\n[2/4] Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Load direction
    print("[3/4] Loading direction...")
    direction, _ = load_direction(args.direction_path)
    direction_rms = float(np.sqrt(np.mean(direction ** 2)))

    # Setup agent
    search_tool = SearchTool(corpus_path=args.corpus_path)
    tools = {"search": search_tool}

    config = AgentConfig(
        max_steps=5, max_tokens_per_step=256, temperature=0.0,
        layer=args.layer, tools=list(tools.keys()), score_mode=args.score_mode,
    )
    agent = ReActAgent(
        model=model, tokenizer=tokenizer, tools=tools,
        config=config, direction=direction, direction_rms=direction_rms,
    )

    # Setup JES policy
    jes_config = JESConfig(tau=args.tau)
    jes_policy = JESStep2OnlyPolicy(config=jes_config, direction=direction)

    # Run JES on all samples
    print("\n[4/4] Running JES evaluation...")
    jes_results = []
    for m in tqdm(manifest, desc="JES"):
        rec = run_jes_episode(agent, m, jes_policy, args.score_mode)
        jes_results.append(rec)

    # Save JES results
    with open(out_dir / "jes_results.jsonl", "w") as f:
        for r in jes_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute stats
    print("\n" + "=" * 70)
    print("  JES EVALUATION RESULTS")
    print("=" * 70)

    # Full dataset stats
    full_stats = compute_paired_stats(baseline_results, jes_results)
    print(f"\n[FULL DATASET] n={full_stats['n']}")
    print(f"  Baseline: {full_stats['baseline_rate']*100:.1f}%  |  JES: {full_stats['jes_rate']*100:.1f}%")
    print(f"  Rescued: {full_stats['rescued']}  |  Regressed: {full_stats['regressed']}  |  Net: {full_stats['net_gain']:+d}")
    print(f"  McNemar p={full_stats.get('mcnemar_p', 'N/A')}")

    # verify_critical subset
    vc_ids = {m["sample_id"] for m in verify_critical}
    if vc_ids:
        vc_stats = compute_paired_stats(baseline_results, jes_results, vc_ids)
        print(f"\n[VERIFY-CRITICAL SUBSET] n={vc_stats['n']}")
        print(f"  Baseline: {vc_stats['baseline_rate']*100:.1f}% (all wrong by definition)")
        print(f"  JES: {vc_stats['jes_rate']*100:.1f}%  <-- HEADLINE RESCUE RATE")
        print(f"  Rescued: {vc_stats['rescued']}")
    else:
        vc_stats = {"n": 0, "message": "No verify_critical samples"}
        print("\n[VERIFY-CRITICAL SUBSET] No samples found!")

    # Cost stats
    jes_cost = compute_cost_stats(jes_results)
    bl_cost = compute_cost_stats(baseline_results)
    print(f"\n[COST]")
    print(f"  Baseline tokens: {bl_cost['tokens']['mean']:.0f} (avg), {bl_cost['tokens']['p95']:.0f} (p95)")
    print(f"  JES tokens:      {jes_cost['tokens']['mean']:.0f} (avg), {jes_cost['tokens']['p95']:.0f} (p95)")
    print(f"  Baseline steps:  {bl_cost['steps']['mean']:.2f} (avg)")
    print(f"  JES steps:       {jes_cost['steps']['mean']:.2f} (avg)")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "manifest_dir": str(manifest_dir),
        "n_total": len(manifest),
        "n_verify_critical": len(verify_critical),
        "n_verify_harmful": len(verify_harmful),
        "n_indifferent": len(indifferent),
        "full_stats": full_stats,
        "verify_critical_stats": vc_stats,
        "cost_baseline": bl_cost,
        "cost_jes": jes_cost,
        "config": {"tau": args.tau, "layer": args.layer},
    }

    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  Results saved to: {out_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

