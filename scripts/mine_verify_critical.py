#!/usr/bin/env python3
"""
Verify-Critical Mining Script.

Mines verify-critical samples from a candidate pool:
- verify_critical: baseline_1hop incorrect AND oracle_2hop correct
- verify_harmful: baseline_1hop correct AND oracle_2hop incorrect  
- indifferent: both same

Usage:
    python scripts/mine_verify_critical.py \
        --data-path data/popqa/popqa_test.jsonl \
        --corpus-path data/popqa/corpus.jsonl \
        --direction-path steering/directions/direction_search_v3.npz \
        --n-samples 200 --out results/verify_mining
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

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.react_loop import ReActAgent, AgentConfig, EpisodeResult
from agent.policies_verify import Baseline1HopPolicy, Oracle2HopPolicy
from datasets.popqa import PopQADataset
from tools.search_tool import SearchTool
from steering.directions import load_direction
from eval.scorers import answer_scorer


def load_model_and_tokenizer(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def run_episode_with_policy(
    agent: ReActAgent,
    sample,
    policy,
    score_mode: str = "any"
) -> Dict:
    """Run a single episode and return structured result."""
    result = agent.run(
        question=sample.question,
        policy=policy,
        gold_answer=sample.answers,
        episode_id=sample.id,
        target_side="positive"  # Should use tool
    )
    
    # Build step records with raw_text
    step_records = []
    for step in result.steps:
        step_rec = {
            "step_idx": step.step_idx,
            "action": step.action,
            "action_input": step.action_input,
            "observation": step.observation[:200] if step.observation else None,
            "final_answer": step.final_answer,
            "margin_before": step.margin_before,
            "tokens": step.tokens_prompt + step.tokens_completion,
            "raw_model_text": step.raw_model_text,
            "parse_failure_reason": step.parse_failure_reason,
        }
        step_records.append(step_rec)
    
    return {
        "sample_id": sample.id,
        "question": sample.question,
        "gold_answer": sample.answer,
        "gold_answers": sample.answers,
        "policy": policy.name,
        "final_answer": result.final_answer,
        "is_correct": result.success,
        "steps": step_records,
        "n_steps": len(result.steps),
        "tool_calls": result.total_tool_calls,
        "total_tokens": result.total_tokens,
        "failure_reason": result.failure_reason,
    }


def mine_samples(
    agent: ReActAgent,
    samples: List,
    out_dir: Path,
    score_mode: str = "any"
) -> Dict:
    """Run baseline_1hop and oracle_2hop on all samples, produce manifest."""
    
    baseline_policy = Baseline1HopPolicy()
    oracle_policy = Oracle2HopPolicy()
    
    baseline_results = []
    oracle_results = []
    
    print("\n[1/2] Running baseline_1hop...")
    for sample in tqdm(samples, desc="baseline_1hop"):
        baseline_policy.reset_episode()
        rec = run_episode_with_policy(agent, sample, baseline_policy, score_mode)
        baseline_results.append(rec)
    
    print("\n[2/2] Running oracle_2hop...")
    for sample in tqdm(samples, desc="oracle_2hop"):
        oracle_policy.reset_episode()
        rec = run_episode_with_policy(agent, sample, oracle_policy, score_mode)
        oracle_results.append(rec)
    
    # Build manifest with labels
    bl_by_id = {r["sample_id"]: r for r in baseline_results}
    or_by_id = {r["sample_id"]: r for r in oracle_results}
    
    manifest = []
    stats = {"verify_critical": 0, "verify_harmful": 0, "indifferent": 0, "total": 0}
    
    for sample in samples:
        sid = sample.id
        bl = bl_by_id[sid]
        orc = or_by_id[sid]
        
        if not bl["is_correct"] and orc["is_correct"]:
            label = "verify_critical"
        elif bl["is_correct"] and not orc["is_correct"]:
            label = "verify_harmful"
        else:
            label = "indifferent"
        
        stats[label] += 1
        stats["total"] += 1
        
        manifest.append({
            "sample_id": sid,
            "question": sample.question,
            "gold_answer": sample.answer,
            "gold_answers": sample.answers,
            "label": label,
            "baseline_correct": bl["is_correct"],
            "oracle_correct": orc["is_correct"],
            "baseline_final_answer": bl["final_answer"],
            "oracle_final_answer": orc["final_answer"],
            "baseline_tool_calls": bl["tool_calls"],
            "oracle_tool_calls": orc["tool_calls"],
        })

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "baseline_results.jsonl", "w") as f:
        for r in baseline_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_dir / "oracle_results.jsonl", "w") as f:
        for r in oracle_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_dir / "manifest.jsonl", "w") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Summary
    print("\n" + "=" * 60)
    print("  MINING SUMMARY")
    print("=" * 60)
    print(f"  Total samples: {stats['total']}")
    print(f"  verify_critical: {stats['verify_critical']} ({100*stats['verify_critical']/stats['total']:.1f}%)")
    print(f"  verify_harmful:  {stats['verify_harmful']} ({100*stats['verify_harmful']/stats['total']:.1f}%)")
    print(f"  indifferent:     {stats['indifferent']} ({100*stats['indifferent']/stats['total']:.1f}%)")
    print("=" * 60)

    # Save stats
    with open(out_dir / "mining_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Mine verify-critical samples")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to PopQA test JSONL")
    parser.add_argument("--corpus-path", type=str, required=True,
                        help="Path to BM25 corpus")
    parser.add_argument("--direction-path", type=str, required=True,
                        help="Path to direction .npz file")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of samples to mine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="results/verify_mining",
                        help="Output directory")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--score-mode", type=str, default="any")
    args = parser.parse_args()

    out_dir = Path(args.out)

    # Load model
    print("[1/4] Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Load direction
    print("[2/4] Loading direction...")
    direction, direction_meta = load_direction(args.direction_path)
    direction_rms = float(np.sqrt(np.mean(direction ** 2)))
    print(f"  Direction shape: {direction.shape}, RMS: {direction_rms:.4f}")

    # Load dataset
    print("[3/4] Loading dataset...")
    dataset = PopQADataset(args.data_path)
    samples = dataset.get_subset(args.n_samples, seed=args.seed)
    print(f"  Selected {len(samples)} samples")

    # Setup search tool
    search_tool = SearchTool(corpus_path=args.corpus_path)
    tools = {"search": search_tool}

    # Create agent
    config = AgentConfig(
        max_steps=5,
        max_tokens_per_step=256,
        temperature=0.0,
        layer=args.layer,
        tools=list(tools.keys()),
        score_mode=args.score_mode,
    )
    agent = ReActAgent(
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        config=config,
        direction=direction,
        direction_rms=direction_rms,
    )

    # Mine
    print("[4/4] Mining verify-critical samples...")
    start_time = time.time()
    stats = mine_samples(agent, samples, out_dir, args.score_mode)
    elapsed = time.time() - start_time

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()

