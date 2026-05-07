#!/usr/bin/env python3
"""
Red Flag Experiment: True test of JES ability to reject misleading tool outputs.

This experiment creates samples where the search tool deliberately returns
WRONG answers. The model should learn to reject these and answer from memory.

This is the DEFINITIVE test for JES's ability to avoid tool over-reliance.
"""

import json
import argparse
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.popqa import PopQADataset, build_popqa_corpus
from tools.search_tool import SearchTool
from tools.calculator_tool import CalculatorTool
from tools.corruption import DeliberateWrongAnswerWrapper
from agent.react_loop import ReActAgent, AgentConfig
from agent.policies import BaselinePolicy, FixedRhoPolicy, JESPolicy, ForcedPolicy
from steering.jes import JESConfig
from steering.directions import load_direction
from eval.metrics import compute_metrics, save_summary


def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_wrong_answers(samples: List, seed: int = 42) -> Dict[str, str]:
    """
    Generate deliberately wrong answers for each sample.
    Uses other samples' answers as wrong answers for diversity.
    """
    rng = random.Random(seed)
    
    # Collect all unique answers
    all_answers = []
    for s in samples:
        for ans in s.answers:
            all_answers.append(ans)
    
    # For each sample, pick a random different answer
    wrong_answers = {}
    for s in samples:
        # Pick answers that are NOT correct for this sample
        correct_set = set(a.lower() for a in s.answers)
        candidates = [a for a in all_answers if a.lower() not in correct_set]
        
        if candidates:
            wrong_answers[s.id] = rng.choice(candidates)
        else:
            # Fallback: just use a generic wrong answer
            wrong_answers[s.id] = "Unknown Entity"
    
    return wrong_answers


def get_policy(policy_name: str, direction: np.ndarray = None, config: dict = None):
    """Create policy from name."""
    config = config or {}

    if policy_name == "baseline":
        return BaselinePolicy()
    elif policy_name.startswith("fixed_"):
        rho = float(policy_name.split("_")[1])
        return FixedRhoPolicy(rho=rho)
    elif policy_name == "jes":
        jes_config = JESConfig(
            tau=config.get("tau", 0.2),
            eps=config.get("eps", 0.02),
            max_rho=config.get("max_rho", 0.25),
        )
        return JESPolicy(config=jes_config, direction=direction)
    elif policy_name == "force_adopt":
        return ForcedPolicy(force_adopt=True)
    elif policy_name == "force_reject":
        return ForcedPolicy(force_adopt=False)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")


def run_experiment(args):
    """Run Red Flag experiment."""
    # Load dataset - use HIGH popularity (easy for model) so it can answer without tool
    dataset = PopQADataset(args.data_path)
    
    # Get samples with high popularity (model can answer from memory)
    all_samples = dataset.samples
    high_pop_samples = [s for s in all_samples if s.s_pop >= args.min_pop]
    print(f"Found {len(high_pop_samples)} samples with s_pop >= {args.min_pop}")
    
    # Random subset
    rng = random.Random(args.seed)
    samples = rng.sample(high_pop_samples, min(args.n_samples, len(high_pop_samples)))
    print(f"Running on {len(samples)} samples")

    # Generate wrong answers for each sample
    wrong_answers = generate_wrong_answers(samples, args.seed)
    print(f"Generated wrong answers for {len(wrong_answers)} samples")
    print(f"Example: {samples[0].id} -> correct: {samples[0].answers[0]}, wrong: {wrong_answers[samples[0].id]}")

    # Build corpus if needed
    corpus_path = Path(args.corpus_path)
    if not corpus_path.exists():
        print(f"Building PopQA corpus at {corpus_path}...")
        build_popqa_corpus(args.data_path, str(corpus_path))

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Load direction
    direction, direction_meta = load_direction(args.direction_path)
    direction_rms = np.sqrt(np.mean(direction ** 2))

    # Setup tools with deliberate wrong answer corruption
    base_search = SearchTool(str(corpus_path), top_k=3)
    corrupted_search = DeliberateWrongAnswerWrapper(
        tool=base_search,
        sample_id_to_wrong_answer=wrong_answers,
        probability=1.0,  # Always corrupt
        seed=args.seed
    )
    calculator = CalculatorTool()
    tools = {"search": corrupted_search, "calculator": calculator}

    # Setup agent
    agent_config = AgentConfig(
        max_steps=args.max_steps,
        layer=args.layer,
        position=args.position,
    )
    agent = ReActAgent(
        model, tokenizer, tools, agent_config,
        direction=direction, direction_rms=direction_rms,
    )

    # Setup policy
    policy_config = {"tau": args.tau, "eps": args.eps, "max_rho": args.rho_max}
    policy = get_policy(args.policy, direction, policy_config)

    # Run episodes
    results = []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in tqdm(samples, desc=f"Running {args.policy}"):
            # Set current sample ID for corruption tracking
            corrupted_search.set_current_sample(sample.id)

            gold_answers = sample.answers

            result = agent.run(
                question=sample.question,
                policy=policy,
                gold_answer=gold_answers,
                episode_id=sample.id,
                target_side="negative",  # Red Flag: should NOT use corrupted tool
            )

            result_dict = result.to_dict()
            # Add corruption info
            result_dict["wrong_answer_injected"] = wrong_answers.get(sample.id, "")
            result_dict["tool_was_corrupted"] = corrupted_search.was_corrupted
            results.append(result_dict)
            f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
            f.flush()

    # Compute and save summary
    summary = compute_metrics(results)
    summary_path = str(output_path).replace(".jsonl", "_summary.json")
    save_summary(summary, summary_path)

    print(f"\n=== Results for {args.policy} on Red Flag Experiment ===")
    print(f"Success rate: {summary.success_rate:.1%}")
    print(f"Mean tool calls: {summary.mean_tool_calls:.2f}")
    print(f"Mean tokens: {summary.mean_tokens:.0f}")
    print(f"Tool corruption rate: {corrupted_search.corruption_rate:.1%}")
    if args.policy == "jes":
        print(f"Mean |ρ|: {summary.mean_abs_rho:.4f}")
        print(f"Achieved rate: {summary.achieved_rate:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Run Red Flag experiment (corrupted tool outputs)")
    parser.add_argument("--data-path", required=True, help="Path to PopQA JSONL file")
    parser.add_argument("--corpus-path", default="data/popqa/corpus.jsonl")
    parser.add_argument("--direction-path", required=True, help="Path to direction NPZ")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--policy", default="baseline",
                        choices=["baseline", "jes", "force_adopt", "force_reject", "fixed_0.1", "fixed_0.2"])
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--min-pop", type=int, default=1000,
                        help="Minimum s_pop (higher = model can answer from memory)")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    # JES parameters
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--rho-max", type=float, default=0.25)

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

