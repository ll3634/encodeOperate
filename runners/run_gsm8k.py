#!/usr/bin/env python3
"""GSM8K experiment runner.

Supports two regimes:
- Red Flag (easy-only): model should NOT use calculator.
- Stealth (hard-only): model should use calculator.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.gsm8k import GSM8KDataset
from tools.calculator_tool import CalculatorTool
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
    """Run GSM8K experiment."""
    # Load dataset
    dataset = GSM8KDataset(args.data_path)
    
    if args.easy_only and args.hard_only:
        raise ValueError("Choose at most one of --easy-only / --hard-only")

    # Filter by difficulty if specified
    fixed_target_side = None
    if args.easy_only:
        samples = dataset.get_easy_subset(args.n_samples, seed=args.seed)
        fixed_target_side = "negative"  # should reject calculator
        print(f"Selected {len(samples)} easy samples (Red-Flag: should reject calculator)")
    elif args.hard_only:
        samples = dataset.get_hard_subset(args.n_samples, seed=args.seed)
        fixed_target_side = "positive"  # should adopt calculator
        print(f"Selected {len(samples)} hard samples (Stealth: should adopt calculator)")
    else:
        samples = dataset.get_subset(args.n_samples, seed=args.seed)
    
    print(f"Running on {len(samples)} samples")
    
    # Show difficulty distribution
    diffs = {}
    for s in samples:
        diffs[s.difficulty] = diffs.get(s.difficulty, 0) + 1
    print(f"Difficulty distribution: {diffs}")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Load direction
    direction, direction_meta = load_direction(args.direction_path)
    direction_rms = np.sqrt(np.mean(direction ** 2))
    
    # Setup tools - only calculator for GSM8K
    calculator = CalculatorTool()
    tools = {"calculator": calculator}
    
    # Setup agent
    agent_config = AgentConfig(
        max_steps=args.max_steps,
        layer=args.layer,
        position=args.position,
    )
    agent = ReActAgent(
        model, tokenizer, tools, agent_config,
        direction=direction, direction_rms=direction_rms
    )
    
    # Setup policy
    policy_config = {"tau": args.tau, "eps": args.eps, "max_rho": args.rho_max}
    policy = get_policy(args.policy, direction, policy_config)
    
    # Run episodes
    results = []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for sample in tqdm(samples, desc=f"Running {args.policy}"):
            # If we sampled a mixed set, decide target_side per sample.
            target_side = fixed_target_side
            if target_side is None:
                target_side = "positive" if getattr(sample, "should_use_tool", False) else "negative"

            result = agent.run(
                question=sample.question,
                policy=policy,
                gold_answer=sample.answer,
                episode_id=sample.id,
                target_side=target_side,
            )
            
            result_dict = result.to_dict()
            result_dict["difficulty"] = sample.difficulty  # Track difficulty
            result_dict["should_use_tool"] = getattr(sample, "should_use_tool", False)
            result_dict["target_side"] = target_side
            results.append(result_dict)
            f.write(json.dumps(result_dict) + "\n")
            f.flush()
    
    # Compute and save summary
    summary = compute_metrics(results)
    summary_path = str(output_path).replace(".jsonl", "_summary.json")
    save_summary(summary, summary_path)
    
    print(f"\n=== Results for {args.policy} on GSM8K ===")
    print(f"Success rate: {summary.success_rate:.1%}")
    print(f"Mean tool calls: {summary.mean_tool_calls:.2f}")
    print(f"Mean tokens: {summary.mean_tokens:.0f}")
    if args.policy == "jes":
        print(f"Mean |ρ|: {summary.mean_abs_rho:.4f}")
        print(f"Achieved rate: {summary.achieved_rate:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Run GSM8K experiments")
    parser.add_argument(
        "--data-path",
        default="data/gsm8k/gsm8k_test.jsonl",
        help="Path to GSM8K JSONL (default: local repo copy)",
    )
    parser.add_argument("--direction-path", required=True, help="Path to direction NPZ")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--policy", default="baseline", 
                        choices=["baseline", "jes", "force_adopt", "force_reject", "fixed_-0.1", "fixed_0.1"])
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--easy-only", action="store_true", help="Only use easy problems (Red Flag: should reject calculator)")
    parser.add_argument("--hard-only", action="store_true", help="Only use hard problems (Stealth: should adopt calculator)")
    # JES parameters
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--rho-max", type=float, default=0.25)
    
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

