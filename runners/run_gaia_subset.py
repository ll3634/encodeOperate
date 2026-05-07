#!/usr/bin/env python3
"""
GAIA conflict subset experiment runner.
Tier 2 application: Real E2E tasks with mixed adopt/reject scenarios.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.gaia import GAIADataset
from tools.search_tool import SearchTool
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
    """Run GAIA subset experiment."""
    # Load dataset
    dataset = GAIADataset(args.data_path)
    
    # Apply heuristic labels if not already labeled
    dataset.apply_heuristic_labels()
    
    # Get conflict subset or all samples
    if args.conflict_only:
        samples = dataset.get_conflict_subset()
    else:
        samples = list(dataset.samples)
    
    samples = samples[:args.n_samples]
    print(f"Running on {len(samples)} samples")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Load direction
    direction, direction_meta = load_direction(args.direction_path)
    direction_rms = np.sqrt(np.mean(direction ** 2))
    
    # Setup tools
    if args.corpus_path and Path(args.corpus_path).exists():
        search = SearchTool(args.corpus_path, top_k=3)
    else:
        def mock_search(query):
            return f"Search results for: {query[:50]}..."
        search = mock_search
    
    calculator = CalculatorTool()
    tools = {"search": search, "calculator": calculator}
    
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
            # Determine target side based on heuristic
            if sample.should_use_tool is True:
                target_side = "positive"
            elif sample.should_use_tool is False:
                target_side = "negative"
            else:
                target_side = "positive"  # Default to adopt for ambiguous
            
            result = agent.run(
                question=sample.question,
                policy=policy,
                gold_answer=sample.answer,
                episode_id=sample.id,
                target_side=target_side,
            )
            
            result_dict = result.to_dict()
            result_dict["should_use_tool"] = sample.should_use_tool
            result_dict["conflict_type"] = sample.conflict_type
            results.append(result_dict)
            f.write(json.dumps(result_dict) + "\n")
            f.flush()
    
    # Compute and save summary
    summary = compute_metrics(results)
    summary_path = str(output_path).replace(".jsonl", "_summary.json")
    save_summary(summary, summary_path)
    
    print(f"\n=== Results for {args.policy} ===")
    print(f"Success rate: {summary.success_rate:.1%}")
    print(f"Mean tool calls: {summary.mean_tool_calls:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Run GAIA subset experiments")
    parser.add_argument("--data-path", required=True, help="Path to GAIA JSONL")
    parser.add_argument("--corpus-path", help="Optional path to search corpus")
    parser.add_argument("--direction-path", required=True, help="Path to direction NPZ")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--policy", default="baseline")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conflict-only", action="store_true")
    # JES parameters
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--rho-max", type=float, default=0.25)
    
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

