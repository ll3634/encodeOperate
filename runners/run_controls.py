#!/usr/bin/env python3
"""
Control experiments runner.
Tier 3 attribution: Random controls, forced baselines, corruption sweeps.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.hotpotqa import HotpotQADataset
from tools.search_tool import SearchTool
from tools.calculator_tool import CalculatorTool
from tools.corruption import CorruptionWrapper, CorruptionConfig
from agent.react_loop import ReActAgent, AgentConfig
from agent.policies import BaselinePolicy, FixedRhoPolicy, JESPolicy, ForcedPolicy, RandomControlPolicy
from steering.jes import JESConfig
from steering.directions import load_direction, generate_random_orthogonal_direction
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


def run_corruption_sweep(args):
    """Run corruption sweep experiment."""
    dataset = HotpotQADataset(args.data_path)
    samples = dataset.get_subset(args.n_samples, seed=args.seed)
    
    model, tokenizer = load_model_and_tokenizer(args.model)
    direction, _ = load_direction(args.direction_path)
    direction_rms = np.sqrt(np.mean(direction ** 2))
    
    corpus_path = Path(args.corpus_path)
    calculator = CalculatorTool()
    
    corruption_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    all_summaries = {}
    
    for p in corruption_levels:
        print(f"\n=== Corruption p={p} ===")
        
        # Create corrupted search tool
        base_search = SearchTool(str(corpus_path), top_k=3)
        if p > 0:
            corruption_config = CorruptionConfig(probability=p, mode="random", seed=args.seed)
            search = CorruptionWrapper(base_search, corruption_config)
        else:
            search = base_search
        
        tools = {"search": search, "calculator": calculator}
        
        agent_config = AgentConfig(max_steps=args.max_steps, layer=args.layer, position=args.position)
        agent = ReActAgent(model, tokenizer, tools, agent_config, direction=direction, direction_rms=direction_rms)
        
        # Run with JES policy
        jes_config = JESConfig(tau=args.tau, eps=args.eps, max_rho=args.rho_max)
        policy = JESPolicy(config=jes_config, direction=direction)
        
        results = []
        for sample in tqdm(samples, desc=f"p={p}"):
            result = agent.run(
                question=sample.question,
                policy=policy,
                gold_answer=sample.answer,
                episode_id=sample.id,
                target_side="positive",
            )
            results.append(result.to_dict())
        
        summary = compute_metrics(results)
        all_summaries[f"p={p}"] = summary.to_dict()
        
        print(f"  Success rate: {summary.success_rate:.1%}")
    
    # Save all summaries
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved corruption sweep results to {output_path}")


def run_random_control(args):
    """Run random direction control experiment."""
    dataset = HotpotQADataset(args.data_path)
    samples = dataset.get_subset(args.n_samples, seed=args.seed)
    
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Load real direction and generate random orthogonal
    real_direction, _ = load_direction(args.direction_path)
    random_direction = generate_random_orthogonal_direction(real_direction, seed=args.seed)
    
    real_rms = np.sqrt(np.mean(real_direction ** 2))
    random_rms = np.sqrt(np.mean(random_direction ** 2))
    
    search = SearchTool(args.corpus_path, top_k=3)
    calculator = CalculatorTool()
    tools = {"search": search, "calculator": calculator}
    
    all_summaries = {}
    
    for name, direction, direction_rms in [
        ("decision_direction", real_direction, real_rms),
        ("random_orthogonal", random_direction, random_rms),
    ]:
        print(f"\n=== Direction: {name} ===")
        
        agent_config = AgentConfig(max_steps=args.max_steps, layer=args.layer, position=args.position)
        agent = ReActAgent(model, tokenizer, tools, agent_config, direction=direction, direction_rms=direction_rms)
        
        jes_config = JESConfig(tau=args.tau, eps=args.eps, max_rho=args.rho_max)
        policy = JESPolicy(config=jes_config, direction=direction)
        
        results = []
        for sample in tqdm(samples, desc=name):
            result = agent.run(
                question=sample.question, policy=policy,
                gold_answer=sample.answer, episode_id=sample.id, target_side="positive",
            )
            results.append(result.to_dict())
        
        summary = compute_metrics(results)
        all_summaries[name] = summary.to_dict()
        print(f"  Success rate: {summary.success_rate:.1%}")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved random control results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run control experiments")
    parser.add_argument("--experiment", required=True, choices=["corruption_sweep", "random_control", "forced_baselines"])
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--direction-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--rho-max", type=float, default=0.25)
    
    args = parser.parse_args()
    
    if args.experiment == "corruption_sweep":
        run_corruption_sweep(args)
    elif args.experiment == "random_control":
        run_random_control(args)


if __name__ == "__main__":
    main()

