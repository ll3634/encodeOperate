#!/usr/bin/env python3
"""PopQA experiment runner.

Replacement for HotpotQA as Tier 1 under-reliance dataset:
model *should* adopt the search tool to answer factual questions.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.popqa import PopQADataset, build_popqa_corpus
from tools.search_tool import SearchTool
from tools.calculator_tool import CalculatorTool
from tools.corruption import CorruptionWrapper, CorruptionConfig
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
        device_map="cuda:0",  # Force GPU, no CPU offload
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
    """Run PopQA experiment."""
    # Load dataset
    dataset = PopQADataset(args.data_path)

    # Filter by popularity if specified
    if args.pop_limit is not None:
        samples = dataset.get_subset_by_popularity(
            args.n_samples, max_pop=args.pop_limit, seed=args.seed
        )
        print(f"Filtered to {len(samples)} samples with s_pop <= {args.pop_limit}")
    else:
        samples = dataset.get_subset(args.n_samples, seed=args.seed)
    print(f"Running on {len(samples)} samples")

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

    # Setup tools
    search = SearchTool(str(corpus_path), top_k=3)
    calculator = CalculatorTool()

    # Apply corruption if specified
    if args.corruption > 0:
        corruption_config = CorruptionConfig(
            probability=args.corruption,
            mode="random",
            seed=args.seed,
        )
        search = CorruptionWrapper(search, corruption_config)

    tools = {"search": search, "calculator": calculator}

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
            # For PopQA we use all acceptable answers (list[str])
            gold_answers = sample.answers

            result = agent.run(
                question=sample.question,
                policy=policy,
                gold_answer=gold_answers,
                episode_id=sample.id,
                target_side="positive",  # PopQA: search is helpful
            )

            result_dict = result.to_dict()
            results.append(result_dict)
            f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
            f.flush()

    # Compute and save summary
    summary = compute_metrics(results)
    summary_path = str(output_path).replace(".jsonl", "_summary.json")
    save_summary(summary, summary_path)

    print(f"\n=== Results for {args.policy} on PopQA ===")
    print(f"Success rate: {summary.success_rate:.1%}")
    print(f"Mean tool calls: {summary.mean_tool_calls:.2f}")
    print(f"Mean tokens: {summary.mean_tokens:.0f}")
    if args.policy == "jes":
        print(f"Mean |ρ|: {summary.mean_abs_rho:.4f}")
        print(f"Achieved rate: {summary.achieved_rate:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Run PopQA experiments")
    parser.add_argument("--data-path", required=True, help="Path to PopQA JSONL file (e.g., data/popqa/popqa_test.jsonl)")
    parser.add_argument("--corpus-path", default="data/popqa/corpus.jsonl")
    parser.add_argument("--direction-path", required=True, help="Path to direction NPZ")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--policy", default="baseline", choices=["baseline", "jes", "force_adopt", "force_reject", "fixed_-0.1", "fixed_-0.2"])
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    # Popularity filter for hard samples
    parser.add_argument("--pop-limit", type=int, default=None,
                        help="Filter samples with s_pop <= pop_limit (lower = harder)")
    # JES parameters
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--rho-max", type=float, default=0.25)
    # Corruption
    parser.add_argument("--corruption", type=float, default=0.0)

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

