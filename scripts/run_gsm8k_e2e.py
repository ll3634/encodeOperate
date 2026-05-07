#!/usr/bin/env python3
"""
GSM8K E2E runner: all four policies + counterfactual hard-subset mining.

Hard/tool-critical subset is defined as:
    baseline WRONG  AND  force_adopt CORRECT
(i.e., the calculator genuinely helps on these samples).

Usage:
  python scripts/run_gsm8k_e2e.py \
      --direction-path steering/directions/direction_calculator_v1.npz \
      --n-samples 200 --out results/gsm8k_200_unified
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
from agent.policies import BaselinePolicy, ForcedPolicy, JESPolicy
from steering.jes import JESConfig
from steering.directions import load_direction
from eval.unified_output import (
    convert_episode_to_record, compute_run_summary,
    write_records, write_summary, make_run_id,
)
from eval.subset_labeling import label_subsets, compute_subset_metrics, save_subset_report
from eval.paired_stats import full_paired_report


def load_model_and_tokenizer(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def run_policy(agent, samples, policy, policy_name, dataset, run_id,
               jes_params=None, bl_by_id=None, fa_by_id=None):
    """Run all samples with a given policy, return unified records."""
    records = []
    for sample in tqdm(samples, desc=f"Running {policy_name}"):
        target_side = "positive"  # GSM8K: calculator can help
        result = agent.run(
            question=sample.question,
            policy=policy,
            gold_answer=sample.answer,
            episode_id=sample.id,
            target_side=target_side,
        )
        ep = result.to_dict()
        ep["difficulty"] = sample.difficulty
        bl_ok = bl_by_id.get(sample.id) if bl_by_id else None
        fa_ok = fa_by_id.get(sample.id) if fa_by_id else None
        rec = convert_episode_to_record(
            ep, run_id=run_id, dataset=dataset,
            jes_params=jes_params if policy_name == "jes" else None,
            baseline_success=bl_ok, force_adopt_success=fa_ok,
        )
        records.append(rec)
    return records


def main():
    parser = argparse.ArgumentParser(description="GSM8K E2E: all policies + hard subset mining")
    parser.add_argument("--data-path", default=None, help="Local GSM8K JSONL (default: HuggingFace)")
    parser.add_argument(
        "--direction-path",
        default="steering/directions/direction_calculator_v1.npz",
        help="Steering direction NPZ (default: calculator_v1; best-practice for GSM8K).",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--rho-max", type=float, default=0.75)
    parser.add_argument("--score-mode", default="numeric",
                        choices=["numeric", "any", "exact"],
                        help="Answer scoring mode (default: numeric; recommended for GSM8K)")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = GSM8KDataset(args.data_path)
    samples = dataset.get_subset(args.n_samples, seed=args.seed)
    print(f"Selected {len(samples)} samples")

    # Load model + direction
    model, tokenizer = load_model_and_tokenizer(args.model)
    direction, _ = load_direction(args.direction_path)
    direction_rms = float(np.sqrt(np.mean(direction ** 2)))

    # Setup agent
    calculator = CalculatorTool()
    agent_config = AgentConfig(max_steps=args.max_steps, layer=args.layer, position=args.position,
                               score_mode=args.score_mode)
    agent = ReActAgent(model, tokenizer, {"calculator": calculator},
                       agent_config, direction=direction, direction_rms=direction_rms)

    jes_params = {"tau": args.tau, "eps": args.eps, "rho_max": args.rho_max,
                  "layer": args.layer, "pos": args.position}

    # ---- Run all four policies ----
    policies = {
        "baseline": BaselinePolicy(),
        "force_adopt": ForcedPolicy(force_adopt=True),
        "force_reject": ForcedPolicy(force_adopt=False),
        "jes": JESPolicy(
            config=JESConfig(tau=args.tau, eps=args.eps, max_rho=args.rho_max),
            direction=direction,
        ),
    }

    all_records = {}
    for pname, policy in policies.items():
        rid = make_run_id("gsm8k", pname, len(samples), args.seed)
        bl_by_id = {r["sample_id"]: r["is_correct"] for r in all_records["baseline"]} if "baseline" in all_records else None
        fa_by_id = {r["sample_id"]: r["is_correct"] for r in all_records["force_adopt"]} if "force_adopt" in all_records else None
        recs = run_policy(agent, samples, policy, pname, "gsm8k", rid,
                          jes_params=jes_params, bl_by_id=bl_by_id, fa_by_id=fa_by_id)
        all_records[pname] = recs
        write_records(recs, str(out / f"{pname}.jsonl"))
        summ = compute_run_summary(
            recs,
            baseline_records=all_records.get("baseline") if pname != "baseline" else None,
            force_adopt_records=all_records.get("force_adopt") if pname != "baseline" else None,
        )
        write_summary(summ, str(out / f"{pname}_summary.json"))
        print(f"  {pname}: {summ.get('success_rate', 0):.1%}")

    # ---- Hard subset mining (counterfactual) ----
    print("\nHard subset mining ...")
    subsets = label_subsets(all_records["baseline"], all_records["force_adopt"])
    for sname, sids in subsets.items():
        print(f"  {sname}: {len(sids)}")
    sub_metrics = compute_subset_metrics(subsets, all_records)
    save_subset_report(subsets, sub_metrics, str(out / "subset_report.json"))

    # ---- Paired stats ----
    print("\nPaired statistics (JES vs Baseline) ...")
    report = full_paired_report(
        all_records["baseline"], all_records["jes"],
        policy_name="jes", indifferent_ids=subsets["indifferent"],
    )
    write_summary(report, str(out / "paired_stats_jes.json"))
    mcn = report["mcnemar"]
    bci = report["bootstrap_success_diff"]
    print(f"  McNemar p={mcn['mcnemar_p']:.4f}")
    print(f"  ΔSuccess: {bci['observed']:+.4f}  CI [{bci['ci_lower']:+.4f}, {bci['ci_upper']:+.4f}]")
    print(f"\nDone. All outputs in {out}/")


if __name__ == "__main__":
    main()

