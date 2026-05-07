#!/usr/bin/env python3
"""
PopQA E2E runner: all four policies in one command.

Runs baseline / force_adopt / force_reject / JES sequentially on the same
sample set, then automatically generates:
  - Unified JSONL + summary per policy
  - Subset labeling (Stealth / RedFlag / Indifferent)
  - Paired statistics (McNemar + bootstrap CI)
  - Publication figures (FigA, Table1, Pareto)

Usage:
  python scripts/run_popqa_e2e.py \
      --data-path data/popqa/popqa_test.jsonl \
      --corpus-path data/popqa/corpus.jsonl \
      --direction-path steering/directions/direction_search_v3.npz \
      --n-samples 500 --out results/popqa_500_unified
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
from reporting.make_figures import fig_a_subset_bars, table1_main_results, fig_c_pareto


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


def main():
    parser = argparse.ArgumentParser(description="PopQA E2E: all policies + stats + figures")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--corpus-path", default="data/popqa/corpus.jsonl")
    parser.add_argument(
        "--direction-path",
        default="steering/directions/direction_search_v3.npz",
        help="Steering direction NPZ (default: search_v3; best-practice for PopQA).",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--rho-max", type=float, default=0.75)
    parser.add_argument("--pop-limit", type=int, default=None,
                        help="Filter s_pop <= pop_limit (lower = harder)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Load dataset
    dataset = PopQADataset(args.data_path)
    if args.pop_limit is not None:
        samples = dataset.get_subset_by_popularity(
            args.n_samples, max_pop=args.pop_limit, seed=args.seed)
    else:
        samples = dataset.get_subset(args.n_samples, seed=args.seed)
    print(f"Selected {len(samples)} samples")

    # Build corpus
    corpus_path = Path(args.corpus_path)
    if not corpus_path.exists():
        build_popqa_corpus(args.data_path, str(corpus_path))

    # Load model + direction
    model, tokenizer = load_model_and_tokenizer(args.model)
    direction, _ = load_direction(args.direction_path)
    direction_rms = float(np.sqrt(np.mean(direction ** 2)))

    # Tools
    search = SearchTool(str(corpus_path), top_k=3)
    calculator = CalculatorTool()
    tools = {"search": search, "calculator": calculator}

    # Agent
    agent_config = AgentConfig(
        max_steps=args.max_steps, layer=args.layer, position=args.position)
    agent = ReActAgent(model, tokenizer, tools, agent_config,
                       direction=direction, direction_rms=direction_rms)

    jes_config = JESConfig(tau=args.tau, eps=args.eps, max_rho=args.rho_max)
    jes_params = {"tau": args.tau, "eps": args.eps, "rho_max": args.rho_max,
                  "layer": args.layer, "pos": args.position}

    policies = {
        "baseline": BaselinePolicy(),
        "force_adopt": ForcedPolicy(force_adopt=True),
        "force_reject": ForcedPolicy(force_adopt=False),
        "jes": JESPolicy(config=jes_config, direction=direction),
    }

    # ---- Run all policies ----
    all_records = {}
    summaries = {}

    for pname, policy in policies.items():
        print(f"\n{'='*50}  {pname}  {'='*50}")
        run_id = make_run_id("popqa", pname, len(samples), args.seed)
        records = []
        for sample in tqdm(samples, desc=pname):
            bl_by_id = {r["sample_id"]: r["is_correct"]
                        for r in all_records["baseline"]} if "baseline" in all_records else None
            fa_by_id = {r["sample_id"]: r["is_correct"]
                        for r in all_records["force_adopt"]} if "force_adopt" in all_records else None

            result = agent.run(
                question=sample.question, policy=policy,
                gold_answer=sample.answers, episode_id=sample.id,
                target_side="positive",
            )
            ep = result.to_dict()
            rec = convert_episode_to_record(
                ep, run_id=run_id, dataset="popqa",
                jes_params=jes_params if pname == "jes" else None,
                baseline_success=bl_by_id.get(sample.id) if bl_by_id else None,
                force_adopt_success=fa_by_id.get(sample.id) if fa_by_id else None,
            )
            records.append(rec)

        all_records[pname] = records
        write_records(records, str(out / f"{pname}.jsonl"))

        bl_recs = all_records.get("baseline") if pname != "baseline" else None
        fa_recs = all_records.get("force_adopt") if pname != "baseline" else None
        summ = compute_run_summary(records,
                                   baseline_records=bl_recs,
                                   force_adopt_records=fa_recs)
        summaries[pname] = summ
        write_summary(summ, str(out / f"{pname}_summary.json"))
        print(f"  {pname}: success={summ['success_rate']:.1%}  "
              f"regress={summ.get('regression_rate',0):.1%}  "
              f"rescue={summ.get('rescue_rate',0):.1%}")

    # ---- Subset labeling ----
    print("\n" + "="*50 + "  Subset labeling  " + "="*50)
    subsets = label_subsets(all_records["baseline"], all_records["force_adopt"])
    for sname, sids in subsets.items():
        print(f"  {sname}: {len(sids)}")

    sub_metrics = compute_subset_metrics(subsets, all_records)
    save_subset_report(subsets, sub_metrics, str(out / "subset_report.json"))

    # ---- Paired statistics (JES vs Baseline) ----
    print("\n" + "="*50 + "  Paired stats  " + "="*50)
    for pname in ["force_adopt", "force_reject", "jes"]:
        if pname not in all_records:
            continue
        report = full_paired_report(
            all_records["baseline"], all_records[pname],
            policy_name=pname, indifferent_ids=subsets["indifferent"],
        )
        write_summary(report, str(out / f"paired_stats_{pname}.json"))
        mcn = report["mcnemar"]
        bci = report["bootstrap_success_diff"]
        print(f"  {pname} vs baseline:")
        print(f"    McNemar p={mcn['mcnemar_p']:.4f}  "
              f"(b={mcn['b_regressed']}, c={mcn['c_rescued']})")
        print(f"    ΔSuccess: {bci['observed']:+.4f}  "
              f"95% CI [{bci['ci_lower']:+.4f}, {bci['ci_upper']:+.4f}]")
        dn = report["do_no_harm"]
        print(f"    Do-no-harm: regress={dn['regression_rate']:.1%}  "
              f"rescue={dn['rescue_rate']:.1%}  net={dn['net_gain']}")

    # ---- Figures ----
    print("\n" + "="*50 + "  Generating figures  " + "="*50)
    table1_main_results(summaries, str(fig_dir / "table1"))
    fig_c_pareto(summaries, str(fig_dir / "fig_c_pareto.png"))
    fig_a_subset_bars(sub_metrics, str(fig_dir / "fig_a_subsets.png"))

    print(f"\n{'='*50}")
    print(f"All outputs in {out}/")
    print(f"Figures in {fig_dir}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

