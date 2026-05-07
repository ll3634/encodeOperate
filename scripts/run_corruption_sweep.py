#!/usr/bin/env python3
"""
Corruption sweep: tool-harmful environment evaluation.

Sweeps corruption probability p ∈ {0.0, 0.1, 0.2, 0.3, 0.5} on PopQA,
running baseline / force_adopt / JES for each p value.

Reports:
  - success vs p curve
  - catastrophic_rate vs p  (baseline-correct at p=0 → policy-wrong at p>0)
  - cost vs p (tokens / tool_calls)

Usage:
  python scripts/run_corruption_sweep.py \
      --data-path data/popqa/popqa_test.jsonl \
      --corpus-path data/popqa/corpus.jsonl \
      --direction-path steering/directions/direction_search_v3.npz \
      --n-samples 200 --out results/corruption_sweep
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
from agent.policies import BaselinePolicy, ForcedPolicy, JESPolicy
from steering.jes import JESConfig
from steering.directions import load_direction
from eval.unified_output import (
    convert_episode_to_record, compute_run_summary,
    write_records, write_summary, make_run_id,
)


DEFAULT_P_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5]
POLICIES = ["baseline", "force_adopt", "jes"]


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


def make_policy(pname, direction, jes_config):
    if pname == "baseline":
        return BaselinePolicy()
    elif pname == "force_adopt":
        return ForcedPolicy(force_adopt=True)
    elif pname == "jes":
        return JESPolicy(config=jes_config, direction=direction)
    else:
        raise ValueError(f"Unknown policy: {pname}")


def run_sweep_cell(agent, samples, policy, policy_name, p_val,
                   dataset, seed, jes_params, bl0_by_id):
    """Run all samples for one (p, policy) cell. Return unified records."""
    run_id = make_run_id(f"{dataset}_p{p_val}", policy_name, len(samples), seed)
    records = []
    for sample in tqdm(samples, desc=f"p={p_val:.1f} {policy_name}", leave=False):
        # Per-sample RNG reset so all policies see identical corruption
        search_tool = agent.tools.get("search")
        if hasattr(search_tool, 'reset_for_sample'):
            search_tool.reset_for_sample(str(sample.id))
        result = agent.run(
            question=sample.question,
            policy=policy,
            gold_answer=sample.answers,
            episode_id=sample.id,
            target_side="positive",
        )
        ep = result.to_dict()
        # baseline_success from p=0 baseline (for catastrophic flag)
        bl0_ok = bl0_by_id.get(sample.id) if bl0_by_id else None
        rec = convert_episode_to_record(
            ep, run_id=run_id, dataset=dataset,
            jes_params=jes_params if policy_name == "jes" else None,
            baseline_success=bl0_ok,
        )
        # Extra corruption metadata
        rec["corruption_p"] = p_val
        records.append(rec)
    return records


def main():
    parser = argparse.ArgumentParser(description="Corruption sweep: success/safety vs corruption p")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--corpus-path", default="data/popqa/corpus.jsonl")
    parser.add_argument(
        "--direction-path",
        default="steering/directions/direction_search_v3.npz",
        help="Steering direction NPZ (default: search_v3; best-practice for PopQA).",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--rho-max", type=float, default=0.75)
    parser.add_argument("--p-values", nargs="+", type=float, default=DEFAULT_P_VALUES)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = PopQADataset(args.data_path)
    samples = dataset.get_subset(args.n_samples, seed=args.seed)
    print(f"Selected {len(samples)} samples")

    # Build corpus if needed
    corpus_path = Path(args.corpus_path)
    if not corpus_path.exists():
        build_popqa_corpus(args.data_path, str(corpus_path))

    # Load model + direction
    model, tokenizer = load_model_and_tokenizer(args.model)
    direction, _ = load_direction(args.direction_path)
    direction_rms = float(np.sqrt(np.mean(direction ** 2)))

    base_search = SearchTool(str(corpus_path), top_k=3)
    calculator = CalculatorTool()

    agent_config = AgentConfig(
        max_steps=args.max_steps, layer=args.layer, position=args.position,
    )

    jes_config = JESConfig(tau=args.tau, eps=args.eps, max_rho=args.rho_max)
    jes_params = {"tau": args.tau, "eps": args.eps, "rho_max": args.rho_max,
                  "layer": args.layer, "pos": args.position}

    # ---- Main sweep loop ----
    # First: run baseline at p=0 to get ground-truth correctness
    bl0_by_id = None  # populated after first baseline run
    sweep_results = {}  # (p, policy) -> summary dict
    all_records_by_cell = {}

    for p_val in sorted(args.p_values):
        print(f"\n{'='*60}")
        print(f"  Corruption p = {p_val}")
        print(f"{'='*60}")

        # Corruption seed for this p level (same for all policies)
        corr_seed = args.seed + int(p_val * 1000)

        for pname in POLICIES:
            # Bug #2 fix: create a FRESH CorruptionWrapper per policy
            # with the SAME seed so all policies see identical corruption.
            if p_val > 0:
                corr_config = CorruptionConfig(
                    probability=p_val, mode="random",
                    seed=corr_seed,
                )
                search = CorruptionWrapper(base_search, corr_config)
            else:
                search = base_search

            tools = {"search": search, "calculator": calculator}
            agent = ReActAgent(
                model, tokenizer, tools, agent_config,
                direction=direction, direction_rms=direction_rms,
            )

            policy = make_policy(pname, direction, jes_config)
            recs = run_sweep_cell(
                agent, samples, policy, pname, p_val,
                "popqa", args.seed, jes_params, bl0_by_id,
            )
            cell_key = (p_val, pname)
            all_records_by_cell[cell_key] = recs

            # Write per-cell JSONL + summary
            tag = f"p{p_val:.1f}_{pname}"
            write_records(recs, str(out / f"{tag}.jsonl"))

            summ = compute_run_summary(recs)
            summ["corruption_p"] = p_val
            # catastrophic rate
            n_cat = sum(1 for r in recs if r.get("flags", {}).get("catastrophic", False))
            summ["catastrophic_count"] = n_cat
            summ["catastrophic_rate"] = n_cat / len(recs) if recs else 0

            write_summary(summ, str(out / f"{tag}_summary.json"))
            sweep_results[cell_key] = summ

            succ = summ["success_rate"]
            cat_r = summ["catastrophic_rate"]
            print(f"  {pname:15s}: success={succ:.1%}  catastrophic={cat_r:.1%}")

            # After baseline p=0, capture ground-truth
            if p_val == 0 and pname == "baseline":
                bl0_by_id = {r["sample_id"]: r["is_correct"] for r in recs}

    # ---- Generate sweep summary + curves ----
    _generate_sweep_report(sweep_results, args.p_values, out)
    print(f"\nAll outputs in {out}/")


def _generate_sweep_report(sweep_results, p_values, out_dir):
    """Generate sweep summary JSON and matplotlib curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_sorted = sorted(p_values)
    summary = {"p_values": p_sorted, "policies": {}}

    for pname in POLICIES:
        pdata = {
            "success_rate": [],
            "catastrophic_rate": [],
            "avg_tokens": [],
            "avg_tool_calls": [],
        }
        for p in p_sorted:
            s = sweep_results.get((p, pname), {})
            pdata["success_rate"].append(s.get("success_rate", 0))
            pdata["catastrophic_rate"].append(s.get("catastrophic_rate", 0))
            pdata["avg_tokens"].append(s.get("avg_tokens_total", 0))
            pdata["avg_tool_calls"].append(s.get("avg_tool_calls", 0))
        summary["policies"][pname] = pdata

    write_summary(summary, str(out_dir / "sweep_summary.json"))

    # Colors
    COLORS = {"baseline": "#2196F3", "force_adopt": "#FF9800", "jes": "#4CAF50"}
    LABELS = {"baseline": "Baseline", "force_adopt": "Force Adopt", "jes": "JES (ours)"}

    # Fig: success & catastrophic vs p (dual y-axis)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    for pname in POLICIES:
        d = summary["policies"][pname]
        c = COLORS.get(pname, "gray")
        lbl = LABELS.get(pname, pname)
        ax1.plot(p_sorted, [v * 100 for v in d["success_rate"]],
                 marker="o", color=c, label=lbl, linewidth=2)
        ax2.plot(p_sorted, [v * 100 for v in d["catastrophic_rate"]],
                 marker="s", color=c, label=lbl, linewidth=2)
        ax3.plot(p_sorted, d["avg_tokens"],
                 marker="^", color=c, label=lbl, linewidth=2)

    for ax, ylabel, title in [
        (ax1, "Success Rate (%)", "Success vs Corruption"),
        (ax2, "Catastrophic Rate (%)", "Catastrophic Failures vs Corruption"),
        (ax3, "Avg Tokens", "Cost vs Corruption"),
    ]:
        ax.set_xlabel("Corruption Probability p", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_xticks(p_sorted)

    plt.tight_layout()
    fig_path = out_dir / "corruption_curves.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved corruption curves: {fig_path}  (+pdf)")


if __name__ == "__main__":
    main()

