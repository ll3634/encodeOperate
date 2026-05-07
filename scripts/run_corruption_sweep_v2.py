#!/usr/bin/env python3
"""
Enhanced corruption sweep: multiple modes × p levels × policies.

Improvements over v1:
 - Multiple corruption modes (random, empty, noise, counterfactual)
 - Step-aware JES tau scheduling + do-no-harm guard
 - force_reject policy included
 - Per-cell JSONL + aggregate sweep summary + multi-panel plots

Usage:
  python scripts/run_corruption_sweep_v2.py \
      --data-path data/popqa/popqa_test.jsonl \
      --corpus-path data/popqa/corpus.jsonl \
      --direction-path steering/directions/direction_search_v3.npz \
      --n-samples 200 --out results/corruption_sweep_v2
"""

import json, argparse, itertools
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

# Import step-aware JES from run_eval
from scripts.run_eval import StepAwareJESPolicy, parse_tau_schedule

DEFAULT_P_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4]
DEFAULT_MODES = ["random", "empty", "noise"]
POLICIES = ["baseline", "force_adopt", "force_reject", "jes"]


def load_model_and_tokenizer(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True)
    model.eval()
    return model, tokenizer


def make_policy(pname, direction, jes_config, tau_schedule=None,
                guard_enabled=False, guard_threshold=-1.0):
    if pname == "baseline":
        return BaselinePolicy()
    elif pname == "force_adopt":
        return ForcedPolicy(force_adopt=True)
    elif pname == "force_reject":
        return ForcedPolicy(force_adopt=False)
    elif pname == "jes":
        return StepAwareJESPolicy(
            base_config=jes_config, direction=direction,
            tau_schedule=tau_schedule or {},
            guard_enabled=guard_enabled,
            guard_threshold=guard_threshold)
    else:
        raise ValueError(f"Unknown policy: {pname}")


def run_sweep_cell(agent, samples, policy, policy_name, p_val, mode,
                   dataset, seed, jes_params, bl0_by_id):
    """Run all samples for one (p, mode, policy) cell."""
    tag = f"{dataset}_p{p_val}_{mode}"
    run_id = make_run_id(tag, policy_name, len(samples), seed)
    records = []
    for sample in tqdm(samples, desc=f"p={p_val:.1f}/{mode}/{policy_name}", leave=False):
        if hasattr(policy, "reset_episode"):
            policy.reset_episode()
        for t in agent.tools.values():
            if hasattr(t, "reset_for_sample"):
                t.reset_for_sample(str(sample.id))

        result = agent.run(
            question=sample.question, policy=policy,
            gold_answer=sample.answers, episode_id=sample.id,
            target_side="positive")
        ep = result.to_dict()
        bl0_ok = bl0_by_id.get(sample.id) if bl0_by_id else None
        rec = convert_episode_to_record(
            ep, run_id=run_id, dataset=dataset,
            jes_params=jes_params if policy_name == "jes" else None,
            baseline_success=bl0_ok)
        rec["corruption_p"] = p_val
        rec["corruption_mode"] = mode
        records.append(rec)
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced corruption sweep: modes × p × policies")
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
    parser.add_argument("--jes-tau-schedule", default=None,
                        help='e.g. "1:3.0,2+:0.5"')
    parser.add_argument("--enable-guard", action="store_true")
    parser.add_argument("--guard-threshold", type=float, default=-1.0)
    parser.add_argument("--p-values", nargs="+", type=float,
                        default=DEFAULT_P_VALUES)
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES,
                        choices=["random", "empty", "noise", "counterfactual"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Dataset
    dataset = PopQADataset(args.data_path)
    samples = dataset.get_subset(args.n_samples, seed=args.seed)
    corpus_path = Path(args.corpus_path)
    if not corpus_path.exists():
        build_popqa_corpus(args.data_path, str(corpus_path))
    print(f"Selected {len(samples)} PopQA samples")

    # Model + direction
    model, tokenizer = load_model_and_tokenizer(args.model)
    direction, _ = load_direction(args.direction_path)
    direction_rms = float(np.sqrt(np.mean(direction ** 2)))

    base_search = SearchTool(str(corpus_path), top_k=3)
    calculator = CalculatorTool()
    agent_config = AgentConfig(
        max_steps=args.max_steps, layer=args.layer, position=args.position)

    jes_config = JESConfig(tau=args.tau, eps=args.eps, max_rho=args.rho_max)
    jes_params = {"tau": args.tau, "eps": args.eps, "rho_max": args.rho_max,
                  "layer": args.layer, "pos": args.position}
    tau_sched = parse_tau_schedule(args.jes_tau_schedule) if args.jes_tau_schedule else {}

    # ---- Main sweep ----
    bl0_by_id = None  # baseline p=0 ground truth
    sweep = {}  # (mode, p, policy) -> summary

    for mode in args.modes:
        for p_val in sorted(args.p_values):
            corr_seed = args.seed + int(p_val * 1000)
            print(f"\n{'='*60}  mode={mode}  p={p_val}  {'='*60}")

            for pname in POLICIES:
                # Fresh corruption wrapper per policy (same seed → identical)
                if p_val > 0:
                    cfg = CorruptionConfig(probability=p_val, mode=mode,
                                            seed=corr_seed)
                    search = CorruptionWrapper(base_search, cfg)
                else:
                    search = base_search

                tools = {"search": search, "calculator": calculator}
                agent = ReActAgent(model, tokenizer, tools, agent_config,
                                   direction=direction,
                                   direction_rms=direction_rms)

                policy = make_policy(pname, direction, jes_config,
                                     tau_sched, args.enable_guard,
                                     args.guard_threshold)
                recs = run_sweep_cell(agent, samples, policy, pname,
                                      p_val, mode, "popqa", args.seed,
                                      jes_params, bl0_by_id)

                tag = f"{mode}_p{p_val:.1f}_{pname}"
                write_records(recs, str(out / f"{tag}.jsonl"))
                summ = compute_run_summary(recs)
                summ.update({"corruption_p": p_val, "corruption_mode": mode})
                n_cat = sum(1 for r in recs
                            if r.get("flags", {}).get("catastrophic", False))
                summ["catastrophic_rate"] = n_cat / len(recs) if recs else 0
                write_summary(summ, str(out / f"{tag}_summary.json"))
                sweep[(mode, p_val, pname)] = summ

                sr = summ["success_rate"]
                cr = summ["catastrophic_rate"]
                print(f"  {pname:15s}: success={sr:.1%}  catastrophic={cr:.1%}")

                if p_val == 0 and pname == "baseline" and bl0_by_id is None:
                    bl0_by_id = {r["sample_id"]: r["is_correct"] for r in recs}

    # ---- Aggregate + plot ----
    _generate_sweep_report(sweep, args.p_values, args.modes, out)
    print(f"\nAll outputs in {out}/")


def _generate_sweep_report(sweep, p_values, modes, out_dir):
    """Generate sweep summary JSON and multi-panel matplotlib figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_sorted = sorted(p_values)
    summary = {"p_values": p_sorted, "modes": modes, "policies": POLICIES,
               "cells": {}}
    for (mode, p, pol), s in sweep.items():
        summary["cells"][f"{mode}_p{p}_{pol}"] = {
            "success_rate": s.get("success_rate", 0),
            "catastrophic_rate": s.get("catastrophic_rate", 0),
            "avg_tokens": s.get("avg_tokens_total", 0),
        }
    write_summary(summary, str(out_dir / "sweep_summary.json"))

    COLORS = {"baseline": "#2196F3", "force_adopt": "#FF9800",
              "force_reject": "#F44336", "jes": "#4CAF50"}
    LABELS = {"baseline": "Baseline", "force_adopt": "Force Adopt",
              "force_reject": "Force Reject", "jes": "JES (ours)"}

    n_modes = len(modes)
    fig, axes = plt.subplots(n_modes, 3, figsize=(16, 5 * n_modes),
                              squeeze=False)
    for mi, mode in enumerate(modes):
        for pol in POLICIES:
            c = COLORS.get(pol, "gray")
            lbl = LABELS.get(pol, pol)
            sr = [sweep.get((mode, p, pol), {}).get("success_rate", 0) * 100
                  for p in p_sorted]
            cr = [sweep.get((mode, p, pol), {}).get("catastrophic_rate", 0) * 100
                  for p in p_sorted]
            tk = [sweep.get((mode, p, pol), {}).get("avg_tokens_total", 0)
                  for p in p_sorted]
            axes[mi, 0].plot(p_sorted, sr, marker="o", color=c, label=lbl, lw=2)
            axes[mi, 1].plot(p_sorted, cr, marker="s", color=c, label=lbl, lw=2)
            axes[mi, 2].plot(p_sorted, tk, marker="^", color=c, label=lbl, lw=2)

        for ax, ylabel, title in [
            (axes[mi, 0], "Success %", f"Success vs p  [{mode}]"),
            (axes[mi, 1], "Catastrophic %", f"Catastrophic vs p  [{mode}]"),
            (axes[mi, 2], "Avg Tokens", f"Cost vs p  [{mode}]"),
        ]:
            ax.set_xlabel("Corruption Probability p")
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_xticks(p_sorted)

    fig.suptitle("Corruption Sweep v2: Multi-Mode Robustness",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig_path = out_dir / "corruption_curves_v2.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}  (+pdf)")


if __name__ == "__main__":
    main()

