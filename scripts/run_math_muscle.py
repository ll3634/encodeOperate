#!/usr/bin/env python3
"""MATH Capability Muscle Experiment.

Pipeline (mirrors GSM8K muscle):
  1) Run baseline/force_adopt/force_reject/jes
  2) Counterfactual label subsets (tool_critical/tool_harmful/indifferent + stealth subdivisions)
  3) Paired stats + headline table

Usage:
  python scripts/run_math_muscle.py \
    --direction-path steering/directions/direction_calculator_v1.npz \
    --n-samples 200 --out results/math_muscle
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.math import MathDataset
from tools.calculator_tool import CalculatorTool
from agent.react_loop import ReActAgent, AgentConfig
from agent.policies import BaselinePolicy, ForcedPolicy
from steering.jes import JESConfig
from steering.directions import load_direction
from eval.unified_output import (
    convert_episode_to_record, write_records, write_summary, make_run_id,
)
from eval.paired_stats import mcnemar_test, bootstrap_ci, do_no_harm_metrics
from scripts.run_eval import StepAwareJESPolicy, parse_tau_schedule
from scripts.label_tool_sensitivity import label_samples, print_summary as print_label_summary


def load_model_and_tokenizer(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    print(f"Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    model.eval()
    return model, tok


def run_policy(agent, samples, policy, run_id: str, jes_params=None):
    records = []
    for s in tqdm(samples, desc=policy.name):
        if hasattr(policy, "reset_episode"):
            policy.reset_episode()
        ep = agent.run(
            question=s.problem,
            policy=policy,
            gold_answer=s.answer,
            episode_id=s.id,
            target_side="positive",
        ).to_dict()
        records.append(convert_episode_to_record(ep, run_id=run_id, dataset="math", jes_params=jes_params))
    return records


def compute_headline(all_records, manifest):
    po = {p: {r["sample_id"]: r for r in recs} for p, recs in all_records.items()}
    out = {}
    for pname in all_records:
        for subset in ["tool_critical", "stealth_choice", "tool_harmful", "indifferent"]:
            if subset == "stealth_choice":
                sids = [m["sample_id"] for m in manifest if m["label"] == "tool_critical" and m.get("subdivision") == "stealth_choice"]
            else:
                sids = [m["sample_id"] for m in manifest if m["label"] == subset]
            if not sids:
                continue
            correct = [po[pname][sid]["is_correct"] for sid in sids if sid in po[pname]]
            n = len(correct)
            out[(pname, subset)] = {"n": n, "success_rate": (sum(correct) / n if n else 0.0)}
    return out


def main():
    ap = argparse.ArgumentParser(description="MATH muscle experiment")
    ap.add_argument("--data-path", default=None)
    ap.add_argument(
        "--direction-path",
        default="steering/directions/direction_calculator_v1.npz",
        help="Steering direction NPZ (default: calculator_v1; best-practice for MATH).",
    )
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--position", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--rho-max", type=float, default=0.75)
    ap.add_argument("--jes-tau-schedule", default=None, help='e.g. "1:3.0,2+:0.5"')
    ap.add_argument("--enable-guard", action="store_true")
    ap.add_argument("--guard-threshold", type=float, default=-1.0)
    ap.add_argument("--score-mode", default="numeric",
                    choices=["numeric", "any", "exact"],
                    help="Answer scoring mode (default: numeric; recommended for MATH)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = MathDataset(args.data_path)
    samples = ds.get_subset(args.n_samples, seed=args.seed)
    print(f"Selected {len(samples)} MATH samples")

    model, tok = load_model_and_tokenizer(args.model)
    direction, _ = load_direction(args.direction_path)
    direction_rms = float(np.sqrt(np.mean(direction ** 2)))

    agent = ReActAgent(
        model, tok, {"calculator": CalculatorTool()},
        AgentConfig(max_steps=args.max_steps, layer=args.layer, position=args.position,
                    score_mode=args.score_mode),
        direction=direction, direction_rms=direction_rms,
    )

    jes_cfg = JESConfig(tau=args.tau, eps=args.eps, max_rho=args.rho_max)
    jes_params = {"tau": args.tau, "eps": args.eps, "rho_max": args.rho_max, "layer": args.layer, "pos": args.position}
    tau_sched = parse_tau_schedule(args.jes_tau_schedule) if args.jes_tau_schedule else {}

    policies = {
        "baseline": BaselinePolicy(),
        "force_adopt": ForcedPolicy(force_adopt=True),
        "force_reject": ForcedPolicy(force_adopt=False),
        "jes": StepAwareJESPolicy(
            base_config=jes_cfg, direction=direction,
            tau_schedule=tau_sched,
            guard_enabled=args.enable_guard,
            guard_threshold=args.guard_threshold,
        ),
    }

    all_records = {}
    for pname, pol in policies.items():
        run_id = make_run_id("math", pname, len(samples), args.seed)
        recs = run_policy(agent, samples, pol, run_id, jes_params if pname == "jes" else None)
        all_records[pname] = recs
        write_records(recs, str(out / f"{pname}.jsonl"))

    # Label
    manifest = label_samples(all_records["baseline"], all_records["force_adopt"], all_records["force_reject"])
    print_label_summary(manifest)
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Paired stats (jes vs baseline)
    bl = {r["sample_id"]: r for r in all_records["baseline"]}
    js = {r["sample_id"]: r for r in all_records["jes"]}
    common = sorted(set(bl) & set(js))
    bl_c = [bl[s]["is_correct"] for s in common]
    js_c = [js[s]["is_correct"] for s in common]
    ind_ids = [m["sample_id"] for m in manifest if m["label"] == "indifferent"]
    paired = {
        "mcnemar": mcnemar_test(bl_c, js_c),
        "bootstrap_success_diff": bootstrap_ci(bl_c, js_c, "success_diff"),
        "do_no_harm": do_no_harm_metrics(all_records["baseline"], all_records["jes"], ind_ids),
    }
    write_summary(paired, str(out / "paired_stats_jes.json"))

    # Headline
    headline = compute_headline(all_records, manifest)
    headline_json = {f"{p}_{s}": v for (p, s), v in headline.items()}
    write_summary(headline_json, str(out / "headline.json"))

    print("\nDone.")


if __name__ == "__main__":
    main()

