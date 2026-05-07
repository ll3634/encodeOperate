#!/usr/bin/env python3
"""
Circuit-Level Behavioral Intervention: KV Group Scaling at attn_L18.

Scales the output of specific KV groups in Layer 18 attention at the
decision point (step 1) to test whether the identified circuit
(attn_L18 KV2 → mlp_L20) can be directly controlled to change behavior.

Conditions:
  - Baseline: FreeGenBaselinePolicy (no intervention)
  - KV2 alpha sweep: α = 0.0, 0.5, 1.0, 1.5, 2.0, 3.0
  - KV0 control: α = 2.0 (specificity control)

Usage:
    python scripts/run_kv_group_scaling.py \
        --data-path data/hotpotqa/hotpot_dev_distractor_v1.json \
        --corpus-path data/hotpotqa/corpus.jsonl \
        --n-samples 500 --seed 42 \
        --reuse-baseline results/l20_rho020_n500/baseline_results.jsonl \
        --out results/kv_group_scaling
"""

import os, sys, json, time, argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from agent.react_loop import ReActAgent, AgentConfig
from agent.policies_verify import FreeGenBaselinePolicy, KVGroupScalingPolicy
from datasets.hotpotqa import HotpotQADataset
from tools.search_tool import SearchTool
from steering.directions import load_direction
from scripts.run_verify_critical_pipeline import (
    run_episode, compute_stats, _has_parse_failure,
)


def load_model_and_tokenizer(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    mdl.eval()
    return mdl, tok


def print_summary_row(tag, stats):
    n = stats["n"]
    bl_2sr = stats.get("bl_second_search_rate", 0) * 100
    po_2sr = stats.get("po_second_search_rate", 0) * 100
    resc = stats.get("rescued", 0)
    regr = stats.get("regressed", 0)
    net = stats.get("net_gain", 0)
    pf = stats.get("parse_failures", 0)
    r_causal = stats.get("rescued_with_more_search", 0)
    purity = stats.get("rescued_causal_pct", 0) * 100
    mcn_p = stats.get("mcnemar_p", float("nan"))
    print(f"  [{tag:<25}] 2ndSR:{bl_2sr:.1f}→{po_2sr:.1f}%  "
          f"Resc:{resc}(causal:{r_causal},{purity:.0f}%)  "
          f"Regr:{regr}  Net(EM):{net:+d}  PF:{pf}/{n}  McNemar_p:{mcn_p:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="KV Group Scaling: Circuit-level behavioral intervention")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--type-filter", default="bridge")
    parser.add_argument("--score-mode", default="exact")
    parser.add_argument("--reuse-baseline", default=None,
                        help="Path to existing baseline_results.jsonl to skip re-running baseline")
    parser.add_argument("--sample-ids", default=None,
                        help="Path to test_sample_ids.json for exact sample reproducibility")
    parser.add_argument("--direction-path",
                        default="steering/directions/direction_search_v3.npz",
                        help="Direction NPZ (needed for agent init, not used for KV scaling)")
    parser.add_argument("--kv2-alphas", type=float, nargs="+",
                        default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
                        help="Alpha values to sweep for KV group 2")
    parser.add_argument("--control-kv-group", type=int, default=0,
                        help="Control KV group (default: 0)")
    parser.add_argument("--control-alpha", type=float, default=2.0,
                        help="Alpha for control KV group")
    parser.add_argument("--out", default="results/kv_group_scaling")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  KV GROUP SCALING: Circuit-Level Behavioral Intervention")
    print("=" * 70)
    print(f"  KV2 alpha sweep: {args.kv2_alphas}")
    print(f"  Control: KV{args.control_kv_group} alpha={args.control_alpha}")
    print(f"  Score mode: {args.score_mode}")

    # Load model
    print("\n[1/4] Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Load direction (needed for agent init even though KV scaling doesn't use it)
    direction, _ = load_direction(args.direction_path, normalize_rms=1.0)
    direction_rms = float(np.sqrt(np.mean(direction ** 2)))

    # Load dataset
    print("[2/4] Loading dataset...")
    dataset = HotpotQADataset(args.data_path)

    if args.sample_ids:
        sample_ids = json.load(open(args.sample_ids))
        all_samples = dataset.get_subset(len(dataset), seed=args.seed,
                                         type_filter=args.type_filter)
        id_set = set(sample_ids)
        samples = [s for s in all_samples if s.id in id_set]
        # Preserve original order
        id_to_sample = {s.id: s for s in samples}
        samples = [id_to_sample[sid] for sid in sample_ids if sid in id_to_sample]
        print(f"  Loaded {len(samples)} samples from sample_ids ({len(sample_ids)} requested)")
    else:
        samples = dataset.get_subset(args.n_samples, seed=args.seed,
                                     type_filter=args.type_filter)
    print(f"  {len(samples)} HotpotQA(type={args.type_filter}) samples")

    # Save sample IDs for reproducibility
    json.dump([s.id for s in samples], open(out_dir / "sample_ids.json", "w"))

    # Setup agent
    search_tool = SearchTool(corpus_path=args.corpus_path)
    tools = {"search": search_tool}
    config = AgentConfig(
        max_steps=5, max_tokens_per_step=256, temperature=0.0,
        layer=20, tools=list(tools.keys()), score_mode=args.score_mode,
    )
    agent = ReActAgent(
        model=model, tokenizer=tokenizer, tools=tools,
        config=config, direction=direction, direction_rms=direction_rms,
    )

    # ── Phase 1: Baseline ─────────────────────────────────────────
    print("\n[3/4] Baseline...")
    if args.reuse_baseline:
        print(f"  Reusing baseline from {args.reuse_baseline}")
        bl_results = []
        with open(args.reuse_baseline) as f:
            for line in f:
                if line.strip():
                    bl_results.append(json.loads(line))
        # Filter to our sample IDs
        our_ids = {s.id for s in samples}
        bl_results = [r for r in bl_results if r["sample_id"] in our_ids]
        print(f"  Loaded {len(bl_results)} baseline results")
    else:
        print("  Running FreeGen baseline...")
        bl_policy = FreeGenBaselinePolicy()
        bl_results = []
        for s in tqdm(samples, desc="freegen_baseline"):
            bl_results.append(run_episode(agent, s, bl_policy, args.score_mode))
        with open(out_dir / "baseline_results.jsonl", "w") as f:
            for r in bl_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    bl_acc = sum(r["is_correct"] for r in bl_results) / len(bl_results)
    bl_second = sum(
        1 for r in bl_results
        if sum(1 for s in r.get("steps", []) if s.get("action") == "search") >= 2
    )
    print(f"  Baseline: acc={bl_acc:.1%}  2ndSR={bl_second}/{len(bl_results)}")

    # ── Phase 2: KV Group Scaling Sweep ──────────────────────────
    print("\n[4/4] Running KV group scaling conditions...")
    all_conditions = {}

    # KV2 alpha sweep
    for alpha_val in args.kv2_alphas:
        tag = f"KV2_a{alpha_val:.1f}"
        print(f"\n  --- {tag}: L18 KV2 (H14-H20) × {alpha_val} ---")
        policy = KVGroupScalingPolicy(layer=18, kv_group=2, alpha=alpha_val)
        run_results = []
        for s in tqdm(samples, desc=tag):
            run_results.append(run_episode(agent, s, policy, args.score_mode))
        # Save raw results
        with open(out_dir / f"{tag}.jsonl", "w") as f:
            for r in run_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats = compute_stats(bl_results, run_results)
        print_summary_row(tag, stats)
        all_conditions[tag] = {
            "layer": 18, "kv_group": 2, "alpha": alpha_val,
            "role": "target", "stats": stats,
        }

    # Control: KV0 amplification
    ctrl_tag = f"KV{args.control_kv_group}_a{args.control_alpha:.1f}"
    print(f"\n  --- {ctrl_tag}: L18 KV{args.control_kv_group} "
          f"(control) × {args.control_alpha} ---")
    ctrl_policy = KVGroupScalingPolicy(
        layer=18, kv_group=args.control_kv_group, alpha=args.control_alpha)
    ctrl_results = []
    for s in tqdm(samples, desc=ctrl_tag):
        ctrl_results.append(run_episode(agent, s, ctrl_policy, args.score_mode))
    with open(out_dir / f"{ctrl_tag}.jsonl", "w") as f:
        for r in ctrl_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ctrl_stats = compute_stats(bl_results, ctrl_results)
    print_summary_row(ctrl_tag, ctrl_stats)
    all_conditions[ctrl_tag] = {
        "layer": 18, "kv_group": args.control_kv_group,
        "alpha": args.control_alpha, "role": "control", "stats": ctrl_stats,
    }

    # ── Report ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CIRCUIT-LEVEL BEHAVIORAL INTERVENTION SUMMARY")
    print("=" * 70)
    print(f"{'Condition':<25} {'2ndSR':>8} {'Resc':>5} {'Causal':>7} "
          f"{'Regr':>5} {'Net':>5} {'PF':>4} {'McNemar':>10}")
    print("-" * 75)
    for tag, cond in all_conditions.items():
        s = cond["stats"]
        sr2 = s.get("po_second_search_rate", 0) * 100
        resc = s.get("rescued", 0)
        r_c = s.get("rescued_with_more_search", 0)
        regr = s.get("regressed", 0)
        net = s.get("net_gain", 0)
        pf = s.get("parse_failures", 0)
        mcn = s.get("mcnemar_p", float("nan"))
        role = cond["role"]
        marker = " ★" if role == "target" else " (ctrl)"
        print(f"{tag + marker:<25} {sr2:>7.1f}% {resc:>5} {r_c:>5}/{resc or 1} "
              f"{regr:>5} {net:>+5} {pf:>4} {mcn:>10.4f}")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "kv_group_scaling_circuit_intervention",
        "dataset": f"HotpotQA(type={args.type_filter})",
        "n_samples": len(samples),
        "score_mode": args.score_mode,
        "baseline_acc": bl_acc,
        "baseline_second_search_rate": bl_second / len(bl_results),
        "conditions": {
            tag: {
                "layer": c["layer"], "kv_group": c["kv_group"],
                "alpha": c["alpha"], "role": c["role"],
                **{k: v for k, v in c["stats"].items()
                   if not isinstance(v, dict)},
            }
            for tag, c in all_conditions.items()
        },
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nReport saved: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()

