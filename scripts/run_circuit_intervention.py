#!/usr/bin/env python3
"""
Circuit-Level Behavioral Intervention Experiment.

Scales the output of specific KV groups in attn_L18 at the decision point
to demonstrate circuit-level control over agent behavior (2nd search rate).

Configs:
  baseline:     No intervention
  amplify_kv2:  Scale KV Group 2 (H14-H20) × alpha
  ablate_kv2:   Zero out KV Group 2 (alpha=0)
  amplify_kv0:  Scale KV Group 0 (H0-H6) × alpha (specificity control)

Usage:
    python scripts/run_circuit_intervention.py \
        --data-path data/hotpotqa/hotpot_dev_distractor_v1.json \
        --corpus-path data/hotpotqa/corpus.jsonl \
        --n-samples 500 --seed 42 --type-filter bridge \
        --out results/circuit_intervention
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
from scripts.run_verify_critical_pipeline import (
    run_episode, compute_stats, compute_activation_stats, _has_parse_failure,
)
from eval.paired_stats import mcnemar_test, bootstrap_ci


def load_model_and_tokenizer(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    p = (model_name or "").lower()
    attn_impl = "eager" if "gemma" in p else "sdpa"
    print(f"  attn_implementation={attn_impl}")
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation=attn_impl,
    )
    mdl.eval()
    return mdl, tok


def print_summary(tag, stats):
    n = stats["n"]
    bl_acc = stats.get("baseline_rate", 0) * 100
    po_acc = stats.get("policy_rate", 0) * 100
    bl_2sr = stats.get("bl_second_search_rate", 0) * 100
    po_2sr = stats.get("po_second_search_rate", 0) * 100
    resc = stats.get("rescued", 0)
    regr = stats.get("regressed", 0)
    net = stats.get("net_gain", 0)
    pf = stats.get("parse_failures", 0)
    r_causal = stats.get("rescued_with_more_search", 0)
    purity = stats.get("rescued_causal_pct", 0) * 100
    mcn_p = stats.get("mcnemar_p", float("nan"))
    print(f"  [{tag}] Acc:{bl_acc:.1f}→{po_acc:.1f}%  2ndSR:{bl_2sr:.1f}→{po_2sr:.1f}%  "
          f"Resc:{resc}(causal:{r_causal},{purity:.0f}%)  Regr:{regr}  "
          f"Net(EM):{net:+d}  PF:{pf}/{n}  McNemar p={mcn_p:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Circuit-level behavioral intervention")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--type-filter", default="bridge")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=int, default=18,
                        help="Attention layer for KV group scaling")
    parser.add_argument("--amplify-alpha", type=float, default=2.0,
                        help="Scaling factor for amplify configs")
    parser.add_argument("--alpha-sweep", type=float, nargs="+", default=None,
                        help="If set, run alpha sweep for KV2: e.g. 0.0 0.5 1.0 1.5 2.0 3.0")
    parser.add_argument("--score-mode", default="exact")
    parser.add_argument("--out", default="results/circuit_intervention")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  CIRCUIT-LEVEL BEHAVIORAL INTERVENTION")
    print("=" * 70)
    print(f"  Layer: {args.layer}, Amplify alpha: {args.amplify_alpha}")
    print(f"  Alpha sweep: {args.alpha_sweep}")

    # Load model
    print("\n[1/3] Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Load dataset
    print("[2/3] Loading dataset...")
    dataset = HotpotQADataset(args.data_path)
    samples = dataset.get_subset(args.n_samples, seed=args.seed, type_filter=args.type_filter)
    print(f"  {len(samples)} samples selected")

    # Setup agent (no direction needed for KV group scaling)
    search_tool = SearchTool(corpus_path=args.corpus_path)
    tools = {"search": search_tool}
    config = AgentConfig(
        max_steps=5, max_tokens_per_step=256, temperature=0.0,
        layer=args.layer, tools=list(tools.keys()), score_mode=args.score_mode,
    )
    # direction/direction_rms not needed for KV group scaling, but agent needs them
    # for margin computation. Use dummy values; margin_fn still works (rho=0).
    agent = ReActAgent(
        model=model, tokenizer=tokenizer, tools=tools,
        config=config, direction=None, direction_rms=1.0,
    )

    # ── Run baseline ──
    print("\n[3/3] Running experiments...")
    print("  Config: baseline (FreeGen, no intervention)")
    bl_policy = FreeGenBaselinePolicy()
    bl_results = []
    for s in tqdm(samples, desc="baseline"):
        bl_results.append(run_episode(agent, s, bl_policy, args.score_mode))

    with open(out_dir / "baseline_results.jsonl", "w") as f:
        for r in bl_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    bl_acc = sum(r["is_correct"] for r in bl_results) / len(bl_results)
    bl_act = compute_activation_stats(bl_results)
    print(f"  Baseline: Acc={bl_acc*100:.1f}%, 2ndSR={bl_act['second_search_activation_rate']*100:.1f}%")



    # ── Define configs to run ──
    if args.alpha_sweep:
        # Alpha sweep mode: KV2 with varying alpha + KV0 controls at max alpha
        configs = [
            (f"kv2_a{a:.1f}", args.layer, 2, a)
            for a in args.alpha_sweep
        ]
        # Add KV0 control at the largest alpha value for specificity
        max_alpha = max(args.alpha_sweep)
        configs.append((f"kv0_a{max_alpha:.1f}", args.layer, 0, max_alpha))
    else:
        # Standard 4-config experiment
        configs = [
            ("amplify_kv2", args.layer, 2, args.amplify_alpha),
            ("ablate_kv2", args.layer, 2, 0.0),
            ("amplify_kv0", args.layer, 0, args.amplify_alpha),
        ]

    all_stats = {}
    for tag, layer, kv_group, alpha in configs:
        print(f"\n  Config: {tag} (L{layer}, KV{kv_group}, α={alpha})")
        policy = KVGroupScalingPolicy(layer=layer, kv_group=kv_group, alpha=alpha)
        run_results = []
        for s in tqdm(samples, desc=tag):
            run_results.append(run_episode(agent, s, policy, args.score_mode))

        with open(out_dir / f"{tag}_results.jsonl", "w") as f:
            for r in run_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        stats = compute_stats(bl_results, run_results)
        all_stats[tag] = stats
        print_summary(tag, stats)

    # ── Final comparison table ──
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    header = f"{'Config':<20} {'Acc%':>6} {'2ndSR%':>7} {'Rescued':>7} {'Causal':>6} {'Purity%':>7} {'Regr':>5} {'Net':>5} {'PF':>4} {'McNemar':>10}"
    print(header)
    print("-" * len(header))

    for tag, stats in all_stats.items():
        po_acc = stats.get("policy_rate", 0) * 100
        po_2sr = stats.get("po_second_search_rate", 0) * 100
        resc = stats.get("rescued", 0)
        r_causal = stats.get("rescued_with_more_search", 0)
        purity = stats.get("rescued_causal_pct", 0) * 100
        regr = stats.get("regressed", 0)
        net = stats.get("net_gain", 0)
        pf = stats.get("parse_failures", 0)
        mcn_p = stats.get("mcnemar_p", float("nan"))
        print(f"{tag:<20} {po_acc:>6.1f} {po_2sr:>7.1f} {resc:>7} {r_causal:>6} {purity:>7.0f} {regr:>5} {net:>+5d} {pf:>4} {mcn_p:>10.4f}")

    # ── Save report ──
    report = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "n_samples": len(samples),
        "baseline_acc": bl_acc,
        "baseline_2ndSR": bl_act["second_search_activation_rate"],
        "configs": {},
    }
    for tag, stats in all_stats.items():
        report["configs"][tag] = {
            "policy_rate": stats.get("policy_rate"),
            "po_second_search_rate": stats.get("po_second_search_rate"),
            "rescued": stats.get("rescued"),
            "rescued_with_more_search": stats.get("rescued_with_more_search"),
            "rescued_causal_pct": stats.get("rescued_causal_pct"),
            "regressed": stats.get("regressed"),
            "net_gain": stats.get("net_gain"),
            "parse_failures": stats.get("parse_failures"),
            "mcnemar_p": stats.get("mcnemar_p"),
            "bl_second_search_rate": stats.get("bl_second_search_rate"),
        }

    report_path = out_dir / "circuit_intervention_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved to {report_path}")


if __name__ == "__main__":
    main()
