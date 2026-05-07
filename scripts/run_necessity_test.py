#!/usr/bin/env python3
"""
Necessity Test: Does L20 steering depend on L18 KV Group 2?

If the two-stage circuit (attn_L18 KV2 -> mlp_L20) is real, then ablating
KV2 at L18 should DIMINISH the effect of L20 steering.

Configs:
  steer_only:        L20 steering (rho=-0.20) + L18 identity (no ablation)
  steer_ablate_kv2:  L20 steering (rho=-0.20) + L18 KV2 ablation (alpha=0)
  steer_ablate_kv0:  L20 steering (rho=-0.20) + L18 KV0 ablation (specificity ctrl)
  baseline:          No intervention (FreeGen)

Expected results if circuit is real:
  steer_only > steer_ablate_kv2 (KV2 ablation weakens steering)
  steer_ablate_kv0 ≈ steer_only (KV0 is irrelevant)

Usage:
    python scripts/run_necessity_test.py \
        --data-path data/hotpotqa/hotpot_dev_distractor_v1.json \
        --corpus-path data/hotpotqa/corpus.jsonl \
        --direction-path steering/directions/direction_search_v3_layer20.npz \
        --n-samples 500 --seed 42 --type-filter bridge \
        --out results/necessity_test
"""

import os, sys, json, time, argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from agent.react_loop import ReActAgent, AgentConfig
from agent.policies_verify import (
    FreeGenBaselinePolicy, FixedRhoStep2OnlyPolicy, SteerPlusAblatePolicy,
)
from datasets.hotpotqa import HotpotQADataset
from tools.search_tool import SearchTool
from scripts.run_verify_critical_pipeline import (
    run_episode, compute_stats, compute_activation_stats, _has_parse_failure,
)
from eval.paired_stats import mcnemar_test, bootstrap_ci
from steering.directions import load_direction


def load_model_and_tokenizer(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    mdl.eval()
    return mdl, tok


def print_summary(tag, stats):
    n = stats["n"]
    po_acc = stats.get("policy_rate", 0) * 100
    po_2sr = stats.get("po_second_search_rate", 0) * 100
    resc = stats.get("rescued", 0)
    r_causal = stats.get("rescued_with_more_search", 0)
    purity = stats.get("rescued_causal_pct", 0) * 100
    regr = stats.get("regressed", 0)
    net = stats.get("net_gain", 0)
    pf = stats.get("parse_failures", 0)
    mcn_p = stats.get("mcnemar_p", float("nan"))
    print(f"  [{tag}] Acc:{po_acc:.1f}%  2ndSR:{po_2sr:.1f}%  "
          f"Resc:{resc}(causal:{r_causal},{purity:.0f}%)  Regr:{regr}  "
          f"Net(EM):{net:+d}  PF:{pf}/{n}  McNemar p={mcn_p:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Necessity test: L20 steering + L18 KV ablation")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--direction-path", required=True)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--type-filter", default="bridge")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--steer-layer", type=int, default=20)
    parser.add_argument("--ablate-layer", type=int, default=18)
    parser.add_argument("--rho", type=float, default=-0.20)
    parser.add_argument("--score-mode", default="exact")
    parser.add_argument("--out", default="results/necessity_test")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  NECESSITY TEST: L20 steering depends on L18 KV2?")
    print("=" * 70)
    print(f"  Steer: L{args.steer_layer} rho={args.rho}")
    print(f"  Ablate: L{args.ablate_layer}")

    # Load model
    print("\n[1/4] Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Load direction
    print("[2/4] Loading steering direction...")
    direction, dir_meta = load_direction(args.direction_path)
    direction_rms = float(np.sqrt(np.mean(direction ** 2)))
    print(f"  RMS={direction_rms:.6f}  Norm={dir_meta['norm']:.4f}")

    # Load dataset
    print("[3/4] Loading dataset...")
    dataset = HotpotQADataset(args.data_path)
    samples = dataset.get_subset(args.n_samples, seed=args.seed, type_filter=args.type_filter)
    print(f"  {len(samples)} samples selected")

    # Setup agent with direction (needed for steering)
    search_tool = SearchTool(corpus_path=args.corpus_path)
    tools = {"search": search_tool}
    config = AgentConfig(
        max_steps=5, max_tokens_per_step=256, temperature=0.0,
        layer=args.steer_layer, tools=list(tools.keys()), score_mode=args.score_mode,
    )
    agent = ReActAgent(
        model=model, tokenizer=tokenizer, tools=tools,
        config=config, direction=direction, direction_rms=direction_rms,
    )

    # Define configs: (tag, policy)
    # All steered configs inject at L19 (= L20 input) so the signal flows
    # through mlp_L20's amplifier.  This makes the comparison fair.
    steer_at = 19  # L19 output = L20 input
    configs = [
        ("baseline", FreeGenBaselinePolicy()),
        ("steer_only", FixedRhoStep2OnlyPolicy(rho=args.rho, steer_layer=steer_at)),
        ("steer_ablate_kv2", SteerPlusAblatePolicy(
            rho=args.rho, ablate_layer=args.ablate_layer, ablate_kv_group=2)),
        ("steer_ablate_kv0", SteerPlusAblatePolicy(
            rho=args.rho, ablate_layer=args.ablate_layer, ablate_kv_group=0)),
    ]

    # Run all configs
    print(f"\n[4/4] Running {len(configs)} configs x {len(samples)} samples...")
    all_results = {}
    all_stats = {}

    for tag, policy in configs:
        print(f"\n  Config: {tag}")
        results = []
        for s in tqdm(samples, desc=tag):
            results.append(run_episode(agent, s, policy, args.score_mode))
        all_results[tag] = results

        with open(out_dir / f"{tag}_results.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute stats (all vs baseline)
    bl_results = all_results["baseline"]
    for tag in ["steer_only", "steer_ablate_kv2", "steer_ablate_kv0"]:
        stats = compute_stats(bl_results, all_results[tag])
        all_stats[tag] = stats
        print_summary(tag, stats)

    # Final comparison table
    bl_acc = sum(r["is_correct"] for r in bl_results) / len(bl_results)
    bl_act = compute_activation_stats(bl_results)
    print("\n" + "=" * 70)
    print("  NECESSITY TEST RESULTS")
    print("=" * 70)
    print(f"  Baseline: Acc={bl_acc*100:.1f}%, 2ndSR={bl_act['second_search_activation_rate']*100:.1f}%\n")
    header = f"{'Config':<22} {'Acc%':>6} {'2ndSR%':>7} {'Rescued':>7} {'Causal':>6} {'Purity%':>7} {'Regr':>5} {'Net':>5} {'PF':>4} {'McNemar':>10}"
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
        print(f"{tag:<22} {po_acc:>6.1f} {po_2sr:>7.1f} {resc:>7} {r_causal:>6} {purity:>7.0f} {regr:>5} {net:>+5d} {pf:>4} {mcn_p:>10.4f}")

    # Key comparison
    s_only = all_stats.get("steer_only", {})
    s_kv2 = all_stats.get("steer_ablate_kv2", {})
    s_kv0 = all_stats.get("steer_ablate_kv0", {})
    print("\n  KEY COMPARISONS:")
    print(f"  steer_only     Net={s_only.get('net_gain',0):+d}  2ndSR={s_only.get('po_second_search_rate',0)*100:.1f}%")
    print(f"  + ablate KV2   Net={s_kv2.get('net_gain',0):+d}  2ndSR={s_kv2.get('po_second_search_rate',0)*100:.1f}%")
    print(f"  + ablate KV0   Net={s_kv0.get('net_gain',0):+d}  2ndSR={s_kv0.get('po_second_search_rate',0)*100:.1f}%")
    delta_kv2 = s_only.get('net_gain',0) - s_kv2.get('net_gain',0)
    delta_kv0 = s_only.get('net_gain',0) - s_kv0.get('net_gain',0)
    print(f"\n  KV2 ablation cost: {delta_kv2:+d} net EM (should be positive if circuit is real)")
    print(f"  KV0 ablation cost: {delta_kv0:+d} net EM (should be ~0 for specificity)")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "n_samples": len(samples),
        "baseline_acc": bl_acc,
        "baseline_2ndSR": bl_act["second_search_activation_rate"],
        "configs": {},
    }
    for tag, stats in all_stats.items():
        report["configs"][tag] = {k: v for k, v in stats.items()
                                   if not isinstance(v, np.floating) or True}
    # Ensure JSON-serializable
    def _convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    report_path = out_dir / "necessity_test_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=_convert)
    print(f"\n  Report saved to {report_path}")


if __name__ == "__main__":
    main()

