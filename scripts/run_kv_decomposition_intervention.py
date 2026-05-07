#!/usr/bin/env python3
"""
KV2 Output Decomposition Patching.

Decomposes attn_L18 KV2's contribution to the attention output along the
L20 evidence probe direction and scales the parallel / orthogonal
components independently at the decision point.

Tests which channel of KV2's output carries the routing effect.

Usage:
    python scripts/run_kv_decomposition_intervention.py \
        --data-path data/hotpotqa/hotpot_dev_distractor_v1.json \
        --corpus-path data/hotpotqa/corpus.jsonl \
        --n-samples 500 --seed 42 \
        --reuse-baseline results/kv_group_scaling_v1/KV2_a1.0.jsonl \
        --out results/kv_decomposition
"""

import os, sys, json, argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from agent.react_loop import ReActAgent, AgentConfig
from agent.policies_verify import FreeGenBaselinePolicy, KVGroupDirectionalScalingPolicy
from datasets.hotpotqa import HotpotQADataset
from tools.search_tool import SearchTool
from steering.directions import load_direction
from scripts.run_verify_critical_pipeline import run_episode, compute_stats


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
    print(f"  [{tag:<26}] 2ndSR:{bl_2sr:.1f}->{po_2sr:.1f}%  "
          f"Resc:{resc}(causal:{r_causal},{purity:.0f}%)  "
          f"Regr:{regr}  Net(EM):{net:+d}  PF:{pf}/{n}  McN_p:{mcn_p:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="KV2 output decomposition patching")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--type-filter", default="bridge")
    parser.add_argument("--score-mode", default="exact")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--kv-group", type=int, default=2)
    parser.add_argument("--evidence-direction-path",
                        default="steering/directions/direction_probe_layer20.npz")
    parser.add_argument("--evidence-direction-key", default="decision_direction")
    parser.add_argument("--reuse-baseline", default=None,
                        help="Path to existing baseline jsonl (identity-equivalent)")
    parser.add_argument("--sample-ids", default=None,
                        help="Optional fixed sample id list (reproducibility)")
    parser.add_argument("--configs", default="identity,orth_a2,orth_a3,par_a2,orth_a0,par_a0",
                        help="Comma-separated config names to run")
    parser.add_argument("--out", default="results/kv_decomposition")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # (alpha_parallel, alpha_orth) per config
    CONFIG_TABLE = {
        "identity":  (1.0, 1.0),
        "orth_a2":   (1.0, 2.0),
        "orth_a3":   (1.0, 3.0),
        "par_a2":    (2.0, 1.0),
        "par_a3":    (3.0, 1.0),
        "orth_a0":   (1.0, 0.0),
        "par_a0":    (0.0, 1.0),
        "kill_both": (0.0, 0.0),
    }

    selected = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in selected:
        if c not in CONFIG_TABLE:
            raise ValueError(f"unknown config '{c}'. known: {list(CONFIG_TABLE)}")

    print("=" * 72)
    print("  KV2 OUTPUT DECOMPOSITION PATCHING")
    print("=" * 72)
    print(f"  Layer {args.layer}, KV group {args.kv_group}")
    print(f"  Evidence direction: {args.evidence_direction_path}")
    print(f"  Configs: {selected}")

    # Load model
    print("\n[1/4] Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Load evidence direction
    print("[2/4] Loading evidence direction...")
    e_dir, _meta = load_direction(
        args.evidence_direction_path,
        key=args.evidence_direction_key,
        normalize_rms=None,
    )
    e_norm = float(np.linalg.norm(e_dir))
    print(f"  evidence_dir: shape={e_dir.shape}  norm={e_norm:.4f}")

    # Load dataset
    print("[3/4] Loading dataset...")
    dataset = HotpotQADataset(args.data_path)

    if args.sample_ids:
        sample_ids = json.load(open(args.sample_ids))
        all_samples = dataset.get_subset(len(dataset), seed=args.seed,
                                         type_filter=args.type_filter)
        id_to_sample = {s.id: s for s in all_samples}
        samples = [id_to_sample[sid] for sid in sample_ids if sid in id_to_sample]
        print(f"  Loaded {len(samples)} samples from sample_ids")
    else:
        samples = dataset.get_subset(args.n_samples, seed=args.seed,
                                     type_filter=args.type_filter)
    print(f"  {len(samples)} HotpotQA(type={args.type_filter}) samples")

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
        config=config, direction=None, direction_rms=1.0,
    )

    # ── Baseline ─────────────────────────────────────────────────
    print("\n[4/4] Baseline...")
    if args.reuse_baseline:
        print(f"  Reusing baseline from {args.reuse_baseline}")
        bl_results = []
        with open(args.reuse_baseline) as f:
            for line in f:
                if line.strip():
                    bl_results.append(json.loads(line))
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

    # ── Sweep ────────────────────────────────────────────────────
    all_conditions = {}
    for cfg in selected:
        a_par, a_orth = CONFIG_TABLE[cfg]
        tag = f"KV{args.kv_group}_{cfg}_ap{a_par:.1f}_ao{a_orth:.1f}"
        print(f"\n  --- {tag}: α_parallel={a_par}, α_orth={a_orth} ---")
        policy = KVGroupDirectionalScalingPolicy(
            layer=args.layer, kv_group=args.kv_group, direction=e_dir,
            alpha_parallel=a_par, alpha_orth=a_orth, tag=cfg,
        )
        run_results = []
        for s in tqdm(samples, desc=tag):
            run_results.append(run_episode(agent, s, policy, args.score_mode))
        with open(out_dir / f"{tag}.jsonl", "w") as f:
            for r in run_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats = compute_stats(bl_results, run_results)
        print_summary_row(tag, stats)
        all_conditions[tag] = {
            "layer": args.layer, "kv_group": args.kv_group,
            "config": cfg, "alpha_parallel": a_par, "alpha_orth": a_orth,
            "stats": stats,
        }

    # ── Report ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  KV2 DECOMPOSITION PATCHING: SUMMARY")
    print("=" * 72)
    header = (f"{'Config':<32} {'α_par':>6} {'α_orth':>7} "
              f"{'2ndSR%':>7} {'Resc':>5} {'Causal':>7} "
              f"{'Regr':>5} {'Net':>5} {'PF':>4} {'McN_p':>8}")
    print(header)
    print("-" * len(header))
    for tag, cond in all_conditions.items():
        s = cond["stats"]
        sr2 = s.get("po_second_search_rate", 0) * 100
        resc = s.get("rescued", 0)
        r_c = s.get("rescued_with_more_search", 0)
        regr = s.get("regressed", 0)
        net = s.get("net_gain", 0)
        pf = s.get("parse_failures", 0)
        mcn = s.get("mcnemar_p", float("nan"))
        print(f"{tag:<32} {cond['alpha_parallel']:>6.1f} "
              f"{cond['alpha_orth']:>7.1f} {sr2:>6.1f}% "
              f"{resc:>5} {r_c:>5}/{resc or 1} "
              f"{regr:>5} {net:>+5} {pf:>4} {mcn:>8.4f}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "kv_group_output_decomposition_patching",
        "dataset": f"HotpotQA(type={args.type_filter})",
        "n_samples": len(samples),
        "score_mode": args.score_mode,
        "layer": args.layer,
        "kv_group": args.kv_group,
        "evidence_direction_path": args.evidence_direction_path,
        "evidence_direction_norm": e_norm,
        "baseline_acc": bl_acc,
        "baseline_second_search_rate": bl_second / len(bl_results),
        "conditions": {
            tag: {
                "layer": c["layer"], "kv_group": c["kv_group"],
                "config": c["config"],
                "alpha_parallel": c["alpha_parallel"],
                "alpha_orth": c["alpha_orth"],
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
