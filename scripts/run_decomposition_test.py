#!/usr/bin/env python3
"""
Decomposition / Mediation Test — p0 decision-only.

Decomposes steering direction s into:
  s_parallel : projection onto evidence-sufficiency probe direction p
  s_perp     : orthogonal remainder (s - s_parallel)

Runs three p0 decision-only conditions at the same rho=-0.20:
  1. Full direction  (s)
  2. Parallel only   (s_parallel)
  3. Perpendicular   (s_perp)

If the steering effect is mediated by the probe direction, s_parallel should
capture most of the gain and s_perp should have little/no effect.

Usage:
    cd tmc/scripts/e2e_agent
    python scripts/run_decomposition_test.py \
        --data-path   data/hotpotqa/hotpot_dev_distractor_v1.json \
        --corpus-path data/hotpotqa/corpus.jsonl \
        --baseline-ids results/l20_fixed_neg_rho020_do_n500/baseline_results.jsonl \
        --rho -0.20 \
        --out results/decomposition_test
"""

import os, sys, json, argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from agent.react_loop import ReActAgent, AgentConfig
from agent.policies_verify import FreeGenBaselinePolicy, TimedRhoStep2OnlyPolicy
from datasets.hotpotqa import HotpotQADataset
from tools.search_tool import SearchTool
from steering.directions import load_direction
from scripts.run_verify_critical_pipeline import (
    run_episode, compute_stats, compute_activation_stats, _has_parse_failure,
)


def load_model_and_tokenizer(model_name, adapter_path: str = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if adapter_path:
        from peft import PeftModel
        print(f"Loading adapter: {adapter_path}")
        mdl = PeftModel.from_pretrained(mdl, adapter_path)
        mdl = mdl.merge_and_unload()
    mdl.eval()
    return mdl, tok


def load_samples_from_ids(dataset, id_file: Path, max_n=None):
    target_ids = set()
    for line in open(id_file):
        target_ids.add(json.loads(line)["sample_id"])
    all_samples = dataset.get_subset(len(target_ids) * 3, seed=42, type_filter="bridge")
    samples = [s for s in all_samples if s.id in target_ids]
    if max_n:
        samples = samples[:max_n]
    print(f"  Loaded {len(samples)} samples matching {len(target_ids)} target IDs")
    return samples


def summarise_run(label, bl_results, run_results):
    fs = compute_stats(bl_results, run_results)
    act = compute_activation_stats(run_results)
    n = fs["n"]
    net_c = fs.get("net_gain_corrected", fs.get("net_gain", 0))
    pf = fs.get("parse_failures", 0)
    rws = fs.get("rescued_with_more_search", 0)
    resc = fs.get("rescued", 0)
    purity = (rws / resc) * 100 if resc > 0 else float("nan")
    print(
        f"  [{label:12s}] steered={fs['policy_rate']*100:.1f}%  "
        f"rescued={resc}  regressed={fs.get('regressed', 0)}  "
        f"net={fs.get('net_gain', 0):+d}  corrected={net_c:+d}  "
        f"PF={pf}/{n}  "
        f"2ndSR={act['second_search_activation_rate']*100:.1f}%  "
        f"purity={purity:.0f}%"
    )
    return fs, act


def main():
    parser = argparse.ArgumentParser(description="Decomposition / Mediation Test")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--dir-full",
                        default="steering/directions/direction_decomp_full_layer20.npz")
    parser.add_argument("--dir-parallel",
                        default="steering/directions/direction_decomp_parallel_layer20.npz")
    parser.add_argument("--dir-perp",
                        default="steering/directions/direction_decomp_perp_layer20.npz")
    parser.add_argument("--baseline-ids", default=None)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rho", type=float, default=-0.20)
    parser.add_argument("--alpha-max", type=float, default=8.0)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-path", default=None,
                        help="Optional PEFT adapter dir to merge on top of --model.")
    parser.add_argument("--score-mode", default="exact")
    parser.add_argument("--out", default="results/decomposition_test")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  DECOMPOSITION / MEDIATION TEST")
    print("=" * 70)
    print(f"  rho={args.rho}  layer={args.layer}  timing=p0 (decision-only)")

    # ── Load model ───────────────────────────────────────────────────────────
    print("\n[1/5] Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model, adapter_path=args.adapter_path)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("[2/5] Loading dataset...")
    dataset = HotpotQADataset(args.data_path)
    if args.baseline_ids and Path(args.baseline_ids).exists():
        samples = load_samples_from_ids(dataset, Path(args.baseline_ids), args.n_samples)
    else:
        samples = dataset.get_subset(args.n_samples, seed=args.seed, type_filter="bridge")
    print(f"  {len(samples)} samples")

    # ── Load directions ───────────────────────────────────────────────────────
    print("[3/5] Loading directions...")
    directions = {}
    for label, path in [("full", args.dir_full), ("parallel", args.dir_parallel),
                        ("perp", args.dir_perp)]:
        d, meta = load_direction(path, normalize_rms=1.0)
        rms = float(np.sqrt(np.mean(d ** 2)))
        print(f"  {label:10s}: norm={np.linalg.norm(d):.4f}  RMS={rms:.6f}")
        directions[label] = (d, rms)

    # ── Setup common search tool ──────────────────────────────────────────────
    search_tool = SearchTool(corpus_path=args.corpus_path)
    tools = {"search": search_tool}

    # ── Run baseline (once, shared across all conditions) ─────────────────────
    # Use the full direction for baseline agent setup (rho=0 so direction doesn't matter)
    d_full, rms_full = directions["full"]
    config = AgentConfig(
        max_steps=5, max_tokens_per_step=256, temperature=0.0,
        layer=args.layer, tools=list(tools.keys()), score_mode=args.score_mode,
    )
    agent = ReActAgent(
        model=model, tokenizer=tokenizer, tools=tools,
        config=config, direction=d_full, direction_rms=rms_full,
    )

    print("\n[4/5] Running baseline (no steering)...")
    bl_policy = FreeGenBaselinePolicy()
    bl_results = []
    for s in tqdm(samples, desc="baseline"):
        bl_results.append(run_episode(agent, s, bl_policy, args.score_mode))

    with open(out_dir / "baseline_results.jsonl", "w") as f:
        for r in bl_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    bl_acc = sum(r["is_correct"] for r in bl_results) / len(bl_results)
    bl_2sr = sum(1 for r in bl_results if r["tool_calls"] >= 2) / len(bl_results)
    print(f"  Baseline: acc={bl_acc*100:.1f}%  2ndSR={bl_2sr*100:.1f}%  n={len(bl_results)}")

    # ── Run three direction conditions (all p0, decision-only) ────────────────
    print("\n[5/5] Running direction conditions (all p0, decision-only)...")
    condition_results = {}

    for label in ["full", "parallel", "perp"]:
        d_vec, d_rms = directions[label]
        print(f"\n  === Direction: {label} ===")

        # Recreate agent with this direction
        agent_cond = ReActAgent(
            model=model, tokenizer=tokenizer, tools=tools,
            config=config, direction=d_vec, direction_rms=d_rms,
        )

        policy = TimedRhoStep2OnlyPolicy(
            rho=args.rho, timing="p0", alpha_max=args.alpha_max
        )
        run_results = []
        for s in tqdm(samples, desc=f"dir_{label}"):
            run_results.append(run_episode(agent_cond, s, policy, args.score_mode))

        with open(out_dir / f"{label}_results.jsonl", "w") as f:
            for r in run_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        fs, act = summarise_run(label, bl_results, run_results)
        condition_results[label] = {
            "direction": label,
            "stats": fs,
            "activation": act,
        }

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  DECOMPOSITION TEST SUMMARY")
    print("=" * 70)
    print(f"  rho={args.rho}  layer={args.layer}  timing=p0  N={len(samples)}")
    print(f"  Baseline: acc={bl_acc*100:.1f}%  2ndSR={bl_2sr*100:.1f}%")
    print()
    print(f"  {'Direction':<12} {'Acc':>6} {'2ndSR':>7} {'Resc':>5} {'Regr':>5} "
          f"{'Net':>5} {'NetC':>5} {'PF':>5} {'viaSearch':>10} {'Purity':>7} {'F1Δ':>7}")
    print("  " + "-" * 85)

    for label in ["full", "parallel", "perp"]:
        res = condition_results[label]
        fs = res["stats"]
        act = res["activation"]
        resc = fs.get("rescued", 0)
        rws = fs.get("rescued_with_more_search", 0)
        purity = f"{rws/resc*100:.0f}%" if resc > 0 else "n/a"
        net_c = fs.get("net_gain_corrected", fs.get("net_gain", 0))
        f1_delta = fs.get("f1_delta", 0.0)
        f1_str = f"{f1_delta:>+.3f}" if f1_delta else "   n/a"
        print(
            f"  {label:<12} "
            f"{fs['policy_rate']*100:>5.1f}%  "
            f"{act['second_search_activation_rate']*100:>5.1f}%  "
            f"{resc:>5}  {fs.get('regressed',0):>5}  "
            f"{fs.get('net_gain',0):>+5}  {net_c:>+5}  "
            f"{fs.get('parse_failures',0):>5}  "
            f"{rws:>10}  {purity:>7}  "
            f"{f1_str}"
        )

    # ── Cosine info for reference ─────────────────────────────────────────────
    d_full_raw = np.load(args.dir_full, allow_pickle=True)["decision_direction"]
    d_par_raw = np.load(args.dir_parallel, allow_pickle=True)["decision_direction"]
    d_perp_raw = np.load(args.dir_perp, allow_pickle=True)["decision_direction"]
    print(f"\n  Direction geometry (pre-normalization):")
    print(f"    ||s_full|| = {np.linalg.norm(d_full_raw):.4f}")
    print(f"    ||s_parallel|| = {np.linalg.norm(d_par_raw):.4f}")
    print(f"    ||s_perp|| = {np.linalg.norm(d_perp_raw):.4f}")
    print(f"    var(s_par)/var(s) = {np.linalg.norm(d_par_raw)**2/np.linalg.norm(d_full_raw)**2:.6f}")

    # ── Save report ───────────────────────────────────────────────────────────
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "rho": args.rho,
        "layer": args.layer,
        "timing": "p0",
        "n_samples": len(samples),
        "baseline_acc": bl_acc,
        "baseline_2nd_search_rate": bl_2sr,
        "direction_files": {
            "full": args.dir_full,
            "parallel": args.dir_parallel,
            "perp": args.dir_perp,
        },
        "direction_geometry": {
            "full_norm": float(np.linalg.norm(d_full_raw)),
            "parallel_norm": float(np.linalg.norm(d_par_raw)),
            "perp_norm": float(np.linalg.norm(d_perp_raw)),
            "var_parallel_fraction": float(np.linalg.norm(d_par_raw)**2 / np.linalg.norm(d_full_raw)**2),
        },
        "conditions": condition_results,
    }
    report_path = out_dir / "decomposition_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()

