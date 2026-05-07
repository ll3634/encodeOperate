#!/usr/bin/env python3
"""
Clean Experiment: Fair comparison for representation steering of tool-use decisions.

Fixes three critical flaws of the original experiment:
  1. Generation parity: Both baseline and steered use identical generation paths
     (FreeGenBaselinePolicy — NO OVERRIDE anywhere).
  2. Configurable steering point: Steer at step 0 (initial search decision) or
     step 1 (second-search decision). Default: step 0.
  3. Strict scoring: Default score-mode is 'exact' (not 'any').

Primary outcome metric: BEHAVIORAL (search rate change), not accuracy.
Secondary metric: accuracy with exact match.

Usage:
    python scripts/run_clean_experiment.py \
        --data-path data/hotpotqa/hotpot_dev_distractor_v1.json \
        --corpus-path data/hotpotqa/corpus.jsonl \
        --n-samples 200 --seed 42 --dataset hotpotqa --type-filter bridge \
        --steer-step 1 \
        --directions random2:steering/directions/direction_random_control_2.npz \
        --fixed-rho-sweep 0.5 1.0 2.0 \
        --score-mode exact \
        --out results/clean_experiment_v1
"""

import os, sys, json, time, argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from agent.react_loop import ReActAgent, AgentConfig
from agent.policies_verify import FreeGenBaselinePolicy, FixedRhoSteerPolicy, EveryStepJESPolicy
from steering.jes import JESConfig
from datasets.popqa import PopQADataset
from datasets.hotpotqa import HotpotQADataset
from tools.search_tool import SearchTool
from steering.directions import load_direction
from scripts.run_verify_critical_pipeline import (
    run_episode, compute_stats, _has_parse_failure,
)


def load_model_and_tokenizer(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    mdl.eval()
    return mdl, tok


def parse_direction_args(raw_list):
    result = []
    for item in raw_list:
        if ":" not in item:
            raise ValueError(f"Direction spec must be 'name:path', got: {item!r}")
        name, path = item.split(":", 1)
        result.append((name.strip(), path.strip()))
    return result


def print_behavioral_summary(tag, stats):
    """Print behavioral + multi-metric accuracy in a compact table row."""
    n = stats["n"]
    bl_sr = stats.get("bl_search_rate", 0) * 100
    po_sr = stats.get("po_search_rate", 0) * 100
    bl_2sr = stats.get("bl_second_search_rate", 0) * 100
    po_2sr = stats.get("po_second_search_rate", 0) * 100
    avg_sd = stats.get("avg_search_count_delta", 0)
    resc = stats.get("rescued", 0)
    regr = stats.get("regressed", 0)
    net = stats.get("net_gain", 0)
    pf = stats.get("parse_failures", 0)
    r_causal = stats.get("rescued_with_more_search", 0)
    # Multi-metric
    bl_f1 = stats.get("bl_f1_mean", 0) * 100
    po_f1 = stats.get("po_f1_mean", 0) * 100
    f1d = stats.get("f1_delta", 0) * 100
    net_c = stats.get("net_gain_contains", 0)
    print(f"  [{tag}] SR:{bl_sr:.0f}→{po_sr:.0f}%  2ndSR:{bl_2sr:.0f}→{po_2sr:.0f}%  "
          f"ΔSearch:{avg_sd:+.3f}  Resc:{resc}(causal:{r_causal})  "
          f"Regr:{regr}  Net(EM):{net:+d}  Net(contains):{net_c:+d}  "
          f"F1:{bl_f1:.1f}→{po_f1:.1f}(Δ{f1d:+.1f})  PF:{pf}/{n}")


def main():
    parser = argparse.ArgumentParser(description="Clean steering experiment (fair baseline)")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--directions", nargs="+", required=True,
                        help="Space-separated 'name:path' pairs")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="hotpotqa", choices=["popqa", "hotpotqa"])
    parser.add_argument("--type-filter", default=None)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--steer-step", type=int, default=0,
                        help="Which step to apply steering. 0=initial decision, 1=second search")
    parser.add_argument("--alpha-max", type=float, default=8.0)
    parser.add_argument("--normalize-rms", type=float, default=1.0)
    parser.add_argument("--fixed-rho-sweep", type=float, nargs="+", default=None,
                        help="Fixed rho values to sweep (mutually exclusive with --jes-adaptive)")
    parser.add_argument("--decision-only", action="store_true", default=False,
                        help="For fixed-rho: limit steering hook to the decision token only "
                             "(mirrors EveryStepJESPolicy behaviour, eliminates generation-artifact confound)")
    parser.add_argument("--jes-adaptive", action="store_true",
                        help="Use EveryStepJESPolicy (adaptive margin-based steering at every step)")
    parser.add_argument("--jes-tau", type=float, nargs="+", default=[0.2],
                        help="JES tau values to sweep (target margin threshold)")
    parser.add_argument("--jes-max-rho", type=float, nargs="+", default=[0.25],
                        help="JES max_rho values to sweep")
    parser.add_argument("--score-mode", default="exact",
                        choices=["exact", "contains", "any"],
                        help="Answer scoring mode. Default: exact (strict)")
    parser.add_argument("--out", default="results/clean_experiment")
    args = parser.parse_args()

    # Validate: must choose one mode
    if not args.jes_adaptive and not args.fixed_rho_sweep:
        parser.error("Must specify either --fixed-rho-sweep or --jes-adaptive")
    if args.jes_adaptive and args.fixed_rho_sweep:
        parser.error("--fixed-rho-sweep and --jes-adaptive are mutually exclusive")

    direction_specs = parse_direction_args(args.directions)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    norm_rms = args.normalize_rms if args.normalize_rms > 0 else None

    do_label = " [decision-only]" if (not args.jes_adaptive and args.decision_only) else ""
    mode_label = "JES Adaptive (every step)" if args.jes_adaptive else f"Fixed rho @ step {args.steer_step}{do_label}"
    print("=" * 70)
    print("  CLEAN EXPERIMENT (Fair Baseline, Behavioral Metrics)")
    print("=" * 70)
    print(f"  Mode: {mode_label}")
    print(f"  Score mode: {args.score_mode} (strict)")
    print(f"  Directions: {[n for n, _ in direction_specs]}")
    if args.jes_adaptive:
        print(f"  JES tau sweep: {args.jes_tau}")
        print(f"  JES max_rho sweep: {args.jes_max_rho}")
    else:
        print(f"  Rho sweep: {args.fixed_rho_sweep}")
    print(f"  Layer: {args.layer}")

    # Load model
    print("\n[1/4] Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Load dataset
    print("[2/4] Loading dataset...")
    if args.dataset == "hotpotqa":
        dataset = HotpotQADataset(args.data_path)
        samples = dataset.get_subset(args.n_samples, seed=args.seed, type_filter=args.type_filter)
        ds_label = f"HotpotQA(type={args.type_filter})"
    else:
        dataset = PopQADataset(args.data_path)
        samples = dataset.get_subset(args.n_samples, seed=args.seed)
        ds_label = "PopQA"
    print(f"  {len(samples)} samples selected")

    # Build agent with first direction (will swap later)
    first_name, first_path = direction_specs[0]
    first_direction, _ = load_direction(first_path, normalize_rms=norm_rms)
    first_direction_rms = float(np.sqrt(np.mean(first_direction ** 2)))

    search_tool = SearchTool(corpus_path=args.corpus_path)
    tools = {"search": search_tool}
    config = AgentConfig(
        max_steps=5, max_tokens_per_step=256, temperature=0.0,
        layer=args.layer, tools=list(tools.keys()), score_mode=args.score_mode,
    )
    agent = ReActAgent(
        model=model, tokenizer=tokenizer, tools=tools,
        config=config, direction=first_direction, direction_rms=first_direction_rms,
    )

    # ── Run FAIR baseline (FreeGen, no OVERRIDE) ──────────────────
    print("\n[3/4] Running FreeGen baseline (no OVERRIDE, same generation path)...")
    bl_policy = FreeGenBaselinePolicy()
    bl_results = []
    for s in tqdm(samples, desc="freegen_baseline"):
        bl_results.append(run_episode(agent, s, bl_policy, args.score_mode))

    with open(out_dir / "baseline_results.jsonl", "w") as f:
        for r in bl_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    bl_acc = sum(r["is_correct"] for r in bl_results) / len(bl_results)
    bl_searches = sum(
        1 for r in bl_results
        if any(s.get("action") == "search" for s in r.get("steps", []))
    )
    bl_second = sum(
        1 for r in bl_results
        if sum(1 for s in r.get("steps", []) if s.get("action") == "search") >= 2
    )
    print(f"  Baseline: acc={bl_acc:.1%}  search_rate={bl_searches}/{len(samples)}  "
          f"2nd_search={bl_second}/{len(samples)}")

    # ── Sweep for each direction ──────────────────────────────────
    if args.jes_adaptive:
        sweep_desc = f"{len(direction_specs)} dirs × {len(args.jes_tau)} tau × {len(args.jes_max_rho)} max_rho"
    else:
        sweep_desc = f"{len(direction_specs)} dirs × {len(args.fixed_rho_sweep)} rho @ step {args.steer_step}"
    print(f"\n[4/4] Steering sweep: {sweep_desc}...")

    all_results = {}
    for dir_name, dir_path in direction_specs:
        print(f"\n{'='*60}")
        print(f"  Direction: {dir_name}  ({dir_path})")
        direction, dir_meta = load_direction(dir_path, normalize_rms=norm_rms)
        direction_rms = float(np.sqrt(np.mean(direction ** 2)))
        print(f"  RMS={dir_meta['original_rms']:.6f} -> normalized={direction_rms:.6f}")

        # Layer consistency check
        try:
            _dir_data = np.load(dir_path, allow_pickle=True)
            if "layer" in _dir_data:
                npz_layer = int(_dir_data["layer"])
                if npz_layer != args.layer:
                    print(f"  ⚠️  LAYER MISMATCH: direction layer {npz_layer} vs --layer={args.layer}")
                    print(f"       Overriding to {npz_layer}")
                    config.layer = npz_layer
                    agent._hidden_rms = None
                else:
                    config.layer = args.layer
        except Exception:
            pass

        agent.direction = direction
        agent.direction_rms = direction_rms

        dir_out = out_dir / dir_name
        dir_out.mkdir(parents=True, exist_ok=True)

        sweep_entries = {}

        if args.jes_adaptive:
            # ── JES Adaptive: sweep tau × max_rho ──
            for tau_val in args.jes_tau:
                for mr_val in args.jes_max_rho:
                    tag = f"jes_tau{tau_val:.2f}_mr{mr_val:.2f}"
                    jes_cfg = JESConfig(
                        tau=tau_val, max_rho=mr_val, alpha_max=args.alpha_max
                    )
                    policy = EveryStepJESPolicy(config=jes_cfg, direction=direction)

                    run_results = []
                    for s in tqdm(samples, desc=f"{dir_name}/{tag}"):
                        run_results.append(run_episode(agent, s, policy, args.score_mode))

                    with open(dir_out / f"{tag}.jsonl", "w") as f:
                        for r in run_results:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")

                    fs = compute_stats(bl_results, run_results)
                    print_behavioral_summary(tag, fs)
                    sweep_entries[tag] = {"stats": fs, "tau": tau_val, "max_rho": mr_val}
        else:
            # ── Fixed rho sweep ──
            for rho_val in args.fixed_rho_sweep:
                do_suffix = "_do" if args.decision_only else ""
                tag = f"step{args.steer_step}_rho{rho_val:+.2f}{do_suffix}"
                policy = FixedRhoSteerPolicy(
                    rho=rho_val, steer_step=args.steer_step, alpha_max=args.alpha_max,
                    decision_only=args.decision_only,
                )

                run_results = []
                for s in tqdm(samples, desc=f"{dir_name}/{tag}"):
                    run_results.append(run_episode(agent, s, policy, args.score_mode))

                with open(dir_out / f"{tag}.jsonl", "w") as f:
                    for r in run_results:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

                fs = compute_stats(bl_results, run_results)
                print_behavioral_summary(tag, fs)
                sweep_entries[tag] = {"stats": fs, "rho": rho_val,
                                      "decision_only": args.decision_only}

        all_results[dir_name] = {
            "sweep": sweep_entries,
            "direction_path": dir_path,
            "original_rms": dir_meta["original_rms"],
        }

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BEHAVIORAL SUMMARY")
    print("=" * 70)
    print(f"{'Dir':<15} {'Config':<22} {'SR_Δ':>6} {'2ndSR_Δ':>8} "
          f"{'ΔSearch':>8} {'Resc':>5} {'Causal':>7} {'Regr':>5} {'Net':>5} {'PF':>4}")
    print("-" * 95)

    for dir_name, res in all_results.items():
        for tag, entry in res["sweep"].items():
            fs = entry["stats"]
            sr_d = fs.get("search_rate_delta", 0) * 100
            sr2_d = fs.get("second_search_rate_delta", 0) * 100
            avg_d = fs.get("avg_search_count_delta", 0)
            resc = fs.get("rescued", 0)
            r_causal = fs.get("rescued_with_more_search", 0)
            regr = fs.get("regressed", 0)
            net = fs.get("net_gain", 0)
            pf = fs.get("parse_failures", 0)
            print(f"{dir_name:<15} {tag:<22} {sr_d:>+5.1f}% {sr2_d:>+7.1f}% "
                  f"{avg_d:>+7.3f} {resc:>5} {r_causal:>5}/{resc or 1} "
                  f"{regr:>5} {net:>+5} {pf:>4}")

    # ── Behavior Flip Matrices ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BEHAVIOR FLIP MATRICES")
    print("=" * 70)
    for dir_name, res in all_results.items():
        for tag, entry in res["sweep"].items():
            fs = entry["stats"]
            for label, key in [("1st Search", "flip_1st_search"),
                               ("2nd Search", "flip_2nd_search")]:
                fm = fs.get(key, {})
                if not fm:
                    continue
                ks = fm.get("keep_search", {})
                ls = fm.get("lost_search", {})
                gs = fm.get("gained_search", {})
                kn = fm.get("keep_no_search", {})
                print(f"\n  {dir_name} / {tag} — {label}:")
                print(f"    {'':>20} Steered:search  Steered:no-search")
                print(f"    {'BL:search':>20}  {ks.get('n',0):>5} "
                      f"(acc {ks.get('bl_acc_rate',0):.0%}→{ks.get('po_acc_rate',0):.0%})"
                      f"  {ls.get('n',0):>5} "
                      f"(acc {ls.get('bl_acc_rate',0):.0%}→{ls.get('po_acc_rate',0):.0%})")
                print(f"    {'BL:no-search':>20}  {gs.get('n',0):>5} "
                      f"(acc {gs.get('bl_acc_rate',0):.0%}→{gs.get('po_acc_rate',0):.0%})"
                      f"  {kn.get('n',0):>5} "
                      f"(acc {kn.get('bl_acc_rate',0):.0%}→{kn.get('po_acc_rate',0):.0%})")

    report = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "clean_v2_jes_adaptive" if args.jes_adaptive else "clean_v1",
        "mode": "jes_adaptive" if args.jes_adaptive else "fixed_rho",
        "decision_only": False if args.jes_adaptive else args.decision_only,
        "fixes": [
            "FreeGen baseline (no OVERRIDE)",
            f"Mode: {mode_label}",
            f"Score mode: {args.score_mode} (strict)",
            f"decision_only={args.decision_only}" if not args.jes_adaptive else "decision_only=True (JES default)",
        ],
        "dataset": ds_label,
        "n_samples": len(samples),
        "score_mode": args.score_mode,
        "layer": config.layer,
        "alpha_max": args.alpha_max,
        "baseline_acc": bl_acc,
        "baseline_search_rate": bl_searches / len(samples),
        "baseline_second_search_rate": bl_second / len(samples),
        "results": {
            dir_name: {
                tag: entry["stats"]
                for tag, entry in res["sweep"].items()
            }
            for dir_name, res in all_results.items()
        },
    }
    with open(out_dir / "clean_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nReport: {out_dir / 'clean_report.json'}")


if __name__ == "__main__":
    main()

