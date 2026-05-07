#!/usr/bin/env python3
"""
Multi-direction comparison: loads model and runs baseline/oracle ONCE,
then sweeps JES for each direction.

Usage:
    python scripts/run_direction_comparison.py \
        --data-path data/hotpotqa/hotpot_dev_distractor_v1.json \
        --corpus-path data/hotpotqa/corpus.jsonl \
        --n-samples 200 --seed 42 --dataset hotpotqa --type-filter bridge \
        --directions search_v1:steering/directions/direction_search_post_runtime_trace_clean_eval200_seed42_bridge_v1.npz \
                     search_v2:steering/directions/direction_search_post_runtime_trace_clean_eval200_seed42_bridge_v2_hook_fixed.npz \
                     calculator:steering/directions/direction_calculator_post_clean_train_v1.npz \
                     v12_post:steering/directions/direction_v12_post_scaled.npz \
                     random1:steering/directions/direction_random_control_1.npz \
                     random2:steering/directions/direction_random_control_2.npz \
        --max-rho-sweep 0.5 1.0 1.5 --tau-sweep 0.0 \
        --out results/direction_comparison_n200
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
    Baseline1HopPolicy, Oracle2HopPolicy, JESStep2OnlyPolicy,
    FixedRhoStep2OnlyPolicy, FixedAlphaStep2OnlyPolicy,
)
from datasets.popqa import PopQADataset
from datasets.hotpotqa import HotpotQADataset
from tools.search_tool import SearchTool
from steering.directions import load_direction
from steering.jes import JESConfig
from scripts.run_verify_critical_pipeline import (
    run_episode, compute_stats, compute_activation_stats, cost_stats,
    _has_parse_failure,
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
    """Parse 'name:path' pairs into [(name, path), ...]"""
    result = []
    for item in raw_list:
        if ":" not in item:
            raise ValueError(f"Direction spec must be 'name:path', got: {item!r}")
        name, path = item.split(":", 1)
        result.append((name.strip(), path.strip()))
    return result


def main():
    parser = argparse.ArgumentParser(description="Multi-direction comparison pipeline")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--directions", nargs="+", required=True,
                        help="Space-separated 'name:path' pairs for directions")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="hotpotqa", choices=["popqa", "hotpotqa"])
    parser.add_argument("--type-filter", default=None)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--tau-sweep", type=float, nargs="+", default=[0.0])
    parser.add_argument("--max-rho-sweep", type=float, nargs="+", default=[0.5])
    parser.add_argument("--alpha-max", type=float, default=8.0)
    parser.add_argument("--normalize-rms", type=float, default=1.0)
    parser.add_argument("--fixed-rho-sweep", type=float, nargs="*", default=None,
                        help="If set, run FixedRhoStep2Only instead of JES. "
                             "Supports negative values for reverse steering.")
    parser.add_argument("--fixed-alpha-sweep", type=float, nargs="*", default=None,
                        help="If set, run FixedAlphaStep2Only with direct alpha values. "
                             "Bypasses rho->alpha conversion and alpha_max clamp.")
    parser.add_argument("--score-mode", default="any")
    parser.add_argument("--out", default="results/direction_comparison")
    args = parser.parse_args()

    direction_specs = parse_direction_args(args.directions)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    norm_rms = args.normalize_rms if args.normalize_rms > 0 else None

    print("=" * 70)
    print("  MULTI-DIRECTION COMPARISON")
    print("=" * 70)
    print(f"Directions ({len(direction_specs)}): {[n for n,_ in direction_specs]}")

    # Load model once
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

    # Load any direction to build agent (for baseline/oracle; direction unused in those phases)
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

    # Shared baseline and oracle (run once)
    print("\n[3/4] Running baseline and oracle (shared across all directions)...")
    bl_results, orc_results = [], []
    for s in tqdm(samples, desc="baseline"):
        bl_results.append(run_episode(agent, s, Baseline1HopPolicy(), args.score_mode))
    for s in tqdm(samples, desc="oracle"):
        orc_results.append(run_episode(agent, s, Oracle2HopPolicy(), args.score_mode))

    # Build manifest
    bl_by_id = {r["sample_id"]: r for r in bl_results}
    orc_by_id = {r["sample_id"]: r for r in orc_results}
    vc_ids, vh_ids, ind_ids = set(), set(), set()
    for s in samples:
        bl_ok = bl_by_id[s.id]["is_correct"]
        orc_ok = orc_by_id[s.id]["is_correct"]
        if not bl_ok and orc_ok:
            vc_ids.add(s.id)
        elif bl_ok and not orc_ok:
            vh_ids.add(s.id)
        else:
            ind_ids.add(s.id)

    print(f"  VC={len(vc_ids)}  VH={len(vh_ids)}  Ind={len(ind_ids)}")
    with open(out_dir / "baseline_results.jsonl", "w") as f:
        for r in bl_results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "oracle_results.jsonl", "w") as f:
        for r in orc_results: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    baseline_acc = sum(r["is_correct"] for r in bl_results) / len(bl_results)

    # Determine sweep mode: FixedAlpha > FixedRho > JES
    use_fixed_alpha = args.fixed_alpha_sweep is not None and len(args.fixed_alpha_sweep) > 0
    use_fixed_rho = (not use_fixed_alpha) and args.fixed_rho_sweep is not None and len(args.fixed_rho_sweep) > 0
    if use_fixed_alpha:
        sweep_grid = [("alpha", a) for a in args.fixed_alpha_sweep]
        mode_label = "FixedAlpha"
    elif use_fixed_rho:
        sweep_grid = [("rho", rho_val) for rho_val in args.fixed_rho_sweep]
        mode_label = "FixedRho"
    else:
        sweep_grid = [("jes", tau, mr) for tau in args.tau_sweep for mr in args.max_rho_sweep]
        mode_label = "JES"

    # Sweep for each direction
    print(f"\n[4/4] {mode_label} sweep for {len(direction_specs)} directions × {len(sweep_grid)} configs...")
    all_direction_results = {}

    for dir_name, dir_path in direction_specs:
        print(f"\n{'='*60}")
        print(f"  Direction: {dir_name}  ({dir_path})")
        direction, dir_meta = load_direction(dir_path, normalize_rms=norm_rms)
        direction_rms = float(np.sqrt(np.mean(direction ** 2)))
        print(f"  RMS={dir_meta['original_rms']:.6f} -> normalized={direction_rms:.6f}")

        # ── Layer-consistency check ──────────────────────────────
        # Read the layer stored in the direction .npz (if present)
        # and warn loudly if it doesn't match the runtime --layer.
        try:
            _dir_data = np.load(dir_path, allow_pickle=True)
            if "layer" in _dir_data:
                npz_layer = int(_dir_data["layer"])
                if npz_layer != args.layer:
                    print(f"  ⚠️  LAYER MISMATCH: direction trained on layer {npz_layer} "
                          f"but runtime --layer={args.layer}")
                    print(f"       Overriding runtime layer to {npz_layer} for this direction.")
                    config.layer = npz_layer
                    # Invalidate cached hidden_rms so it re-calibrates at correct layer
                    agent._hidden_rms = None
                else:
                    config.layer = args.layer
        except Exception:
            pass  # If we can't read it, keep using --layer

        agent.direction = direction
        agent.direction_rms = direction_rms

        dir_out = out_dir / dir_name
        dir_out.mkdir(parents=True, exist_ok=True)

        best_net, best_tag = -999, None
        sweep_entries = {}

        for grid_entry in sweep_grid:
            sweep_type = grid_entry[0]
            if sweep_type == "alpha":
                alpha_val = grid_entry[1]
                tag = f"fixed_alpha{alpha_val:+.2f}"
                policy = FixedAlphaStep2OnlyPolicy(alpha=alpha_val)
            elif sweep_type == "rho":
                rho_val = grid_entry[1]
                tag = f"fixed_rho{rho_val:+.2f}"
                policy = FixedRhoStep2OnlyPolicy(rho=rho_val, alpha_max=args.alpha_max)
            else:
                tau_val, mr_val = grid_entry[1], grid_entry[2]
                tag = f"tau{tau_val:.2f}_rho{mr_val:.2f}"
                jes_config = JESConfig(tau=tau_val, max_rho=mr_val, alpha_max=args.alpha_max)
                policy = JESStep2OnlyPolicy(config=jes_config, direction=direction)

            run_results = []
            for s in tqdm(samples, desc=f"{dir_name}/{tag}"):
                run_results.append(run_episode(agent, s, policy, args.score_mode))

            with open(dir_out / f"{tag}.jsonl", "w") as f:
                for r in run_results: f.write(json.dumps(r, ensure_ascii=False) + "\n")

            fs = compute_stats(bl_results, run_results)
            vs = compute_stats(bl_results, run_results, vc_ids) if vc_ids else {"n": 0}
            net_c = fs.get("net_gain_corrected", fs.get("net_gain", 0))
            pf = fs.get("parse_failures", 0)
            print(f"  [{tag}] rescued={fs.get('rescued',0)} regrsd={fs.get('regressed',0)} "
                  f"net={fs.get('net_gain',0):+d} corrected={net_c:+d} pf={pf}/{fs['n']}")

            entry = {"full_stats": fs, "vc_stats": vs}
            if sweep_type == "alpha":
                entry["fixed_alpha"] = alpha_val
            elif sweep_type == "rho":
                entry["fixed_rho"] = grid_entry[1]
            else:
                entry["tau"] = grid_entry[1]
                entry["max_rho"] = grid_entry[2]
            sweep_entries[tag] = entry

            if net_c > best_net:
                best_net, best_tag = net_c, tag

        all_direction_results[dir_name] = {
            "best_tag": best_tag, "best_net_gain_corrected": best_net,
            "sweep": sweep_entries, "direction_path": dir_path,
            "original_rms": dir_meta["original_rms"],
        }

    # Summary table
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY (best config per direction)")
    print("=" * 70)
    print(f"{'Direction':<20} {'Cfg':<20} {'BL%':>5} {'JES%':>6} {'Resc':>5} {'Regr':>5} "
          f"{'Net':>5} {'NetC':>5} {'PF':>5} {'VCrsc':>6}")
    print("-" * 80)

    summary = {}
    for dir_name, res in all_direction_results.items():
        btag = res["best_tag"]
        fs = res["sweep"][btag]["full_stats"]
        vs = res["sweep"][btag]["vc_stats"]
        n = fs["n"]
        bl_pct = f"{fs['baseline_rate']*100:.1f}%"
        jes_pct = f"{fs['policy_rate']*100:.1f}%"
        net = fs.get("net_gain", 0)
        netc = fs.get("net_gain_corrected", net)
        pf = fs.get("parse_failures", 0)
        vc_resc = vs.get("rescued_genuine", vs.get("rescued", 0)) if vs["n"] > 0 else "-"
        print(f"{dir_name:<20} {btag:<20} {bl_pct:>5} {jes_pct:>6} "
              f"{fs.get('rescued',0):>5} {fs.get('regressed',0):>5} "
              f"{net:>+5} {netc:>+5} {pf:>5} {str(vc_resc):>6}")
        summary[dir_name] = {
            "best_config": btag, "n": n, "baseline_acc": baseline_acc,
            "jes_acc": fs["policy_rate"], "rescued": fs.get("rescued",0),
            "regressed": fs.get("regressed",0), "net_gain": net,
            "net_gain_corrected": netc, "parse_failures": pf,
            "parse_failure_rate": pf/n, "vc_rescued_genuine": vc_resc,
            "vc_n": vs["n"],
        }

    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": ds_label, "n_samples": len(samples),
        "n_vc": len(vc_ids), "n_vh": len(vh_ids), "n_ind": len(ind_ids),
        "alpha_max": args.alpha_max, "normalize_rms": norm_rms,
        "mode": mode_label,
        "tau_sweep": args.tau_sweep, "max_rho_sweep": args.max_rho_sweep,
        "fixed_rho_sweep": args.fixed_rho_sweep,
        "fixed_alpha_sweep": args.fixed_alpha_sweep,
        "summary": summary, "full_results": all_direction_results,
    }
    with open(out_dir / "comparison_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {out_dir / 'comparison_report.json'}")


if __name__ == "__main__":
    main()


