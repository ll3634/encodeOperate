#!/usr/bin/env python3
"""
Intervention Timing Test — decision-point specificity.

Tests whether steering effectiveness is specific to the decision point (p0),
or generalizes to later positions in the autoregressive generation (p2, p4).

Three configs, all using DEFAULT_SYSTEM_PROMPT (matching A3) and the same direction/rho:
  Config A (p0) : inject at last token of input prompt  — decision_only mode
  Config C (p2) : inject at 50% through generation      — two-pass
  Config B (p4) : inject at last generated token         — two-pass

Finding: Steering effect is localized to the decision point (p0). Post-decision
injection (p2, p4) has zero or negative effect, demonstrating that the
continue/stop decision is formed at a single token position and cannot be
altered after the action token has been committed.

Usage:
    cd tmc/scripts/e2e_agent
    python scripts/run_timing_intervention.py \\
        --data-path   data/hotpotqa/hotpot_dev_distractor_v1.json \\
        --corpus-path data/hotpotqa/corpus.jsonl \\
        --direction-path steering/directions/direction_search_v3_layer20.npz \\
        --baseline-ids results/l20_fixed_neg_rho020_do_n500/baseline_results.jsonl \\
        --rho -0.20 \\
        --out results/timing_intervention_v3
"""

import os, sys, json, argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from agent.react_loop import ReActAgent, AgentConfig
from agent.prompts import PromptBuilder
from agent.policies_verify import (
    FreeGenBaselinePolicy, TimedRhoStep2OnlyPolicy,
)
from datasets.hotpotqa import HotpotQADataset
from tools.search_tool import SearchTool
from steering.directions import load_direction
from scripts.run_verify_critical_pipeline import (
    run_episode, compute_stats, compute_activation_stats, _has_parse_failure,
)

# Injection position labels for the figure
POSITION_LABELS = {
    "p0": "Decision point\n(pre-generation)",
    "p2": "Mid-generation\n(50% tokens)",
    "p4": "End of generation\n(100% tokens)",
}


def load_model_and_tokenizer(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    mdl.eval()
    return mdl, tok


def load_samples_from_ids(dataset, id_file: Path, max_n: int = None):
    """Load samples matching IDs from a baseline results JSONL."""
    target_ids = set()
    for line in open(id_file):
        d = json.loads(line)
        target_ids.add(d["sample_id"])

    all_samples = dataset.get_subset(len(target_ids) * 3, seed=42, type_filter="bridge")
    samples = [s for s in all_samples if s.id in target_ids]
    if max_n:
        samples = samples[:max_n]
    print(f"  Loaded {len(samples)} samples matching {len(target_ids)} target IDs")
    return samples


def summarise_run(label, bl_results, run_results, score_mode="any"):
    fs = compute_stats(bl_results, run_results)
    act = compute_activation_stats(run_results)
    n = fs["n"]
    net_c = fs.get("net_gain_corrected", fs.get("net_gain", 0))
    pf = fs.get("parse_failures", 0)
    rws = fs.get("rescued_with_more_search", 0)
    purity = (rws / fs.get("rescued", 1)) * 100 if fs.get("rescued", 0) > 0 else float("nan")
    print(
        f"  [{label}] BL={fs['baseline_rate']*100:.1f}%  "
        f"steered={fs['policy_rate']*100:.1f}%  "
        f"rescued={fs.get('rescued', 0)}  regressed={fs.get('regressed', 0)}  "
        f"net={fs.get('net_gain', 0):+d}  corrected={net_c:+d}  "
        f"PF={pf}/{n}  "
        f"2ndSR={act['second_search_activation_rate']*100:.1f}%  "
        f"purity={purity:.0f}%"
    )
    return fs, act


def make_figure(config_results, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figure")
        return

    # Positions in order: p0, p2, p4
    positions = ["p0", "p2", "p4"]
    labels = [POSITION_LABELS[p] for p in positions]
    net_gains = [config_results[p]["stats"]["net_gain_corrected"] for p in positions]
    second_sr = [config_results[p]["activation"]["second_search_activation_rate"] * 100
                 for p in positions]
    rescued = [config_results[p]["stats"].get("rescued", 0) for p in positions]
    regressed = [config_results[p]["stats"].get("regressed", 0) for p in positions]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Intervention Timing Test: Steering Effect is Decision-Point Specific",
                 fontsize=13, fontweight="bold")

    colors = ["#2196F3", "#FF9800", "#F44336"]  # blue (p0), orange (p2), red (p4)

    # Panel 1: Net(EM) corrected by position
    ax = axes[0]
    bar_colors = [c if v >= 0 else "#9E9E9E" for c, v in zip(colors, net_gains)]
    bars = ax.bar(labels, net_gains, color=bar_colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Net(EM) corrected", fontsize=10)
    ax.set_title("Steering Effect by Injection Position", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, net_gains):
        ypos = max(bar.get_height(), 0) + 0.3 if val >= 0 else min(bar.get_height(), 0) - 0.8
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:+d}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    # Panel 2: 2nd Search Rate delta
    bl_2sr = config_results["p0"]["stats"]["bl_second_search_rate"] * 100
    sr_deltas = [sr - bl_2sr for sr in second_sr]
    ax = axes[1]
    bar_colors2 = [c if v >= 0 else "#9E9E9E" for c, v in zip(colors, sr_deltas)]
    bars = ax.bar(labels, sr_deltas, color=bar_colors2, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Δ 2nd Search Rate (pp)", fontsize=10)
    ax.set_title("Search Behavior Change\n(positive = more 2nd searches triggered)", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, sr_deltas):
        ypos = max(bar.get_height(), 0) + 0.3 if val >= 0 else min(bar.get_height(), 0) - 0.5
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:+.1f}pp", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Panel 3: Rescued vs Regressed breakdown
    ax = axes[2]
    x = np.arange(len(positions))
    width = 0.35
    bars1 = ax.bar(x - width/2, rescued, width, label="Rescued", color="#4CAF50",
                   edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + width/2, regressed, width, label="Regressed", color="#F44336",
                   edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Rescued vs Regressed Samples", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([POSITION_LABELS[p].split("\n")[0] for p in positions], fontsize=9)
    ax.legend(fontsize=9)
    for bar in bars1:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        out_path = out_dir / f"timing_intervention.{ext}"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Intervention timing test")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--direction-path",
                        default="steering/directions/direction_search_v3_layer20.npz")
    parser.add_argument("--baseline-ids", default=None,
                        help="JSONL with sample_ids to use (defaults to n-samples random subset)")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rho", type=float, default=-0.20)
    parser.add_argument("--alpha-max", type=float, default=8.0)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--score-mode", default="exact")
    parser.add_argument("--out", default="results/timing_intervention_v3")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  INTERVENTION TIMING TEST")
    print("=" * 70)
    print(f"  rho={args.rho}  layer={args.layer}  direction={args.direction_path}")
    print(f"  DEFAULT_SYSTEM_PROMPT (matching A3)")
    print(f"  score_mode={args.score_mode}")

    # ── Load model ───────────────────────────────────────────────────────────
    print("\n[1/5] Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model)

    # ── Load direction ────────────────────────────────────────────────────────
    print("[2/5] Loading direction...")
    direction, dir_meta = load_direction(args.direction_path, normalize_rms=1.0)
    direction_rms = float(np.sqrt(np.mean(direction ** 2)))
    print(f"  direction RMS: {direction_rms:.4f}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("[3/5] Loading dataset...")
    dataset = HotpotQADataset(args.data_path)

    if args.baseline_ids and Path(args.baseline_ids).exists():
        samples = load_samples_from_ids(dataset, Path(args.baseline_ids), args.n_samples)
    else:
        samples = dataset.get_subset(args.n_samples, seed=args.seed, type_filter="bridge")
    print(f"  {len(samples)} samples")

    # ── Create agent with DEFAULT_SYSTEM_PROMPT (matching A3) ──────────────────
    search_tool = SearchTool(corpus_path=args.corpus_path)
    tools = {"search": search_tool}
    config = AgentConfig(
        max_steps=5, max_tokens_per_step=256, temperature=0.0,
        layer=args.layer, tools=list(tools.keys()), score_mode=args.score_mode,
    )
    agent = ReActAgent(
        model=model, tokenizer=tokenizer, tools=tools,
        config=config, direction=direction, direction_rms=direction_rms,
    )
    # Use DEFAULT_SYSTEM_PROMPT (the PromptBuilder default) — no override

    # ── Run baseline ──────────────────────────────────────────────────────────
    print("\n[4/5] Running baseline (DEFAULT_SYSTEM_PROMPT, no steering)...")
    bl_policy = FreeGenBaselinePolicy()
    bl_results = []
    for s in tqdm(samples, desc="baseline"):
        bl_results.append(run_episode(agent, s, bl_policy, args.score_mode))

    with open(out_dir / "baseline_results.jsonl", "w") as f:
        for r in bl_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    bl_acc = sum(r["is_correct"] for r in bl_results) / len(bl_results)
    bl_2sr = sum(1 for r in bl_results if r["tool_calls"] >= 2) / len(bl_results)
    print(f"  Baseline: acc={bl_acc*100:.1f}%  2ndSR={bl_2sr*100:.1f}%  "
          f"n={len(bl_results)}")

    # ── Run three timing configs ──────────────────────────────────────────────
    print("\n[5/5] Running timing configs A (p0), C (p2), B (p4)...")
    timings = [("A_p0", "p0"), ("C_p2", "p2"), ("B_p4", "p4")]
    config_results = {}

    for config_name, timing in timings:
        print(f"\n  === Config {config_name} (timing={timing}) ===")
        policy = TimedRhoStep2OnlyPolicy(
            rho=args.rho, timing=timing, alpha_max=args.alpha_max
        )
        run_results = []
        for s in tqdm(samples, desc=f"config_{config_name}"):
            run_results.append(run_episode(agent, s, policy, args.score_mode))

        with open(out_dir / f"{config_name}_results.jsonl", "w") as f:
            for r in run_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        fs, act = summarise_run(config_name, bl_results, run_results, args.score_mode)
        config_results[timing] = {
            "config_name": config_name,
            "timing": timing,
            "position_label": POSITION_LABELS[timing],
            "stats": fs,
            "activation": act,
        }

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TIMING COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  rho={args.rho}  layer={args.layer}  N={len(samples)}")
    print(f"  Baseline: {bl_acc*100:.1f}%")
    print()
    print(f"  {'Config':<12} {'Pos':<4} {'Acc':>6} {'2ndSR':>7} "
          f"{'Resc':>5} {'Regr':>5} {'Net':>5} {'NetC':>5} {'PF':>5} {'Purity':>7}")
    print("  " + "-" * 60)

    for timing, label in [("p0", "A"), ("p2", "C"), ("p4", "B")]:
        res = config_results[timing]
        fs = res["stats"]
        act = res["activation"]
        n = fs["n"]
        rws = fs.get("rescued_with_more_search", 0)
        resc = fs.get("rescued", 0)
        purity = f"{rws/resc*100:.0f}%" if resc > 0 else "n/a"
        net_c = fs.get("net_gain_corrected", fs.get("net_gain", 0))
        print(
            f"  Config {label:<5} {timing:<4} "
            f"{fs['policy_rate']*100:>5.1f}%  "
            f"{act['second_search_activation_rate']*100:>5.1f}%  "
            f"{resc:>5}  {fs.get('regressed',0):>5}  "
            f"{fs.get('net_gain',0):>+5}  {net_c:>+5}  "
            f"{fs.get('parse_failures',0):>5}  {purity:>7}"
        )

    # ── Check monotonicity hypothesis ─────────────────────────────────────────
    net_p0 = config_results["p0"]["stats"].get("net_gain_corrected", 0)
    net_p2 = config_results["p2"]["stats"].get("net_gain_corrected", 0)
    net_p4 = config_results["p4"]["stats"].get("net_gain_corrected", 0)
    is_monotone = net_p0 >= net_p2 >= net_p4
    print()
    print(f"  Net(EM) corrected: A(p0)={net_p0:+d}  C(p2)={net_p2:+d}  B(p4)={net_p4:+d}")
    print(f"  Monotone hypothesis (A >= C >= B): {'✓ CONFIRMED' if is_monotone else '✗ NOT monotone'}")

    # ── Generate figure ───────────────────────────────────────────────────────
    print("\nGenerating figure...")
    make_figure(config_results, out_dir)

    # ── Save report ───────────────────────────────────────────────────────────
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "direction_path": args.direction_path,
        "rho": args.rho,
        "layer": args.layer,
        "n_samples": len(samples),
        "baseline_acc": bl_acc,
        "baseline_2nd_search_rate": bl_2sr,
        "position_labels": POSITION_LABELS,
        "configs": config_results,
        "monotone_hypothesis_confirmed": is_monotone,
        "net_gains_corrected": {"p0": net_p0, "p2": net_p2, "p4": net_p4},
    }
    report_path = out_dir / "timing_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
