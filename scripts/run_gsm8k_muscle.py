#!/usr/bin/env python3
"""
GSM8K Capability Muscle Experiment.

Demonstrates that JES recovers tool-critical samples (especially stealth_choice)
while maintaining safety on tool-harmful and indifferent subsets.

Pipeline:
 1. Run all policies: baseline, force_adopt, force_reject, jes
 2. Label counterfactual subsets (tool_critical, tool_harmful, indifferent + stealth)
 3. Compute macro/micro metrics + paired stats
 4. Print headline results for paper

Usage:
  python scripts/run_gsm8k_muscle.py \
      --direction-path steering/directions/direction_calculator_v1.npz \
      --n-samples 300 --out results/gsm8k_muscle
"""

import json, argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.gsm8k import GSM8KDataset
from tools.calculator_tool import CalculatorTool
from agent.react_loop import ReActAgent, AgentConfig
from agent.policies import BaselinePolicy, ForcedPolicy
from steering.jes import JESConfig
from steering.directions import load_direction
from eval.unified_output import (
    convert_episode_to_record, compute_run_summary,
    write_records, write_summary, make_run_id,
)
from eval.paired_stats import mcnemar_test, bootstrap_ci, do_no_harm_metrics

# Import step-aware JES from run_eval
from scripts.run_eval import StepAwareJESPolicy, parse_tau_schedule
# Import labeling logic from label_tool_sensitivity
from scripts.label_tool_sensitivity import label_samples, print_summary as print_label_summary


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


def run_all_policies(agent, samples, policies, jes_params, out_dir):
    """Run all policies, write JSONL + summary, return dict of records."""
    all_records = {}
    for pname, policy in policies.items():
        run_id = make_run_id("gsm8k", pname, len(samples), 42)
        records = []
        print(f"\n{'='*50}  {pname}  {'='*50}")
        bl_by_id = {r["sample_id"]: r["is_correct"]
                     for r in all_records["baseline"]} if "baseline" in all_records else None
        fa_by_id = {r["sample_id"]: r["is_correct"]
                     for r in all_records["force_adopt"]} if "force_adopt" in all_records else None

        for sample in tqdm(samples, desc=pname):
            if hasattr(policy, "reset_episode"):
                policy.reset_episode()
            result = agent.run(
                question=sample.question, policy=policy,
                gold_answer=sample.answer, episode_id=sample.id,
                target_side="positive")
            ep = result.to_dict()
            bl_ok = bl_by_id.get(sample.id) if bl_by_id else None
            fa_ok = fa_by_id.get(sample.id) if fa_by_id else None
            rec = convert_episode_to_record(
                ep, run_id=run_id, dataset="gsm8k",
                jes_params=jes_params if pname == "jes" else None,
                baseline_success=bl_ok, force_adopt_success=fa_ok)
            records.append(rec)

        all_records[pname] = records
        write_records(records, str(out_dir / f"{pname}.jsonl"))
        summ = compute_run_summary(records)
        write_summary(summ, str(out_dir / f"{pname}_summary.json"))
        print(f"  {pname}: success={summ['success_rate']:.1%}  "
              f"avg_tc={summ['avg_tool_calls']:.2f}  "
              f"avg_tokens={summ['avg_tokens_total']:.0f}")
    return all_records


def compute_headline(all_records, manifest):
    """Compute the muscle experiment headline numbers."""
    label_map = {r["sample_id"]: r for r in manifest}
    results = {}
    for pname, recs in all_records.items():
        po_by_id = {r["sample_id"]: r for r in recs}
        for subset in ["tool_critical", "stealth_choice", "tool_harmful", "indifferent"]:
            if subset in ("stealth_choice", "stealth_query", "stealth_format"):
                sids = [m["sample_id"] for m in manifest
                        if m["label"] == "tool_critical"
                        and m.get("subdivision") == subset]
            else:
                sids = [m["sample_id"] for m in manifest
                        if m["label"] == subset]
            if not sids:
                continue
            correct = [po_by_id[s]["is_correct"] for s in sids if s in po_by_id]
            n = len(correct)
            sr = sum(correct) / n if n else 0
            results[(pname, subset)] = {"n": n, "success_rate": sr}
    return results


def print_headline(headline, all_records):
    """Print publication-ready headline table."""
    print(f"\n{'='*70}")
    print("  GSM8K MUSCLE EXPERIMENT — HEADLINE RESULTS")
    print(f"{'='*70}")
    subsets = ["tool_critical", "stealth_choice", "tool_harmful", "indifferent"]
    policies = ["baseline", "force_adopt", "force_reject", "jes"]
    hdr = f"{'Policy':18s}" + "".join(f"{s:>18s}" for s in subsets)
    print(hdr)
    print("-" * len(hdr))
    for pname in policies:
        if pname not in all_records:
            continue
        row = f"{pname:18s}"
        for subset in subsets:
            key = (pname, subset)
            if key in headline:
                h = headline[key]
                row += f"  {h['success_rate']*100:5.1f}% (n={h['n']:3d})"
            else:
                row += f"{'':>18s}"
        print(row)

    # JES vs Baseline delta on stealth_choice
    jes_sc = headline.get(("jes", "stealth_choice"))
    bl_sc = headline.get(("baseline", "stealth_choice"))
    if jes_sc and bl_sc:
        delta = jes_sc["success_rate"] - bl_sc["success_rate"]
        print(f"\n★ JES stealth_choice recovery: "
              f"{bl_sc['success_rate']*100:.1f}% → {jes_sc['success_rate']*100:.1f}% "
              f"(Δ = {delta*100:+.1f}pp)")


def main():
    parser = argparse.ArgumentParser(
        description="GSM8K muscle experiment: all policies + counterfactual headline")
    parser.add_argument("--data-path", default=None,
                        help="Local GSM8K JSONL (default: HuggingFace)")
    parser.add_argument("--gsm-hard", action="store_true",
                        help="Use GSM-Hard (large-number variant) to induce calculator-critical cases")
    parser.add_argument(
        "--direction-path",
        default="steering/directions/direction_calculator_v1.npz",
        help="Steering direction NPZ (default: calculator_v1; best-practice for GSM8K).",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=5)
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
    parser.add_argument("--score-mode", default="numeric",
                        choices=["numeric", "any", "exact"],
                        help="Answer scoring mode (default: numeric; recommended for GSM8K)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = GSM8KDataset(args.data_path, gsm_hard=args.gsm_hard)
    samples = dataset.get_subset(args.n_samples, seed=args.seed)
    variant = "GSM-Hard" if args.gsm_hard else "GSM8K"
    print(f"Selected {len(samples)} {variant} samples")

    # Load model + direction
    model, tokenizer = load_model_and_tokenizer(args.model)
    direction, _ = load_direction(args.direction_path)
    direction_rms = float(np.sqrt(np.mean(direction ** 2)))

    # Agent setup
    calculator = CalculatorTool()
    agent_config = AgentConfig(
        max_steps=args.max_steps, layer=args.layer, position=args.position,
        score_mode=args.score_mode)
    agent = ReActAgent(model, tokenizer, {"calculator": calculator},
                       agent_config, direction=direction,
                       direction_rms=direction_rms)

    jes_config = JESConfig(tau=args.tau, eps=args.eps, max_rho=args.rho_max)
    jes_params = {"tau": args.tau, "eps": args.eps, "rho_max": args.rho_max,
                  "layer": args.layer, "pos": args.position}
    tau_sched = parse_tau_schedule(args.jes_tau_schedule) if args.jes_tau_schedule else {}

    # Build policies
    policies = {
        "baseline": BaselinePolicy(),
        "force_adopt": ForcedPolicy(force_adopt=True),
        "force_reject": ForcedPolicy(force_adopt=False),
        "jes": StepAwareJESPolicy(
            base_config=jes_config, direction=direction,
            tau_schedule=tau_sched,
            guard_enabled=args.enable_guard,
            guard_threshold=args.guard_threshold),
    }

    # Step 1: Run all policies
    all_records = run_all_policies(agent, samples, policies, jes_params, out)

    # Step 2: Counterfactual labeling
    print("\n" + "="*60 + "  Counterfactual Labeling  " + "="*60)
    manifest = label_samples(all_records["baseline"], all_records["force_adopt"])
    print_label_summary(manifest)
    manifest_path = out / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Step 3: Paired stats (JES vs Baseline)
    print("\n" + "="*60 + "  Paired Statistics  " + "="*60)
    bl_by_id = {r["sample_id"]: r for r in all_records["baseline"]}
    jes_by_id = {r["sample_id"]: r for r in all_records["jes"]}
    common = sorted(set(bl_by_id) & set(jes_by_id))
    bl_c = [bl_by_id[s]["is_correct"] for s in common]
    jes_c = [jes_by_id[s]["is_correct"] for s in common]
    mcn = mcnemar_test(bl_c, jes_c)
    bci = bootstrap_ci(bl_c, jes_c, "success_diff")
    ind_ids = [m["sample_id"] for m in manifest if m["label"] == "indifferent"]
    dn = do_no_harm_metrics(all_records["baseline"], all_records["jes"], ind_ids)
    paired_report = {"mcnemar": mcn, "bootstrap_success_diff": bci, "do_no_harm": dn}
    write_summary(paired_report, str(out / "paired_stats_jes.json"))
    print(f"  McNemar p={mcn['mcnemar_p']:.4f}  "
          f"(b={mcn['b_regressed']}, c={mcn['c_rescued']})")
    print(f"  ΔSuccess: {bci['observed']:+.4f}  "
          f"95% CI [{bci['ci_lower']:+.4f}, {bci['ci_upper']:+.4f}]")
    print(f"  Regression: {dn['regression_rate']:.1%}  "
          f"Rescue: {dn['rescue_rate']:.1%}  Net: {dn['net_gain']}")

    # Step 4: Headline
    headline = compute_headline(all_records, manifest)
    print_headline(headline, all_records)

    # Save headline as JSON
    headline_json = {f"{p}_{s}": v for (p, s), v in headline.items()}
    write_summary(headline_json, str(out / "headline.json"))

    print(f"\nDone. All outputs in {out}/")


if __name__ == "__main__":
    main()

