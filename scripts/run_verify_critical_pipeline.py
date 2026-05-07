#!/usr/bin/env python3
"""
Verify-Critical Pipeline: Mine -> JES Eval -> Report

Single command to run the full verify-critical mining and evaluation pipeline.

Usage:
    python scripts/run_verify_critical_pipeline.py \
        --data-path data/popqa/popqa_test.jsonl \
        --corpus-path data/popqa/corpus.jsonl \
        --direction-path steering/directions/direction_search_v3.npz \
        --n-samples 200 --out results/verify_critical_v1
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from agent.react_loop import ReActAgent, AgentConfig
from agent.policies_verify import Baseline1HopPolicy, Oracle2HopPolicy, JESStep2OnlyPolicy
from agent.policies import BaselinePolicy
from datasets.popqa import PopQADataset
from datasets.hotpotqa import HotpotQADataset
from tools.search_tool import SearchTool
from steering.directions import load_direction
from steering.jes import JESConfig
from eval.paired_stats import mcnemar_test, bootstrap_ci
from scripts.control_budget_diagnosis import compute_diagnosis, print_diagnosis


def load_model_and_tokenizer(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def run_episode(agent, sample, policy, score_mode="any"):
    """Run single episode, return result dict."""
    policy.reset_episode()
    result = agent.run(
        question=sample.question,
        policy=policy,
        gold_answer=sample.answers,
        episode_id=sample.id,
        target_side="positive"
    )
    
    steps = []
    for s in result.steps:
        steps.append({
            "step_idx": s.step_idx,
            "action": s.action,
            "action_input": s.action_input,
            "observation": s.observation[:200] if s.observation else None,
            "final_answer": s.final_answer,
            "margin_before": s.margin_before,
            "steering": s.steering,
            "raw_model_text": s.raw_model_text,
            "parse_failure_reason": s.parse_failure_reason,
        })
    
    # Compute multi-metric scores (EM is primary via result.success / score_mode)
    from eval.scorers import compute_f1, contains_answer, exact_match
    fa = result.final_answer or ""
    golds = sample.answers if isinstance(sample.answers, list) else [sample.answers]
    f1_score = max((compute_f1(fa, g) for g in golds), default=0.0)
    contains_correct = any(contains_answer(fa, g) for g in golds)
    em_correct = any(exact_match(fa, g) for g in golds)

    return {
        "sample_id": sample.id,
        "question": sample.question,
        "gold_answer": sample.answer,
        "gold_answers": sample.answers,
        "policy": policy.name,
        "final_answer": result.final_answer,
        "is_correct": result.success,
        "em_correct": em_correct,
        "f1_score": f1_score,
        "contains_correct": contains_correct,
        "steps": steps,
        "n_steps": len(result.steps),
        "tool_calls": result.total_tool_calls,
        "total_tokens": result.total_tokens,
        "failure_reason": result.failure_reason,
    }


def _has_parse_failure(result):
    """Check if any step in the episode had a parse failure.

    Previously this filtered to ``steering.action == "jes_steering"`` but that
    field is never written into the steering payload, so the function silently
    returned False for every episode.  Since every step in a JES episode is
    steered, checking for parse failures on *any* step is both correct and safe.
    """
    for step in result.get("steps", []):
        if step.get("parse_failure_reason"):
            return True
    return False


def _get_step(result, step_idx):
    for step in result.get("steps", []):
        if step.get("step_idx") == step_idx:
            return step
    return None


def compute_activation_stats(results, subset_ids=None):
    """Compute step-1 behavior and true second-search activation metrics."""
    by_id = {r["sample_id"]: r for r in results}
    if subset_ids:
        selected_ids = sorted(set(subset_ids) & set(by_id.keys()))
    else:
        selected_ids = sorted(by_id.keys())

    if not selected_ids:
        return {
            "step1_search_count": 0,
            "step1_search_rate": 0.0,
            "step1_final_count": 0,
            "step1_final_rate": 0.0,
            "step1_parse_failure_count": 0,
            "step1_parse_failure_rate": 0.0,
            "step1_other_count": 0,
            "step1_other_rate": 0.0,
            "step1_missing_count": 0,
            "step1_missing_rate": 0.0,
            "second_search_activation_count": 0,
            "second_search_activation_rate": 0.0,
        }

    counts = {
        "step1_search_count": 0,
        "step1_final_count": 0,
        "step1_parse_failure_count": 0,
        "step1_other_count": 0,
        "step1_missing_count": 0,
        "second_search_activation_count": 0,
    }

    for sample_id in selected_ids:
        result = by_id[sample_id]
        step1 = _get_step(result, 1)
        if step1 is None:
            counts["step1_missing_count"] += 1
        elif step1.get("parse_failure_reason"):
            counts["step1_parse_failure_count"] += 1
        elif step1.get("action") == "search":
            counts["step1_search_count"] += 1
        elif step1.get("action") == "final" or (
            step1.get("action") is None and step1.get("final_answer") is not None
        ):
            counts["step1_final_count"] += 1
        else:
            counts["step1_other_count"] += 1

        if result.get("tool_calls", 0) > 1:
            counts["second_search_activation_count"] += 1

    n = len(selected_ids)
    return {
        **counts,
        "step1_search_rate": counts["step1_search_count"] / n,
        "step1_final_rate": counts["step1_final_count"] / n,
        "step1_parse_failure_rate": counts["step1_parse_failure_count"] / n,
        "step1_other_rate": counts["step1_other_count"] / n,
        "step1_missing_rate": counts["step1_missing_count"] / n,
        "second_search_activation_rate": counts["second_search_activation_count"] / n,
    }


def compute_stats(bl_results, policy_results, subset_ids=None):
    """Compute paired stats including parse-failure-corrected metrics."""
    bl_by_id = {r["sample_id"]: r for r in bl_results}
    po_by_id = {r["sample_id"]: r for r in policy_results}

    if subset_ids:
        common = sorted(subset_ids & set(bl_by_id.keys()) & set(po_by_id.keys()))
    else:
        common = sorted(set(bl_by_id.keys()) & set(po_by_id.keys()))

    if not common:
        return {"n": 0}

    bl_correct = [bl_by_id[sid]["is_correct"] for sid in common]
    po_correct = [po_by_id[sid]["is_correct"] for sid in common]

    n = len(common)
    rescued = sum(1 for b, p in zip(bl_correct, po_correct) if not b and p)
    regressed = sum(1 for b, p in zip(bl_correct, po_correct) if b and not p)

    # Parse failure tracking (corrected metrics)
    n_parse_failures = sum(1 for sid in common if _has_parse_failure(po_by_id[sid]))
    rescued_accidental = sum(
        1 for sid in common
        if not bl_by_id[sid]["is_correct"]
        and po_by_id[sid]["is_correct"]
        and _has_parse_failure(po_by_id[sid])
    )
    rescued_genuine = rescued - rescued_accidental

    # ── Behavioral metrics: search rate & count changes ──────────
    def _count_searches(r):
        return sum(1 for s in r.get("steps", []) if s.get("action") == "search")

    bl_searches = [_count_searches(bl_by_id[sid]) for sid in common]
    po_searches = [_count_searches(po_by_id[sid]) for sid in common]

    bl_search_rate = sum(1 for s in bl_searches if s > 0) / n
    po_search_rate = sum(1 for s in po_searches if s > 0) / n
    bl_second_search_rate = sum(1 for s in bl_searches if s >= 2) / n
    po_second_search_rate = sum(1 for s in po_searches if s >= 2) / n
    avg_search_delta = sum(p - b for b, p in zip(bl_searches, po_searches)) / n

    # Rescued samples that ALSO changed search behavior (causal evidence)
    rescued_with_more_search = sum(
        1 for sid in common
        if not bl_by_id[sid]["is_correct"] and po_by_id[sid]["is_correct"]
        and _count_searches(po_by_id[sid]) > _count_searches(bl_by_id[sid])
    )

    # ── Behavior flip matrix (1st search & 2nd search) ────────────
    # 2×2 matrix: rows = baseline behavior, cols = steered behavior
    # Each cell also tracks accuracy to answer: did the flip help?
    def _build_flip_matrix(bl_counts, po_counts, bl_correct_list, po_correct_list, threshold):
        """Build a 2x2 flip matrix at a given search-count threshold.
        threshold=1: did the sample search at all?
        threshold=2: did the sample do a 2nd search?
        """
        cells = {
            "keep_search": {"n": 0, "bl_acc": 0, "po_acc": 0},
            "lost_search": {"n": 0, "bl_acc": 0, "po_acc": 0},
            "gained_search": {"n": 0, "bl_acc": 0, "po_acc": 0},
            "keep_no_search": {"n": 0, "bl_acc": 0, "po_acc": 0},
        }
        for bc, pc, ba, pa in zip(bl_counts, po_counts, bl_correct_list, po_correct_list):
            bl_did = bc >= threshold
            po_did = pc >= threshold
            if bl_did and po_did:
                key = "keep_search"
            elif bl_did and not po_did:
                key = "lost_search"
            elif not bl_did and po_did:
                key = "gained_search"
            else:
                key = "keep_no_search"
            cells[key]["n"] += 1
            cells[key]["bl_acc"] += int(ba)
            cells[key]["po_acc"] += int(pa)
        # Convert acc sums to rates
        for cell in cells.values():
            cn = cell["n"]
            cell["bl_acc_rate"] = cell["bl_acc"] / cn if cn > 0 else 0.0
            cell["po_acc_rate"] = cell["po_acc"] / cn if cn > 0 else 0.0
        return cells

    flip_1st = _build_flip_matrix(bl_searches, po_searches, bl_correct, po_correct, threshold=1)
    flip_2nd = _build_flip_matrix(bl_searches, po_searches, bl_correct, po_correct, threshold=2)

    # ── Multi-metric scoring (F1, contains) ─────────────────────
    bl_f1 = [bl_by_id[sid].get("f1_score", 0.0) for sid in common]
    po_f1 = [po_by_id[sid].get("f1_score", 0.0) for sid in common]
    bl_contains = [bl_by_id[sid].get("contains_correct", False) for sid in common]
    po_contains = [po_by_id[sid].get("contains_correct", False) for sid in common]

    # Contains-based rescued/regressed (more lenient than EM)
    rescued_contains = sum(1 for b, p in zip(bl_contains, po_contains) if not b and p)
    regressed_contains = sum(1 for b, p in zip(bl_contains, po_contains) if b and not p)

    stats = {
        "n": n,
        "baseline_rate": sum(bl_correct) / n,
        "policy_rate": sum(po_correct) / n,
        "rescued": rescued,
        "regressed": regressed,
        "net_gain": rescued - regressed,
        "rescue_rate": rescued / n,
        "regression_rate": regressed / n,
        # Corrected metrics (excluding parse-failure-induced rescues)
        "parse_failures": n_parse_failures,
        "rescued_genuine": rescued_genuine,
        "rescued_accidental": rescued_accidental,
        "net_gain_corrected": rescued_genuine - regressed,
        # ── Multi-metric answer quality ──────────────────────────
        "bl_f1_mean": sum(bl_f1) / n,
        "po_f1_mean": sum(po_f1) / n,
        "f1_delta": (sum(po_f1) - sum(bl_f1)) / n,
        "bl_contains_rate": sum(bl_contains) / n,
        "po_contains_rate": sum(po_contains) / n,
        "rescued_contains": rescued_contains,
        "regressed_contains": regressed_contains,
        "net_gain_contains": rescued_contains - regressed_contains,
        # Behavioral metrics (primary for causal validation)
        "bl_search_rate": bl_search_rate,
        "po_search_rate": po_search_rate,
        "search_rate_delta": po_search_rate - bl_search_rate,
        "bl_second_search_rate": bl_second_search_rate,
        "po_second_search_rate": po_second_search_rate,
        "second_search_rate_delta": po_second_search_rate - bl_second_search_rate,
        "avg_search_count_delta": avg_search_delta,
        "rescued_with_more_search": rescued_with_more_search,
        "rescued_causal_pct": rescued_with_more_search / rescued if rescued > 0 else 0.0,
        # Behavior flip matrices
        "flip_1st_search": flip_1st,
        "flip_2nd_search": flip_2nd,
    }
    stats.update(compute_activation_stats(policy_results, set(common)))

    if n > 0:
        mcn = mcnemar_test(bl_correct, po_correct)
        stats["mcnemar_p"] = mcn["mcnemar_p"]

        boot = bootstrap_ci(bl_correct, po_correct, metric="success_diff")
        stats["success_diff"] = boot["observed"]
        stats["success_diff_ci"] = (boot["ci_lower"], boot["ci_upper"])

    return stats


def cost_stats(results):
    """Cost percentiles."""
    tokens = [r["total_tokens"] for r in results]
    steps = [r["n_steps"] for r in results]
    tools = [r["tool_calls"] for r in results]
    return {
        "tokens_mean": np.mean(tokens), "tokens_p95": np.percentile(tokens, 95),
        "steps_mean": np.mean(steps), "steps_p95": np.percentile(steps, 95),
        "tool_calls_mean": np.mean(tools),
    }



def generate_report_md(report: dict, out_path: Path):
    """Generate report.md from report dict."""
    lines = [
        "# Verify-Critical Evaluation Report",
        f"\n**Generated**: {report['timestamp']}",
        f"\n**Dataset**: {report.get('dataset_label', 'PopQA')} (n={report['n_total']})",
        "",
        "## Summary",
        "",
        "| Subset | N | Baseline | JES | Rescued | Regressed | Net | ParseFail | Acc.Rescue | Corr.Rescue | Corr.Net |",
        "|--------|---|----------|-----|---------|-----------|-----|-----------|------------|-------------|----------|",
    ]

    for name, stats in [("Full", report["full_stats"]),
                        ("Verify-Critical", report["vc_stats"])]:
        if stats["n"] > 0:
            ngc = stats.get("net_gain_corrected")
            ngc_str = f"{ngc:+d}" if isinstance(ngc, int) else str(ngc) if ngc is not None else "?"
            lines.append(
                f"| {name} | {stats['n']} | {stats['baseline_rate']*100:.1f}% | "
                f"{stats['policy_rate']*100:.1f}% | {stats['rescued']} | "
                f"{stats['regressed']} | {stats['net_gain']:+d} | "
                f"{stats.get('parse_failures', '?')} | "
                f"{stats.get('rescued_accidental', '?')} | "
                f"{stats.get('rescued_genuine', '?')} | "
                f"{ngc_str} |"
            )

    fs = report["full_stats"]
    lines.extend([
        "",
        "## Verify-Critical Headline",
        "",
        f"- **verify_critical samples**: {report['n_vc']} ({100*report['n_vc']/report['n_total']:.1f}%)",
        f"- **JES rescue rate on verify_critical**: {report['vc_stats'].get('policy_rate', 0)*100:.1f}%",
        "",
        "## Corrected Metrics (Full Dataset)",
        "",
        f"- Parse-failure episodes: {fs.get('parse_failures', '?')} / {fs['n']} "
        f"({100*fs.get('parse_failures',0)/fs['n']:.1f}%)",
        f"- Rescued (raw): {fs.get('rescued', '?')}",
        f"- Rescued (accidental — parse-failure episodes that answered correctly): "
        f"{fs.get('rescued_accidental', '?')}",
        f"- Rescued (genuine): {fs.get('rescued_genuine', '?')}",
        f"- Regressed: {fs.get('regressed', '?')}",
        f"- Net gain (raw): {fs.get('net_gain', '?'):+d}" if isinstance(fs.get('net_gain'), int) else
        f"- Net gain (raw): {fs.get('net_gain', '?')}",
        f"- **Net gain (corrected): {fs.get('net_gain_corrected', '?'):+d}**" if isinstance(fs.get('net_gain_corrected'), int) else
        f"- **Net gain (corrected): {fs.get('net_gain_corrected', '?')}**",
        "",
        "## Second-search Activation",
        "",
        f"- Full: step1_search={fs.get('step1_search_count', 0)}/{fs['n']} "
        f"({100*fs.get('step1_search_rate', 0.0):.1f}%), "
        f"step1_final={fs.get('step1_final_count', 0)}/{fs['n']} "
        f"({100*fs.get('step1_final_rate', 0.0):.1f}%), "
        f"step1_parse_failure={fs.get('step1_parse_failure_count', 0)}/{fs['n']} "
        f"({100*fs.get('step1_parse_failure_rate', 0.0):.1f}%), "
        f"actual second search={fs.get('second_search_activation_count', 0)}/{fs['n']} "
        f"({100*fs.get('second_search_activation_rate', 0.0):.1f}%)",
    ])

    if report["vc_stats"]["n"] > 0:
        vs = report["vc_stats"]
        lines.extend([
            f"- Verify-Critical: step1_search={vs.get('step1_search_count', 0)}/{vs['n']} "
            f"({100*vs.get('step1_search_rate', 0.0):.1f}%), "
            f"step1_final={vs.get('step1_final_count', 0)}/{vs['n']} "
            f"({100*vs.get('step1_final_rate', 0.0):.1f}%), "
            f"step1_parse_failure={vs.get('step1_parse_failure_count', 0)}/{vs['n']} "
            f"({100*vs.get('step1_parse_failure_rate', 0.0):.1f}%), "
            f"actual second search={vs.get('second_search_activation_count', 0)}/{vs['n']} "
            f"({100*vs.get('second_search_activation_rate', 0.0):.1f}%)",
            "",
        ])

    lines.extend([
        "## Statistical Tests",
        "",
    ])

    if "mcnemar_p" in report["full_stats"]:
        lines.append(f"- McNemar p-value (full): {report['full_stats']['mcnemar_p']:.4f}")
    if "success_diff_ci" in report["full_stats"]:
        ci = report["full_stats"]["success_diff_ci"]
        lines.append(f"- Success diff 95% CI: [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")

    lines.extend([
        "",
        "## Cost Analysis",
        "",
        "| Policy | Tokens (mean) | Tokens (p95) | Steps (mean) |",
        "|--------|---------------|--------------|--------------|",
    ])

    for name, cost in [("Baseline", report["cost_bl"]), ("JES", report["cost_jes"])]:
        lines.append(
            f"| {name} | {cost['tokens_mean']:.0f} | {cost['tokens_p95']:.0f} | "
            f"{cost['steps_mean']:.2f} |"
        )

    # Sweep summary table (only when multiple configs were run)
    if "sweep_summary" in report and len(report["sweep_summary"]) > 1:
        lines.extend([
            "",
            "## Sweep Config Summary",
            "",
            "| Config | τ | ρ_max | Rescued | Regr | Net | ParseFail | Step1Search | 2ndSearch | Acc | Corr.Rescue | Corr.Net |",
            "|--------|---|-------|---------|------|-----|-----------|-------------|-----------|-----|-------------|----------|",
        ])
        best_tag = report.get("best_config", "")
        for tag, e in report["sweep_summary"].items():
            marker = " ✓" if tag == best_tag else ""
            ngc = e.get("net_gain_corrected", "?")
            ngc_str = f"{ngc:+d}" if isinstance(ngc, int) else str(ngc)
            lines.append(
                f"| {tag}{marker} | {e['tau']} | {e['max_rho']} | "
                f"{e.get('rescued','?')} | {e.get('regressed','?')} | {e.get('net_gain','?'):+d} | "
                f"{e.get('parse_failures','?')} | {e.get('step1_search_count','?')} | "
                f"{e.get('second_search_activation_count','?')} | {e.get('rescued_accidental','?')} | "
                f"{e.get('rescued_genuine','?')} | {ngc_str} |"
                if isinstance(e.get('net_gain'), int)
                else
                f"| {tag}{marker} | {e['tau']} | {e['max_rho']} | "
                f"{e.get('rescued','?')} | {e.get('regressed','?')} | {e.get('net_gain','?')} | "
                f"{e.get('parse_failures','?')} | {e.get('step1_search_count','?')} | "
                f"{e.get('second_search_activation_count','?')} | {e.get('rescued_accidental','?')} | "
                f"{e.get('rescued_genuine','?')} | {ngc_str} |"
            )

    lines.extend([
        "",
        "## Mining Distribution",
        "",
        f"- verify_critical: {report['n_vc']}",
        f"- verify_harmful: {report['n_vh']}",
        f"- indifferent: {report['n_ind']}",
    ])

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Full verify-critical pipeline")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--corpus-path", type=str, required=True)
    parser.add_argument("--direction-path", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of samples to run. Omit or set to -1 for the full dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="results/verify_critical_v1")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--tau", type=float, default=1.5,
                        help="Legacy single-tau (ignored when --tau-sweep is set)")
    parser.add_argument("--tau-sweep", type=float, nargs="+", default=None,
                        help="tau_step2 values to sweep, e.g. 0.0 0.1")
    parser.add_argument("--max-rho-sweep", type=float, nargs="+", default=None,
                        help="max_rho values to sweep, e.g. 0.25 0.75 1.5")
    parser.add_argument("--score-mode", type=str, default="any")
    parser.add_argument("--dataset", type=str, default="popqa",
                        choices=["popqa", "hotpotqa"],
                        help="Dataset to use: popqa or hotpotqa")
    parser.add_argument("--type-filter", type=str, default=None,
                        help="Filter by question type, e.g. 'bridge' for HotpotQA")
    parser.add_argument("--alpha-max", type=float, default=8.0,
                        help="Maximum |alpha| safety clamp (prevents format corruption). "
                             "Default 8.0 based on empirical format-stability analysis.")
    parser.add_argument("--normalize-rms", type=float, default=1.0,
                        help="Normalize direction to this target RMS before use. "
                             "Set to 1.0 (default) so that rho maps to the same alpha "
                             "for ALL directions, ensuring fair alpha_max clipping. "
                             "Set to 0 or negative to disable normalization.")
    args = parser.parse_args()

    # Build sweep grid
    tau_values = args.tau_sweep if args.tau_sweep else [args.tau]
    max_rho_values = args.max_rho_sweep if args.max_rho_sweep else [0.25]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # Load resources
    print("=" * 70)
    print("  VERIFY-CRITICAL PIPELINE")
    print("=" * 70)

    print("\n[1/5] Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model)

    print("[2/5] Loading direction...")
    norm_rms = args.normalize_rms if args.normalize_rms > 0 else None
    direction, dir_meta = load_direction(args.direction_path, normalize_rms=norm_rms)
    direction_rms = float(np.sqrt(np.mean(direction ** 2)))
    print(f"  Original RMS: {dir_meta['original_rms']:.6f}  Norm: {dir_meta['original_norm']:.4f}")
    print(f"  Loaded   RMS: {direction_rms:.6f}  Norm: {dir_meta['norm']:.4f}")
    if norm_rms is not None:
        print(f"  Normalized to target RMS = {norm_rms}")
        expected_alpha = 0.5 * (0.82 / direction_rms)  # rough estimate at rho=0.5
        print(f"  Estimated alpha at rho=0.5: {expected_alpha:.2f} (alpha_max={args.alpha_max})")

    print("[3/5] Loading dataset...")
    if args.dataset == "hotpotqa":
        dataset = HotpotQADataset(args.data_path)
        samples = dataset.get_subset(args.n_samples, seed=args.seed, type_filter=args.type_filter)
        ds_label = f"HotpotQA"
        if args.type_filter:
            ds_label += f" (type={args.type_filter})"
    else:
        dataset = PopQADataset(args.data_path)
        samples = dataset.get_subset(args.n_samples, seed=args.seed)
        ds_label = "PopQA"
    print(f"  Selected {len(samples)} {ds_label} samples")

    # Setup agent
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

    # Phase 1: Mining
    print("\n[4/5] PHASE 1: Mining...")
    baseline_policy = Baseline1HopPolicy()
    oracle_policy = Oracle2HopPolicy()

    bl_results = []
    orc_results = []

    print("  Running baseline_1hop...")
    for s in tqdm(samples, desc="baseline"):
        bl_results.append(run_episode(agent, s, baseline_policy, args.score_mode))

    print("  Running oracle_2hop...")
    for s in tqdm(samples, desc="oracle"):
        orc_results.append(run_episode(agent, s, oracle_policy, args.score_mode))

    # Build manifest
    bl_by_id = {r["sample_id"]: r for r in bl_results}
    orc_by_id = {r["sample_id"]: r for r in orc_results}

    manifest = []
    for s in samples:
        bl_ok = bl_by_id[s.id]["is_correct"]
        orc_ok = orc_by_id[s.id]["is_correct"]
        if not bl_ok and orc_ok:
            label = "verify_critical"
        elif bl_ok and not orc_ok:
            label = "verify_harmful"
        else:
            label = "indifferent"
        manifest.append({"sample_id": s.id, "label": label})

    vc_ids = {m["sample_id"] for m in manifest if m["label"] == "verify_critical"}
    vh_ids = {m["sample_id"] for m in manifest if m["label"] == "verify_harmful"}
    ind_ids = {m["sample_id"] for m in manifest if m["label"] == "indifferent"}

    print(f"\n  Mining results:")
    print(f"    verify_critical: {len(vc_ids)}")
    print(f"    verify_harmful: {len(vh_ids)}")
    print(f"    indifferent: {len(ind_ids)}")

    # Save mining outputs
    with open(out_dir / "baseline_results.jsonl", "w") as f:
        for r in bl_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_dir / "oracle_results.jsonl", "w") as f:
        for r in orc_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_dir / "manifest.jsonl", "w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")

    # Phase 2: JES sweep over (tau, max_rho) grid
    sweep_grid = [(tau, mr) for tau in tau_values for mr in max_rho_values]
    print(f"\n[5/7] PHASE 2: JES Sweep ({len(sweep_grid)} configs)...")
    print(f"  tau_values:     {tau_values}")
    print(f"  max_rho_values: {max_rho_values}")

    all_sweep_results = {}
    best_config = None
    best_net_gain = -999

    for tau_val, mr_val in sweep_grid:
        tag = f"tau{tau_val:.2f}_rho{mr_val:.2f}"
        print(f"\n  --- JES config: tau={tau_val}, max_rho={mr_val}, alpha_max={args.alpha_max} ---")
        jes_config = JESConfig(tau=tau_val, max_rho=mr_val, alpha_max=args.alpha_max)
        jes_policy = JESStep2OnlyPolicy(config=jes_config, direction=direction)

        jes_results = []
        for s in tqdm(samples, desc=f"JES({tag})"):
            jes_results.append(run_episode(agent, s, jes_policy, args.score_mode))

        # Save per-config results
        with open(out_dir / f"jes_results_{tag}.jsonl", "w") as f:
            for r in jes_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        full_stats = compute_stats(bl_results, jes_results)
        vc_stats = compute_stats(bl_results, jes_results, vc_ids) if vc_ids else {"n": 0}

        sweep_entry = {
            "tau": tau_val, "max_rho": mr_val, "tag": tag,
            "full_stats": full_stats, "vc_stats": vc_stats,
            "cost_jes": cost_stats(jes_results),
        }
        all_sweep_results[tag] = sweep_entry

        net = full_stats.get("net_gain", 0)
        net_corrected = full_stats.get("net_gain_corrected", net)
        pf = full_stats.get("parse_failures", 0)
        acc = full_stats.get("rescued_accidental", 0)
        print(
            f"    Full: rescued={full_stats.get('rescued',0)} "
            f"(accidental={acc}) regressed={full_stats.get('regressed',0)} "
            f"net={net:+d} | corrected net={net_corrected:+d} | parse_fail={pf}"
        )
        print(
            f"    Activation: step1_search={full_stats.get('step1_search_count',0)}/{full_stats['n']} "
            f"({100*full_stats.get('step1_search_rate',0.0):.1f}%), "
            f"step1_final={full_stats.get('step1_final_count',0)}/{full_stats['n']} "
            f"({100*full_stats.get('step1_final_rate',0.0):.1f}%), "
            f"step1_parse_fail={full_stats.get('step1_parse_failure_count',0)}/{full_stats['n']} "
            f"({100*full_stats.get('step1_parse_failure_rate',0.0):.1f}%), "
            f"2nd_search={full_stats.get('second_search_activation_count',0)}/{full_stats['n']} "
            f"({100*full_stats.get('second_search_activation_rate',0.0):.1f}%)"
        )
        if vc_stats["n"] > 0:
            print(f"    VC:   rescued={vc_stats.get('rescued',0)} "
                  f"(genuine={vc_stats.get('rescued_genuine','?')})/{vc_stats['n']}")
            print(
                f"    VC Activation: step1_search={vc_stats.get('step1_search_count',0)}/{vc_stats['n']} "
                f"({100*vc_stats.get('step1_search_rate',0.0):.1f}%), "
                f"2nd_search={vc_stats.get('second_search_activation_count',0)}/{vc_stats['n']} "
                f"({100*vc_stats.get('second_search_activation_rate',0.0):.1f}%)"
            )

        # Select best config by corrected net gain (excludes accidental rescues).
        # Falls back to raw net_gain when corrected metric is unavailable.
        if net_corrected > best_net_gain:
            best_net_gain = net_corrected
            best_config = tag

    # Also save a canonical "jes_results.jsonl" from the best config (or first)
    best_tag = best_config or list(all_sweep_results.keys())[0]
    import shutil
    best_jes_path = out_dir / f"jes_results_{best_tag}.jsonl"
    if best_jes_path.exists():
        shutil.copy(best_jes_path, out_dir / "jes_results.jsonl")

    # Phase 3: Control Budget Diagnosis
    print(f"\n[6/7] PHASE 3: Control Budget Diagnosis...")
    # Load the best JES results for diagnosis
    jes_for_diag = []
    with open(out_dir / "jes_results.jsonl") as f:
        for line in f:
            if line.strip():
                jes_for_diag.append(json.loads(line))

    diag = compute_diagnosis(bl_results, orc_results, jes_for_diag)
    print_diagnosis(diag)

    with open(out_dir / "diagnosis.json", "w") as f:
        json.dump(diag, f, indent=2, ensure_ascii=False)

    # Phase 4: Build final report
    print(f"\n[7/7] PHASE 4: Final Report...")
    best_sweep = all_sweep_results[best_tag]
    cost_bl = cost_stats(bl_results)

    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset_label": ds_label,
        "n_total": len(samples),
        "n_vc": len(vc_ids), "n_vh": len(vh_ids), "n_ind": len(ind_ids),
        "full_stats": best_sweep["full_stats"],
        "vc_stats": best_sweep["vc_stats"],
        "cost_bl": cost_bl,
        "cost_jes": best_sweep["cost_jes"],
        "config": {
            "tau": best_sweep["tau"], "max_rho": best_sweep["max_rho"],
            "alpha_max": args.alpha_max,
            "normalize_rms": norm_rms,
            "direction_original_rms": dir_meta["original_rms"],
            "direction_loaded_rms": direction_rms,
            "layer": args.layer, "model": args.model,
        },
        "sweep_summary": {tag: {
            "tau": e["tau"], "max_rho": e["max_rho"],
            # Raw metrics
            "rescued": e["full_stats"].get("rescued", 0),
            "regressed": e["full_stats"].get("regressed", 0),
            "net_gain": e["full_stats"].get("net_gain", 0),
            "vc_rescued": e["vc_stats"].get("rescued", 0),
            # Corrected metrics (primary headline)
            "parse_failures": e["full_stats"].get("parse_failures", 0),
            "rescued_accidental": e["full_stats"].get("rescued_accidental", 0),
            "rescued_genuine": e["full_stats"].get("rescued_genuine", 0),
            "net_gain_corrected": e["full_stats"].get("net_gain_corrected", 0),
            "vc_rescued_genuine": e["vc_stats"].get("rescued_genuine", 0),
            "step1_search_count": e["full_stats"].get("step1_search_count", 0),
            "step1_search_rate": e["full_stats"].get("step1_search_rate", 0.0),
            "step1_final_count": e["full_stats"].get("step1_final_count", 0),
            "step1_parse_failure_count": e["full_stats"].get("step1_parse_failure_count", 0),
            "second_search_activation_count": e["full_stats"].get("second_search_activation_count", 0),
            "second_search_activation_rate": e["full_stats"].get("second_search_activation_rate", 0.0),
        } for tag, e in all_sweep_results.items()},
        "best_config": best_tag,
        "verdict": diag["verdict"],
    }

    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    generate_report_md(report, out_dir / "report.md")

    # Print summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"\nBest config: {best_tag}")
    print(f"  alpha_max: {args.alpha_max}")
    fs = best_sweep["full_stats"]
    print(f"\n[FULL DATASET] n={fs['n']}")
    print(f"  Baseline: {fs['baseline_rate']*100:.1f}%")
    print(f"  JES:      {fs['policy_rate']*100:.1f}%")
    print(f"  Rescued: {fs['rescued']} (genuine={fs.get('rescued_genuine','?')}, accidental={fs.get('rescued_accidental','?')})")
    print(f"  Regressed: {fs['regressed']} | Net: {fs['net_gain']:+d} (corrected: {fs.get('net_gain_corrected', '?'):+d})")
    print(f"  Parse failures: {fs.get('parse_failures', '?')}/{fs['n']} ({100*fs.get('parse_failures',0)/fs['n']:.1f}%)")
    print(
        f"  Activation: step1_search={fs.get('step1_search_count',0)}/{fs['n']} "
        f"({100*fs.get('step1_search_rate',0.0):.1f}%), "
        f"step1_final={fs.get('step1_final_count',0)}/{fs['n']} "
        f"({100*fs.get('step1_final_rate',0.0):.1f}%), "
        f"step1_parse_fail={fs.get('step1_parse_failure_count',0)}/{fs['n']} "
        f"({100*fs.get('step1_parse_failure_rate',0.0):.1f}%), "
        f"2nd_search={fs.get('second_search_activation_count',0)}/{fs['n']} "
        f"({100*fs.get('second_search_activation_rate',0.0):.1f}%)"
    )

    vs = best_sweep["vc_stats"]
    if vs["n"] > 0:
        print(f"\n[VERIFY-CRITICAL] n={vs['n']}")
        print(f"  Baseline: {vs['baseline_rate']*100:.1f}% (0% by definition)")
        print(f"  JES:      {vs['policy_rate']*100:.1f}%  ← HEADLINE RESCUE RATE")
        print(f"  Rescued: {vs['rescued']} (genuine={vs.get('rescued_genuine','?')}, accidental={vs.get('rescued_accidental','?')})")
        print(f"  Parse failures: {vs.get('parse_failures', '?')}/{vs['n']}")
        print(
            f"  Activation: step1_search={vs.get('step1_search_count',0)}/{vs['n']} "
            f"({100*vs.get('step1_search_rate',0.0):.1f}%), "
            f"step1_final={vs.get('step1_final_count',0)}/{vs['n']} "
            f"({100*vs.get('step1_final_rate',0.0):.1f}%), "
            f"step1_parse_fail={vs.get('step1_parse_failure_count',0)}/{vs['n']} "
            f"({100*vs.get('step1_parse_failure_rate',0.0):.1f}%), "
            f"2nd_search={vs.get('second_search_activation_count',0)}/{vs['n']} "
            f"({100*vs.get('second_search_activation_rate',0.0):.1f}%)"
        )

    print(f"\n[VERDICT] {diag['verdict']}")
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Results saved to: {out_dir}")
    print(f"Report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()

