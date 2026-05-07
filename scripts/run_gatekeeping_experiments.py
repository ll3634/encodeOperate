#!/usr/bin/env python3
"""
Evidence-Seeking Controller — Gatekeeping Experiments (Gates 1-3).

Gate 1: Step-wise decision-margin distribution
  - Log decision-aligned margin at each step (right before Action:/Final commitment)
  - Report fraction of near-boundary decisions (|margin| < m0) for step1 vs step2+

Gate 2: Step2+ behavior change with forced push-to-Action
  - For near-boundary step2+ samples, apply intervention to increase Action probability
  - Measure: additional tool call rate, query change rate, correctness delta

Gate 3: Direction slope test on step2+
  - Sweep rho in {-r, 0, +r} on step2+ decisions, estimate dm/dρ
  - If slope is weak/unstable, flag for step2+-specific direction extraction

Usage:
  cd tmc/scripts/e2e_agent
  python scripts/run_gatekeeping_experiments.py \
      --data-path data/popqa/popqa_test.jsonl \
      --corpus-path data/popqa/corpus.jsonl \
      --direction-path steering/directions/direction_search_v3.npz \
      --n-samples 100 --out results/gatekeeping_v1
"""

import json
import argparse
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.popqa import PopQADataset, build_popqa_corpus
from tools.search_tool import SearchTool
from tools.calculator_tool import CalculatorTool
from agent.react_loop import ReActAgent, AgentConfig
from agent.policies import BaselinePolicy, Policy, SteeringDecision
from steering.directions import load_direction


# ============================================================================
# Gate 1: Step-wise margin logging (baseline run with per-step margin recording)
# ============================================================================

def run_gate1(agent, samples, out_dir):
    """Run baseline agent, record decision-aligned margin at every step.

    Returns list of dicts: one per decision point, with fields:
      sample_id, step_idx, margin, action_taken (Action/Final), is_step1
    """
    print("\n" + "=" * 60)
    print("  GATE 1: Step-wise decision-margin distribution")
    print("=" * 60)

    policy = BaselinePolicy()
    all_decisions = []

    for sample in tqdm(samples, desc="Gate1-baseline"):
        result = agent.run(
            question=sample.question,
            policy=policy,
            gold_answer=sample.answers,
            episode_id=sample.id,
            target_side="positive",
        )
        for step in result.steps:
            margin = step.margin_before
            if margin is None:
                continue
            action_taken = "Final" if step.final_answer else "Action"
            all_decisions.append({
                "sample_id": sample.id,
                "step_idx": step.step_idx,
                "margin": float(margin),
                "action_taken": action_taken,
                "is_step1": step.step_idx == 0,
                "is_correct": result.success,
                "question": sample.question[:200],
            })

    # Save raw decisions
    dec_path = out_dir / "gate1_decisions.jsonl"
    with open(dec_path, "w") as f:
        for d in all_decisions:
            f.write(json.dumps(d, default=str) + "\n")

    # Analyze
    step1 = [d for d in all_decisions if d["is_step1"]]
    step2p = [d for d in all_decisions if not d["is_step1"]]

    thresholds = [0.5, 1.0, 1.5, 2.0, 3.0]
    report = {
        "total_decisions": len(all_decisions),
        "step1_count": len(step1),
        "step2p_count": len(step2p),
        "step1_margins": _margin_stats([d["margin"] for d in step1]),
        "step2p_margins": _margin_stats([d["margin"] for d in step2p]),
        "near_boundary_fractions": {},
    }

    for m0 in thresholds:
        s1_near = sum(1 for d in step1 if abs(d["margin"]) < m0)
        s2_near = sum(1 for d in step2p if abs(d["margin"]) < m0)
        report["near_boundary_fractions"][f"m0={m0}"] = {
            "step1": s1_near / max(len(step1), 1),
            "step2p": s2_near / max(len(step2p), 1),
            "step1_count": s1_near,
            "step2p_count": s2_near,
        }

    # Print summary
    print(f"\n  Total decisions: {len(all_decisions)}")
    print(f"  Step 1: {len(step1)}  |  Step 2+: {len(step2p)}")
    if step1:
        print(f"  Step 1 margin: mean={np.mean([d['margin'] for d in step1]):.3f}, "
              f"std={np.std([d['margin'] for d in step1]):.3f}")
    if step2p:
        print(f"  Step 2+ margin: mean={np.mean([d['margin'] for d in step2p]):.3f}, "
              f"std={np.std([d['margin'] for d in step2p]):.3f}")
    print("\n  Near-boundary fractions (|margin| < m0):")
    for m0 in thresholds:
        nb = report["near_boundary_fractions"][f"m0={m0}"]
        print(f"    m0={m0}: step1={nb['step1']:.1%} ({nb['step1_count']}), "
              f"step2+={nb['step2p']:.1%} ({nb['step2p_count']})")

    # Gate 1 pass/fail: need sufficient step2+ decisions (prey exists)
    MIN_STEP2P_COUNT = 5
    gate_pass = len(step2p) >= MIN_STEP2P_COUNT
    report["gate_pass"] = gate_pass
    report["min_step2p_required"] = MIN_STEP2P_COUNT

    if gate_pass:
        print(f"\n  ✅ GATE 1 PASS: {len(step2p)} step2+ decisions found (need ≥ {MIN_STEP2P_COUNT})")
    else:
        print(f"\n  ❌ GATE 1 FAIL: Only {len(step2p)} step2+ decisions (need ≥ {MIN_STEP2P_COUNT})")

    # Save report
    with open(out_dir / "gate1_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    return all_decisions, report


def _margin_stats(margins):
    """Compute summary statistics for a list of margins."""
    if not margins:
        return {"count": 0}
    arr = np.array(margins)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "frac_positive": float(np.mean(arr > 0)),
        "frac_negative": float(np.mean(arr < 0)),
    }


# ============================================================================
# Step2-only forced policy for Gate 2
# ============================================================================

class Step2OnlyForcePolicy(Policy):
    """Force push-to-Action ONLY at step 2+ (step_idx >= 1).

    Step 1 (step_idx=0) is left unsteered (baseline behavior).
    At step 2+, uses direction-sign-agnostic forced adopt logic.
    """

    def __init__(self, rho_magnitude: float = 0.5, alpha_max: float = 2000.0):
        self.rho_magnitude = rho_magnitude
        self.alpha_max = alpha_max
        self._step_counter = 0

    @property
    def name(self) -> str:
        return "step2_only_force_adopt"

    def reset_episode(self):
        self._step_counter = 0

    def decide(self, margin_fn, target_side, hidden_rms, direction_rms):
        step_idx = self._step_counter
        self._step_counter += 1

        m0 = margin_fn(0.0)

        # Step 1: no intervention
        if step_idx == 0:
            return SteeringDecision(
                rho=0.0, alpha=0.0, policy_name=self.name,
                m_before=m0,
                details={"step_idx": step_idx, "intervened": False},
            )

        # Step 2+: force push-to-Action (direction-sign-agnostic)
        rho_pos = float(self.rho_magnitude)
        rho_neg = -float(self.rho_magnitude)
        m_pos = margin_fn(rho_pos)
        m_neg = margin_fn(rho_neg)

        # Pick rho that maximizes margin (pushes toward Action)
        rho = rho_pos if m_pos >= m_neg else rho_neg
        alpha = rho * (hidden_rms / direction_rms)
        alpha = float(np.clip(alpha, -self.alpha_max, self.alpha_max))

        return SteeringDecision(
            rho=rho, alpha=alpha, policy_name=self.name,
            m_before=m0,
            details={
                "step_idx": step_idx,
                "intervened": True,
                "rho_selected": rho,
                "margin_at_+rho": m_pos,
                "margin_at_-rho": m_neg,
            },
        )


# ============================================================================
# Gate 2: Step2+ behavior change with forced push-to-Action
# ============================================================================

def run_gate2(agent, samples, gate1_decisions, out_dir, m0_threshold=1.5):
    """Run Step2OnlyForcePolicy on samples that had near-boundary step2+ decisions.

    Compares against baseline to measure:
      - Additional tool call rate
      - Query change rate
      - Correctness delta
    """
    print("\n" + "=" * 60)
    print("  GATE 2: Step2+ behavior change with forced push-to-Action")
    print("=" * 60)

    # Identify samples with near-boundary step2+ decisions
    near_boundary_samples = set()
    for d in gate1_decisions:
        if not d["is_step1"] and abs(d["margin"]) < m0_threshold:
            near_boundary_samples.add(d["sample_id"])

    print(f"  Near-boundary step2+ samples (|margin| < {m0_threshold}): "
          f"{len(near_boundary_samples)}")

    if not near_boundary_samples:
        print("  WARNING: No near-boundary step2+ samples found. Gate 2 skipped.")
        report = {"skipped": True, "reason": "no_near_boundary_samples",
                  "m0_threshold": m0_threshold}
        with open(out_dir / "gate2_report.json", "w") as f:
            json.dump(report, f, indent=2)
        return report

    # Filter samples
    target_samples = [s for s in samples if s.id in near_boundary_samples]

    # Run baseline on target samples
    baseline_policy = BaselinePolicy()
    baseline_results = {}
    for sample in tqdm(target_samples, desc="Gate2-baseline"):
        result = agent.run(
            question=sample.question, policy=baseline_policy,
            gold_answer=sample.answers, episode_id=sample.id,
            target_side="positive",
        )
        baseline_results[sample.id] = result

    # Run Step2OnlyForce on target samples
    force_policy = Step2OnlyForcePolicy(rho_magnitude=0.5)
    force_results = {}
    for sample in tqdm(target_samples, desc="Gate2-step2force"):
        result = agent.run(
            question=sample.question, policy=force_policy,
            gold_answer=sample.answers, episode_id=sample.id,
            target_side="positive",
        )
        force_results[sample.id] = result

    # Compare
    comparisons = []
    for sid in near_boundary_samples:
        if sid not in baseline_results or sid not in force_results:
            continue
        bl = baseline_results[sid]
        fc = force_results[sid]

        bl_tool_calls = bl.total_tool_calls
        fc_tool_calls = fc.total_tool_calls
        extra_calls = fc_tool_calls - bl_tool_calls

        # Check if queries changed
        bl_queries = [s.action_input for s in bl.steps
                      if s.action and s.action.lower() == "search"]
        fc_queries = [s.action_input for s in fc.steps
                      if s.action and s.action.lower() == "search"]
        query_changed = bl_queries != fc_queries

        comparisons.append({
            "sample_id": sid,
            "bl_success": bl.success,
            "fc_success": fc.success,
            "bl_tool_calls": bl_tool_calls,
            "fc_tool_calls": fc_tool_calls,
            "extra_tool_calls": extra_calls,
            "query_changed": query_changed,
            "bl_final": bl.final_answer,
            "fc_final": fc.final_answer,
            "bl_steps": len(bl.steps),
            "fc_steps": len(fc.steps),
            "rescued": (not bl.success) and fc.success,
            "regressed": bl.success and (not fc.success),
        })

    # Aggregate
    n = len(comparisons)
    if n == 0:
        print("  WARNING: No valid comparisons. Gate 2 inconclusive.")
        report = {"skipped": True, "reason": "no_valid_comparisons"}
        with open(out_dir / "gate2_report.json", "w") as f:
            json.dump(report, f, indent=2)
        return report

    n_extra_calls = sum(1 for c in comparisons if c["extra_tool_calls"] > 0)
    n_query_changed = sum(1 for c in comparisons if c["query_changed"])
    n_rescued = sum(1 for c in comparisons if c["rescued"])
    n_regressed = sum(1 for c in comparisons if c["regressed"])
    bl_success_rate = sum(1 for c in comparisons if c["bl_success"]) / n
    fc_success_rate = sum(1 for c in comparisons if c["fc_success"]) / n

    report = {
        "m0_threshold": m0_threshold,
        "n_near_boundary_samples": len(near_boundary_samples),
        "n_compared": n,
        "additional_tool_call_rate": n_extra_calls / n,
        "query_change_rate": n_query_changed / n,
        "baseline_success_rate": bl_success_rate,
        "force_success_rate": fc_success_rate,
        "correctness_delta": fc_success_rate - bl_success_rate,
        "n_rescued": n_rescued,
        "n_regressed": n_regressed,
        "net_gain": n_rescued - n_regressed,
        "comparisons": comparisons,
    }

    print(f"\n  Compared: {n} samples")
    print(f"  Additional tool call rate: {n_extra_calls}/{n} = {n_extra_calls/n:.1%}")
    print(f"  Query change rate: {n_query_changed}/{n} = {n_query_changed/n:.1%}")
    print(f"  Baseline success: {bl_success_rate:.1%}")
    print(f"  Force success: {fc_success_rate:.1%}")
    print(f"  Δ correctness: {fc_success_rate - bl_success_rate:+.1%}")
    print(f"  Rescued: {n_rescued}  Regressed: {n_regressed}  Net: {n_rescued - n_regressed}")

    with open(out_dir / "gate2_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report




# ============================================================================
# Gate 3: Direction slope test on step2+
# ============================================================================

def run_gate3(agent, samples, gate1_decisions, out_dir,
              m0_threshold=1.5, rho_probe=0.25):
    """Test whether the step1 direction vector has stable slope on step2+ decisions.

    For each near-boundary step2+ decision, sweep rho in {-r, 0, +r} and
    estimate dm/dρ via central difference.
    """
    print("\n" + "=" * 60)
    print("  GATE 3: Direction slope test on step2+")
    print("=" * 60)

    # Identify near-boundary step2+ decisions
    near_boundary = [d for d in gate1_decisions
                     if not d["is_step1"] and abs(d["margin"]) < m0_threshold]

    print(f"  Near-boundary step2+ decisions (|margin| < {m0_threshold}): "
          f"{len(near_boundary)}")

    if not near_boundary:
        print("  WARNING: No near-boundary step2+ decisions. Gate 3 skipped.")
        report = {"skipped": True, "reason": "no_near_boundary_step2p",
                  "m0_threshold": m0_threshold}
        with open(out_dir / "gate3_report.json", "w") as f:
            json.dump(report, f, indent=2)
        return report

    # For each near-boundary step2+ decision, we need to reconstruct the
    # exact prompt state at that step and probe margins at different rho.
    # We do this by re-running the agent up to that step.
    sample_map = {s.id: s for s in samples}
    policy = BaselinePolicy()

    slope_results = []
    for dec in tqdm(near_boundary, desc="Gate3-slope"):
        sid = dec["sample_id"]
        target_step = dec["step_idx"]
        sample = sample_map.get(sid)
        if sample is None:
            continue

        # Re-run agent to collect messages at the target step
        # We use a special "probe" approach: run baseline, capture messages
        # at each step, then probe margin at the target step
        slopes = _probe_slope_at_step(
            agent, sample, target_step, rho_probe
        )
        if slopes is not None:
            slope_results.append({
                "sample_id": sid,
                "step_idx": target_step,
                "margin_baseline": dec["margin"],
                **slopes,
            })

    if not slope_results:
        print("  WARNING: No slope measurements obtained. Gate 3 inconclusive.")
        report = {"skipped": True, "reason": "no_slope_measurements"}
        with open(out_dir / "gate3_report.json", "w") as f:
            json.dump(report, f, indent=2)
        return report

    # Analyze slopes
    slopes_arr = np.array([r["slope"] for r in slope_results])
    abs_slopes = np.abs(slopes_arr)

    report = {
        "rho_probe": rho_probe,
        "m0_threshold": m0_threshold,
        "n_probed": len(slope_results),
        "slope_stats": {
            "mean": float(np.mean(slopes_arr)),
            "std": float(np.std(slopes_arr)),
            "median": float(np.median(slopes_arr)),
            "mean_abs": float(np.mean(abs_slopes)),
            "frac_positive": float(np.mean(slopes_arr > 0)),
            "frac_negative": float(np.mean(slopes_arr < 0)),
            "frac_abs_gt_0.05": float(np.mean(abs_slopes > 0.05)),
            "frac_abs_gt_0.1": float(np.mean(abs_slopes > 0.1)),
            "frac_abs_gt_0.5": float(np.mean(abs_slopes > 0.5)),
        },
        "sign_consistency": _sign_consistency(slopes_arr),
        "slope_results": slope_results,
    }

    print(f"\n  Probed: {len(slope_results)} step2+ decisions")
    print(f"  Slope mean: {np.mean(slopes_arr):.4f} ± {np.std(slopes_arr):.4f}")
    print(f"  Slope |mean|: {np.mean(abs_slopes):.4f}")
    print(f"  Sign consistency: {report['sign_consistency']:.1%}")
    print(f"  |slope| > 0.05: {np.mean(abs_slopes > 0.05):.1%}")
    print(f"  |slope| > 0.1: {np.mean(abs_slopes > 0.1):.1%}")
    print(f"  |slope| > 0.5: {np.mean(abs_slopes > 0.5):.1%}")

    # Gate pass/fail
    slope_effective = np.mean(abs_slopes > 0.05) >= 0.5  # >50% have meaningful slope
    sign_consistent = report["sign_consistency"] >= 0.7   # >70% same sign
    gate_pass = slope_effective and sign_consistent
    report["gate_pass"] = gate_pass
    report["slope_effective"] = slope_effective
    report["sign_consistent"] = sign_consistent

    if gate_pass:
        print(f"\n  ✅ GATE 3 PASS: Direction has stable slope on step2+")
    else:
        print(f"\n  ❌ GATE 3 FAIL: Direction slope is weak/unstable on step2+")
        if not slope_effective:
            print(f"    → Need step2+-specific direction extraction")
        if not sign_consistent:
            print(f"    → Sign inconsistency: direction may flip on step2+")

    with open(out_dir / "gate3_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report


def _probe_slope_at_step(agent, sample, target_step_idx, rho_probe):
    """Re-run agent to reconstruct prompt at target_step_idx, then probe margin.

    Returns dict with m_neg, m0, m_pos, slope, or None if step not reached.
    """
    from agent.prompts import PromptBuilder

    # Run baseline to reconstruct trajectory up to target step
    policy = BaselinePolicy()
    prompt_builder = agent.prompt_builder
    messages = prompt_builder.build_full_prompt(sample.question, [])
    steps_history = []

    for step_idx in range(agent.config.max_steps):
        if step_idx == target_step_idx:
            # We're at the target step — probe margins at different rho
            m0 = agent._compute_margin(messages, rho=0.0)
            m_pos = agent._compute_margin(messages, rho=rho_probe)
            m_neg = agent._compute_margin(messages, rho=-rho_probe)

            slope = (m_pos - m_neg) / (2 * rho_probe)
            return {
                "m_neg": float(m_neg),
                "m0": float(m0),
                "m_pos": float(m_pos),
                "slope": float(slope),
                "rho_probe": float(rho_probe),
            }

        # Generate step normally (no steering)
        def margin_fn(rho):
            return agent._compute_margin(messages, rho)

        steering_decision = policy.decide(
            margin_fn, "positive",
            agent._get_hidden_rms(),
            agent.direction_rms or 1.0
        )

        completion, _, _ = agent._generate_step(messages, steering_decision)

        from agent.prompts import parse_action
        parsed = parse_action(completion)

        if parsed["final_answer"]:
            # Agent finished before reaching target step
            return None

        if parsed["action"] and parsed["action"].lower() in agent.tools:
            tool_name = parsed["action"].lower()
            tool_input = parsed["action_input"] or ""
            try:
                observation = agent.tools[tool_name](tool_input)
                observation = str(observation)[:500]
            except Exception as e:
                observation = f"Error: {str(e)}"
        else:
            observation = "No valid action parsed"

        steps_history.append({
            "action": parsed["action"],
            "action_input": parsed["action_input"],
            "observation": observation,
        })
        messages = prompt_builder.build_full_prompt(
            sample.question, steps_history
        )

    return None  # Didn't reach target step


def _sign_consistency(slopes):
    """Fraction of slopes that share the majority sign."""
    if len(slopes) == 0:
        return 0.0
    n_pos = np.sum(slopes > 0)
    n_neg = np.sum(slopes < 0)
    return float(max(n_pos, n_neg) / len(slopes))



# ============================================================================
# Model / tokenizer loader
# ============================================================================

def load_model_and_tokenizer(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


# ============================================================================
# Final summary report
# ============================================================================

def generate_final_report(gate1_report, gate2_report, gate3_report, out_dir):
    """Compile a unified summary across all 3 gates."""
    summary = {
        "gate1": {
            "total_decisions": gate1_report.get("total_decisions", 0),
            "step1_count": gate1_report.get("step1_count", 0),
            "step2p_count": gate1_report.get("step2p_count", 0),
            "step1_margin_mean": gate1_report.get("step1_margins", {}).get("mean"),
            "step2p_margin_mean": gate1_report.get("step2p_margins", {}).get("mean"),
            "has_step2p_prey": gate1_report.get("step2p_count", 0) > 0,
        },
        "gate2": {},
        "gate3": {},
        "overall_verdict": "UNKNOWN",
    }

    # Gate 2 summary
    if gate2_report.get("skipped"):
        summary["gate2"] = {"skipped": True, "reason": gate2_report.get("reason")}
    else:
        summary["gate2"] = {
            "n_compared": gate2_report.get("n_compared", 0),
            "additional_tool_call_rate": gate2_report.get("additional_tool_call_rate"),
            "correctness_delta": gate2_report.get("correctness_delta"),
            "net_gain": gate2_report.get("net_gain"),
        }

    # Gate 3 summary
    if gate3_report.get("skipped"):
        summary["gate3"] = {"skipped": True, "reason": gate3_report.get("reason")}
    else:
        summary["gate3"] = {
            "n_probed": gate3_report.get("n_probed", 0),
            "mean_abs_slope": gate3_report.get("slope_stats", {}).get("mean_abs"),
            "sign_consistency": gate3_report.get("sign_consistency"),
            "gate_pass": gate3_report.get("gate_pass"),
        }

    # Overall verdict — use each gate's own pass/fail judgment
    g1_pass = gate1_report.get("gate_pass", False)
    g2_pass = (not gate2_report.get("skipped")) and gate2_report.get("net_gain", 0) >= 0
    g3_pass = gate3_report.get("gate_pass", False)

    if g1_pass and g2_pass and g3_pass:
        verdict = "ALL GATES PASS — proceed with Evidence-Seeking Controller"
    elif g1_pass and g3_pass:
        verdict = "GATE 1+3 PASS — direction works, but Gate 2 needs investigation"
    elif g1_pass:
        verdict = "GATE 1 PASS only — step2+ prey exists but direction/behavior needs work"
    else:
        verdict = "GATES FAIL — insufficient step2+ decisions or direction ineffective"

    summary["overall_verdict"] = verdict
    summary["gate_pass"] = {"gate1": g1_pass, "gate2": g2_pass, "gate3": g3_pass}

    print("\n" + "=" * 60)
    print("  FINAL VERDICT")
    print("=" * 60)
    print(f"  {verdict}")
    print(f"  Gate 1 (prey exists):   {'✅' if g1_pass else '❌'}")
    print(f"  Gate 2 (behavior Δ):    {'✅' if g2_pass else '❌'}")
    print(f"  Gate 3 (slope stable):  {'✅' if g3_pass else '❌'}")

    with open(out_dir / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary



# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evidence-Seeking Controller — Gatekeeping Experiments (Gates 1-3)")

    parser.add_argument("--data-path", required=True,
                        help="Path to popqa_test.jsonl")
    parser.add_argument("--corpus-path", default="data/popqa/corpus.jsonl",
                        help="Path to search corpus JSONL")
    parser.add_argument("--direction-path",
                        default="steering/directions/direction_search_v3.npz",
                        help="Direction vector NPZ file")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name or path")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of samples to evaluate")
    parser.add_argument("--max-steps", type=int, default=10,
                        help="Max ReAct steps per episode")
    parser.add_argument("--layer", type=int, default=12,
                        help="Hidden state intervention layer")
    parser.add_argument("--position", type=int, default=-1,
                        help="Token position for intervention (-1 = last)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--m0-threshold", type=float, default=1.5,
                        help="Near-boundary threshold |margin| < m0")
    parser.add_argument("--rho-probe", type=float, default=0.25,
                        help="Rho magnitude for slope probing (Gate 3)")
    parser.add_argument("--out", required=True,
                        help="Output directory for reports")
    parser.add_argument("--gates", default="1,2,3",
                        help="Comma-separated list of gates to run (default: 1,2,3)")

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gates_to_run = [int(g.strip()) for g in args.gates.split(",")]

    # Save args for reproducibility
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load dataset
    print("\n[1/4] Loading dataset...")
    dataset = PopQADataset(args.data_path)
    samples = dataset.get_subset(args.n_samples, seed=args.seed)
    print(f"  Selected {len(samples)} samples")

    # Build corpus if needed
    corpus_path = Path(args.corpus_path)
    if not corpus_path.exists():
        print(f"  Corpus not found at {corpus_path}, building...")
        build_popqa_corpus(args.data_path, str(corpus_path))

    # Load model + direction
    print("\n[2/4] Loading model and direction...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    direction, _ = load_direction(args.direction_path)
    direction_rms = float(np.sqrt(np.mean(direction ** 2)))
    print(f"  Direction: {args.direction_path}")
    print(f"  Direction RMS: {direction_rms:.6f}")

    # Build tools and agent
    print("\n[3/4] Creating agent...")
    search = SearchTool(str(corpus_path), top_k=3)
    tools = {"search": search, "calculator": CalculatorTool()}

    agent_config = AgentConfig(
        max_steps=args.max_steps,
        layer=args.layer,
        position=args.position,
    )
    agent = ReActAgent(
        model, tokenizer, tools, agent_config,
        direction=direction, direction_rms=direction_rms,
    )

    # Run gates
    print("\n[4/4] Running gatekeeping experiments...")
    t0 = time.time()

    gate1_decisions, gate1_report = None, {"step2p_count": 0}
    gate2_report = {"skipped": True, "reason": "not_run"}
    gate3_report = {"skipped": True, "reason": "not_run"}

    if 1 in gates_to_run:
        gate1_decisions, gate1_report = run_gate1(agent, samples, out_dir)

    if 2 in gates_to_run:
        if gate1_decisions is None:
            # Try to load from previous gate1 run
            g1_path = out_dir / "gate1_decisions.jsonl"
            if g1_path.exists():
                print("  Loading Gate 1 decisions from disk...")
                gate1_decisions = []
                with open(g1_path) as f:
                    for line in f:
                        gate1_decisions.append(json.loads(line))
            else:
                print("  ERROR: Gate 2 requires Gate 1 decisions. Run Gate 1 first.")
        if gate1_decisions:
            gate2_report = run_gate2(
                agent, samples, gate1_decisions, out_dir,
                m0_threshold=args.m0_threshold,
            )

    if 3 in gates_to_run:
        if gate1_decisions is None:
            g1_path = out_dir / "gate1_decisions.jsonl"
            if g1_path.exists():
                print("  Loading Gate 1 decisions from disk...")
                gate1_decisions = []
                with open(g1_path) as f:
                    for line in f:
                        gate1_decisions.append(json.loads(line))
            else:
                print("  ERROR: Gate 3 requires Gate 1 decisions. Run Gate 1 first.")
        if gate1_decisions:
            gate3_report = run_gate3(
                agent, samples, gate1_decisions, out_dir,
                m0_threshold=args.m0_threshold,
                rho_probe=args.rho_probe,
            )

    # Final summary
    final = generate_final_report(gate1_report, gate2_report, gate3_report, out_dir)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Results saved to: {out_dir}")


if __name__ == "__main__":
    main()