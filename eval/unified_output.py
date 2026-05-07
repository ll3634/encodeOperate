#!/usr/bin/env python3
"""
Unified output format for E2E agent evaluation.

Provides:
 - convert_episode_to_record(): EpisodeResult dict → extended JSONL record
 - compute_run_summary(): list[record] → per-run summary JSON
 - write_records() / load_records(): JSONL I/O helpers

All experiments MUST use this module for output to guarantee a single,
auditable, publication-grade format.
"""

import json
import hashlib
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_commit_hash() -> str:
    """Return short git commit hash or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, q))


# ---------------------------------------------------------------------------
# Per-sample record conversion
# ---------------------------------------------------------------------------

def convert_episode_to_record(
    episode: Dict[str, Any],
    *,
    run_id: str,
    dataset: str,
    split: str = "test",
    jes_params: Optional[Dict] = None,
    baseline_success: Optional[bool] = None,
    force_adopt_success: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Convert a raw EpisodeResult.to_dict() into the unified JSONL record.

    Parameters
    ----------
    episode : dict from EpisodeResult.to_dict()
    run_id  : unique identifier for this experiment run
    dataset : 'popqa', 'gsm8k', etc.
    split   : 'test' (default)
    jes_params : dict with {tau, rho_max, eps, layer, pos} when policy==jes
    baseline_success : used for computing flags (regressed / rescued)
    force_adopt_success : used for computing flags (tool_unnecessary)
    """
    steps = episode.get("steps", [])
    totals = episode.get("totals", {})
    policy_name = episode.get("policy", "unknown")
    is_correct = bool(episode.get("success", False))

    # -- tool_trace & decision_trace --
    tool_trace = []
    decision_trace = []
    for s in steps:
        if s.get("action") and s["action"].lower() != "finish":
            tool_trace.append({
                "step": s["step_idx"],
                "tool_name": s.get("action", ""),
                "tool_input": (s.get("action_input") or "")[:200],
                "tool_output_summary": (s.get("observation") or "")[:200],
                "adopted": True,
                "corruption_applied": s.get("corruption_applied", False),
            })
        steering = s.get("steering") or {}
        if steering or s.get("margin_before") is not None:
            rho = steering.get("rho_used", steering.get("rho", 0.0))
            alpha = steering.get("alpha_used", steering.get("alpha", 0.0))

            # Keep step indexing explicit:
            # - step_idx in EpisodeResult is 0-based (0..)
            # - JES tau schedules and guard_triggered_steps are naturally 1-based (1..)
            step0 = s["step_idx"]
            step1 = step0 + 1

            # Preserve additional steering details (when present) for auditability.
            # We keep this as a small, explicit allow-list to avoid bloating JSONL.
            _steering_allow = {
                "policy_name",
                "guard_triggered",
                "tau_effective",
                "eps_effective",
                "m_target",
                "m_before",
                "m_after",
                "slope",
                "rho_star_raw",
                "rho_used",
                "alpha_used",
                "alpha_clipped",
                "saturated",
                "already_satisfied",
                "step",
            }
            steering_details = {k: steering.get(k) for k in _steering_allow if k in steering}

            decision_trace.append({
                "step": step0,
                "step_1b": step1,
                "margin_before": s.get("margin_before"),
                "margin_after": s.get("margin_after") or steering.get("m_after"),
                "rho": rho,
                "alpha": alpha,
                "saturated": steering.get("saturated", False),
                "already_satisfied": steering.get("already_satisfied", False),
                "guard_triggered": bool(steering.get("guard_triggered", False)),
                "tau_effective": steering.get("tau_effective"),
                "m_target": steering.get("m_target"),
                "steering_details": steering_details or None,
            })

    # -- flags --
    flags: Dict[str, bool] = {}
    if baseline_success is not None:
        flags["regressed"] = baseline_success and not is_correct
        flags["rescued"] = (not baseline_success) and is_correct
    if force_adopt_success is not None:
        flags["tool_unnecessary"] = is_correct and (not force_adopt_success)
    flags["catastrophic"] = bool(
        any(t.get("corruption_applied", False) for t in tool_trace)
        and baseline_success and not is_correct
    )

    # -- tokens breakdown --
    tok_prompt = sum(s.get("tokens_prompt", 0) for s in steps)
    tok_compl = sum(s.get("tokens_completion", 0) for s in steps)

    return {
        "run_id": run_id,
        "dataset": dataset,
        "split": split,
        "sample_id": str(episode.get("id", "")),
        "question": (episode.get("question") or "")[:500],
        "gold_answer": episode.get("gold_answer"),
        "policy_name": policy_name,
        "jes_params": jes_params or {},
        "final_answer": episode.get("final_answer"),
        "is_correct": is_correct,
        "tool_calls": totals.get("tool_calls", 0),
        "tokens_prompt": tok_prompt,
        "tokens_completion": tok_compl,
        "tokens_total": totals.get("total_tokens", tok_prompt + tok_compl),
        "steps": len(steps),
        "wall_time_ms": totals.get("total_wall_time_ms", 0.0),
        "tool_trace": tool_trace,
        "decision_trace": decision_trace,
        "flags": flags,
        "failure_reason": episode.get("failure_reason"),
    }


# ---------------------------------------------------------------------------
# Per-run summary
# ---------------------------------------------------------------------------

def compute_run_summary(
    records: List[Dict[str, Any]],
    *,
    baseline_records: Optional[List[Dict[str, Any]]] = None,
    force_adopt_records: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Compute per-run summary from a list of unified records.

    If *baseline_records* and *force_adopt_records* are provided, subset
    breakdown (Stealth / RedFlag / Indifferent) is included.
    """
    if not records:
        return {}

    n = len(records)
    successes = [r["is_correct"] for r in records]
    tokens_total = [r["tokens_total"] for r in records]
    tool_calls = [r["tool_calls"] for r in records]
    n_steps = [r["steps"] for r in records]

    # Regression / rescue counts
    # Prefer computing directly vs baseline_records when available.
    # (Relying on per-record flags is only valid if convert_episode_to_record()
    # was called with baseline_success/force_adopt_success.
    n_paired_vs_baseline = 0
    if baseline_records:
        bl_ok_by_id = {r["sample_id"]: bool(r.get("is_correct", False)) for r in baseline_records}
        common_ids = [r["sample_id"] for r in records if r.get("sample_id") in bl_ok_by_id]
        n_paired_vs_baseline = len(common_ids)
        n_regressed = sum(
            1 for r in records
            if r.get("sample_id") in bl_ok_by_id
            and bl_ok_by_id[r["sample_id"]]
            and (not bool(r.get("is_correct", False)))
        )
        n_rescued = sum(
            1 for r in records
            if r.get("sample_id") in bl_ok_by_id
            and (not bl_ok_by_id[r["sample_id"]])
            and bool(r.get("is_correct", False))
        )
    else:
        n_regressed = sum(1 for r in records if r.get("flags", {}).get("regressed", False))
        n_rescued = sum(1 for r in records if r.get("flags", {}).get("rescued", False))

    # Saturation (JES only)
    n_saturated = 0
    n_steered = 0
    for r in records:
        for d in r.get("decision_trace", []):
            if d.get("rho", 0) != 0:
                n_steered += 1
                if d.get("saturated", False):
                    n_saturated += 1

    summary: Dict[str, Any] = {
        "dataset": records[0].get("dataset", ""),
        "n": n,
        "policy_name": records[0].get("policy_name", ""),
        "run_id": records[0].get("run_id", ""),
        "git_commit": _git_commit_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success_rate": sum(successes) / n,
        "n_success": sum(successes),
        "avg_tool_calls": float(np.mean(tool_calls)),
        "avg_steps": float(np.mean(n_steps)),
        "avg_tokens_total": float(np.mean(tokens_total)),
        "p50_tokens": _percentile(tokens_total, 50),
        "p90_tokens": _percentile(tokens_total, 90),
        "p95_tokens": _percentile(tokens_total, 95),
        "n_paired_vs_baseline": n_paired_vs_baseline,
        "regression_rate": n_regressed / (n_paired_vs_baseline or n) if n else 0,
        "rescue_rate": n_rescued / (n_paired_vs_baseline or n) if n else 0,
        "net_gain": n_rescued - n_regressed,
        "saturation_rate": n_saturated / n_steered if n_steered else 0.0,
    }

    # Subset breakdown (if baseline + force_adopt provided)
    if baseline_records and force_adopt_records:
        bl_by_id = {r["sample_id"]: r["is_correct"] for r in baseline_records}
        fa_by_id = {r["sample_id"]: r["is_correct"] for r in force_adopt_records}
        policy_by_id = {r["sample_id"]: r for r in records}

        subsets = {"stealth": [], "red_flag": [], "indifferent": []}
        for sid in bl_by_id:
            bl_ok = bl_by_id.get(sid, False)
            fa_ok = fa_by_id.get(sid, False)
            if not bl_ok and fa_ok:
                subsets["stealth"].append(sid)
            elif bl_ok and not fa_ok:
                subsets["red_flag"].append(sid)
            else:
                subsets["indifferent"].append(sid)

        for sname, sids in subsets.items():
            s_records = [policy_by_id[sid] for sid in sids if sid in policy_by_id]
            s_n = len(s_records)
            s_correct = sum(r["is_correct"] for r in s_records)
            # Compute regressions/rescues directly vs baseline for the subset.
            s_regressed = sum(
                1 for sid in sids
                if sid in policy_by_id
                and bl_by_id.get(sid, False)
                and (not bool(policy_by_id[sid].get("is_correct", False)))
            )
            s_rescued = sum(
                1 for sid in sids
                if sid in policy_by_id
                and (not bl_by_id.get(sid, False))
                and bool(policy_by_id[sid].get("is_correct", False))
            )
            summary[sname] = {
                "count": s_n,
                "success": s_correct / s_n if s_n else 0,
                "rescues": s_rescued,
                "regressions": s_regressed,
            }

    return summary


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_records(records: List[Dict], path: str) -> None:
    """Write records to JSONL file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def load_records(path: str) -> List[Dict]:
    """Load unified records from JSONL."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_summary(summary: Dict, path: str) -> None:
    """Write summary JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)


def make_run_id(dataset: str, policy: str, n: int, seed: int = 42) -> str:
    """Generate a deterministic run_id."""
    raw = f"{dataset}_{policy}_n{n}_s{seed}_{_git_commit_hash()}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]

