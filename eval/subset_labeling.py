#!/usr/bin/env python3
"""
Counterfactual subset labeling for E2E evaluation.

Three subsets defined purely by *counterfactual* outcomes:
  - Stealth  : baseline FAILS  AND  force_adopt SUCCEEDS
               → tool genuinely helps; JES should *recover* these.
  - Red Flag : baseline SUCCEEDS  AND  force_adopt FAILS
               → tool genuinely hurts; JES should *protect* these.
  - Indifferent : all other cases
               → tool doesn't flip the outcome either way.

These definitions are deterministic given the (baseline, force_adopt) pair.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set


# ---------------------------------------------------------------------------
# Core labeling
# ---------------------------------------------------------------------------

def label_subsets(
    baseline_records: List[Dict],
    force_adopt_records: List[Dict],
) -> Dict[str, List[str]]:
    """
    Classify sample IDs into Stealth / RedFlag / Indifferent.

    Parameters
    ----------
    baseline_records : unified JSONL records (policy='baseline')
    force_adopt_records : unified JSONL records (policy='force_adopt')

    Returns
    -------
    dict with keys 'stealth', 'red_flag', 'indifferent' → list of sample_id
    """
    bl_by_id = {r["sample_id"]: r["is_correct"] for r in baseline_records}
    fa_by_id = {r["sample_id"]: r["is_correct"] for r in force_adopt_records}

    common = sorted(set(bl_by_id) & set(fa_by_id))

    stealth: List[str] = []
    red_flag: List[str] = []
    indifferent: List[str] = []

    for sid in common:
        bl_ok = bl_by_id[sid]
        fa_ok = fa_by_id[sid]
        if (not bl_ok) and fa_ok:
            stealth.append(sid)
        elif bl_ok and (not fa_ok):
            red_flag.append(sid)
        else:
            indifferent.append(sid)

    return {
        "stealth": stealth,
        "red_flag": red_flag,
        "indifferent": indifferent,
    }


def label_subsets_from_jsonl(
    baseline_path: str,
    force_adopt_path: str,
) -> Dict[str, List[str]]:
    """Convenience: load raw JSONL (old format) and classify."""
    def _load(path):
        recs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    recs.append({
                        "sample_id": str(d.get("id", d.get("sample_id", ""))),
                        "is_correct": d.get("success", d.get("is_correct", False)),
                    })
        return recs
    return label_subsets(_load(baseline_path), _load(force_adopt_path))


# ---------------------------------------------------------------------------
# Subset metrics (per-policy, per-subset)
# ---------------------------------------------------------------------------

def compute_subset_metrics(
    subsets: Dict[str, List[str]],
    policy_records_map: Dict[str, List[Dict]],
) -> Dict[str, Dict[str, Dict]]:
    """
    For each (subset, policy) pair compute success_rate, rescues, regressions, cost.

    Parameters
    ----------
    subsets : output of label_subsets()
    policy_records_map : {policy_name: list_of_unified_records}

    Returns
    -------
    Nested dict: subset_name → policy_name → metrics dict
    """
    result = {}
    for sname, sids in subsets.items():
        sid_set = set(sids)
        result[sname] = {}
        for pname, records in policy_records_map.items():
            recs = [r for r in records if r["sample_id"] in sid_set]
            if not recs:
                continue
            n = len(recs)
            n_correct = sum(r["is_correct"] for r in recs)
            n_regressed = sum(1 for r in recs if r.get("flags", {}).get("regressed", False))
            n_rescued = sum(1 for r in recs if r.get("flags", {}).get("rescued", False))
            avg_tokens = sum(r.get("tokens_total", 0) for r in recs) / n
            avg_tools = sum(r.get("tool_calls", 0) for r in recs) / n
            result[sname][pname] = {
                "count": n,
                "success_rate": n_correct / n,
                "success_count": n_correct,
                "rescues": n_rescued,
                "regressions": n_regressed,
                "avg_tokens": round(avg_tokens, 1),
                "avg_tool_calls": round(avg_tools, 2),
            }
    return result


def save_subset_report(
    subsets: Dict[str, List[str]],
    subset_metrics: Dict,
    output_path: str,
) -> None:
    """Save subset analysis to JSON file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "subset_sizes": {k: len(v) for k, v in subsets.items()},
        "sample_ids": subsets,
        "metrics": subset_metrics,
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

