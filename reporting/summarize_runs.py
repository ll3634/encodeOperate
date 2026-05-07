#!/usr/bin/env python3
"""
Summarize existing JSONL result files into the unified format.

Reads *old-format* per-sample JSONL produced by the existing runners,
converts each sample to the unified record, then generates the unified
per-run summary JSON.

Usage:
  python -m reporting.summarize_runs \
      --baseline results/popqa_500/baseline_500.jsonl \
      --force-adopt results/popqa_500/force_adopt_500.jsonl \
      --force-reject results/popqa_500/force_reject_500.jsonl \
      --jes results/popqa_500/jes_500.jsonl \
      --dataset popqa --out results/popqa_500_unified
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.unified_output import (
    convert_episode_to_record, compute_run_summary,
    write_records, write_summary, make_run_id,
)
from eval.subset_labeling import label_subsets, compute_subset_metrics, save_subset_report
from eval.paired_stats import full_paired_report


def _load_old_jsonl(path: str) -> List[Dict]:
    """Load old-format JSONL from runners."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def convert_old_to_unified(
    old_records: List[Dict],
    *,
    dataset: str,
    policy_name: str,
    run_id: str,
    jes_params: Optional[Dict] = None,
    baseline_by_id: Optional[Dict] = None,
    force_adopt_by_id: Optional[Dict] = None,
) -> List[Dict]:
    """Convert a list of old-format records to unified records."""
    unified = []
    for ep in old_records:
        sid = str(ep.get("id", ""))
        bl_ok = baseline_by_id.get(sid, {}).get("success") if baseline_by_id else None
        fa_ok = force_adopt_by_id.get(sid, {}).get("success") if force_adopt_by_id else None
        rec = convert_episode_to_record(
            ep,
            run_id=run_id,
            dataset=dataset,
            jes_params=jes_params,
            baseline_success=bl_ok,
            force_adopt_success=fa_ok,
        )
        unified.append(rec)
    return unified


def main():
    parser = argparse.ArgumentParser(description="Convert old JSONL to unified format + generate summaries")
    parser.add_argument("--baseline", required=True, help="Baseline JSONL")
    parser.add_argument("--force-adopt", required=True, help="Force-adopt JSONL")
    parser.add_argument("--force-reject", required=True, help="Force-reject JSONL")
    parser.add_argument("--jes", required=True, help="JES JSONL")
    parser.add_argument("--dataset", default="popqa", help="Dataset name")
    parser.add_argument("--jes-params", default=None, help="JES params JSON string")
    parser.add_argument("--out", required=True, help="Output directory for unified results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jes_params = json.loads(args.jes_params) if args.jes_params else {}

    # Load old-format data
    print("Loading old-format JSONL files ...")
    old = {
        "baseline": _load_old_jsonl(args.baseline),
        "force_adopt": _load_old_jsonl(args.force_adopt),
        "force_reject": _load_old_jsonl(args.force_reject),
        "jes": _load_old_jsonl(args.jes),
    }
    for pname, recs in old.items():
        print(f"  {pname}: {len(recs)} samples")

    # Build ID lookups for cross-policy flags
    bl_by_id = {str(r["id"]): r for r in old["baseline"]}
    fa_by_id = {str(r["id"]): r for r in old["force_adopt"]}

    # Convert each policy
    all_unified: Dict[str, List[Dict]] = {}
    summaries: Dict[str, Dict] = {}

    for pname, recs in old.items():
        run_id = make_run_id(args.dataset, pname, len(recs), args.seed)
        jp = jes_params if pname == "jes" else None
        unified = convert_old_to_unified(
            recs, dataset=args.dataset, policy_name=pname,
            run_id=run_id, jes_params=jp,
            baseline_by_id=bl_by_id, force_adopt_by_id=fa_by_id,
        )
        all_unified[pname] = unified

        # Write unified JSONL
        write_records(unified, str(out / f"{pname}.jsonl"))

        # Compute + write summary
        bl_recs = all_unified.get("baseline")
        fa_recs = all_unified.get("force_adopt")
        summ = compute_run_summary(
            unified,
            baseline_records=bl_recs if pname != "baseline" else None,
            force_adopt_records=fa_recs if pname != "baseline" else None,
        )
        summaries[pname] = summ
        write_summary(summ, str(out / f"{pname}_summary.json"))
        print(f"  {pname}: success={summ.get('success_rate', 0):.1%}  "
              f"regress={summ.get('regression_rate', 0):.1%}  "
              f"rescue={summ.get('rescue_rate', 0):.1%}")

    # Subset labeling
    print("\nLabeling subsets ...")
    subsets = label_subsets(all_unified["baseline"], all_unified["force_adopt"])
    for sname, sids in subsets.items():
        print(f"  {sname}: {len(sids)}")

    sub_metrics = compute_subset_metrics(subsets, all_unified)
    save_subset_report(subsets, sub_metrics, str(out / "subset_report.json"))

    # Paired stats (jes vs baseline)
    print("\nPaired statistics (JES vs Baseline) ...")
    report = full_paired_report(
        all_unified["baseline"], all_unified["jes"],
        policy_name="jes",
        indifferent_ids=subsets["indifferent"],
    )
    write_summary(report, str(out / "paired_stats_jes.json"))
    mcn = report["mcnemar"]
    bci = report["bootstrap_success_diff"]
    print(f"  McNemar p={mcn['mcnemar_p']:.4f}  "
          f"(b={mcn['b_regressed']}, c={mcn['c_rescued']})")
    print(f"  Bootstrap ΔSuccess: {bci['observed']:+.4f}  "
          f"95% CI [{bci['ci_lower']:+.4f}, {bci['ci_upper']:+.4f}]")

    print(f"\nAll unified outputs written to {out}/")


if __name__ == "__main__":
    main()

