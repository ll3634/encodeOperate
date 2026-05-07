#!/usr/bin/env python3
"""
Counterfactual subset labeler with stealth subdivisions.

Input:  baseline.jsonl + force_adopt.jsonl (+ optional force_reject.jsonl)
Output: manifest.jsonl — one row per sample with label + subdivision.

Labels:
  tool_critical  : baseline FAILS  AND force_adopt SUCCEEDS
  tool_harmful   : baseline SUCCEEDS AND force_adopt FAILS
  indifferent    : all other

Stealth subdivisions (within tool_critical):
  stealth_choice : baseline made 0 tool calls (chose not to use tool)
  stealth_query  : baseline called tool but fewer times than force_adopt
  stealth_format : baseline called tool same/more times — tool output handling differs

Usage:
  python scripts/label_tool_sensitivity.py \
      --baseline results/popqa/baseline.jsonl \
      --force-adopt results/popqa/force_adopt.jsonl \
      --force-reject results/popqa/force_reject.jsonl \
      --out results/popqa/manifest.jsonl
"""

import json, argparse
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.unified_output import load_records


def label_samples(
    baseline_recs: List[Dict],
    force_adopt_recs: List[Dict],
    force_reject_recs: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Label each sample with tool sensitivity + stealth subdivision."""
    bl_by_id = {r["sample_id"]: r for r in baseline_recs}
    fa_by_id = {r["sample_id"]: r for r in force_adopt_recs}
    fr_by_id = {r["sample_id"]: r for r in (force_reject_recs or [])}
    common = sorted(set(bl_by_id) & set(fa_by_id))

    manifest = []
    for sid in common:
        bl = bl_by_id[sid]
        fa = fa_by_id[sid]
        bl_ok = bl["is_correct"]
        fa_ok = fa["is_correct"]
        bl_tc = bl.get("tool_calls", 0)
        fa_tc = fa.get("tool_calls", 0)

        # Primary label
        if (not bl_ok) and fa_ok:
            label = "tool_critical"
        elif bl_ok and (not fa_ok):
            label = "tool_harmful"
        else:
            label = "indifferent"

        # Stealth subdivision (only for tool_critical)
        subdivision = None
        if label == "tool_critical":
            if bl_tc == 0:
                subdivision = "stealth_choice"
            elif bl_tc < fa_tc:
                subdivision = "stealth_query"
            else:
                subdivision = "stealth_format"

        row = {
            "sample_id": sid,
            "label": label,
            "subdivision": subdivision,
            "baseline_correct": bl_ok,
            "force_adopt_correct": fa_ok,
            "baseline_tool_calls": bl_tc,
            "force_adopt_tool_calls": fa_tc,
            "baseline_tokens": bl.get("tokens_total", 0),
            "force_adopt_tokens": fa.get("tokens_total", 0),
        }
        if fr_by_id and sid in fr_by_id:
            fr = fr_by_id[sid]
            row["force_reject_correct"] = fr["is_correct"]
            row["force_reject_tool_calls"] = fr.get("tool_calls", 0)

        manifest.append(row)
    return manifest


def print_summary(manifest: List[Dict]):
    """Print summary counts."""
    from collections import Counter
    labels = Counter(r["label"] for r in manifest)
    subdivs = Counter(r["subdivision"] for r in manifest if r["subdivision"])
    n = len(manifest)
    print(f"\n{'='*50}  Subset Labeling Summary  {'='*50}")
    print(f"Total samples: {n}")
    for lbl in ["tool_critical", "tool_harmful", "indifferent"]:
        cnt = labels.get(lbl, 0)
        print(f"  {lbl:20s}: {cnt:4d}  ({cnt/n*100:.1f}%)")
    if subdivs:
        print(f"\n  tool_critical subdivisions:")
        for sub in ["stealth_choice", "stealth_query", "stealth_format"]:
            cnt = subdivs.get(sub, 0)
            print(f"    {sub:20s}: {cnt:4d}")


def main():
    parser = argparse.ArgumentParser(description="Counterfactual subset labeler")
    parser.add_argument("--baseline", required=True, help="baseline.jsonl path")
    parser.add_argument("--force-adopt", required=True, help="force_adopt.jsonl path")
    parser.add_argument("--force-reject", default=None, help="force_reject.jsonl (optional)")
    parser.add_argument("--out", required=True, help="Output manifest.jsonl")
    args = parser.parse_args()

    bl_recs = load_records(args.baseline)
    fa_recs = load_records(args.force_adopt)
    fr_recs = load_records(args.force_reject) if args.force_reject else None

    manifest = label_samples(bl_recs, fa_recs, fr_recs)
    print_summary(manifest)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nManifest written to {out}  ({len(manifest)} rows)")


if __name__ == "__main__":
    main()

