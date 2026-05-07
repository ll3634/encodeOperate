#!/usr/bin/env python3
"""
Control Budget Diagnosis.

Reads baseline, oracle, and JES results from a verify-critical pipeline run
and produces a diagnostic report on whether the dataset is a viable target
for decision-only JES steering.

Key outputs:
  1. verify-critical density (n_vc / n_total)
  2. rho* distribution for verify-critical samples at step 1
  3. Flip-feasibility table: for each max_rho budget, how many VC samples
     have |rho*| <= budget?
  4. Go / No-Go verdict based on density >= 5% and feasible flip rate.

Usage:
    python scripts/control_budget_diagnosis.py \
        --results-dir results/verify_critical_v4
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_jsonl(path: Path) -> List[Dict]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def extract_step1_jes_data(jes_result: Dict) -> Optional[Dict]:
    """Extract JES steering data from step 1 (the decision point)."""
    for step in jes_result.get("steps", []):
        steering = step.get("steering", {})
        if steering.get("action") == "jes_steering" and steering.get("step", -1) == 1:
            return {
                "sample_id": jes_result["sample_id"],
                "m_before": steering.get("m_before", step.get("margin_before")),
                "m_after": steering.get("m_after"),
                "m_target": steering.get("m_target"),
                "slope": steering.get("slope"),
                "rho_star_raw": steering.get("rho_star_raw"),
                "rho_used": steering.get("rho_used"),
                "achieved": steering.get("achieved", False),
                "clipped": steering.get("clipped", False),
                "saturated": steering.get("saturated", False),
                "unstable": steering.get("unstable", False),
                "eps_effective": steering.get("eps_effective"),
            }
    return None


def compute_diagnosis(
    baseline_results: List[Dict],
    oracle_results: List[Dict],
    jes_results: List[Dict],
    rho_budgets: List[float] = None,
) -> Dict[str, Any]:
    """Compute the full control budget diagnosis."""
    if rho_budgets is None:
        rho_budgets = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

    bl_by_id = {r["sample_id"]: r for r in baseline_results}
    orc_by_id = {r["sample_id"]: r for r in oracle_results}
    jes_by_id = {r["sample_id"]: r for r in jes_results}

    n_total = len(baseline_results)

    # Classify samples
    vc_ids, vh_ids, ind_ids = set(), set(), set()
    for sid in bl_by_id:
        bl_ok = bl_by_id[sid]["is_correct"]
        orc_ok = orc_by_id.get(sid, {}).get("is_correct", False)
        if not bl_ok and orc_ok:
            vc_ids.add(sid)
        elif bl_ok and not orc_ok:
            vh_ids.add(sid)
        else:
            ind_ids.add(sid)

    vc_density = len(vc_ids) / n_total if n_total > 0 else 0.0

    # Extract rho* for VC samples
    vc_jes_data = []
    for sid in vc_ids:
        if sid in jes_by_id:
            data = extract_step1_jes_data(jes_by_id[sid])
            if data:
                vc_jes_data.append(data)

    # Extract rho* for ALL samples (for full distribution)
    all_jes_data = []
    for sid in jes_by_id:
        data = extract_step1_jes_data(jes_by_id[sid])
        if data:
            all_jes_data.append(data)

    # rho* distribution for VC samples
    vc_rho_stars = []
    for d in vc_jes_data:
        rho_raw = d.get("rho_star_raw")
        if rho_raw is not None and np.isfinite(rho_raw):
            vc_rho_stars.append(abs(rho_raw))

    # rho* distribution for ALL samples
    all_rho_stars = []
    for d in all_jes_data:
        rho_raw = d.get("rho_star_raw")
        if rho_raw is not None and np.isfinite(rho_raw):
            all_rho_stars.append(abs(rho_raw))

    # Flip feasibility table
    flip_table = []
    for budget in rho_budgets:
        feasible_vc = sum(1 for r in vc_rho_stars if r <= budget)
        feasible_all = sum(1 for r in all_rho_stars if r <= budget)
        flip_table.append({
            "max_rho": budget,
            "vc_feasible": feasible_vc,
            "vc_total": len(vc_rho_stars),
            "vc_rate": feasible_vc / len(vc_rho_stars) if vc_rho_stars else 0.0,
            "all_feasible": feasible_all,
            "all_total": len(all_rho_stars),
            "all_rate": feasible_all / len(all_rho_stars) if all_rho_stars else 0.0,
        })

    # Margin distribution at step 1 (all samples)
    all_margins = [d["m_before"] for d in all_jes_data if d.get("m_before") is not None]
    vc_margins = [d["m_before"] for d in vc_jes_data if d.get("m_before") is not None]

    # Slope distribution
    all_slopes = [abs(d["slope"]) for d in all_jes_data
                  if d.get("slope") is not None and np.isfinite(d["slope"])]

    # Unstable fraction
    n_unstable_vc = sum(1 for d in vc_jes_data if d.get("unstable", False))
    n_unstable_all = sum(1 for d in all_jes_data if d.get("unstable", False))

    # JES rescue stats on VC
    vc_jes_correct = sum(1 for sid in vc_ids
                         if jes_by_id.get(sid, {}).get("is_correct", False))

    # Go/No-Go
    go_density = vc_density >= 0.05
    go_feasible = (len(vc_rho_stars) > 0 and
                   sum(1 for r in vc_rho_stars if r <= 1.5) / len(vc_rho_stars) >= 0.3)
    go_rescue = vc_jes_correct > 0
    verdict = "GO" if (go_density and (go_feasible or go_rescue)) else "NO-GO"

    return {
        "n_total": n_total,
        "n_vc": len(vc_ids), "n_vh": len(vh_ids), "n_ind": len(ind_ids),
        "vc_density": vc_density,
        "vc_jes_correct": vc_jes_correct,
        "vc_rho_star_distribution": {
            "n": len(vc_rho_stars),
            "mean": float(np.mean(vc_rho_stars)) if vc_rho_stars else None,
            "median": float(np.median(vc_rho_stars)) if vc_rho_stars else None,
            "p25": float(np.percentile(vc_rho_stars, 25)) if vc_rho_stars else None,
            "p75": float(np.percentile(vc_rho_stars, 75)) if vc_rho_stars else None,
            "p90": float(np.percentile(vc_rho_stars, 90)) if vc_rho_stars else None,
            "min": float(np.min(vc_rho_stars)) if vc_rho_stars else None,
            "max": float(np.max(vc_rho_stars)) if vc_rho_stars else None,
            "values": [float(r) for r in sorted(vc_rho_stars)],
        },
        "all_rho_star_distribution": {
            "n": len(all_rho_stars),
            "mean": float(np.mean(all_rho_stars)) if all_rho_stars else None,
            "median": float(np.median(all_rho_stars)) if all_rho_stars else None,
            "p25": float(np.percentile(all_rho_stars, 25)) if all_rho_stars else None,
            "p75": float(np.percentile(all_rho_stars, 75)) if all_rho_stars else None,
        },
        "vc_margin_step1": {
            "n": len(vc_margins),
            "mean": float(np.mean(vc_margins)) if vc_margins else None,
            "min": float(np.min(vc_margins)) if vc_margins else None,
            "max": float(np.max(vc_margins)) if vc_margins else None,
        },
        "all_margin_step1": {
            "n": len(all_margins),
            "mean": float(np.mean(all_margins)) if all_margins else None,
            "std": float(np.std(all_margins)) if all_margins else None,
        },
        "slope_distribution": {
            "n": len(all_slopes),
            "mean": float(np.mean(all_slopes)) if all_slopes else None,
            "median": float(np.median(all_slopes)) if all_slopes else None,
        },
        "unstable": {"vc": n_unstable_vc, "all": n_unstable_all},
        "flip_feasibility": flip_table,
        "verdict": verdict,
        "verdict_details": {
            "density_ok": go_density,
            "feasibility_ok": go_feasible,
            "rescue_observed": go_rescue,
        },
        "per_vc_sample": vc_jes_data,
    }


def print_diagnosis(diag: Dict):
    """Pretty-print the diagnosis."""
    print("=" * 70)
    print("  CONTROL BUDGET DIAGNOSIS")
    print("=" * 70)

    print(f"\n[1] Sample Distribution (n={diag['n_total']})")
    print(f"    verify_critical:  {diag['n_vc']}  ({diag['vc_density']*100:.1f}%)")
    print(f"    verify_harmful:   {diag['n_vh']}")
    print(f"    indifferent:      {diag['n_ind']}")

    rd = diag["vc_rho_star_distribution"]
    print(f"\n[2] rho* Distribution (verify-critical, n={rd['n']})")
    if rd["n"] > 0:
        print(f"    mean={rd['mean']:.3f}  median={rd['median']:.3f}")
        print(f"    p25={rd['p25']:.3f}  p75={rd['p75']:.3f}  p90={rd['p90']:.3f}")
        print(f"    min={rd['min']:.3f}  max={rd['max']:.3f}")
        print(f"    values: {rd['values']}")
    else:
        print("    (no finite rho* values)")

    md = diag["vc_margin_step1"]
    print(f"\n[3] Step-1 Margin (verify-critical, n={md['n']})")
    if md["n"] > 0:
        print(f"    mean={md['mean']:.3f}  min={md['min']:.3f}  max={md['max']:.3f}")

    print(f"\n[4] Flip-Feasibility Table (verify-critical)")
    print(f"    {'max_rho':>8} | {'feasible':>8} / {'total':>5} | {'rate':>6} || {'all_feas':>8} / {'all_tot':>7} | {'all_rate':>8}")
    print(f"    {'-'*8}-+-{'-'*8}---{'-'*5}-+-{'-'*6}-++-{'-'*8}---{'-'*7}-+-{'-'*8}")
    for row in diag["flip_feasibility"]:
        print(f"    {row['max_rho']:8.2f} | {row['vc_feasible']:8d} / {row['vc_total']:5d} | "
              f"{row['vc_rate']*100:5.1f}% || {row['all_feasible']:8d} / {row['all_total']:7d} | "
              f"{row['all_rate']*100:7.1f}%")

    print(f"\n[5] JES Rescue on VC: {diag['vc_jes_correct']} / {diag['n_vc']}")
    print(f"    Unstable slopes: VC={diag['unstable']['vc']}, All={diag['unstable']['all']}")

    v = diag["verdict_details"]
    print(f"\n[6] VERDICT: *** {diag['verdict']} ***")
    print(f"    density >= 5%:           {'YES' if v['density_ok'] else 'NO'}")
    print(f"    feasible flips (rho<=1.5 >= 30%): {'YES' if v['feasibility_ok'] else 'NO'}")
    print(f"    any rescue observed:     {'YES' if v['rescue_observed'] else 'NO'}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Control Budget Diagnosis")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory with baseline/oracle/jes results JSONL")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path (default: <results-dir>/diagnosis.json)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    bl = load_jsonl(results_dir / "baseline_results.jsonl")
    orc = load_jsonl(results_dir / "oracle_results.jsonl")
    jes = load_jsonl(results_dir / "jes_results.jsonl")

    print(f"Loaded: {len(bl)} baseline, {len(orc)} oracle, {len(jes)} JES results")

    diag = compute_diagnosis(bl, orc, jes)
    print_diagnosis(diag)

    out_path = Path(args.out) if args.out else results_dir / "diagnosis.json"
    with open(out_path, "w") as f:
        json.dump(diag, f, indent=2, ensure_ascii=False)
    print(f"\nDiagnosis saved to: {out_path}")


if __name__ == "__main__":
    main()
