#!/usr/bin/env python3
"""
One-command report generation from unified result directories.

Reads existing unified JSONL files and generates:
  - Summary JSONs (per-policy)
  - Subset labeling + report
  - Paired statistics (McNemar + bootstrap CI)
  - All publication figures (FigA, Table1, Pareto)

Usage:
  # From existing unified outputs:
  python scripts/report_all.py --results-dir results/popqa_500_unified

  # From old-format outputs (convert first):
  python -m reporting.summarize_runs \
      --baseline results/popqa_500/baseline_500.jsonl \
      --force-adopt results/popqa_500/force_adopt_500.jsonl \
      --force-reject results/popqa_500/force_reject_500.jsonl \
      --jes results/popqa_500/jes_500.jsonl \
      --out results/popqa_500_unified
  python scripts/report_all.py --results-dir results/popqa_500_unified
"""

import json
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.unified_output import load_records, compute_run_summary, write_summary
from eval.subset_labeling import label_subsets, compute_subset_metrics, save_subset_report
from eval.paired_stats import full_paired_report
from reporting.make_figures import fig_a_subset_bars, table1_main_results, fig_c_pareto


def main():
    parser = argparse.ArgumentParser(description="Generate all reports from unified results")
    parser.add_argument("--results-dir", required=True, help="Dir with unified JSONL files")
    parser.add_argument("--fig-dir", default=None, help="Figure output dir (default: results-dir/figures)")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rdir = Path(args.results_dir)
    fig_dir = Path(args.fig_dir) if args.fig_dir else rdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load all policy JSONL files ----
    all_records = {}
    for jpath in sorted(rdir.glob("*.jsonl")):
        pname = jpath.stem  # e.g., "baseline", "force_adopt", etc.
        # Skip files that look like sweep cells (p0.1_baseline.jsonl)
        if pname.startswith("p") and "_" in pname:
            continue
        recs = load_records(str(jpath))
        if recs:
            all_records[pname] = recs
            print(f"  Loaded {pname}: {len(recs)} records")

    if not all_records:
        print(f"No JSONL files found in {rdir}")
        return

    # ---- Summaries ----
    print("\n--- Summaries ---")
    summaries = {}
    for pname, recs in all_records.items():
        bl_recs = all_records.get("baseline") if pname != "baseline" else None
        fa_recs = all_records.get("force_adopt") if pname != "baseline" else None
        summ = compute_run_summary(recs, baseline_records=bl_recs,
                                   force_adopt_records=fa_recs)
        summaries[pname] = summ
        write_summary(summ, str(rdir / f"{pname}_summary.json"))
        print(f"  {pname}: success={summ['success_rate']:.1%}  n={summ['n']}")

    # ---- Subset labeling ----
    subsets = None
    if "baseline" in all_records and "force_adopt" in all_records:
        print("\n--- Subset labeling ---")
        subsets = label_subsets(all_records["baseline"], all_records["force_adopt"])
        for sname, sids in subsets.items():
            print(f"  {sname}: {len(sids)}")

        sub_metrics = compute_subset_metrics(subsets, all_records)
        save_subset_report(subsets, sub_metrics, str(rdir / "subset_report.json"))

    # ---- Paired statistics ----
    if "baseline" in all_records:
        print("\n--- Paired statistics ---")
        for pname in ["force_adopt", "force_reject", "jes"]:
            if pname not in all_records:
                continue
            ind_ids = subsets["indifferent"] if subsets else None
            report = full_paired_report(
                all_records["baseline"], all_records[pname],
                policy_name=pname, indifferent_ids=ind_ids,
                n_bootstrap=args.n_bootstrap, seed=args.seed,
            )
            write_summary(report, str(rdir / f"paired_stats_{pname}.json"))
            mcn = report["mcnemar"]
            bci = report["bootstrap_success_diff"]
            dn = report["do_no_harm"]
            print(f"  {pname} vs baseline:")
            print(f"    McNemar p={mcn['mcnemar_p']:.4f}")
            print(f"    ΔSuccess: {bci['observed']:+.4f}  "
                  f"95% CI [{bci['ci_lower']:+.4f}, {bci['ci_upper']:+.4f}]")
            print(f"    Do-no-harm: regress={dn['regression_rate']:.1%}  "
                  f"rescue={dn['rescue_rate']:.1%}  net={dn['net_gain']}")

    # ---- Figures ----
    print("\n--- Figures ---")
    table1_main_results(summaries, str(fig_dir / "table1"))
    fig_c_pareto(summaries, str(fig_dir / "fig_c_pareto.png"))

    if subsets:
        subset_report_path = rdir / "subset_report.json"
        if subset_report_path.exists():
            with open(subset_report_path) as f:
                sr = json.load(f)
            fig_a_subset_bars(sr.get("metrics", {}), str(fig_dir / "fig_a_subsets.png"))

    print(f"\nAll reports written to {rdir}/")
    print(f"All figures written to {fig_dir}/")


if __name__ == "__main__":
    main()

