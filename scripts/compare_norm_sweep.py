#!/usr/bin/env python3
"""
Compare results from the normalized sweep experiment across directions.

Supports multi-random baseline: reads random_seed0..N-1 results and
computes distribution statistics (mean, std, 95%CI) for comparison.

Usage:
    python scripts/compare_norm_sweep.py <results_root> [--n-random 10]
"""

import json
import os
import sys
import argparse
import numpy as np


def load_report(root, label):
    """Load report.json for a given label, return None if missing."""
    path = os.path.join(root, label, "report.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def extract_metrics(rpt):
    """Extract key metrics from a report dict."""
    fs = rpt["full_stats"]
    vs = rpt.get("vc_stats", {})
    return {
        "net": fs.get("net_gain", 0),
        "corr": fs.get("net_gain_corrected", 0),
        "rescued": fs.get("rescued", 0),
        "genuine": fs.get("rescued_genuine", 0),
        "regressed": fs.get("regressed", 0),
        "parse_fail": fs.get("parse_failures", 0),
        "vc_rescued": vs.get("rescued", 0),
        "vc_genuine": vs.get("rescued_genuine", 0),
        "vc_n": vs.get("n", 0),
        "delta_acc": fs.get("policy_rate", 0) - fs.get("baseline_rate", 0),
    }


def print_direction_row(label, rpt, hdr_len):
    """Print one direction's summary + sweep rows."""
    fs = rpt["full_stats"]
    vs = rpt.get("vc_stats", {})
    cfg = rpt.get("config", {})

    mr = cfg.get("max_rho", "?")
    m = extract_metrics(rpt)
    print(
        f"{label:<16} | {str(mr):>8} | {m['net']:>+5d} | {m['corr']:>+5d} "
        f"| {m['rescued']:>7d} | {m['genuine']:>7d} | {m['regressed']:>7d} "
        f"| {m['parse_fail']:>5d} | {m['vc_rescued']:>3d}/{m['vc_n']:<3d}"
    )

    sweep = rpt.get("sweep_summary", {})
    for tag in sorted(sweep.keys()):
        s = sweep[tag]
        print(
            f"  {'':>14} | {s['max_rho']:>8.2f} | {s['net_gain']:>+5d} | {'':>5} "
            f"| {s['rescued']:>7d} | {'':>7} | {s['regressed']:>7d} "
            f"| {'':>5} | {s.get('vc_rescued', 0):>3d}/{'':>3}"
        )
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", help="Root directory of sweep results")
    parser.add_argument("--n-random", type=int, default=0,
                        help="Number of random directions (random_seed0..N-1)")
    args = parser.parse_args()

    root = args.results_root
    targeted_labels = ["search_post", "v12_post"]

    # Header
    hdr = (
        f"{'Direction':<16} | {'max_rho':>8} | {'Net':>5} | {'Corr.':>5} "
        f"| {'Rescued':>7} | {'Genuine':>7} | {'Regress':>7} "
        f"| {'PFail':>5} | {'VC_Resc':>8}"
    )
    hdr_len = len(hdr)

    # ---- Section 1: Targeted directions ----
    print("=" * hdr_len)
    print("  TARGETED DIRECTIONS (full rho sweep)")
    print("=" * hdr_len)
    print(hdr)
    print("-" * hdr_len)

    for label in targeted_labels:
        rpt = load_report(root, label)
        if rpt is None:
            print(f"{label:<16} | ** MISSING **")
            continue
        print_direction_row(label, rpt, hdr_len)

    # ---- Section 2: Random baseline distribution ----
    if args.n_random > 0:
        print("=" * hdr_len)
        print(f"  RANDOM BASELINE DISTRIBUTION ({args.n_random} directions, single rho)")
        print("=" * hdr_len)
        print(hdr)
        print("-" * hdr_len)

        random_metrics = []
        for i in range(args.n_random):
            label = f"random_seed{i}"
            rpt = load_report(root, label)
            if rpt is None:
                print(f"{label:<16} | ** MISSING **")
                continue
            m = extract_metrics(rpt)
            random_metrics.append({"label": label, "rpt": rpt, **m})
            cfg = rpt.get("config", {})
            mr = cfg.get("max_rho", "?")
            print(
                f"{label:<16} | {str(mr):>8} | {m['net']:>+5d} | {m['corr']:>+5d} "
                f"| {m['rescued']:>7d} | {m['genuine']:>7d} | {m['regressed']:>7d} "
                f"| {m['parse_fail']:>5d} | {m['vc_rescued']:>3d}/{m['vc_n']:<3d}"
            )
            # Per-rho sweep details for each random
            sweep = rpt.get("sweep_summary", {})
            for tag in sorted(sweep.keys()):
                s = sweep[tag]
                print(
                    f"  {'':>14} | {s['max_rho']:>8.2f} | {s['net_gain']:>+5d} | {'':>5} "
                    f"| {s['rescued']:>7d} | {'':>7} | {s['regressed']:>7d} "
                    f"| {'':>5} | {s.get('vc_rescued', 0):>3d}/{'':>3}"
                )

        if random_metrics:
            print()
            print("-" * hdr_len)
            print("  RANDOM DISTRIBUTION SUMMARY:")
            for key in ["net", "corr", "rescued", "genuine", "regressed",
                         "parse_fail", "vc_rescued", "delta_acc"]:
                vals = [m[key] for m in random_metrics]
                arr = np.array(vals, dtype=float)
                mean = np.mean(arr)
                std = np.std(arr, ddof=1) if len(arr) > 1 else 0
                ci_lo = np.percentile(arr, 2.5)
                ci_hi = np.percentile(arr, 97.5)
                fmt = ".1%" if key == "delta_acc" else ".1f"
                print(
                    f"    {key:<12}: mean={mean:{fmt}}  std={std:{fmt}}  "
                    f"95%CI=[{ci_lo:{fmt}}, {ci_hi:{fmt}}]  "
                    f"range=[{np.min(arr):{fmt}}, {np.max(arr):{fmt}}]"
                )

            # ---- Section 3: Verdict ----
            print()
            print("=" * hdr_len)
            print("  VERDICT: Direction vs Random")
            print("=" * hdr_len)
            corr_vals = np.array([m["corr"] for m in random_metrics], dtype=float)
            rand_mean = np.mean(corr_vals)
            rand_std = np.std(corr_vals, ddof=1) if len(corr_vals) > 1 else 0
            rand_ci_hi = np.percentile(corr_vals, 97.5)

            for label in targeted_labels:
                rpt = load_report(root, label)
                if rpt is None:
                    continue
                m = extract_metrics(rpt)
                # Use rho=0.5 result for fair comparison with random
                sweep = rpt.get("sweep_summary", {})
                rho05_net = None
                for tag, s in sweep.items():
                    if abs(s.get("max_rho", 0) - 0.5) < 0.01:
                        rho05_net = s.get("net_gain", None)
                        break
                dir_corr = m["corr"]
                z = (dir_corr - rand_mean) / rand_std if rand_std > 0 else float('inf')
                above = "YES" if dir_corr > rand_ci_hi else "NO"
                print(
                    f"  {label:<16}: corr_net={dir_corr:+d}  "
                    f"random_mean={rand_mean:.1f}  z={z:.2f}  "
                    f"above_95%CI({rand_ci_hi:.1f})? {above}"
                )

    # ---- Fairness check ----
    print()
    print("=" * hdr_len)
    print("FAIRNESS CHECK (all directions should show similar loaded_rms):")
    all_labels = targeted_labels + [f"random_seed{i}" for i in range(args.n_random)]
    for label in all_labels:
        rpt = load_report(root, label)
        if rpt is None:
            continue
        cfg = rpt.get("config", {})
        orig_rms = cfg.get("direction_original_rms", None)
        loaded_rms = cfg.get("direction_loaded_rms", None)
        norm_target = cfg.get("normalize_rms", None)
        if orig_rms is not None:
            print(
                f"  {label:<16} | orig_rms={orig_rms:.6f} "
                f"-> loaded_rms={loaded_rms:.6f} (target={norm_target})"
            )


if __name__ == "__main__":
    main()

