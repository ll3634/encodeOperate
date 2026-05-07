#!/usr/bin/env python3
"""Verify that patched cross_model_full.py reruns reproduce the pre-patch
aggregate metrics to 3 significant figures (relative diff < 5e-3).

OQ3: Qwen is skipped (no pre-patch baseline file exists).
"""
import json
import math
import os
import sys

MODELS = ["mistral_v2", "gemma2_v2", "llama31_v2", "r1distill_v2"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

ANCHORS = [
    ("paired_corruption", "AB_ratio_action"),
    ("paired_corruption", "MW_action_p"),
    ("evidence_probe", "auroc_mean"),
]
TOL_REL = 5e-3
TOL_LOG10 = 0.005  # for very small numbers (<1e-3) compare on log10


def relcmp(a, b):
    """Return (passed, diff_metric, metric_kind)."""
    if a == 0 and b == 0:
        return True, 0.0, "abs"
    if a == 0 or b == 0:
        return abs(a - b) < TOL_REL, abs(a - b), "abs"
    # Use log10 distance for very small p-values to avoid float-noise inflation.
    if abs(a) < 1e-3 or abs(b) < 1e-3:
        d = abs(math.log10(abs(a)) - math.log10(abs(b)))
        return d < TOL_LOG10, d, "log10"
    rd = abs(a - b) / max(abs(a), abs(b))
    return rd < TOL_REL, rd, "rel"


def fmt_diff(d, kind):
    if kind == "log10":
        return f"{d:.2e} (log10)"
    return f"{d:.2e}"


def main():
    n_fail = 0
    rows = []
    for m in MODELS:
        d_old = os.path.join(RESULTS, f"cross_model_{m}", "full_results_pre_patch.json")
        d_new = os.path.join(RESULTS, f"cross_model_{m}", "full_results.json")
        if not (os.path.isfile(d_old) and os.path.isfile(d_new)):
            print(f"FAIL  {m}: missing baseline or new file ({d_old} / {d_new})")
            n_fail += 1
            continue
        old = json.load(open(d_old))
        new = json.load(open(d_new))

        for parent, key in ANCHORS:
            a = old[parent][key]
            b = new[parent][key]
            ok, d, kind = relcmp(a, b)
            status = "OK  " if ok else "FAIL"
            rows.append((m, f"{parent}.{key}", a, b, d, kind, ok))
            if not ok:
                n_fail += 1

        # 4th anchor: peak-layer AUROC from layer_sweep (peak_evidence_layer)
        peak = str(new["peak_evidence_layer"])
        a = old["layer_sweep"][peak]["auroc"]
        b = new["layer_sweep"][peak]["auroc"]
        ok, d, kind = relcmp(a, b)
        status = "OK  " if ok else "FAIL"
        rows.append((m, f"layer_sweep[L{peak}].auroc", a, b, d, kind, ok))
        if not ok:
            n_fail += 1

    # Print table.
    print()
    print(f"{'model':<14} {'anchor':<32} {'old':>16} {'new':>16} {'diff':>14} {'status':>6}")
    print("-" * 102)
    for m, anchor, a, b, d, kind, ok in rows:
        a_s = f"{a:.6g}"
        b_s = f"{b:.6g}"
        d_s = fmt_diff(d, kind)
        status = "OK" if ok else "FAIL"
        print(f"{m:<14} {anchor:<32} {a_s:>16} {b_s:>16} {d_s:>14} {status:>6}")
    print()
    print(f"Total checks: {len(rows)}  Failures: {n_fail}")
    if n_fail == 0:
        print("Verification gate passed.")
        sys.exit(0)
    else:
        print("Verification gate FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
