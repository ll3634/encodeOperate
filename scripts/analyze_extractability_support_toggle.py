#!/usr/bin/env python3
"""Analyse Extractability-Support-Missingness toggle eval.

Produces summary.json + report.md with:
  - per-cell rates (search, stop, commit-W, EM, parse-failure)
  - McNemar on commit-W for T0 vs N0, S0 vs T0, T1 vs T0
  - Wilcoxon signed-rank + paired-permutation on margin_label for the same pairs
  - bootstrap 95% CIs for rate differences and margin shifts
  - per-schema breakdown
"""
import argparse, json, math, random, statistics as st
from collections import defaultdict
from pathlib import Path

try:
    from scipy.stats import wilcoxon, binomtest
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    if HAVE_SCIPY:
        return binomtest(k, n, 0.5, alternative="two-sided").pvalue
    p = 0.0
    for i in range(k + 1):
        p += math.comb(n, i) * 0.5 ** n
    return min(1.0, 2 * p)


def paired_perm(xs, ys, n_iter=10000, seed=0):
    rng = random.Random(seed)
    d = [a - b for a, b in zip(xs, ys)]
    obs = sum(d) / len(d)
    cnt = 0
    for _ in range(n_iter):
        s = sum((x if rng.random() < 0.5 else -x) for x in d) / len(d)
        if abs(s) >= abs(obs) - 1e-12: cnt += 1
    return obs, (cnt + 1) / (n_iter + 1)


def bootstrap_ci(xs, n_iter=5000, seed=1, alpha=0.05):
    rng = random.Random(seed); n = len(xs); out = []
    for _ in range(n_iter):
        out.append(sum(xs[rng.randrange(n)] for _ in range(n)) / n)
    out.sort()
    return out[int(alpha/2 * n_iter)], out[int((1-alpha/2) * n_iter)]


def diff_ci(xs, ys, n_iter=5000, seed=2, alpha=0.05):
    rng = random.Random(seed); n = len(xs); out = []
    for _ in range(n_iter):
        idx = [rng.randrange(n) for _ in range(n)]
        out.append(sum(xs[i] for i in idx)/n - sum(ys[i] for i in idx)/n)
    out.sort()
    return out[int(alpha/2 * n_iter)], out[int((1-alpha/2) * n_iter)]


def pair_binary(a_rows, b_rows, key):
    ka = {r["sample_id"]: int(bool(r.get(key))) for r in a_rows}
    kb = {r["sample_id"]: int(bool(r.get(key))) for r in b_rows}
    ids = sorted(set(ka) & set(kb))
    return ids, [ka[i] for i in ids], [kb[i] for i in ids]


def pair_float(a_rows, b_rows, key):
    ka = {r["sample_id"]: r[key] for r in a_rows}
    kb = {r["sample_id"]: r[key] for r in b_rows}
    ids = sorted(set(ka) & set(kb))
    return ids, [ka[i] for i in ids], [kb[i] for i in ids]


def cell_stats(rows):
    n = len(rows)
    return {
        "n": n,
        "search_rate":  sum(1 for r in rows if r["action_type"] == "search") / max(1, n),
        "stop_rate":    sum(1 for r in rows if r["action_type"] == "stop") / max(1, n),
        "commit_W":     sum(1 for r in rows if r.get("contains_W")) / max(1, n),
        "em":           sum(1 for r in rows if r.get("em") == 1) / max(1, n),
        "parse_fail":   sum(1 for r in rows if r.get("parse_failure")) / max(1, n),
        "mean_ml":      st.fmean(r["margin_label"]        for r in rows),
        "mean_mft":     st.fmean(r["margin_first_token"]  for r in rows),
    }


def paired_tests(A, B, label_a, label_b):
    _, ya, yb = pair_binary(A, B, "contains_W")
    b = sum(1 for x, y in zip(ya, yb) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(ya, yb) if x == 0 and y == 1)
    p_mc = mcnemar_exact(b, c)
    _, sa, sb = pair_binary(A, B, "action_type")   # 1=search, 0=stop
    sa = [1 if r["action_type"] == "search" else 0 for r in A]
    sb = [1 if r["action_type"] == "search" else 0 for r in B]
    _, fa, fb = pair_float(A, B, "margin_label")
    obs_diff, p_perm = paired_perm(fa, fb)
    try:
        w_stat = wilcoxon(fa, fb).pvalue if HAVE_SCIPY else None
    except ValueError:
        w_stat = 1.0
    lo, hi = diff_ci(fa, fb)
    return {
        "pair": f"{label_a} vs {label_b}",
        "n_pairs": len(fa),
        "commitW_delta": (sum(ya)-sum(yb))/max(1,len(ya)),
        "commitW_mcnemar_b": b, "commitW_mcnemar_c": c, "commitW_p": p_mc,
        "margin_label_delta": obs_diff,
        "margin_label_ci95":  [lo, hi],
        "margin_label_perm_p": p_perm,
        "margin_label_wilcoxon_p": w_stat,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="results/extractability_support_toggle/eval_results.jsonl")
    ap.add_argument("--out-dir", default="results/extractability_support_toggle")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.eval)]
    by_cond = defaultdict(list)
    for r in rows: by_cond[r["condition"]].append(r)

    cells = {c: cell_stats(by_cond[c]) for c in ("N0", "T0", "T1", "S0")}
    pairs = [
        paired_tests(by_cond["T0"], by_cond["N0"], "T0", "N0"),   # extractability effect
        paired_tests(by_cond["T1"], by_cond["T0"], "T1", "T0"),   # missingness cue
        paired_tests(by_cond["S0"], by_cond["T0"], "S0", "T0"),   # support-on-top-of-extractability
    ]

    # Per-schema commit-W per condition.
    by_sch = defaultdict(lambda: defaultdict(list))
    for r in rows: by_sch[r["schema_type"]][r["condition"]].append(r)
    per_schema = {sch: {c: {"n": len(by_sch[sch][c]),
                            "commit_W": sum(int(r.get("contains_W", 0)) for r in by_sch[sch][c]) / max(1, len(by_sch[sch][c])),
                            "search":   sum(1 for r in by_sch[sch][c] if r["action_type"] == "search") / max(1, len(by_sch[sch][c])),
                            "em":       sum(1 for r in by_sch[sch][c] if r.get("em") == 1) / max(1, len(by_sch[sch][c]))}
                        for c in ("N0","T0","T1","S0")} for sch in sorted(by_sch)}

    summary = {"cells": cells, "paired_tests": pairs, "per_schema": per_schema,
               "n_total": len(rows), "scipy_available": HAVE_SCIPY}
    out_dir = Path(args.out_dir)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
