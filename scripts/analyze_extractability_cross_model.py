#!/usr/bin/env python3
"""Per-model summary + paired stats for cross-model extractability replication.

Writes summary_{tag}.json for each --eval input, and a combined cross-model JSON.
"""
import argparse, json, math, statistics as st
from collections import defaultdict
from pathlib import Path

try:
    from scipy.stats import binomtest, wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    if HAVE_SCIPY:
        return binomtest(k, n, 0.5, alternative="two-sided").pvalue
    p = sum(math.comb(n, i) * 0.5 ** n for i in range(k + 1))
    return min(1.0, 2 * p)


def _fa(r):
    """First-action token (CLAUDE.md §4.9 #8 fix). Falls back to action_type
    for legacy rows that pre-date the field."""
    return r.get("first_action_token") or (
        "search" if r.get("action_type") == "search" else
        "stop"   if r.get("action_type") == "stop" else "parse_fail")


def cell(rows):
    n = max(1, len(rows))
    n_stop = sum(1 for r in rows if _fa(r) == "stop") or 1
    return {
        "n": len(rows),
        "first_search_rate": sum(1 for r in rows if _fa(r) == "search") / n,
        "first_stop_rate":   sum(1 for r in rows if _fa(r) == "stop")   / n,
        "first_parse_fail":  sum(1 for r in rows if _fa(r) == "parse_fail") / n,
        "commit_W":               sum(1 for r in rows if r.get("contains_W")) / n,
        "commit_W_among_stops":   sum(1 for r in rows if _fa(r) == "stop" and r.get("contains_W")) / n_stop,
        "em":          sum(1 for r in rows if r.get("em") == 1) / n,
        "parse_fail":  sum(1 for r in rows if r.get("parse_failure")) / n,
        "mean_ml":     st.fmean(r["margin_label"]       for r in rows),
        "mean_mft":    st.fmean(r["margin_first_token"] for r in rows),
        "mean_margin_post": (st.fmean([r["margin_post"] for r in rows if r.get("margin_post") is not None])
                              if any(r.get("margin_post") is not None for r in rows) else None),
    }


def paired(A, B, key_int):
    """Paired McNemar on a binary attribute and Wilcoxon on margin_label."""
    ka = {r["sample_id"]: int(bool(r.get(key_int))) for r in A}
    kb = {r["sample_id"]: int(bool(r.get(key_int))) for r in B}
    ids = sorted(set(ka) & set(kb))
    a, b = [ka[i] for i in ids], [kb[i] for i in ids]
    b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    p_mc = mcnemar_exact(b10, b01)

    da = {r["sample_id"]: r["margin_label"] for r in A}
    db = {r["sample_id"]: r["margin_label"] for r in B}
    fa, fb = [da[i] for i in ids], [db[i] for i in ids]
    delta_ml = (sum(fa) - sum(fb)) / max(1, len(fa))
    try:
        p_w = wilcoxon(fa, fb).pvalue if HAVE_SCIPY else None
    except ValueError:
        p_w = 1.0
    return {
        "n_pairs": len(ids),
        "delta_rate": (sum(a) - sum(b)) / max(1, len(a)),
        "mcnemar_b10": b10, "mcnemar_b01": b01, "mcnemar_p": p_mc,
        "delta_margin_label": delta_ml,
        "wilcoxon_p_margin": p_w,
    }


def per_schema(rows, conds):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows: by[r["schema_type"]][r["condition"]].append(r)
    out = {}
    for sch in sorted(by):
        out[sch] = {}
        for c in conds:
            xs = by[sch][c]
            n = max(1, len(xs))
            out[sch][c] = {
                "n": len(xs),
                "first_search": sum(1 for r in xs if _fa(r) == "search") / n,
                "first_stop":   sum(1 for r in xs if _fa(r) == "stop")   / n,
                "commit_W": sum(int(r.get("contains_W", 0)) for r in xs) / n,
                "em":       sum(1 for r in xs if r.get("em") == 1) / n,
            }
    return out


def summarise(rows, tag, conds=("N0", "T0", "S0"), include_T1=False):
    if include_T1: conds = tuple(list(conds) + ["T1"])
    by = defaultdict(list)
    for r in rows: by[r["condition"]].append(r)

    cells = {c: cell(by[c]) for c in conds if by[c]}
    contrasts = {}
    if "T0" in by and "N0" in by:
        contrasts["T0_vs_N0"] = {
            "commit_W":    paired(by["T0"], by["N0"], "contains_W"),
            "first_is_search":   paired(
                [{**r, "_search": int(_fa(r) == "search")} for r in by["T0"]],
                [{**r, "_search": int(_fa(r) == "search")} for r in by["N0"]],
                "_search",
            ),
            "first_is_stop":   paired(
                [{**r, "_stop": int(_fa(r) == "stop")} for r in by["T0"]],
                [{**r, "_stop": int(_fa(r) == "stop")} for r in by["N0"]],
                "_stop",
            ),
        }
    if "S0" in by and "T0" in by:
        contrasts["S0_vs_T0"] = {
            "commit_W":    paired(by["S0"], by["T0"], "contains_W"),
            "em":          paired(by["S0"], by["T0"], "em"),
        }
    return {
        "tag": tag,
        "n_records": len(rows),
        "cells": cells,
        "contrasts": contrasts,
        "per_schema": per_schema(rows, conds),
    }


def qual_failures(rows, max_per_cell=3):
    out = {}
    for cond in ("N0", "T0", "S0", "T1"):
        sub = [r for r in rows if r["condition"] == cond]
        if not sub: continue
        out[cond] = {
            "parse_failures":  [r["raw_output"] for r in sub if r.get("parse_failure")][:max_per_cell],
            "commit_W_examples": [r["raw_output"] for r in sub if r.get("contains_W")][:max_per_cell],
            "first_search_examples": [r["raw_output"] for r in sub if _fa(r) == "search"][:max_per_cell],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", nargs="+", required=True,
                    help="path:tag pairs, e.g. results/.../eval_results_qwen.jsonl:qwen2_5_7b")
    ap.add_argument("--out-dir", default="results/cross_model_extractability")
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    combined = {"models": {}, "scipy_available": HAVE_SCIPY}
    for spec in args.eval:
        path, tag = spec.rsplit(":", 1)
        rows = [json.loads(l) for l in open(path)]
        s = summarise(rows, tag)
        s["qualitative"] = qual_failures(rows)
        with open(out_dir / f"summary_{tag}.json", "w") as f:
            json.dump(s, f, indent=2)
        combined["models"][tag] = {k: v for k, v in s.items() if k != "qualitative"}

    with open(out_dir / "summary_cross_model.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
