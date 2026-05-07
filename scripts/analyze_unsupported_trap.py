#!/usr/bin/env python3
"""Analysis for unsupported-answer-trap benchmark.

Reports per-condition commitment patterns:
  - search / stop / pf
  - commits_W  (contains the extractable wrong candidate W)
  - commits_V  (contains the gold V; V is absent from Trap-B0/B1 by construction)
  - hedge      (explicit uncertainty phrasing)

Paired contrasts:
  Trap-B1 vs Trap-B0 : effect of task_missingness cue under a High-L wrong-candidate trap
  True-D0 vs Trap-B0 : baseline commitment swap when the true bridge + V is provided

McNemar is computed for search_rate and commits_W.
"""
import argparse, json, re
from collections import defaultdict
from math import comb
from pathlib import Path
from statistics import mean, median


HEDGE_PATTERNS = (
    "not specified", "unknown", "cannot be determined", "not mentioned",
    "does not specify", "cannot be definitively", "not explicitly",
    "insufficient", "not provided",
)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) * 2
    return min(tail / (2 ** n), 1.0)


def _annotate(rows, pairs):
    for r in rows:
        p = pairs.get((r["sample_id"], r["condition_id"]), {})
        V = (p.get("V") or "").lower()
        W = (p.get("W") or "").lower()
        ans = (r.get("final_answer") or "").lower()
        r["V"] = p.get("V")
        r["W"] = p.get("W")
        r["schema"] = p.get("schema")
        r["commits_V"] = bool(V) and V in ans
        r["commits_W"] = bool(W) and W in ans
        r["hedge"] = any(pat in ans for pat in HEDGE_PATTERNS)


def per_condition(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["condition_id"]].append(r)
    out = {}
    for k, v in sorted(by.items()):
        n = len(v)
        out[k] = {
            "n": n,
            "search_rate": sum(1 for x in v if x["action_type"] == "search") / n,
            "stop_rate":   sum(1 for x in v if x["action_type"] == "stop") / n,
            "pf_rate":     sum(1 for x in v if x["parse_failure"]) / n,
            "commits_W_rate": sum(1 for x in v if x["commits_W"]) / n,
            "commits_V_rate": sum(1 for x in v if x["commits_V"]) / n,
            "hedge_rate":     sum(1 for x in v if x["hedge"]) / n,
            "em_rate":     sum(1 for x in v if x.get("em") == 1) / n,
            "mean_margin": mean(x["margin"] for x in v),
            "median_margin": median(x["margin"] for x in v),
        }
    return out


def paired_contrast(rows, a_cond, b_cond):
    by_sid = defaultdict(dict)
    for r in rows:
        by_sid[r["sample_id"]][r["condition_id"]] = r
    pairs = [(by_sid[sid][a_cond], by_sid[sid][b_cond])
             for sid in by_sid if a_cond in by_sid[sid] and b_cond in by_sid[sid]]
    if not pairs:
        return {}
    # McNemar on search-rate (0/1).
    a_s = [1 if p[0]["action_type"] == "search" else 0 for p in pairs]
    b_s = [1 if p[1]["action_type"] == "search" else 0 for p in pairs]
    b_to_a = sum(1 for ai, bi in zip(a_s, b_s) if ai == 1 and bi == 0)
    a_to_b = sum(1 for ai, bi in zip(a_s, b_s) if ai == 0 and bi == 1)
    p_search = mcnemar(b_to_a, a_to_b)
    # McNemar on commits_W.
    a_w = [1 if p[0]["commits_W"] else 0 for p in pairs]
    b_w = [1 if p[1]["commits_W"] else 0 for p in pairs]
    bw_to_aw = sum(1 for ai, bi in zip(a_w, b_w) if ai == 1 and bi == 0)
    aw_to_bw = sum(1 for ai, bi in zip(a_w, b_w) if ai == 0 and bi == 1)
    p_w = mcnemar(bw_to_aw, aw_to_bw)
    d_margin = [p[0]["margin"] - p[1]["margin"] for p in pairs]
    return {
        "n_pairs": len(pairs),
        f"{a_cond}_search_rate": sum(a_s) / len(a_s),
        f"{b_cond}_search_rate": sum(b_s) / len(b_s),
        "delta_search": (sum(a_s) - sum(b_s)) / len(a_s),
        "mcnemar_search_discordant": {a_cond: b_to_a, b_cond: a_to_b},
        "mcnemar_search_p": p_search,
        f"{a_cond}_commits_W_rate": sum(a_w) / len(a_w),
        f"{b_cond}_commits_W_rate": sum(b_w) / len(b_w),
        "delta_commits_W": (sum(a_w) - sum(b_w)) / len(a_w),
        "mcnemar_commits_W_discordant": {a_cond: bw_to_aw, b_cond: aw_to_bw},
        "mcnemar_commits_W_p": p_w,
        "mean_margin_delta": mean(d_margin),
        "median_margin_delta": median(d_margin),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="results/unsupported_trap/eval_results.jsonl")
    ap.add_argument("--pairs", default="results/unsupported_trap/pairs.jsonl")
    ap.add_argument("--out", default="results/unsupported_trap/summary.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.eval)]
    pairs = {(p["sample_id"], p["condition_id"]): p
             for p in (json.loads(l) for l in open(args.pairs))}
    _annotate(rows, pairs)

    per_cond = per_condition(rows)
    contrasts = {
        "Trap-B1_vs_Trap-B0": paired_contrast(rows, "Trap-B1", "Trap-B0"),
        "True-D0_vs_Trap-B0": paired_contrast(rows, "True-D0", "Trap-B0"),
        "True-D0_vs_Trap-B1": paired_contrast(rows, "True-D0", "Trap-B1"),
    }
    per_schema = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per_schema[r.get("schema") or "?"][r["condition_id"]].append(r)
    per_schema_out = {
        sch: {
            cond: {
                "n": len(v),
                "commits_W_rate": sum(1 for x in v if x["commits_W"]) / max(1, len(v)),
                "commits_V_rate": sum(1 for x in v if x["commits_V"]) / max(1, len(v)),
                "search_rate":    sum(1 for x in v if x["action_type"] == "search") / max(1, len(v)),
                "mean_margin":    mean(x["margin"] for x in v) if v else None,
            } for cond, v in d.items()
        } for sch, d in per_schema.items()
    }

    summary = {
        "n_total_records": len(rows),
        "per_condition": per_cond,
        "contrasts": contrasts,
        "per_schema": per_schema_out,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
