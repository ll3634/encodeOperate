#!/usr/bin/env python3
"""Analysis for controlled L/G/M benchmark.

Computes per-condition rates (search / stop / pf / em), paired
margin statistics across conditions, and McNemar p-values for
search rate on the three key contrasts:
  - B1 vs B0  : effect of missingness cue (L and G held Low)
  - C0 vs B0  : effect of global sufficiency (L held Low)
  - D0 vs C0  : effect of direct answer sentence (G held High)
"""
import argparse, json, math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def mcnemar(b, c):
    """Two-sided exact McNemar p-value for b discordant pairs (1->0) and c (0->1)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X <= k) + P(X >= n-k) under Binomial(n, 0.5), two-sided.
    from math import comb
    tail = sum(comb(n, i) for i in range(k + 1)) * 2
    p = tail / (2 ** n)
    return min(p, 1.0)


def load_results(eval_path, pairs_path=None):
    rows = [json.loads(l) for l in open(eval_path)]
    if pairs_path:
        pairs = {(p["sample_id"], p["condition_id"]): p
                 for p in (json.loads(l) for l in open(pairs_path))}
        for r in rows:
            key = (r["sample_id"], r["condition_id"])
            if key in pairs:
                r["schema"] = pairs[key].get("schema")
                r["gold_answer"] = pairs[key].get("gold_answer")
    return rows


def per_condition(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["condition_id"]].append(r)
    out = {}
    for k, v in sorted(by.items()):
        margins = [x["margin"] for x in v]
        action_types = [x["action_type"] for x in v]
        pf = sum(1 for x in v if x["parse_failure"])
        em = sum(1 for x in v if x.get("em") == 1)
        n_s = sum(1 for a in action_types if a == "search")
        n_st = sum(1 for a in action_types if a == "stop")
        out[k] = {
            "n": len(v),
            "search_rate": n_s / len(v),
            "stop_rate": n_st / len(v),
            "pf_rate": pf / len(v),
            "em_rate": em / len(v),
            "mean_margin": mean(margins),
            "median_margin": median(margins),
            "n_search": n_s, "n_stop": n_st, "n_pf": pf, "n_em": em,
        }
    return out


def paired_contrast(rows, a_cond, b_cond, attr="action_type"):
    """Paired contrast a vs b on the `attr` field. Returns McNemar for
    search-rate (0/1) and mean margin delta."""
    by_sid = defaultdict(dict)
    for r in rows:
        by_sid[r["sample_id"]][r["condition_id"]] = r
    pairs = [(by_sid[sid][a_cond], by_sid[sid][b_cond])
             for sid in by_sid
             if a_cond in by_sid[sid] and b_cond in by_sid[sid]]
    if not pairs:
        return {}
    # Search=1, not-search=0.
    a_s = [1 if p[0]["action_type"] == "search" else 0 for p in pairs]
    b_s = [1 if p[1]["action_type"] == "search" else 0 for p in pairs]
    b_to_a = sum(1 for ai, bi in zip(a_s, b_s) if ai == 1 and bi == 0)  # b->a flipped to search
    a_to_b = sum(1 for ai, bi in zip(a_s, b_s) if ai == 0 and bi == 1)
    p = mcnemar(b_to_a, a_to_b)
    d_margin = [p[0]["margin"] - p[1]["margin"] for p in pairs]
    return {
        "n_pairs": len(pairs),
        f"{a_cond}_search_rate": sum(a_s) / len(a_s),
        f"{b_cond}_search_rate": sum(b_s) / len(b_s),
        "delta_search": (sum(a_s) - sum(b_s)) / len(a_s),
        "mcnemar_discordant_" + a_cond: b_to_a,
        "mcnemar_discordant_" + b_cond: a_to_b,
        "mcnemar_p": p,
        "mean_margin_delta": mean(d_margin),
        "median_margin_delta": median(d_margin),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="results/controlled_lgm/eval_results.jsonl")
    ap.add_argument("--pairs", default="results/controlled_lgm/pairs.jsonl")
    ap.add_argument("--out", default="results/controlled_lgm/summary.json")
    args = ap.parse_args()

    rows = load_results(args.eval, args.pairs)
    per_cond = per_condition(rows)
    contrasts = {
        "B1_vs_B0": paired_contrast(rows, "B1", "B0"),
        "C0_vs_B0": paired_contrast(rows, "C0", "B0"),
        "D0_vs_C0": paired_contrast(rows, "D0", "C0"),
        "D0_vs_B0": paired_contrast(rows, "D0", "B0"),
    }
    # Per-schema breakdown of search rate.
    per_schema = defaultdict(lambda: defaultdict(list))
    for r in rows:
        sch = r.get("schema") or "?"
        per_schema[sch][r["condition_id"]].append(r)
    per_schema_out = {
        sch: {
            cond: {
                "n": len(v),
                "search_rate": sum(1 for x in v if x["action_type"] == "search") / max(1, len(v)),
                "mean_margin": mean([x["margin"] for x in v]) if v else None,
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
