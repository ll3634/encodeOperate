#!/usr/bin/env python3
"""Aggregate sampling-decoding eval into per-sample probabilities, paired
permutation tests, bootstrap CIs, and quality reports.

Reads `sampling_eval_results_qwen.jsonl` (rows per (sample_id, condition, seed))
and writes:
  - per_sample_sampling_summary.csv
  - sampling_summary_qwen.json
  - parse_failure_report.md
  - report.md
"""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np


def majority(vals):
    if not vals:
        return None
    c = {}
    for v in vals:
        c[v] = c.get(v, 0) + 1
    best_v, best_n = None, -1
    for v, n in sorted(c.items(), key=lambda x: str(x[0])):
        if n > best_n:
            best_v, best_n = v, n
    return best_v


def per_sample_aggregate(rows):
    """Collapse seeds → per (sample_id, condition) row of probabilities/majorities."""
    bucket = defaultdict(list)
    for r in rows:
        bucket[(r["sample_id"], r["condition"])].append(r)
    out = {}
    for key, rs in bucket.items():
        n = len(rs)
        is_search = [int((r["action_type"] or r["first_action_token"]) == "search") for r in rs]
        is_stop   = [int((r["action_type"] or r["first_action_token"]) == "stop")   for r in rs]
        commit_w  = [int(r["contains_W"]) for r in rs]
        contains_gold = [int(r["contains_gold"]) for r in rs]
        em        = [r["em"] for r in rs if r["em"] is not None]
        pf        = [int(bool(r["parse_failure"])) for r in rs]
        mb        = [int(bool(r["malformed_both"])) for r in rs]
        out[key] = {
            "sample_id": key[0], "condition": key[1],
            "schema_type": rs[0].get("schema_type"),
            "K": n,
            "P_search":   float(np.mean(is_search)),
            "P_stop":     float(np.mean(is_stop)),
            "P_commitW":  float(np.mean(commit_w)),
            "P_containsGold": float(np.mean(contains_gold)),
            "P_em":       (float(np.mean(em)) if em else None),
            "P_parseFail": float(np.mean(pf)),
            "P_malformedBoth": float(np.mean(mb)),
            "majority_action": majority([r["action_type"] for r in rs]),
            "majority_commitW": int(sum(commit_w) > n / 2),
            "std_search":  float(np.std(is_search,  ddof=0)),
            "std_commitW": float(np.std(commit_w,   ddof=0)),
            "mean_n_tokens": float(np.mean([r["n_generated_tokens"] for r in rs])),
        }
    return out


def paired_perm_sign_flip(diffs, n_iter=10000, seed=42):
    diffs = np.asarray(diffs, dtype=np.float64)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return float("nan"), float("nan"), 0
    obs = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_iter, len(diffs)))
    null = (signs * diffs).mean(axis=1)
    p = float((np.abs(null) >= abs(obs) - 1e-12).mean())
    return obs, p, len(diffs)


def bootstrap_ci_mean(diffs, n_iter=10000, seed=42, alpha=0.05):
    diffs = np.asarray(diffs, dtype=np.float64)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_iter, len(diffs)))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def mcnemar_exact(b, c):
    """Exact two-sided binomial McNemar: discordant pairs (b, c)."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_one = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return float(min(1.0, 2 * p_one))


def paired_test_block(per, condA, condB, key):
    sids = sorted({s for (s, c) in per if c in (condA, condB)})
    diffs = []
    for sid in sids:
        a, b = per.get((sid, condA)), per.get((sid, condB))
        if a is None or b is None or a[key] is None or b[key] is None:
            continue
        diffs.append(b[key] - a[key])
    obs, p, n = paired_perm_sign_flip(diffs, n_iter=10000, seed=42)
    lo, hi = bootstrap_ci_mean(diffs, n_iter=10000, seed=42)
    return {
        "n_pairs": n, "mean_delta": obs,
        "perm_p_two_sided": p,
        "ci95_lower": lo, "ci95_upper": hi,
        "median_delta": float(np.median(diffs)) if diffs else float("nan"),
    }


def mcnemar_block(per, condA, condB, vote_key):
    sids = sorted({s for (s, c) in per if c in (condA, condB)})
    b = c = a = d = 0  # b = A=0,B=1 ; c = A=1,B=0
    for sid in sids:
        ra, rb = per.get((sid, condA)), per.get((sid, condB))
        if ra is None or rb is None:
            continue
        va, vb = int(ra[vote_key]), int(rb[vote_key])
        if va == 0 and vb == 1: b += 1
        elif va == 1 and vb == 0: c += 1
        elif va == 1 and vb == 1: a += 1
        else: d += 1
    return {"a_11": a, "b_01": b, "c_10": c, "d_00": d, "n": a + b + c + d,
            "p_exact": mcnemar_exact(b, c)}


def write_csv(per_sample, path):
    rows = list(per_sample.values())
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in sorted(rows, key=lambda x: (x["condition"], x["sample_id"])):
            w.writerow(r)


def write_parse_failure_report(rows, per_sample, conds, path, model_label):
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    lines = [f"# Parse-failure / quality report — {model_label}", ""]
    lines += ["| condition | N gens | parse_fail % | malformed_both % | "
              "mean_n_tokens | per-seed mean(std P_search) | "
              "per-seed mean(std P_commitW) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for c in conds:
        rs = by_cond.get(c, [])
        if not rs:
            lines.append(f"| {c} | 0 | - | - | - | - | - |"); continue
        pf = 100.0 * np.mean([int(r["parse_failure"]) for r in rs])
        mb = 100.0 * np.mean([int(r["malformed_both"]) for r in rs])
        ml = float(np.mean([r["n_generated_tokens"] for r in rs]))
        per = [v for (k, v) in per_sample.items() if k[1] == c]
        s_std = float(np.mean([p["std_search"]  for p in per])) if per else float("nan")
        w_std = float(np.mean([p["std_commitW"] for p in per])) if per else float("nan")
        lines.append(f"| {c} | {len(rs)} | {pf:.2f} | {mb:.2f} | {ml:.1f} | "
                     f"{s_std:.4f} | {w_std:.4f} |")
    Path(path).write_text("\n".join(lines) + "\n")


def _fmt_p(p, B=10000):
    if p is None or (isinstance(p, float) and (p != p)):
        return "n/a"
    if p <= 0.0:
        return f"<{1.0/B:.0e} (0/{B} permutations)"
    return f"{p:.4g}"


def write_report(summary, conds, path, model_label):
    L = [f"# Decoding-robustness report — {model_label}",
         "",
         f"Sampling: temp={summary['config']['temperature']} "
         f"top_p={summary['config']['top_p']} "
         f"K_seeds={summary['config']['n_seeds']} "
         f"max_new_tokens={summary['config']['max_new_tokens']}",
         "",
         f"Per-condition N (unique sample_ids): " +
         ", ".join(f"{c}={summary['n_per_condition'].get(c,0)}" for c in conds),
         ""]
    L += ["## Primary paired tests (sign-flip permutation, B=10000; bootstrap CI95 on Δ)", ""]
    for pair_label, key, blk in summary["paired"]:
        L += [f"### {pair_label}  (metric: `{key}`)",
              f"- n_pairs           = **{blk['n_pairs']}**",
              f"- mean Δ (B−A)     = **{blk['mean_delta']:+.4f}** "
              f"(median {blk['median_delta']:+.4f})",
              f"- bootstrap CI95   = [{blk['ci95_lower']:+.4f}, {blk['ci95_upper']:+.4f}]",
              f"- permutation p    = **{_fmt_p(blk['perm_p_two_sided'])}**", ""]
    L += ["## Majority-vote McNemar (per-sample majority across the K seeds)", ""]
    for pair_label, vote_key, blk in summary["mcnemar"]:
        L += [f"### {pair_label}  (vote: `{vote_key}`)",
              f"- 2×2 table: a={blk['a_11']} (both 1), b={blk['b_01']} "
              f"(A=0→B=1), c={blk['c_10']} (A=1→B=0), d={blk['d_00']} (both 0); n={blk['n']}",
              f"- exact two-sided p = **{_fmt_p(blk['p_exact'])}**", ""]
    L += ["## Interpretation", ""]
    L += [summary["interpretation_text"], ""]
    Path(path).write_text("\n".join(L))


def interpret(summary, conds):
    primary = {label: blk for label, _, blk in summary["paired"]}
    s_blk = primary.get("ΔP(search)  T0 − N0")
    w_blk = primary.get("ΔP(commit-W)  T0 − N0")
    pf_lines = []
    for c in conds:
        nrec = summary["quality"]["n_per_condition_gens"].get(c, 0)
        pf = summary["quality"]["parse_failure_pct"].get(c, 0.0)
        pf_lines.append(f"{c}: pf={pf:.1f}%/{nrec} gens")
    qstr = "; ".join(pf_lines)

    def sign(blk):
        if blk is None: return None
        return "+" if blk["mean_delta"] > 0 else ("-" if blk["mean_delta"] < 0 else "0")
    ds = sign(s_blk); dw = sign(w_blk)
    sig_s = (s_blk is not None and s_blk["perm_p_two_sided"] < 0.05)
    sig_w = (w_blk is not None and w_blk["perm_p_two_sided"] < 0.05)

    if ds == "-" and dw == "+" and sig_s and sig_w:
        verdict = ("**STRONG**: T0 reduces P(search) and increases P(commit-W) "
                   "under sampling decoding. The extractability claim is NOT a "
                   "greedy artifact.")
    elif (ds == "-" and sig_s) or (dw == "+" and sig_w):
        verdict = ("**PARTIAL**: direction of effect holds under sampling but "
                   "only one of the two metrics reaches conventional significance.")
    else:
        verdict = ("**FAILURE TO REPLICATE under sampling**: the extractability "
                   "claim should be scoped to greedy / low-temperature decoding.")
    return verdict + f"\n\nQuality controls: {qstr}."


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp", required=True,
                    help="sampling_eval_results_*.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-label", default="qwen")
    ap.add_argument("--conditions", nargs="+", default=["N0", "T0", "S0"])
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp)]
    rows = [r for r in rows if r["condition"] in args.conditions]
    per = per_sample_aggregate(rows)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(per, out_dir / "per_sample_sampling_summary.csv")

    paired = []
    if "N0" in args.conditions and "T0" in args.conditions:
        paired.append(("ΔP(search)  T0 − N0", "P_search",
                       paired_test_block(per, "N0", "T0", "P_search")))
        paired.append(("ΔP(commit-W)  T0 − N0", "P_commitW",
                       paired_test_block(per, "N0", "T0", "P_commitW")))
    if "S0" in args.conditions and "T0" in args.conditions:
        paired.append(("ΔP(commit-W)  T0 − S0", "P_commitW",
                       paired_test_block(per, "S0", "T0", "P_commitW")))
        paired.append(("ΔP(search)  T0 − S0", "P_search",
                       paired_test_block(per, "S0", "T0", "P_search")))
    mc = []
    if "N0" in args.conditions and "T0" in args.conditions:
        mc.append(("majority commit-W  N0 vs T0", "majority_commitW",
                   mcnemar_block(per, "N0", "T0", "majority_commitW")))

    n_per_cond = {c: len({k[0] for k in per if k[1] == c}) for c in args.conditions}
    n_per_cond_gens = {c: sum(1 for r in rows if r["condition"] == c) for c in args.conditions}
    pf_pct = {c: float(100.0 * np.mean([int(r["parse_failure"])
                                        for r in rows if r["condition"] == c]))
              if n_per_cond_gens.get(c, 0) else 0.0 for c in args.conditions}

    summary = {
        "model_label": args.model_label,
        "config": {"temperature": args.temperature, "top_p": args.top_p,
                   "max_new_tokens": args.max_new_tokens,
                   "n_seeds": len({r["seed"] for r in rows})},
        "n_per_condition": n_per_cond,
        "paired": paired,
        "mcnemar": mc,
        "quality": {"parse_failure_pct": pf_pct,
                    "n_per_condition_gens": n_per_cond_gens,
                    "malformed_both_pct": {
                        c: float(100.0 * np.mean(
                            [int(r["malformed_both"])
                             for r in rows if r["condition"] == c])) if n_per_cond_gens.get(c, 0) else 0.0
                        for c in args.conditions},
                    "mean_n_tokens": {
                        c: float(np.mean([r["n_generated_tokens"]
                                          for r in rows if r["condition"] == c])) if n_per_cond_gens.get(c, 0) else 0.0
                        for c in args.conditions}},
    }
    summary["interpretation_text"] = interpret(summary, args.conditions)
    with open(out_dir / f"sampling_summary_{args.model_label}.json", "w") as f:
        json.dump(summary, f, indent=2)
    write_parse_failure_report(rows, per, args.conditions,
                               out_dir / "parse_failure_report.md", args.model_label)
    write_report(summary, args.conditions,
                 out_dir / "report.md", args.model_label)
    print("[done]")
    for label, _, blk in paired:
        print(f"  {label:35s} n={blk['n_pairs']:3d}  "
              f"meanΔ={blk['mean_delta']:+.4f}  "
              f"CI=[{blk['ci95_lower']:+.4f},{blk['ci95_upper']:+.4f}]  "
              f"perm_p={blk['perm_p_two_sided']:.4g}")
    for label, _, blk in mc:
        print(f"  {label:35s}  b={blk['b_01']} c={blk['c_10']}  "
              f"exact_p={blk['p_exact']:.4g}")


if __name__ == "__main__":
    main()

