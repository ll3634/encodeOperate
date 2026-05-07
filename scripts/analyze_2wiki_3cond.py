"""Analyze 2Wiki extractability eval (N0/T0/S0, no T1) for any model."""
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


def diff_ci(xs, ys, n_iter=5000, seed=1, alpha=0.05):
    rng = random.Random(seed); n = len(xs); out = []
    for _ in range(n_iter):
        idx = [rng.randrange(n) for _ in range(n)]
        out.append(sum(xs[i] for i in idx)/n - sum(ys[i] for i in idx)/n)
    out.sort()
    return out[int(alpha/2 * n_iter)], out[int((1-alpha/2) * n_iter)]


def cell_stats(rows):
    n = len(rows)
    if n == 0: return {"n": 0}
    ml = [r["margin_label"] for r in rows if r.get("margin_label") is not None]
    mft = [r["margin_first_token"] for r in rows if r.get("margin_first_token") is not None]
    return {
        "n": n,
        "search_rate":  sum(1 for r in rows if r.get("action_type") == "search") / n,
        "stop_rate":    sum(1 for r in rows if r.get("action_type") == "stop") / n,
        "commit_W":     sum(1 for r in rows if r.get("contains_W")) / n,
        "em":           sum(1 for r in rows if r.get("em") == 1) / n,
        "parse_fail":   sum(1 for r in rows if r.get("parse_failure")) / n,
        "mean_ml":      st.fmean(ml) if ml else None,
        "mean_mft":     st.fmean(mft) if mft else None,
    }


def paired_tests(A, B, label_a, label_b):
    ka = {r["sample_id"]: int(bool(r.get("contains_W"))) for r in A}
    kb = {r["sample_id"]: int(bool(r.get("contains_W"))) for r in B}
    ids = sorted(set(ka) & set(kb))
    ya = [ka[i] for i in ids]; yb = [kb[i] for i in ids]
    b = sum(1 for x, y in zip(ya, yb) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(ya, yb) if x == 0 and y == 1)
    p_mc = mcnemar_exact(b, c)
    fa = {r["sample_id"]: r.get("margin_label") for r in A}
    fb = {r["sample_id"]: r.get("margin_label") for r in B}
    common = [i for i in ids if fa[i] is not None and fb[i] is not None]
    fa_l = [fa[i] for i in common]; fb_l = [fb[i] for i in common]
    if fa_l:
        obs_diff, p_perm = paired_perm(fa_l, fb_l)
        try: w_p = wilcoxon(fa_l, fb_l).pvalue if HAVE_SCIPY else None
        except ValueError: w_p = 1.0
        lo, hi = diff_ci(fa_l, fb_l)
    else:
        obs_diff = p_perm = w_p = lo = hi = None
    return {
        "pair": f"{label_a} vs {label_b}",
        "n_pairs": len(ids),
        "commitW_delta": (sum(ya)-sum(yb))/max(1,len(ya)),
        "commitW_mcnemar_b": b, "commitW_mcnemar_c": c, "commitW_p": p_mc,
        "margin_label_delta": obs_diff,
        "margin_label_ci95":  [lo, hi] if lo is not None else None,
        "margin_label_perm_p": p_perm,
        "margin_label_wilcoxon_p": w_p,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.eval)]
    by_cond = defaultdict(list)
    for r in rows: by_cond[r["condition"]].append(r)

    cells = {c: cell_stats(by_cond[c]) for c in ("N0", "T0", "S0")}
    pairs = [
        paired_tests(by_cond["T0"], by_cond["N0"], "T0", "N0"),
        paired_tests(by_cond["S0"], by_cond["T0"], "S0", "T0"),
        paired_tests(by_cond["S0"], by_cond["N0"], "S0", "N0"),
    ]

    out = {"cells": cells, "paired_tests": pairs,
           "n_total": len(rows), "model": args.model}
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2))

    # Brief markdown
    lines = [f"# 2Wiki Extractability ({args.model})", ""]
    lines.append("| cond | n | search | stop | commit-W | EM | mean_ml | mean_mft |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for c in ("N0","T0","S0"):
        s = cells[c]
        lines.append(f"| {c} | {s['n']} | {s['search_rate']:.2f} | {s['stop_rate']:.2f} | "
                     f"{s['commit_W']:.2f} | {s['em']:.2f} | "
                     f"{s['mean_ml']:+.2f} | {s['mean_mft']:+.2f} |")
    lines += ["", "| pair | n | dCommitW | McNemar (b,c) | p | dMargin | perm p |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for p in pairs:
        lines.append(f"| {p['pair']} | {p['n_pairs']} | {p['commitW_delta']:+.3f} | "
                     f"({p['commitW_mcnemar_b']},{p['commitW_mcnemar_c']}) | "
                     f"{p['commitW_p']:.2e} | "
                     f"{p['margin_label_delta']:+.3f} | "
                     f"{p['margin_label_perm_p']:.4f} |")
    (out_dir / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
