#!/usr/bin/env python3
"""Analyze anti-cue specificity (2 targets x 4 cues) eval."""
import argparse, json
from pathlib import Path

import numpy as np
from scipy import stats

CUES_ALL = ["neutral", "task_missingness", "generic_incompleteness", "action_directive"]
TARGETS = ["sf", "distractor"]


def perm_two(x, n=20000, seed=0):
    x = np.asarray(x); rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, len(x)))
    null = (signs * x).mean(axis=1); obs = x.mean()
    return float((null <= obs).mean()), float((np.abs(null) >= abs(obs)).mean())


def boot_ci(x, n=20000, seed=1):
    x = np.asarray(x); rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def summarize(name, x):
    x = np.asarray(x)
    m, lo, hi = boot_ci(x)
    p_less, p_two = perm_two(x)
    w_p = float(stats.wilcoxon(x, alternative="two-sided").pvalue) if len(x) > 3 and np.any(x != 0) else float("nan")
    pos = int((x > 0).sum()); neg = int((x < 0).sum())
    s_p = float(stats.binomtest(max(pos, neg), pos + neg, 0.5, alternative="greater").pvalue) if pos + neg > 0 else float("nan")
    return {
        "name": name, "n": int(len(x)),
        "mean": float(m), "median": float(np.median(x)),
        "trimmed_mean_5pct": float(stats.trim_mean(x, 0.05)),
        "ci95": [float(lo), float(hi)],
        "perm_p_one_less": p_less, "perm_p_two_sided": p_two,
        "wilcoxon_p_two_sided": w_p,
        "sign_test_p_two_sided": s_p,
        "n_pos": pos, "n_neg": neg,
    }


def fmt(s):
    return (f'  {s["name"]:55s} N={s["n"]:3d}  '
            f'mean={s["mean"]:+7.3f}  med={s["median"]:+7.3f}  '
            f'CI=[{s["ci95"][0]:+.3f},{s["ci95"][1]:+.3f}]  '
            f'p2={s["perm_p_two_sided"]:.4g}  +{s["n_pos"]}/-{s["n_neg"]}')


def wls_design(rows, feats, cues):
    """Build regression design: margin ~ cue dummies + target + cue:target + feat controls + sample FE."""
    cue_levels = [c for c in cues if c != "neutral"]
    X_cols = []
    names = []
    y = np.array([r["margin"] for r in rows], dtype=float)
    # sample fixed effect (demean y per sample)
    sids = sorted(set(r["sample_id"] for r in rows))
    sid_to_idx = {s: i for i, s in enumerate(sids)}
    sid_mean = {}
    for r in rows:
        sid_mean.setdefault(r["sample_id"], []).append(r["margin"])
    sid_mean = {k: float(np.mean(v)) for k, v in sid_mean.items()}
    y_c = np.array([r["margin"] - sid_mean[r["sample_id"]] for r in rows])
    target_d = np.array([1.0 if r["target"] == "distractor" else 0.0 for r in rows])
    X_cols.append(target_d); names.append("target=dist")
    for c in cue_levels:
        d = np.array([1.0 if r["cue"] == c else 0.0 for r in rows])
        X_cols.append(d); names.append(f"cue={c}")
    for c in cue_levels:
        d = np.array([1.0 if (r["cue"] == c and r["target"] == "distractor") else 0.0 for r in rows])
        X_cols.append(d); names.append(f"cue={c}:dist")
    for fname in feats:
        v = np.array([r["feat"][fname] if isinstance(r["feat"][fname], (int, float)) else float(r["feat"][fname])
                      for r in rows], dtype=float)
        v = v - v.mean()
        X_cols.append(v); names.append(f"feat:{fname}")
    X = np.column_stack(X_cols)
    # OLS with intercept
    X = np.column_stack([np.ones(len(y_c)), X])
    names = ["const"] + names
    XtX = X.T @ X
    try:
        beta = np.linalg.solve(XtX, X.T @ y_c)
        resid = y_c - X @ beta
        sigma2 = (resid @ resid) / max(1, len(y_c) - X.shape[1])
        var_beta = sigma2 * np.linalg.inv(XtX)
        se = np.sqrt(np.diag(var_beta))
        t = beta / se
        p = 2 * (1 - stats.norm.cdf(np.abs(t)))
    except np.linalg.LinAlgError:
        beta = np.full(X.shape[1], np.nan); se = t = p = beta.copy()
    return [{"name": n, "beta": float(b), "se": float(s), "t": float(ti), "p": float(pi)}
            for n, b, s, ti, pi in zip(names, beta, se, t, p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/anti_cue_specificity/pairs.jsonl")
    ap.add_argument("--eval",  default="results/anti_cue_specificity/eval_results.jsonl")
    ap.add_argument("--out",   default="results/anti_cue_specificity/summary.json")
    args = ap.parse_args()

    pairs = {(r["sample_id"], r["condition_id"]): r
             for r in (json.loads(l) for l in open(args.pairs))}
    evs = [json.loads(l) for l in open(args.eval)]

    rows = []
    for e in evs:
        k = (e["sample_id"], e["condition_id"])
        if k not in pairs:
            continue
        p = pairs[k]
        rows.append({
            "sample_id": e["sample_id"],
            "target": e["target"], "cue": e["cue"],
            "condition_id": e["condition_id"],
            "margin": float(e["margin"]),
            "action_type": e["action_type"],
            "parse_failure": bool(e["parse_failure"]),
            "em": e.get("em"),
            "feat": p["feat"],
        })
    print(f"[info] loaded {len(rows)} eval rows")

    cues_present = sorted(set(r["cue"] for r in rows),
                          key=lambda c: CUES_ALL.index(c) if c in CUES_ALL else 99)
    if "neutral" not in cues_present:
        raise RuntimeError("neutral cue missing from eval data")
    CUES = cues_present
    print(f"[info] cues present: {CUES}")

    # Per-cell stats
    cells = {}
    for r in rows:
        cells.setdefault(r["condition_id"], []).append(r)
    per_cell = {}
    for cid, rs in cells.items():
        margs = np.array([r["margin"] for r in rs])
        sr = np.mean([r["action_type"] == "search" for r in rs])
        stp = np.mean([r["action_type"] == "stop"   for r in rs])
        pf = np.mean([r["parse_failure"] for r in rs])
        ems = [r["em"] for r in rs if r["em"] is not None]
        per_cell[cid] = {
            "n": int(len(rs)),
            "margin_mean": float(margs.mean()),
            "margin_median": float(np.median(margs)),
            "margin_std": float(margs.std(ddof=1)),
            "2ndSR": float(sr), "stop_rate": float(stp),
            "parse_failure_rate": float(pf),
            "em_rate": float(np.mean(ems)) if ems else None, "em_n": int(len(ems)),
        }
    print("\n=== Per-cell stats ===")
    print(f'{"cell":36s} n  margin_mean  med    2ndSR   stop   PF')
    for t in TARGETS:
        for c in CUES:
            cid = f"{t}_{c}"; s = per_cell[cid]
            print(f'  {cid:34s} {s["n"]:3d}  {s["margin_mean"]:+7.3f}  {s["margin_median"]:+7.3f}  '
                  f'{s["2ndSR"]:5.2%}  {s["stop_rate"]:5.2%}  {s["parse_failure_rate"]:5.2%}')

    # Build per-sample pivot: margin[sid][target][cue]
    sids = sorted(set(r["sample_id"] for r in rows))
    pivot = {s: {t: {} for t in TARGETS} for s in sids}
    act   = {s: {t: {} for t in TARGETS} for s in sids}
    for r in rows:
        pivot[r["sample_id"]][r["target"]][r["cue"]] = r["margin"]
        act[r["sample_id"]][r["target"]][r["cue"]] = r["action_type"]

    def vec(f):
        return np.array([f(s) for s in sids])

    # Pairwise contrasts vs neutral within target
    contrasts = {}
    for t in TARGETS:
        for c in CUES:
            if c == "neutral":
                continue
            d = vec(lambda s, t=t, c=c: pivot[s][t][c] - pivot[s][t]["neutral"])
            contrasts[f"{c} - neutral | {t}"] = summarize(f"({c}) - neutral | {t}", d)
    print("\n=== Pairwise contrasts vs neutral ===")
    for s in contrasts.values(): print(fmt(s))

    # Locality interaction per cue
    interactions = {}
    for c in [x for x in CUES if x != "neutral"]:
        d = vec(lambda s, c=c: (pivot[s]["sf"][c] - pivot[s]["sf"]["neutral"])
                               - (pivot[s]["distractor"][c] - pivot[s]["distractor"]["neutral"]))
        interactions[f"locality[{c}] = SF-effect - dist-effect"] = summarize(
            f"locality[{c}] (SF-effect - dist-effect)", d)

    # Semantic specificity (pooled and per target): only pairs available among CUES
    pooled = {}
    non_neutral = [x for x in CUES if x != "neutral"]
    for label, filt in [("pooled", lambda s, c: 0.5*(pivot[s]["sf"][c]-pivot[s]["sf"]["neutral"])
                                                 + 0.5*(pivot[s]["distractor"][c]-pivot[s]["distractor"]["neutral"])),
                         ("SF",   lambda s, c: pivot[s]["sf"][c] - pivot[s]["sf"]["neutral"]),
                         ("dist", lambda s, c: pivot[s]["distractor"][c] - pivot[s]["distractor"]["neutral"])]:
        for i in range(len(non_neutral)):
            for j in range(i + 1, len(non_neutral)):
                a, b = non_neutral[i], non_neutral[j]
                d = vec(lambda s, f=filt, a=a, b=b: f(s, a) - f(s, b))
                pooled[f"{a} vs {b} | {label}"] = summarize(
                    f"{a} - {b} [{label}]", d)
    print("\n=== Locality interactions ===")
    for s in interactions.values(): print(fmt(s))
    if pooled:
        print("\n=== Semantic specificity & directive upper bound ===")
        for s in pooled.values(): print(fmt(s))

    # Flip counts (low=neutral->search vs high=cue->stop and vice versa)
    flips = {}
    for t in TARGETS:
        for c in CUES:
            if c == "neutral":
                continue
            L1H0 = sum(1 for s in sids
                       if act[s][t]["neutral"] == "search" and act[s][t][c] != "search")
            L0H1 = sum(1 for s in sids
                       if act[s][t]["neutral"] != "search" and act[s][t][c] == "search")
            b = max(L1H0, L0H1); mc = stats.binomtest(min(L1H0, L0H1), L1H0 + L0H1, 0.5) if L1H0 + L0H1 > 0 else None
            flips[f"{t}_{c}"] = {"L1H0": int(L1H0), "L0H1": int(L0H1),
                                 "mcnemar_p_two_sided": (float(mc.pvalue) if mc else None)}
    print("\n=== 2ndSR flip counts (neutral -> cue) ===")
    for k, v in flips.items():
        print(f'  {k:40s}  L1H0(neu_search->cue_nosearch)={v["L1H0"]:2d}  '
              f'L0H1(neu_nosearch->cue_search)={v["L0H1"]:2d}  p={v["mcnemar_p_two_sided"]}')

    # Regression with confound controls + sample fixed effect (within-subject centering)
    feats_ctrl = ["tok_len", "q_overlap", "entity_count", "copula_count"]
    reg = wls_design(rows, feats_ctrl, CUES)
    print("\n=== Regression (within-subject centered) ===")
    for r in reg:
        print(f'  {r["name"]:30s}  beta={r["beta"]:+7.4f}  se={r["se"]:.4f}  t={r["t"]:+.3f}  p={r["p"]:.4g}')

    out = {
        "per_cell": per_cell,
        "contrasts_vs_neutral": contrasts,
        "locality_interactions": interactions,
        "semantic_specificity_and_directive": pooled,
        "flip_counts_2ndSR": flips,
        "regression": reg,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
