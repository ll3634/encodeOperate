#!/usr/bin/env python3
"""Factorial 2x3 analysis: target_location (SF / distractor) x wrapper_semantics (neutral / commitment / anti).

Reads 4 paired eval files where "low" condition is always neutral at the same target:
  sf_commitment_vs_neutral, sf_anti_vs_neutral,
  distractor_commitment_vs_neutral, distractor_anti_vs_neutral

Reconstructs per-sample margins for all 6 cells, then runs paired contrasts
and interaction tests.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
from scipy import stats


def load_paired(pairs_path, eval_path):
    pairs = {json.loads(l)["sample_id"]: json.loads(l) for l in open(pairs_path)}
    evs = [json.loads(l) for l in open(eval_path)]
    lo = {r["sample_id"]: r for r in evs if r["condition"] == "low"}
    hi = {r["sample_id"]: r for r in evs if r["condition"] == "high"}
    return pairs, lo, hi


def perm_p(x, n=20000, seed=0):
    x = np.asarray(x); rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, len(x)))
    null = (signs * x).mean(axis=1); obs = x.mean()
    return float((null <= obs).mean()), float((np.abs(null) >= abs(obs)).mean())


def boot_ci(x, n=20000, seed=1):
    x = np.asarray(x); rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def summarize(name, x):
    m, lo, hi = boot_ci(x)
    p_less, p_two = perm_p(x)
    w = stats.wilcoxon(x, alternative="less").pvalue if len(x) > 3 else float("nan")
    return {
        "name": name, "n": int(len(x)), "mean": m, "median": float(np.median(x)),
        "ci95": [lo, hi], "perm_p_one_less": p_less, "perm_p_two_sided": p_two,
        "pct_neg": float((x < 0).mean()), "pct_pos": float((x > 0).mean()),
        "wilcoxon_p_less": float(w),
    }


def fmt(s):
    return (f'  {s["name"]:38s} N={s["n"]:3d}  mean={s["mean"]:+.3f}  '
            f'median={s["median"]:+.3f}  CI=[{s["ci95"][0]:+.3f}, {s["ci95"][1]:+.3f}]  '
            f'perm_two={s["perm_p_two_sided"]:.4g}  pct<0={s["pct_neg"]:.0%}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/factorial_2x3")
    ap.add_argument("--out", default="results/factorial_2x3/factorial_report.json")
    args = ap.parse_args()

    R = Path(args.root)
    cells = {
        "sf_commit":   (R / "sf_commitment_vs_neutral"),
        "sf_anti":     (R / "sf_anti_vs_neutral"),
        "dist_commit": (R / "distractor_commitment_vs_neutral"),
        "dist_anti":   (R / "distractor_anti_vs_neutral"),
    }
    data = {}
    for k, d in cells.items():
        p, lo, hi = load_paired(d / "pairs.jsonl", d / "eval_results.jsonl")
        data[k] = {"pairs": p, "low": lo, "high": hi}

    ids = sorted(set.intersection(*[set(v["pairs"].keys()) & set(v["low"].keys()) & set(v["high"].keys())
                                    for v in data.values()]))
    print(f"[info] N paired across all 4 cells = {len(ids)}")

    def arr(cell, cond, field="margin"):
        src = data[cell]["low" if cond == "low" else "high"]
        return np.array([src[i][field] for i in ids])

    # Per-cell margins (4 high cells). "low" margins (neutral) should match across cells at same target:
    m_neu_sf   = (arr("sf_commit",   "low") + arr("sf_anti",   "low")) / 2
    m_neu_dist = (arr("dist_commit", "low") + arr("dist_anti", "low")) / 2
    m_com_sf   = arr("sf_commit",   "high")
    m_ant_sf   = arr("sf_anti",     "high")
    m_com_dist = arr("dist_commit", "high")
    m_ant_dist = arr("dist_anti",   "high")

    # Sanity: neutral margins at same target should be near-identical
    neu_sf_disc  = np.abs(arr("sf_commit", "low")   - arr("sf_anti", "low")).mean()
    neu_dst_disc = np.abs(arr("dist_commit", "low") - arr("dist_anti", "low")).mean()
    print(f"[sanity] mean |neu_sf_commit - neu_sf_anti|   = {neu_sf_disc:.4f}")
    print(f"[sanity] mean |neu_dist_commit - neu_dist_anti| = {neu_dst_disc:.4f}")

    cell_means = {
        "neutral_sf":    float(m_neu_sf.mean()),
        "commitment_sf": float(m_com_sf.mean()),
        "anti_sf":       float(m_ant_sf.mean()),
        "neutral_dist":  float(m_neu_dist.mean()),
        "commitment_dist": float(m_com_dist.mean()),
        "anti_dist":     float(m_ant_dist.mean()),
    }
    print("\n=== 2x3 cell means (margin) ===")
    print(f'            neutral    commit    anti')
    print(f'  sf      {cell_means["neutral_sf"]:+7.3f}  {cell_means["commitment_sf"]:+7.3f}  {cell_means["anti_sf"]:+7.3f}')
    print(f'  dist    {cell_means["neutral_dist"]:+7.3f}  {cell_means["commitment_dist"]:+7.3f}  {cell_means["anti_dist"]:+7.3f}')

    # Planned paired contrasts (within-target, diff from neutral)
    contrasts = {
        "commit-neutral | SF":   m_com_sf - m_neu_sf,
        "anti-neutral   | SF":   m_ant_sf - m_neu_sf,
        "commit-neutral | dist": m_com_dist - m_neu_dist,
        "anti-neutral   | dist": m_ant_dist - m_neu_dist,
    }
    contrast_stats = {k: summarize(k, v) for k, v in contrasts.items()}
    print("\n=== Within-target paired contrasts vs neutral ===")
    for s in contrast_stats.values():
        print(fmt(s))

    # Interactions: locality of commitment, locality of anti, and commit-vs-anti asymmetry
    inter = {
        "commit locality  (commit-neutral)|SF - (commit-neutral)|dist":
            contrasts["commit-neutral | SF"] - contrasts["commit-neutral | dist"],
        "anti locality    (anti-neutral)|SF   - (anti-neutral)|dist":
            contrasts["anti-neutral   | SF"] - contrasts["anti-neutral   | dist"],
        "asymmetry        (commit-neutral)|SF + (anti-neutral)|SF":
            contrasts["commit-neutral | SF"] + contrasts["anti-neutral   | SF"],
    }
    inter_stats = {k: summarize(k, v) for k, v in inter.items()}
    print("\n=== Interaction / asymmetry contrasts ===")
    for s in inter_stats.values():
        print(fmt(s))

    # 2ndSR rates per cell
    def sr(cell, cond):
        src = data[cell]["low" if cond == "low" else "high"]
        return np.array([int(src[i]["action_type"] == "search") for i in ids])
    sr_cells = {
        "sf_commit":   sr("sf_commit", "high"),   "sf_anti":  sr("sf_anti", "high"),
        "dist_commit": sr("dist_commit", "high"), "dist_anti": sr("dist_anti", "high"),
        "sf_neutral":  sr("sf_commit", "low"),    "dist_neutral": sr("dist_commit", "low"),
    }
    print("\n=== 2ndSR per cell ===")
    for k, v in sr_cells.items():
        print(f'  {k:20s} {v.mean():.2%}  (n_search={int(v.sum())}/{len(v)})')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "n_paired": len(ids),
        "cell_means": cell_means,
        "contrasts": contrast_stats,
        "interactions": inter_stats,
        "sr_cells": {k: float(v.mean()) for k, v in sr_cells.items()},
        "neutral_replicate_check": {"sf": float(neu_sf_disc), "dist": float(neu_dst_disc)},
    }, open(args.out, "w"), indent=2)
    print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
