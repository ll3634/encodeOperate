#!/usr/bin/env python3
"""Paired-ratio significance test — Evidence vs D1/D2/D3/D4.

Pre-registration (R1+R2+R3, mirrors OCFT/audit convention):
  STEP 0 : 5x5 cosine matrix from probe directions
  STEP 1 : per-prompt r_D_i = |Δm_par_D_i| / |Δm_full_i|; skip |full_i| < 0.01
  STEP 2 : 10 pairwise paired-permutation tests on r_D_a - r_D_b
           (10,000 sign-flips, two-sided), Bonferroni α = 0.005
  STEP 3 : single-link clustering on Bonferroni-significant edges

NO new GPU runs; reads only cached .npz / .npy artefacts.
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np

SEED = 20260502
B_BOOT = 2000
N_PERM = 10000
ALPHA = 0.05
MIN_FULL_ABS = 0.01

ROOT = Path("results/ocft")
OUT = ROOT / "paired_ratio_test"
OUT.mkdir(parents=True, exist_ok=True)

DIR_FILES = {
    "E":  "steering/directions/direction_probe_layer20.npz",
    "D1": "results/ocft/per_candidate/D1_source/direction.npy",
    "D2": "results/ocft/per_candidate/D2_action_prior/direction.npy",
    "D3": "results/ocft/per_candidate/D3_candidate_present/direction.npy",
    "D4": "results/ocft/per_candidate/D4_obs_length/direction.npy",
}
SHIFT_FILES = {
    "E":  "results/decomposition_ci_null/per_example_shifts.npz",
    "D1": "results/ocft/per_example_shifts_D1_source.npz",
    "D2": "results/ocft/per_example_shifts_D2_action_prior.npz",
    "D3": "results/ocft/per_example_shifts_D3_candidate_present.npz",
    "D4": "results/ocft/per_example_shifts_D4_obs_length.npz",
}
NAMES = ["E", "D1", "D2", "D3", "D4"]


def load_dir(name):
    f = DIR_FILES[name]
    if f.endswith(".npz"):
        return np.load(f, allow_pickle=True)["decision_direction"].astype(np.float64)
    return np.load(f).astype(np.float64)


def cos(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def boot_mean_ci(x, B=B_BOOT, level=95.0, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(x.mean()), float(lo), float(hi)


def perm_paired(d, n_perm=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    obs = float(d.mean())
    n = len(d)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    null_means = (signs * d).mean(axis=1)
    p = float((np.abs(null_means) >= abs(obs) - 1e-15).mean())
    p = max(p, 1.0 / n_perm)
    return obs, p


def main():
    # STEP 0 ──────────────────────────────────────────────────────────────────
    dirs = {n: load_dir(n) for n in NAMES}
    cmat = {a: {b: cos(dirs[a], dirs[b]) for b in NAMES} for a in NAMES}
    with open(OUT / "cosine_matrix.json", "w") as f:
        json.dump(cmat, f, indent=2)

    # STEP 1 ──────────────────────────────────────────────────────────────────
    full = np.load(SHIFT_FILES["E"], allow_pickle=True)["full"].astype(np.float64)
    par = {n: np.load(SHIFT_FILES[n], allow_pickle=True)["parallel"].astype(np.float64)
           for n in NAMES}
    sids = np.load(SHIFT_FILES["E"], allow_pickle=True)["sample_ids"]

    keep = np.abs(full) >= MIN_FULL_ABS
    n_skip = int((~keep).sum())
    if n_skip > 10:
        raise SystemExit(f"DATA INTEGRITY: {n_skip} prompts skipped (>10 threshold)")

    full_k = full[keep]
    ratios = {n: np.abs(par[n][keep]) / np.abs(full_k) for n in NAMES}
    n_eff = int(keep.sum())

    # per-prompt CSV
    with open(OUT / "per_prompt_ratios.csv", "w") as f:
        f.write("sample_id," + ",".join(NAMES) + ",full_shift\n")
        kept_ids = sids[keep]
        for i, sid in enumerate(kept_ids):
            row = [str(sid)] + [f"{ratios[n][i]:.6f}" for n in NAMES] + [f"{full_k[i]:+.6f}"]
            f.write(",".join(row) + "\n")

    # STEP 2 ──────────────────────────────────────────────────────────────────
    pairs = list(combinations(NAMES, 2))
    n_pairs = len(pairs)
    bonf_alpha = ALPHA / n_pairs

    pairwise = []
    for i, (a, b) in enumerate(pairs):
        d = ratios[a] - ratios[b]
        mean_diff, ci_lo, ci_hi = boot_mean_ci(d, seed=SEED + i)
        _, p_raw = perm_paired(d, seed=SEED + 1000 + i)
        p_bonf = min(1.0, p_raw * n_pairs)
        pairwise.append({
            "a": a, "b": b,
            "mean_a": float(ratios[a].mean()),
            "mean_b": float(ratios[b].mean()),
            "mean_diff": mean_diff,
            "ci_low": ci_lo, "ci_high": ci_hi,
            "p_raw": p_raw, "p_bonf": p_bonf,
            "significant": bool(p_raw <= bonf_alpha),
        })

    # STEP 3 — single-link clustering on non-significant edges ───────────────
    parent = {n: n for n in NAMES}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for r in pairwise:
        if not r["significant"]:
            union(r["a"], r["b"])

    clusters = {}
    for n in NAMES:
        clusters.setdefault(find(n), []).append(n)
    cluster_list = []
    for i, members in enumerate(sorted(clusters.values(),
                                       key=lambda m: -np.mean([ratios[x].mean() for x in m]))):
        mean_r = float(np.mean([ratios[x].mean() for x in members]))
        cluster_list.append({"id": i + 1, "members": members, "mean_ratio": mean_r})

    cluster_assignment = {"alpha": ALPHA, "bonferroni_alpha": bonf_alpha,
                          "n_pairs": n_pairs, "clusters": cluster_list}
    with open(OUT / "cluster_assignment.json", "w") as f:
        json.dump(cluster_assignment, f, indent=2)
    with open(OUT / "pairwise_results.json", "w") as f:
        json.dump({"n_eff": n_eff, "n_skipped": n_skip,
                   "min_full_abs_threshold": MIN_FULL_ABS,
                   "n_perm": N_PERM, "n_boot": B_BOOT,
                   "alpha": ALPHA, "bonferroni_alpha": bonf_alpha,
                   "pairwise": pairwise}, f, indent=2)

    write_report(cmat, ratios, pairwise, cluster_list, n_eff, n_skip, bonf_alpha)


if __name__ == "__main__":
    from paired_ratio_report import write_report
    main()
