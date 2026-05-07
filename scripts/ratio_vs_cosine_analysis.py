#!/usr/bin/env python3
"""Ratio-vs-cosine scatter analysis for K=200 random + 6 named directions
on the L20 §3 operating point.

NO new GPU runs. Random direction vectors regenerated from the SEED used
in scripts/decomposition_ci_null.py (SEED=20260429); per-direction shifts
loaded from results/decomposition_ci_null/per_example_shifts.npz['random']
(shape K=200 x N=100). Named-direction shifts pulled from cached OCFT files.
"""
import json
from pathlib import Path

import numpy as np

SEED_RAND = 20260429   # must match decomposition_ci_null.SEED
DIM = 3584
K = 200
N = 100

EVIDENCE_BIN = (0.005, 0.025)
D3_BIN = (0.025, 0.055)
EVIDENCE_BIN_WIDE = (0.001, 0.040)

OUT = Path("results/ocft/ratio_vs_cosine"); OUT.mkdir(parents=True, exist_ok=True)

DIR_FILES = {
    # Canonical §3 evidence direction (matches direction_decomp_parallel_layer20.npz)
    "E":  "results/phase1_probe/probe_direction_l20.npz",
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


def normalize_rms(d, t=1.0):
    r = float(np.sqrt(np.mean(d ** 2)))
    return d * (t / r) if r > 1e-12 else d


def gen_random_dirs(dim, n, seed):
    rng = np.random.RandomState(seed)
    return np.stack([normalize_rms(rng.randn(dim).astype(np.float32), 1.0)
                     for _ in range(n)])


def load_named_dir(name):
    f = DIR_FILES[name]
    if f.endswith(".npz"):
        return np.load(f, allow_pickle=True)["decision_direction"].astype(np.float64)
    return np.load(f).astype(np.float64)


def cos_to(A, V):  # V: (K,D) or (D,)
    A = np.asarray(A, np.float64); V = np.asarray(V, np.float64)
    if V.ndim == 1:
        return float(np.dot(A, V) / (np.linalg.norm(A) * np.linalg.norm(V)))
    return (V @ A) / (np.linalg.norm(V, axis=1) * np.linalg.norm(A))


def perm_below_median(value, pool, n_perm=10000, seed=0):
    """One-sided test: is `value` significantly BELOW median(pool)?
    H0: value is exchangeable with pool elements.
    Pool the value with the population, draw a random element of size 1,
    p = P(draw <= value | H0). Use bootstrap-like permutation: sample with
    replacement from pool ∪ {value} and compute empirical CDF at `value`.
    """
    rng = np.random.RandomState(seed)
    combined = np.concatenate([pool, [value]])
    draws = rng.choice(combined, size=n_perm, replace=True)
    p = float((draws <= value).mean())
    return max(p, 1.0 / n_perm)


def bin_stats(name, value, pool, lo, hi, seed):
    return {
        "direction": name,
        "value": float(value),
        "bin_lo": float(lo), "bin_hi": float(hi),
        "n_in_bin": int(len(pool)),
        "bin_min": float(np.min(pool)) if len(pool) else None,
        "bin_p25": float(np.percentile(pool, 25)) if len(pool) else None,
        "bin_median": float(np.median(pool)) if len(pool) else None,
        "bin_mean": float(np.mean(pool)) if len(pool) else None,
        "bin_p75": float(np.percentile(pool, 75)) if len(pool) else None,
        "bin_max": float(np.max(pool)) if len(pool) else None,
        "percentile_of_value": (
            float((pool <= value).mean() * 100.0) if len(pool) else None),
        "p_value_below_median": (
            perm_below_median(value, pool, seed=seed) if len(pool) else None),
    }


def main():
    print(f"[init] regenerating K={K} random unit-RMS directions seed={SEED_RAND}")
    R = gen_random_dirs(DIM, K, SEED_RAND)
    A = np.load("steering/directions/direction_decomp_full_layer20.npz",
                allow_pickle=True)["decision_direction"].astype(np.float64)
    print(f"  A shape={A.shape} rms={np.sqrt(np.mean(A**2)):.4f}")

    cos_R = cos_to(A, R)              # (K,)
    abs_cos_R = np.abs(cos_R)

    # Cached random shifts (K,N): rand[k,i] = margin_with_R_k - margin_baseline
    rand_sh = np.load("results/decomposition_ci_null/per_example_shifts.npz",
                      allow_pickle=True)["random"].astype(np.float64)
    full_sh = np.load(SHIFT_FILES["E"], allow_pickle=True)["full"].astype(np.float64)
    par_sh = {n: np.load(SHIFT_FILES[n], allow_pickle=True)["parallel"].astype(np.float64)
              for n in ["E", "D1", "D2", "D3", "D4"]}
    assert rand_sh.shape == (K, N), rand_sh.shape

    full_mean = float(full_sh.mean())
    full_abs_mean = abs(full_mean)
    print(f"  Δm_full mean = {full_mean:+.4f}  (denominator)")

    # Random per-direction aggregate ratio
    rand_dir_means = rand_sh.mean(axis=1)             # (K,) signed
    rand_signed_ratio = rand_dir_means / full_mean    # (K,) signed
    rand_abs_ratio = np.abs(rand_dir_means) / full_abs_mean  # (K,)

    # Named-direction aggregate ratio (signed and abs, same convention)
    named = {}
    for n in ["E", "D1", "D2", "D3", "D4"]:
        v = load_named_dir(n)
        c = cos_to(A, v)
        m = float(par_sh[n].mean())
        named[n] = {"cos_to_A": c, "abs_cos": abs(c),
                    "mean_par_shift": m,
                    "signed_ratio": m / full_mean,
                    "abs_ratio": abs(m) / full_abs_mean}
        print(f"  {n}: cos={c:+.4f}  mean_par={m:+.4f}  abs_ratio={named[n]['abs_ratio']:.4f}")

    # Spearman + linear trend
    from scipy.stats import spearmanr, linregress
    rho, pval = spearmanr(abs_cos_R, rand_abs_ratio)
    lin = linregress(abs_cos_R, rand_abs_ratio)
    print(f"\n[trend] Spearman ρ={rho:+.3f} p={pval:.3g}")
    print(f"[trend] linregress slope={lin.slope:+.3f} intercept={lin.intercept:+.3f} R²={lin.rvalue**2:.3f} p={lin.pvalue:.3g}")

    # STEP 3 — evidence bin
    e_val = named["E"]["abs_ratio"]
    e_cos = named["E"]["abs_cos"]
    mask_e = (abs_cos_R >= EVIDENCE_BIN[0]) & (abs_cos_R <= EVIDENCE_BIN[1])
    pool_e = rand_abs_ratio[mask_e]
    if len(pool_e) < 10:
        print(f"[bin] evidence: only {len(pool_e)} dirs in {EVIDENCE_BIN}; widening to {EVIDENCE_BIN_WIDE}")
        mask_e = (abs_cos_R >= EVIDENCE_BIN_WIDE[0]) & (abs_cos_R <= EVIDENCE_BIN_WIDE[1])
        pool_e = rand_abs_ratio[mask_e]
        e_bin = EVIDENCE_BIN_WIDE
        e_widened = True
    else:
        e_bin = EVIDENCE_BIN
        e_widened = False
    e_stats = bin_stats("E", e_val, pool_e, e_bin[0], e_bin[1], seed=11)
    e_stats["bin_widened"] = e_widened
    e_stats["abs_cos"] = e_cos
    print(f"\n[E bin {e_bin}] n={e_stats['n_in_bin']} median={e_stats['bin_median']:.4f} "
          f"E_value={e_val:.4f} pct={e_stats['percentile_of_value']:.1f} "
          f"p_below_median={e_stats['p_value_below_median']:.4f}")

    # STEP 4 — D3 bin
    d3_val = named["D3"]["abs_ratio"]
    d3_cos = named["D3"]["abs_cos"]
    mask_d3 = (abs_cos_R >= D3_BIN[0]) & (abs_cos_R <= D3_BIN[1])
    pool_d3 = rand_abs_ratio[mask_d3]
    d3_stats = bin_stats("D3", d3_val, pool_d3, D3_BIN[0], D3_BIN[1], seed=22)
    d3_stats["abs_cos"] = d3_cos
    # also one-sided ABOVE-median
    if len(pool_d3) > 0:
        rng = np.random.RandomState(33)
        combined = np.concatenate([pool_d3, [d3_val]])
        d3_stats["p_value_above_median"] = float((rng.choice(combined, 10000, replace=True) >= d3_val).mean())
        d3_stats["p_value_above_median"] = max(d3_stats["p_value_above_median"], 1e-4)
    print(f"[D3 bin {D3_BIN}] n={d3_stats['n_in_bin']} median={d3_stats['bin_median']:.4f} "
          f"D3_value={d3_val:.4f} pct={d3_stats['percentile_of_value']:.1f}")

    # Save artefacts
    scatter = []
    for k in range(K):
        scatter.append({"label": f"R{k:03d}", "kind": "random",
                        "cos": float(cos_R[k]), "abs_cos": float(abs_cos_R[k]),
                        "signed_ratio": float(rand_signed_ratio[k]),
                        "abs_ratio": float(rand_abs_ratio[k])})
    for n, d in named.items():
        scatter.append({"label": n, "kind": "named",
                        "cos": float(d["cos_to_A"]), "abs_cos": float(d["abs_cos"]),
                        "signed_ratio": float(d["signed_ratio"]),
                        "abs_ratio": float(d["abs_ratio"])})
    with open(OUT / "scatter_data.json", "w") as f:
        json.dump({"K": K, "N": N, "denom_full_mean": full_mean,
                   "rows": scatter}, f, indent=2)
    with open(OUT / "binned_analysis.json", "w") as f:
        json.dump({"evidence": e_stats, "D3": d3_stats,
                   "spearman": {"rho": float(rho), "p": float(pval)},
                   "linregress": {"slope": float(lin.slope),
                                  "intercept": float(lin.intercept),
                                  "rsq": float(lin.rvalue**2),
                                  "p": float(lin.pvalue)},
                   "named_directions": named},
                  f, indent=2)

    make_plot(abs_cos_R, rand_abs_ratio, named, lin, e_bin, D3_BIN)
    write_report(named, e_stats, d3_stats, rho, pval, lin)


if __name__ == "__main__":
    from ratio_vs_cosine_io import make_plot, write_report
    main()
