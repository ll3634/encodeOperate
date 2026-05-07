#!/usr/bin/env python3
"""Per-prompt heterogeneity of evidence-parallel causal effect.
No GPU. All cached. See task spec for verdict scheme.
"""
from __future__ import annotations
import json, sys, math
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.nullspace_rotation_io import load_directions, _u

OUT = Path("results/per_prompt_heterogeneity")
OUT.mkdir(parents=True, exist_ok=True)

SHIFT_PATH = "results/decomposition_ci_null/per_example_shifts.npz"
LABEL_PATH = "results/phase1_probe/labels.jsonl"
ACT_PATH = "results/phase1_probe/activations_multilayer.npz"

RNG = np.random.default_rng(20240101)


def bimodality_coefficient(x: np.ndarray) -> float:
    """SAS bimodality coefficient: BC = (g^2 + 1) / (k + 3(n-1)^2/((n-2)(n-3)))
    where g = skew, k = excess kurt. BC > 0.555 suggests bimodality."""
    n = len(x)
    g = stats.skew(x, bias=False)
    k = stats.kurtosis(x, fisher=True, bias=False)
    correction = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float((g * g + 1.0) / (k + correction))


def hartigan_dip(x: np.ndarray, n_boot: int = 2000):
    """Hartigan-Hartigan dip statistic with bootstrap p-value vs Uniform null.
    No diptest pkg available; implement minimal version."""
    x_sorted = np.sort(np.asarray(x, dtype=np.float64))
    def _dip(xs):
        n = len(xs)
        if n < 4:
            return 0.0
        ecdf_low = np.arange(n) / n
        ecdf_hi = (np.arange(n) + 1) / n
        # GCM/LCM via PAV is complex; use practical proxy: max gap between
        # ECDF and best unimodal envelope approximated by linear interpolant
        # between min/max. This is a SIMPLIFIED dip; for rigorous test rely on BC.
        a = (ecdf_low + ecdf_hi) / 2
        # Distance from ECDF to nearest unimodal CDF approximated by piecewise-linear.
        lin = (xs - xs[0]) / max(xs[-1] - xs[0], 1e-12)
        return float(np.max(np.abs(a - lin)))
    obs = _dip(x_sorted)
    boots = np.empty(n_boot)
    n = len(x_sorted)
    for b in range(n_boot):
        u = RNG.uniform(x_sorted[0], x_sorted[-1], n)
        boots[b] = _dip(u)
    p = float((boots >= obs).mean())
    return {"dip_proxy": obs, "p_uniform_null": p,
            "note": "simplified ECDF-vs-linear proxy; treat with BC together"}


def gaussian_mixture_2(x):
    from sklearn.mixture import GaussianMixture
    g1 = GaussianMixture(n_components=1, random_state=0).fit(x.reshape(-1, 1))
    g2 = GaussianMixture(n_components=2, random_state=0, n_init=5).fit(x.reshape(-1, 1))
    bic1, bic2 = g1.bic(x.reshape(-1, 1)), g2.bic(x.reshape(-1, 1))
    means = sorted(g2.means_.ravel().tolist())
    weights = g2.weights_.tolist()
    return {"bic_1comp": float(bic1), "bic_2comp": float(bic2),
            "delta_bic": float(bic1 - bic2),
            "means_2comp_sorted": means,
            "weights_2comp": weights,
            "favor_2comp": bool(bic2 < bic1 - 6)}


def main():
    # ---- Load shifts ----
    sh = np.load(SHIFT_PATH, allow_pickle=True)
    sample_ids = [str(s) for s in sh["sample_ids"]]
    parallel = sh["parallel"].astype(np.float64)
    full = sh["full"].astype(np.float64)
    perp = sh["perp"].astype(np.float64)
    baseline = sh["baseline"].astype(np.float64)
    random_shifts = sh["random"].astype(np.float64)  # (200, 100)
    N = len(sample_ids)
    print(f"[load] N={N}  shifts: parallel/full/perp loaded")

    # ---- Load labels and align ----
    by_id = {}
    for ln in open(LABEL_PATH):
        r = json.loads(ln)
        by_id[r["sample_id"]] = r
    lab = [by_id[s] for s in sample_ids]
    print(f"[align] labels matched: {len(lab)}/{N}")

    # ---- Load activations and align ----
    act = np.load(ACT_PATH)
    act_sids = [str(s) for s in act["sample_ids"]]
    sid2idx = {s: i for i, s in enumerate(act_sids)}
    H = np.array([act["layer_20"][sid2idx[s]] for s in sample_ids], dtype=np.float64)
    print(f"[align] hidden states layer_20: shape={H.shape}")

    # ---- Directions ----
    dirs = load_directions()
    A, E = dirs["A"], dirs["E"]
    print(f"[geom] cos(E,A)={float(E@A):+.5f}  ||A||={np.linalg.norm(A):.4f}  ||E||={np.linalg.norm(E):.4f}")

    # ---- STEP 1: ratio r_i ----
    abs_full = np.abs(full)
    abs_par = np.abs(parallel)
    eps = 1e-6
    r = abs_par / np.maximum(abs_full, eps)
    # cap at 5 for visualisation when |full| is tiny
    r_capped = np.clip(r, 0, 5)
    print(f"[r] median={np.median(r):.3f}  mean={r.mean():.3f}  p90={np.percentile(r,90):.3f}  max={r.max():.3f}")
    n_full_tiny = int((abs_full < 0.25).sum())
    print(f"[r] |full|<0.25 (denominator-tiny): {n_full_tiny}/{N}")

    # Distribution shape tests on r and on parallel directly
    bc_r = bimodality_coefficient(r_capped)
    bc_par = bimodality_coefficient(parallel)
    bc_par_abs = bimodality_coefficient(abs_par)
    dip_r = hartigan_dip(r_capped, n_boot=1000)
    dip_par = hartigan_dip(parallel, n_boot=1000)
    gmm_r = gaussian_mixture_2(r_capped)
    gmm_par = gaussian_mixture_2(parallel)
    print(f"[dist] BC(r)={bc_r:.3f}  BC(parallel)={bc_par:.3f}  BC(|par|)={bc_par_abs:.3f}")
    print(f"[dist] dip_proxy(r)={dip_r['dip_proxy']:.3f} p={dip_r['p_uniform_null']:.3f}")
    print(f"[dist] GMM r delta_bic={gmm_r['delta_bic']:+.2f} favor_2={gmm_r['favor_2comp']}")
    print(f"[dist] GMM par delta_bic={gmm_par['delta_bic']:+.2f} favor_2={gmm_par['favor_2comp']}")

    bins, edges = np.histogram(r_capped, bins=20, range=(0, 5))
    dist_out = {
        "n": int(N),
        "r_stats": {"mean": float(r.mean()), "median": float(np.median(r)),
                    "p25": float(np.percentile(r, 25)), "p75": float(np.percentile(r, 75)),
                    "p90": float(np.percentile(r, 90)), "max": float(r.max())},
        "denominator_tiny_count": n_full_tiny,
        "bimodality_coefficient": {"r_capped": bc_r, "parallel": bc_par,
                                   "abs_parallel": bc_par_abs,
                                   "threshold_for_bimodal": 0.555},
        "dip_proxy_r": dip_r,
        "dip_proxy_parallel": dip_par,
        "gmm_r": gmm_r,
        "gmm_parallel": gmm_par,
        "histogram_r_capped": {"bins": [float(x) for x in bins], "edges": [float(x) for x in edges]},
        "verdict_distribution": (
            "BIMODAL" if (bc_r > 0.555 and gmm_r["favor_2comp"]) else
            "HEAVY_TAIL" if np.percentile(r, 90) > 1.0 else "UNIMODAL_TIGHT"),
    }
    json.dump(dist_out, open(OUT / "distribution_analysis.json", "w"), indent=2, default=float)
    print(f"[dist] verdict: {dist_out['verdict_distribution']}")

    # save common arrays for steps 2-4
    np.savez_compressed(OUT / "intermediate.npz",
                        sample_ids=np.array(sample_ids), r=r, parallel=parallel,
                        full=full, perp=perp, baseline=baseline, H=H, E=E, A=A,
                        random_shifts=random_shifts)
    print("[save] distribution_analysis.json + intermediate.npz")


if __name__ == "__main__":
    main()
