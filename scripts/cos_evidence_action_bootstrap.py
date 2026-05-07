#!/usr/bin/env python3
"""95% bootstrap CI for cos(d_evidence, d_action) on Qwen2.5-7B at L20.

Resampling unit: sample_id, on both sides.
  Evidence side (n=486 from results/phase1_probe/activations_multilayer.npz):
    Resample 486 (sample_id, label, h_L20) tuples with replacement.
    Refit a logistic regression with the same hyperparameters as
    phase1_multilayer_probe.train_probe (StandardScaler + balanced LR, C=1, lbfgs)
    on ALL bootstrap samples; recover the unit direction in the ORIGINAL
    feature space via w_orig = clf.coef_[0] / scaler.scale_, normalized.

  Action side (n=200 PopQA samples from extract_action_persample_l20.py):
    Resample 200 (margin, h_L20) tuples with replacement.
    Recompute p20/p80 of margins; rebuild low/high groups.
    Direction = h_low_mean - h_high_mean, normalized.
    Skip iterations where either group is empty (rare).

Cosine = dot of the two unit vectors.

Outputs (results/cos_evidence_action_ci.json):
  point_estimate  - cos using full data (logreg evidence dir vs full action dir)
  ci_lower, ci_upper - 2.5%/97.5% percentiles of B bootstrap cosines
  n_bootstrap     - number of valid bootstrap iterations
  noise_floor     - mean |cos| over n_bootstrap pairs of random unit
                    Gaussian directions in the same dimension (3584)
  noise_floor_p975 - 97.5%-quantile of |cos| under the random null
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def fit_evidence_dir_scaled(Xs_b, yb, scale_full, seed):
    """LR fit on pre-scaled features; return unit direction in RAW feature space."""
    if (yb == 0).sum() < 2 or (yb == 1).sum() < 2:
        return None
    clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=200,
                             solver="lbfgs", random_state=seed)
    clf.fit(Xs_b, yb)
    w = clf.coef_[0] / scale_full
    n = np.linalg.norm(w)
    return (w / n).astype(np.float64) if n > 0 else None


def fit_action_dir(margins, H):
    p20 = np.percentile(margins, 20)
    p80 = np.percentile(margins, 80)
    low_mask  = margins <= p20
    high_mask = margins >= p80
    if low_mask.sum() < 2 or high_mask.sum() < 2:
        return None
    d = H[low_mask].mean(axis=0) - H[high_mask].mean(axis=0)
    n = np.linalg.norm(d)
    return (d / n).astype(np.float64) if n > 0 else None


def _one_iter(b, Xs, y, scale_full, H, m, seed_root, n_e, n_a):
    rng = np.random.default_rng(seed_root + b)
    idx_e = rng.integers(0, n_e, size=n_e)
    idx_a = rng.integers(0, n_a, size=n_a)
    d_e = fit_evidence_dir_scaled(Xs[idx_e], y[idx_e], scale_full, seed=42)
    if d_e is None:
        return b, np.nan
    d_a = fit_action_dir(m[idx_a], H[idx_a])
    if d_a is None:
        return b, np.nan
    return b, float(np.dot(d_e, d_a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-npz",
                    default="results/phase1_probe/activations_multilayer.npz")
    ap.add_argument("--action-npz",
                    default="results/cos_evidence_action_ci/popqa_l20_persample.npz")
    ap.add_argument("--out", default="results/cos_evidence_action_ci.json")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()

    # Evidence side
    e = np.load(args.evidence_npz, allow_pickle=True)
    X_e = e["layer_20"].astype(np.float32)
    y_e = e["y"].astype(np.int32)
    print(f"[evidence] X={X_e.shape} y0={int((y_e==0).sum())} y1={int((y_e==1).sum())}")

    # Action side
    a = np.load(args.action_npz, allow_pickle=True)
    H_a = a["hidden"].astype(np.float32)
    m_a = a["margins"].astype(np.float64)
    print(f"[action]   H={H_a.shape} margins:[{m_a.min():.2f},{m_a.max():.2f}] med={np.median(m_a):.2f}")

    # Pre-scale evidence features once (full-data scaler)
    sc_full = StandardScaler().fit(X_e)
    Xs_full = sc_full.transform(X_e).astype(np.float32)
    scale_full = sc_full.scale_.astype(np.float64)

    # Point estimate (full-data directions, both refit here for consistency)
    n_e, n_a, dim = len(y_e), len(m_a), X_e.shape[1]
    d_e_full = fit_evidence_dir_scaled(Xs_full, y_e, scale_full, seed=args.seed)
    d_a_full = fit_action_dir(m_a, H_a)
    point = float(np.dot(d_e_full, d_a_full))
    print(f"[point estimate] cos(d_evidence, d_action) = {point:+.6f}")

    # Bootstrap (parallel)
    print(f"[bootstrap] B={args.n_bootstrap}  n_jobs={args.n_jobs}  ...")
    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
        delayed(_one_iter)(b, Xs_full, y_e, scale_full, H_a, m_a, args.seed, n_e, n_a)
        for b in range(args.n_bootstrap))
    cos_boot = np.full(args.n_bootstrap, np.nan, dtype=np.float64)
    for b, c in results:
        cos_boot[b] = c
    skipped = int(np.sum(np.isnan(cos_boot)))

    valid_cos = cos_boot[~np.isnan(cos_boot)]
    ci_lo, ci_hi = np.percentile(valid_cos, [2.5, 97.5])
    print(f"\n[bootstrap] n_valid={len(valid_cos)} skipped={skipped}")
    print(f"            CI95% = [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"            mean = {valid_cos.mean():+.4f} median = {np.median(valid_cos):+.4f}")

    # Noise floor: random unit Gaussians in same dim
    n_noise = len(valid_cos)
    rng2 = np.random.default_rng(args.seed + 1)
    a_rand = rng2.standard_normal(size=(n_noise, dim)).astype(np.float64)
    b_rand = rng2.standard_normal(size=(n_noise, dim)).astype(np.float64)
    a_rand /= np.linalg.norm(a_rand, axis=1, keepdims=True)
    b_rand /= np.linalg.norm(b_rand, axis=1, keepdims=True)
    cos_noise = np.einsum("ij,ij->i", a_rand, b_rand)
    nf_mean = float(np.mean(np.abs(cos_noise)))
    nf_p975 = float(np.quantile(np.abs(cos_noise), 0.975))
    print(f"[noise floor] mean|cos|={nf_mean:.5f}  q97.5|cos|={nf_p975:.5f}  (dim={dim})")

    crosses_zero  = (ci_lo <= 0.0 <= ci_hi)
    crosses_floor = (ci_lo <= -nf_p975) or (ci_hi >= nf_p975) or (-nf_p975 <= ci_lo and ci_hi <= nf_p975)
    out = {
        "point_estimate": point,
        "ci_lower": float(ci_lo),
        "ci_upper": float(ci_hi),
        "n_bootstrap": int(len(valid_cos)),
        "n_bootstrap_attempted": int(args.n_bootstrap),
        "n_skipped": int(skipped),
        "bootstrap_mean": float(valid_cos.mean()),
        "bootstrap_median": float(np.median(valid_cos)),
        "noise_floor": nf_mean,
        "noise_floor_q975_abs": nf_p975,
        "ci_crosses_zero": bool(crosses_zero),
        "ci_within_noise_floor_band": bool(-nf_p975 <= ci_lo and ci_hi <= nf_p975),
        "dim": int(dim),
        "n_evidence_samples": int(n_e),
        "n_action_samples": int(n_a),
        "evidence_source": args.evidence_npz,
        "action_source": args.action_npz,
        "seed": int(args.seed),
        "elapsed_seconds": time.time() - t0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
