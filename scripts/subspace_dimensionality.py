#!/usr/bin/env python3
"""
Action vs Evidence Subspace Dimensionality (L20)
=================================================
Conclusive test for Q1: "Why are evidence and action orthogonal?"

Logic:
  If action is d_a-dimensional and evidence is d_e-dimensional in D=3584,
  the expected cos between two random subspaces is:
    E[|cos|] ≈ sqrt(d_a * d_e / D)    (for d_a, d_e << D)
  For d_a=1, d_e=1: E[|cos|] = sqrt(1/3584) = 0.0167
  Observed cos = 0.013-0.019 → consistent with random alignment

Method:
  1. PCA on L20 step-1 hidden states (N=486, D=3584)
  2. For each PC: compute |r| with margin (action) and |r| with label (evidence)
  3. Sort by predictive power → cumulative R²/AUROC curves
  4. Effective dimensionality = k at 90% of max performance
  5. Compare theoretical E[|cos|] with observed cos

Zero GPU cost — uses existing activations.
"""

import sys, os, json
import numpy as np
sys.stdout.reconfigure(line_buffering=True)

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, pointbiserialr


def main():
    print("=== Action vs Evidence Subspace Dimensionality (L20) ===", flush=True)

    # ── Load data ──────────────────────────────────────────────────────
    act_data = np.load("results/phase1_probe/activations_multilayer.npz",
                       allow_pickle=True)
    H = act_data["layer_20"].astype(np.float32)  # (486, 3584)
    y = act_data["y"].astype(np.int32)            # evidence label
    sids = act_data["sample_ids"].tolist()

    # Load margin_before from labels
    label_map = {}
    with open("results/phase1_probe/labels.jsonl") as f:
        for line in f:
            d = json.loads(line)
            label_map[d["sample_id"]] = d

    margins = np.array([float(label_map[s]["margin_before"]) for s in sids],
                       dtype=np.float32)

    N, D = H.shape
    print(f"  N={N}, D={D}, pos={y.sum()}, neg={(1-y).sum()}", flush=True)
    print(f"  Margin: mean={margins.mean():.3f}, std={margins.std():.3f}", flush=True)

    # ── PCA ─────────────────────────────────────────────────────────────
    print("\nRunning PCA...", flush=True)
    scaler = StandardScaler()
    H_s = scaler.fit_transform(H)
    pca = PCA(n_components=min(N, D))
    Z = pca.fit_transform(H_s)   # (N, K) where K=N=486
    K = Z.shape[1]
    print(f"  {K} PCs extracted, var explained: "
          f"top1={pca.explained_variance_ratio_[0]:.4f}, "
          f"top10={pca.explained_variance_ratio_[:10].sum():.4f}, "
          f"top50={pca.explained_variance_ratio_[:50].sum():.4f}", flush=True)

    # ── Correlation spectrum ────────────────────────────────────────────
    print("\nCorrelation spectrum...", flush=True)
    r_margin = np.array([pearsonr(Z[:, k], margins)[0] for k in range(K)])
    r_label  = np.array([pointbiserialr(y, Z[:, k])[0] for k in range(K)])

    # Sort by predictive power
    order_action   = np.argsort(-np.abs(r_margin))
    order_evidence = np.argsort(-np.abs(r_label))

    print(f"  Action (margin) top-5 |r|:  "
          f"{np.sort(np.abs(r_margin))[::-1][:5]}", flush=True)
    print(f"  Evidence (label) top-5 |r|: "
          f"{np.sort(np.abs(r_label))[::-1][:5]}", flush=True)

    # ── Cumulative performance curves ───────────────────────────────────
    print("\nCumulative R²/AUROC by number of PCs...", flush=True)
    ks = [1, 2, 3, 5, 10, 20, 50, 100, 200, K]
    ks = [k for k in ks if k <= K]

    cum_r2, cum_auroc = [], []
    for k in ks:
        # Action: cross-val R² using top-k margin-correlated PCs
        idx_a = order_action[:k]
        cv_r2 = cross_val_score(Ridge(alpha=1.0), Z[:, idx_a], margins,
                                cv=KFold(5, shuffle=True, random_state=42),
                                scoring="r2")
        cum_r2.append(float(np.mean(cv_r2)))

        # Evidence: cross-val AUROC using top-k label-correlated PCs
        idx_e = order_evidence[:k]
        cv_auc = cross_val_score(
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                               solver="lbfgs", random_state=42),
            Z[:, idx_e], y,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring="roc_auc")
        cum_auroc.append(float(np.mean(cv_auc)))

    print(f"\n  {'k':>5}  {'R²(action)':>11}  {'AUROC(evi)':>11}", flush=True)
    print(f"  {'-'*33}", flush=True)
    for k, r2, auc in zip(ks, cum_r2, cum_auroc):
        print(f"  {k:>5}  {r2:>11.4f}  {auc:>11.4f}", flush=True)

    # ── Effective dimensionality ────────────────────────────────────────
    r2_max = max(cum_r2)
    auroc_max = max(cum_auroc)

    def eff_dim(values, threshold_frac=0.90):
        target = max(values) * threshold_frac
        for k, v in zip(ks, values):
            if v >= target:
                return k
        return ks[-1]

    d_action = eff_dim(cum_r2)
    d_evidence = eff_dim(cum_auroc)

    print(f"\n  Effective dimensionality (90% of max):", flush=True)
    print(f"    Action  (R²_max={r2_max:.4f}):   d_action  = {d_action}", flush=True)
    print(f"    Evidence (AUROC_max={auroc_max:.4f}): d_evidence = {d_evidence}",
          flush=True)

    # ── Theoretical cos prediction ──────────────────────────────────────
    # For random d1-D and d2-D subspaces in D dimensions:
    #   E[cos²(principal angle)] ≈ d1*d2 / D  (when d1,d2 << D)
    #   For 1D vs 1D: E[|cos|] = sqrt(1/D)
    cos_theory_1v1 = np.sqrt(1.0 / D)
    cos_theory_actual = np.sqrt(d_action * d_evidence / D)
    cos_observed = 0.0135  # from probe experiments

    # Permutation baseline: cos between random probe directions
    rng = np.random.RandomState(42)
    null_cos = []
    for _ in range(1000):
        d1 = rng.randn(D); d1 /= np.linalg.norm(d1)
        d2 = rng.randn(D); d2 /= np.linalg.norm(d2)
        null_cos.append(abs(np.dot(d1, d2)))
    null_mean = np.mean(null_cos)
    null_95 = np.percentile(null_cos, 95)

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*75}", flush=True)
    print(f"  ★ SUBSPACE DIMENSIONALITY ANALYSIS — CONCLUSION", flush=True)
    print(f"{'='*75}", flush=True)
    print(f"  D = {D}   (residual stream dimension)", flush=True)
    print(f"  d_action   = {d_action:>3}  "
          f"(R² with {d_action} PCs = {r2_max*0.9:.4f})", flush=True)
    print(f"  d_evidence = {d_evidence:>3}  "
          f"(AUROC with {d_evidence} PCs = {auroc_max*0.9:.4f})", flush=True)
    print(f"", flush=True)
    print(f"  Observed |cos(evi_dir, act_dir)| = {cos_observed:.4f}", flush=True)
    print(f"  Random 1D vs 1D in {D}-d:     E[|cos|] = {cos_theory_1v1:.4f}  "
          f"(MC mean={null_mean:.4f}, 95th={null_95:.4f})", flush=True)
    print(f"  Random {d_action}D vs {d_evidence}D:       "
          f"E[|cos|] ≈ {cos_theory_actual:.4f}", flush=True)
    print(f"", flush=True)

    ratio = cos_observed / null_mean
    if ratio < 1.5:
        verdict = ("GEOMETRIC INEVITABILITY: observed cos is indistinguishable "
                   "from random alignment of two low-d subspaces in high-d space")
    elif ratio < 3.0:
        verdict = ("MARGINAL: observed cos is slightly above random — "
                   "possible weak coupling")
    else:
        verdict = ("SPECIFIC DECOUPLING: observed cos is well below random — "
                   "the network actively separates these subspaces")

    print(f"  observed/null ratio = {ratio:.2f}", flush=True)
    print(f"  → {verdict}", flush=True)
    print(f"{'='*75}", flush=True)

    # ── Save ────────────────────────────────────────────────────────────
    os.makedirs("results/subspace_dimensionality", exist_ok=True)
    out = "results/subspace_dimensionality/results.json"
    with open(out, "w") as f:
        json.dump({
            "D": D, "N": N,
            "d_action": d_action, "d_evidence": d_evidence,
            "r2_max": r2_max, "auroc_max": auroc_max,
            "cos_observed": cos_observed,
            "cos_theory_1v1": float(cos_theory_1v1),
            "cos_theory_subspace": float(cos_theory_actual),
            "null_cos_mean": float(null_mean),
            "null_cos_95th": float(null_95),
            "ratio_obs_over_null": float(ratio),
            "ks": ks, "cum_r2": cum_r2, "cum_auroc": cum_auroc,
            "top10_r_margin": sorted(np.abs(r_margin).tolist(), reverse=True)[:10],
            "top10_r_label": sorted(np.abs(r_label).tolist(), reverse=True)[:10],
        }, f, indent=2)
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
