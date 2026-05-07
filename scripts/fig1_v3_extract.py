#!/usr/bin/env python3
"""Figure 1 v3 — extract a redesigned set of named directions at L20 with
HONEST out-of-fold AUROC (5-fold CV) for every probe-trained direction.

Evidence family (7) : LR / Ridge / mean-diff / PCA at L20  +  cross-layer
                      LR probes at L12 / L16 / L24, all evaluated on L20
                      activations against y_A (Method A, n_sf >= 1).
Operative family (3): D3'_perp, D1_perp, joint(D3'+D1)_perp  (causally
                      validated elsewhere).
Inert controls   (2): D2_bal, D4_obs_length.
Reference        (1): A_L20  (action axis).
Random null cloud(K): K=20 unit vectors uniformly sampled in null(A).

For each direction we save: cos(d, A), OOF mean AUROC (+ std across folds),
plus a unit-norm perp version for the GPU intervention pass.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "results/fig1_v3"; OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260502
K_FOLDS = 5
K_RAND  = 20

# ── load cached activations + labels ─────────────────────────────────────
acts = np.load(ROOT / "results/phase1_probe/activations_multilayer.npz",
               allow_pickle=True)
X = {L: acts[f"layer_{L}"].astype(np.float32) for L in (12, 16, 20, 24)}
sample_ids = acts["sample_ids"]
y_A = acts["y"].astype(int)
print(f"[load] N={len(sample_ids)}  y_A pos={y_A.sum()}/{len(y_A)}")

A = np.load(ROOT / "steering/directions/direction_decomp_full_layer20.npz",
            allow_pickle=True)["decision_direction"].astype(np.float32)
A_hat = A / np.linalg.norm(A)
def unit(v): return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)
def perp(v):
    p = v - float(np.dot(v, A_hat)) * A_hat
    return unit(p) if np.linalg.norm(p) > 1e-8 else v

# ── extractors that also return OOF AUROC on L20 (X_eval=X[20]) ─────────
def cv_extract(extractor, X_train, y, X_eval=None, k=K_FOLDS):
    """Train `extractor` on (X_train, y) per-fold, score d·X_eval[test]."""
    if X_eval is None: X_eval = X_train
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    aurocs = []
    for tr, te in skf.split(X_train, y):
        d = extractor(X_train[tr], y[tr])
        s = X_eval[te] @ d
        aurocs.append(max(roc_auc_score(y[te], s),
                          1 - roc_auc_score(y[te], s)))
    d_full = extractor(X_train, y)                  # for intervention
    return unit(d_full), float(np.mean(aurocs)), float(np.std(aurocs))

def lr(Xt, y):    return unit(LogisticRegression(C=1.0, max_iter=2000,
                              class_weight="balanced",
                              random_state=SEED).fit(Xt, y).coef_.ravel())
def ridge(Xt, y): return unit(RidgeClassifier(alpha=1.0, class_weight="balanced",
                              random_state=SEED).fit(Xt, y).coef_.ravel())
def mdiff(Xt, y): return unit(Xt[y == 1].mean(0) - Xt[y == 0].mean(0))
def pca1(Xt, y):
    d = Xt[y == 1].mean(0) - Xt[y == 0].mean(0)
    Xc = Xt - Xt.mean(0)
    v = np.linalg.svd(Xc, full_matrices=False)[2][0]
    return unit(v * np.sign(np.dot(v, d)))

# ── load existing OCFT/probe directions ─────────────────────────────────
def load_npy(p): return np.load(ROOT / p).astype(np.float32)
D3p_raw = unit(load_npy("results/d3_balanced_control/direction_D3prime_no_S0.npy"))
D1_raw  = unit(load_npy("results/ocft/per_candidate/D1_source/direction.npy"))
D2b_raw = unit(load_npy("results/d2_balanced_retrain/direction_D2prime_balanced.npy"))
D4_raw  = unit(load_npy("results/ocft/per_candidate/D4_obs_length/direction.npy"))
joint_raw = unit(perp(D3p_raw) + perp(D1_raw))

# ── build registry: (name, vector, family, layer, oof_auroc, std) ───────
registry = []
print("\n[evidence family — OOF AUROC]")
for name, ext, layer in [
        ("E1_LR_L20",       lr,    20),
        ("E2_Ridge_L20",    ridge, 20),
        ("E3_MeanDiff_L20", mdiff, 20),
        ("E4_PCA_L20",      pca1,  20),
        ("ExL12_LR",        lr,    12),
        ("ExL16_LR",        lr,    16),
        ("ExL24_LR",        lr,    24)]:
    d, mu, sd = cv_extract(ext, X[layer], y_A, X_eval=X[20])
    registry.append((name, d, "evidence" if layer == 20 else "evidence_x",
                     layer, mu, sd))
    print(f"  {name:<16s} L{layer:>2d}  OOF AUROC={mu:.3f} ± {sd:.3f}  "
          f"cos·A={float(np.dot(d, A_hat)):+.3f}")

print("\n[operative / inert / reference — fixed directions]")
def fixed_auroc(d):
    s = X[20] @ d
    return max(roc_auc_score(y_A, s), 1 - roc_auc_score(y_A, s)), 0.0
for name, d, fam in [
        ("O1_D3prime",    D3p_raw,   "operative"),
        ("O2_D1_source",  D1_raw,    "operative"),
        ("O3_joint",      joint_raw, "operative"),
        ("I1_D2bal",      D2b_raw,   "inert"),
        ("I2_D4_obslen",  D4_raw,    "inert"),
        ("A_L20",         unit(A),   "reference")]:
    mu, sd = fixed_auroc(d)
    registry.append((name, d, fam, 20, float(mu), float(sd)))
    print(f"  {name:<16s} L20  AUROC={mu:.3f}  cos·A={float(np.dot(d, A_hat)):+.3f}")

# ── random null cloud ───────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
print(f"\n[random null cloud K={K_RAND}]")
for k in range(K_RAND):
    g = rng.standard_normal(len(A_hat)).astype(np.float32)
    d = perp(g)
    mu, _ = fixed_auroc(d)
    registry.append((f"rand{k:02d}", d, "random", 20, float(mu), 0.0))

# ── persist ─────────────────────────────────────────────────────────────
direction_arrs, perp_arrs, summary = {"A_hat": A_hat}, {}, {}
for name, d, fam, layer, mu, sd in registry:
    direction_arrs[name] = d
    cos_A = float(np.dot(d, A_hat))
    perp_arrs[name + "_perp"] = perp(d) if abs(cos_A) < 0.999 else d
    summary[name] = dict(family=fam, layer=int(layer),
                         cos_A=cos_A, oof_auroc=mu, oof_auroc_std=sd)
np.savez(OUT / "directions.npz", **direction_arrs, **perp_arrs)
json.dump({"convention": "OOF AUROC = mean of 5-fold CV; AUROC=max(s,1-s); "
                         "label = y_A (Method A, n_sf>=1)",
           "n_samples": int(len(y_A)), "k_folds": K_FOLDS,
           "k_random": K_RAND, "directions": summary},
          open(OUT / "extract_summary.json", "w"), indent=2)
print(f"\n[saved] {OUT/'directions.npz'}\n[saved] {OUT/'extract_summary.json'}")
