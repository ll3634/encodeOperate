#!/usr/bin/env python3
"""P2.1 — Compute a 2D PCA embedding of the directions used in Figure 1.

Layout space: null(A) (i.e. the action axis is collapsed to the origin).
We project all candidate directions into null(A), L2-normalise, then run
PCA on the stack {D3'_perp, D1_perp, D4_perp, D2bal_perp, E_perp,
joint_perp} together with K=20 random unit directions in null(A) (the
exact same K=20 used by results/d3_perp_vs_random_null/) so that the
embedding axes are interpretable as "the two directions of largest
variance among the operative+null candidates".

Output: results/fig1_pca_embedding.json with 2D coordinates and
per-direction explained-variance contributions.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path("tmc/scripts/e2e_agent")
OUT = ROOT / "results" / "fig1_pca_embedding"
OUT.mkdir(parents=True, exist_ok=True)


def project_to_null(v, A_hat):
    p = v - float(np.dot(v, A_hat)) * A_hat
    return (p / np.linalg.norm(p)).astype(np.float32)


def main():
    # --- A axis ---
    A = np.load(ROOT / "steering/directions/direction_decomp_full_layer20.npz",
                allow_pickle=True)["decision_direction"].astype(np.float32)
    A_hat = A / np.linalg.norm(A)

    # --- Source directions ---
    E   = np.load(ROOT / "results/phase1_probe/probe_direction_l20.npz",
                  allow_pickle=True)["decision_direction"].astype(np.float32)
    D1  = np.load(ROOT / "results/ocft/per_candidate/D1_source/direction.npy").astype(np.float32)
    D4  = np.load(ROOT / "results/ocft/per_candidate/D4_obs_length/direction.npy").astype(np.float32)
    D2b = np.load(ROOT / "results/d2_balanced_retrain/direction_D2prime_balanced.npy").astype(np.float32)
    D3p = np.load(ROOT / "results/d3_balanced_control/direction_D3prime_no_S0.npy").astype(np.float32)

    perps = {}
    for name, v in [("D3p", D3p), ("D1", D1), ("D4", D4),
                    ("D2bal", D2b), ("E", E)]:
        perps[name] = project_to_null(v / np.linalg.norm(v), A_hat)
    # Joint
    joint_raw = perps["D3p"] + perps["D1"]
    perps["joint"] = (joint_raw / np.linalg.norm(joint_raw)).astype(np.float32)

    # --- Random nulls (same RNG_SEED & K as d3_perp_vs_random_null) ---
    K_RANDOM = 20
    RNG_SEED = 20260424
    rng = np.random.default_rng(RNG_SEED)
    randoms = []
    for k in range(K_RANDOM):
        r = rng.standard_normal(A_hat.shape[0]).astype(np.float32)
        r /= np.linalg.norm(r)
        randoms.append(project_to_null(r, A_hat))
    randoms = np.stack(randoms)

    # --- Stack matrix for PCA: (D3p, D1, joint, D4, D2bal, E) + K randoms ---
    names = ["D3p", "D1", "joint", "D4", "D2bal", "E"] + [f"r{k:02d}" for k in range(K_RANDOM)]
    X = np.stack([perps["D3p"], perps["D1"], perps["joint"],
                  perps["D4"], perps["D2bal"], perps["E"]] + list(randoms))
    print(f"X shape = {X.shape}  (rows = directions, cols = hidden dim)")

    # --- PCA via SVD on the centred stack ---
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # Top-2 components
    coords = (U[:, :2] * S[:2])  # shape (n_dir, 2)
    var_explained = (S ** 2) / (S ** 2).sum()
    print(f"PC1 explains {var_explained[0]*100:.1f}%  PC2 {var_explained[1]*100:.1f}%  "
          f"(top-2 cumulative: {(var_explained[:2].sum())*100:.1f}%)")

    # --- Bias check: rotate so that D3p sits on positive x-axis ---
    # Find rotation R such that R @ coord(D3p) = (||coord(D3p)||, 0)
    d3_xy = coords[names.index("D3p")]
    theta = -np.arctan2(d3_xy[1], d3_xy[0])
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
    coords_rot = coords @ R.T  # rotate every point

    # --- Persist ---
    out = {
        "_meta": {
            "method": "PCA (SVD) on centred stack of {6 named + 20 random} unit perp directions in null(A)",
            "rotation": "post-PCA: rotated so that D3' lies along positive x-axis (zero rotation in the embedding plane has no semantic meaning; this convention aligns the figure with the strongest-operative direction).",
            "var_explained_PC1_pct": float(var_explained[0] * 100),
            "var_explained_PC2_pct": float(var_explained[1] * 100),
            "var_explained_PC1_PC2_cum_pct": float(var_explained[:2].sum() * 100),
            "K_RANDOM": K_RANDOM, "RNG_SEED": RNG_SEED,
        },
        "coordinates": {n: [float(x), float(y)] for n, (x, y) in zip(names, coords_rot)},
        "norms_in_full_dim": {n: float(np.linalg.norm(perps.get(n) if n in perps else randoms[int(n[1:])])) for n in names},
        "cos_with_each_other": {
            "cos(D3p, D1)":   float(np.dot(perps["D3p"], perps["D1"])),
            "cos(D3p, E)":    float(np.dot(perps["D3p"], perps["E"])),
            "cos(D3p, D4)":   float(np.dot(perps["D3p"], perps["D4"])),
            "cos(D1,  E)":    float(np.dot(perps["D1"],  perps["E"])),
            "cos(D1,  D4)":   float(np.dot(perps["D1"],  perps["D4"])),
            "cos(E,   D4)":   float(np.dot(perps["E"],   perps["D4"])),
            "cos(joint,D3p)": float(np.dot(perps["joint"], perps["D3p"])),
            "cos(joint,D1)":  float(np.dot(perps["joint"], perps["D1"])),
            "cos(D2bal,D3p)": float(np.dot(perps["D2bal"], perps["D3p"])),
            "cos(D2bal,D1)":  float(np.dot(perps["D2bal"], perps["D1"])),
            "cos(D2bal,E)":   float(np.dot(perps["D2bal"], perps["E"])),
        },
    }
    out_file = OUT / "embedding.json"
    json.dump(out, open(out_file, "w"), indent=2)
    print(f"\nSaved: {out_file}")
    print("Coordinates (rotated so D3p on +x):")
    for n, (x, y) in zip(names, coords_rot):
        print(f"  {n:>8s}: ({x:+.4f}, {y:+.4f})")


if __name__ == "__main__":
    main()
