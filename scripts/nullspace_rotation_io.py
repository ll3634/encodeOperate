#!/usr/bin/env python3
"""Geometry construction & verification for null-space rotation scan.

E(theta) = c * A_hat + sqrt(1 - c^2) * (cos(theta) * E_perp_hat + sin(theta) * X_perp_hat)

where c = cos(E, A) = E_hat . A_hat, and X is one of {D3, D1, random_seed42}.
Returns 19 unique unit-norm directions (theta=0 shared across paths).
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

A_PATH = "steering/directions/direction_decomp_full_layer20.npz"
E_PATH = "results/phase1_probe/probe_direction_l20.npz"
D3_PATH = "results/ocft/per_candidate/D3_candidate_present/direction.npy"
D1_PATH = "results/ocft/per_candidate/D1_source/direction.npy"

ANGLES_DEG = [0, 15, 30, 45, 60, 75, 90]
PATHS = ["E_to_D3", "E_to_D1", "E_to_random"]
RANDOM_SEED = 42


def _u(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def load_directions() -> Dict[str, np.ndarray]:
    A = np.load(A_PATH, allow_pickle=True)["decision_direction"].astype(np.float64)
    E = np.load(E_PATH, allow_pickle=True)["decision_direction"].astype(np.float64)
    D3 = np.load(D3_PATH).astype(np.float64)
    D1 = np.load(D1_PATH).astype(np.float64)
    return {"A": _u(A), "E": _u(E), "D3": _u(D3), "D1": _u(D1)}


def perp_component(V: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Component of V orthogonal to A (A assumed unit-norm)."""
    return V - float(V @ A) * A


def make_random_perp(A: np.ndarray, E_perp: np.ndarray, seed: int = RANDOM_SEED) -> np.ndarray:
    """Random unit vector in null(A), additionally orthogonal to E_perp.

    This ensures the E->random rotation moves through a different
    null-space subspace than the E->D3 / E->D1 rotations.
    """
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(A.shape[0]).astype(np.float64)
    # project out A and E_perp
    r = r - float(r @ A) * A
    r = r - float(r @ E_perp) * E_perp
    return _u(r)


def construct_family(dirs: Dict[str, np.ndarray]) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict]:
    """Build E(theta) for the three paths.  Returns nested dict and a meta dict.

    For every path we form an orthonormal basis {E_perp_hat, X_orth_hat} of a
    2D subspace of null(A) by Gram-Schmidt:
        X_orth = (X - (X.A)A) - ((X - (X.A)A) . E_perp_hat) * E_perp_hat
    Then E(theta) = c*A + sqrt(1-c^2) * (cos*E_perp_hat + sin*X_orth_hat).

    Properties (exact by construction):
        ||E(theta)|| = 1 for all theta
        cos(E(theta), A) = c for all theta
        E(0) = E_hat exactly
        E(pi/2) = direction in {cos A = c} that is maximally aligned with the
                  X-component orthogonal to E in null(A).
    """
    A = dirs["A"]
    E = dirs["E"]
    c = float(E @ A)
    E_perp = perp_component(E, A)
    E_perp_hat = _u(E_perp)
    sqrt_term = math.sqrt(max(0.0, 1.0 - c * c))

    raw_targets: Dict[str, np.ndarray] = {}
    raw_targets["E_to_D3"] = _u(perp_component(dirs["D3"], A))
    raw_targets["E_to_D1"] = _u(perp_component(dirs["D1"], A))
    raw_targets["E_to_random"] = make_random_perp(A, E_perp_hat)

    targets: Dict[str, np.ndarray] = {}
    cos_to_E_perp: Dict[str, float] = {}
    cos_after_orth: Dict[str, float] = {}
    for nm, t in raw_targets.items():
        cos_to_E_perp[nm] = float(E_perp_hat @ t)
        # Gram-Schmidt against E_perp_hat
        t_orth = t - cos_to_E_perp[nm] * E_perp_hat
        # also re-project out A in case of fp drift
        t_orth = t_orth - float(t_orth @ A) * A
        t_hat = _u(t_orth)
        # Verify orthonormality of basis
        assert abs(float(t_hat @ A)) < 1e-10, f"{nm}: not orth to A"
        assert abs(float(t_hat @ E_perp_hat)) < 1e-10, f"{nm}: not orth to E_perp"
        assert abs(np.linalg.norm(t_hat) - 1.0) < 1e-10, f"{nm}: not unit"
        targets[nm] = t_hat
        cos_after_orth[nm] = float(t_hat @ raw_targets[nm])

    family: Dict[str, Dict[int, np.ndarray]] = {p: {} for p in PATHS}
    for p, X_hat in targets.items():
        for theta_deg in ANGLES_DEG:
            theta = math.radians(theta_deg)
            v = c * A + sqrt_term * (math.cos(theta) * E_perp_hat + math.sin(theta) * X_hat)
            family[p][theta_deg] = v
    meta = {
        "c": c,
        "sqrt_one_minus_c2": sqrt_term,
        "E_perp_norm": float(np.linalg.norm(E_perp)),
        "cos_raw_target_with_E_perp": cos_to_E_perp,
        "cos_orth_target_with_raw_target": cos_after_orth,
    }
    return family, meta


def verify(family: Dict[str, Dict[int, np.ndarray]],
           dirs: Dict[str, np.ndarray]) -> Tuple[List[Dict], float, bool]:
    """Returns (rows, max_cos_dev_from_c, theta0_matches_E)."""
    A, E, D3, D1 = dirs["A"], dirs["E"], dirs["D3"], dirs["D1"]
    c = float(E @ A)
    rows: List[Dict] = []
    max_dev = 0.0
    theta0_match = True
    for p in PATHS:
        for th, v in family[p].items():
            n = float(np.linalg.norm(v))
            cA = float(v @ A)
            cE = float(v @ E)
            cD3 = float(v @ D3)
            cD1 = float(v @ D1)
            dev = abs(cA - c)
            max_dev = max(max_dev, dev)
            rows.append({
                "path": p, "theta_deg": int(th),
                "norm": n, "cos_with_A": cA, "cos_dev_from_c": dev,
                "cos_with_E": cE, "cos_with_D3": cD3, "cos_with_D1": cD1,
            })
            if th == 0 and abs(cE - 1.0) > 1e-10:
                theta0_match = False
    return rows, max_dev, theta0_match


def unique_directions(family: Dict[str, Dict[int, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Map name -> direction vector.  theta=0 shared across paths as 'E_theta0'."""
    out: Dict[str, np.ndarray] = {}
    out["E_theta0"] = family["E_to_D3"][0]  # any path; identical at theta=0
    for p in PATHS:
        for th in ANGLES_DEG:
            if th == 0:
                continue
            out[f"{p}__theta{th:02d}"] = family[p][th]
    return out


if __name__ == "__main__":
    dirs = load_directions()
    family, meta = construct_family(dirs)
    rows, max_dev, t0_ok = verify(family, dirs)
    print(f"c = cos(E, A) = {meta['c']:+.6f}")
    print(f"sqrt(1-c^2)  = {meta['sqrt_one_minus_c2']:.6f}")
    print(f"theta0 matches E exactly: {t0_ok}")
    print(f"max |cos(E(theta), A) - c| = {max_dev:.3e}")
    print(f"unique directions: {len(unique_directions(family))}")
    print()
    print(f"{'path':>14s} {'theta':>5s}  {'norm':>8s} {'cosA':>9s} {'cosE':>9s} "
          f"{'cosD3':>9s} {'cosD1':>9s}")
    for r in rows:
        print(f"{r['path']:>14s} {r['theta_deg']:>5d}  "
              f"{r['norm']:.6f} {r['cos_with_A']:+.6f} {r['cos_with_E']:+.6f} "
              f"{r['cos_with_D3']:+.6f} {r['cos_with_D1']:+.6f}")
