"""Phase D-3: Decompose FT action direction into evidence-parallel /
evidence-perpendicular components, and report all key cosines.

Inputs (paths configurable):
  - evidence_dir_base : results/phase1_probe/probe_direction_l20.npz
  - evidence_dir_ft   : results/phase1_probe_ft_balanced/probe_direction_l20.npz
  - action_dir_base   : steering/directions/direction_search_v3_layer20.npz
  - action_dir_ft     : steering/directions/direction_search_v3_layer20_ft_balanced.npz

Outputs (under steering/directions/):
  - direction_decomp_full_layer20_ft.npz
  - direction_decomp_parallel_layer20_ft.npz
  - direction_decomp_perp_layer20_ft.npz
  - results/ft_phaseD/decomposition.json
"""
import json
from pathlib import Path

import numpy as np


def load_dir(path: str, key: str = "decision_direction") -> np.ndarray:
    d = np.load(path)
    return d[key].astype(np.float32)


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def project_decompose(action: np.ndarray, evidence_unit: np.ndarray):
    parallel_coef = float(np.dot(action, evidence_unit))
    parallel = parallel_coef * evidence_unit
    perp = action - parallel
    return parallel, perp, parallel_coef


def main():
    out_dir_steer = Path("steering/directions"); out_dir_steer.mkdir(parents=True, exist_ok=True)
    out_dir_res = Path("results/ft_phaseD"); out_dir_res.mkdir(parents=True, exist_ok=True)

    paths = {
        "evidence_base": "results/phase1_probe/probe_direction_l20.npz",
        "evidence_ft":   "results/phase1_probe_ft_balanced/probe_direction_l20.npz",
        "action_base":   "steering/directions/direction_search_v3_layer20.npz",
        "action_ft":     "steering/directions/direction_search_v3_layer20_ft_balanced.npz",
    }

    ev_base = load_dir(paths["evidence_base"])    # already unit-norm
    ev_ft   = load_dir(paths["evidence_ft"])      # already unit-norm
    act_base = load_dir(paths["action_base"])     # norm ~ 25.5
    act_ft   = load_dir(paths["action_ft"])       # norm ~ 25.5

    # Sanity checks on norms
    print("=== Norms ===")
    print(f"  evidence_base : {np.linalg.norm(ev_base):.4f}")
    print(f"  evidence_ft   : {np.linalg.norm(ev_ft):.4f}")
    print(f"  action_base   : {np.linalg.norm(act_base):.4f}")
    print(f"  action_ft     : {np.linalg.norm(act_ft):.4f}")

    ev_base_u = unit(ev_base)
    ev_ft_u   = unit(ev_ft)
    act_base_u = unit(act_base)
    act_ft_u   = unit(act_ft)

    # Pairwise cosines
    print("\n=== Pairwise cosines (unit vectors) ===")
    cos_evbase_evft   = float(np.dot(ev_base_u, ev_ft_u))
    cos_actbase_actft = float(np.dot(act_base_u, act_ft_u))
    cos_evbase_actbase = float(np.dot(ev_base_u, act_base_u))
    cos_evbase_actft   = float(np.dot(ev_base_u, act_ft_u))
    cos_evft_actbase   = float(np.dot(ev_ft_u,   act_base_u))
    cos_evft_actft     = float(np.dot(ev_ft_u,   act_ft_u))
    print(f"  cos(ev_base,  ev_ft)    = {cos_evbase_evft:+.4f}   (does FT rotate evidence axis?)")
    print(f"  cos(act_base, act_ft)   = {cos_actbase_actft:+.4f}   (does FT rotate action axis?)")
    print(f"  cos(ev_base,  act_base) = {cos_evbase_actbase:+.4f}  ('canonical' baseline ≈ -0.0135)")
    print(f"  cos(ev_base,  act_ft)   = {cos_evbase_actft:+.4f}   (FT action vs base evidence)")
    print(f"  cos(ev_ft,    act_base) = {cos_evft_actbase:+.4f}   (base action vs FT evidence)")
    print(f"  cos(ev_ft,    act_ft)   = {cos_evft_actft:+.4f}   (FT-FT in-system; KEY M1 vs M2)")

    # Decompose action_ft along ev_ft
    print("\n=== Decompose action_ft into (parallel || ev_ft) + (perp ⟂ ev_ft) ===")
    parallel, perp, p_coef = project_decompose(act_ft, ev_ft_u)
    p_norm = float(np.linalg.norm(parallel))
    pp_norm = float(np.linalg.norm(perp))
    full_norm = float(np.linalg.norm(act_ft))
    print(f"  parallel coef (action_ft · ev_ft_u) = {p_coef:+.6f}")
    print(f"  || parallel || = {p_norm:.4f}    || perp || = {pp_norm:.4f}")
    print(f"  || full ||     = {full_norm:.4f}  (sanity: sqrt(p^2 + pp^2) = "
          f"{np.sqrt(p_norm**2 + pp_norm**2):.4f})")
    print(f"  parallel/full  = {(p_norm/full_norm)*100:.3f}%   (energy in evidence direction)")

    # Save FT decomposition NPZs (mirror base format)
    layer = 20
    np.savez(out_dir_steer / "direction_decomp_full_layer20_ft.npz",
             decision_direction=act_ft.astype(np.float32),
             layer=layer, method="decomposition_full_ft", component="full",
             cos_with_probe_ft=float(np.dot(unit(act_ft), ev_ft_u)))
    np.savez(out_dir_steer / "direction_decomp_parallel_layer20_ft.npz",
             decision_direction=parallel.astype(np.float32),
             layer=layer, method="decomposition_parallel_ft",
             component="parallel_to_probe_ft",
             cos_with_probe_ft=float(np.dot(unit(parallel), ev_ft_u))
                if p_norm > 1e-8 else 0.0,
             original_norm=full_norm,
             component_norm=p_norm)
    np.savez(out_dir_steer / "direction_decomp_perp_layer20_ft.npz",
             decision_direction=perp.astype(np.float32),
             layer=layer, method="decomposition_perpendicular_ft",
             component="perpendicular_to_probe_ft",
             cos_with_probe_ft=float(np.dot(unit(perp), ev_ft_u)),
             original_norm=full_norm,
             component_norm=pp_norm)

    summary = {
        "norms": {k: float(np.linalg.norm(load_dir(v))) for k, v in paths.items()},
        "cosines": {
            "cos(ev_base,  ev_ft)":     cos_evbase_evft,
            "cos(act_base, act_ft)":    cos_actbase_actft,
            "cos(ev_base,  act_base)":  cos_evbase_actbase,
            "cos(ev_base,  act_ft)":    cos_evbase_actft,
            "cos(ev_ft,    act_base)":  cos_evft_actbase,
            "cos(ev_ft,    act_ft)":    cos_evft_actft,
        },
        "decomposition_action_ft_along_ev_ft": {
            "parallel_coef": p_coef,
            "parallel_norm": p_norm,
            "perp_norm":     pp_norm,
            "full_norm":     full_norm,
            "parallel_share_pct": (p_norm / full_norm) * 100,
        },
        "files_written": [
            "steering/directions/direction_decomp_full_layer20_ft.npz",
            "steering/directions/direction_decomp_parallel_layer20_ft.npz",
            "steering/directions/direction_decomp_perp_layer20_ft.npz",
        ],
    }
    out_path = out_dir_res / "decomposition.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
