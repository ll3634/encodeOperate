#!/usr/bin/env python3
"""Rerun the E->D3 rotation path with D3'_no_S0 endpoint for Figure 3 / Figure 1
consistency.

Reuses construction logic from `nullspace_rotation_io.construct_family` and
the additive-injection helper from `nullspace_injection_run`.
Reuses cached cohort (sample_ids, baseline) from
`results/evidence_erasure_test/nullspace_injection/per_prompt_margins.npz`,
which is identical to the §3 / decomposition_ci_null cohort (N=100).

Output:
  results/figure3_rotation_d3prime/
    new_directions_d3prime.npz
    new_injection_results.json
    figure3_unified.json
    sanity_check_report.json
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from nullspace_rotation_io import (load_directions as load_dirs_orig,
                                   perp_component, _u, ANGLES_DEG)
from nullspace_injection_run import (LAYER, RHO, HIDDEN_RMS, to_rms1,
                                     inject_margin, margin_from_logits)
from agent.prompts import ACTION_TOKENS
from dose_response_erasure import rebuild_prompts

OUT = Path("results/figure3_rotation_d3prime")
OUT.mkdir(parents=True, exist_ok=True)

D3PRIME_PATH = "results/d3_balanced_control/direction_D3prime_no_S0.npy"
CACHED_INJ = "results/evidence_erasure_test/nullspace_injection/per_prompt_margins.npz"
CACHED_FIG = "results/evidence_erasure_test/nullspace_injection/figure_injection_rotation.json"


def build_d3prime_family(A, E, D3p):
    """Construct E(theta) for E->D3'_no_S0 using same formula as nullspace_rotation_io."""
    c = float(E @ A)
    E_perp = perp_component(E, A)
    E_perp_hat = _u(E_perp)
    sqrt_term = math.sqrt(max(0.0, 1.0 - c * c))
    raw = _u(perp_component(D3p, A))
    cos_to_E = float(E_perp_hat @ raw)
    t_orth = raw - cos_to_E * E_perp_hat
    t_orth = t_orth - float(t_orth @ A) * A
    X_hat = _u(t_orth)
    family = {}
    for th in ANGLES_DEG:
        rad = math.radians(th)
        v = c * A + sqrt_term * (math.cos(rad) * E_perp_hat + math.sin(rad) * X_hat)
        family[th] = v
    meta = {"c": c, "sqrt_one_minus_c2": sqrt_term,
            "cos_raw_target_with_E_perp": cos_to_E,
            "endpoint": "D3prime_no_S0"}
    return family, meta, X_hat


def main():
    t0 = time.time()
    # ---- 1. Load directions (A, E, D1, D3 original, plus new D3'_no_S0) ----
    dirs = load_dirs_orig()  # uses float64
    A, E = dirs["A"], dirs["E"]
    D3p = _u(np.load(D3PRIME_PATH).astype(np.float64))
    print(f"[geom] cos(D3'_no_S0, D3_orig)  = {float(D3p @ dirs['D3']):+.5f}")
    print(f"[geom] cos(D3'_no_S0, A)         = {float(D3p @ A):+.5f}")
    print(f"[geom] cos(D3'_no_S0, E)         = {float(D3p @ E):+.5f}")

    family, meta, X_hat = build_d3prime_family(A, E, D3p)
    print(f"[meta] c=cos(E,A)={meta['c']:+.6f}  sqrt(1-c^2)={meta['sqrt_one_minus_c2']:.6f}")
    print(f"[meta] cos(raw D3'_perp, E_perp_hat)={meta['cos_raw_target_with_E_perp']:+.5f}")

    # Verify construction: ||v||=1, cos(v,A)=c
    target_c = meta["c"]
    max_dev = 0.0
    for th, v in family.items():
        nrm = float(np.linalg.norm(v))
        cA = float(v @ A)
        max_dev = max(max_dev, abs(cA - target_c))
        assert abs(nrm - 1.0) < 1e-9, f"theta{th} norm={nrm}"
    print(f"[verify] max |cos(v,A)-c| = {max_dev:.2e} (target <1e-3)")
    if max_dev >= 1e-3:
        sys.exit(f"STOP: construction off")

    # Save constructed directions (float32 for storage)
    arr = np.stack([family[th].astype(np.float32) for th in ANGLES_DEG], axis=0)
    np.savez_compressed(OUT / "new_directions_d3prime.npz",
                        thetas=np.array(ANGLES_DEG, dtype=np.int32),
                        vectors=arr,
                        c_target=target_c, meta=json.dumps(meta))

    # ---- 2. Cached cohort + baseline ----
    inj_cache = np.load(ROOT / CACHED_INJ, allow_pickle=True)
    sample_ids = [str(s) for s in inj_cache["sample_ids"]]
    base = inj_cache["baseline"].astype(np.float32)
    cached_E_theta0 = inj_cache["E_theta0"].astype(np.float32)
    cached_dm_E0 = float((cached_E_theta0 - base).mean())
    print(f"[cache] N={len(sample_ids)}  baseline mean={base.mean():+.4f}  "
          f"cached E_theta0 dm={cached_dm_E0:+.4f}")

    # Theta=0 reproducibility: by construction it is identical to cached E_theta0
    # (because c, E_perp_hat, sqrt_term, cos(0)=1, sin(0)=0 -> v = c*A + sqrt*E_perp_hat = E_hat)
    cos_theta0 = float(family[0] @ (cached_E_theta0_dir := _u(np.load(
        ROOT / "results/phase1_probe/probe_direction_l20.npz",
        allow_pickle=True)["decision_direction"].astype(np.float64))))
    print(f"[sanity] cos(family[0], E_hat) = {cos_theta0:.10f}  (must be ~1.0)")

    # ---- 3. Load model and run 6 NEW directions (theta=15..90) ----
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("\n[load] Qwen/Qwen2.5-7B-Instruct")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    prompts = rebuild_prompts(tok, sample_ids)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True).eval()
    tool_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]

    new_thetas = [th for th in ANGLES_DEG if th != 0]
    margins = {}
    tg = time.time()
    for k, th in enumerate(new_thetas):
        vec_rms1 = to_rms1(family[th].astype(np.float32))
        per = np.empty(len(prompts), dtype=np.float32)
        for i, p in enumerate(prompts):
            per[i] = inject_margin(model, tok, p, vec_rms1, RHO, tool_ids, fin_ids)
        margins[th] = per
        eta = (time.time() - tg) / (k + 1) * (len(new_thetas) - k - 1)
        print(f"  [{k+1}/{len(new_thetas)}] theta={th:>2d}  mean_dm={float((per-base).mean()):+.4f}  ETA={eta:.0f}s")
    margins[0] = cached_E_theta0  # reuse cached for theta=0
    print(f"[gpu] {time.time()-tg:.0f}s")

    # Save per-prompt margins
    np.savez_compressed(OUT / "per_prompt_margins.npz",
        sample_ids=np.array(sample_ids), baseline=base,
        **{f"theta{th:02d}": margins[th] for th in ANGLES_DEG})

    # ---- 4. Bootstrap CI per theta ----
    rng = np.random.default_rng(12345)
    N = len(base)
    points = []
    for th in ANGLES_DEG:
        dm = margins[th] - base
        n_boot = 2000
        boot_means = np.array([dm[rng.integers(0, N, N)].mean() for _ in range(n_boot)])
        ci_lo, ci_hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
        boot_abs = np.array([np.abs(dm[rng.integers(0, N, N)]).mean() for _ in range(n_boot)])
        ci_alo, ci_ahi = float(np.percentile(boot_abs, 2.5)), float(np.percentile(boot_abs, 97.5))
        points.append({"theta": int(th), "dm": float(dm.mean()), "dm_ci": [ci_lo, ci_hi],
                       "abs_mean_dm": float(abs(dm.mean())),
                       "mean_abs_dm": float(np.abs(dm).mean()),
                       "mean_abs_dm_ci": [ci_alo, ci_ahi],
                       "flip_rate": float(np.mean((margins[th] > 0) != (base > 0)))})
    json.dump({"path": "E_to_D3prime_no_S0", "constant_cos_D_A": target_c,
               "rho": RHO, "N": N, "points": points,
               "endpoint_meta": meta},
              open(OUT / "new_injection_results.json", "w"), indent=2, default=float)

    # ---- 5. Sanity report ----
    cached_fig = json.load(open(ROOT / CACHED_FIG))
    cached_E_to_D1 = {p["theta"]: p for p in cached_fig["paths"]["E_to_D1"]}
    cached_E_to_random = {p["theta"]: p for p in cached_fig["paths"]["E_to_random"]}
    cached_E_to_D3 = {p["theta"]: p for p in cached_fig["paths"]["E_to_D3"]}
    new_p0 = points[0]["dm"]
    sanity = {
        "theta_0_check": {
            "new_dm_E_theta0": new_p0,
            "cached_E_to_D1_theta0_dm": cached_E_to_D1[0]["dm"],
            "cached_E_to_random_theta0_dm": cached_E_to_random[0]["dm"],
            "max_abs_diff": max(abs(new_p0 - cached_E_to_D1[0]["dm"]),
                                abs(new_p0 - cached_E_to_random[0]["dm"])),
            "tolerance": 0.02,
            "passes": max(abs(new_p0 - cached_E_to_D1[0]["dm"]),
                          abs(new_p0 - cached_E_to_random[0]["dm"])) < 0.02,
            "notes": "By reuse of cached E_theta0 array, exact match expected.",
        },
        "theta_90_check": {
            "new_dm_E_to_D3prime_theta90": points[-1]["dm"],
            "cached_E_to_D3orig_theta90_dm": cached_E_to_D3[90]["dm"],
            "cached_E_to_D1_theta90_dm": cached_E_to_D1[90]["dm"],
            "expected_band": [-1.0, -0.4],
            "in_band": -1.0 <= points[-1]["dm"] <= -0.4,
        },
    }
    json.dump(sanity, open(OUT / "sanity_check_report.json", "w"), indent=2, default=float)

    # ---- 6. Unified figure3 JSON ----
    unified = {
        "paths": {
            "E_to_D3prime": {
                "thetas": [int(p["theta"]) for p in points],
                "dm_means": [p["dm"] for p in points],
                "ci_lows": [p["dm_ci"][0] for p in points],
                "ci_highs": [p["dm_ci"][1] for p in points],
                "source": "new (D3'_no_S0 endpoint, this run)",
            },
            "E_to_D1": {
                "thetas": [p["theta"] for p in cached_fig["paths"]["E_to_D1"]],
                "dm_means": [p["dm"] for p in cached_fig["paths"]["E_to_D1"]],
                "ci_lows": [p["dm_ci"][0] for p in cached_fig["paths"]["E_to_D1"]],
                "ci_highs": [p["dm_ci"][1] for p in cached_fig["paths"]["E_to_D1"]],
                "source": "cached from figure_injection_rotation.json",
            },
            "E_to_random": {
                "thetas": [p["theta"] for p in cached_fig["paths"]["E_to_random"]],
                "dm_means": [p["dm"] for p in cached_fig["paths"]["E_to_random"]],
                "ci_lows": [p["dm_ci"][0] for p in cached_fig["paths"]["E_to_random"]],
                "ci_highs": [p["dm_ci"][1] for p in cached_fig["paths"]["E_to_random"]],
                "source": "cached from figure_injection_rotation.json",
            },
        },
        "constants": {"cos_A_target": target_c, "rho": RHO, "N": N,
                      "operator": "additive injection RMS-matched (alpha = rho * HIDDEN_RMS / d_rms)",
                      "hidden_rms": HIDDEN_RMS, "layer": LAYER},
        "references": cached_fig.get("reference_lines", {}),
    }
    json.dump(unified, open(OUT / "figure3_unified.json", "w"), indent=2, default=float)
    print(f"\n[done] {time.time()-t0:.0f}s total")
    print(f"  theta=0 sanity diff = {sanity['theta_0_check']['max_abs_diff']:.4e}  passes={sanity['theta_0_check']['passes']}")
    print(f"  theta=90 dm = {points[-1]['dm']:+.4f}  in expected band={sanity['theta_90_check']['in_band']}")


if __name__ == "__main__":
    main()
