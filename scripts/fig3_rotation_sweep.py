#!/usr/bin/env python3
"""Figure 3 — angular sweep through null(A).

For each path (a, b) we sample θ in {0, 15, 30, 45, 60, 75, 90}, build the
unit perp direction
    v(θ) = cos(θ) · a_perp + sin(θ) · b_perp_orth     (b_perp_orth = b_perp Gram–Schmidted against a_perp)
and apply the locked perp flip protocol (factor=2.0) at L20 on the matched
N=100 cohort. All directions are unit-norm and orthogonal to A.

Paths sweep the operative subspace from operative ↔ inert and operative ↔ operative
to expose continuous angular structure (the "rotation degree-ness").
"""
import json, time, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from agent.prompts import ACTION_TOKENS
from evidence_erasure_test import (
    ProjectionFlipHook, forward_margin, build_p0_prompt,
    LAYER, SEED, N,
)

OUT = Path("results/fig3_geometry"); OUT.mkdir(parents=True, exist_ok=True)
THETAS = [0, 15, 30, 45, 60, 75, 90]


def proj_null(v, A_hat):
    p = v - float(np.dot(v, A_hat)) * A_hat
    return (p / np.linalg.norm(p)).astype(np.float32)


def gs_orth(a, b):
    """Component of b orthogonal to a, unit-normalised."""
    o = b - float(np.dot(a, b)) * a
    return (o / np.linalg.norm(o)).astype(np.float32)


def main():
    print(f"[init] L{LAYER} N={N} thetas={THETAS}")
    A = np.load("steering/directions/direction_decomp_full_layer20.npz",
                allow_pickle=True)["decision_direction"].astype(np.float32)
    A_hat = A / np.linalg.norm(A)

    E   = np.load("results/phase1_probe/probe_direction_l20.npz",
                  allow_pickle=True)["decision_direction"].astype(np.float32)
    D1  = np.load("results/ocft/per_candidate/D1_source/direction.npy").astype(np.float32)
    D4  = np.load("results/ocft/per_candidate/D4_obs_length/direction.npy").astype(np.float32)
    D3p = np.load("results/d3_balanced_control/direction_D3prime_no_S0.npy").astype(np.float32)

    perp = {n: proj_null(v / np.linalg.norm(v), A_hat) for n, v in
            [("E", E), ("D1", D1), ("D4", D4), ("D3p", D3p)]}

    # Random direction in null(A), seed fixed for reproducibility
    rng_dir = np.random.default_rng(20260424)
    r = rng_dir.standard_normal(A_hat.shape[0]).astype(np.float32)
    perp["random"] = proj_null(r / np.linalg.norm(r), A_hat)

    # Define paths: (start, end) -- start at θ=0, end at θ=90
    PATHS = [
        ("D3p_to_D1",     "D3p", "D1"),    # operative → operative (KEY: should stay HIGH)
        ("D3p_to_E",      "D3p", "E"),     # operative → inert
        ("D3p_to_D4",     "D3p", "D4"),    # operative → inert
        ("D3p_to_random", "D3p", "random"),# operative → null (control)
        ("E_to_D3p",      "E",   "D3p"),   # inert → operative (climb)
        ("E_to_D1",       "E",   "D1"),    # inert → operative (climb)
    ]

    paths_dirs = {}
    for name, a_key, b_key in PATHS:
        a, b = perp[a_key], perp[b_key]
        b_o = gs_orth(a, b)
        steps = []
        for th in THETAS:
            r = np.deg2rad(th)
            v = (np.cos(r) * a + np.sin(r) * b_o).astype(np.float32)
            v /= np.linalg.norm(v)
            assert abs(float(np.dot(v, A_hat))) < 1e-4
            steps.append(v)
        paths_dirs[name] = steps
        print(f"  {name}: cos(a,b)={float(np.dot(a, b)):+.4f}")

    # Cohort
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os
    MODEL_PATH = os.environ.get("MODEL_PATH", "/home/featurize/work/models/Qwen2.5-7B-Instruct")
    print(f"\n[load] {MODEL_PATH}")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    labels = [json.loads(l) for l in open("results/phase1_probe/labels.jsonl")]
    bl_map = {}
    with open("results/l20_rho020_n500/baseline_results.jsonl") as f:
        for line in f:
            ep = json.loads(line); bl_map[ep["sample_id"]] = ep
    prompts, sample_ids = [], []
    for ld in labels:
        ep = bl_map.get(ld["sample_id"])
        if not ep or not ep.get("steps"): continue
        s0 = ep["steps"][0]
        if not s0.get("observation"): continue
        prompts.append(build_p0_prompt(tok, ld["question"], s0["action_input"], s0["observation"]))
        sample_ids.append(ld["sample_id"])
        if len(prompts) >= N: break
    cached = np.load("results/evidence_erasure_test/per_prompt_margins.npz")
    assert sample_ids == list(cached["sample_ids"]), "cohort mismatch"
    base_cached = cached["baseline"].astype(np.float32)
    print(f"[prompts] N={len(prompts)} (matched cached cohort)")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True).eval()
    tool_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]

    n = len(prompts)
    results = {}
    rng_b = np.random.default_rng(SEED); B = 2000
    t0 = time.time()
    total = sum(len(v) for v in paths_dirs.values())
    done = 0
    for path_name, steps in paths_dirs.items():
        per_theta = []
        for th, v in zip(THETAS, steps):
            margins = np.zeros(n, dtype=np.float32)
            for i, p in enumerate(prompts):
                hf = lambda v=v: ProjectionFlipHook(model, v, factor=2.0)
                margins[i] = forward_margin(model, tok, p, hf, tool_ids, fin_ids)
            dm = margins - base_cached
            idx = rng_b.integers(0, n, size=(B, n))
            sm = np.abs(dm[idx].mean(axis=1))
            lo, hi = np.percentile(sm, [2.5, 97.5])
            per_theta.append({
                "theta_deg": th,
                "abs_signed_mean_dm": float(abs(dm.mean())),
                "ci_low": float(lo), "ci_high": float(hi),
                "flip_rate": float((np.sign(margins) != np.sign(base_cached)).mean()),
            })
            done += 1
            print(f"  [{done:>2d}/{total}] {path_name} θ={th:>2d}: "
                  f"|Δm|={abs(dm.mean()):.4f}  CI=[{lo:.3f},{hi:.3f}]  "
                  f"({(time.time()-t0):.0f}s elapsed)")
        results[path_name] = per_theta
    print(f"[done] {time.time() - t0:.0f}s total")

    json.dump({"_meta": {"N": n, "thetas": THETAS, "protocol": "perp flip x2 in null(A)"},
               "paths": results},
              open(OUT / "rotation_sweep.json", "w"), indent=2)
    print(f"\nSaved: {OUT / 'rotation_sweep.json'}")


if __name__ == "__main__":
    main()
