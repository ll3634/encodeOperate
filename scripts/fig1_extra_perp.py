#!/usr/bin/env python3
"""Figure 1 extension: project {D4, D2', joint(D3'+D1)} into null(A) and run
the locked flip protocol (N=100 paired, factor=2.0) so that all five
directions sit in the same coordinate convention.

Reuses cohort/baseline cache from results/evidence_erasure_test/.
Random null in null(A) is reused from results/d3_perp_vs_random_null/.
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
    boot_mean_ci, LAYER, SEED, N,
)

OUT = Path("results/fig1_extra_perp"); OUT.mkdir(parents=True, exist_ok=True)


def project_to_null(v, A_hat):
    p = v - float(np.dot(v, A_hat)) * A_hat
    return (p / np.linalg.norm(p)).astype(np.float32)


def main():
    print(f"[init] L{LAYER} N={N}")

    # Action axis
    A = np.load("steering/directions/direction_decomp_full_layer20.npz",
                allow_pickle=True)["decision_direction"].astype(np.float32)
    A_hat = A / np.linalg.norm(A)

    # Source directions (all unit-norm at file)
    D1 = np.load("results/ocft/per_candidate/D1_source/direction.npy").astype(np.float32)
    D4 = np.load("results/ocft/per_candidate/D4_obs_length/direction.npy").astype(np.float32)
    D2b = np.load("results/d2_balanced_retrain/direction_D2prime_balanced.npy").astype(np.float32)
    D3p = np.load("results/d3_balanced_control/direction_D3prime_no_S0.npy").astype(np.float32)

    # Project into null(A) and renormalise
    D1_perp  = project_to_null(D1 / np.linalg.norm(D1), A_hat)
    D4_perp  = project_to_null(D4 / np.linalg.norm(D4), A_hat)
    D2b_perp = project_to_null(D2b / np.linalg.norm(D2b), A_hat)
    D3p_perp = project_to_null(D3p / np.linalg.norm(D3p), A_hat)
    # Joint = unit-normalised mean of two perp directions
    joint_raw = D3p_perp + D1_perp
    joint = (joint_raw / np.linalg.norm(joint_raw)).astype(np.float32)

    # Sanity
    for name, v in [("D4_perp", D4_perp), ("D2bal_perp", D2b_perp),
                    ("joint_D3pD1_perp", joint)]:
        c = float(np.dot(v, A_hat))
        assert abs(c) < 1e-5, f"{name} not orthogonal to A: cos={c}"
        print(f"  {name}: cos(_,A)={c:+.2e}  ||v||={np.linalg.norm(v):.4f}")
    print(f"  cos(D3p_perp, D1_perp) = {float(np.dot(D3p_perp, D1_perp)):+.4f}")
    print(f"  cos(joint, D3p_perp)   = {float(np.dot(joint, D3p_perp)):+.4f}")
    print(f"  cos(joint, D1_perp)    = {float(np.dot(joint, D1_perp)):+.4f}")

    np.savez(OUT / "directions.npz",
             A_hat=A_hat, D1_perp=D1_perp, D4_perp=D4_perp,
             D2bal_perp=D2b_perp, D3p_perp=D3p_perp, joint=joint)

    # Cohort: same locked N=100 prompts as evidence_erasure_test
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

    cond_specs = [
        ("D4_perp",          D4_perp),
        ("D2bal_perp",       D2b_perp),
        ("joint_D3pD1_perp", joint),
    ]
    n = len(prompts)
    margins = {c: np.zeros(n, dtype=np.float32) for c, _ in cond_specs}
    t0 = time.time()
    for i, p in enumerate(prompts):
        for c, d in cond_specs:
            hf = lambda d=d: ProjectionFlipHook(model, d, factor=2.0)
            margins[c][i] = forward_margin(model, tok, p, hf, tool_ids, fin_ids)
        if (i + 1) % 10 == 0 or i == 0:
            eta = (time.time() - t0) / (i + 1) * (n - i - 1)
            print(f"  [{i+1:>3d}/{n}]  ETA={eta:.0f}s")
    print(f"[done] forwards: {time.time() - t0:.0f}s")

    np.savez(OUT / "per_prompt_margins.npz",
             sample_ids=np.array(sample_ids), baseline=base_cached,
             **{c: margins[c] for c, _ in cond_specs})

    def stats(m_cond, base):
        dm = (m_cond - base).astype(np.float32)
        signed = float(dm.mean())
        rng_b = np.random.default_rng(SEED); B = 2000
        idx = rng_b.integers(0, len(dm), size=(B, len(dm)))
        sm = np.abs(dm[idx].mean(axis=1))
        lo_sm, hi_sm = np.percentile(sm, [2.5, 97.5])
        flip_rate = float((np.sign(dm) != np.sign(base)).mean())
        return {"signed_mean_dm": signed,
                "abs_signed_mean_dm": abs(signed),
                "abs_signed_mean_dm_ci": [float(lo_sm), float(hi_sm)],
                "flip_rate_sign_change": flip_rate}

    out = {"N": n, "convention": "abs_signed_mean_dm matches figure_spectrum.json |dm_flip|",
           "directions": {c: stats(margins[c], base_cached) for c, _ in cond_specs}}
    json.dump(out, open(OUT / "results.json", "w"), indent=2)
    print("\n[results]")
    for c, s in out["directions"].items():
        print(f"  {c}: |signed dm|={s['abs_signed_mean_dm']:.4f}  "
              f"CI=[{s['abs_signed_mean_dm_ci'][0]:.4f},{s['abs_signed_mean_dm_ci'][1]:.4f}]  "
              f"flip_rate={s['flip_rate_sign_change']:.2f}")


if __name__ == "__main__":
    main()
