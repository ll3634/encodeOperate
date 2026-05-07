#!/usr/bin/env python3
"""Test A_revised: D3'_no_S0_perp / D1_perp vs random null distribution.

For X in {D3'_no_S0, D1}:
  X_perp = X - (X . A_hat) A_hat       (project onto null(A))
  X_perp /= ||X_perp||                  (renormalize to unit)

K=20 random unit vectors r_k -> r_k_perp = r_k - (r_k . A_hat) A_hat,
renormalized.  All directions are unit-norm vectors strictly in null(A).

Inject each direction at L20 last-token p0 with flip factor=2.0 on the
locked Figure 1 cohort (N=100 prompts; baseline reused from
results/evidence_erasure_test/per_prompt_margins.npz).

Decision rule (per X):
  X_perp |signed mean dm| > p95(random null) AND > 5x mean(random null)
    -> GENUINE_OPERATIVE
  X_perp |signed mean dm| within [p5, p95] of random null
    -> NOT_DISTINGUISHABLE_FROM_RANDOM
  otherwise -> PARTIAL
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

OUT = Path("results/d3_perp_vs_random_null"); OUT.mkdir(parents=True, exist_ok=True)
K_RANDOM = 20
RNG_SEED = 20260424


def project_to_null(v, A_hat):
    p = v - float(np.dot(v, A_hat)) * A_hat
    return (p / np.linalg.norm(p)).astype(np.float32)


def main():
    print(f"[init] L{LAYER} N={N} K_RANDOM={K_RANDOM} RNG_SEED={RNG_SEED}")

    # ---- Load directions ----
    A = np.load("steering/directions/direction_decomp_full_layer20.npz",
                allow_pickle=True)["decision_direction"].astype(np.float32)
    A_hat = A / np.linalg.norm(A)

    D3 = np.load("results/d3_balanced_control/direction_D3prime_no_S0.npy"
                 ).astype(np.float32)
    D1 = np.load("results/ocft/per_candidate/D1_source/direction.npy"
                 ).astype(np.float32)

    X_dirs = {
        "D3prime_no_S0": D3 / np.linalg.norm(D3),
        "D1_source":     D1 / np.linalg.norm(D1),
    }

    # ---- Build perp directions ----
    perps = {}
    for name, X in X_dirs.items():
        cos_with_A = float(np.dot(X, A_hat))
        Xp = project_to_null(X, A_hat)
        perps[name] = Xp
        # sanity: cos with A should be ~0
        assert abs(float(np.dot(Xp, A_hat))) < 1e-5, "perp not orthogonal"
        print(f"  {name}: original cos_with_A={cos_with_A:+.5f}  "
              f"perp cos_with_A={float(np.dot(Xp, A_hat)):+.2e}")

    # ---- Random unit directions in null(A) ----
    rng = np.random.default_rng(RNG_SEED)
    randoms = []
    for k in range(K_RANDOM):
        r = rng.standard_normal(A_hat.shape[0]).astype(np.float32)
        r /= np.linalg.norm(r)
        randoms.append(project_to_null(r, A_hat))
    randoms = np.stack(randoms)
    print(f"  {K_RANDOM} random_perp directions: dim={randoms.shape[1]}")

    # Save direction matrix
    np.savez(OUT / "directions.npz",
             D3prime_no_S0_perp=perps["D3prime_no_S0"],
             D1_source_perp=perps["D1_source"],
             random_perp=randoms,
             A_hat=A_hat)

    # ---- Build prompts (same locked cohort) ----
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\n[load] Qwen/Qwen2.5-7B-Instruct")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    labels = [json.loads(l) for l in open("results/phase1_probe/labels.jsonl")]
    bl_map = {}
    with open("results/l20_rho020_n500/baseline_results.jsonl") as f:
        for line in f: ep = json.loads(line); bl_map[ep["sample_id"]] = ep
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
    cached_sids = list(cached["sample_ids"])
    assert sample_ids == cached_sids, "prompt cohort mismatch"
    base_cached = cached["baseline"].astype(np.float32)
    print(f"[prompts] N={len(prompts)} (matched cached cohort)")

    # ---- Forward passes ----
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True).eval()
    tool_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]

    cond_specs = [("D3prime_no_S0_perp", perps["D3prime_no_S0"]),
                  ("D1_source_perp",     perps["D1_source"])]
    for k in range(K_RANDOM):
        cond_specs.append((f"random_perp_{k:02d}", randoms[k]))

    n = len(prompts)
    margins = {c: np.zeros(n, dtype=np.float32) for c, _ in cond_specs}
    t0 = time.time()
    for i, p in enumerate(prompts):
        for c, d in cond_specs:
            hf = lambda d=d: ProjectionFlipHook(model, d, factor=2.0)  # flip
            margins[c][i] = forward_margin(model, tok, p, hf, tool_ids, fin_ids)
        if (i + 1) % 10 == 0 or i == 0:
            eta = (time.time() - t0) / (i + 1) * (n - i - 1)
            print(f"  [{i+1:>3d}/{n}]  ETA={eta:.0f}s")
    print(f"[done] forwards: {time.time() - t0:.0f}s")

    np.savez(OUT / "per_prompt_margins.npz",
             sample_ids=np.array(sample_ids), baseline=base_cached,
             **{c: margins[c] for c, _ in cond_specs})

    # ---- Stats ----
    def stats(m_cond, base):
        dm = (m_cond - base).astype(np.float32)
        signed = float(dm.mean())
        m_abs, lo_abs, hi_abs = boot_mean_ci(np.abs(dm))
        rng_b = np.random.default_rng(SEED); B = 2000
        idx = rng_b.integers(0, len(dm), size=(B, len(dm)))
        sm = np.abs(dm[idx].mean(axis=1))
        lo_sm, hi_sm = np.percentile(sm, [2.5, 97.5])
        return {"signed_mean_dm": signed,
                "abs_signed_mean_dm": abs(signed),
                "abs_signed_mean_dm_ci": [float(lo_sm), float(hi_sm)],
                "mean_abs_dm": float(m_abs),
                "mean_abs_dm_ci": [float(lo_abs), float(hi_abs)]}

    per_dir = {c: stats(margins[c], base_cached) for c, _ in cond_specs}
    rand_signed = np.array([per_dir[f"random_perp_{k:02d}"]["abs_signed_mean_dm"]
                            for k in range(K_RANDOM)])
    rand_meanabs = np.array([per_dir[f"random_perp_{k:02d}"]["mean_abs_dm"]
                             for k in range(K_RANDOM)])

    null = {
        "K": K_RANDOM,
        "abs_signed_mean_dm": {
            "mean": float(rand_signed.mean()), "std": float(rand_signed.std()),
            "p5": float(np.percentile(rand_signed, 5)),
            "p25": float(np.percentile(rand_signed, 25)),
            "p50": float(np.percentile(rand_signed, 50)),
            "p75": float(np.percentile(rand_signed, 75)),
            "p95": float(np.percentile(rand_signed, 95)),
            "max": float(rand_signed.max()),
            "all_values": rand_signed.tolist(),
        },
        "mean_abs_dm": {
            "mean": float(rand_meanabs.mean()), "std": float(rand_meanabs.std()),
            "p5": float(np.percentile(rand_meanabs, 5)),
            "p95": float(np.percentile(rand_meanabs, 95)),
            "max": float(rand_meanabs.max()),
            "all_values": rand_meanabs.tolist(),
        },
    }

    def verdict(x_val, null_p5, null_p95, null_mean):
        if x_val > null_p95 and x_val > 5 * null_mean: return "GENUINE_OPERATIVE"
        if null_p5 <= x_val <= null_p95: return "NOT_DISTINGUISHABLE_FROM_RANDOM"
        return "PARTIAL"

    out = {"K": K_RANDOM, "N": n,
           "convention": "abs_signed_mean_dm matches figure_spectrum.json |Δm_flip|",
           "directions": {
               "D3prime_no_S0_perp": per_dir["D3prime_no_S0_perp"],
               "D1_source_perp":     per_dir["D1_source_perp"],
           },
           "random_null": null,
           "decision": {}}
    for X in ("D3prime_no_S0_perp", "D1_source_perp"):
        x = per_dir[X]["abs_signed_mean_dm"]
        m = null["abs_signed_mean_dm"]
        pct = float((rand_signed < x).mean() * 100)
        v = verdict(x, m["p5"], m["p95"], m["mean"])
        out["decision"][X] = {
            "x_abs_signed_mean_dm": x,
            "null_mean": m["mean"], "null_p5": m["p5"], "null_p95": m["p95"],
            "null_max": m["max"], "x_percentile_in_null": pct,
            "ratio_x_over_null_mean": x / m["mean"] if m["mean"] > 0 else float("inf"),
            "verdict": v,
        }
    json.dump(out, open(OUT / "results.json", "w"), indent=2)

    print("\n[null distribution: |signed mean Δm_flip|]")
    print(f"  K={K_RANDOM}  mean={null['abs_signed_mean_dm']['mean']:.4f}  "
          f"std={null['abs_signed_mean_dm']['std']:.4f}")
    print(f"  p5={null['abs_signed_mean_dm']['p5']:.4f}  "
          f"p50={null['abs_signed_mean_dm']['p50']:.4f}  "
          f"p95={null['abs_signed_mean_dm']['p95']:.4f}  "
          f"max={null['abs_signed_mean_dm']['max']:.4f}")
    print("\n[decision]")
    for X, d in out["decision"].items():
        print(f"  {X}: |signed Δm_flip|={d['x_abs_signed_mean_dm']:.4f}  "
              f"({d['ratio_x_over_null_mean']:.1f}x null mean, "
              f"percentile={d['x_percentile_in_null']:.0f}%)")
        print(f"    -> {d['verdict']}")


if __name__ == "__main__":
    main()
