#!/usr/bin/env python3
"""Random-direction erasure control + figure data for Evidence Erasure Test.

Reuses cached baseline / erase_{E,A} / flip_{E,A} margins from
results/evidence_erasure_test/per_prompt_margins.npz.

Adds erasure + flip on:
  - D1..D4 from results/ocft/per_candidate/*/direction.npy
  - K=20 random unit directions in R^3584 (seed=42),
    each with |cos(r_k, A)| < 0.05 and |cos(r_k, E)| < 0.05.

Writes:
  results/evidence_erasure_test/random_control/random_directions.npy
  results/evidence_erasure_test/random_control/per_direction_results.json
  results/evidence_erasure_test/random_control/random_erasure_report.md
  results/evidence_erasure_test/figure_scatter_AvsE.json
  results/evidence_erasure_test/figure_spectrum.json
  results/evidence_erasure_test/figure_strip.json
"""
import json, time
from pathlib import Path
import numpy as np
import sys, torch
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS
from evidence_erasure_test import (ProjectionFlipHook, build_p0_prompt,
                                   forward_margin, LAYER)

K_RANDOM = 20
RNG_SEED = 42
ORTHO_THRESH = 0.05
N_BOOT = 2000
BOOT_SEED = 20260502

OUT_TEST = Path("results/evidence_erasure_test")
OUT      = OUT_TEST / "random_control"; OUT.mkdir(parents=True, exist_ok=True)


def sample_random_unit(rng, dim, n_check):
    while True:
        v = rng.standard_normal(dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-12
        bad = False
        for d in n_check:
            if abs(float(v @ d)) > ORTHO_THRESH:
                bad = True; break
        if not bad:
            return v


def boot_mean_ci(x, B=N_BOOT, level=95.0, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(x.mean()), float(lo), float(hi)


def load_directions():
    E = np.load("results/phase1_probe/probe_direction_l20.npz",
                allow_pickle=True)["decision_direction"].astype(np.float32)
    A = np.load("steering/directions/direction_decomp_full_layer20.npz",
                allow_pickle=True)["decision_direction"].astype(np.float32)
    E_hat = E / (np.linalg.norm(E) + 1e-12)
    A_hat = A / (np.linalg.norm(A) + 1e-12)
    named = {"E": E_hat, "A": A_hat}
    for d, name in [("D1_source", "D1"), ("D2_action_prior", "D2"),
                    ("D3_candidate_present", "D3"), ("D4_obs_length", "D4")]:
        v = np.load(f"results/ocft/per_candidate/{d}/direction.npy").astype(np.float32)
        v /= np.linalg.norm(v) + 1e-12
        named[name] = v
    return named


def rebuild_prompts(tok, sample_ids):
    label_data = [json.loads(l) for l in open("results/phase1_probe/labels.jsonl")]
    by_id = {ld["sample_id"]: ld for ld in label_data}
    bl = {}
    with open("results/l20_rho020_n500/baseline_results.jsonl") as f:
        for line in f:
            ep = json.loads(line); bl[ep["sample_id"]] = ep
    prompts = []
    for sid in sample_ids:
        ld = by_id[sid]; ep = bl[sid]; s0 = ep["steps"][0]
        prompts.append(build_p0_prompt(
            tok, ld["question"], s0["action_input"], s0["observation"]))
    return prompts


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache = np.load(OUT_TEST / "per_prompt_margins.npz")
    sample_ids = [str(s) for s in cache["sample_ids"]]
    base = cache["baseline"].astype(np.float32)
    cached = {c: cache[c].astype(np.float32) for c in
              ["baseline", "erase_E", "flip_E", "erase_A", "flip_A"]}
    n = len(sample_ids)
    print(f"[cache] N={n} sample_ids loaded; baseline mean={base.mean():+.3f}")

    named = load_directions()
    E_hat, A_hat = named["E"], named["A"]
    print(f"  cos(E,A)={float(E_hat @ A_hat):+.4f}")
    for nm in ["D1", "D2", "D3", "D4"]:
        d = named[nm]
        print(f"  {nm}: |cos(·,A)|={abs(float(d@A_hat)):.4f}  "
              f"|cos(·,E)|={abs(float(d@E_hat)):.4f}")

    rng = np.random.default_rng(RNG_SEED)
    randoms = []
    n_resample = 0
    while len(randoms) < K_RANDOM:
        attempts_before = n_resample
        v = rng.standard_normal(3584).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-12
        if abs(float(v @ A_hat)) > ORTHO_THRESH or abs(float(v @ E_hat)) > ORTHO_THRESH:
            n_resample += 1; continue
        randoms.append(v)
    R = np.stack(randoms, 0)
    np.save(OUT / "random_directions.npy", R)
    print(f"[random] K={K_RANDOM} unit dirs (seed={RNG_SEED}); "
          f"resamples={n_resample}; "
          f"max|cos·A|={max(abs(float(R[k]@A_hat)) for k in range(K_RANDOM)):.4f}; "
          f"max|cos·E|={max(abs(float(R[k]@E_hat)) for k in range(K_RANDOM)):.4f}")

    print("\n[load] Qwen/Qwen2.5-7B-Instruct")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    prompts = rebuild_prompts(tok, sample_ids)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True).eval()
    tool_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]

    # Build the list of directions to run NEW erasure on
    new_dirs = [("D1", named["D1"]), ("D2", named["D2"]),
                ("D3", named["D3"]), ("D4", named["D4"])]
    new_dirs += [(f"r_{k+1:02d}", R[k]) for k in range(K_RANDOM)]
    print(f"[plan] new dirs: {len(new_dirs)} × 2 cond × {n} = "
          f"{len(new_dirs)*2*n} forwards")

    new_margins = {}
    t0 = time.time()
    for di, (name, vec) in enumerate(new_dirs):
        m_e = np.zeros(n, dtype=np.float32)
        m_f = np.zeros(n, dtype=np.float32)
        for i, p in enumerate(prompts):
            m_e[i] = forward_margin(model, tok, p,
                lambda v=vec: ProjectionFlipHook(model, v, factor=1.0),
                tool_ids, fin_ids)
            m_f[i] = forward_margin(model, tok, p,
                lambda v=vec: ProjectionFlipHook(model, v, factor=2.0),
                tool_ids, fin_ids)
        new_margins[name] = {"erase": m_e, "flip": m_f}
        elapsed = time.time() - t0
        eta = elapsed / (di + 1) * (len(new_dirs) - di - 1)
        print(f"  [{di+1:>2d}/{len(new_dirs)}] {name:>5s}  "
              f"|Δm_e|={abs((m_e-base).mean()):.4f}  "
              f"|Δm_f|={abs((m_f-base).mean()):.4f}  ETA={eta:.0f}s")

    np.savez(OUT / "new_margins.npz", base=base,
             **{f"{n}_{c}": new_margins[n][c] for n in new_margins for c in ("erase","flip")})
    from random_erasure_io import analyse_and_write
    analyse_and_write(cached, new_margins, named, R, sample_ids, n_resample,
                      OUT_TEST, OUT)


if __name__ == "__main__":
    main()
