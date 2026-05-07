#!/usr/bin/env python3
"""Community-aligned operative baselines at L20 for Figure 1.

Extracts three standard directions at L20 using the same labeled activations
as the rest of the paper (phase1_probe/activations_multilayer.npz):

  O_caa_l20   : Mean-Diff (CAA)  Rimsky et al. ACL 2024
                 mean(high-margin acts) - mean(low-margin acts)  at L20
                 (splits on margin_before, bottom/top 20th percentile)

  O_repE_l20  : PCA  (RepE)  Zou et al. 2023
                 PC1 of per-pair difference vectors (high - low)

  O_itiprobe  : LR probe (ITI-style)  Li et al. 2023
                 LR coef trained on binary margin label at L20
                 (label = 1 if margin_before > median, else 0)

Then projects each into null(A) (perp protocol, factor=2.0, N=100)
and measures |Δm|, exactly matching fig1_v3_run_perp.py.

All three should be operative if the action subspace is genuinely encoded at L20.
Cross-check: cos(each, A) ≈ 0 after perp projection (by construction).
"""
import json, time, sys, os
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from agent.prompts import ACTION_TOKENS
from evidence_erasure_test import (
    ProjectionFlipHook, forward_margin, build_p0_prompt, LAYER, SEED, N,
)

OUT = ROOT / "results/fig1_O_community"; OUT.mkdir(parents=True, exist_ok=True)


def unit(v):
    v = np.asarray(v, np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


def perp_to_A(v, A_hat):
    return unit(v - float(np.dot(v, A_hat)) * A_hat)


def build_l20_directions(A_hat):
    """Compute three community-standard L20 directions."""
    acts = np.load(ROOT / "results/phase1_probe/activations_multilayer.npz",
                   allow_pickle=True)
    X = acts["layer_20"].astype(np.float32)          # (N, D)
    labels = [json.loads(l)
              for l in open(ROOT / "results/phase1_probe/labels.jsonl")]
    margins = np.array([l["margin_before"] for l in labels], dtype=np.float32)

    # percentile split (matching search_v3 protocol)
    p20, p80 = np.percentile(np.abs(margins), [20, 80])
    idx_low  = np.where(np.abs(margins) <= p20)[0]   # weak stop bias
    idx_high = np.where(np.abs(margins) >= p80)[0]   # strong stop bias
    print(f"  margin split: low N={len(idx_low)} (|m|<={p20:.2f}), "
          f"high N={len(idx_high)} (|m|>={p80:.2f})")

    X_lo, X_hi = X[idx_low], X[idx_high]

    # 1. CAA / Mean-Diff  (community standard)
    d_caa = unit(X_hi.mean(0) - X_lo.mean(0))
    print(f"  CAA: cos_A={float(np.dot(d_caa, A_hat)):+.4f}")

    # 2. RepE / PCA — PC1 of paired-difference matrix
    # pair each high with a low (min-size), compute difference vectors
    n_pairs = min(len(idx_high), len(idx_low))
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx_high); rng.shuffle(idx_low)
    diff_mat = X_hi[:n_pairs] - X_lo[:n_pairs]   # (P, D)
    pca = PCA(n_components=1)
    pca.fit(diff_mat)
    d_repE = unit(pca.components_[0])
    # align sign with CAA direction
    if np.dot(d_repE, d_caa) < 0:
        d_repE = -d_repE
    print(f"  RepE: cos_A={float(np.dot(d_repE, A_hat)):+.4f}  "
          f"cos_CAA={float(np.dot(d_repE, d_caa)):+.4f}")

    # 3. ITI / LR probe  — train on binary margin label
    y_bin = (np.abs(margins) >= np.median(np.abs(margins))).astype(int)
    lr = LogisticRegression(C=1.0, max_iter=2000,
                            class_weight="balanced",
                            random_state=SEED).fit(X, y_bin)
    d_iti = unit(lr.coef_.ravel())
    if np.dot(d_iti, d_caa) < 0:
        d_iti = -d_iti
    print(f"  ITI:  cos_A={float(np.dot(d_iti, A_hat)):+.4f}  "
          f"cos_CAA={float(np.dot(d_iti, d_caa)):+.4f}")

    dirs = {
        "O_caa_l20":  perp_to_A(d_caa,  A_hat),
        "O_repE_l20": perp_to_A(d_repE, A_hat),
        "O_iti_l20":  perp_to_A(d_iti,  A_hat),
    }
    for k, v in dirs.items():
        print(f"  {k}: cos_A_after_perp={float(np.dot(v, A_hat)):+.2e}  ||v||={np.linalg.norm(v):.4f}")
    return dirs


def compute_stats(m_cond, base):
    dm = (m_cond - base).astype(np.float32)
    signed = float(dm.mean())
    rng = np.random.default_rng(SEED); B = 2000
    idx = rng.integers(0, len(dm), (B, len(dm)))
    sm = np.abs(dm[idx].mean(axis=1))
    lo, hi = np.percentile(sm, [2.5, 97.5])
    return {"signed_mean_dm": signed,
            "abs_signed_mean_dm": abs(signed),
            "abs_signed_mean_dm_ci": [float(lo), float(hi)]}


def main():
    A_hat = unit(np.load(
        ROOT / "results/fig1_v3/directions.npz",
        allow_pickle=True)["A_hat"])

    print("[build L20 directions]")
    dirs = build_l20_directions(A_hat)

    MODEL_ID = os.environ.get(
        "MODEL_ID", "/home/featurize/work/models/Qwen2.5-7B-Instruct")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    labels = [json.loads(l)
              for l in open(ROOT / "results/phase1_probe/labels.jsonl")]
    bl_map = {}
    with open(ROOT / "results/l20_rho020_n500/baseline_results.jsonl") as f:
        for line in f:
            ep = json.loads(line); bl_map[ep["sample_id"]] = ep
    prompts, sample_ids = [], []
    for ld in labels:
        ep = bl_map.get(ld["sample_id"])
        if not ep or not ep.get("steps"): continue
        s0 = ep["steps"][0]
        if not s0.get("observation"): continue
        prompts.append(build_p0_prompt(
            tok, ld["question"], s0["action_input"], s0["observation"]))
        sample_ids.append(ld["sample_id"])
        if len(prompts) >= N: break
    cached = np.load(ROOT / "results/evidence_erasure_test/per_prompt_margins.npz")
    assert sample_ids == list(cached["sample_ids"])
    base = cached["baseline"].astype(np.float32)
    print(f"\n[cohort] N={len(prompts)}  baseline mean={base.mean():+.3f}")

    from transformers import AutoModelForCausalLM
    print(f"\n[load model] {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True).eval()
    tool_ids = [tok.encode(t, add_special_tokens=False)[0]
                for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0]
                for t in ACTION_TOKENS["finish"]]

    n = len(prompts)
    margins = {k: np.zeros(n, np.float32) for k in dirs}
    t0 = time.time()
    print("\n[injections]")
    for i, p in enumerate(prompts):
        for k, v in dirs.items():
            hf = lambda v=v: ProjectionFlipHook(model, v, factor=2.0)
            margins[k][i] = forward_margin(model, tok, p, hf, tool_ids, fin_ids)
        if (i + 1) % 5 == 0 or i == 0:
            eta = (time.time() - t0) / (i + 1) * (n - i - 1)
            print(f"  [{i+1:>3d}/{n}]  ETA={eta:.0f}s")

    null_p95 = 0.1226
    out = {"N": n, "factor": 2.0, "layer": LAYER,
           "null_p95": null_p95, "directions": {}}
    rows = []
    for k, v in dirs.items():
        s = compute_stats(margins[k], base)
        s["cos_A"] = float(np.dot(v, A_hat))
        s["above_null"] = s["abs_signed_mean_dm"] > null_p95
        out["directions"][k] = s
        rows.append((k, s))

    json.dump(out, open(OUT / "results.json", "w"), indent=2)
    np.savez(OUT / "per_prompt_margins.npz",
             sample_ids=np.array(sample_ids), baseline=base,
             **{k: margins[k] for k in dirs})

    print("\n[results]")
    for k, s in sorted(rows, key=lambda x: x[1]["abs_signed_mean_dm"], reverse=True):
        flag = "** OPERATIVE **" if s["above_null"] else "  below null  "
        print(f"  {k:<14s}  |Δm|={s['abs_signed_mean_dm']:.3f}  "
              f"CI=[{s['abs_signed_mean_dm_ci'][0]:.3f},"
              f"{s['abs_signed_mean_dm_ci'][1]:.3f}]  {flag}")
    print(f"\n[saved] {OUT/'results.json'}")


if __name__ == "__main__":
    main()
