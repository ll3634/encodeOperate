#!/usr/bin/env python3
"""New O-side baselines + Amnesic complement test for Figure 1.

Experiments
-----------
1. Amnesic complement:
   O1_amnesic = O1 (D3') with E1-subspace component removed.
   If |Δm| is preserved → O1's operative effect is independent of E1.

2. New operative baselines (independent extraction methods):
   O_sae_proj   : SAE-projected CAA direction  (mechanistic grounding)
   O_post_v1    : post-search contrast direction (different context)
   O_meandiff   : mean-diff baseline = standard CAA

3. SAE individual features (top-3):
   O_sae_f1/f2/f3

All injected with perp protocol (project into null(A), factor=2.0, N=100),
matching fig1_v3_run_perp.py exactly.
"""
import json, time, sys, os
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from agent.prompts import ACTION_TOKENS
from evidence_erasure_test import (
    ProjectionFlipHook, forward_margin, build_p0_prompt, LAYER, SEED, N,
)

OUT = ROOT / "results/fig1_new_O"; OUT.mkdir(parents=True, exist_ok=True)


def unit(v):
    v = np.asarray(v, np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


def perp_to_A(v, A_hat):
    return unit(v - float(np.dot(v, A_hat)) * A_hat)


def load_direction(path, key="decision_direction"):
    d = np.load(path, allow_pickle=True)
    return unit(d[key].astype(np.float32))


def build_directions(A_hat, E1_hat):
    """Return dict of name → unit perp-to-A direction."""
    dirs = {}

    # ── 1. Amnesic complement: O1 with E1 component removed ──────────────
    O1 = unit(np.load(ROOT / "results/d3_balanced_control/direction_D3prime_no_S0.npy"))
    O1_noE1 = unit(O1 - float(np.dot(O1, E1_hat)) * E1_hat)
    dirs["O1_amnesic"] = perp_to_A(O1_noE1, A_hat)
    print(f"  O1_amnesic: cos(O1,E1)={float(np.dot(O1,E1_hat)):+.4f} "
          f"→ cos(O1_noE1,E1)={float(np.dot(O1_noE1,E1_hat)):+.4f}  "
          f"cos(O1_noE1,A)={float(np.dot(O1_noE1,A_hat)):+.4f}")

    # ── 2. New O baselines ────────────────────────────────────────────────
    candidates = {
        "O_sae_proj":  ROOT / "steering/directions/direction_sae_projected_caa.npz",
        "O_post_v1":   ROOT / "steering/directions/direction_search_post_clean_eval200_seed42_bridge_v1.npz",
        "O_meandiff":  ROOT / "steering/directions/direction_probe_mean_diff.npz",
        "O_sae_f1":    ROOT / "steering/directions/direction_sae_feature_rank1_f112115.npz",
        "O_sae_f2":    ROOT / "steering/directions/direction_sae_feature_rank2_f112616.npz",
        "O_sae_f3":    ROOT / "steering/directions/direction_sae_feature_rank3_f91348.npz",
    }
    for name, path in candidates.items():
        v = load_direction(path)
        vp = perp_to_A(v, A_hat)
        dirs[name] = vp
        print(f"  {name:<16s}: cos_A={float(np.dot(v,A_hat)):+.4f}  "
              f"cos_ev={float(np.dot(v,E1_hat)):+.4f}  "
              f"||perp||={np.linalg.norm(vp):.4f}")

    return dirs


def run_injections(model, tok, prompts, base, dirs, tool_ids, fin_ids):
    n = len(prompts)
    margins = {k: np.zeros(n, np.float32) for k in dirs}
    t0 = time.time()
    for i, p in enumerate(prompts):
        for k, v in dirs.items():
            hf = lambda v=v: ProjectionFlipHook(model, v, factor=2.0)
            margins[k][i] = forward_margin(model, tok, p, hf, tool_ids, fin_ids)
        if (i + 1) % 5 == 0 or i == 0:
            eta = (time.time() - t0) / (i + 1) * (n - i - 1)
            print(f"  [{i+1:>3d}/{n}]  ETA={eta:.0f}s")
    return margins


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
    # ── load directions ───────────────────────────────────────────────────
    fig_dirs = np.load(ROOT / "results/fig1_v3/directions.npz", allow_pickle=True)
    A_hat = fig_dirs["A_hat"].astype(np.float32)
    E1_hat = fig_dirs["E1_LR_L20"].astype(np.float32)
    print("[directions]")
    dirs = build_directions(A_hat, E1_hat)
    print(f"  total: {len(dirs)} directions\n")

    # ── cohort ────────────────────────────────────────────────────────────
    MODEL_ID = os.environ.get("MODEL_ID",
                              "/home/featurize/work/models/Qwen2.5-7B-Instruct")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    labels = [json.loads(l) for l in open(ROOT / "results/phase1_probe/labels.jsonl")]
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
        prompts.append(build_p0_prompt(tok, ld["question"],
                                       s0["action_input"], s0["observation"]))
        sample_ids.append(ld["sample_id"])
        if len(prompts) >= N: break
    cached = np.load(ROOT / "results/evidence_erasure_test/per_prompt_margins.npz")
    assert sample_ids == list(cached["sample_ids"]), "cohort mismatch"
    base = cached["baseline"].astype(np.float32)
    print(f"[cohort] N={len(prompts)}  baseline mean={base.mean():+.3f}\n")

    # ── model ─────────────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM
    print(f"[load model] {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True).eval()
    tool_ids = [tok.encode(t, add_special_tokens=False)[0]
                for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0]
                for t in ACTION_TOKENS["finish"]]

    # ── run ───────────────────────────────────────────────────────────────
    print("[injections]")
    margins = run_injections(model, tok, prompts, base, dirs, tool_ids, fin_ids)

    # ── stats ─────────────────────────────────────────────────────────────
    # Also load AUROC for each direction for the output
    acts = np.load(ROOT / "results/phase1_probe/activations_multilayer.npz",
                   allow_pickle=True)
    X20 = acts["layer_20"].astype(np.float32)
    y_A = acts["y"].astype(int)
    from sklearn.metrics import roc_auc_score

    null_p95 = 0.1226  # from d3_perp_vs_random_null
    out = {"N": len(prompts), "factor": 2.0, "layer": LAYER,
           "null_p95": null_p95, "directions": {}}
    rows = []
    for k, v in dirs.items():
        s = compute_stats(margins[k], base)
        scores = X20 @ v
        auroc = max(roc_auc_score(y_A, scores), 1 - roc_auc_score(y_A, scores))
        s["oof_auroc"] = float(auroc)
        s["cos_A"] = float(np.dot(v, A_hat))
        s["cos_E1"] = float(np.dot(v, E1_hat))
        above = s["abs_signed_mean_dm"] > null_p95
        s["above_null"] = above
        out["directions"][k] = s
        rows.append((k, s))

    json.dump(out, open(OUT / "results.json", "w"), indent=2)
    np.savez(OUT / "per_prompt_margins.npz",
             sample_ids=np.array(sample_ids), baseline=base,
             **{k: margins[k] for k in dirs})

    print("\n[results — sorted by |Δm|_perp]")
    for k, s in sorted(rows, key=lambda x: x[1]["abs_signed_mean_dm"], reverse=True):
        flag = "** OPERATIVE **" if s["above_null"] else "  below null  "
        print(f"  {k:<18s}  AUROC={s['oof_auroc']:.3f}  "
              f"|Δm|={s['abs_signed_mean_dm']:.3f}  "
              f"CI=[{s['abs_signed_mean_dm_ci'][0]:.3f},"
              f"{s['abs_signed_mean_dm_ci'][1]:.3f}]  {flag}")
    print(f"\n[saved] {OUT/'results.json'}")


if __name__ == "__main__":
    main()
