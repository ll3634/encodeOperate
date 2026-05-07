#!/usr/bin/env python3
"""Figure 1 v3 — causal pass: for every direction in fig1_v3/directions.npz,
inject its perp version at L20 last-token (factor=2.0) and measure
|signed mean Δm| against the cached §3 baseline (N=100 cohort).

Reuses cohort + baseline from results/evidence_erasure_test/ exactly.
A_L20 is skipped on the perp side (perp(A)=0); we keep its full-direction
flip number from previous experiments only as reference plotted by hand.
"""
import json, time, sys, os
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from agent.prompts import ACTION_TOKENS
from evidence_erasure_test import (
    ProjectionFlipHook, forward_margin, build_p0_prompt, LAYER, SEED, N,
)

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "results/fig1_v3"; OUT.mkdir(parents=True, exist_ok=True)


def main():
    print(f"[init] L{LAYER} N={N}")
    arrs = np.load(OUT / "directions.npz", allow_pickle=True)
    A_hat = arrs["A_hat"].astype(np.float32)
    summary = json.load(open(OUT / "extract_summary.json"))["directions"]

    # Pick perp arrays for every direction except A_L20 (perp undefined).
    perp_keys = [k for k in arrs.files
                 if k.endswith("_perp") and not k.startswith("A_L20")]
    print(f"[directions] {len(perp_keys)} perp directions to inject")
    for k in perp_keys[:5]:
        v = arrs[k]
        print(f"  sample {k}: cos·A={float(np.dot(v, A_hat)):+.2e}  ||v||={np.linalg.norm(v):.4f}")

    # Cohort: same locked N=100 prompts as evidence_erasure_test (HF cache OK).
    from transformers import AutoModelForCausalLM, AutoTokenizer
    MODEL_ID = os.environ.get("MODEL_ID", "/home/featurize/work/models/Qwen2.5-7B-Instruct")
    print(f"\n[load tok] {MODEL_ID}")
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
        prompts.append(build_p0_prompt(tok, ld["question"], s0["action_input"], s0["observation"]))
        sample_ids.append(ld["sample_id"])
        if len(prompts) >= N: break
    cached = np.load(ROOT / "results/evidence_erasure_test/per_prompt_margins.npz")
    assert sample_ids == list(cached["sample_ids"]), "cohort mismatch"
    base = cached["baseline"].astype(np.float32)
    n = len(prompts)
    print(f"[prompts] N={n}  baseline mean margin={base.mean():+.3f}")

    print(f"\n[load model]")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True).eval()
    tool_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]

    margins = {k: np.zeros(n, dtype=np.float32) for k in perp_keys}
    t0 = time.time()
    for i, p in enumerate(prompts):
        for k in perp_keys:
            v = arrs[k]
            hf = lambda v=v: ProjectionFlipHook(model, v, factor=2.0)
            margins[k][i] = forward_margin(model, tok, p, hf, tool_ids, fin_ids)
        if (i + 1) % 5 == 0 or i == 0:
            eta = (time.time() - t0) / (i + 1) * (n - i - 1)
            print(f"  [{i+1:>3d}/{n}]  ETA={eta:.0f}s")
    print(f"[done] forwards: {time.time() - t0:.0f}s")

    np.savez(OUT / "per_prompt_margins.npz",
             sample_ids=np.array(sample_ids), baseline=base,
             **{k: margins[k] for k in perp_keys})

    # ── stats ─────────────────────────────────────────────────────────
    def stats(m_cond, base):
        dm = (m_cond - base).astype(np.float32)
        signed = float(dm.mean())
        rng_b = np.random.default_rng(SEED); B = 2000
        idx = rng_b.integers(0, len(dm), size=(B, len(dm)))
        sm = np.abs(dm[idx].mean(axis=1))
        lo, hi = np.percentile(sm, [2.5, 97.5])
        return {"signed_mean_dm": signed,
                "abs_signed_mean_dm": abs(signed),
                "abs_signed_mean_dm_ci": [float(lo), float(hi)],
                "flip_rate_sign_change":
                    float((np.sign(dm) != np.sign(base)).mean())}

    out = {"N": n, "factor": 2.0, "layer": LAYER,
           "convention": "abs_signed_mean_dm matches §3 |Δm_flip|",
           "directions": {}}
    for k in perp_keys:
        s = stats(margins[k], base)
        canonical = k[:-len("_perp")]
        s.update({"family": summary[canonical]["family"],
                  "layer_extracted": summary[canonical]["layer"],
                  "cos_A": summary[canonical]["cos_A"],
                  "oof_auroc": summary[canonical]["oof_auroc"],
                  "oof_auroc_std": summary[canonical]["oof_auroc_std"]})
        out["directions"][canonical] = s
    json.dump(out, open(OUT / "results.json", "w"), indent=2)

    print("\n[results — sorted by |Δm|_perp desc]")
    rows = sorted(out["directions"].items(),
                  key=lambda kv: kv[1]["abs_signed_mean_dm"], reverse=True)
    for c, s in rows:
        print(f"  {c:<16s} fam={s['family']:<11s} "
              f"AUROC={s['oof_auroc']:.3f}  "
              f"|Δm|_perp={s['abs_signed_mean_dm']:.3f}  "
              f"CI=[{s['abs_signed_mean_dm_ci'][0]:.3f},"
              f"{s['abs_signed_mean_dm_ci'][1]:.3f}]")


if __name__ == "__main__":
    main()
