#!/usr/bin/env python3
"""Null-Space Injection Rotation -- §4.1 additive injection on E(theta) directions.

Hook: steering.hook_utils.SteeringHook (the SAME function as §4.1).
Operating point: rho=+0.20, layer=20, position=-1, max_interventions=1, RMS-normalised.

The cached §3 'parallel' direction is -E (cos=-1.0); injecting parallel at
rho=-0.20 yielded mean dm = -0.157.  Equivalently, injecting +E at rho=+0.20
should reproduce the same -0.157.  This is verification V1.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from agent.prompts import ACTION_TOKENS
from steering.hook_utils import SteeringHook
from dose_response_erasure import rebuild_prompts

LAYER = 20
RHO = +0.20
HIDDEN_RMS = 0.65
OUT = Path("results/evidence_erasure_test/nullspace_injection")
OUT.mkdir(parents=True, exist_ok=True)


def margin_from_logits(logits, tool_ids, fin_ids):
    lp = torch.log_softmax(logits, dim=-1)
    return (torch.logsumexp(lp[tool_ids], 0) -
            torch.logsumexp(lp[fin_ids], 0)).item()


def inject_margin(model, tok, prompt, direction_rms1, rho, tool_ids, fin_ids):
    """§4.1 protocol: alpha = rho * (HIDDEN_RMS / d_rms); h += alpha * d.
    direction_rms1 must already have RMS=1, so alpha = rho*HIDDEN_RMS.
    """
    device = next(model.parameters()).device
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    if direction_rms1 is not None and abs(rho) > 1e-8:
        d_rms = float(np.sqrt(np.mean(direction_rms1 ** 2)))
        alpha = rho * (HIDDEN_RMS / d_rms)
        with SteeringHook(model, direction_rms1, alpha, layer=LAYER,
                          position=-1, max_interventions=1):
            with torch.no_grad():
                logits = model(ids).logits[0, -1, :]
    else:
        with torch.no_grad():
            logits = model(ids).logits[0, -1, :]
    return margin_from_logits(logits, tool_ids, fin_ids)


def load_unit_dirs():
    """Returns (rotation_dirs, anchor_dirs, A_hat, E_hat, D3_hat) all unit-L2."""
    z = np.load(ROOT / "results/evidence_erasure_test/nullspace_rotation/"
                       "constructed_directions.npz", allow_pickle=True)
    rot = {nm: v.astype(np.float32) for nm, v in zip(z["names"], z["vectors"])}
    A = np.load(ROOT / "steering/directions/direction_decomp_full_layer20.npz",
                allow_pickle=True)["decision_direction"].astype(np.float32)
    E = np.load(ROOT / "results/phase1_probe/probe_direction_l20.npz",
                allow_pickle=True)["decision_direction"].astype(np.float32)
    D3 = np.load(ROOT / "results/ocft/per_candidate/D3_candidate_present/"
                        "direction.npy").astype(np.float32)
    A /= np.linalg.norm(A); E /= np.linalg.norm(E); D3 /= np.linalg.norm(D3)
    anchors = {"A_anchor": A, "D3_anchor": D3}
    return rot, anchors, A, E, D3


def to_rms1(unit_vec):
    """Unit-L2 vector (norm=1) -> RMS=1 by multiplying by sqrt(d)."""
    d = unit_vec.shape[0]
    return (unit_vec * np.sqrt(d)).astype(np.float32)


def main():
    t0 = time.time()
    rot, anchors, A_hat, E_hat, D3_hat = load_unit_dirs()
    # ---- Verification V2: cos(·, A) within 0.001 of -0.013 ----
    target = -0.013456
    devs = {nm: float(v @ A_hat) - target for nm, v in rot.items()}
    max_dev = max(abs(d) for d in devs.values())
    print(f"[V2] max |cos(E_theta, A) - (-0.013456)| = {max_dev:.2e}  "
          f"(tolerance 1e-3): {'PASS' if max_dev < 1e-3 else 'FAIL'}")
    if max_dev >= 1e-3:
        print("[V2] FAIL -- aborting"); sys.exit(2)
    # ---- Verification V3: hook function identity ----
    print(f"[V3] injection hook = SteeringHook from {SteeringHook.__module__} "
          f"({Path(sys.modules[SteeringHook.__module__].__file__).resolve()})")
    print(f"[V3] §4.1 reference = same SteeringHook (used in "
          f"scripts/decomposition_ci_null.py:compute_margin)")

    # ---- Cached baseline + sample_ids (same as §3 / decomp_ci_null) ----
    cache = np.load(ROOT / "results/decomposition_ci_null/per_example_shifts.npz",
                    allow_pickle=True)
    sample_ids = [str(s) for s in cache["sample_ids"]]
    base = cache["baseline"].astype(np.float32)
    cached_par_shift = cache["parallel"].astype(np.float32)  # = §3 'E injection' (sign-flipped)
    print(f"[cache] N={len(sample_ids)}, baseline mean={base.mean():.4f}, "
          f"cached parallel shift mean={cached_par_shift.mean():.4f}")

    # ---- Build job list ----
    jobs = []  # (group, name, theta_or_None, rms1_dir)
    for ang in [0, 15, 30, 45, 60, 75, 90]:
        if ang == 0:
            jobs.append(("E_to_D3", "E_theta0", 0, to_rms1(rot["E_theta0"])))
        else:
            jobs.append(("E_to_D3", f"E_to_D3__theta{ang:02d}", ang,
                         to_rms1(rot[f"E_to_D3__theta{ang:02d}"])))
    for ang in [0, 15, 30, 45, 60, 75, 90]:
        nm = "E_theta0" if ang == 0 else f"E_to_D1__theta{ang:02d}"
        jobs.append(("E_to_D1", nm, ang, to_rms1(rot[nm])))
    for ang in [0, 15, 30, 45, 60, 75, 90]:
        nm = "E_theta0" if ang == 0 else f"E_to_random__theta{ang:02d}"
        jobs.append(("E_to_random", nm, ang, to_rms1(rot[nm])))
    for nm, v in anchors.items():
        jobs.append(("anchor", nm, None, to_rms1(v)))
    unique_names = sorted({j[1] for j in jobs})
    print(f"[plan] groups=3 paths + 2 anchors; unique dirs = {len(unique_names)}; "
          f"forwards = {len(unique_names)} x {len(sample_ids)} = "
          f"{len(unique_names)*len(sample_ids)}")

    # ---- Load model ----
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("[load] Qwen/Qwen2.5-7B-Instruct")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                                        trust_remote_code=True)
    prompts = rebuild_prompts(tok, sample_ids)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True).eval()
    tool_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]

    # ---- Run unique dirs once, broadcast to paths in post ----
    margins = {}; tg = time.time()
    name2vec = {nm: vec for (_, nm, _, vec) in jobs}
    for k, nm in enumerate(unique_names):
        vec = name2vec[nm]
        per = np.empty(len(prompts), dtype=np.float32)
        for i, p in enumerate(prompts):
            per[i] = inject_margin(model, tok, p, vec, RHO, tool_ids, fin_ids)
        margins[nm] = per
        eta = (time.time() - tg) / (k + 1) * (len(unique_names) - k - 1)
        dm = float((per - base).mean())
        print(f"  [{k+1:>2d}/{len(unique_names)}] {nm:<28s}  mean_dm={dm:+.4f}  ETA={eta:5.1f}s")
    print(f"[gpu] {time.time()-tg:.0f}s")

    np.savez_compressed(OUT / "per_prompt_margins.npz",
        sample_ids=np.array(sample_ids), baseline=base, **margins)
    from nullspace_injection_report import write_report
    write_report(OUT, margins, base, jobs, cached_par_shift, A_hat, E_hat, D3_hat)
    print(f"[done] {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
