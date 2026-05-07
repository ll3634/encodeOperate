#!/usr/bin/env python3
"""Null-space rotation scan: matched-geometry erasure of E(theta).

For 19 unique unit-norm directions on three rotation paths (E->D3,
E->D1, E->random) with cos(.,A) = c = -0.013 fixed, run
projection-erasure (alpha=1.0) and sign-flip (alpha=2.0) on the
cached N=100 prompt set.  Reuses cached margins for theta=0 (= E).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from agent.prompts import ACTION_TOKENS
from evidence_erasure_test import (ProjectionFlipHook, build_p0_prompt,
                                   forward_margin, LAYER)
from dose_response_erasure import rebuild_prompts
from nullspace_rotation_io import (load_directions, construct_family, verify,
                                   unique_directions, ANGLES_DEG, PATHS)

OUT = Path("results/evidence_erasure_test/nullspace_rotation"); OUT.mkdir(parents=True, exist_ok=True)
FIG_ROT = Path("results/evidence_erasure_test/figure_nullspace_rotation.json")
FIG_SCT = Path("results/evidence_erasure_test/figure_nullspace_scatter.json")
N_BOOT = 2000
SEED_BOOT = 12345


def boot_ci(x: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED_BOOT):
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = x[idx].mean()
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return float(x.mean()), lo, hi


def main():
    t0 = time.time()
    # ---------- 1. construction & verification ----------
    dirs = load_directions()
    family, meta = construct_family(dirs)
    rows, max_dev, t0_ok = verify(family, dirs)
    if max_dev > 0.002:
        sys.exit(f"STOP: max cos deviation {max_dev:.4e} > 0.002 (construction broken)")
    if not t0_ok:
        sys.exit("STOP: theta=0 does not recover E exactly")
    uniq = unique_directions(family)  # 19 entries
    print(f"[verify] 19 unique directions; max |cos-c|={max_dev:.2e}; theta0_match={t0_ok}")

    # save constructed_directions.npz + verification_table.json
    arr = np.stack([uniq[k] for k in uniq.keys()], axis=0).astype(np.float32)
    np.savez_compressed(OUT / "constructed_directions.npz",
                        names=np.array(list(uniq.keys())), vectors=arr)
    (OUT / "verification_table.json").write_text(json.dumps(
        {"meta": meta, "rows": rows, "max_cos_deviation": max_dev,
         "theta_0_matches_E": t0_ok}, indent=2, default=float))

    # ---------- 2. cached references ----------
    cache_main = np.load("results/evidence_erasure_test/per_prompt_margins.npz")
    cache_hid  = np.load("results/evidence_erasure_test/dose_response/per_prompt_margins_alpha_new.npz")
    sample_ids = [str(s) for s in cache_main["sample_ids"]]
    base = cache_main["baseline"].astype(np.float32)
    H = cache_hid["hidden_L20"].astype(np.float32)
    cached_margins = {
        "E_theta0": {1.0: cache_main["erase_E"].astype(np.float32),
                     2.0: cache_main["flip_E"].astype(np.float32)},
    }
    # Determine which (name, alpha) pairs are NEW
    new_jobs = []
    for nm in uniq.keys():
        for alpha in (1.0, 2.0):
            if nm in cached_margins and alpha in cached_margins[nm]:
                continue
            new_jobs.append((nm, alpha))
    n_new = len(new_jobs)
    print(f"[plan] cached: 1 dir x 2 alpha = 2; new: {n_new} (= {n_new // 2} dirs x 2 alpha)")
    print(f"[plan] forwards: {n_new * len(sample_ids)} = {n_new} * N=100")

    # ---------- 3. GPU forwards ----------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("\n[load] Qwen/Qwen2.5-7B-Instruct")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    prompts = rebuild_prompts(tok, sample_ids)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True).eval()
    tool_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]

    measured: dict = {nm: {} for nm in uniq.keys()}
    measured["E_theta0"][1.0] = cached_margins["E_theta0"][1.0]
    measured["E_theta0"][2.0] = cached_margins["E_theta0"][2.0]
    t_start = time.time()
    for j, (nm, alpha) in enumerate(new_jobs):
        v = uniq[nm].astype(np.float32)
        per = np.empty(len(prompts), dtype=np.float32)
        for i, p in enumerate(prompts):
            per[i] = forward_margin(
                model, tok, p,
                lambda v=v, a=alpha: ProjectionFlipHook(model, v, factor=a),
                tool_ids, fin_ids)
        measured[nm][alpha] = per
        eta = (time.time() - t_start) / (j + 1) * (n_new - j - 1)
        print(f"  [{j + 1:>3d}/{n_new}] {nm:<30s} alpha={alpha}  "
              f"mean|dm|={float(np.abs(per - base).mean()):.4f}  ETA={eta:5.1f}s")

    # ---------- 4. analysis ----------
    print("\n[analysis] bootstrap CIs (B=2000) and projection magnitudes")
    points_per_path = {p: [] for p in PATHS}
    scatter = []
    for nm, vec in uniq.items():
        proj = (H @ vec.astype(np.float32))
        proj_mean = float(np.abs(proj).mean())
        proj_sd   = float(np.abs(proj).std())
        de = measured[nm][1.0] - base
        df = measured[nm][2.0] - base
        m_e, lo_e, hi_e = boot_ci(np.abs(de))
        m_f, lo_f, hi_f = boot_ci(np.abs(df))
        flip_e = float(np.mean(np.sign(de) != 0) * np.mean((measured[nm][1.0] > 0) != (base > 0)))
        flip_e = float(np.mean((measured[nm][1.0] > 0) != (base > 0)))
        flip_f = float(np.mean((measured[nm][2.0] > 0) != (base > 0)))
        rec = {
            "name": nm, "mean_proj_magnitude": proj_mean,
            "sd_proj_magnitude": proj_sd,
            "dm_erase_mean": m_e, "dm_erase_ci": [lo_e, hi_e],
            "dm_flip_mean":  m_f, "dm_flip_ci":  [lo_f, hi_f],
            "flip_rate_erase": flip_e, "flip_rate_flip": flip_f,
        }
        scatter.append({"name": nm, "mean_proj_magnitude": proj_mean,
                        "dm_erase_abs": m_e, "dm_flip_abs": m_f})
        # attach to per-path output
        if nm == "E_theta0":
            for p in PATHS:
                points_per_path[p].append((0, rec))
        else:
            p, th = nm.split("__theta")
            points_per_path[p].append((int(th), rec))

    # ---------- 5. output JSON ----------
    from nullspace_rotation_io import _u  # noqa
    A_hat, E_hat, D3_hat, D1_hat = dirs["A"], dirs["E"], dirs["D3"], dirs["D1"]
    figure = {"constant_cos": meta["c"],
              "verification": {"max_cos_deviation": max_dev,
                               "theta_0_matches_E": t0_ok},
              "paths": {}}
    for p in PATHS:
        items = sorted(points_per_path[p], key=lambda x: x[0])
        path_pts = []
        for th, rec in items:
            v = uniq["E_theta0"] if th == 0 else uniq[f"{p}__theta{th:02d}"]
            path_pts.append({
                "theta_deg": int(th),
                "cos_with_A": float(v @ A_hat),
                "cos_with_E": float(v @ E_hat),
                "cos_with_D3": float(v @ D3_hat),
                "cos_with_D1": float(v @ D1_hat),
                **rec,
            })
        figure["paths"][p] = {
            "description": f"Rotation from E toward {p.split('_to_')[1]} in null(A)",
            "angles_deg": ANGLES_DEG, "points": path_pts}
    FIG_ROT.write_text(json.dumps(figure, indent=2, default=float))
    FIG_SCT.write_text(json.dumps({"description": "All 19 dirs: |proj| vs |dm|",
                                   "points": scatter}, indent=2, default=float))
    (OUT / "per_direction_results.json").write_text(json.dumps(
        figure, indent=2, default=float))

    # ---------- 6. report ----------
    from nullspace_report import write_report
    write_report(OUT, figure, meta, max_dev, t0_ok, base, measured)
    print(f"\n[done] {time.time()-t0:.0f}s total")
    print(f"[saved] {FIG_ROT}\n[saved] {FIG_SCT}\n[saved] {OUT}/")


if __name__ == "__main__":
    main()
