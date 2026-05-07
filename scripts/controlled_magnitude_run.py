#!/usr/bin/env python3
"""Driver: controlled-magnitude erasure spectrum + null-space rotation at fixed c."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from agent.prompts import ACTION_TOKENS
from dose_response_erasure import rebuild_prompts
from controlled_magnitude_erasure import (
    ControlledMagnitudeHook, forward_margin, boot_ci,
    load_all_directions, build_rotation_directions, OUT, RANDOM_K)

SPEC_DIR_ORDER = (["A", "E", "D3", "D1", "D2", "D4"]
                  + [f"r_{i+1:02d}" for i in range(RANDOM_K)])


def main():
    t0 = time.time()
    # ---------- 1. Load cached state ----------
    cache_main = np.load("results/evidence_erasure_test/per_prompt_margins.npz")
    cache_hid  = np.load("results/evidence_erasure_test/dose_response/per_prompt_margins_alpha_new.npz")
    sample_ids = [str(s) for s in cache_main["sample_ids"]]
    base = cache_main["baseline"].astype(np.float32)
    H = cache_hid["hidden_L20"].astype(np.float64)  # (100, 3584)

    # ---------- 2. Choose c_E and c_A ----------
    dirs = load_all_directions()
    proj_E = float(np.mean(np.abs(H @ dirs["E"])))
    proj_A = float(np.mean(np.abs(H @ dirs["A"])))
    print(f"[c] mean|h.E_hat| = c_E = {proj_E:.6f}")
    print(f"[c] mean|h.A_hat| = c_A = {proj_A:.6f}")
    if proj_E < 1e-3:
        print("[FLAG] c_E < 0.001; using floor 0.01")
        c_E = max(proj_E, 0.01)
    else:
        c_E = proj_E
    c_A = proj_A
    magnitudes = {"c_E": c_E, "c_A": c_A}

    # Per-direction projection magnitude (for diagnostics)
    proj_per_dir = {nm: float(np.mean(np.abs(H @ vec))) for nm, vec in dirs.items()}
    rot = build_rotation_directions()
    proj_per_rot = {nm: float(np.mean(np.abs(H @ vec))) for nm, vec in rot.items()}

    # ---------- 3. Plan ----------
    n_spec = len(SPEC_DIR_ORDER)
    n_rot = len(rot)
    n_jobs = (n_spec + n_rot) * 2 * len(sample_ids)
    print(f"[plan] spectrum: {n_spec} dirs, rotation: {n_rot} dirs; "
          f"2 magnitudes; total forwards = {n_jobs}")

    # ---------- 4. GPU ----------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("\n[load] Qwen/Qwen2.5-7B-Instruct")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    prompts = rebuild_prompts(tok, sample_ids)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True).eval()
    tool_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]

    spec_results = {nm: {} for nm in SPEC_DIR_ORDER}
    rot_results = {nm: {} for nm in rot.keys()}
    job_idx = 0
    n_dir_total = (n_spec + n_rot) * 2
    t_start = time.time()
    for tag, dir_set, store in [("spec", dirs, spec_results), ("rot", rot, rot_results)]:
        keys = SPEC_DIR_ORDER if tag == "spec" else list(rot.keys())
        for nm in keys:
            v = dir_set[nm].astype(np.float32)
            for c_label, c_val in magnitudes.items():
                per = np.empty(len(prompts), dtype=np.float32)
                for i, p in enumerate(prompts):
                    per[i] = forward_margin(
                        model, tok, p,
                        lambda v=v, c=c_val: ControlledMagnitudeHook(model, v, c=c),
                        tool_ids, fin_ids)
                store[nm][c_label] = per
                job_idx += 1
                eta = (time.time() - t_start) / job_idx * (n_dir_total - job_idx)
                dm = float((per - base).mean())
                print(f"  [{job_idx:>3d}/{n_dir_total}] {tag:<4s} {nm:<28s} "
                      f"{c_label}={c_val:.4f}  mean_dm={dm:+.4f}  ETA={eta:5.1f}s")

    print(f"\n[gpu] {time.time()-t_start:.0f}s")

    # ---------- 5. Analyse ----------
    # Spectrum
    spec_records = []
    for nm in SPEC_DIR_ORDER:
        rec = {"name": nm, "type": (
            "action" if nm == "A" else
            "evidence" if nm == "E" else
            "ocft_candidate" if nm in ("D3", "D1", "D2", "D4") else "random"),
            "mean_proj_magnitude": proj_per_dir[nm]}
        for c_label in ("c_E", "c_A"):
            de = spec_results[nm][c_label] - base
            m, lo, hi = boot_ci(de)
            am, alo, ahi = boot_ci(np.abs(de))
            flip = float(np.mean((spec_results[nm][c_label] > 0) != (base > 0)))
            rec[f"dm_at_{c_label}"] = m; rec[f"dm_at_{c_label}_ci"] = [lo, hi]
            rec[f"abs_dm_at_{c_label}"] = am; rec[f"abs_dm_at_{c_label}_ci"] = [alo, ahi]
            rec[f"flip_rate_{c_label}"] = flip
        spec_records.append(rec)

    # Rotation at fixed c
    rotation_blocks = {}
    for c_label in ("c_E", "c_A"):
        dms, cis = [], []
        for ang in [0, 15, 30, 45, 60, 75, 90]:
            key = f"E_to_D3__theta{ang:02d}"
            de = rot_results[key][c_label] - base
            m, lo, hi = boot_ci(np.abs(de))
            dms.append(m); cis.append([lo, hi])
        rotation_blocks[f"nullspace_rotation_at_{c_label}"] = {
            "path": "E_to_D3", "angles_deg": [0, 15, 30, 45, 60, 75, 90],
            "dm_controlled": dms, "dm_controlled_ci": cis,
            "constant_c": magnitudes[c_label],
        }

    figure = {"magnitudes": magnitudes, "directions": spec_records, **rotation_blocks}
    (OUT / "figure_controlled_spectrum.json").write_text(
        json.dumps(figure, indent=2, default=float))
    (OUT / "per_direction_results.json").write_text(
        json.dumps({"directions": spec_records,
                    "rotation_proj_magnitudes": proj_per_rot,
                    "magnitudes": magnitudes}, indent=2, default=float))
    (OUT / "rotation_at_fixed_c.json").write_text(
        json.dumps(rotation_blocks, indent=2, default=float))

    # Save raw per-prompt margins for reproducibility
    raw = {"baseline": base, "magnitudes_c_E": np.array([c_E]), "magnitudes_c_A": np.array([c_A])}
    for nm in SPEC_DIR_ORDER:
        for c_label in ("c_E", "c_A"):
            raw[f"{nm}_{c_label}"] = spec_results[nm][c_label]
    for nm in rot.keys():
        for c_label in ("c_E", "c_A"):
            raw[f"{nm}_{c_label}"] = rot_results[nm][c_label]
    np.savez_compressed(OUT / "per_prompt_margins.npz", **raw)

    # Render report
    from controlled_report import write_report
    write_report(OUT, figure, magnitudes, proj_per_dir, proj_per_rot)
    print(f"\n[done] {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
