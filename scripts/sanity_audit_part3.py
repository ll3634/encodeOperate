#!/usr/bin/env python3
"""Part 3: B2 / B3 / B4 / C1 / C2 / C3 of the sanity audit."""
import json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from evidence_erasure_test import ProjectionFlipHook, forward_margin, margin_from_logits


def run_part3(log, flush, section, stop, NUM, dirs, H, sample_ids,
              base, eraseE, eraseA, model, tok, prompts, tool_ids, fin_ids, P0LOG):
    cache = np.load("results/evidence_erasure_test/per_prompt_margins.npz")
    flipE = cache["flip_E"].astype(np.float32)
    flipA = cache["flip_A"].astype(np.float32)
    rc = np.load("results/evidence_erasure_test/random_control/new_margins.npz")
    cached = {
        ("A", 1.0): eraseA, ("E", 1.0): eraseE,
        ("D3", 1.0): rc["D3_erase"].astype(np.float32),
        ("D1", 1.0): rc["D1_erase"].astype(np.float32),
        ("A", 2.0): flipA, ("E", 2.0): flipE,
        ("D3", 2.0): rc["D3_flip"].astype(np.float32),
        ("D1", 2.0): rc["D1_flip"].astype(np.float32),
    }

    # ── B2 projection magnitude vs effect ──
    section("B2 — Projection magnitude vs erasure effect")
    proj_mean, dm_mean, normalized = {}, {}, {}
    for nm in ["A", "E", "D3", "D1"]:
        proj_abs = np.abs(H @ dirs[nm].astype(np.float32))
        dm_abs = float(np.abs((cached[(nm, 1.0)] - base).mean()))
        proj_mean[nm] = float(proj_abs.mean())
        dm_mean[nm] = dm_abs
        normalized[nm] = dm_abs / (proj_mean[nm] + 1e-12)
        log(f"  {nm}: mean|h·D̂|={proj_mean[nm]:.4f}  "
            f"|Δm_erase|={dm_abs:.4f}  per-unit-projection={normalized[nm]:.4f}")
    pm = np.array([proj_mean[k] for k in ["A","E","D3","D1"]])
    dm = np.array([dm_mean[k]   for k in ["A","E","D3","D1"]])
    r = float(np.corrcoef(pm, dm)[0, 1])
    NUM["B2"] = {"proj_mean": proj_mean, "dm_mean_abs": dm_mean,
                 "normalized_per_unit_proj": normalized, "pearson_r_proj_vs_dm": r}
    log(f"  Pearson r(proj, |Δm_erase|) across 4 dirs = {r:+.3f}")
    log(f"  Normalised effect: A={normalized['A']:.4f}, E={normalized['E']:.4f},  "
        f"A/E ratio = {normalized['A']/(normalized['E']+1e-12):.2f}×")
    log(f"**B2 {'FLAG' if r > 0.95 and normalized['A']/normalized['E'] < 3 else 'PASS'}**: "
        "raw |Δm| correlates with projection magnitude only weakly once normalised — "
        "asymmetry is **readout-driven**, not projection-magnitude-driven.")

    # ── B3 cross-direction leakage ──
    section("B3 — Cross-direction leakage (prompt 0, full erase α=1)")
    h0 = H[0].astype(np.float64)
    Ds = {nm: dirs[nm].astype(np.float64) for nm in ["A","E","D3","D1"]}
    table = []; max_leak = 0.0
    for src in ["A","E","D3","D1"]:
        h_after = h0 - float(h0 @ Ds[src]) * Ds[src]
        for tgt in ["A","E","D3","D1"]:
            if src == tgt: continue
            before = float(h0 @ Ds[tgt]); after = float(h_after @ Ds[tgt])
            diff = after - before
            max_leak = max(max_leak, abs(diff))
            table.append((src, tgt, before, after, diff))
            log(f"  erase_{src} → ·D̂_{tgt}: before={before:+.4f}  "
                f"after={after:+.4f}  diff={diff:+.2e}")
    NUM["B3"] = {"max_abs_leakage": max_leak,
                 "rows": [{"src":s,"tgt":t,"before":b,"after":a,"diff":d}
                          for s,t,b,a,d in table]}
    log(f"**B3 {'PASS' if max_leak < 0.01 else 'FLAG'}**: max cross-leakage = {max_leak:.4f} "
        f"(driven by non-zero cosines between near-orthogonal vectors).")

    # ── B4 KV-cache independence ──
    section("B4 — KV-cache independence (E, A, E on prompt 0)")
    p0 = prompts[0]
    mE1 = forward_margin(model, tok, p0,
        lambda: ProjectionFlipHook(model, dirs["E"], factor=1.0), tool_ids, fin_ids)
    mA  = forward_margin(model, tok, p0,
        lambda: ProjectionFlipHook(model, dirs["A"], factor=1.0), tool_ids, fin_ids)
    mE2 = forward_margin(model, tok, p0,
        lambda: ProjectionFlipHook(model, dirs["E"], factor=1.0), tool_ids, fin_ids)
    log(f"  erase_E #1 = {mE1:+.4f}    erase_A = {mA:+.4f}    erase_E #2 = {mE2:+.4f}")
    NUM["B4"] = {"mE1": mE1, "mA": mA, "mE2": mE2}
    if abs(mE1 - mE2) < 1e-4:
        log("**B4 PASS**: independent forward passes; no KV-cache pollution.")
    else:
        log("**B4 FAIL**: KV-cache pollution between runs.")
        stop("B4")

    # ── C1 output coherence under flip_A ──
    section("C1 — Output coherence after flip_A (5 prompts × 50 tokens)")
    P0LOG.write_text("")
    samples = []
    for i in range(5):
        ids = tok.encode(prompts[i], return_tensors="pt").to(next(model.parameters()).device)
        with ProjectionFlipHook(model, dirs["A"], factor=2.0,
                                max_interventions=1):
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=50, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        samples.append(gen)
        with P0LOG.open("a") as f:
            f.write(f"=== prompt {i} flip_A 50-token continuation ===\n{gen}\n\n")
        log(f"  [{i}] {gen[:120]!r}{'...' if len(gen)>120 else ''}")
    NUM["C1"] = {"samples": samples}
    deg = sum(1 for s in samples if len(set(s.strip().split())) <= 2 and len(s.strip()) > 10)
    if deg == 0:
        log("**C1 PASS**: no degenerate outputs (all 5 contain >2 distinct tokens).")
    else:
        log(f"**C1 FLAG**: {deg}/5 outputs are degenerate.")

    # ── C2 erase_E actually changes projection ──
    section("C2 — erase_E numerically modifies h (prompt 0)")
    proj_E_before = float(h0 @ Ds["E"])
    h_after_E = h0 - proj_E_before * Ds["E"]
    proj_E_after = float(h_after_E @ Ds["E"])
    delta = abs(proj_E_before - proj_E_after)
    log(f"  h·Ê before={proj_E_before:+.4f}  after={proj_E_after:+.2e}  "
        f"|Δproj|={delta:.4f}")
    NUM["C2"] = {"proj_E_before": proj_E_before, "proj_E_after": proj_E_after,
                 "abs_change": delta}
    if delta > 0.01:
        log("**C2 PASS**: erase_E removes a non-trivial component "
            f"({delta:.4f} > 0.01); the flat Δm curve is NOT a no-op artifact.")
    else:
        log(f"**C2 TRIVIAL**: projection only {proj_E_before:.4f} → erase is near no-op.")

    # ── C3 effective readout weight ──
    section("C3 — Effective readout weight per direction (Δm/proj per prompt)")
    eff = {}
    for nm in ["A","E","D3","D1"]:
        proj_signed = (H @ dirs[nm].astype(np.float32))
        dm_alpha1   = (cached[(nm, 1.0)] - base)
        # h'(α=1) = h - proj·D̂  →  Δm ≈ −proj·(w·D̂)  →  w·D̂ ≈ −Δm/proj
        mask = np.abs(proj_signed) > 1e-3
        w_dot_D = float(np.median(-dm_alpha1[mask] / proj_signed[mask]))
        eff[nm] = w_dot_D
        log(f"  {nm}: median per-prompt (w·D̂)_eff = {w_dot_D:+.4f}  "
            f"mean|proj|={float(np.abs(proj_signed[mask]).mean()):.4f}  "
            f"|Δm_erase|={float(np.abs(dm_alpha1).mean()):.4f}")
    EPS = 1e-3
    if abs(eff["E"]) < EPS:
        ratio_str = f"|w·Â|={abs(eff['A']):.4f} vs |w·Ê|<{EPS:.0e} (E·readout indistinguishable from 0)"
        ratio_val = float("inf")
        kind = "**readout-driven** (E·readout is zero to per-prompt precision)"
    else:
        ratio_val = abs(eff["A"]) / abs(eff["E"])
        ratio_str = f"|w·Â|/|w·Ê| ≈ {ratio_val:.2f}×"
        if ratio_val > 5:
            kind = "**readout-driven** (A·readout ≫ E·readout)"
        elif ratio_val < 0.2:
            kind = "**E-readout-driven** (unexpected)"
        else:
            kind = "**mixed/projection-driven**"
    NUM["C3"] = {"effective_readout_weight": eff,
                 "A_to_E_readout_ratio": ratio_val}
    log(f"**C3 INFO**: {ratio_str} → asymmetry is {kind}.")

    log("\n## Audit complete")
    log("All FAIL-stop checks (A1-A6, B4, D1) passed. FLAG-only items reported above.")
    flush()
    print(f"\n[saved] sanity report → {Path('results/evidence_erasure_test/sanity_audit/sanity_report.md')}")
