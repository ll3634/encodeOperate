#!/usr/bin/env python3
"""Sanity (3): D3 par/perp injection on its NATIVE LGM corpus
(extractability_support_toggle/pairs.jsonl) — per-cell |par|/|full| ratio.

If D3 is genuinely an "availability" direction, the per-cell pattern on the
LGM toggle corpus (where T0/T1 = candidate present, N0/S0 control) should
be more discriminative than the HotpotQA-native cell breakdown. This script
runs only D3 (not D_evidence) and only on this corpus; it does not modify
any §3 / OCFT artefact.

Pipeline mirrors scripts/ocft_run_injection.py exactly: same Qwen2.5-7B,
L20 p0 last token, ρ=-0.20, hidden_rms=0.65, normalize_rms=1.0, single
intervention.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import SteeringHook
from steering.directions import load_direction

LAYER = 20
RHO = -0.20
HIDDEN_RMS = 0.65
SEED = 20260502


def build_p0_prompt(tok, question, obs):
    pb = PromptBuilder(tools=["search", "calculator"])
    query = f"about: {question[:80]}"
    steps = [{"action": "search", "action_input": query, "observation": obs}]
    msgs = pb.build_full_prompt(question, steps)
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)


def margin_from_logits(logits, tool_ids, fin_ids):
    import torch
    lp = torch.log_softmax(logits, dim=-1)
    return (torch.logsumexp(lp[tool_ids], 0) -
            torch.logsumexp(lp[fin_ids], 0)).item()


def compute_margin(model, tok, prompt, direction, rho, layer, tool_ids, fin_ids):
    import torch
    device = next(model.parameters()).device
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    if direction is not None and abs(rho) > 1e-8:
        d_rms = float(np.sqrt(np.mean(direction ** 2)))
        alpha = rho * HIDDEN_RMS / max(d_rms, 1e-8)
        with SteeringHook(model, direction, alpha, layer=layer,
                          position=-1, max_interventions=1):
            with torch.no_grad():
                logits = model(ids).logits[0, -1, :]
    else:
        with torch.no_grad():
            logits = model(ids).logits[0, -1, :]
    return margin_from_logits(logits, tool_ids, fin_ids)


def boot_ratio_ci(num, denom, B=2000, level=95.0, rng=None):
    rng = rng if rng is not None else np.random.default_rng(SEED)
    n = len(num)
    eps = 1e-12
    if n <= 1:
        v = float(abs(num.mean()) / (abs(denom.mean()) + eps))
        return {"point": v, "ci_low": v, "ci_high": v, "n": int(n)}
    idx = rng.integers(0, n, size=(B, n))
    nb = num[idx].mean(axis=1); db = denom[idx].mean(axis=1)
    rb = np.abs(nb) / (np.abs(db) + eps)
    lo, hi = np.percentile(rb, [(100 - level) / 2, 100 - (100 - level) / 2])
    return {"point": float(abs(num.mean()) / (abs(denom.mean()) + eps)),
            "ci_low": float(lo), "ci_high": float(hi), "n": int(n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/extractability_support_toggle/pairs.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out-dir", default="results/ocft/d3_native_lgm")
    ap.add_argument("--full-dir", default="steering/directions/direction_decomp_full_layer20.npz")
    ap.add_argument("--par-dir", default="steering/directions/direction_decomp_parallel_D3_candidate_present_layer20.npz")
    ap.add_argument("--perp-dir", default="steering/directions/direction_decomp_perp_D3_candidate_present_layer20.npz")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[init] L{LAYER} ρ={RHO} hidden_rms={HIDDEN_RMS}")
    print(f"[load] directions …")
    d_full, _ = load_direction(args.full_dir, normalize_rms=1.0)
    d_par,  _ = load_direction(args.par_dir,  normalize_rms=1.0)
    d_perp, _ = load_direction(args.perp_dir, normalize_rms=1.0)
    print(f"  full RMS={np.sqrt(np.mean(d_full**2)):.3f}  "
          f"par RMS={np.sqrt(np.mean(d_par**2)):.3f}  "
          f"perp RMS={np.sqrt(np.mean(d_perp**2)):.3f}")

    pairs = [json.loads(l) for l in open(args.pairs)]
    print(f"[pairs] N={len(pairs)}  cells: "
          f"{ {c: sum(1 for p in pairs if p['condition']==c) for c in ('N0','T0','T1','S0')} }")

    print(f"\n[load] {args.model}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True).eval()

    tool_ids = [tok.encode(t, add_special_tokens=False)[0]
                for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0]
                for t in ACTION_TOKENS["finish"]]

    n = len(pairs)
    base = np.zeros(n); full_m = np.zeros(n); par_m = np.zeros(n); perp_m = np.zeros(n)
    cell = np.array([p["condition"] for p in pairs])

    t0 = time.time()
    for i, rec in enumerate(pairs):
        p = build_p0_prompt(tok, rec["question"], rec["obs"][:1500])
        base[i]   = compute_margin(model, tok, p, None,  0.0, LAYER, tool_ids, fin_ids)
        full_m[i] = compute_margin(model, tok, p, d_full, RHO, LAYER, tool_ids, fin_ids)
        par_m[i]  = compute_margin(model, tok, p, d_par,  RHO, LAYER, tool_ids, fin_ids)
        perp_m[i] = compute_margin(model, tok, p, d_perp, RHO, LAYER, tool_ids, fin_ids)
        if (i+1) % 10 == 0 or i == 0:
            eta = (time.time()-t0)/(i+1) * (n-i-1)
            print(f"  [{i+1:>3d}/{n}]  cell={rec['condition']}  ETA={eta:.0f}s")
    print(f"[done] forwards: {time.time()-t0:.0f}s")

    full_sh = full_m - base; par_sh = par_m - base; perp_sh = perp_m - base
    np.savez(out_dir / "per_example_shifts_D3_lgm.npz",
             baseline=base, full=full_sh, parallel=par_sh, perp=perp_sh,
             cell=cell, sample_ids=np.array([p["sample_id"] for p in pairs]))

    # Per-cell ratio
    rng = np.random.default_rng(SEED)
    rows = {}
    for c in ("N0", "T0", "T1", "S0", "all", "candidate_present", "candidate_absent"):
        if c == "all":
            mask = np.ones(n, bool)
        elif c == "candidate_present":
            mask = np.isin(cell, ["T0", "T1", "S0"])  # any cell with W in obs
        elif c == "candidate_absent":
            mask = cell == "N0"
        else:
            mask = cell == c
        rows[c] = {
            "n": int(mask.sum()),
            "mean_full": float(full_sh[mask].mean()),
            "mean_par":  float(par_sh[mask].mean()),
            "mean_perp": float(perp_sh[mask].mean()),
            "ratio_par_over_full_abs":
                boot_ratio_ci(par_sh[mask], full_sh[mask], rng=rng),
            "ratio_perp_over_full_abs":
                boot_ratio_ci(perp_sh[mask], full_sh[mask], rng=rng),
        }
    with open(out_dir / "per_cell_lgm.json", "w") as f:
        json.dump({"config": {"layer": LAYER, "rho": RHO,
                              "hidden_rms": HIDDEN_RMS,
                              "model": args.model, "N": n},
                   "cells": rows}, f, indent=2, default=float)

    print("\n=== Per-cell |par|/|full| ratio (D3 on its NATIVE LGM corpus) ===")
    print(f"{'cell':>20s}  {'n':>4s}  {'Δm_full':>10s}  {'Δm_par':>10s}  "
          f"{'|par|/|full|':>14s}  {'95% CI':>20s}")
    for c, r in rows.items():
        ci = r["ratio_par_over_full_abs"]
        print(f"{c:>20s}  {r['n']:>4d}  {r['mean_full']:>+10.3f}  "
              f"{r['mean_par']:>+10.3f}  {ci['point']:>14.3f}  "
              f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]")
    print(f"\n[done] wrote {out_dir}/")


if __name__ == "__main__":
    main()
