#!/usr/bin/env python3
"""Part 2 of sanity audit: D1 swap test (early), A5/A6, B1-B4, C1-C3."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from agent.prompts import ACTION_TOKENS
from evidence_erasure_test import (ProjectionFlipHook, build_p0_prompt,
                                   forward_margin, margin_from_logits, LAYER)
from dose_response_erasure import rebuild_prompts


def run_part2(LINES, NUM, dirs, H, sample_ids, OUT, P0LOG, REPORT, NUM_OUT):
    GLOBAL_FAIL = [False]

    def log(msg):
        print(msg); LINES.append(msg)

    def flush():
        REPORT.write_text("\n".join(LINES) + "\n")
        NUM_OUT.write_text(json.dumps(NUM, indent=2, default=float))

    def stop(tag):
        GLOBAL_FAIL[0] = True
        log(f"\n*** STOP: {tag} failed ***"); flush(); sys.exit(2)

    def section(t):
        log(f"\n## {t}")

    # Load model + tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log("\n[load] Qwen/Qwen2.5-7B-Instruct (for GPU checks)")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    prompts = rebuild_prompts(tok, sample_ids)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True).eval()
    tool_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]

    cache = np.load("results/evidence_erasure_test/per_prompt_margins.npz")
    base_cached = cache["baseline"].astype(np.float32)
    erase_E_c = cache["erase_E"].astype(np.float32)
    erase_A_c = cache["erase_A"].astype(np.float32)

    # ── D1 swap test (CRITICAL, early) ──
    section("D1 — Direction-swap pipeline test (CRITICAL)")
    p0 = prompts[0]
    swap_E_via_E = forward_margin(model, tok, p0,
        lambda v=dirs["E"]: ProjectionFlipHook(model, v, factor=1.0),
        tool_ids, fin_ids)
    swap_E_via_A = forward_margin(model, tok, p0,
        lambda v=dirs["A"]: ProjectionFlipHook(model, v, factor=1.0),
        tool_ids, fin_ids)
    log(f"  prompt 0 erase using direction=E (matches cached erase_E): {swap_E_via_E:+.4f}  "
        f"vs cached erase_E[0]={erase_E_c[0]:+.4f}")
    log(f"  prompt 0 erase using direction=A (matches cached erase_A): {swap_E_via_A:+.4f}  "
        f"vs cached erase_A[0]={erase_A_c[0]:+.4f}")
    NUM["D1"] = {"swap_E_via_E": swap_E_via_E, "cached_eraseE_p0": float(erase_E_c[0]),
                 "swap_E_via_A": swap_E_via_A, "cached_eraseA_p0": float(erase_A_c[0])}
    matches_E = abs(swap_E_via_E - erase_E_c[0]) < 1e-3
    matches_A = abs(swap_E_via_A - erase_A_c[0]) < 1e-3
    if matches_E and matches_A:
        log("**D1 PASS**: changing the direction vector swaps the effect "
            "(direction is causally what the hook acts on; no hardcoded ablation).")
    else:
        log("**D1 FAIL**: direction vector does not control the effect — pipeline bug.")
        stop("D1")

    # ── A5 alpha application ──
    section("A5 — α=0.50 projection halving + cached-margin reproduction (prompt 0)")
    h0 = H[0].astype(np.float64)
    A_ = dirs["A"].astype(np.float64); proj0 = float(h0 @ A_)
    h_half = h0 - 0.50 * proj0 * A_
    proj_after = float(h_half @ A_)
    expected = 0.50 * proj0
    log(f"  before: h·Â={proj0:+.6f};  after α=0.50: h'·Â={proj_after:+.6f}; "
        f"expected={expected:+.6f}")
    cached_a050 = np.load("results/evidence_erasure_test/dose_response/"
                          "per_prompt_margins_alpha_new.npz")["A_alpha0p50"][0]
    re_run = forward_margin(model, tok, p0,
        lambda v=dirs["A"]: ProjectionFlipHook(model, v, factor=0.5),
        tool_ids, fin_ids)
    log(f"  cached A_α=0.50 margin[0]={float(cached_a050):+.4f}  re-run={re_run:+.4f}")
    NUM["A5"] = {"proj_after_half": proj_after, "expected": expected,
                 "cached_margin": float(cached_a050), "re_run_margin": re_run}
    if abs(proj_after - expected) < 1e-5 and abs(re_run - cached_a050) < 1e-3:
        log("**A5 PASS**: α=0.50 halves projection; cached margin reproduces.")
    else:
        log("**A5 FAIL**: α=0.50 does not halve projection or margin not reproducible.")
        stop("A5")

    # ── A6 baseline determinism ──
    section("A6 — Baseline determinism (prompt 0, two runs)")
    m1 = forward_margin(model, tok, p0, None, tool_ids, fin_ids)
    m2 = forward_margin(model, tok, p0, None, tool_ids, fin_ids)
    log(f"  baseline pass 1: {m1:+.6f}    pass 2: {m2:+.6f}    diff: {m2-m1:+.2e}")
    NUM["A6"] = {"m1": m1, "m2": m2}
    if abs(m1 - m2) < 1e-5:
        log("**A6 PASS**: baseline deterministic.")
    else:
        log("**A6 FAIL**: baseline not deterministic.")
        stop("A6")

    # ── B1 norm preservation + energy fractions ──
    section("B1 — Norm preservation & energy fractions (full N=100)")
    ratios = {}; energy = {}
    for nm in ["A", "E", "D3", "D1"]:
        D = dirs[nm].astype(np.float64)
        proj = (H @ D)
        h_after = H - proj[:, None].astype(np.float32) * D[None, :].astype(np.float32)
        norm_h  = np.linalg.norm(H, axis=1)
        norm_h2 = np.linalg.norm(h_after, axis=1)
        nr = float((norm_h2 / (norm_h + 1e-12)).mean())
        ef = float((np.abs(proj) / (norm_h + 1e-12)).mean())
        ratios[nm] = nr; energy[nm] = ef
        log(f"  erase_{nm}: mean ||h'||/||h|| = {nr:.4f}   "
            f"mean |h·D̂|/||h|| = {ef:.4f}")
    NUM["B1"] = {"norm_ratios": ratios, "energy_fractions": energy}
    flag = ratios["A"] < 0.95
    log(f"**B1 {'FLAG' if flag else 'PASS'}**: erase_A removes "
        f"{(1-ratios['A'])*100:.2f}% of h energy "
        f"({'>5% — confound risk' if flag else '<5% — safe'}).")

    # save partial then continue
    flush()
    return run_part3(log, flush, section, stop, NUM, dirs, H, sample_ids,
                     base_cached, erase_E_c, erase_A_c, model, tok,
                     prompts, tool_ids, fin_ids, P0LOG)


def run_part3(*args):
    from sanity_audit_part3 import run_part3 as _r
    return _r(*args)
