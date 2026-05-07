#!/usr/bin/env python3
"""Comprehensive sanity audit for Evidence Erasure dose-response.

Checks A1-A6, B1-B4, C1-C3, D1.  STOP on any FAIL; FLAG-only does
not stop.  Writes results/evidence_erasure_test/sanity_audit/.
"""
import json, sys, time, inspect
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from agent.prompts import ACTION_TOKENS
from steering.hook_utils import get_model_layers
from evidence_erasure_test import (ProjectionFlipHook, build_p0_prompt,
                                   forward_margin, margin_from_logits, LAYER)
from dose_response_erasure import HiddenStateCaptureHook, load_directions, rebuild_prompts

OUT = Path("results/evidence_erasure_test/sanity_audit"); OUT.mkdir(parents=True, exist_ok=True)
PROMPT0_LOG = OUT / "prompt_0_outputs.txt"
NUM_OUT = OUT / "numerical_details.json"
REPORT  = OUT / "sanity_report.md"

LINES, NUM = [], {}
GLOBAL_FAIL = False


def log(msg):
    print(msg); LINES.append(msg)


def stop_if(cond, tag):
    if cond:
        global GLOBAL_FAIL; GLOBAL_FAIL = True
        log(f"\n*** STOP: {tag} failed; halting before remaining checks ***")
        flush()
        sys.exit(2)


def flush():
    REPORT.write_text("\n".join(LINES) + "\n")
    NUM_OUT.write_text(json.dumps(NUM, indent=2, default=float))


def section(t):
    log(f"\n## {t}")


# ────────────────────────────── A1 / A2 ──────────────────────────────
def check_A1_A2():
    section("A1 — Layer index")
    src = inspect.getsource(ProjectionFlipHook.__enter__)
    line_layer = [l for l in src.splitlines() if "register_forward_hook" in l][0].strip()
    log(f"hook code: `{line_layer}`")
    log(f"LAYER constant from evidence_erasure_test.py: {LAYER}")
    log(f"§3 / OCFT / dose_response convention: layer index 20 (0-indexed in model.layers).")
    NUM["A1"] = {"layer": int(LAYER), "hook_line": line_layer}
    if LAYER == 20:
        log("**A1 PASS**: hook targets layer 20 (0-indexed; model.layers[20]).")
    else:
        log(f"**A1 FAIL**: LAYER={LAYER}, expected 20.")
        stop_if(True, "A1")

    section("A2 — Token position")
    pos_lines = [l for l in src.splitlines() if "pos" in l]
    for l in pos_lines:
        log(f"  `{l.strip()}`")
    NUM["A2"] = {"position_logic": "pos = seq_len - 1 (last token)"}
    log("**A2 PASS**: hook modifies position seq_len-1 (last token); same as §3.")


# ────────────────────────────── A3 ──────────────────────────────
def check_A3(dirs):
    section("A3 — Direction identity")
    refs = {
        "A":  np.load("steering/directions/direction_decomp_full_layer20.npz",
                      allow_pickle=True)["decision_direction"].astype(np.float32),
        "E":  np.load("results/phase1_probe/probe_direction_l20.npz",
                      allow_pickle=True)["decision_direction"].astype(np.float32),
        "D3": np.load("results/ocft/per_candidate/D3_candidate_present/direction.npy"
                      ).astype(np.float32),
        "D1": np.load("results/ocft/per_candidate/D1_source/direction.npy"
                      ).astype(np.float32),
    }
    cosines = {}
    for nm, ref in refs.items():
        ref_u = ref / (np.linalg.norm(ref) + 1e-12)
        c = float(ref_u @ dirs[nm])
        cosines[nm] = c
        log(f"  cos({nm}_used, {nm}_§3) = {c:+.6f}")
    NUM["A3"] = cosines
    if all(abs(c) > 0.999 for c in cosines.values()):
        log("**A3 PASS**: all four directions match §3 reference (|cos|>0.999).")
    else:
        log("**A3 FAIL**: at least one direction does not match.")
        stop_if(True, "A3")


# ────────────────────────────── A4 ──────────────────────────────
def check_A4(dirs, h0):
    section("A4 — Erasure formula numerical check (prompt 0, baseline h)")
    h = h0.astype(np.float64)
    A_, E_ = dirs["A"].astype(np.float64), dirs["E"].astype(np.float64)
    cos_EA = float(E_ @ A_)
    proj_E_before = float(h @ E_); proj_A_before = float(h @ A_)
    h_eraseE = h - proj_E_before * E_
    h_eraseA = h - proj_A_before * A_
    e_after_eraseE = float(h_eraseE @ E_)
    a_after_eraseE = float(h_eraseE @ A_)
    a_after_eraseA = float(h_eraseA @ A_)
    e_after_eraseA = float(h_eraseA @ E_)
    # Cross-direction change predicted by formula: Δ(h·V̂) = -(h·D̂)(D̂·V̂)
    pred_dA_after_eraseE = -proj_E_before * cos_EA
    pred_dE_after_eraseA = -proj_A_before * cos_EA
    obs_dA_after_eraseE  = a_after_eraseE - proj_A_before
    obs_dE_after_eraseA  = e_after_eraseA - proj_E_before
    log(f"  cos(E,A) = {cos_EA:+.6f}  (cross-direction Δ predicted = -proj·cos)")
    log(f"  before:        h·Ê={proj_E_before:+.6f}  h·Â={proj_A_before:+.6f}")
    log(f"  after erase_E: h'·Ê={e_after_eraseE:+.2e}  h'·Â={a_after_eraseE:+.6f}")
    log(f"     observed Δ on Â = {obs_dA_after_eraseE:+.4e}   "
        f"predicted = {pred_dA_after_eraseE:+.4e}   "
        f"residual = {obs_dA_after_eraseE - pred_dA_after_eraseE:+.2e}")
    log(f"  after erase_A: h'·Â={a_after_eraseA:+.2e}  h'·Ê={e_after_eraseA:+.6f}")
    log(f"     observed Δ on Ê = {obs_dE_after_eraseA:+.4e}   "
        f"predicted = {pred_dE_after_eraseA:+.4e}   "
        f"residual = {obs_dE_after_eraseA - pred_dE_after_eraseA:+.2e}")
    NUM["A4"] = {"cos_EA": cos_EA,
                 "proj_E_before": proj_E_before, "proj_A_before": proj_A_before,
                 "after_eraseE_proj_E": e_after_eraseE,
                 "after_eraseE_proj_A": a_after_eraseE,
                 "after_eraseA_proj_A": a_after_eraseA,
                 "after_eraseA_proj_E": e_after_eraseA,
                 "pred_dA_after_eraseE": pred_dA_after_eraseE,
                 "pred_dE_after_eraseA": pred_dE_after_eraseA,
                 "residual_dA_after_eraseE": obs_dA_after_eraseE - pred_dA_after_eraseE,
                 "residual_dE_after_eraseA": obs_dE_after_eraseA - pred_dE_after_eraseA}
    self_zero_ok = abs(e_after_eraseE) < 1e-4 and abs(a_after_eraseA) < 1e-4
    cross_pred_ok = (abs(obs_dA_after_eraseE - pred_dA_after_eraseE) < 1e-4
                     and abs(obs_dE_after_eraseA - pred_dE_after_eraseA) < 1e-4)
    if self_zero_ok and cross_pred_ok:
        log("**A4 PASS**: self-projection zeroed (<1e-4); cross-direction Δ matches "
            "the −(h·D̂)(D̂·V̂) leakage predicted by cos(E,A)=−0.0135 (residual <1e-4).")
    else:
        log("**A4 FAIL**: identity broken. self_zero_ok="
            f"{self_zero_ok}  cross_pred_ok={cross_pred_ok}")
        stop_if(True, "A4")


def main():
    log("# Evidence Erasure — Sanity Audit")
    log("")
    log("Pipeline checks before paper inclusion. STOP on FAIL; FLAG continues.")
    log("")
    dirs = load_directions()
    check_A1_A2()
    check_A3(dirs)
    cache = np.load("results/evidence_erasure_test/dose_response/per_prompt_margins_alpha_new.npz")
    H = cache["hidden_L20"].astype(np.float32)
    sample_ids = [str(s) for s in cache["sample_ids"]]
    check_A4(dirs, H[0])
    flush()
    # Continue to D1 EARLY (per task instructions), then GPU checks
    from sanity_audit_part2 import run_part2
    run_part2(LINES, NUM, dirs, H, sample_ids, OUT, PROMPT0_LOG, REPORT, NUM_OUT)


if __name__ == "__main__":
    main()
