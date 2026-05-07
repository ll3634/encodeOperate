#!/usr/bin/env python3
"""Diagnostic: compare new-model baseline against cached baseline AND
verify ProjectionFlipHook actually changes logits when applied with a
known-strong direction (A_hat itself, factor=2.0)."""
import json, sys, os, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from agent.prompts import ACTION_TOKENS
from evidence_erasure_test import (
    ProjectionFlipHook, forward_margin, build_p0_prompt, LAYER, N,
)

MODEL_PATH = os.environ.get("MODEL_PATH", "/home/featurize/work/models/Qwen2.5-7B-Instruct")

def main():
    print(f"[load] {MODEL_PATH}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    labels = [json.loads(l) for l in open("results/phase1_probe/labels.jsonl")]
    bl_map = {}
    with open("results/l20_rho020_n500/baseline_results.jsonl") as f:
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
        if len(prompts) >= 10: break  # only 10 for diagnostic

    cached = np.load("results/evidence_erasure_test/per_prompt_margins.npz")
    cached_sids = list(cached["sample_ids"])
    cached_base = cached["baseline"].astype(np.float32)
    cached_flipA = cached["flip_A"].astype(np.float32)
    cidx = [cached_sids.index(s) for s in sample_ids]
    print(f"[cohort] N={len(prompts)} matched")

    A = np.load("steering/directions/direction_decomp_full_layer20.npz",
                allow_pickle=True)["decision_direction"].astype(np.float32)
    A_hat = A / np.linalg.norm(A)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True).eval()
    tool_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]

    print("\n[per-prompt baseline + flip_A check (10 prompts)]")
    print(f"{'sid':<12s} {'cached_base':>10s} {'new_base':>10s} {'Δbase':>8s}  "
          f"{'cached_flipA':>12s} {'new_flipA':>10s}")
    for j, p in enumerate(prompts):
        b_new = forward_margin(model, tok, p, None, tool_ids, fin_ids)
        f_new = forward_margin(model, tok, p,
            lambda: ProjectionFlipHook(model, A_hat, factor=2.0),
            tool_ids, fin_ids)
        print(f"{sample_ids[j]:<12s} {cached_base[cidx[j]]:>+10.3f} {b_new:>+10.3f} "
              f"{b_new-cached_base[cidx[j]]:>+8.3f}  "
              f"{cached_flipA[cidx[j]]:>+12.3f} {f_new:>+10.3f}")


if __name__ == "__main__":
    main()
