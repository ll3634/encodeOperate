#!/usr/bin/env python3
"""Identity-patch sanity for Gemma-2 hooks under eager attention.

For each patched layer L, on N items: (1) baseline forward, (2) forward with
hook patching layer L's last-token output to its OWN natural activation. Reports
||h_after - h_before|| / ||h_before|| (should be 0; nonzero means hook broken)
and |margin_baseline - margin_patched| (should be 0; nonzero means downstream
state is corrupted, e.g. sliding-window cache mismatch).

Spike at any layer => eager isn't engaged or hook corrupts the residual; STOP
the Gemma circuit run per Phase 0.5 §0.2.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.prompts import PromptBuilder
from steering.hook_utils import get_model_layers
from scripts.patch_L20_localise_full_residual import (
    MultiSitePatcher, make_margin_ids, build_prompt, do_forward,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/anti_cue_tm_n100/pairs.jsonl")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--layers", required=True,
                    help="comma-separated layer indices to test (e.g. 21,22,...,41)")
    ap.add_argument("--n-items", type=int, default=5)
    args = ap.parse_args()

    layer_idxs = [int(x) for x in args.layers.split(",")]
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    p = args.model_path.lower()
    attn_impl = "eager" if "gemma" in p else "sdpa"
    print(f"[info] attn_implementation={attn_impl}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation=attn_impl,
    )
    model.eval()
    actual_attn = model.config._attn_implementation
    assert actual_attn == attn_impl, f"attn_impl rolled back: {actual_attn}"
    print(f"[info] model.config._attn_implementation={actual_attn}", flush=True)
    device = next(model.parameters()).device

    layers = get_model_layers(model)
    SITES = {f"layer_out_L{L}": layers[L] for L in layer_idxs}
    tool_ids, fin_ids = make_margin_ids(tok)
    builder = PromptBuilder()

    recs = [json.loads(l) for l in open(args.pairs)]
    sids = sorted(set(r["sample_id"] for r in recs))[: args.n_items]
    by_sid = {s: {} for s in sids}
    for r in recs:
        if r["sample_id"] in by_sid and r["target"] == "distractor" and r["cue"] == "task_missingness":
            by_sid[r["sample_id"]][("distractor", "task_missingness")] = r

    patcher = MultiSitePatcher(SITES)
    rows = []
    for s in sids:
        rec = by_sid[s][("distractor", "task_missingness")]
        prompt = build_prompt(builder, tok, rec, args.model_path)
        # Stage 1: baseline forward (capture natural)
        patcher.reset_run()
        with patcher:
            m_base = do_forward(model, tok, prompt, device, tool_ids, fin_ids, patcher)
        natural = {site: patcher.captured[site].copy() for site in SITES}
        # Stage 2: identity-patch each site at last token, observe margin + post-patch capture
        for L in layer_idxs:
            site = f"layer_out_L{L}"
            patcher.reset_run()
            patcher.patch_vecs[site] = natural[site]  # IDENTITY patch
            with patcher:
                m_patched = do_forward(model, tok, prompt, device, tool_ids, fin_ids, patcher)
            after = patcher.captured[site]
            before = natural[site]
            denom = float(np.linalg.norm(before)) + 1e-12
            rel = float(np.linalg.norm(after - before)) / denom
            rows.append({
                "sample_id": s, "layer": L,
                "rel_norm_after_minus_before": rel,
                "abs_norm_diff": float(np.linalg.norm(after - before)),
                "norm_before": float(np.linalg.norm(before)),
                "margin_baseline": m_base,
                "margin_patched": m_patched,
                "abs_margin_diff": float(abs(m_base - m_patched)),
            })

    # Summarize per layer
    by_L = {}
    for r in rows:
        by_L.setdefault(r["layer"], []).append(r)
    summary = {"attn_impl": actual_attn, "n_items": len(sids), "layers": {}}
    for L, rs in sorted(by_L.items()):
        rels = [r["rel_norm_after_minus_before"] for r in rs]
        margs = [r["abs_margin_diff"] for r in rs]
        summary["layers"][L] = {
            "rel_norm_max": max(rels), "rel_norm_mean": float(np.mean(rels)),
            "abs_margin_diff_max": max(margs),
            "abs_margin_diff_mean": float(np.mean(margs)),
        }
    summary["rows"] = rows
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)

    print(f"[wrote] {args.out}")
    print(f"\n=== identity-patch hook sanity (N={len(sids)}) ===")
    print(f"{'L':>3s}  {'rel_norm_max':>12s}  {'abs_margin_diff_max':>20s}")
    bad = 0
    for L, s in summary["layers"].items():
        flag = ""
        if s["rel_norm_max"] > 1e-3 or s["abs_margin_diff_max"] > 1e-3:
            flag = "  <-- SPIKE"
            bad += 1
        print(f"{L:>3d}  {s['rel_norm_max']:>12.6f}  {s['abs_margin_diff_max']:>20.6f}{flag}")
    if bad:
        print(f"\nFAIL: {bad}/{len(summary['layers'])} layers spike under identity patch.")
        sys.exit(2)
    print("\nPASS: hook is identity on all tested layers.")


if __name__ == "__main__":
    main()
