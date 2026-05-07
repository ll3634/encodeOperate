#!/usr/bin/env python3
"""Sweep upstream layer-output patches to localise where the L20 decision-token
locality content enters the residual.

Same N=100 (sf_tm, dist_tm) pairs and same patcher mechanics as
patch_L20_localise_full_residual.py. For each layer L in --layer-range, patch
sf_tm -> dist_tm at the LAST TOKEN of `layers[L]` output (= post-layer-L
residual = everything accumulated by layers 0..L at the decision token), then
read the margin off the LM head.

Recovery as a function of L tells us at which layer the locality content
becomes "ready":
  * L=0 baseline: only embeddings differ at the patched token (~0% expected)
  * L=19 == the existing pre_only condition (~0.537 expected)
  * monotone rise between → localisation curve

Output:
  results/task_missingness_L20_split_upstream/patch_results.jsonl
  results/task_missingness_L20_split_upstream/summary.json
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import get_model_layers
from scripts.patch_L20_localise_full_residual import (
    MultiSitePatcher, make_margin_ids, build_prompt, do_forward,
    perm_p_paired, boot_ci,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/anti_cue_tm_n100/pairs.jsonl")
    ap.add_argument("--out-dir", default="results/task_missingness_L20_split_upstream")
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--layer-range", default="0:20",
                    help="python-style start:stop for layers to sweep (default 0:20 = 0..19)")
    ap.add_argument("--mismatched-perm", default=None,
                    help="JSON with {'map': {sid: donor_sid}}; if set, donor sf_tm "
                         "comes from map[sid] instead of from sid (specificity control)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    lo, hi = (int(x) for x in args.layer_range.split(":"))
    layer_idxs = list(range(lo, hi))

    records = [json.loads(l) for l in open(args.pairs)]
    sids = sorted(set(r["sample_id"] for r in records))
    if args.limit: sids = sids[:args.limit]
    need = {("sf","task_missingness"), ("distractor","task_missingness")}
    by_sid = {s: {} for s in sids}
    for r in records:
        if r["sample_id"] in by_sid and (r["target"], r["cue"]) in need:
            by_sid[r["sample_id"]][(r["target"], r["cue"])] = r
    print(f"[info] {len(sids)} samples, sweeping layers {layer_idxs[0]}..{layer_idxs[-1]}")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    p = args.model_path.lower()
    attn_impl = "eager" if "gemma" in p else "sdpa"
    print(f"[info] attn_implementation={attn_impl}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype,
                                                 device_map="auto", trust_remote_code=True,
                                                 attn_implementation=attn_impl)
    model.eval(); device = next(model.parameters()).device
    layers = get_model_layers(model)
    SITES = {f"layer_out_L{L}": layers[L] for L in layer_idxs}
    tool_ids, fin_ids = make_margin_ids(tok)
    builder = PromptBuilder()

    patcher = MultiSitePatcher(SITES)
    natural = {s: {} for s in sids}

    # Stage 1: natural captures for sf_tm and dist_tm at every swept layer
    t0 = time.time()
    for i, s in enumerate(sids):
        for cell in [("sf", "task_missingness"), ("distractor", "task_missingness")]:
            prompt = build_prompt(builder, tok, by_sid[s][cell], args.model_path)
            patcher.reset_run()
            with patcher:
                m = do_forward(model, tok, prompt, device, tool_ids, fin_ids, patcher)
            natural[s][cell] = {"margin": m, "prompt": prompt,
                                "activations": dict(patcher.captured)}
        if (i + 1) % 25 == 0 or i + 1 == len(sids):
            print(f"  [stage1 {i+1}/{len(sids)}] {time.time()-t0:.1f}s")

    donor_map = None
    if args.mismatched_perm:
        with open(args.mismatched_perm) as fp:
            donor_map = json.load(fp).get("map", {})
        missing = [s for s in sids if s not in donor_map or donor_map[s] not in natural]
        if missing:
            raise SystemExit(f"mismatched-perm missing donors for {len(missing)} sids "
                             f"(first: {missing[:3]})")
        print(f"[info] mismatched-donor mode: {len(donor_map)} entries from {args.mismatched_perm}")

    # Stage 2: for each layer, patch sf_tm -> dist_tm
    rows_path = out_dir / "patch_results.jsonl"
    n_written = 0; t0 = time.time()
    with open(rows_path, "w") as f:
        for li, L in enumerate(layer_idxs):
            site = f"layer_out_L{L}"
            for s in sids:
                donor_sid = donor_map[s] if donor_map else s
                src = natural[donor_sid][("sf", "task_missingness")]
                tgt = natural[s][("distractor", "task_missingness")]
                patcher.reset_run()
                patcher.patch_vecs[site] = src["activations"][site]
                with patcher:
                    m_p = do_forward(model, tok, tgt["prompt"], device, tool_ids, fin_ids, patcher)
                row = {
                    "sample_id": s, "donor_sid": donor_sid, "layer": L, "site": site,
                    "margin_source_sf_tm": src["margin"],
                    "margin_target_dist_tm": tgt["margin"],
                    "margin_patched": m_p,
                    "delta_margin": m_p - tgt["margin"],
                    "locality_gap": src["margin"] - tgt["margin"],
                    "action_target_natural": "search" if tgt["margin"] > 0 else "stop",
                    "action_patched":        "search" if m_p > 0 else "stop",
                }
                f.write(json.dumps(row) + "\n"); f.flush(); n_written += 1
            print(f"  [stage2 layer {L} ({li+1}/{len(layer_idxs)})] {time.time()-t0:.1f}s  rows={n_written}")
    print(f"[wrote] {rows_path}  ({n_written} rows)")

    summarize(rows_path, out_dir / "summary.json", layer_idxs)


def summarize(in_path, out_json, layer_idxs):
    rows = [json.loads(l) for l in open(in_path)]
    by_L = {L: [] for L in layer_idxs}
    for r in rows: by_L[r["layer"]].append(r)

    summary = {"layers": {}}
    for L, rs in by_L.items():
        d = np.array([r["delta_margin"] for r in rs])
        gaps = np.array([r["locality_gap"] for r in rs])
        mask = gaps > 0.5
        rec = np.array([r["delta_margin"] / r["locality_gap"]
                        if abs(r["locality_gap"]) > 0.01 else np.nan for r in rs])
        rec_pos = rec[mask & np.isfinite(rec)]
        lo_d, hi_d = boot_ci(d)
        rlo, rhi = (boot_ci(rec_pos) if len(rec_pos) >= 5 else (float("nan"), float("nan")))
        flips_s2s = sum(1 for r in rs if r["action_target_natural"] == "stop"   and r["action_patched"] == "search")
        flips_se2st = sum(1 for r in rs if r["action_target_natural"] == "search" and r["action_patched"] == "stop")
        summary["layers"][L] = {
            "n": len(rs),
            "delta_margin_mean": float(d.mean()),
            "delta_margin_median": float(np.median(d)),
            "delta_margin_ci95": [lo_d, hi_d],
            "perm_p_paired_two_sided": perm_p_paired(d),
            "recovery_pos_gap_n": int(mask.sum()),
            "recovery_pos_gap_mean":   float(rec_pos.mean())   if len(rec_pos) else float("nan"),
            "recovery_pos_gap_median": float(np.median(rec_pos)) if len(rec_pos) else float("nan"),
            "recovery_pos_gap_ci95":   [rlo, rhi],
            "flip_stop_to_search": flips_s2s,
            "flip_search_to_stop": flips_se2st,
        }
    base_rows = by_L[layer_idxs[0]]
    summary["natural_baseline"] = {
        "margin_sf_tm_mean":   float(np.mean([r["margin_source_sf_tm"]   for r in base_rows])),
        "margin_dist_tm_mean": float(np.mean([r["margin_target_dist_tm"] for r in base_rows])),
        "locality_gap_mean":   float(np.mean([r["locality_gap"]          for r in base_rows])),
    }
    json.dump(summary, open(out_json, "w"), indent=2)
    print(f"[wrote] {out_json}")
    print(f"\n=== upstream sweep (sf_tm -> dist_tm, N={len(base_rows)}) ===")
    print(f"natural gap = {summary['natural_baseline']['locality_gap_mean']:+.3f}")
    print(f"{'L':>3s}  {'Δmargin':>8s}  {'CI95':>22s}  {'perm_p':>7s}  {'recovery':>8s}  {'rec_CI':>22s}  flips(s→S/S→s)")
    for L in layer_idxs:
        s = summary["layers"][L]
        ci = s["delta_margin_ci95"]; rci = s["recovery_pos_gap_ci95"]
        print(f"{L:>3d}  {s['delta_margin_mean']:+8.3f}  "
              f"[{ci[0]:+7.3f},{ci[1]:+7.3f}]  {s['perm_p_paired_two_sided']:.4f}  "
              f"{s['recovery_pos_gap_mean']:+8.3f}  [{rci[0]:+7.3f},{rci[1]:+7.3f}]  "
              f"{s['flip_stop_to_search']:>2d}/{s['flip_search_to_stop']:>2d}")


if __name__ == "__main__":
    main()
