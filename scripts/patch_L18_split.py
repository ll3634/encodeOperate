#!/usr/bin/env python3
"""Split the +0.547 L18 layer-output recovery into attn vs mlp.

The upstream sweep (`patch_L20_localise_upstream_sweep.py`) showed the
task_missingness decision-token locality margin appears sharply at L18 in the
residual: recovery jumps from ~0% at L17 to +0.547 at L18, then plateaus at L19.

This script asks: of L18's contribution, how much is attn vs mlp?

Same N=100 sf_tm/dist_tm pairs, same MultiSitePatcher mechanics. Conditions:
  pre_only         : layers[17] output[last]               (pre-L18 residual = L17 baseline)
  attn_only        : layers[18].self_attn output[last]      (L18 attention output)
  mlp_only         : layers[18].mlp output[last]            (L18 MLP output)
  pre_plus_attn    : layers[17] + layers[18].self_attn      (additivity probe)
  full_layer       : layers[18] output[last]                (positive control, = +0.547)

Output:
  results/task_missingness_L18_split/patch_results.jsonl
  results/task_missingness_L18_split/summary.json
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.prompts import PromptBuilder
from steering.hook_utils import get_model_layers
from scripts.patch_L20_localise_full_residual import (
    MultiSitePatcher, make_margin_ids, build_prompt, do_forward,
    perm_p_paired, boot_ci,
)


def build_conds(layer: int):
    L = layer
    return [
        ("pre_only",      [f"resid_pre_L{L}"]),
        ("attn_only",     [f"attn_out_L{L}"]),
        ("mlp_only",      [f"mlp_out_L{L}"]),
        ("pre_plus_attn", [f"resid_pre_L{L}", f"attn_out_L{L}"]),
        ("full_layer",    [f"layer_out_L{L}"]),
    ]


# Backwards-compat alias for any caller that still imports the constant
# (Qwen L18 default).
CONDS = build_conds(18)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/anti_cue_tm_n100/pairs.jsonl")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory (e.g. results/attack3_closure/circuit_<model>_L<layer>)")
    ap.add_argument("--model-path", required=True,
                    help="HF model path (e.g. Qwen/Qwen2.5-7B-Instruct, "
                         "mistralai/Mistral-7B-Instruct-v0.3, unsloth/gemma-2-9b-it)")
    ap.add_argument("--layer", type=int, required=True,
                    help="Decoder block index L for the split "
                         "(Qwen=18, Mistral=16, Gemma=25/23 per ledger)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=None)
    return ap.parse_args(argv)


def main():
    args = parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(l) for l in open(args.pairs)]
    sids = sorted(set(r["sample_id"] for r in records))
    if args.limit: sids = sids[:args.limit]
    need = {("sf","task_missingness"), ("distractor","task_missingness")}
    by_sid = {s: {} for s in sids}
    for r in records:
        if r["sample_id"] in by_sid and (r["target"], r["cue"]) in need:
            by_sid[r["sample_id"]][(r["target"], r["cue"])] = r
    print(f"[info] {len(sids)} samples  layer=L{args.layer}  model={args.model_path}")

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
    L = args.layer
    SITES = {
        f"resid_pre_L{L}":  layers[L - 1],
        f"attn_out_L{L}":   layers[L].self_attn,
        f"mlp_out_L{L}":    layers[L].mlp,
        f"layer_out_L{L}":  layers[L],
    }
    conds = build_conds(L)
    tool_ids, fin_ids = make_margin_ids(tok)
    builder = PromptBuilder()

    patcher = MultiSitePatcher(SITES)
    natural = {s: {} for s in sids}

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

    rows_path = out_dir / "patch_results.jsonl"
    n_written = 0; t0 = time.time()
    with open(rows_path, "w") as f:
        for i, s in enumerate(sids):
            src = natural[s][("sf", "task_missingness")]
            tgt = natural[s][("distractor", "task_missingness")]
            for cname, site_list in conds:
                patcher.reset_run()
                for site in site_list:
                    patcher.patch_vecs[site] = src["activations"][site]
                with patcher:
                    m_p = do_forward(model, tok, tgt["prompt"], device, tool_ids, fin_ids, patcher)
                row = {
                    "sample_id": s, "condition": cname, "sites": site_list,
                    "margin_source_sf_tm": src["margin"],
                    "margin_target_dist_tm": tgt["margin"],
                    "margin_patched": m_p,
                    "delta_margin": m_p - tgt["margin"],
                    "locality_gap": src["margin"] - tgt["margin"],
                    "action_target_natural": "search" if tgt["margin"] > 0 else "stop",
                    "action_patched":        "search" if m_p > 0 else "stop",
                }
                f.write(json.dumps(row) + "\n"); f.flush(); n_written += 1
            if (i + 1) % 20 == 0 or i + 1 == len(sids):
                print(f"  [stage2 {i+1}/{len(sids)}] {time.time()-t0:.1f}s  rows={n_written}")
    print(f"[wrote] {rows_path}  ({n_written} rows)")
    summarize(rows_path, out_dir / "summary.json")


def summarize(in_path, out_json):
    rows = [json.loads(l) for l in open(in_path)]
    by_cond = {}
    for r in rows: by_cond.setdefault(r["condition"], []).append(r)
    summary = {"conditions": {}}
    for cond, rs in by_cond.items():
        d = np.array([r["delta_margin"] for r in rs])
        gaps = np.array([r["locality_gap"] for r in rs])
        mask = gaps > 0.5
        rec = np.array([r["delta_margin"] / r["locality_gap"]
                        if abs(r["locality_gap"]) > 0.01 else np.nan for r in rs])
        rec_pos = rec[mask & np.isfinite(rec)]
        lo, hi = boot_ci(d)
        rlo, rhi = (boot_ci(rec_pos) if len(rec_pos) >= 5 else (float("nan"), float("nan")))
        flips_s2s = sum(1 for r in rs if r["action_target_natural"] == "stop"   and r["action_patched"] == "search")
        flips_se2st = sum(1 for r in rs if r["action_target_natural"] == "search" and r["action_patched"] == "stop")
        summary["conditions"][cond] = {
            "n": len(rs), "sites": rs[0]["sites"],
            "delta_margin_mean": float(d.mean()),
            "delta_margin_median": float(np.median(d)),
            "delta_margin_ci95": [lo, hi],
            "perm_p_paired_two_sided": perm_p_paired(d),
            "recovery_pos_gap_mean":   float(rec_pos.mean())   if len(rec_pos) else float("nan"),
            "recovery_pos_gap_median": float(np.median(rec_pos)) if len(rec_pos) else float("nan"),
            "recovery_pos_gap_ci95":   [rlo, rhi],
            "flip_stop_to_search": flips_s2s,
            "flip_search_to_stop": flips_se2st,
        }
    base_rows = by_cond[list(by_cond)[0]]
    summary["natural_baseline"] = {
        "margin_sf_tm_mean":   float(np.mean([r["margin_source_sf_tm"]   for r in base_rows])),
        "margin_dist_tm_mean": float(np.mean([r["margin_target_dist_tm"] for r in base_rows])),
        "locality_gap_mean":   float(np.mean([r["locality_gap"]          for r in base_rows])),
    }
    json.dump(summary, open(out_json, "w"), indent=2)
    print(f"[wrote] {out_json}")
    print(f"\n=== L18 split (sf_tm -> dist_tm, N={len(base_rows)}) ===")
    print(f"natural gap = {summary['natural_baseline']['locality_gap_mean']:+.3f}")
    print(f"{'cond':16s} {'Δmargin':>8s}  {'CI95':>22s}  {'perm_p':>7s}  {'recovery':>8s}  flips(s→S/S→s)")
    for cond, s in summary["conditions"].items():
        ci = s["delta_margin_ci95"]
        print(f"{cond:16s} {s['delta_margin_mean']:+8.3f}  "
              f"[{ci[0]:+7.3f},{ci[1]:+7.3f}]  {s['perm_p_paired_two_sided']:.4f}  "
              f"{s['recovery_pos_gap_mean']:+8.3f}  "
              f"{s['flip_stop_to_search']:>2d}/{s['flip_search_to_stop']:>2d}")


if __name__ == "__main__":
    main()
