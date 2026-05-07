#!/usr/bin/env python3
"""Gemma-2-9B-it sparse residual formation sweep.

Cross-prompt residual patching: at the LAST TOKEN of layers[L].output, copy
the activation from sf_tm (evidence present) into dist_tm (evidence absent).
Measure first-token margin (Action vs Final) shift. This locates where in
depth the evidence→action information becomes present in the residual stream.

Adapts patch_L20_localise_upstream_sweep.py for Gemma:
  - chat template safety (system→user merge)
  - sparse layer set across Gemma's 42 layers
  - same MultiSitePatcher mechanics

Pairs: results/anti_cue_tm_n100/pairs.jsonl (HotpotQA, model-agnostic).
Output: results/gemma_circuit_sanity/exp1_residual_sweep_<tag>/
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS  # noqa: E402
from steering.hook_utils import get_model_layers          # noqa: E402
from scripts.patch_L20_localise_full_residual import (    # noqa: E402
    MultiSitePatcher, make_margin_ids, margin_from_logits,
    perm_p_paired, boot_ci,
)
from scripts.cross_model_full import apply_chat_template_safe  # noqa: E402

# Gemma 42-layer sparse set (depth: 0 25 40 50 60 65 70 75 80 90 100 %)
GEMMA_SPARSE_LAYERS = [0, 10, 16, 21, 25, 27, 29, 31, 33, 37, 41]


def build_prompt_chat_safe(builder, tok, rec):
    steps = [{"action": "search", "action_input": f"about: {rec['question'][:80]}",
              "observation": rec["obs"]}]
    msgs = builder.build_full_prompt(rec["question"], steps)
    return apply_chat_template_safe(tok, msgs, add_generation_prompt=True)


def do_forward(model, tok, prompt, device, tool_ids, fin_ids):
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(ids)
    return margin_from_logits(out.logits[0, -1, :], tool_ids, fin_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/anti_cue_tm_n100/pairs.jsonl")
    ap.add_argument("--out-dir", default="results/gemma_circuit_sanity/exp1_residual_sweep")
    ap.add_argument("--model-path", default="unsloth/gemma-2-9b-it")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--layers", default=",".join(str(L) for L in GEMMA_SPARSE_LAYERS),
                    help="comma-separated layer indices (default: Gemma sparse set)")
    ap.add_argument("--mismatched-perm", default=None,
                    help="JSON file with {'map': {sid: donor_sid}} for mismatched control")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    layer_idxs = [int(L) for L in args.layers.split(",")]

    records = [json.loads(l) for l in open(args.pairs)]
    sids_all = sorted(set(r["sample_id"] for r in records))
    sids = sids_all[:args.limit] if args.limit else sids_all
    need = {("sf", "task_missingness"), ("distractor", "task_missingness")}
    by_sid = {s: {} for s in sids}
    for r in records:
        if r["sample_id"] in by_sid and (r["target"], r["cue"]) in need:
            by_sid[r["sample_id"]][(r["target"], r["cue"])] = r
    # Drop sids missing either cell.
    sids = [s for s in sids if len(by_sid[s]) == 2]
    print(f"[info] N={len(sids)} samples, layers={layer_idxs}")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"[info] loading {args.model_path} dtype={args.dtype}")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True)
    model.eval(); device = next(model.parameters()).device
    layers = get_model_layers(model)
    n_layers = len(layers)
    bad = [L for L in layer_idxs if L < 0 or L >= n_layers]
    if bad: raise SystemExit(f"layer(s) out of range [0,{n_layers}): {bad}")
    SITES = {f"layer_out_L{L}": layers[L] for L in layer_idxs}
    tool_ids, fin_ids = make_margin_ids(tok)
    builder = PromptBuilder()
    print(f"[info] tool_ids={tool_ids}  fin_ids={fin_ids}  n_layers={n_layers}")

    patcher = MultiSitePatcher(SITES)
    natural = {s: {} for s in sids}

    # Stage 1: natural captures (all sites in one forward per (sid, cell))
    t0 = time.time()
    for i, s in enumerate(sids):
        for cell in [("sf", "task_missingness"), ("distractor", "task_missingness")]:
            prompt = build_prompt_chat_safe(builder, tok, by_sid[s][cell])
            patcher.reset_run()
            with patcher:
                m = do_forward(model, tok, prompt, device, tool_ids, fin_ids)
            natural[s][cell] = {"margin": m, "prompt": prompt,
                                "activations": dict(patcher.captured)}
        if (i + 1) % 10 == 0 or i + 1 == len(sids):
            print(f"  [stage1 {i+1}/{len(sids)}] {time.time()-t0:.1f}s")

    donor_map = None
    if args.mismatched_perm:
        donor_map = json.load(open(args.mismatched_perm)).get("map", {})
        miss = [s for s in sids if s not in donor_map or donor_map[s] not in natural]
        if miss:
            raise SystemExit(f"mismatched-perm missing donors for {len(miss)} sids")
        print(f"[info] mismatched-donor mode: {len(donor_map)} entries")

    # Stage 2: per-layer sf_tm -> dist_tm patch
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
                    m_p = do_forward(model, tok, tgt["prompt"], device, tool_ids, fin_ids)
                row = {
                    "sample_id": s, "donor_sid": donor_sid, "layer": L, "site": site,
                    "margin_source_sf_tm":   src["margin"],
                    "margin_target_dist_tm": tgt["margin"],
                    "margin_patched":        m_p,
                    "delta_margin":          m_p - tgt["margin"],
                    "locality_gap":          src["margin"] - tgt["margin"],
                    "action_target_natural": "search" if tgt["margin"] > 0 else "stop",
                    "action_patched":        "search" if m_p > 0 else "stop",
                }
                f.write(json.dumps(row) + "\n"); f.flush(); n_written += 1
            print(f"  [stage2 layer {L} ({li+1}/{len(layer_idxs)})] "
                  f"{time.time()-t0:.1f}s rows={n_written}")
    print(f"[wrote] {rows_path}  ({n_written} rows)")

    summarize(rows_path, out_dir / "summary.json", layer_idxs, args)


def summarize(in_path, out_json, layer_idxs, args):
    rows = [json.loads(l) for l in open(in_path)]
    by_L = {L: [] for L in layer_idxs}
    for r in rows: by_L[r["layer"]].append(r)
    summary = {"model": args.model_path, "pairs": args.pairs,
               "mismatched_perm": args.mismatched_perm, "layers": {}}
    for L, rs in by_L.items():
        d = np.array([r["delta_margin"] for r in rs])
        gaps = np.array([r["locality_gap"] for r in rs])
        mask = gaps > 0.5
        rec = np.array([r["delta_margin"] / r["locality_gap"]
                        if abs(r["locality_gap"]) > 0.01 else np.nan for r in rs])
        rec_pos = rec[mask & np.isfinite(rec)]
        lo_d, hi_d = boot_ci(d)
        rlo, rhi = (boot_ci(rec_pos) if len(rec_pos) >= 5 else (float("nan"), float("nan")))
        flips_s2s   = sum(1 for r in rs if r["action_target_natural"] == "stop"   and r["action_patched"] == "search")
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
    print(f"\n=== Gemma residual sweep (sf_tm -> dist_tm, N={len(base_rows)}) ===")
    print(f"natural gap = {summary['natural_baseline']['locality_gap_mean']:+.3f}")
    print(f"{'L':>3s} {'Δmargin':>8s} {'CI95':>22s} {'perm_p':>7s} {'recovery':>8s} {'rec_CI':>22s}  flips(s→S/S→s)")
    for L in layer_idxs:
        s = summary["layers"][L]
        ci = s["delta_margin_ci95"]; rci = s["recovery_pos_gap_ci95"]
        print(f"{L:>3d} {s['delta_margin_mean']:+8.3f} "
              f"[{ci[0]:+7.3f},{ci[1]:+7.3f}] {s['perm_p_paired_two_sided']:.4f} "
              f"{s['recovery_pos_gap_mean']:+8.3f} [{rci[0]:+7.3f},{rci[1]:+7.3f}]  "
              f"{s['flip_stop_to_search']:>2d}/{s['flip_search_to_stop']:>2d}")


if __name__ == "__main__":
    main()
