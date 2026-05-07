#!/usr/bin/env python3
"""C2-Audit Step 1: Extract Qwen2.5-7B action_dir at L20 via the SAME pipeline
used for Qwen3-32B in cross_model_full.py.

Sweeps L14-L26 every 2 (50%-95% depth in 28 layers, matching the 32B protocol's
L32-L60 every 2 in 64 layers), reuses collect_popqa_multilayer +
extract_action_dir_from_popqa from cross_model_full.py, and saves the L20
action direction (forced per audit brief) plus per-layer Spearman quality so
the auto-detected peak can be cross-checked against the L20 anchor.

Output: results/qwen_7b_normalization_audit/directions_L20.npz
        results/qwen_7b_normalization_audit/sweep_quality.json
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))
from cross_model_full import (  # noqa: E402
    collect_popqa_multilayer,
    extract_action_dir_from_popqa,
)
from steering.hook_utils import get_model_layers  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/featurize/work/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--popqa-path", default="data/popqa/popqa_test.jsonl")
    ap.add_argument("--n-popqa", type=int, default=400)
    ap.add_argument("--target-layer", default="20",
                    help="Forced layer for direction save (audit anchor). "
                         "Use 'auto' to save at the sweep's auto-peak layer.")
    ap.add_argument("--sweep-layers", default=None,
                    help="Comma-separated explicit sweep layer list, e.g. "
                         "'28,30,32,34,36,38,40'. If unset, uses 50%-95% "
                         "depth every 2 layers.")
    ap.add_argument("--out-dir", default="results/qwen_7b_normalization_audit")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True); model.eval()
    n_layers = len(get_model_layers(model))
    D = model.config.hidden_size
    print(f"[ok] n_layers={n_layers} hidden={D}")

    # Bit-faithful to cross_model_full.py: 50%-95% depth, every 2 layers.
    if args.sweep_layers:
        sweep_layers = sorted({int(x) for x in args.sweep_layers.split(",") if x.strip()})
    else:
        sweep_start = int(n_layers * 0.5)
        sweep_end   = int(n_layers * 0.95)
        sweep_layers = list(range(sweep_start, sweep_end, 2))
    forced_anchor = None if str(args.target_layer).lower() == "auto" else int(args.target_layer)
    if forced_anchor is not None and forced_anchor not in sweep_layers:
        sweep_layers = sorted(set(sweep_layers + [forced_anchor]))
    print(f"[sweep] layers={sweep_layers}  target={args.target_layer}")

    print(f"\n=== Collect PopQA states (N={args.n_popqa}, all sweep layers) ===")
    popqa_by_layer = collect_popqa_multilayer(
        model, tok, args.popqa_path, sweep_layers, n=args.n_popqa)

    print("\n=== Per-layer action_dir quality (Spearman |rho|) ===")
    per_layer = {}
    dirs = {}
    for li in sweep_layers:
        d, q, mstats = extract_action_dir_from_popqa(popqa_by_layer[li])
        if d is None:
            print(f"  L{li}: degenerate (norm=0); skipped")
            continue
        dirs[li] = d
        per_layer[str(li)] = {
            "action_dir_quality": float(q),
            "margin_mean": mstats["mean"],
            "margin_std":  mstats["std"],
            "spearman_r":  mstats["spearman_r"],
            "n_low":       mstats["n_low"],
            "n_high":      mstats["n_high"],
        }
        print(f"  L{li}: quality={q:.4f}  margin_range=[{mstats['min']:.2f},{mstats['max']:.2f}]")

    peak_layer = int(max(per_layer, key=lambda k: per_layer[k]["action_dir_quality"]))
    print(f"\n[auto-peak] L{peak_layer}  quality={per_layer[str(peak_layer)]['action_dir_quality']:.4f}")
    save_layer = peak_layer if forced_anchor is None else forced_anchor
    print(f"[save-at]   L{save_layer}  quality={per_layer[str(save_layer)]['action_dir_quality']:.4f}")

    # Save sweep summary
    (out / "sweep_quality.json").write_text(json.dumps({
        "model": args.model,
        "n_layers": n_layers,
        "hidden_size": D,
        "n_popqa": args.n_popqa,
        "sweep_layers": sweep_layers,
        "auto_peak_layer": peak_layer,
        "forced_target_layer": (None if forced_anchor is None else int(forced_anchor)),
        "saved_layer": int(save_layer),
        "per_layer": per_layer,
    }, indent=2))
    print(f"[save] {out/'sweep_quality.json'}")

    # Save action_dir + a placeholder evidence_dir (zeros) so the existing
    # c2_step1_action_margins_qwen3_32b.py loader works unchanged. The audit
    # only consumes the action-projection fields downstream.
    action_dir_L = dirs[save_layer]
    assert abs(np.linalg.norm(action_dir_L.astype(np.float64)) - 1.0) < 1e-5, "non-unit"
    evidence_zero = np.zeros(D, dtype=np.float32)
    npz_path = out / f"directions_L{save_layer}.npz"
    np.savez(npz_path,
             action_dir=action_dir_L,
             evidence_dir=evidence_zero,
             L_act=int(save_layer),
             L_evi=int(save_layer),
             cos_action_evidence=0.0,
             evidence_auroc=float("nan"),
             action_quality=per_layer[str(save_layer)]["action_dir_quality"])
    print(f"[save] {npz_path}  (action_dir L2={np.linalg.norm(action_dir_L):.6f})")


if __name__ == "__main__":
    main()
