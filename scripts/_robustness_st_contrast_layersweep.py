#!/usr/bin/env python3
"""Layer-sweep variant of _robustness_st_contrast.py.

Reads <st-root>/L<layer>/raw_h_<ds>.npz for each layer in --layers, runs the
same 5-seed train/test sweep on each, plus the cross-dataset transfer test
at the centre layer (peak). Aggregates everything into a single JSON.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

# Reuse the building blocks already validated in the n=50 robustness pass.
from _robustness_st_contrast import (  # noqa: E402
    load_split, fit_direction, project_cells,
    paired_delta_norm, cell_means_raw, is_canonical,
    seed_sweep, cross_dataset,
)


def summarize_sweep(sweep):
    deltas = [s["delta_S0_T0_norm"]["mean_delta"] for s in sweep]
    ds_pd = [s["delta_S0_T0_norm"]["paired_d"] for s in sweep]
    canon = [s["canonical"] for s in sweep]
    return {
        "n_seeds": len(sweep),
        "mean_delta_mean": float(np.mean(deltas)),
        "mean_delta_std":  float(np.std(deltas, ddof=1)),
        "paired_d_mean":   float(np.mean(ds_pd)),
        "paired_d_std":    float(np.std(ds_pd, ddof=1)),
        "canonical_rate":  float(np.mean(canon)),
        "canonical_count": int(sum(canon)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--st-root", required=True,
                    help="Dir containing L<layer>/raw_h_<ds>.npz subdirs.")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--peak-layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="unknown")
    args = ap.parse_args()

    R = Path(args.st_root)
    payload = {"model": args.model, "peak_layer": args.peak_layer,
               "layers": args.layers, "per_layer": {}}

    print(f"\n##### {args.model}  layers={args.layers}  peak={args.peak_layer} #####")
    for L in args.layers:
        layer_dir = R / f"L{L}"
        per_layer = {"layer": L, "datasets": {}}
        loaded = {}
        for ds in ("hotpotqa", "musique"):
            p = layer_dir / f"raw_h_{ds}.npz"
            if not p.exists():
                print(f"[skip] {p}"); continue
            H, by, layer_id = load_split(p)
            assert layer_id == L, f"layer mismatch {layer_id}!={L}"
            loaded[ds] = (H, by)
            common = sorted(set(by["N0"]) & set(by["T0"]) & set(by["S0"]))
            sweep = seed_sweep(H, by, common, n_seeds=5)
            summary = summarize_sweep(sweep)
            per_layer["datasets"][ds] = {
                "n_common_sids": len(common),
                "per_seed": sweep,
                "summary_S0_T0_norm": summary,
            }
            print(f"  L{L} {ds}: n={len(common)}  "
                  f"\u0394={summary['mean_delta_mean']:+.4f}\u00b1{summary['mean_delta_std']:.4f}  "
                  f"d={summary['paired_d_mean']:+.3f}\u00b1{summary['paired_d_std']:.3f}  "
                  f"canon={summary['canonical_count']}/{summary['n_seeds']}")

        if L == args.peak_layer and "hotpotqa" in loaded and "musique" in loaded:
            H_h, by_h = loaded["hotpotqa"]; H_m, by_m = loaded["musique"]
            per_layer["cross_dataset"] = {
                "hotpot_dir_on_musique_cells": cross_dataset(H_h, by_h, H_m, by_m),
                "musique_dir_on_hotpot_cells": cross_dataset(H_m, by_m, H_h, by_h),
            }
            print(f"\n  L{L} cross-dataset:")
            for k, v in per_layer["cross_dataset"].items():
                d = v["delta_S0_T0_norm"]
                print(f"    {k}: canonical={v['canonical']}  "
                      f"\u0394={d['mean_delta']:+.4f}  d={d['paired_d']:+.3f}")
        payload["per_layer"][f"L{L}"] = per_layer

    # Convenience aggregate: per-layer headline numbers per dataset.
    agg = {}
    for ds in ("hotpotqa", "musique"):
        rows = []
        for L in args.layers:
            blk = payload["per_layer"].get(f"L{L}", {}).get("datasets", {}).get(ds)
            if not blk: continue
            s = blk["summary_S0_T0_norm"]
            rows.append({
                "L": L,
                "delta_mean": s["mean_delta_mean"],
                "delta_std":  s["mean_delta_std"],
                "d_mean":     s["paired_d_mean"],
                "d_std":      s["paired_d_std"],
                "canonical_count": s["canonical_count"],
            })
        agg[ds] = rows
    payload["aggregate"] = agg

    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
