#!/usr/bin/env python3
"""Audit 4 — Decompose an adapter-extracted action direction along an
adapter-extracted evidence direction, producing the (full / parallel / perp)
NPZs used by `ft_in_adapter_decomposition.py`.

This is the parameterized analogue of `_phaseD3_decompose_ft.py`.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_dir(path, key="decision_direction"):
    d = np.load(path)
    return d[key].astype(np.float32)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-dir", required=True)
    ap.add_argument("--action-dir", required=True)
    ap.add_argument("--out-full", required=True)
    ap.add_argument("--out-parallel", required=True)
    ap.add_argument("--out-perp", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--tag", default="ctrl_n0")
    ap.add_argument("--layer", type=int, default=20)
    args = ap.parse_args()

    ev = load_dir(args.evidence_dir)
    act = load_dir(args.action_dir)
    ev_u = unit(ev)

    p_coef = float(np.dot(act, ev_u))
    parallel = p_coef * ev_u
    perp = act - parallel

    p_norm = float(np.linalg.norm(parallel))
    pp_norm = float(np.linalg.norm(perp))
    full_norm = float(np.linalg.norm(act))

    Path(args.out_full).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_full,
             decision_direction=act.astype(np.float32),
             layer=args.layer,
             method=f"decomposition_full_{args.tag}",
             component="full",
             cos_with_probe=float(np.dot(unit(act), ev_u)))
    np.savez(args.out_parallel,
             decision_direction=parallel.astype(np.float32),
             layer=args.layer,
             method=f"decomposition_parallel_{args.tag}",
             component=f"parallel_to_probe_{args.tag}",
             cos_with_probe=float(np.dot(unit(parallel), ev_u))
                if p_norm > 1e-8 else 0.0,
             original_norm=full_norm,
             component_norm=p_norm)
    np.savez(args.out_perp,
             decision_direction=perp.astype(np.float32),
             layer=args.layer,
             method=f"decomposition_perpendicular_{args.tag}",
             component=f"perpendicular_to_probe_{args.tag}",
             cos_with_probe=float(np.dot(unit(perp), ev_u)),
             original_norm=full_norm,
             component_norm=pp_norm)

    summary = {
        "tag": args.tag,
        "layer": args.layer,
        "evidence_dir": args.evidence_dir,
        "action_dir": args.action_dir,
        "evidence_norm": float(np.linalg.norm(ev)),
        "action_norm": full_norm,
        "parallel_coef": p_coef,
        "parallel_norm": p_norm,
        "perp_norm": pp_norm,
        "parallel_share_norm_pct": (p_norm / full_norm) * 100,
        "var_parallel_fraction": (p_norm ** 2) / (full_norm ** 2),
        "cos_action_evidence": float(np.dot(unit(act), ev_u)),
        "files_written": [args.out_full, args.out_parallel, args.out_perp],
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2))

    print(f"=== Decomposition ({args.tag}, L{args.layer}) ===")
    print(f"  evidence_norm = {summary['evidence_norm']:.4f}")
    print(f"  action_norm   = {full_norm:.4f}")
    print(f"  parallel_norm = {p_norm:.4f}  ({summary['parallel_share_norm_pct']:.3f}%)")
    print(f"  perp_norm     = {pp_norm:.4f}")
    print(f"  cos(act, ev)  = {summary['cos_action_evidence']:+.5f}")
    print(f"[saved] {args.summary_json}")


if __name__ == "__main__":
    main()
