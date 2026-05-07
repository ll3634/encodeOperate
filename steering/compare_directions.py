#!/usr/bin/env python3
"""Compare two steering direction NPZ files (cosine similarity, norms, RMS).

This is meant for reviewer-proof reporting of whether e.g. search-direction and
calculator-direction are effectively the same representation.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_vec(path: str, key: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {path}. Available: {list(data.keys())}")
    return data[key].astype(np.float32).reshape(-1)


def main():
    p = argparse.ArgumentParser(description="Compare two direction vectors")
    p.add_argument("--a", required=True, help="NPZ path A")
    p.add_argument("--b", required=True, help="NPZ path B")
    p.add_argument("--key", default="decision_direction", help="NPZ key (default: decision_direction)")
    p.add_argument("--out", default=None, help="Optional JSON output path")
    args = p.parse_args()

    va = load_vec(args.a, args.key)
    vb = load_vec(args.b, args.key)
    if va.shape != vb.shape:
        raise ValueError(f"Shape mismatch: {va.shape} vs {vb.shape}")

    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    ra = float(np.sqrt(np.mean(va ** 2)))
    rb = float(np.sqrt(np.mean(vb ** 2)))
    cos = float(np.dot(va, vb) / (na * nb + 1e-12))
    cos_abs = float(abs(cos))

    summary = {
        "a": str(args.a),
        "b": str(args.b),
        "key": args.key,
        "dim": int(va.shape[0]),
        "norm_a": na,
        "norm_b": nb,
        "rms_a": ra,
        "rms_b": rb,
        "cosine": cos,
        "abs_cosine": cos_abs,
        "note": "If abs_cosine ~ 1.0, the directions are essentially the same up to sign.",
    }

    print("=== Direction Comparison ===")
    print(f"A: {args.a}")
    print(f"B: {args.b}")
    print(f"dim={summary['dim']}  cosine={cos:+.6f}  abs_cosine={cos_abs:.6f}")
    print(f"norm_a={na:.4f} rms_a={ra:.6f}")
    print(f"norm_b={nb:.4f} rms_b={rb:.6f}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON: {out}")


if __name__ == "__main__":
    main()
