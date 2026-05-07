#!/usr/bin/env python3
"""Compute S0-vs-T0 contrastive direction from a train split of paired cells,
project the held-out test split's N0/T0/S0 cells onto it, report cell ordering
and Δ(T0-N0), Δ(S0-N0), Δ(S0-T0) (raw, normalized, paired-d).

Train/test split: by sample_id with a fixed seed (no leakage across
conditions for the same sid).
"""
import argparse, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np


def cell(vals):
    a = np.asarray(vals, dtype=np.float64)
    n = a.size
    if n == 0: return {"n": 0}
    return {"n": int(n), "mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if n > 1 else 0.0,
            "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max())}


def paired(a_pairs, b_pairs):
    am = dict(a_pairs); bm = dict(b_pairs)
    ks = sorted(set(am) & set(bm))
    diffs = np.array([bm[k] - am[k] for k in ks], dtype=np.float64)
    n = diffs.size
    if n == 0: return None
    mn = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if n > 1 else 0.0
    cd = mn / sd if sd > 0 else float("nan")
    return {"n_pairs": int(n), "mean_delta": mn, "paired_d": cd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True,
                    help="Dir containing raw_h_<ds>.npz files.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="unknown")
    ap.add_argument("--seed", type=int, default=20260502)
    ap.add_argument("--train-frac", type=float, default=0.5)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    results = {}
    for ds in ("hotpotqa", "musique"):
        npz_path = Path(args.raw_dir) / f"raw_h_{ds}.npz"
        if not npz_path.exists():
            print(f"[skip] {npz_path} missing"); continue
        d = np.load(npz_path)
        H = d["H"].astype(np.float64)
        sids = d["sids"]; conds = d["conds"]
        L = int(d["L"])

        # Index by (sid, cond)
        by_cond_sid = defaultdict(dict)  # by_cond_sid[cond][sid] = h_idx
        for i, (s, c) in enumerate(zip(sids, conds)):
            by_cond_sid[str(c)][str(s)] = i

        # All sids appearing in S0 AND T0 AND N0
        common_sids = sorted(
            set(by_cond_sid["S0"]) & set(by_cond_sid["T0"]) & set(by_cond_sid["N0"]))
        # Train/test split by sid
        order = list(common_sids); rng.shuffle(order)
        n_train = int(round(len(order) * args.train_frac))
        train_sids = sorted(order[:n_train])
        test_sids  = sorted(order[n_train:])

        # Direction = mean(h_S0_train) - mean(h_T0_train); unit-norm
        H_S0_tr = H[[by_cond_sid["S0"][s] for s in train_sids]].mean(0)
        H_T0_tr = H[[by_cond_sid["T0"][s] for s in train_sids]].mean(0)
        direction = H_S0_tr - H_T0_tr
        nrm = float(np.linalg.norm(direction))
        if nrm < 1e-12:
            results[ds] = {"error": "degenerate direction"}; continue
        direction /= nrm

        # Project test cells (raw and ||h||-normalized)
        per_cond_raw = defaultdict(list)        # cond -> list of (sid, raw_proj)
        per_cond_norm = defaultdict(list)       # cond -> list of (sid, raw_proj/||h||)
        for c in ("N0", "T0", "S0"):
            for s in test_sids:
                if s not in by_cond_sid[c]: continue
                h = H[by_cond_sid[c][s]]
                p = float(h @ direction)
                hn = float(np.linalg.norm(h))
                per_cond_raw[c].append((s, p))
                per_cond_norm[c].append((s, p / hn if hn > 0 else 0.0))

        cells_raw  = {c: cell([v for _, v in per_cond_raw[c]])  for c in ("N0","T0","S0")}
        cells_norm = {c: cell([v for _, v in per_cond_norm[c]]) for c in ("N0","T0","S0")}
        deltas_raw = {
            "T0_minus_N0": paired(per_cond_raw["N0"], per_cond_raw["T0"]),
            "S0_minus_N0": paired(per_cond_raw["N0"], per_cond_raw["S0"]),
            "S0_minus_T0": paired(per_cond_raw["T0"], per_cond_raw["S0"]),
        }
        deltas_norm = {
            "T0_minus_N0": paired(per_cond_norm["N0"], per_cond_norm["T0"]),
            "S0_minus_N0": paired(per_cond_norm["N0"], per_cond_norm["S0"]),
            "S0_minus_T0": paired(per_cond_norm["T0"], per_cond_norm["S0"]),
        }
        order_raw = sorted([(c, cells_raw[c]["mean"]) for c in ("N0","T0","S0")],
                           key=lambda x: x[1], reverse=True)
        order_str = " > ".join(f"{c}({v:+.4f})" for c, v in order_raw)
        canonical = (order_raw[0][0] == "S0" and order_raw[1][0] == "T0"
                     and order_raw[2][0] == "N0")

        print(f"\n=== {ds} (L={L}) ===")
        print(f"  train sids: {len(train_sids)}, test sids: {len(test_sids)}, ||dir||_raw={nrm:.3f}")
        print(f"  test cell ordering (raw): {order_str}")
        print(f"  CANONICAL (S0>T0>N0): {canonical}")
        for k, v in deltas_norm.items():
            print(f"  {k}_norm: Δ={v['mean_delta']:+.5f}  paired_d={v['paired_d']:+.3f}  n={v['n_pairs']}")

        results[ds] = {
            "L": L, "n_train_sids": len(train_sids), "n_test_sids": len(test_sids),
            "direction_raw_norm": nrm,
            "test_cells_raw": cells_raw, "test_cells_normalized": cells_norm,
            "test_deltas_raw": deltas_raw, "test_deltas_normalized": deltas_norm,
            "test_cell_ordering_raw_desc": order_str,
            "canonical_S0_T0_N0": canonical,
        }

    with open(args.out, "w") as f:
        json.dump({"model": args.model, "datasets": results}, f, indent=2)
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
