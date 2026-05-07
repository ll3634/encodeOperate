#!/usr/bin/env python3
"""Robustness for the S0-T0 contrastive direction:
  (A) Seed sweep: 5 random train/test splits per (model, dataset). Report
      mean ± std of Δ(S0-T0)_norm and paired_d, plus canonical-ordering rate.
  (B) Cross-dataset generalization: train direction on dataset X (all 50 sids),
      project dataset Y test cells (all 50 sids). Report cell ordering and Δ.
Reads raw_h_<ds>.npz produced by _extract_h_for_st_contrast.py.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np


def load_split(npz_path):
    d = np.load(npz_path)
    H = d["H"].astype(np.float64)
    sids = [str(s) for s in d["sids"]]
    conds = [str(c) for c in d["conds"]]
    L = int(d["L"])
    by = defaultdict(dict)  # by[cond][sid] = row idx
    for i, (s, c) in enumerate(zip(sids, conds)):
        by[c][s] = i
    return H, by, L


def fit_direction(H, by, train_sids):
    h_s0 = H[[by["S0"][s] for s in train_sids if s in by["S0"]]].mean(0)
    h_t0 = H[[by["T0"][s] for s in train_sids if s in by["T0"]]].mean(0)
    v = h_s0 - h_t0
    n = float(np.linalg.norm(v))
    return (v / n if n > 1e-12 else v), n


def project_cells(H, by, sids, direction):
    out = {}  # cond -> list of (sid, raw_proj, normed_proj)
    for c in ("N0", "T0", "S0"):
        out[c] = []
        for s in sids:
            if s not in by[c]:
                continue
            h = H[by[c][s]]
            p = float(h @ direction)
            hn = float(np.linalg.norm(h))
            out[c].append((s, p, p / hn if hn > 0 else 0.0))
    return out


def paired_delta_norm(per_cond, a, b):
    am = {sid: pn for sid, _, pn in per_cond[a]}
    bm = {sid: pn for sid, _, pn in per_cond[b]}
    ks = sorted(set(am) & set(bm))
    diffs = np.array([bm[k] - am[k] for k in ks])
    if diffs.size == 0:
        return None
    sd = float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0
    return {
        "n": int(diffs.size),
        "mean_delta": float(diffs.mean()),
        "paired_d": float(diffs.mean() / sd) if sd > 0 else float("nan"),
    }


def cell_means_raw(per_cond):
    return {c: float(np.mean([p for _, p, _ in per_cond[c]])) if per_cond[c] else None
            for c in ("N0", "T0", "S0")}


def is_canonical(means):
    return (means["S0"] is not None and means["T0"] is not None
            and means["N0"] is not None
            and means["S0"] > means["T0"] > means["N0"])


def seed_sweep(H, by, common_sids, n_seeds=5, base_seed=20260502, train_frac=0.5):
    out = []
    for k in range(n_seeds):
        seed = base_seed + k
        rng = np.random.default_rng(seed)
        order = list(common_sids); rng.shuffle(order)
        n_tr = int(round(len(order) * train_frac))
        tr = sorted(order[:n_tr]); te = sorted(order[n_tr:])
        v, vn = fit_direction(H, by, tr)
        per = project_cells(H, by, te, v)
        means = cell_means_raw(per)
        d_st = paired_delta_norm(per, "T0", "S0")
        d_tn = paired_delta_norm(per, "N0", "T0")
        out.append({
            "seed": seed, "n_train": len(tr), "n_test": len(te),
            "raw_dir_norm": vn,
            "test_means_raw": means,
            "canonical": is_canonical(means),
            "delta_S0_T0_norm": d_st,
            "delta_T0_N0_norm": d_tn,
        })
    return out


def cross_dataset(H_tr, by_tr, H_te, by_te):
    # Use ALL train sids common to S0+T0 for direction; project ALL test sids
    tr_sids = sorted(set(by_tr["S0"]) & set(by_tr["T0"]))
    te_sids = sorted(set(by_te["N0"]) & set(by_te["T0"]) & set(by_te["S0"]))
    v, vn = fit_direction(H_tr, by_tr, tr_sids)
    per = project_cells(H_te, by_te, te_sids, v)
    means = cell_means_raw(per)
    return {
        "n_train_sids": len(tr_sids), "n_test_sids": len(te_sids),
        "raw_dir_norm": vn, "test_means_raw": means,
        "canonical": is_canonical(means),
        "delta_S0_T0_norm": paired_delta_norm(per, "T0", "S0"),
        "delta_T0_N0_norm": paired_delta_norm(per, "N0", "T0"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--st-dir", required=True,
                    help="Dir with raw_h_<ds>.npz (the st_contrast/ folder).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="unknown")
    args = ap.parse_args()

    R = Path(args.st_dir)
    payload = {"model": args.model, "seed_sweep": {}, "cross_dataset": {}}

    loaded = {}
    for ds in ("hotpotqa", "musique"):
        p = R / f"raw_h_{ds}.npz"
        if not p.exists(): continue
        H, by, L = load_split(p)
        loaded[ds] = (H, by, L)
        common = sorted(set(by["N0"]) & set(by["T0"]) & set(by["S0"]))
        sweep = seed_sweep(H, by, common, n_seeds=5)
        deltas = [s["delta_S0_T0_norm"]["mean_delta"] for s in sweep]
        ds_pd = [s["delta_S0_T0_norm"]["paired_d"] for s in sweep]
        canon = [s["canonical"] for s in sweep]
        payload["seed_sweep"][ds] = {
            "L": L, "n_common_sids": len(common),
            "per_seed": sweep,
            "summary_S0_T0_norm": {
                "mean_delta_mean": float(np.mean(deltas)),
                "mean_delta_std":  float(np.std(deltas, ddof=1)),
                "paired_d_mean":   float(np.mean(ds_pd)),
                "paired_d_std":    float(np.std(ds_pd, ddof=1)),
                "canonical_rate":  float(np.mean(canon)),
            },
        }
        print(f"\n=== {args.model} {ds} (L={L}) seed sweep n=5 ===")
        s = payload["seed_sweep"][ds]["summary_S0_T0_norm"]
        print(f"  Δ(S0-T0)_norm  mean={s['mean_delta_mean']:+.4f}  std={s['mean_delta_std']:.4f}")
        print(f"  paired_d       mean={s['paired_d_mean']:+.3f}  std={s['paired_d_std']:.3f}")
        print(f"  canonical rate {s['canonical_rate']:.2f} ({sum(canon)}/{len(canon)})")

    if "hotpotqa" in loaded and "musique" in loaded:
        H_h, by_h, _ = loaded["hotpotqa"]; H_m, by_m, _ = loaded["musique"]
        payload["cross_dataset"]["hotpot_dir_on_musique_cells"] = cross_dataset(H_h, by_h, H_m, by_m)
        payload["cross_dataset"]["musique_dir_on_hotpot_cells"] = cross_dataset(H_m, by_m, H_h, by_h)
        print(f"\n=== {args.model} cross-dataset generalization ===")
        for k, v in payload["cross_dataset"].items():
            d = v["delta_S0_T0_norm"]
            print(f"  {k}: canonical={v['canonical']}  Δ(S0-T0)_norm={d['mean_delta']:+.4f}  d={d['paired_d']:+.3f}  means={v['test_means_raw']}")

    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
