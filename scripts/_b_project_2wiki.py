#!/usr/bin/env python3
"""B: project hotpotqa-trained S0-vs-T0 direction onto 2WikiMultiHop cells.

For each model: fit direction from existing n200/L<peak>/raw_h_hotpotqa.npz,
project onto st_contrast_2wiki/L<peak>/raw_h_2wiki.npz, report cell means,
paired_d, canonical ordering. Also fit a 2wiki-own direction (all 50 sids)
as a within-task control.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from _robustness_st_contrast import (  # noqa: E402
    load_split, fit_direction, project_cells,
    paired_delta_norm, cell_means_raw, is_canonical,
)

MODELS = [
    ("qwen2_5_7b",   20),
    ("qwen2_5_14b",  46),
    ("qwen2_5_32b",  50),
    ("qwen3_32b",    52),
]
RESULTS_ROOT = _HERE.parent / "results" / "scaling_difficulty_audit"


def cell_summary(per):
    means_raw = cell_means_raw(per)
    means_norm = {c: float(np.mean([pn for _, _, pn in per[c]])) if per[c] else None
                  for c in ("N0", "T0", "S0")}
    return {
        "means_raw":  means_raw,
        "means_norm": means_norm,
        "canonical":  is_canonical(means_raw),
        "delta_S0_T0_norm": paired_delta_norm(per, "T0", "S0"),
        "delta_T0_N0_norm": paired_delta_norm(per, "N0", "T0"),
    }


def project_one(model_dir: str, peak: int):
    out = {"model": model_dir, "peak_layer": peak}
    train_path = (RESULTS_ROOT / model_dir / "st_contrast_n200"
                  / f"L{peak}" / "raw_h_hotpotqa.npz")
    test_path  = (RESULTS_ROOT / model_dir / "st_contrast_2wiki"
                  / f"L{peak}" / "raw_h_2wiki.npz")
    if not train_path.exists():
        out["error"] = f"missing train: {train_path}"; return out
    if not test_path.exists():
        out["error"] = f"missing test: {test_path}"; return out
    H_tr, by_tr, _ = load_split(str(train_path))
    H_te, by_te, _ = load_split(str(test_path))

    tr_sids = sorted(set(by_tr["S0"]) & set(by_tr["T0"]))
    te_sids = sorted(set(by_te["N0"]) & set(by_te["T0"]) & set(by_te["S0"]))
    out["n_train_sids"] = len(tr_sids)
    out["n_test_sids"]  = len(te_sids)

    # (1) Cross-task: hotpot-trained direction on 2wiki cells
    direction, raw_n = fit_direction(H_tr, by_tr, tr_sids)
    per_cross = project_cells(H_te, by_te, te_sids, direction)
    out["cross_task_hotpot_dir_on_2wiki"] = {
        "raw_dir_norm": float(raw_n), **cell_summary(per_cross),
    }

    # (2) Within-task control: 2wiki-trained direction on 2wiki cells (in-fit)
    direction_w, raw_n_w = fit_direction(H_te, by_te, te_sids)
    per_within = project_cells(H_te, by_te, te_sids, direction_w)
    out["within_task_2wiki_dir_in_fit"] = {
        "raw_dir_norm": float(raw_n_w), **cell_summary(per_within),
    }

    # (3) Direction-similarity sanity: cosine(hotpot_dir, 2wiki_dir)
    out["cos_hotpot_2wiki_dir"] = float(direction @ direction_w)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS_ROOT / "b_2wiki_projection.json"))
    ap.add_argument("--models", nargs="+", default=None,
                    help="Subset of models to process (e.g. qwen2_5_7b qwen3_32b)")
    args = ap.parse_args()

    payload = {"per_model": {}}
    pending = []
    for m, L in MODELS:
        if args.models and m not in args.models:
            continue
        r = project_one(m, L)
        payload["per_model"][m] = r
        if "error" in r:
            print(f"[{m}] SKIP — {r['error']}", flush=True)
            pending.append(m)
            continue
        cx = r["cross_task_hotpot_dir_on_2wiki"]
        wi = r["within_task_2wiki_dir_in_fit"]
        d_st = cx["delta_S0_T0_norm"]; d_st_w = wi["delta_S0_T0_norm"]
        print(f"\n=== {m} (L{L}) ===", flush=True)
        print(f"  n_train_sids={r['n_train_sids']}  n_test_sids={r['n_test_sids']}", flush=True)
        print("  cross-task (hotpot dir → 2wiki cells):", flush=True)
        print(f"    means_raw  = {cx['means_raw']}", flush=True)
        print(f"    canonical  = {cx['canonical']}  Δ(S0-T0)_norm={d_st['mean_delta']:+.4f}  d={d_st['paired_d']:+.3f}", flush=True)
        print("  within-task (2wiki dir → 2wiki cells, in-fit):", flush=True)
        print(f"    means_raw  = {wi['means_raw']}", flush=True)
        print(f"    canonical  = {wi['canonical']}  Δ(S0-T0)_norm={d_st_w['mean_delta']:+.4f}  d={d_st_w['paired_d']:+.3f}", flush=True)
        print(f"  cos(hotpot_dir, 2wiki_dir) = {r['cos_hotpot_2wiki_dir']:+.3f}", flush=True)

    # Compact summary table
    print("\n=== summary table ===")
    print(f"{'model':<14}{'L':>4}  {'cross Δ_norm':>13} {'cross d':>9} {'cross can.':>11}  "
          f"{'within Δ_norm':>14} {'within d':>10} {'cos':>7}")
    for m, L in MODELS:
        r = payload["per_model"].get(m)
        if not r or "error" in r:
            print(f"{m:<14}{L:>4}  {'-- pending --':>13}")
            continue
        cx = r["cross_task_hotpot_dir_on_2wiki"]; wi = r["within_task_2wiki_dir_in_fit"]
        print(f"{m:<14}{L:>4}  "
              f"{cx['delta_S0_T0_norm']['mean_delta']:>+13.4f} "
              f"{cx['delta_S0_T0_norm']['paired_d']:>+9.3f} "
              f"{str(cx['canonical']):>11}  "
              f"{wi['delta_S0_T0_norm']['mean_delta']:>+14.4f} "
              f"{wi['delta_S0_T0_norm']['paired_d']:>+10.3f} "
              f"{r['cos_hotpot_2wiki_dir']:>+7.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[save] {args.out}")
    if pending:
        print(f"[note] pending models: {pending}")


if __name__ == "__main__":
    main()
