#!/usr/bin/env python3
"""Aggregate Robustness A (prompt paraphrase) + B (obs style) results.

Each input is `path:tag` where `tag` encodes model_variant_style, e.g.
`qwen_v2_factcard`. Cells: per (model, prompt_variant, obs_style, condition).
Contrasts: T0 vs N0 within each (model, variant, style)."""
import argparse, json, math, statistics as st
from collections import defaultdict
from pathlib import Path

try:
    from scipy.stats import binomtest, wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    if HAVE_SCIPY:
        return binomtest(k, n, 0.5, alternative="two-sided").pvalue
    p = sum(math.comb(n, i) * 0.5 ** n for i in range(k + 1))
    return min(1.0, 2 * p)


def _fa(r):
    return r.get("first_action_token") or (
        "search" if r.get("action_type") == "search" else
        "stop"   if r.get("action_type") == "stop" else "parse_fail")


def cell(rows):
    if not rows:
        return {"n": 0}
    n = len(rows)
    return {
        "n": n,
        "first_search": sum(1 for r in rows if _fa(r) == "search") / n,
        "first_stop":   sum(1 for r in rows if _fa(r) == "stop")   / n,
        "first_parse_fail":  sum(1 for r in rows if _fa(r) == "parse_fail") / n,
        "commit_W":     sum(1 for r in rows if r.get("contains_W")) / n,
        "em":           sum(1 for r in rows if r.get("em") == 1) / n,
        "mean_ml":      st.fmean(r["margin_label"] for r in rows),
        "mean_margin_post": (st.fmean([r["margin_post"] for r in rows if r.get("margin_post") is not None])
                              if any(r.get("margin_post") is not None for r in rows) else None),
    }


def paired_T0_vs_N0(t0_rows, n0_rows):
    """Return dict with paired contrasts for first_search, first_stop,
    commit_W and margin_label."""
    by_t = {r["sample_id"]: r for r in t0_rows}
    by_n = {r["sample_id"]: r for r in n0_rows}
    ids = sorted(set(by_t) & set(by_n))
    out = {"n_pairs": len(ids)}
    for key, getter in (
        ("first_search", lambda r: int(_fa(r) == "search")),
        ("first_stop",   lambda r: int(_fa(r) == "stop")),
        ("commit_W",     lambda r: int(bool(r.get("contains_W")))),
    ):
        a = [getter(by_t[i]) for i in ids]
        b = [getter(by_n[i]) for i in ids]
        b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
        b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
        out[key] = {
            "T0_rate": sum(a) / max(1, len(a)),
            "N0_rate": sum(b) / max(1, len(b)),
            "delta_T0_minus_N0": (sum(a) - sum(b)) / max(1, len(a)),
            "mcnemar_b10": b10, "mcnemar_b01": b01,
            "mcnemar_p": mcnemar_exact(b10, b01),
        }
    a = [by_t[i]["margin_label"] for i in ids]
    b = [by_n[i]["margin_label"] for i in ids]
    delta_ml = (sum(a) - sum(b)) / max(1, len(a))
    try:
        p_w = wilcoxon(a, b).pvalue if HAVE_SCIPY else None
    except ValueError:
        p_w = 1.0
    out["margin_label"] = {
        "T0_mean": st.fmean(a), "N0_mean": st.fmean(b),
        "delta_T0_minus_N0": delta_ml, "wilcoxon_p": p_w,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", nargs="+", required=True,
                    help="path:model:variant:style tuples, "
                         "e.g. results/.../qwen_v2_factcard.jsonl:qwen2_5_7b:v2:factcard")
    ap.add_argument("--out", default="results/cross_model_extractability/robustness/summary.json")
    args = ap.parse_args()

    grouped = defaultdict(lambda: defaultdict(list))  # (model,var,style) -> cond -> rows
    for spec in args.eval:
        parts = spec.split(":")
        path, model_tag, variant, style = parts[0], parts[1], parts[2], parts[3]
        for line in open(path):
            r = json.loads(line)
            r.setdefault("prompt_variant", variant)
            r.setdefault("obs_style", style)
            grouped[(model_tag, variant, style)][r["condition"]].append(r)

    summary = {"scipy_available": HAVE_SCIPY, "configs": {}}
    rows_for_table = []
    for key, by_cond in sorted(grouped.items()):
        model_tag, variant, style = key
        cells = {c: cell(by_cond[c]) for c in by_cond}
        contrast = (paired_T0_vs_N0(by_cond.get("T0", []), by_cond.get("N0", []))
                    if "T0" in by_cond and "N0" in by_cond else None)
        cfg_key = f"{model_tag}|{variant}|{style}"
        summary["configs"][cfg_key] = {"cells": cells, "T0_vs_N0": contrast}
        if contrast:
            rows_for_table.append({
                "model": model_tag, "variant": variant, "style": style,
                "n_pairs": contrast["n_pairs"],
                "N0_first_search": contrast["first_search"]["N0_rate"],
                "T0_first_search": contrast["first_search"]["T0_rate"],
                "delta_first_search": contrast["first_search"]["delta_T0_minus_N0"],
                "p_search": contrast["first_search"]["mcnemar_p"],
                "N0_commit_W": contrast["commit_W"]["N0_rate"],
                "T0_commit_W": contrast["commit_W"]["T0_rate"],
                "delta_commit_W": contrast["commit_W"]["delta_T0_minus_N0"],
                "p_commitW": contrast["commit_W"]["mcnemar_p"],
                "delta_ml": contrast["margin_label"]["delta_T0_minus_N0"],
                "p_ml": contrast["margin_label"]["wilcoxon_p"],
            })
    summary["table"] = rows_for_table
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    # Pretty print
    hdr = ("model", "var", "style", "N0sr", "T0sr", "Δsr", "p", "N0cW", "T0cW", "ΔcW", "p", "Δml", "p_ml")
    print("  ".join(f"{h:>10}" for h in hdr))
    for row in rows_for_table:
        cells = (row["model"][:10], row["variant"], row["style"],
                 f"{row['N0_first_search']:.2f}", f"{row['T0_first_search']:.2f}",
                 f"{row['delta_first_search']:+.2f}", f"{row['p_search']:.1e}",
                 f"{row['N0_commit_W']:.2f}", f"{row['T0_commit_W']:.2f}",
                 f"{row['delta_commit_W']:+.2f}", f"{row['p_commitW']:.1e}",
                 f"{row['delta_ml']:+.2f}", f"{row['p_ml']:.1e}" if row['p_ml'] else "n/a")
        print("  ".join(f"{c:>10}" for c in cells))


if __name__ == "__main__":
    main()
