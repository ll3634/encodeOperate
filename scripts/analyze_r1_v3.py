#!/usr/bin/env python3
"""
Analyze R1 v3 results and compare to v2.

Produces:
  - Corrected 2ndSR/Stop rates (first-token classification vs v2 parse)
  - Confusion matrix: sign(margin_B) vs first_action_token
  - Margin distributions (A vs B) for 1SF/2SF
  - Samples where v2 and v3-first disagree
  - 3-model spectrum table (Qwen / Mistral / R1 corrected)
"""
import json, sys
import numpy as np
from pathlib import Path
from collections import Counter


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def fmt_pct(n, d):
    return f"{n}/{d}={n/d*100:.1f}%" if d else "N/A"


def summarize(name, rows, v2_rows=None):
    n = len(rows)
    valid = [r for r in rows if r.get("margin_B_true") is not None]
    print(f"\n{'='*72}\n  {name}  (N={n}, valid_margin_B={len(valid)})\n{'='*72}")

    # First-token-based classification (true decision)
    ft = Counter(r["action_type_first"] for r in rows)
    print(f"  First-action-token classification (TRUE decision at pos B):")
    print(f"    search: {fmt_pct(ft.get('search',0), n)}")
    print(f"    stop:   {fmt_pct(ft.get('stop',0), n)}")
    print(f"    other:  {fmt_pct(ft.get('other',0), n)}")
    print(f"    None (no </think>): {fmt_pct(ft.get(None,0), n)}")

    # v2-style (full-gen parse)
    v2p = Counter(r["action_type_v2"] for r in rows)
    print(f"  v2-style parse (Final-priority, BUG):")
    print(f"    search: {fmt_pct(v2p.get('search',0), n)}")
    print(f"    stop:   {fmt_pct(v2p.get('stop',0), n)}")
    print(f"    None:   {fmt_pct(v2p.get(None,0), n)}")

    # Hidden search count
    hidden = sum(1 for r in rows if r["action_type_first"] == "search"
                 and r["action_type_v2"] == "stop")
    print(f"  Hidden search (first=search but v2=stop, hallucinated obs+Final): {hidden}/{n}")

    # Margin stats
    mA = np.array([r["margin_A_v2eq"] for r in valid])
    mB = np.array([r["margin_B_true"] for r in valid])
    print(f"  Margin A (v2-equivalent, BUGGY): mean={mA.mean():+.3f} ± {mA.std():.3f}  median={np.median(mA):+.3f}")
    print(f"  Margin B (TRUE decision pos):    mean={mB.mean():+.3f} ± {mB.std():.3f}  median={np.median(mB):+.3f}")

    # Confusion matrix: sign(margin_B) vs first_action
    srch = [r for r in valid if r["action_type_first"] == "search"]
    stp  = [r for r in valid if r["action_type_first"] == "stop"]
    tp = sum(1 for r in srch if r["margin_B_true"] > 0)
    fn = sum(1 for r in srch if r["margin_B_true"] <= 0)
    fp = sum(1 for r in stp  if r["margin_B_true"] > 0)
    tn = sum(1 for r in stp  if r["margin_B_true"] <= 0)
    print(f"  Confusion sign(margin_B) x first_action:")
    print(f"                    first=search  first=stop")
    print(f"    margin_B > 0     TP={tp:<6d}   FP={fp}")
    print(f"    margin_B ≤ 0     FN={fn:<6d}   TN={tn}")
    if srch: print(f"    Mean margin_B | first=search: {np.mean([r['margin_B_true'] for r in srch]):+.3f}")
    if stp:  print(f"    Mean margin_B | first=stop:   {np.mean([r['margin_B_true'] for r in stp ]):+.3f}")

    # v2 comparison
    if v2_rows is not None:
        print(f"\n  --- v2 vs v3 comparison ---")
        v2_n = len(v2_rows)
        v2_search = sum(1 for r in v2_rows if r.get("action_type") == "search")
        v2_stop   = sum(1 for r in v2_rows if r.get("action_type") == "stop")
        v2_mpost  = np.array([r["margin_post"] for r in v2_rows if r.get("margin_post") is not None])
        print(f"    v2 2ndSR: {fmt_pct(v2_search, v2_n)}  v2 stop: {fmt_pct(v2_stop, v2_n)}")
        print(f"    v2 margin_post: mean={v2_mpost.mean():+.3f}  (should match v3 margin_A)")

    return {
        "n": n, "first_search": ft.get("search", 0), "first_stop": ft.get("stop", 0),
        "v2_search": v2p.get("search", 0), "v2_stop": v2p.get("stop", 0),
        "hidden_search": hidden,
        "margin_A_mean": float(mA.mean()) if len(mA) else None,
        "margin_B_mean": float(mB.mean()) if len(mB) else None,
        "margin_B_search_mean": float(np.mean([r["margin_B_true"] for r in srch])) if srch else None,
        "margin_B_stop_mean":   float(np.mean([r["margin_B_true"] for r in stp ])) if stp  else None,
        "confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3-dir", default="results/sf_counterfactual_r1_v3")
    ap.add_argument("--v2-dir", default="results/sf_counterfactual_r1_v2")
    args = ap.parse_args()
    v3_dir = Path(args.v3_dir)
    v2_dir = Path(args.v2_dir)

    summaries = {}
    for cond in ("1sf", "2sf"):
        v3 = load_jsonl(v3_dir / f"r1_{cond}_trajectories_v3.jsonl")
        v2 = load_jsonl(v2_dir / f"r1_{cond}_trajectories.jsonl") if (v2_dir / f"r1_{cond}_trajectories.jsonl").exists() else None
        summaries[cond] = summarize(f"1SF condition (evidence-insufficient)" if cond == "1sf"
                                    else "2SF condition (evidence-sufficient)", v3, v2)

    # Cross-condition contrast
    s1, s2 = summaries["1sf"], summaries["2sf"]
    print(f"\n{'='*72}\n  EVIDENCE-SENSITIVITY (1SF - 2SF)\n{'='*72}")
    d_ft = s1["first_search"]/s1["n"] - s2["first_search"]/s2["n"]
    d_v2 = (s1["v2_search"]/s1["n"]) - (s2["v2_search"]/s2["n"])
    d_mB = s1["margin_B_mean"] - s2["margin_B_mean"]
    d_mA = s1["margin_A_mean"] - s2["margin_A_mean"]
    print(f"  Δ 2ndSR (first-token, TRUE):  {d_ft*100:+.1f}pp")
    print(f"  Δ 2ndSR (v2 parse, BUGGY):    {d_v2*100:+.1f}pp")
    print(f"  Δ Margin B (TRUE):            {d_mB:+.3f}")
    print(f"  Δ Margin A (v2-equivalent):   {d_mA:+.3f}")

    # 3-model spectrum
    print(f"\n{'='*72}\n  3-MODEL SPECTRUM (for paper table)\n{'='*72}")
    print(f"  Model           1SF_2ndSR    2SF_2ndSR    Δ(pp)    Note")
    print(f"  {'-'*72}")
    # Qwen / Mistral pulled from existing analysis if available
    for mdl in ("qwen", "mistral"):
        p = Path(f"results/sf_counterfactual/analysis_{mdl}.json")
        if p.exists():
            a = json.loads(p.read_text())
            r1 = a.get("1sf", {}).get("search_rate")
            r2 = a.get("2sf", {}).get("search_rate")
            if r1 is not None and r2 is not None:
                print(f"  {mdl:<15s} {r1*100:>6.1f}%      {r2*100:>6.1f}%      {(r1-r2)*100:+.1f}")
    r1_1 = s1["first_search"]/s1["n"]
    r1_2 = s2["first_search"]/s2["n"]
    print(f"  {'R1 (v3 fix)':<15s} {r1_1*100:>6.1f}%      {r1_2*100:>6.1f}%      {(r1_1-r1_2)*100:+.1f}  <- USE THIS")
    print(f"  {'R1 (v2 buggy)':<15s} {s1['v2_search']/s1['n']*100:>6.1f}%      {s2['v2_search']/s2['n']*100:>6.1f}%      {d_v2*100:+.1f}  <- DO NOT USE")

    (v3_dir / "analysis_r1_v3.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nSaved {v3_dir / 'analysis_r1_v3.json'}")


if __name__ == "__main__":
    main()
