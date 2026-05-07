#!/usr/bin/env python3
"""
Analyze results from sf_counterfactual.py
==========================================
Three layers:
  1. Population-level: 2ndSR, stop rate, EM, margin (paired McNemar / Wilcoxon)
  2. Conditional: split by 1SF-EM (first-hop sufficient vs insufficient)
  3. Margin: paired delta distribution

Usage:
  cd tmc/scripts/e2e_agent
  python scripts/analyze_sf_counterfactual.py --model qwen
  python scripts/analyze_sf_counterfactual.py --model qwen --outdir results/sf_counterfactual
"""

import json, argparse
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon, mannwhitneyu
from statsmodels.stats.contingency_tables import mcnemar


SEP = "=" * 60


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def mcnemar_test(a_search, b_search):
    """McNemar test for paired binary outcomes. a_search, b_search: bool arrays."""
    n01 = int(np.sum(~a_search & b_search))   # 1sf=stop, 2sf=search
    n10 = int(np.sum(a_search & ~b_search))   # 1sf=search, 2sf=stop
    table = [[int(np.sum(a_search & b_search)), n10],
             [n01, int(np.sum(~a_search & ~b_search))]]
    if (n01 + n10) == 0:
        return 1.0, n01, n10
    result = mcnemar([[table[0][0], table[0][1]], [table[1][0], table[1][1]]], exact=False, correction=True)
    return result.pvalue, n01, n10


def analyze(r1, r2, label="ALL", margin_key="margin"):
    """Analyze a paired list of 1sf and 2sf records."""
    # Align by sample_id
    id2_1sf = {r["sample_id"]: r for r in r1}
    id2_2sf = {r["sample_id"]: r for r in r2}
    common = sorted(set(id2_1sf) & set(id2_2sf))

    r1a = [id2_1sf[sid] for sid in common]
    r2a = [id2_2sf[sid] for sid in common]
    n = len(common)

    # "search" = genuine Action: search issued.
    # "hallucinated_obs" (Mistral '[N] Title: ...' continuations) = implicit stop, NOT search.
    # "search_continuation" is a retired label; kept here for backward compat with old files.
    search_1 = np.array([r["action_type"] == "search" for r in r1a])
    search_2 = np.array([r["action_type"] == "search" for r in r2a])
    # parse_failure: action_type is None (unknown/budget-out) AND not a hallucinated_obs
    pf_1 = np.array([r.get("action_type") is None for r in r1a])
    pf_2 = np.array([r.get("action_type") is None for r in r2a])
    # Use specified margin key; skip samples where it's None
    def get_margin(r):
        v = r.get(margin_key)
        return v if v is not None else np.nan
    margin_1 = np.array([get_margin(r) for r in r1a])
    margin_2 = np.array([get_margin(r) for r in r2a])
    em_1 = np.array([r["em"] if r["em"] is not None else np.nan for r in r1a])
    em_2 = np.array([r["em"] if r["em"] is not None else np.nan for r in r2a])

    # McNemar on 2ndSR (exclude budget-out / None action_type for behavioral analysis)
    pval_mcnemar, n01, n10 = mcnemar_test(search_1, search_2)

    # Wilcoxon on paired margin delta (skip NaN pairs)
    valid_both = ~np.isnan(margin_1) & ~np.isnan(margin_2)
    delta_margin = (margin_2 - margin_1)[valid_both]
    try:
        _, pval_wilcoxon = wilcoxon(delta_margin) if len(delta_margin) >= 10 else (None, np.nan)
    except Exception:
        pval_wilcoxon = np.nan

    print(f"\n{SEP}")
    print(f"SUBSET: {label}  (N={n}, margin_key={margin_key})")
    print(SEP)
    print(f"  Parse failure:  1SF={pf_1.mean()*100:.1f}%  2SF={pf_2.mean()*100:.1f}%")
    print(f"  2ndSR:          1SF={search_1.mean()*100:.1f}%  2SF={search_2.mean()*100:.1f}%"
          f"  Δ={search_2.mean()*100 - search_1.mean()*100:+.1f}%")
    print(f"  McNemar p:      {pval_mcnemar:.4f}  (n01={n01}, n10={n10})")
    m1v = margin_1[valid_both]; m2v = margin_2[valid_both]
    print(f"  Mean margin ({margin_key}): 1SF={np.nanmean(margin_1):.3f}  2SF={np.nanmean(margin_2):.3f}"
          f"  Δ={delta_margin.mean():+.3f}  (N_valid={valid_both.sum()})")
    print(f"  Wilcoxon p (margin Δ): {pval_wilcoxon:.4f}")

    # EM
    valid_em1 = ~np.isnan(em_1)
    valid_em2 = ~np.isnan(em_2)
    if valid_em1.any():
        print(f"  EM (if stop):   1SF={np.nanmean(em_1)*100:.1f}% (N={valid_em1.sum()})"
              f"  2SF={np.nanmean(em_2)*100:.1f}% (N={valid_em2.sum()})")

    def _f(v):
        """Convert to float, return None for NaN (safe for JSON serialization)."""
        fv = float(v)
        return None if (fv != fv) else fv  # NaN != NaN is True

    return {
        "label": label, "n": n,
        "sr_1sf": float(search_1.mean()), "sr_2sf": float(search_2.mean()),
        "pf_1sf": float(pf_1.mean()), "pf_2sf": float(pf_2.mean()),
        "mcnemar_p": float(pval_mcnemar), "n01": n01, "n10": n10,
        "n_valid_margin": int(valid_both.sum()),
        # Use nanmean to skip NaN samples (e.g. R1 budget-out with no margin_post)
        "mean_margin_1sf": _f(np.nanmean(margin_1)), "mean_margin_2sf": _f(np.nanmean(margin_2)),
        "mean_delta_margin": _f(delta_margin.mean()) if len(delta_margin) else None,
        "std_delta_margin":  _f(delta_margin.std())  if len(delta_margin) else None,
        "wilcoxon_p": _f(pval_wilcoxon),
        "em_1sf": _f(np.nanmean(em_1)) if valid_em1.any() else None,
        "em_2sf": _f(np.nanmean(em_2)) if valid_em2.any() else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen",
                        help="Model tag: qwen, mistral, r1")
    parser.add_argument("--outdir", default="results/sf_counterfactual")
    parser.add_argument("--margin-key", default=None,
                        help="Which margin to use: 'margin', 'margin_pre', 'margin_post'. "
                             "Default: 'margin_post' for r1, 'margin' for others.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    f1 = outdir / f"{args.model}_1sf_trajectories.jsonl"
    f2 = outdir / f"{args.model}_2sf_trajectories.jsonl"

    # R1 uses a different directory and filename prefix
    is_r1 = (args.model == "r1")
    if is_r1:
        outdir = Path(args.outdir.replace("sf_counterfactual", "sf_counterfactual_r1")
                      if "sf_counterfactual_r1" not in args.outdir else args.outdir)
        f1 = outdir / "r1_1sf_trajectories.jsonl"
        f2 = outdir / "r1_2sf_trajectories.jsonl"
    else:
        f1 = outdir / f"{args.model}_1sf_trajectories.jsonl"
        f2 = outdir / f"{args.model}_2sf_trajectories.jsonl"

    # Determine margin key
    margin_key = args.margin_key
    if margin_key is None:
        margin_key = "margin_post" if is_r1 else "margin"

    if not f1.exists() or not f2.exists():
        print(f"ERROR: Missing trajectory files in {outdir}")
        print(f"  Expected: {f1.name}, {f2.name}")
        return

    r1 = load_jsonl(f1)
    r2 = load_jsonl(f2)
    print(f"Loaded: {len(r1)} 1SF records, {len(r2)} 2SF records")

    # ── Layer 1: Population-level ──────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"LAYER 1: POPULATION-LEVEL  (margin_key={margin_key})")
    results_all = analyze(r1, r2, label="ALL", margin_key=margin_key)

    # ── Layer 2: Conditional on 1SF-EM ────────────────────────────────────
    id2_1sf = {r["sample_id"]: r for r in r1}
    id2_2sf = {r["sample_id"]: r for r in r2}
    common = sorted(set(id2_1sf) & set(id2_2sf))

    sufficient_ids   = [sid for sid in common if id2_1sf[sid].get("em") == 1]
    insufficient_ids = [sid for sid in common if id2_1sf[sid].get("em") == 0]
    unknown_ids      = [sid for sid in common if id2_1sf[sid].get("em") is None]

    print(f"\n{SEP}")
    print("LAYER 2: CONDITIONAL (1SF-EM split)")
    print(f"  First-hop sufficient (1SF EM=1):   N={len(sufficient_ids)}")
    print(f"  First-hop insufficient (1SF EM=0): N={len(insufficient_ids)}")
    print(f"  Unknown (1SF no final answer):     N={len(unknown_ids)}")

    results_suf = results_insuf = None
    if len(sufficient_ids) >= 5:
        r1_suf = [id2_1sf[sid] for sid in sufficient_ids]
        r2_suf = [id2_2sf[sid] for sid in sufficient_ids]
        results_suf = analyze(r1_suf, r2_suf, "FIRST-HOP SUFFICIENT", margin_key=margin_key)
    if len(insufficient_ids) >= 5:
        r1_ins = [id2_1sf[sid] for sid in insufficient_ids]
        r2_ins = [id2_2sf[sid] for sid in insufficient_ids]
        results_insuf = analyze(r1_ins, r2_ins, "FIRST-HOP INSUFFICIENT", margin_key=margin_key)

    # ── Layer 3: Margin delta distribution ────────────────────────────────
    print(f"\n{SEP}")
    print(f"LAYER 3: MARGIN DELTA DISTRIBUTION ({margin_key})")
    r1a = [id2_1sf[sid] for sid in common]
    r2a = [id2_2sf[sid] for sid in common]

    def safe_get(r, key):
        v = r.get(key)
        return v if v is not None else np.nan

    m1 = np.array([safe_get(r, margin_key) for r in r1a])
    m2 = np.array([safe_get(r, margin_key) for r in r2a])
    valid = ~np.isnan(m1) & ~np.isnan(m2)
    delta = (m2 - m1)[valid]
    print(f"  N valid pairs: {valid.sum()}/{len(common)}")
    print(f"  Δmargin (2SF - 1SF): mean={delta.mean():+.3f}, median={np.median(delta):+.3f}")
    print(f"  Fraction 2SF < 1SF (more stop):  {(delta < 0).mean()*100:.1f}%")
    q25, q75 = np.percentile(delta, [25, 75])
    print(f"  IQR: [{q25:+.3f}, {q75:+.3f}]")

    # ── Save JSON ──────────────────────────────────────────────────────────
    output = {
        "model": args.model,
        "margin_key": margin_key,
        "population": results_all,
        "sufficient": results_suf,
        "insufficient": results_insuf,
        "margin_delta": {
            "n_valid": int(valid.sum()),
            "mean": float(delta.mean()), "median": float(np.median(delta)),
            "std": float(delta.std()), "q25": float(q25), "q75": float(q75),
            "frac_more_stop": float((delta < 0).mean()),
        }
    }
    out_path = outdir / f"analysis_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved analysis to {out_path}")


if __name__ == "__main__":
    main()
