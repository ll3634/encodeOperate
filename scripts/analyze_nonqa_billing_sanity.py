#!/usr/bin/env python3
"""
Analysis + report generator for the calculator-assisted billing-verification
sanity check.

Inputs (from a single run dir):
  config.json
  parsed_outputs.jsonl

Outputs (written to the same run dir):
  summary.json   condition rates + paired contrasts (overwrites the inline summary)
  report.md      human-readable report

Pilot go criteria (per spec):
  S0 commit_W >= 0.75
  parse_failure_rate <= 0.10
  T0 - N0 commit_W delta >= +0.20
  T0 - IC commit_W delta >= +0.15
  N0 commit_W <= 0.20 preferred (flagged if higher)
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


CONDITIONS = ["N0", "T0", "IC", "S0"]


def load_parsed(path: Path) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def by_item_condition(rows: List[dict]) -> Dict[Tuple[str, str], dict]:
    out = {}
    for r in rows:
        out[(r["item_id"], r["condition"])] = r
    return out


# ---------------------------------------------------------------------------
# Condition-level rates
# ---------------------------------------------------------------------------

def condition_rates(rows: List[dict]) -> Dict[str, dict]:
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    out = {}
    for c in CONDITIONS:
        rs = by_cond.get(c, [])
        n = len(rs)
        out[c] = {
            "n": n,
            "commit_W_rate": sum(r["commit_W"] for r in rs) / n if n else 0.0,
            "first_is_search_rate": sum(r["first_is_search"] for r in rs) / n if n else 0.0,
            "first_is_final_rate": sum(r["first_is_final"] for r in rs) / n if n else 0.0,
            "parse_failure_rate": sum(r["parse_failure"] for r in rs) / n if n else 0.0,
        }
    overall_pf = sum(r["parse_failure"] for r in rows) / len(rows) if rows else 0.0
    out["_overall_parse_failure_rate"] = overall_pf
    return out


# ---------------------------------------------------------------------------
# Paired contrasts (McNemar + bootstrap)
# ---------------------------------------------------------------------------

def mcnemar_exact_binom_p(b: int, c: int) -> float:
    """Exact two-sided McNemar via scipy.stats.binomtest(min(b,c), n=b+c, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        # Fallback exact two-sided binomial computation
        k = min(b, c)
        # P(X <= k) under Bin(n, 0.5)
        p_one = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
        return min(1.0, 2.0 * p_one)


def paired_contrast(
    cond_a: str, cond_b: str,
    by_ic: Dict[Tuple[str, str], dict],
    item_ids: List[str],
    n_boot: int = 10000,
    seed: int = 12345,
) -> dict:
    """
    Paired contrast on commit_W between two conditions.

    delta = mean(commit_W[cond_a]) - mean(commit_W[cond_b])
    Returns delta, b, c, mcnemar_p, bootstrap CI.
    """
    pairs = []
    for iid in item_ids:
        ra = by_ic.get((iid, cond_a))
        rb = by_ic.get((iid, cond_b))
        if ra is None or rb is None:
            continue
        pairs.append((int(ra["commit_W"]), int(rb["commit_W"])))
    n = len(pairs)
    if n == 0:
        return {"n": 0}

    a_arr = np.array([p[0] for p in pairs])
    b_arr = np.array([p[1] for p in pairs])
    delta = float(a_arr.mean() - b_arr.mean())

    # McNemar discordances on commit_W
    b_disc = int(((a_arr == 1) & (b_arr == 0)).sum())  # a-yes, b-no
    c_disc = int(((a_arr == 0) & (b_arr == 1)).sum())  # a-no, b-yes
    p_mcn = mcnemar_exact_binom_p(b_disc, c_disc)

    # Paired bootstrap on item indices
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    idx_pool = np.arange(n)
    for i in range(n_boot):
        idx = rng.choice(idx_pool, size=n, replace=True)
        deltas[i] = a_arr[idx].mean() - b_arr[idx].mean()
    lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))

    return {
        "n": n,
        "rate_a": float(a_arr.mean()),
        "rate_b": float(b_arr.mean()),
        "delta": delta,
        "b_a_yes_b_no": b_disc,
        "c_a_no_b_yes": c_disc,
        "mcnemar_p": p_mcn,
        "bootstrap_ci_95": [lo, hi],
        "n_boot": n_boot,
    }


# ---------------------------------------------------------------------------
# Pilot gates
# ---------------------------------------------------------------------------

def evaluate_pilot_gates(rates: Dict[str, dict],
                         contrasts: Dict[str, dict]) -> dict:
    s0 = rates["S0"]["commit_W_rate"]
    n0 = rates["N0"]["commit_W_rate"]
    pf = rates["_overall_parse_failure_rate"]
    d_t0_n0 = contrasts["T0_vs_N0"]["delta"]
    d_t0_ic = contrasts["T0_vs_IC"]["delta"]

    gates = {
        "S0_commit_W_ge_0.75": (s0 >= 0.75, s0),
        "parse_failure_le_0.10": (pf <= 0.10, pf),
        "T0_minus_N0_ge_0.20": (d_t0_n0 >= 0.20, d_t0_n0),
        "T0_minus_IC_ge_0.15": (d_t0_ic >= 0.15, d_t0_ic),
        "N0_commit_W_le_0.20_preferred": (n0 <= 0.20, n0),
    }
    must_pass = [
        "S0_commit_W_ge_0.75",
        "parse_failure_le_0.10",
        "T0_minus_N0_ge_0.20",
        "T0_minus_IC_ge_0.15",
    ]
    pilot_pass = all(gates[g][0] for g in must_pass)
    return {
        "gates": {k: {"passed": v[0], "value": v[1]} for k, v in gates.items()},
        "must_pass_keys": must_pass,
        "pilot_pass": pilot_pass,
    }


# ---------------------------------------------------------------------------
# Examples (for the report)
# ---------------------------------------------------------------------------

def pick_examples(rows: List[dict], by_ic, item_ids, k: int = 3):
    s0_ok, t0_fail, n0_ok = [], [], []
    for iid in item_ids:
        s0 = by_ic.get((iid, "S0"))
        t0 = by_ic.get((iid, "T0"))
        n0 = by_ic.get((iid, "N0"))
        if s0 and s0["commit_W"] == 1 and len(s0_ok) < k:
            s0_ok.append(s0)
        if t0 and t0["commit_W"] == 1 and len(t0_fail) < k:
            t0_fail.append(t0)
        if n0 and n0["first_is_search"] == 1 and len(n0_ok) < k:
            n0_ok.append(n0)
    return s0_ok, t0_fail, n0_ok


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(out_md: Path, config: dict, rates: dict,
                 contrasts: dict, pilot_eval: dict,
                 examples: Tuple[list, list, list]):
    s0_ok, t0_fail, n0_ok = examples

    def fmt_pct(x): return f"{x*100:.1f}%"
    def fmt_p(p):
        if p < 1e-4:
            return f"{p:.2e}"
        return f"{p:.4f}"
    def fmt_ci(ci):
        return f"[{ci[0]*100:+.1f}%, {ci[1]*100:+.1f}%]"

    lines = []
    lines.append("# Calculator-assisted billing verification sanity check")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This is a behavior-only non-QA sanity check. It tests whether unsupported "
        "candidate amount availability increases premature finalization in a realistic "
        "billing-verification setting. It does not test the mechanism chain and does not "
        "claim universal agent generality."
    )
    lines.append("")
    lines.append("## Run config")
    lines.append("")
    lines.append(f"- model: `{config.get('model')}`")
    lines.append(f"- n_items: {config.get('n_items')}  (pilot={config.get('pilot')})")
    lines.append(f"- temperature: {config.get('temperature')}, "
                 f"top_p: {config.get('top_p')}, "
                 f"max_new_tokens: {config.get('max_new_tokens')}")
    lines.append(f"- seed: {config.get('seed')}")
    lines.append("")

    lines.append("## Condition rates")
    lines.append("")
    lines.append("| Condition | n | commit_W | first_is_search | first_is_final | parse_failure |")
    lines.append("|---|---|---|---|---|---|")
    for c in CONDITIONS:
        r = rates[c]
        lines.append(
            f"| {c} | {r['n']} | {fmt_pct(r['commit_W_rate'])} | "
            f"{fmt_pct(r['first_is_search_rate'])} | "
            f"{fmt_pct(r['first_is_final_rate'])} | "
            f"{fmt_pct(r['parse_failure_rate'])} |"
        )
    lines.append("")
    lines.append(f"Overall parse-failure rate: **{fmt_pct(rates['_overall_parse_failure_rate'])}**")
    lines.append("")

    lines.append("## Pilot go/no-go gates")
    lines.append("")
    lines.append("| Gate | Passed | Value |")
    lines.append("|---|---|---|")
    for k, v in pilot_eval["gates"].items():
        flag = "✅" if v["passed"] else "❌"
        val = v["value"]
        if "rate" in k or k.startswith("S0") or k.startswith("N0") or k.startswith("parse"):
            v_str = fmt_pct(val) if abs(val) <= 1.5 else f"{val:.4f}"
        else:
            v_str = fmt_pct(val)
        lines.append(f"| {k} | {flag} | {v_str} |")
    lines.append("")
    lines.append(f"**pilot_pass = {pilot_eval['pilot_pass']}** "
                 f"(must-pass = {', '.join(pilot_eval['must_pass_keys'])})")
    lines.append("")

    lines.append("## Paired contrasts on commit_W")
    lines.append("")
    lines.append("| Contrast | n | rate_a | rate_b | delta | b (a=1,b=0) | c (a=0,b=1) | McNemar p | 95% CI |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for label, key in [
        ("T0 vs N0", "T0_vs_N0"),
        ("T0 vs IC", "T0_vs_IC"),
        ("S0 vs T0", "S0_vs_T0"),
    ]:
        ct = contrasts.get(key, {})
        if ct.get("n", 0) == 0:
            lines.append(f"| {label} | 0 | – | – | – | – | – | – | – |")
            continue
        lines.append(
            f"| {label} | {ct['n']} | {fmt_pct(ct['rate_a'])} | "
            f"{fmt_pct(ct['rate_b'])} | {fmt_pct(ct['delta'])} | "
            f"{ct['b_a_yes_b_no']} | {ct['c_a_no_b_yes']} | "
            f"{fmt_p(ct['mcnemar_p'])} | {fmt_ci(ct['bootstrap_ci_95'])} |"
        )
    lines.append("")

    lines.append("## Parse-failure diagnostics")
    lines.append("")
    lines.append("| Condition | n | parse_failure | first_is_search | first_is_final |")
    lines.append("|---|---|---|---|---|")
    for c in CONDITIONS:
        r = rates[c]
        lines.append(
            f"| {c} | {r['n']} | {fmt_pct(r['parse_failure_rate'])} | "
            f"{fmt_pct(r['first_is_search_rate'])} | "
            f"{fmt_pct(r['first_is_final_rate'])} |"
        )
    lines.append("")

    def _ex(title, lst):
        lines.append(f"### {title}")
        lines.append("")
        if not lst:
            lines.append("(none)")
            lines.append("")
            return
        for r in lst:
            lines.append(f"- `{r['item_id']}` [{r['condition']}] W={r['W_str']}  "
                         f"→ raw: `{r['first_line']!r}`")
        lines.append("")

    lines.append("## Example raw outputs")
    lines.append("")
    _ex("3 successful S0 (commit_W = 1)", s0_ok)
    _ex("3 T0 failures (commit_W = 1; the targeted error)", t0_fail)
    _ex("3 N0 correct searches (first_is_search = 1)", n0_ok)

    lines.append("## Recommended paper sentence (only if successful)")
    lines.append("")
    lines.append(
        "> We also test a calculator-assisted billing-verification setting, where the "
        "model must either search for missing invoice fields or finalize a payable amount. "
        "Making an unsupported candidate amount available in the tool observation increases "
        "finalization to that amount relative to both a no-candidate condition and a matched "
        "irrelevant-note control. We use this as a non-QA surface-domain sanity check, not "
        "as evidence of universal agent generality."
    )
    lines.append("")
    lines.append("## Caveats / blacklist")
    lines.append("")
    lines.append(
        "Do not derive any of the following claims from this single sanity check:"
    )
    lines.append("")
    for bl in [
        "This proves the mechanism generalizes to all real-world tools.",
        "This is a calculator failure.",
        "The model cannot do arithmetic.",
        "The circuit generalizes to billing.",
        "This rules out all RAG artifacts.",
        "Evidence is ignored.",
        "A3 fixes billing answers.",
    ]:
        lines.append(f"- ❌ \"{bl}\"")
    lines.append("")

    out_md.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Run directory written by the runner.")
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    rows = load_parsed(run_dir / "parsed_outputs.jsonl")
    by_ic = by_item_condition(rows)
    item_ids = sorted({r["item_id"] for r in rows})

    rates = condition_rates(rows)

    contrasts = {
        "T0_vs_N0": paired_contrast("T0", "N0", by_ic, item_ids, n_boot=args.n_boot),
        "T0_vs_IC": paired_contrast("T0", "IC", by_ic, item_ids, n_boot=args.n_boot),
        "S0_vs_T0": paired_contrast("S0", "T0", by_ic, item_ids, n_boot=args.n_boot),
    }
    pilot_eval = evaluate_pilot_gates(rates, contrasts)
    examples = pick_examples(rows, by_ic, item_ids, k=3)

    summary = {
        "n_items": len(item_ids),
        "model": config.get("model"),
        "by_condition": {c: rates[c] for c in CONDITIONS},
        "overall_parse_failure_rate": rates["_overall_parse_failure_rate"],
        "contrasts": contrasts,
        "pilot_eval": pilot_eval,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(run_dir / "report.md", config, rates, contrasts, pilot_eval, examples)

    print(f"[analyze] wrote: {run_dir/'summary.json'}")
    print(f"[analyze] wrote: {run_dir/'report.md'}")
    print(f"[analyze] pilot_pass = {pilot_eval['pilot_pass']}")
    for c in CONDITIONS:
        r = rates[c]
        print(f"  {c}: commit_W={r['commit_W_rate']*100:.1f}% "
              f"search={r['first_is_search_rate']*100:.1f}% "
              f"final={r['first_is_final_rate']*100:.1f}% "
              f"pf={r['parse_failure_rate']*100:.1f}%")
    for k, ct in contrasts.items():
        if ct.get("n", 0) == 0:
            continue
        print(f"  {k}: delta={ct['delta']*100:+.1f}%, "
              f"McNemar p={ct['mcnemar_p']:.4f}, "
              f"95% CI=[{ct['bootstrap_ci_95'][0]*100:+.1f}%, "
              f"{ct['bootstrap_ci_95'][1]*100:+.1f}%]")


if __name__ == "__main__":
    main()

