#!/usr/bin/env python3
"""
Analyzer + report generator for the multi-turn ReAct meeting-scheduling
non-QA sanity check.

Supports two inputs:
  --prefilled_dir   run dir for mode='prefilled' (multi-turn ReAct prefill)
  --single_shot_dir run dir for mode='single_shot' (collapsed user prompt)

Within each mode:
  Paired contrasts T0 vs {N0, IC, S0} on commit_W_anywhere,
  hallucinated_observation, final_present.

Across modes (structural ablation):
  Paired contrast prefilled-T0 vs single_shot-T0 on the same metrics.

Outputs (written to --out_dir, defaults to prefilled_dir):
  summary.json
  report.md
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


_CONDITION_ORDER = ["N0", "T0", "T_partial", "T_edge", "IC", "S0"]
METRICS = ["commit_W_anywhere", "hallucinated_observation", "final_present"]


def detect_conditions(rows: List[dict]) -> List[str]:
    found = {r["condition"] for r in rows}
    return [c for c in _CONDITION_ORDER if c in found]


def detect_contrasts(conditions: List[str]) -> List[Tuple[str, str, str]]:
    """Return list of (label, cond_a, cond_b) for within-mode paired contrasts."""
    cs = set(conditions)
    out: List[Tuple[str, str, str]] = []
    if "T0" in cs and "N0" in cs: out.append(("T0_vs_N0", "T0", "N0"))
    if "T0" in cs and "IC" in cs: out.append(("T0_vs_IC", "T0", "IC"))
    if "S0" in cs and "T0" in cs: out.append(("S0_vs_T0", "S0", "T0"))
    if "T_partial" in cs and "N0" in cs: out.append(("T_partial_vs_N0", "T_partial", "N0"))
    if "T_edge" in cs and "N0" in cs: out.append(("T_edge_vs_N0", "T_edge", "N0"))
    if "T0" in cs and "T_partial" in cs: out.append(("T0_vs_T_partial", "T0", "T_partial"))
    if "T_partial" in cs and "T_edge" in cs: out.append(("T_partial_vs_T_edge", "T_partial", "T_edge"))
    return out


def load_parsed(path: Path) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def by_item_condition(rows: List[dict]) -> Dict[Tuple[str, str], dict]:
    return {(r["item_id"], r["condition"]): r for r in rows}


def condition_rates(rows: List[dict], conditions: List[str]) -> Dict[str, dict]:
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    out = {}
    for c in conditions:
        rs = by_cond.get(c, [])
        n = len(rs)
        d = {"n": n}
        for m in METRICS + ["first_is_action", "first_is_final", "parse_failure", "commit_W"]:
            d[m + "_rate"] = sum(r[m] for r in rs) / n if n else 0.0
        out[c] = d
    out["_overall_parse_failure_rate"] = (
        sum(r["parse_failure"] for r in rows) / len(rows) if rows else 0.0
    )
    return out


def mcnemar_exact_binom_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        k = min(b, c)
        p_one = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
        return min(1.0, 2.0 * p_one)


def paired_contrast(
    a_rows: Dict[Tuple[str, str], dict],
    b_rows: Dict[Tuple[str, str], dict],
    a_key: Tuple[str, str],   # (label_a, condition_a)  — for documentation only
    b_key: Tuple[str, str],
    item_ids: List[str],
    metric: str,
    n_boot: int = 10000,
    seed: int = 12345,
) -> dict:
    """Generic paired contrast on a binary metric over item_ids."""
    a_cond = a_key[1]
    b_cond = b_key[1]
    pairs = []
    for iid in item_ids:
        ra = a_rows.get((iid, a_cond))
        rb = b_rows.get((iid, b_cond))
        if ra is None or rb is None:
            continue
        pairs.append((int(ra[metric]), int(rb[metric])))
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    a_arr = np.array([p[0] for p in pairs])
    b_arr = np.array([p[1] for p in pairs])
    delta = float(a_arr.mean() - b_arr.mean())
    b_disc = int(((a_arr == 1) & (b_arr == 0)).sum())
    c_disc = int(((a_arr == 0) & (b_arr == 1)).sum())
    p_mcn = mcnemar_exact_binom_p(b_disc, c_disc)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    idx_pool = np.arange(n)
    for i in range(n_boot):
        idx = rng.choice(idx_pool, size=n, replace=True)
        deltas[i] = a_arr[idx].mean() - b_arr[idx].mean()
    lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    return {
        "n": n,
        "metric": metric,
        "label_a": f"{a_key[0]}:{a_cond}",
        "label_b": f"{b_key[0]}:{b_cond}",
        "rate_a": float(a_arr.mean()),
        "rate_b": float(b_arr.mean()),
        "delta": delta,
        "b_a_yes_b_no": b_disc,
        "c_a_no_b_yes": c_disc,
        "mcnemar_p": p_mcn,
        "bootstrap_ci_95": [lo, hi],
        "n_boot": n_boot,
    }


def pick_examples(by_ic: Dict[Tuple[str, str], dict],
                  item_ids: List[str], k: int = 3):
    t0_trap, n0_search, s0_correct = [], [], []
    for iid in item_ids:
        t0 = by_ic.get((iid, "T0"))
        n0 = by_ic.get((iid, "N0"))
        s0 = by_ic.get((iid, "S0"))
        if t0 and t0["commit_W_anywhere"] == 1 and len(t0_trap) < k:
            t0_trap.append(t0)
        if n0 and n0["commit_W_anywhere"] == 0 and len(n0_search) < k:
            n0_search.append(n0)
        if s0 and s0["commit_W"] == 1 and len(s0_correct) < k:
            s0_correct.append(s0)
    return t0_trap, n0_search, s0_correct


def fmt_pct(x): return f"{x*100:.1f}%"
def fmt_p(p): return f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"
def fmt_ci(ci): return f"[{ci[0]*100:+.1f}%, {ci[1]*100:+.1f}%]"


def write_report(out_md: Path,
                 cfg_pre: dict, rates_pre: dict,
                 cfg_ss: dict | None, rates_ss: dict | None,
                 within_pre: dict, within_ss: dict | None,
                 cross: dict | None,
                 examples_pre, examples_ss,
                 conditions_pre: List[str],
                 conditions_ss: List[str] | None,
                 contrasts_pre: List[Tuple[str, str, str]],
                 contrasts_ss: List[Tuple[str, str, str]] | None):

    scenario = (cfg_pre.get("scenario") or "react_meeting")
    is_v2 = scenario == "react_meeting_v2"
    is_codesearch = scenario == "react_codesearch"
    has_gradient = "T_partial" in conditions_pre and "T_edge" in conditions_pre

    L: List[str] = []
    if is_codesearch:
        L.append("# Multi-turn ReAct code-search sanity check (premature-stop carving)")
        L.append("")
        L.append("Non-QA scenario: the agent is asked to find the canonical (production) "
                 "implementation of a function in a fictional repository, replying with "
                 "a `file:line`. A single `code_search` Action+Observation is prefilled. "
                 "In T0 the Observation lists three hits — a legacy/ implementation whose "
                 "name matches the user query exactly, the canonical implementation "
                 "(different name, production path), and a tests/ fixture; the trap is "
                 "the legacy path. In S0 the legacy hit is removed (canonical only). "
                 "In N0 no hit contains a matching definition. In IC the tool returns a "
                 "service-status message with no hits at all.")
    elif is_v2:
        L.append("# Multi-turn ReAct meeting-scheduling sanity check (v2: premature-stop carving)")
        L.append("")
        L.append("Non-QA scenario: the model receives a meeting-scheduling task whose "
                 "user prompt states an explicit negative constraint (e.g. "
                 "\"Bob has a 1:00pm-2:00pm standup; the meeting MUST NOT overlap this block\"). "
                 "A single joint-availability tool call is prefilled. In T0 the prefilled "
                 "Observation reports a joint slot that VIOLATES the constraint (the trap). "
                 "In S0 the slot RESPECTS the constraint. In N0 the tool reports no slot. "
                 "In IC the observation contains a tool-internal status line with no slot at all.")
        if has_gradient:
            L.append("")
            L.append("**Trap-difficulty gradient** (this run): T0 = full violation "
                     "(slot fully inside forbidden block, 30/30 min overlap); "
                     "T_partial = ~15/30 min overlap (straddles boundary); "
                     "T_edge = touches boundary, ZERO overlap (logically valid; control "
                     "for \"model rejects anything near the constraint\"); "
                     "S0 = well-separated valid slot.")
    else:
        L.append("# Multi-turn ReAct meeting-scheduling sanity check")
        L.append("")
        L.append("Non-QA scenario: the model receives a meeting-scheduling task with a "
                 "calendar tool. The first tool call (lookup of person A) is prefilled, "
                 "and the model is placed at the decision point for whether to issue a "
                 "second tool call or finalize. T0 makes a candidate (trap) slot "
                 "extractable from A's calendar; N0/IC do not; S0 already provides the "
                 "joint slot.")
    L.append("")
    L.append("## Run config (prefilled)")
    L.append(f"- model: `{cfg_pre.get('model')}`  n_items: {cfg_pre.get('n_items')}  "
             f"max_new_tokens: {cfg_pre.get('max_new_tokens')}  seed: {cfg_pre.get('seed')}")
    if cfg_ss is not None:
        L.append("")
        L.append("## Run config (single_shot)")
        L.append(f"- model: `{cfg_ss.get('model')}`  n_items: {cfg_ss.get('n_items')}  "
                 f"max_new_tokens: {cfg_ss.get('max_new_tokens')}  seed: {cfg_ss.get('seed')}")
    L.append("")

    def _rates_table(rates: dict, title: str, conds: List[str]):
        L.append(f"## Condition rates — {title}")
        L.append("")
        L.append("| Cond | n | commit_W_anywhere | hallucinated_obs | final_present | "
                 "first_is_action(p0) | first_is_final(p0) | parse_failure |")
        L.append("|---|---|---|---|---|---|---|---|")
        for c in conds:
            r = rates[c]
            L.append(
                f"| {c} | {r['n']} | {fmt_pct(r['commit_W_anywhere_rate'])} | "
                f"{fmt_pct(r['hallucinated_observation_rate'])} | "
                f"{fmt_pct(r['final_present_rate'])} | "
                f"{fmt_pct(r['first_is_action_rate'])} | "
                f"{fmt_pct(r['first_is_final_rate'])} | "
                f"{fmt_pct(r['parse_failure_rate'])} |"
            )
        L.append("")
        L.append(f"Overall parse-failure rate: **{fmt_pct(rates['_overall_parse_failure_rate'])}**")
        L.append("")

    _rates_table(rates_pre, "prefilled (multi-turn ReAct)", conditions_pre)
    if rates_ss is not None:
        _rates_table(rates_ss, "single_shot (collapsed user prompt)",
                     conditions_ss or conditions_pre)

    if has_gradient:
        # Sole-purpose gradient table: commit_W_anywhere across T0 / T_partial /
        # T_edge with N0 baseline. The headline of this run.
        L.append("## Trap-difficulty gradient (commit_W_anywhere, prefilled)")
        L.append("")
        L.append("| Cond | overlap with constraint | n | commit_W_anywhere | first_is_final |")
        L.append("|---|---|---|---|---|")
        labels = {"T0": "30/30 min (full)",
                  "T_partial": "15/30 min (partial)",
                  "T_edge": "0/30 min, touches boundary (valid)",
                  "N0": "no slot in obs",
                  "S0": "0/30 min, well-separated (valid)"}
        for c in ["T0", "T_partial", "T_edge", "S0", "N0"]:
            if c not in conditions_pre:
                continue
            r = rates_pre[c]
            L.append(f"| {c} | {labels[c]} | {r['n']} | "
                     f"{fmt_pct(r['commit_W_anywhere_rate'])} | "
                     f"{fmt_pct(r['final_present_rate'])} |")
        L.append("")

    def _within_table(d: dict, title: str, contrasts: List[Tuple[str, str, str]]):
        L.append(f"## Within-mode paired contrasts — {title}")
        L.append("")
        L.append("| Metric | Contrast | n | rate_a | rate_b | delta | b | c | McNemar p | 95% CI |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for key, _ca, _cb in contrasts:
            for m in METRICS:
                ct = d.get(f"{key}__{m}", {})
                if ct.get("n", 0) == 0:
                    continue
                L.append(
                    f"| {m} | {key.replace('_',' ')} | {ct['n']} | "
                    f"{fmt_pct(ct['rate_a'])} | {fmt_pct(ct['rate_b'])} | "
                    f"{fmt_pct(ct['delta'])} | {ct['b_a_yes_b_no']} | "
                    f"{ct['c_a_no_b_yes']} | {fmt_p(ct['mcnemar_p'])} | "
                    f"{fmt_ci(ct['bootstrap_ci_95'])} |"
                )
        L.append("")

    _within_table(within_pre, "prefilled", contrasts_pre)
    if within_ss is not None:
        _within_table(within_ss, "single_shot",
                      contrasts_ss or contrasts_pre)

    if cross is not None:
        L.append("## Structural ablation: prefilled vs single_shot at T0 (paired by item)")
        L.append("")
        L.append(
            "This is the headline ablation. Prior 8 single-shot pilots (billing, "
            "meeting v1, tax v1/v2/v3) failed to elicit any commit-to-trap effect on "
            "Qwen2.5-7B-Instruct. Switching to a multi-turn ReAct prefill (system → "
            "user → assistant scratchpad with one observation → decision point) on the "
            "same trap content unlocks the effect. The contrast below isolates the "
            "structural change."
        )
        L.append("")
        L.append("| Metric | n | rate_pre(T0) | rate_ss(T0) | delta | b | c | McNemar p | 95% CI |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for m in METRICS:
            ct = cross.get(m, {})
            if ct.get("n", 0) == 0:
                continue
            L.append(
                f"| {m} | {ct['n']} | {fmt_pct(ct['rate_a'])} | "
                f"{fmt_pct(ct['rate_b'])} | {fmt_pct(ct['delta'])} | "
                f"{ct['b_a_yes_b_no']} | {ct['c_a_no_b_yes']} | "
                f"{fmt_p(ct['mcnemar_p'])} | {fmt_ci(ct['bootstrap_ci_95'])} |"
            )
        L.append("")

    def _ex_block(title: str, lst):
        L.append(f"### {title}")
        L.append("")
        if not lst:
            L.append("(none)")
            L.append("")
            return
        for r in lst:
            raw = (r.get("raw") or "").replace("\n", "\\n")
            L.append(f"- `{r['item_id']}` [{r['condition']}] W={r['W_str']}  "
                     f"final_text={r.get('final_text')!r}")
            L.append(f"    raw: `{raw}`")
        L.append("")

    L.append("## Example raw outputs")
    L.append("")
    L.append("### Prefilled (multi-turn ReAct)")
    L.append("")
    t0_trap_pre, n0_pre, s0_pre = examples_pre
    _ex_block("3 T0 trap commits (commit_W_anywhere = 1)", t0_trap_pre)
    _ex_block("3 N0 non-commits (commit_W_anywhere = 0)", n0_pre)
    _ex_block("3 S0 correct joint-slot finalizations (commit_W = 1)", s0_pre)

    if examples_ss is not None:
        L.append("### Single_shot")
        L.append("")
        t0_trap_ss, n0_ss, s0_ss = examples_ss
        _ex_block("Up to 3 T0 trap commits (commit_W_anywhere = 1)", t0_trap_ss)
        _ex_block("3 N0 non-commits (commit_W_anywhere = 0)", n0_ss)
        _ex_block("3 S0 (commit_W = 1 if any)", s0_ss)

    L.append("## Caveats / blacklist")
    L.append("")
    if is_codesearch:
        bls = [
            "This proves agent dev assistants always pick stale code in production.",
            "Single-shot code-search agents are immune to premature commitment in general.",
            "The model fails to read the legacy/ path component — S0 with the same "
            "structure picks the canonical path; what fails in T0 is comparison of the "
            "name-match prior against the path-prefix evidence at the decision point.",
            "This validates the L18-KV2 → L20-MLP circuit on non-QA tasks. (Mechanism "
            "co-localization on this scenario is not yet measured.)",
            "The single-shot/prefilled gap means single-shot agents never commit "
            "prematurely — it means that for THIS surface, on Qwen2.5-7B-Instruct, the "
            "prefill-into-decision-point structure is the trigger.",
        ]
    elif is_v2:
        bls = [
            "This proves the mechanism generalizes to all real-world tools.",
            "Single-shot agents are immune to premature commitment in general.",
            "Prefilled-T0 commits to the trap because the model failed to read the "
            "constraint — the same model finalizes the constraint-respecting slot in "
            "S0 with the same prompt structure, so the constraint IS being read; what "
            "fails is constraint-checking against the surface candidate at p0.",
            "This validates the L18-KV2 → L20-MLP circuit on non-QA tasks. (Mechanism "
            "co-localization on this scenario is not yet measured.)",
            "The single-shot/prefilled gap means single-shot agents never commit "
            "prematurely — it means that for THIS surface, on Qwen2.5-7B-Instruct, "
            "the prefill-into-decision-point structure is the trigger; other models "
            "and other surfaces may differ.",
        ]
    else:
        bls = [
            "This proves the mechanism generalizes to all real-world tools.",
            "Single-shot agents are immune to premature commitment.",
            "The trap fires at p0 — it does NOT; the model issues a second action at p0 "
            "and then hallucinates the second observation in the body of that action.",
            "The hallucinated-observation pathology is novel — it is a known agent-safety "
            "failure mode; this experiment shows it is the mechanism here.",
            "This validates the L18-KV2 → L20-MLP circuit on non-QA tasks.",
        ]
    for bl in bls:
        L.append(f"- ❌ \"{bl}\"")
    L.append("")
    out_md.write_text("\n".join(L))



def _within_mode(by_ic, item_ids, mode_label: str, n_boot: int,
                 contrasts: List[Tuple[str, str, str]]) -> dict:
    out = {}
    for key, c_a, c_b in contrasts:
        for m in METRICS:
            ct = paired_contrast(
                by_ic, by_ic,
                (mode_label, c_a), (mode_label, c_b),
                item_ids, m, n_boot=n_boot,
            )
            out[f"{key}__{m}"] = ct
    return out


def _cross_mode(by_ic_pre, by_ic_ss, item_ids, n_boot: int) -> dict:
    out = {}
    for m in METRICS:
        out[m] = paired_contrast(
            by_ic_pre, by_ic_ss,
            ("prefilled", "T0"), ("single_shot", "T0"),
            item_ids, m, n_boot=n_boot,
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefilled_dir", required=True,
                    help="Run dir for mode='prefilled'.")
    ap.add_argument("--single_shot_dir", default=None,
                    help="Run dir for mode='single_shot' (optional; enables structural ablation).")
    ap.add_argument("--out_dir", default=None,
                    help="Where to write summary.json + report.md (default: --prefilled_dir).")
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    pre_dir = Path(args.prefilled_dir)
    out_dir = Path(args.out_dir) if args.out_dir else pre_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_pre = json.loads((pre_dir / "config.json").read_text())
    rows_pre = load_parsed(pre_dir / "parsed_outputs.jsonl")
    by_ic_pre = by_item_condition(rows_pre)
    item_ids_pre = sorted({r["item_id"] for r in rows_pre})
    conditions_pre = detect_conditions(rows_pre)
    contrasts_pre = detect_contrasts(conditions_pre)
    rates_pre = condition_rates(rows_pre, conditions_pre)
    within_pre = _within_mode(by_ic_pre, item_ids_pre, "prefilled", args.n_boot,
                              contrasts_pre)
    examples_pre = pick_examples(by_ic_pre, item_ids_pre, k=3)

    cfg_ss = rates_ss = within_ss = examples_ss = cross = None
    conditions_ss = contrasts_ss = None
    if args.single_shot_dir:
        ss_dir = Path(args.single_shot_dir)
        cfg_ss = json.loads((ss_dir / "config.json").read_text())
        rows_ss = load_parsed(ss_dir / "parsed_outputs.jsonl")
        by_ic_ss = by_item_condition(rows_ss)
        item_ids_ss = sorted({r["item_id"] for r in rows_ss})
        conditions_ss = detect_conditions(rows_ss)
        contrasts_ss = detect_contrasts(conditions_ss)
        rates_ss = condition_rates(rows_ss, conditions_ss)
        within_ss = _within_mode(by_ic_ss, item_ids_ss, "single_shot", args.n_boot,
                                 contrasts_ss)
        examples_ss = pick_examples(by_ic_ss, item_ids_ss, k=3)
        item_ids_common = sorted(set(item_ids_pre) & set(item_ids_ss))
        cross = _cross_mode(by_ic_pre, by_ic_ss, item_ids_common, args.n_boot)

    summary = {
        "prefilled": {
            "config": cfg_pre,
            "n_items": len(item_ids_pre),
            "conditions": conditions_pre,
            "by_condition": {c: rates_pre[c] for c in conditions_pre},
            "overall_parse_failure_rate": rates_pre["_overall_parse_failure_rate"],
            "within_mode_contrasts": within_pre,
        },
    }
    if cfg_ss is not None:
        summary["single_shot"] = {
            "config": cfg_ss,
            "n_items": len(item_ids_ss),
            "conditions": conditions_ss,
            "by_condition": {c: rates_ss[c] for c in conditions_ss},
            "overall_parse_failure_rate": rates_ss["_overall_parse_failure_rate"],
            "within_mode_contrasts": within_ss,
        }
        summary["structural_ablation_prefilled_vs_single_shot_T0"] = cross

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(out_dir / "report.md",
                 cfg_pre, rates_pre, cfg_ss, rates_ss,
                 within_pre, within_ss, cross,
                 examples_pre, examples_ss,
                 conditions_pre, conditions_ss,
                 contrasts_pre, contrasts_ss)

    print(f"[analyze] wrote: {out_dir/'summary.json'}")
    print(f"[analyze] wrote: {out_dir/'report.md'}")
    print()
    print("=== prefilled within-mode (commit_W_anywhere) ===")
    for key, _ca, _cb in contrasts_pre:
        ct = within_pre.get(f"{key}__commit_W_anywhere")
        if not ct or ct.get("n", 0) == 0:
            continue
        print(f"  {key}: rate_a={ct['rate_a']*100:.1f}%  rate_b={ct['rate_b']*100:.1f}%  "
              f"delta={ct['delta']*100:+.1f}%  McNemar p={ct['mcnemar_p']:.4g}  "
              f"CI={[round(x*100,1) for x in ct['bootstrap_ci_95']]}")
    if cross is not None:
        print()
        print("=== structural ablation prefilled-T0 vs single_shot-T0 ===")
        for m in METRICS:
            ct = cross[m]
            print(f"  {m}: rate_pre={ct['rate_a']*100:.1f}%  rate_ss={ct['rate_b']*100:.1f}%  "
                  f"delta={ct['delta']*100:+.1f}%  McNemar p={ct['mcnemar_p']:.4g}  "
                  f"CI={[round(x*100,1) for x in ct['bootstrap_ci_95']]}")


if __name__ == "__main__":
    main()
