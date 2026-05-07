#!/usr/bin/env python3
"""
Analysis + report generator for the tax-inclusive total sanity check.

Reuses statistical machinery (paired McNemar + bootstrap CI) from the billing
analyzer; writes a tax-specific report.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent))

from analyze_nonqa_billing_sanity import (
    CONDITIONS,
    load_parsed,
    by_item_condition,
    condition_rates,
    paired_contrast,
    evaluate_pilot_gates,
    pick_examples,
)


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
    lines.append("# Tax-inclusive total sanity check")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This is a behavior-only non-QA sanity check. The agent must compute the "
        "after-tax total payable on an invoice. After one tool observation, it must "
        "either query the tax rate or commit a final amount. Conditions:"
    )
    lines.append("")
    lines.append("- **N0**: Line-item count + jurisdiction; no money. (W = subtotal value)")
    lines.append("- **T0**: Line-item count + `Subtotal: $X (pre-tax)` + jurisdiction. **Trap**: tax rate still missing. (W = subtotal value)")
    lines.append("- **IC**: Line-item count + length-matched irrelevant note (invoice date + email) + jurisdiction. (W = subtotal value)")
    lines.append("- **S0**: Subtotal + Tax rate + Total all visible. (W = total value)")
    lines.append("")
    lines.append(
        "Primary metric `commit_W` = the model finalizes `AMOUNT=` exactly equal to W. "
        "In N0/T0/IC, W is the **subtotal**: a commit_W=1 event in T0 is the targeted "
        "trap (model treats the pre-tax subtotal as the final answer). In N0/IC the "
        "subtotal value is not present in the observation, so commit_W is a "
        "near-zero-by-construction baseline. In S0, W is the **total**: commit_W=1 is "
        "the correct competence behavior."
    )
    lines.append("")
    lines.append("## Run config")
    lines.append("")
    lines.append(f"- model: `{config.get('model')}`")
    lines.append(f"- n_items: {config.get('n_items')}  (pilot={config.get('pilot')})")
    lines.append(f"- max_new_tokens: {config.get('max_new_tokens')}")
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
        v_str = fmt_pct(val) if abs(val) <= 1.5 else f"{val:.4f}"
        lines.append(f"| {k} | {flag} | {v_str} |")
    lines.append("")
    lines.append(f"**pilot_pass = {pilot_eval['pilot_pass']}** "
                 f"(must-pass = {', '.join(pilot_eval['must_pass_keys'])})")
    lines.append("")

    lines.append("## Paired contrasts on commit_W")
    lines.append("")
    lines.append("| Contrast | n | rate_a | rate_b | delta | b (a=1,b=0) | c (a=0,b=1) | McNemar p | 95% CI |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for label, key in [("T0 vs N0", "T0_vs_N0"),
                       ("T0 vs IC", "T0_vs_IC"),
                       ("S0 vs T0", "S0_vs_T0")]:
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
    _ex("3 T0 failures (commit_W = subtotal; the targeted error)", t0_fail)
    _ex("3 N0 correct searches (first_is_search = 1)", n0_ok)

    lines.append("## Caveats / blacklist")
    lines.append("")
    lines.append("Do not derive any of the following claims from this single sanity check:")
    lines.append("")
    for bl in [
        "This proves the QA mechanism generalizes to all tool-using settings.",
        "Real-world invoice-processing agents fail this exact way at this exact rate.",
        "The circuit identified in the QA experiments is the same circuit firing here.",
        "The model cannot perform tax computations.",
        "This rules out RAG / context-length artifacts (only the IC contrast does).",
    ]:
        lines.append(f"- ❌ \"{bl}\"")
    lines.append("")

    out_md.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
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
