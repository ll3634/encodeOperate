#!/usr/bin/env python3
"""Analyse the MuSiQue N0/T0/S0 toggle eval, mirroring
analyze_extractability_support_toggle.py but without T1 / per-schema breakdown.

Outputs summary.json + report.md side-by-side with the eval JSONL. The HotpotQA
result is loaded for a directional cross-benchmark comparison, but no inferential
test is run between the two; this is a defensive directional replication, not a
benchmark comparison study.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from analyze_extractability_support_toggle import (   # noqa: E402
    cell_stats, paired_tests,
)

CONDITIONS = ("N0", "T0", "S0")


def write_report(summary, hotpot_summary, out_path):
    cells = summary["cells"]
    pairs = summary["paired_tests"]
    pair_T0_N0 = next(p for p in pairs if p["pair"] == "T0 vs N0")
    pair_S0_T0 = next(p for p in pairs if p["pair"] == "S0 vs T0")

    def fmt(x, p=3):
        return f"{x:+.{p}f}" if isinstance(x, (int, float)) else str(x)

    lines = []
    lines.append("# MuSiQue Extractability-Support Replication\n")
    lines.append("Defensive cross-benchmark check of the N0/T0/S0 extractability-support toggle.\n")
    lines.append(f"**Model**: Qwen/Qwen2.5-7B-Instruct · **Dataset**: MuSiQue 2-hop bridge "
                 f"(`musique_ans_v1.0_dev`) · **N** = {cells['N0']['n']} samples × 3 conditions.\n")
    lines.append("Same eval pipeline as the HotpotQA toggle (`eval_extractability_cross_model.run_one`), "
                 "with the cleaner system prompt (no \"Your first word must be Action or Final\" line).\n")
    lines.append("## 1. Per-cell rates\n")
    lines.append("| cond | search | stop | commit-W | EM | parse_fail | mean ml | mean mft |")
    lines.append("|:-:|---:|---:|---:|---:|---:|---:|---:|")
    for c in CONDITIONS:
        s = cells[c]
        lines.append(f"| {c} | {s['search_rate']:.2f} | {s['stop_rate']:.2f} | "
                     f"{s['commit_W']:.2f} | {s['em']:.2f} | {s['parse_fail']:.2f} | "
                     f"{s['mean_ml']:+.2f} | {s['mean_mft']:+.2f} |")
    lines.append("")
    lines.append("`ml` = teacher-forced label margin "
                 "`logP(search\\nAction Input: | Action:) − logP(Final Answer: | Action:)`. "
                 "Positive ⇒ prefers search; negative ⇒ prefers stop.\n")

    lines.append("## 2. Paired causal contrasts (paired by sample_id)\n")
    lines.append("| contrast | Δ commit-W | McNemar (b, c) | p | Δ ml | 95% CI | perm p | Wilcoxon p |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for p in pairs:
        lo, hi = p["margin_label_ci95"]
        wp = p["margin_label_wilcoxon_p"]
        wp_s = "n/a" if wp is None else f"{wp:.3g}"
        lines.append(f"| **{p['pair']}** | {p['commitW_delta']:+.3f} | "
                     f"({p['commitW_mcnemar_b']}, {p['commitW_mcnemar_c']}) | "
                     f"{p['commitW_p']:.3g} | {p['margin_label_delta']:+.3f} | "
                     f"[{lo:+.2f}, {hi:+.2f}] | {p['margin_label_perm_p']:.3g} | {wp_s} |")
    lines.append("")

    if hotpot_summary is not None:
        lines.append("## 3. Cross-benchmark directional comparison\n")
        lines.append("Side-by-side magnitudes for the two main contrasts. No inferential test "
                     "across benchmarks; this is a directional replication check only.\n")
        lines.append("| contrast | metric | HotpotQA | MuSiQue |")
        lines.append("|---|---|---:|---:|")
        h_cells = hotpot_summary["cells"]
        h_T0_N0 = next(p for p in hotpot_summary["paired_tests"] if p["pair"] == "T0 vs N0")
        h_S0_T0 = next(p for p in hotpot_summary["paired_tests"] if p["pair"] == "S0 vs T0")
        lines.append(f"| T0 − N0 (extractability) | Δ commit-W | "
                     f"{h_T0_N0['commitW_delta']:+.2f} | {pair_T0_N0['commitW_delta']:+.2f} |")
        lines.append(f"| T0 − N0 (extractability) | Δ ml      | "
                     f"{h_T0_N0['margin_label_delta']:+.2f} | {pair_T0_N0['margin_label_delta']:+.2f} |")
        lines.append(f"| S0 − T0 (support)        | Δ commit-W | "
                     f"{h_S0_T0['commitW_delta']:+.2f} | {pair_S0_T0['commitW_delta']:+.2f} |")
        lines.append(f"| S0 − T0 (support)        | Δ ml      | "
                     f"{h_S0_T0['margin_label_delta']:+.2f} | {pair_S0_T0['margin_label_delta']:+.2f} |")
        lines.append(f"| N0 commit-W rate         | (sanity)   | "
                     f"{h_cells['N0']['commit_W']:.2f} | {cells['N0']['commit_W']:.2f} |")
        lines.append(f"| T0 commit-W rate         | (extract.) | "
                     f"{h_cells['T0']['commit_W']:.2f} | {cells['T0']['commit_W']:.2f} |")
        lines.append("")

    lines.append("## 4. Verdict\n")
    extract_dir = "directional"
    if pair_T0_N0["commitW_p"] < 0.05 and pair_T0_N0["commitW_delta"] > 0:
        extract_dir = "**replicates**"
    elif pair_T0_N0["commitW_delta"] > 0 and pair_T0_N0["margin_label_perm_p"] < 0.05:
        extract_dir = "**directional (margin-only)**"
    else:
        extract_dir = "did **not** replicate"
    lines.append(f"- Extractability effect (T0 vs N0): {extract_dir} on MuSiQue "
                 f"(Δ commit-W = {pair_T0_N0['commitW_delta']:+.2f}, "
                 f"Δ ml = {pair_T0_N0['margin_label_delta']:+.2f} nats, "
                 f"perm p = {pair_T0_N0['margin_label_perm_p']:.3g}).")
    lines.append(f"- Support-on-top effect (S0 vs T0): "
                 f"Δ commit-W = {pair_S0_T0['commitW_delta']:+.2f}, "
                 f"Δ ml = {pair_S0_T0['margin_label_delta']:+.2f} nats, "
                 f"perm p = {pair_S0_T0['margin_label_perm_p']:.3g}.")
    lines.append("- Cross-benchmark scope: this is a defensive external-validity replication, "
                 "not a benchmark comparison study. Interpretation should be limited to "
                 "directional consistency with HotpotQA, not magnitude equivalence.\n")

    Path(out_path).write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval",
        default="results/second_benchmark_extractability/qwen/eval.jsonl")
    ap.add_argument("--out-dir",
        default="results/second_benchmark_extractability/qwen")
    ap.add_argument("--hotpot-summary",
        default="results/extractability_support_toggle/summary.json",
        help="HotpotQA toggle summary.json for directional comparison (optional).")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.eval)]
    by_cond = defaultdict(list)
    for r in rows: by_cond[r["condition"]].append(r)

    cells = {c: cell_stats(by_cond[c]) for c in CONDITIONS}
    pairs = [
        paired_tests(by_cond["T0"], by_cond["N0"], "T0", "N0"),  # extractability
        paired_tests(by_cond["S0"], by_cond["T0"], "S0", "T0"),  # support-on-top
        paired_tests(by_cond["S0"], by_cond["N0"], "S0", "N0"),  # full sufficiency
    ]
    summary = {"cells": cells, "paired_tests": pairs,
               "n_total": len(rows), "dataset": "musique_2hop_bridge",
               "model": "Qwen/Qwen2.5-7B-Instruct"}
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    hotpot = None
    if Path(args.hotpot_summary).exists():
        hotpot = json.loads(Path(args.hotpot_summary).read_text())

    write_report(summary, hotpot, out_dir / "report.md")
    print(json.dumps(summary, indent=2))
    print(f"\n[done] -> {out_dir}/summary.json, {out_dir}/report.md")


if __name__ == "__main__":
    main()
