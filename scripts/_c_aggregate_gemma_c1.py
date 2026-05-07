#!/usr/bin/env python3
"""C: aggregate existing Gemma-2-9B-it eval.jsonl files into c1-format
behavioral_baseline.json so the scaling table can fold in the non-Qwen row.
Reuses cell_stats / paired_tests from analyze_extractability_support_toggle.
"""
import json, sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))
from analyze_extractability_support_toggle import cell_stats, paired_tests  # noqa

CONDITIONS = ("N0", "T0", "S0")
SOURCES = {
    "hotpotqa": "results/cross_benchmark_clean_prompt_hotpotqa/gemma/eval.jsonl",
    "musique":  "results/second_benchmark_extractability/gemma/eval.jsonl",
}
E2E_ROOT = _HERE.parent
OUT_DIR = E2E_ROOT / "results" / "gemma_2_9b_scale_check" / "c1"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_dataset = {}
    for ds, rel in SOURCES.items():
        eval_path = E2E_ROOT / rel
        if not eval_path.exists():
            print(f"[skip] {ds}: missing {eval_path}"); continue
        recs = [json.loads(l) for l in open(eval_path)]
        recs = [r for r in recs if r.get("condition") in CONDITIONS]
        by = defaultdict(list)
        for r in recs:
            by[r["condition"]].append(r)
        cells = {c: cell_stats(by[c]) for c in CONDITIONS if by[c]}
        pairs = []
        if by.get("T0") and by.get("N0"):
            pairs.append(paired_tests(by["T0"], by["N0"], "T0", "N0"))
        if by.get("S0") and by.get("T0"):
            pairs.append(paired_tests(by["S0"], by["T0"], "S0", "T0"))
        if by.get("S0") and by.get("N0"):
            pairs.append(paired_tests(by["S0"], by["N0"], "S0", "N0"))
        per_dataset[ds] = {
            "eval_jsonl": str(eval_path), "n_records": len(recs),
            "cells_gemma2_9b": cells, "paired_tests_gemma2_9b": pairs,
        }
        print(f"\n=== gemma-2-9b-it {ds} ({len(recs)} recs) ===")
        for c in CONDITIONS:
            if c not in cells: continue
            cs = cells[c]
            print(f"  {c}: n={cs['n']:3d}  search={cs['search_rate']:.2f}  "
                  f"commit_W={cs['commit_W']:.2f}  mean_ml={cs['mean_ml']:+.3f}  "
                  f"em={cs['em']:.2f}  pf={cs['parse_fail']:.2f}")
        for p in pairs:
            print(f"  {p['pair']}: ΔcommitW={p['commitW_delta']:+.2f}  "
                  f"mcnemar_p={p['commitW_p']:.4f}  Δmargin={p['margin_label_delta']:+.3f}  "
                  f"perm_p={p['margin_label_perm_p']:.4f}")

    out = {"model": "google/gemma-2-9b-it",
           "datasets": per_dataset, "source_note": "Aggregated from existing Gemma eval.jsonl"}
    out_path = OUT_DIR / "behavioral_baseline.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
