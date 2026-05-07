#!/usr/bin/env python3
"""C1: Qwen3-32B N0/T0/S0 behavioural baseline on HotpotQA + MuSiQue.

Loads the 32B model once and runs `run_one` from
eval_extractability_cross_model.py over both pairs files, then aggregates
per-condition commit-W / first-search rate / margin and 7B reference deltas
into a single behavioral_baseline.json.

Path B framing reminder: a LOWER 32B commit-W vs 7B is the substantive result;
do not "fix" it by switching prompts.
"""
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))
from eval_extractability_cross_model import run_one, is_r1_model  # noqa: E402
from analyze_extractability_support_toggle import cell_stats, paired_tests  # noqa: E402

DATASETS = {
    "hotpotqa": "results/extractability_support_toggle/pairs.jsonl",
    "musique":  "results/second_benchmark_extractability/pairs.jsonl",
}
CONDITIONS = ("N0", "T0", "S0")

# 7B Qwen2.5 reference (from cross_model_behavior_alignment/aligned_model_table.json
# and second_benchmark_extractability/qwen/summary.json)
QWEN25_7B_REF = {
    "hotpotqa": {
        "n_per_cell": 50,
        "cells": {
            "N0": {"first_search_rate": 0.96, "commit_W": 0.00, "mean_margin_label":  7.854},
            "T0": {"first_search_rate": 0.56, "commit_W": 0.44, "mean_margin_label":  2.936},
            "S0": {"first_search_rate": 0.00, "commit_W": 1.00, "mean_margin_label": -7.651},
        },
        "delta_T0_minus_N0": {"commit_W":  0.44, "margin_label": -4.918},
    },
    "musique": {
        "n_per_cell": 50,
        "cells": {
            "N0": {"first_search_rate": 0.08, "commit_W": 0.02, "mean_margin_label": 12.961},
            "T0": {"first_search_rate": 0.06, "commit_W": 0.46, "mean_margin_label": 10.322},
            "S0": {"first_search_rate": 0.04, "commit_W": 0.58, "mean_margin_label":  9.349},
        },
        "delta_T0_minus_N0": {"commit_W":  0.44, "margin_label": -2.639},
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/home/featurize/work/models/Qwen3-32B")
    ap.add_argument("--out-dir", default="results/qwen3_32b_scale_check/c1")
    ap.add_argument("--limit", type=int, default=None,
                    help="Per-condition cap (mainly for smoke testing).")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True); model.eval()
    device = next(model.parameters()).device
    is_r1 = is_r1_model(args.model_path)
    print(f"[ok] device={device} is_r1={is_r1}")

    per_dataset = {}
    for ds_name, pairs_path in DATASETS.items():
        rows = [json.loads(l) for l in open(pairs_path)]
        rows = [r for r in rows
                if (r.get("condition") or r.get("condition_id")) in CONDITIONS]
        if args.limit:
            kept, ct = [], defaultdict(int)
            for r in rows:
                c = r.get("condition") or r.get("condition_id")
                if ct[c] < args.limit:
                    kept.append(r); ct[c] += 1
            rows = kept
        print(f"\n[ds] {ds_name}: {len(rows)} records ({pairs_path})")

        eval_path = out_dir / f"eval_{ds_name}.jsonl"
        t0 = time.time()
        with open(eval_path, "w") as f:
            for i, rec in enumerate(rows, 1):
                row = run_one(rec, model, tok, device, args.max_new_tokens,
                              is_r1=is_r1, prompt_variant="v1", obs_style="factcard")
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
                if i % 10 == 0 or i == len(rows):
                    print(f"  [{i}/{len(rows)}] {time.time()-t0:.1f}s")

        recs = [json.loads(l) for l in open(eval_path)]
        by_cond = defaultdict(list)
        for r in recs:
            by_cond[r["condition"]].append(r)
        cells = {c: cell_stats(by_cond[c]) for c in CONDITIONS if by_cond[c]}
        pairs_stats = []
        if by_cond.get("T0") and by_cond.get("N0"):
            pairs_stats.append(paired_tests(by_cond["T0"], by_cond["N0"], "T0", "N0"))
        if by_cond.get("S0") and by_cond.get("T0"):
            pairs_stats.append(paired_tests(by_cond["S0"], by_cond["T0"], "S0", "T0"))
        if by_cond.get("S0") and by_cond.get("N0"):
            pairs_stats.append(paired_tests(by_cond["S0"], by_cond["N0"], "S0", "N0"))

        per_dataset[ds_name] = {
            "eval_jsonl": str(eval_path),
            "n_records": len(recs),
            "cells_32b": cells,
            "paired_tests_32b": pairs_stats,
            "ref_qwen2_5_7b": QWEN25_7B_REF[ds_name],
        }

    # cross-dataset deltas (32B vs 7B)
    deltas = {}
    for ds, blk in per_dataset.items():
        ref = blk["ref_qwen2_5_7b"]["cells"]
        d = {}
        for c in CONDITIONS:
            if c not in blk["cells_32b"]:
                continue
            cur = blk["cells_32b"][c]
            d[c] = {
                "commit_W_32b": cur["commit_W"],
                "commit_W_7b":  ref[c]["commit_W"],
                "delta_commit_W_32b_minus_7b": cur["commit_W"] - ref[c]["commit_W"],
                "first_search_32b": cur["search_rate"],
                "first_search_7b":  ref[c]["first_search_rate"],
                "mean_ml_32b": cur["mean_ml"],
                "mean_ml_7b":  ref[c]["mean_margin_label"],
            }
        deltas[ds] = d

    out = {
        "model_32b": args.model_path,
        "model_7b_ref": "Qwen/Qwen2.5-7B-Instruct",
        "datasets": per_dataset,
        "scale_deltas": deltas,
        "max_new_tokens": args.max_new_tokens,
    }
    (out_dir / "behavioral_baseline.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] -> {out_dir}/behavioral_baseline.json")


if __name__ == "__main__":
    main()
