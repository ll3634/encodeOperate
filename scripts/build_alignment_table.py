#!/usr/bin/env python3
"""Build aligned_model_table.json joining behavior cells with representation numbers.

Inputs:
  Behavior summaries (per-model summary_*.json from analyze_extractability_cross_model.py)
  Representation results (cross_model_*_v2/full_results.json)

Output: results/cross_model_behavior_alignment/aligned_model_table.json
"""
import json
from pathlib import Path

ROOT = Path("results")
OUT  = ROOT / "cross_model_behavior_alignment" / "aligned_model_table.json"

MODELS = [
    {
        "tag": "qwen2_5_7b",
        "display": "Qwen2.5-7B-Instruct",
        "hf_path": "Qwen/Qwen2.5-7B-Instruct",
        "behavior": ROOT / "cross_model_extractability/summary_qwen2_5_7b.json",
        # Qwen primary: AUROC and cos sourced from project corpus (see CLAUDE.md
        # \u00a74.2 / \u00a74.3 / \u00a74.6); paired corruption from main paired report.
        "representation_override": {
            "evidence_auroc": 0.862,
            "cos_action_evidence": -0.0135,
            "ab_ratio_action": 1.83,
            "ab_pvalue_action": 4e-4,
            "evi_layer": 20, "act_layer": 20,
            "n_layers": 28, "hidden": 3584,
            "n_corruption_pairs": 50,
            "source": "project main paired report (N=50)",
        },
    },
    {
        "tag": "mistral_7b_v03",
        "display": "Mistral-7B-Instruct-v0.3",
        "hf_path": "mistralai/Mistral-7B-Instruct-v0.3",
        "behavior": ROOT / "cross_model_extractability/summary_mistral_7b_v03.json",
        "representation": ROOT / "cross_model_mistral_v2/full_results.json",
    },
    {
        "tag": "gemma_2_9b_it",
        "display": "Gemma-2-9B-it",
        "hf_path": "unsloth/gemma-2-9b-it",
        "behavior": ROOT / "cross_model_behavior_alignment/summary_gemma_2_9b_it.json",
        "representation": ROOT / "cross_model_gemma2_v2/full_results.json",
    },
    # Boundary cases (appendix)
    {
        "tag": "llama_3_1_8b",
        "display": "Llama-3.1-8B-Instruct (boundary)",
        "hf_path": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "behavior": None,
        "representation": ROOT / "cross_model_llama31_v2/full_results.json",
    },
    {
        "tag": "r1_distill_qwen_7b",
        "display": "DeepSeek-R1-Distill-Qwen-7B (boundary)",
        "hf_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "behavior": ROOT / "cross_model_extractability/summary_r1_distill_qwen_7b.json",
        "representation": ROOT / "cross_model_r1distill_v2/full_results.json",
    },
]


def load_behavior(path):
    if path is None or not path.exists():
        return None
    s = json.load(open(path))
    cells = s["cells"]
    contrasts = s["contrasts"]
    return {
        "n_records": s["n_records"],
        "cells": {
            c: {
                "first_search_rate": cells[c]["first_search_rate"],
                "first_stop_rate":   cells[c]["first_stop_rate"],
                "commit_W":          cells[c]["commit_W"],
                "em":                cells[c]["em"],
                "parse_fail":        cells[c]["parse_fail"],
                "mean_label_margin": cells[c]["mean_ml"],
            } for c in ("N0", "T0", "S0")
        },
        "contrast_T0_vs_N0": {
            "delta_search": contrasts["T0_vs_N0"]["first_is_search"]["delta_rate"],
            "delta_commit_W": contrasts["T0_vs_N0"]["commit_W"]["delta_rate"],
            "delta_margin_label": contrasts["T0_vs_N0"]["commit_W"]["delta_margin_label"],
            "mcnemar_p_search": contrasts["T0_vs_N0"]["first_is_search"]["mcnemar_p"],
            "mcnemar_p_commit_W": contrasts["T0_vs_N0"]["commit_W"]["mcnemar_p"],
            "wilcoxon_p_margin": contrasts["T0_vs_N0"]["commit_W"]["wilcoxon_p_margin"],
        },
        "contrast_S0_vs_T0": {
            "delta_commit_W": contrasts["S0_vs_T0"]["commit_W"]["delta_rate"],
            "delta_em":       contrasts["S0_vs_T0"]["em"]["delta_rate"],
            "delta_margin_label": contrasts["S0_vs_T0"]["commit_W"]["delta_margin_label"],
            "mcnemar_p_commit_W": contrasts["S0_vs_T0"]["commit_W"]["mcnemar_p"],
            "mcnemar_p_em":       contrasts["S0_vs_T0"]["em"]["mcnemar_p"],
            "wilcoxon_p_margin":  contrasts["S0_vs_T0"]["commit_W"]["wilcoxon_p_margin"],
        },
    }


def load_representation(spec):
    if "representation_override" in spec:
        return spec["representation_override"]
    p = spec.get("representation")
    if p is None or not p.exists():
        return None
    d = json.load(open(p))
    return {
        "evidence_auroc": d["evidence_probe"]["auroc_mean"],
        "cos_action_evidence": d["orthogonality"]["cos_same_layer"],
        "cos_action_evidence_xlayer": d["orthogonality"]["cos_action_evidence"],
        "ab_ratio_action":  d["paired_corruption"]["AB_ratio_action"],
        "ab_pvalue_action": d["paired_corruption"]["MW_action_p"],
        "evi_layer": d["peak_evidence_layer"],
        "act_layer": d["peak_action_layer"],
        "n_layers": d["n_layers"],
        "hidden":   d["hidden_size"],
        "n_corruption_pairs": d["paired_corruption"]["n_samples"],
        "source": str(p),
    }


def main():
    out = {"models": {}}
    for spec in MODELS:
        out["models"][spec["tag"]] = {
            "display": spec["display"],
            "hf_path": spec["hf_path"],
            "behavior": load_behavior(spec["behavior"]),
            "representation": load_representation(spec),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[wrote] {OUT}")


if __name__ == "__main__":
    main()
