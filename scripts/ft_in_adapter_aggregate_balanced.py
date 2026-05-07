#!/usr/bin/env python3
"""Audit 4 — Aggregate the existing balanced 3-condition ReAct decomposition
(`results/ft_phaseD/decomposition_ftdirs_on_ft/decomposition_report.json`) into
the format requested by the audit task.

Reads the per-condition `*_results.jsonl` if present to compute additional
per-example diagnostics; otherwise relies on aggregated stats already present
in `decomposition_report.json`.

Output: results/ft_in_adapter_d4/balanced_decomposition.json
"""
import json
from pathlib import Path

SRC = Path("results/ft_phaseD/decomposition_ftdirs_on_ft/decomposition_report.json")
OUT = Path("results/ft_in_adapter_d4/balanced_decomposition.json")


def cond_summary(name, c):
    s = c["stats"]
    return {
        "direction": name,
        "n": s["n"],
        "baseline_em": s["baseline_rate"],
        "policy_em": s["policy_rate"],
        "rescued": s["rescued"],
        "regressed": s["regressed"],
        "net_em": s["net_gain"],
        "net_em_corrected": s.get("net_gain_corrected"),
        "rescue_rate": s["rescue_rate"],
        "regression_rate": s["regression_rate"],
        "parse_failures": s["parse_failures"],
        "rescued_genuine": s.get("rescued_genuine"),
        "rescued_accidental": s.get("rescued_accidental"),
        "rescued_causal_pct_search": s["rescued_causal_pct"],
        "bl_second_search_rate": s["bl_second_search_rate"],
        "po_second_search_rate": s["po_second_search_rate"],
        "second_search_rate_delta": s["second_search_rate_delta"],
        "bl_search_rate": s["bl_search_rate"],
        "po_search_rate": s["po_search_rate"],
        "search_rate_delta": s["search_rate_delta"],
        "avg_search_count_delta": s["avg_search_count_delta"],
        "bl_f1_mean": s["bl_f1_mean"],
        "po_f1_mean": s["po_f1_mean"],
        "f1_delta": s["f1_delta"],
        "mcnemar_p": s["mcnemar_p"],
        "success_diff": s["success_diff"],
        "success_diff_ci": s["success_diff_ci"],
    }


def main():
    rep = json.loads(SRC.read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    full = rep["conditions"]["full"]["stats"]
    par  = rep["conditions"]["parallel"]["stats"]
    perp = rep["conditions"]["perp"]["stats"]
    geom = rep["direction_geometry"]

    out = {
        "source": str(SRC),
        "model": rep["model"],
        "adapter_tag": "balanced",
        "adapter_path": "adapters/qwen_balanced_v1",
        "rho": rep["rho"],
        "layer": rep["layer"],
        "timing": rep["timing"],
        "n_samples": rep["n_samples"],
        "baseline_em": rep["baseline_acc"],
        "baseline_2nd_search_rate": rep["baseline_2nd_search_rate"],
        "direction_files": rep["direction_files"],
        "direction_geometry": {
            **geom,
            "parallel_share_norm_pct": (
                geom["parallel_norm"] / geom["full_norm"]) * 100,
        },
        "conditions": {
            "full":     cond_summary("full",     rep["conditions"]["full"]),
            "parallel": cond_summary("parallel", rep["conditions"]["parallel"]),
            "perp":     cond_summary("perp",     rep["conditions"]["perp"]),
        },
        "headline_metrics_2nd_search_rate": {
            "baseline":  rep["baseline_2nd_search_rate"],
            "full":      full["po_second_search_rate"],
            "parallel":  par ["po_second_search_rate"],
            "perp":      perp["po_second_search_rate"],
            "delta_full":     full["second_search_rate_delta"],
            "delta_parallel": par ["second_search_rate_delta"],
            "delta_perp":     perp["second_search_rate_delta"],
            "ratio_perp_over_full":
                (perp["second_search_rate_delta"] /
                 full["second_search_rate_delta"]
                 if abs(full["second_search_rate_delta"]) > 1e-12 else None),
            "ratio_parallel_over_full":
                (par["second_search_rate_delta"] /
                 full["second_search_rate_delta"]
                 if abs(full["second_search_rate_delta"]) > 1e-12 else None),
        },
        "headline_metrics_em": {
            "baseline_em": rep["baseline_acc"],
            "full_em":     full["policy_rate"],
            "parallel_em": par ["policy_rate"],
            "perp_em":     perp["policy_rate"],
            "net_full":     full["net_gain"],
            "net_parallel": par ["net_gain"],
            "net_perp":     perp["net_gain"],
            "mcnemar_p_full":     full["mcnemar_p"],
            "mcnemar_p_parallel": par ["mcnemar_p"],
            "mcnemar_p_perp":     perp["mcnemar_p"],
        },
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[saved] {OUT}")
    h = out["headline_metrics_2nd_search_rate"]
    print(f"  baseline 2ndSR:   {h['baseline']:.4f}")
    print(f"  full     2ndSR:   {h['full']:.4f}  (Δ {h['delta_full']:+.4f})")
    print(f"  parallel 2ndSR:   {h['parallel']:.4f}  (Δ {h['delta_parallel']:+.4f})")
    print(f"  perp     2ndSR:   {h['perp']:.4f}  (Δ {h['delta_perp']:+.4f})")
    print(f"  perp/full ratio:  {h['ratio_perp_over_full']*100:+.2f}%")
    print(f"  par /full ratio:  {h['ratio_parallel_over_full']*100:+.2f}%")


if __name__ == "__main__":
    main()
