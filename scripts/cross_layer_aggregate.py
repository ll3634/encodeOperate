#!/usr/bin/env python3
"""Aggregate per-model cross_layer_cos trajectories into one figure JSON +
draft PNG, and emit the DECORRELATION/UNIFORM verdict.

Inputs:
  results/cross_layer_cos/{qwen25,llama31,mistral,r1distill,gemma2}/trajectory.json

Outputs:
  results/cross_layer_cos/figure_layer_trajectory.json
  results/cross_layer_cos/figure_layer_trajectory.png
  results/cross_layer_cos/verdict.md
"""

import json
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("results/cross_layer_cos")

MODELS = [
    ("qwen25",     "Qwen2.5-7B-Instruct",            20, "#1f77b4", "o"),
    ("llama31",    "Llama-3.1-8B-Instruct",          28, "#d62728", "s"),
    ("mistral",    "Mistral-7B-Instruct-v0.3",       28, "#2ca02c", "D"),
    ("r1distill",  "DeepSeek-R1-Distill-Qwen-7B",    22, "#9467bd", "^"),
    ("gemma2",     "Gemma-2-9B-Instruct",            37, "#ff7f0e", "v"),
]

DECORR_THRESHOLD = 0.25


def load(model_dir):
    p = ROOT / model_dir / "trajectory.json"
    if not p.exists():
        return None
    return json.load(open(p))


def main():
    figure_data = {
        "experiment": "cross_model_layer_trajectory_cos_evidence_action",
        "decorr_threshold": DECORR_THRESHOLD,
        "models": [],
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    summary_rows = []
    for key, label, peak, color, marker in MODELS:
        d = load(key)
        if d is None:
            print(f"[skip] {key}: no trajectory.json")
            continue
        rows = d["rows"]
        layers_all = np.array([r["layer"] for r in rows])
        cos_all = np.array([r.get("cos_point", np.nan) for r in rows], dtype=float)
        cos_lo_all = np.array([r.get("cos_ci_lo", np.nan) for r in rows], dtype=float)
        cos_hi_all = np.array([r.get("cos_ci_hi", np.nan) for r in rows], dtype=float)
        auroc_all = np.array([r.get("auroc", np.nan) for r in rows], dtype=float)
        a_qual_all = np.array([r.get("action_dir_quality", np.nan) for r in rows], dtype=float)

        keep = np.isfinite(cos_all) & np.isfinite(auroc_all)
        if not keep.any():
            print(f"[skip] {key}: no finite rows")
            continue
        layers = layers_all[keep]
        cos = cos_all[keep]
        cos_lo = cos_lo_all[keep]
        cos_hi = cos_hi_all[keep]
        auroc = auroc_all[keep]
        a_qual = a_qual_all[keep]

        # |cos| panel
        abs_cos = np.abs(cos)
        ax1.errorbar(layers, abs_cos,
                     yerr=[abs_cos - np.abs(cos_lo).clip(0, None),
                           np.maximum(np.abs(cos_hi), abs_cos) - abs_cos],
                     marker=marker, ls="-", lw=1.4, ms=5,
                     capsize=2, color=color, label=f"{label}  (L_peak={peak})")
        # Decision-layer marker
        if peak in layers.tolist():
            i = layers.tolist().index(peak)
            ax1.scatter([peak], [abs_cos[i]], s=140, facecolors="none",
                        edgecolors=color, linewidths=1.6, zorder=5)

        # Diagnostics: AUROC + action quality
        ax2.plot(layers, auroc, marker=marker, ls="-", lw=1.2, ms=4,
                 color=color, label=f"{label} AUROC")
        ax2.plot(layers, a_qual, marker=marker, ls=":", lw=1.0, ms=4,
                 color=color, alpha=0.55)

        # Per-model summary
        max_abs = float(np.max(abs_cos))
        argmax_l = int(layers[int(np.argmax(abs_cos))])
        peak_idx = int(np.argmin(np.abs(layers - peak))) if peak in layers else -1
        peak_cos = float(cos[peak_idx]) if peak_idx >= 0 else float("nan")
        peak_abs = float(abs_cos[peak_idx]) if peak_idx >= 0 else float("nan")
        any_decorr = bool(np.any(abs_cos >= DECORR_THRESHOLD))
        early_layers = layers <= int(d["n_layers"] * 0.4)
        early_max = float(np.max(abs_cos[early_layers])) if early_layers.any() else float("nan")

        figure_data["models"].append({
            "key": key, "label": label,
            "model_id": d["model"],
            "n_layers": d["n_layers"],
            "decision_layer": peak,
            "sweep_layers": layers.tolist(),
            "cos": cos.tolist(),
            "cos_ci_lo": cos_lo.tolist(),
            "cos_ci_hi": cos_hi.tolist(),
            "auroc": auroc.tolist(),
            "action_dir_quality": a_qual.tolist(),
            "max_abs_cos": max_abs,
            "max_abs_cos_layer": argmax_l,
            "early_max_abs_cos": early_max,
            "decision_cos": peak_cos,
            "decision_abs_cos": peak_abs,
            "exceeded_decorr_threshold": any_decorr,
        })
        summary_rows.append((label, max_abs, argmax_l, peak_abs, any_decorr, early_max))

    # Panel 1 cosmetics
    ax1.axhline(DECORR_THRESHOLD, ls="--", color="black", lw=0.8, alpha=0.6,
                label=f"DECORRELATION threshold |cos|={DECORR_THRESHOLD}")
    ax1.axhline(0, color="grey", lw=0.4, alpha=0.5)
    ax1.set_xlabel("layer index")
    ax1.set_ylabel(r"$|\cos(E_{\mathrm{layer}}, A_{\mathrm{layer}})|$  (95% bootstrap)")
    ax1.set_title("E/A near-orthogonality across model depth")
    ax1.set_yscale("symlog", linthresh=0.05)
    ax1.set_ylim(-0.005, 0.6)
    ax1.grid(linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=7, framealpha=0.95)

    # Panel 2 cosmetics
    ax2.set_xlabel("layer index")
    ax2.set_ylabel("evidence AUROC (solid) / action |Spearman| (dotted)")
    ax2.set_title("Probe quality diagnostics")
    ax2.set_ylim(0.45, 1.0)
    ax2.grid(linestyle=":", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=7, framealpha=0.95)

    out_png = ROOT / "figure_layer_trajectory.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=160)
    print(f"[save] {out_png}")

    # Verdict
    overall_max = max(r[1] for r in summary_rows) if summary_rows else 0.0
    any_decorr = any(r[4] for r in summary_rows)
    verdict = "DECORRELATION" if any_decorr else "UNIFORM"
    figure_data["overall_max_abs_cos"] = overall_max
    figure_data["any_model_exceeds_threshold"] = any_decorr
    figure_data["verdict"] = verdict

    out_json = ROOT / "figure_layer_trajectory.json"
    with open(out_json, "w") as f:
        json.dump(figure_data, f, indent=2)
    print(f"[save] {out_json}")

    # Verdict markdown
    lines = [
        "# Cross-Model Layer-wise cos(E, A) Trajectory — Verdict",
        "",
        f"**Verdict: {verdict}**  (max |cos| across all models/layers = {overall_max:.4f}; "
        f"DECORRELATION threshold = {DECORR_THRESHOLD})",
        "",
        "| Model | n_layers | decision L | |cos|@decision | max |cos| | argmax L | early-layer max | exceeds 0.25? |",
        "|---|---:|---:|---:|---:|---:|---:|:--:|",
    ]
    for label, mx, ar, pcos, dec, early in summary_rows:
        lines.append(f"| {label} | — | — | {pcos:.4f} | {mx:.4f} | L{ar} | {early:.4f} | {'YES' if dec else 'no'} |")
    out_md = ROOT / "verdict.md"
    out_md.write_text("\n".join(lines))
    print(f"[save] {out_md}")
    print(f"\n=== VERDICT: {verdict} ===")
    print(f"  overall max |cos| = {overall_max:.4f}")


if __name__ == "__main__":
    main()
