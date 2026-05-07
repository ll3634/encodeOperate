#!/usr/bin/env python3
"""Companion plot to Figure 1: per-model layer sweep of Δ(S0−T0)_norm.

Shows that the scale-invariance (Δ ≈ 0.32) is a plateau across L_peak ± 1, not
a single-layer artefact. Each model traced with its family colour; L_peak marked.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("tmc/scripts/e2e_agent/results/scaling_difficulty_audit")

MODELS = [
    ("Qwen2.5-7B-Instruct",  "Qwen2.5", 20, "o", "-",
     ROOT / "qwen2_5_7b/st_contrast_n200/layersweep_robustness.json"),
    ("Qwen2.5-14B-Instruct", "Qwen2.5", 46, "s", "-",
     ROOT / "qwen2_5_14b/st_contrast_n200/layersweep_robustness.json"),
    ("Qwen2.5-32B-Instruct", "Qwen2.5", 50, "D", "-",
     ROOT / "qwen2_5_32b/st_contrast_n200/layersweep_robustness_peak50.json"),
    ("Qwen3-32B",            "Qwen3",   52, "^", "-",
     ROOT / "qwen3_32b/st_contrast_n200/layersweep_robustness.json"),
]

FAMILY_COLOR = {"Qwen2.5": "#1f77b4", "Qwen3": "#d62728"}
SHADE = {"Qwen2.5-7B-Instruct": 1.4, "Qwen2.5-14B-Instruct": 1.0,
         "Qwen2.5-32B-Instruct": 0.65, "Qwen3-32B": 1.0}

OUT_DIR = Path("paper/figures"); OUT_DIR.mkdir(parents=True, exist_ok=True)


def shade(rgb_hex, factor):
    rgb = np.array([int(rgb_hex[1:3], 16), int(rgb_hex[3:5], 16), int(rgb_hex[5:7], 16)])
    rgb = np.clip(rgb * factor, 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(rgb)


def main():
    fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)

    for lbl, fam, peak, marker, ls, jp in MODELS:
        d = json.load(open(jp))
        rows = d["aggregate"]["hotpotqa"]
        # plot relative to L_peak so x is comparable across models
        Lrel = np.array([r["L"] - peak for r in rows])
        m = np.array([r["delta_mean"] for r in rows])
        s = np.array([r["delta_std"] for r in rows])

        c = shade(FAMILY_COLOR[fam], SHADE[lbl])
        ax.errorbar(Lrel, m, yerr=1.96 * s, marker=marker, ls=ls, lw=1.4, ms=6,
                    capsize=3, color=c, label=f"{lbl}  (L_peak={peak})")
        # Mark L_peak
        i_peak = int(np.where(Lrel == 0)[0][0])
        ax.scatter([0], [m[i_peak]], s=140, facecolors="none",
                   edgecolors=c, linewidths=1.6, zorder=5)

    # Plateau band [0.30, 0.37] from §1.5
    ax.axhspan(0.30, 0.37, color="grey", alpha=0.12,
               label="cross-scale plateau [0.30, 0.37]")
    ax.axvline(0, color="black", lw=0.6, alpha=0.4)

    ax.set_xlabel("layer offset from each model's L_peak")
    ax.set_ylabel(r"$\Delta(\mathrm{S0}-\mathrm{T0})_{\mathrm{norm}}$  (HotpotQA, mean \u00b1 95% CI)")
    ax.set_title("Layer-sweep plateau: invariance survives L_peak \u00b1 1 in all four models")
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xlim(-2.5, 2.5)
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="lower center", fontsize=8, ncol=2, framealpha=0.95)

    for ext in ("png", "pdf"):
        out = OUT_DIR / f"fig1_layer_sweep.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=180 if ext == "png" else None)
        print(f"[save] {out}")


if __name__ == "__main__":
    main()
