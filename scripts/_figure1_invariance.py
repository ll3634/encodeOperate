#!/usr/bin/env python3
"""Figure 1 for §1: scale-invariance of evidence magnitude vs family-sensitivity of reliability.

Two-panel figure across four checkpoints:
  - Panel A: Δ(S0−T0)_norm at L_peak (mean ± 5-seed std).
  - Panel B: paired d at L_peak (mean ± 5-seed std).
Family-coloured bars; same-family Qwen2.5 chain (7B → 14B → 32B) shares one colour,
Qwen3-32B uses a contrasting colour to show the cross-family jump in B.

Inputs are the layersweep_robustness*.json files written by
_robustness_st_contrast_layersweep.py. Output paper/figures/fig1_invariance.{png,pdf}.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("tmc/scripts/e2e_agent/results/scaling_difficulty_audit")

# (label, params_B, family, peak_layer, json path).
MODELS = [
    ("Qwen2.5-7B-Instruct",  7,  "Qwen2.5", 20,
     ROOT / "qwen2_5_7b/st_contrast_n200/layersweep_robustness.json"),
    ("Qwen2.5-14B-Instruct", 14, "Qwen2.5", 46,
     ROOT / "qwen2_5_14b/st_contrast_n200/layersweep_robustness.json"),
    ("Qwen2.5-32B-Instruct", 32, "Qwen2.5", 50,
     ROOT / "qwen2_5_32b/st_contrast_n200/layersweep_robustness_peak50.json"),
    ("Qwen3-32B",            32, "Qwen3",   52,
     ROOT / "qwen3_32b/st_contrast_n200/layersweep_robustness.json"),
]

FAMILY_COLOR = {"Qwen2.5": "#1f77b4", "Qwen3": "#d62728"}

OUT_DIR = Path("paper/figures"); OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_peak(path: Path, peak: int):
    """Pull the (Δ_mean, Δ_std, d_mean, d_std) at peak layer for HotpotQA."""
    d = json.load(open(path))
    rows = d["aggregate"]["hotpotqa"]
    row = next(r for r in rows if r["L"] == peak)
    return row["delta_mean"], row["delta_std"], row["d_mean"], row["d_std"]


def main():
    labels, params, families, peaks, vals = [], [], [], [], []
    for lbl, p, fam, peak, jp in MODELS:
        d_m, d_s, dpd_m, dpd_s = load_peak(jp, peak)
        labels.append(lbl); params.append(p); families.append(fam); peaks.append(peak)
        vals.append((d_m, d_s, dpd_m, dpd_s))
        print(f"{lbl:24s} L{peak:>2d}  Δ={d_m:+.4f}\u00b1{d_s:.4f}  "
              f"d={dpd_m:+.3f}\u00b1{dpd_s:.3f}")

    delta_m  = np.array([v[0] for v in vals])
    delta_s  = np.array([v[1] for v in vals])
    pd_m     = np.array([v[2] for v in vals])
    pd_s     = np.array([v[3] for v in vals])
    colors   = [FAMILY_COLOR[f] for f in families]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    x = np.arange(len(labels))

    # --- Panel A: Δ magnitude (scale-invariance) ---
    ax = axes[0]
    bars = ax.bar(x, delta_m, yerr=1.96 * delta_s, color=colors,
                  edgecolor="black", linewidth=0.8, capsize=4, error_kw=dict(lw=1.0))
    # cross-model mean band
    band_lo, band_hi = delta_m.mean() - delta_m.std(ddof=1), delta_m.mean() + delta_m.std(ddof=1)
    ax.axhspan(band_lo, band_hi, color="grey", alpha=0.15,
               label=f"4-model mean \u00b11sd: {delta_m.mean():.3f}\u00b1{delta_m.std(ddof=1):.3f}")
    ax.axhline(delta_m.mean(), color="grey", linestyle="--", linewidth=0.8)
    for xi, v, s in zip(x, delta_m, delta_s):
        ax.text(xi, v + 1.96 * s + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l.split('-')[0]}\n{p}B  L{pk}" for l, p, pk in zip(labels, params, peaks)],
                       fontsize=9)
    ax.set_ylabel(r"$\Delta(\mathrm{S0}-\mathrm{T0})_{\mathrm{norm}}$  (mean \u00b1 95% CI, 5 seeds)")
    ax.set_title("A. Evidence magnitude is scale- and family-invariant")
    ax.set_ylim(0, max(delta_m + 1.96 * delta_s) * 1.25)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # --- Panel B: paired-d (reliability) ---
    ax = axes[1]
    ax.bar(x, pd_m, yerr=1.96 * pd_s, color=colors,
           edgecolor="black", linewidth=0.8, capsize=4, error_kw=dict(lw=1.0))
    for xi, v, s in zip(x, pd_m, pd_s):
        ax.text(xi, v + 1.96 * s + 0.18, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    # within-Qwen2.5 chain
    qwen25_idx = [i for i, f in enumerate(families) if f == "Qwen2.5"]
    ax.plot(np.array(qwen25_idx), pd_m[qwen25_idx], "o--",
            color=FAMILY_COLOR["Qwen2.5"], lw=1.0, ms=5,
            label="Qwen2.5 family chain (no scaling trend)")
    # cross-family arrow at 32B
    i32_2_5 = next(i for i, (l, f) in enumerate(zip(labels, families))
                   if f == "Qwen2.5" and "32B" in l)
    i32_3   = next(i for i, (l, f) in enumerate(zip(labels, families))
                   if f == "Qwen3"   and "32B" in l)
    ax.annotate("", xy=(i32_3, pd_m[i32_3]), xytext=(i32_2_5, pd_m[i32_2_5]),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=FAMILY_COLOR["Qwen3"]))
    ax.text((i32_2_5 + i32_3) / 2, (pd_m[i32_2_5] + pd_m[i32_3]) / 2 + 0.4,
            "+111% cross-family\n@ 32B", ha="center", va="bottom", fontsize=8.5,
            color=FAMILY_COLOR["Qwen3"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l.split('-')[0]}\n{p}B  L{pk}" for l, p, pk in zip(labels, params, peaks)],
                       fontsize=9)
    ax.set_ylabel("paired d  (mean \u00b1 95% CI, 5 seeds)")
    ax.set_title("B. Per-sample reliability is family-, not scale-, sensitive")
    ax.set_ylim(0, max(pd_m + 1.96 * pd_s) * 1.20)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Family legend (proxy patches in shared-figure space)
    from matplotlib.patches import Patch
    family_handles = [Patch(facecolor=FAMILY_COLOR[f], edgecolor="black", label=f)
                      for f in ("Qwen2.5", "Qwen3")]
    fig.legend(handles=family_handles, loc="lower center",
               ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("HotpotQA, L_peak, N=200 paired sids, 5-seed train/test", fontsize=10,
                 y=1.02)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"fig1_invariance.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=180 if ext == "png" else None)
        print(f"[save] {out}")


if __name__ == "__main__":
    main()
