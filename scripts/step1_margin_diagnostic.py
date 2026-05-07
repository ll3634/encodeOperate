#!/usr/bin/env python3
"""Persist step1_margin distributions across the 5 cross-model artifacts.

Reads per_sample.npz from each cross_model_*_v2 directory and writes
results/step1_margin_distributions/{summary.json, README.md}.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "step1_margin_distributions")

MODELS = [
    ("qwen25_7b",   "cross_model_qwen25_v2",   "Qwen/Qwen2.5-7B-Instruct"),
    ("mistral_7b",  "cross_model_mistral_v2",  "mistralai/Mistral-7B-Instruct-v0.3"),
    ("llama31_8b",  "cross_model_llama31_v2",  "unsloth/Meta-Llama-3.1-8B-Instruct"),
    ("gemma2_9b",   "cross_model_gemma2_v2",   "unsloth/gemma-2-9b-it"),
    ("r1distill_7b","cross_model_r1distill_v2","deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
]

VERDICTS = {
    "qwen25_7b":    "stop-biased: 411/486 negative, mean -5.93, std 5.10; matches the published \u00a74.6 baseline-margin \u2248 -7 anchor.",
    "mistral_7b":   "spread-with-decision-pressure: 27.2% of samples within |m|<0.5, the highest near-boundary fraction across the 5 families.",
    "llama31_8b":   "balanced: 249 positive / 228 negative, mean -0.07; the most evenly split decision locus across the 5 families.",
    "gemma2_9b":    "mild stop-bias: 290/486 negative, mean -1.67; non-saturated.",
    "r1distill_7b": "saturated: 486/486 positive, range [+2.06, +7.75]; step1 sign-label is degenerate (only family with sign-uniformity = 1.000).",
}


def classify(sign_unif, frac_near):
    if sign_unif >= 0.99 and frac_near < 0.05:
        return "SATURATED"
    if sign_unif >= 0.95:
        return "near-saturated"
    if frac_near > 0.30:
        return "boundary-rich"
    return "spread"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for short, sub, hf in MODELS:
        npz = os.path.join(ROOT, "results", sub, "per_sample.npz")
        z = np.load(npz, allow_pickle=False)
        m = z["step1_margin"].astype(np.float64)
        n = int(len(m))
        n_pos = int((m > 0).sum())
        n_neg = int((m < 0).sum())
        n_zero = int((m == 0).sum())
        n_near = int((np.abs(m) < 0.5).sum())
        sign_unif = float(max(n_pos, n_neg) / n)
        counts, edges = np.histogram(m, bins=10)
        rows.append({
            "model_short": short,
            "model_hf": hf,
            "input_npz": os.path.relpath(npz, ROOT),
            "n": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_zero": n_zero,
            "n_near_boundary_0.5": n_near,
            "min": float(m.min()),
            "max": float(m.max()),
            "mean": float(m.mean()),
            "std": float(m.std()),
            "median": float(np.median(m)),
            "p10": float(np.percentile(m, 10)),
            "p25": float(np.percentile(m, 25)),
            "p75": float(np.percentile(m, 75)),
            "p90": float(np.percentile(m, 90)),
            "sign_uniformity": sign_unif,
            "fraction_near_boundary_0.5": float(n_near / n),
            "histogram_bin_edges": [float(x) for x in edges],
            "histogram_counts": [int(c) for c in counts],
            "regime": classify(sign_unif, n_near / n),
            "verdict": VERDICTS[short],
        })

    summary = {
        "spec_version": "step1-margin-diagnostic-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "definition": (
            "step1_margin = logit(Action_token) - logit(Final_token) at the last "
            "input position of the step-1 prompt (post first-search observation, "
            "pre first-action emission). Captured by cross_model_full.py:"
            "collect_step1_states. n=486 HotpotQA bridge samples for all models."
        ),
        "rows": rows,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    write_readme(rows)
    print(f"wrote {os.path.join(OUT_DIR, 'summary.json')}")
    print(f"wrote {os.path.join(OUT_DIR, 'README.md')}")


def write_readme(rows):
    md = ["# step1_margin distributions \u2014 5-model comparative diagnostic\n"]
    md.append("spec_version: step1-margin-diagnostic-v1\n")
    md.append("step1_margin = logit(Action_token) \u2212 logit(Final_token) at the "
              "last input position of the step-1 prompt (post first-search observation).\n")
    md.append("Source: per_sample.npz['step1_margin'] from each cross_model_*_v2 directory. "
              "n=486 HotpotQA bridge samples per model.\n")
    md.append("## Distribution table\n")
    md.append("| model | n | min | p10 | med | p90 | max | mean | std | n_pos | n_neg | n |m|<0.5 | sign-unif | regime |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(
            f"| {r['model_short']} | {r['n']} | "
            f"{r['min']:+.3f} | {r['p10']:+.3f} | {r['median']:+.3f} | "
            f"{r['p90']:+.3f} | {r['max']:+.3f} | "
            f"{r['mean']:+.3f} | {r['std']:.3f} | "
            f"{r['n_pos']} | {r['n_neg']} | {r['n_near_boundary_0.5']} | "
            f"{r['sign_uniformity']:.3f} | {r['regime']} |"
        )
    md.append("")
    md.append("## Per-model verdict\n")
    for r in rows:
        md.append(f"- **{r['model_short']}** \u2014 {r['verdict']}")
    md.append("")
    md.append("## Histograms (10 bins)\n")
    for r in rows:
        md.append(f"### {r['model_short']}  range [{r['min']:+.3f}, {r['max']:+.3f}]")
        edges = r["histogram_bin_edges"]
        counts = r["histogram_counts"]
        cmax = max(counts) if counts else 1
        md.append("```")
        for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
            bar = "#" * int(50 * c / cmax)
            md.append(f"  [{lo:+7.3f},{hi:+7.3f}]  {c:4d}  {bar}")
        md.append("```")
        md.append("")
    with open(os.path.join(OUT_DIR, "README.md"), "w") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    main()
