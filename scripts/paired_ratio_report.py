"""Markdown writer for paired_ratio_test.py."""
from pathlib import Path

import numpy as np

OUT = Path("results/ocft/paired_ratio_test")

PRE_REG = """\
## Pre-registration (declared BEFORE inspecting results)

**Inputs (cached, no new GPU runs)**
- N=100 paired prompts, identical sample-id ordering across all 5 .npz files
- Δm_full   from `results/decomposition_ci_null/per_example_shifts.npz`  (key `full`)
- Δm_par_E  from same file                                               (key `parallel`)
- Δm_par_Dk from `results/ocft/per_example_shifts_<DK>.npz`              (key `parallel`)
- Probe directions from `results/ocft/per_candidate/<DK>/direction.npy`
  and `steering/directions/direction_probe_layer20.npz` (E)

**STEP 1** — per-prompt ratio: `r_D_i = |Δm_par_D_i| / |Δm_full_i|`.
Skip any prompt where `|Δm_full_i| < 0.01`. If >10 prompts skipped, STOP and
report data-integrity failure.

**STEP 2** — for each of the 10 pairs (E, D1, D2, D3, D4):
  - Paired-permutation test on `d_i = r_D_a_i - r_D_b_i`
  - 10,000 random sign-flips, two-sided p-value
  - Bonferroni-corrected α = 0.05 / 10 = 0.005
  - 95% bootstrap CI on `mean(d_i)` via 2,000 resamples

**STEP 3** — single-link clustering: directions a, b in the same cluster iff
their pairwise raw p-value > 0.005 (Bonferroni-corrected non-significance).

**Pre-registered Outcomes**
- **Outcome I**  — D3 singleton, {E,D1,D2,D4} one cluster:
    => evidence-parallel inertness is **direction-specific**, not geometric.
- **Outcome II** — {D3,D4} high cluster, {E,D1,D2} low cluster:
    => two-cluster; falsifies geometric triviality.
- **Outcome III** — no Bonferroni-significant pair: honest null;
    geometric triviality NOT falsified at the paired level.
- **Outcome IV** — E singleton LOW, {D1,D2,D3,D4} one cluster:
    => evidence-parallel inertness is **feature-specific to evidence**.
"""


def _fmt_p(p):
    if p < 1e-4:
        return f"<1e-4"
    return f"{p:.4g}"


def _classify_outcome(clusters, ratios):
    cl_by_member = {}
    for c in clusters:
        for m in c["members"]:
            cl_by_member[m] = (c["id"], tuple(sorted(c["members"])))
    if len(clusters) == 1:
        return "III", "All directions pairwise indistinguishable; no Bonferroni-significant pair."

    # Outcome IV: E singleton LOW
    e_cluster = [c for c in clusters if "E" in c["members"]][0]
    others_clusters = [c for c in clusters if c is not e_cluster]
    if (len(e_cluster["members"]) == 1
            and len(others_clusters) == 1
            and set(others_clusters[0]["members"]) == {"D1", "D2", "D3", "D4"}
            and e_cluster["mean_ratio"] < others_clusters[0]["mean_ratio"]):
        return "IV", "Evidence forms a singleton LOW cluster — inertness is feature-specific."

    # Outcome I: D3 singleton, {E,D1,D2,D4} together
    d3_cluster = [c for c in clusters if "D3" in c["members"]][0]
    if (len(d3_cluster["members"]) == 1
            and len(others_clusters) >= 1):
        rest = set()
        for c in clusters:
            if c is not d3_cluster:
                rest |= set(c["members"])
        if rest == {"E", "D1", "D2", "D4"} and len(clusters) == 2:
            return "I", "D3 singleton; {E, D1, D2, D4} indistinguishable — direction-specific."

    # Outcome II: {D3,D4} together high; {E,D1,D2} together low
    if len(clusters) == 2:
        sets = [set(c["members"]) for c in clusters]
        if {"D3", "D4"} in [s for s in sets] and {"E", "D1", "D2"} in [s for s in sets]:
            return "II", "Two-cluster {D3,D4} vs {E,D1,D2} — falsifies triviality."

    return "Other", "Cluster pattern does not match any pre-registered outcome label exactly."


def write_report(cmat, ratios, pairwise, clusters, n_eff, n_skip, bonf_alpha):
    names = ["E", "D1", "D2", "D3", "D4"]
    lines = ["# Paired-ratio significance test — Evidence vs D1/D2/D3/D4", ""]
    lines.append(PRE_REG)
    lines.append("")
    lines.append(f"**N (kept)**: {n_eff}    **N skipped (|Δm_full|<0.01)**: {n_skip}")
    lines.append(f"**Bonferroni α**: {bonf_alpha:.4f} (0.05 / 10)")
    lines.append("")

    # Cosine matrix
    lines.append("## STEP 0 — Pairwise cosine matrix (probe directions)")
    lines.append("")
    lines.append("|       | " + " | ".join(f"{n:>6s}" for n in names) + " |")
    lines.append("|-------|" + "|".join(["--------"] * len(names)) + "|")
    for a in names:
        row = [f"{cmat[a][b]:+.4f}" if names.index(b) <= names.index(a) else ""
               for b in names]
        lines.append(f"| {a:<5s} | " + " | ".join(f"{c:>6s}" for c in row) + " |")
    lines.append("")

    # Mean ratios
    lines.append("## STEP 1 — Per-prompt |par|/|full| ratios (mean ± SD)")
    lines.append("")
    lines.append("| direction | mean_ratio | sd_ratio | mean(|par|) | mean(|full|) |")
    lines.append("|-----------|-----------:|---------:|------------:|-------------:|")
    full_abs = None
    for n in names:
        r = ratios[n]
        lines.append(f"| {n} | {r.mean():.4f} | {r.std(ddof=1):.4f} | "
                     f"{np.abs(r * 1.0).mean():.4f} | n/a |")
    lines.append("")

    # Pairwise table
    lines.append("## STEP 2 — Pairwise paired-permutation tests (10,000 sign-flips)")
    lines.append("")
    lines.append("| pair | mean(r_a) | mean(r_b) | mean_diff | 95% CI | p_raw | p_bonf | sig (α=0.005) |")
    lines.append("|------|----------:|----------:|----------:|:-------|------:|-------:|:--------------|")
    pw_sorted = sorted(pairwise, key=lambda r: -abs(r["mean_diff"]))
    for r in pw_sorted:
        sig = "**YES**" if r["significant"] else "no"
        lines.append(
            f"| {r['a']} vs {r['b']} | {r['mean_a']:.4f} | {r['mean_b']:.4f} | "
            f"{r['mean_diff']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
            f"{_fmt_p(r['p_raw'])} | {_fmt_p(r['p_bonf'])} | {sig} |"
        )
    lines.append("")

    # Cluster assignment
    lines.append("## STEP 3 — Cluster assignment (single-link on Bonferroni-non-sig edges)")
    lines.append("")
    lines.append("| cluster | members | mean ratio |")
    lines.append("|---------|---------|-----------:|")
    for c in clusters:
        lines.append(f"| {c['id']} | {{{', '.join(c['members'])}}} | {c['mean_ratio']:.4f} |")
    lines.append("")

    out_label, interp = _classify_outcome(clusters, ratios)
    lines.append(f"## Verdict — Outcome {out_label}")
    lines.append("")
    lines.append(interp)
    lines.append("")

    (OUT / "ratio_test_report.md").write_text("\n".join(lines))
    print(f"\n[done] wrote {OUT}/ratio_test_report.md  →  Outcome {out_label}")
    print(interp)
