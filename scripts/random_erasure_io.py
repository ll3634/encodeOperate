#!/usr/bin/env python3
"""Analysis + figure JSON writer for random_erasure_control."""
import json
from pathlib import Path
import numpy as np

N_BOOT = 2000
SEED = 20260502


def boot_abs_mean_ci(x, B=N_BOOT, level=95.0, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    means = np.abs(x[idx].mean(axis=1))
    lo, hi = np.percentile(means, [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(np.abs(x.mean())), float(lo), float(hi)


def signed_mean(x):
    return float(x.mean())


def classify_outcome(absdm_E, absflip_E, rand_abs_e, rand_abs_f):
    p5e, p95e = float(np.percentile(rand_abs_e, 5)), float(np.percentile(rand_abs_e, 95))
    p5f, p95f = float(np.percentile(rand_abs_f, 5)), float(np.percentile(rand_abs_f, 95))
    inside_e = p5e <= absdm_E <= p95e
    inside_f = p5f <= absflip_E <= p95f
    if inside_e and inside_f:
        return "R1", (p5e, p95e, p5f, p95f)
    if absdm_E > p95e or absflip_E > p95f:
        return "R2", (p5e, p95e, p5f, p95f)
    if absdm_E < p5e:
        return "R3", (p5e, p95e, p5f, p95f)
    return "R1", (p5e, p95e, p5f, p95f)


def analyse_and_write(cached, new_margins, named, R, sample_ids,
                      n_resample, out_test: Path, out: Path):
    out_test = Path(out_test); out = Path(out)
    base = cached["baseline"]
    A_hat, E_hat = named["A"], named["E"]

    # Per-direction stats (named + random)
    dir_stats = []

    def add(name, type_, vec, m_erase, m_flip, auroc=None):
        de = m_erase - base
        df = m_flip - base
        am_e, lo_e, hi_e = boot_abs_mean_ci(de)
        am_f, lo_f, hi_f = boot_abs_mean_ci(df)
        dir_stats.append({
            "name": name, "type": type_,
            "cos_with_A": float(vec @ A_hat) if vec is not None else 1.0,
            "cos_with_E": float(vec @ E_hat) if vec is not None else None,
            "auroc": auroc,
            "dm_erase_signed": signed_mean(de),
            "dm_flip_signed":  signed_mean(df),
            "dm_erase": am_e, "dm_erase_ci": [lo_e, hi_e],
            "dm_flip":  am_f, "dm_flip_ci":  [lo_f, hi_f],
        })

    # A and E from cached margins
    add("A", "action", A_hat, cached["erase_A"], cached["flip_A"], auroc=None)
    add("E", "evidence", E_hat, cached["erase_E"], cached["flip_E"], auroc=0.862)
    aurocs = {"D1": 1.00, "D2": 1.00, "D3": 1.00, "D4": 0.99}
    for name in ["D1", "D2", "D3", "D4"]:
        add(name, "ocft_candidate", named[name],
            new_margins[name]["erase"], new_margins[name]["flip"],
            auroc=aurocs.get(name))
    for k in range(R.shape[0]):
        nm = f"r_{k+1:02d}"
        add(nm, "random", R[k],
            new_margins[nm]["erase"], new_margins[nm]["flip"])

    # Random aggregate
    rand_abs_e = np.array([d["dm_erase"] for d in dir_stats if d["type"] == "random"])
    rand_abs_f = np.array([d["dm_flip"]  for d in dir_stats if d["type"] == "random"])
    rand_summary = {
        "K": int(R.shape[0]),
        "abs_dm_erase_mean": float(rand_abs_e.mean()),
        "abs_dm_erase_sd":   float(rand_abs_e.std(ddof=1)),
        "abs_dm_erase_p5_p95": [float(np.percentile(rand_abs_e, 5)),
                                float(np.percentile(rand_abs_e, 95))],
        "abs_dm_flip_mean": float(rand_abs_f.mean()),
        "abs_dm_flip_sd":   float(rand_abs_f.std(ddof=1)),
        "abs_dm_flip_p5_p95": [float(np.percentile(rand_abs_f, 5)),
                               float(np.percentile(rand_abs_f, 95))],
        "n_resample": int(n_resample),
    }

    # E vs random outcome
    E_row = next(d for d in dir_stats if d["name"] == "E")
    A_row = next(d for d in dir_stats if d["name"] == "A")
    outcome, (p5e, p95e, p5f, p95f) = classify_outcome(
        E_row["dm_erase"], E_row["dm_flip"], rand_abs_e, rand_abs_f)
    print(f"\n[outcome] {outcome}")
    print(f"  E erase |Δm|={E_row['dm_erase']:.4f}  random p5..p95=[{p5e:.4f}, {p95e:.4f}]")
    print(f"  E flip  |Δm|={E_row['dm_flip']:.4f}  random p5..p95=[{p5f:.4f}, {p95f:.4f}]")
    print(f"  A erase |Δm|={A_row['dm_erase']:.4f}  A flip |Δm|={A_row['dm_flip']:.4f}")

    with open(out / "per_direction_results.json", "w") as f:
        json.dump({"directions": dir_stats, "random_summary": rand_summary,
                   "outcome": outcome,
                   "config": {"layer": 20, "K": int(R.shape[0]),
                              "model": "Qwen/Qwen2.5-7B-Instruct",
                              "n_boot": N_BOOT}}, f, indent=2)

    # Figure A: scatter erase_A vs erase_E
    fig_a = {"prompts": [
        {"id": str(sid),
         "dm_erase_A": float(cached["erase_A"][i] - base[i]),
         "dm_erase_E": float(cached["erase_E"][i] - base[i]),
         "dm_flip_A":  float(cached["flip_A"][i]  - base[i]),
         "dm_flip_E":  float(cached["flip_E"][i]  - base[i]),
         "baseline_margin": float(base[i])}
        for i, sid in enumerate(sample_ids)]}
    (out_test / "figure_scatter_AvsE.json").write_text(json.dumps(fig_a, indent=2))

    # Figure B: spectrum
    fig_b = {"directions": [
        {k: d[k] for k in ("name","type","cos_with_A","auroc",
                           "dm_erase","dm_flip","dm_erase_ci","dm_flip_ci")}
        for d in dir_stats]}
    (out_test / "figure_spectrum.json").write_text(json.dumps(fig_b, indent=2))

    # Figure C: strip
    erase_r_mean = np.zeros(len(sample_ids), dtype=np.float64)
    K = R.shape[0]
    for k in range(K):
        nm = f"r_{k+1:02d}"
        erase_r_mean += (new_margins[nm]["erase"] - base)
    erase_r_mean /= K
    fig_c = {"conditions": {
        "erase_E":      [float(x) for x in (cached["erase_E"] - base)],
        "flip_E":       [float(x) for x in (cached["flip_E"]  - base)],
        "erase_A":      [float(x) for x in (cached["erase_A"] - base)],
        "flip_A":       [float(x) for x in (cached["flip_A"]  - base)],
        "erase_r_mean": [float(x) for x in erase_r_mean],
    }}
    (out_test / "figure_strip.json").write_text(json.dumps(fig_c, indent=2))

    write_report(out, dir_stats, rand_summary, outcome, A_row, E_row,
                 (p5e, p95e, p5f, p95f), n_resample, len(sample_ids))


def write_report(out, dir_stats, rand_summary, outcome,
                 A_row, E_row, ranges, n_resample, N):
    p5e, p95e, p5f, p95f = ranges
    flipA_abs = next(d["dm_flip"] for d in dir_stats if d["name"] == "A")
    rows = []
    order = ["A", "E", "D3", "D1", "D2", "D4"]
    for nm in order:
        d = next(x for x in dir_stats if x["name"] == nm)
        rows.append(
            f"| {nm} | {d['type']} | {abs(d['cos_with_A']):.4f} "
            f"| {d['dm_erase']:.4f} | {d['dm_flip']:.4f} "
            f"| {d['dm_flip']/flipA_abs:.3f} |")
    rs = rand_summary
    rows.append(
        f"| random (mean±sd) | random | — "
        f"| {rs['abs_dm_erase_mean']:.4f}±{rs['abs_dm_erase_sd']:.4f} "
        f"| {rs['abs_dm_flip_mean']:.4f}±{rs['abs_dm_flip_sd']:.4f} "
        f"| {rs['abs_dm_flip_mean']/flipA_abs:.3f} |")

    md = [
        "# Random-Direction Erasure Control — Evidence Erasure Test",
        "",
        "## Pre-registration",
        "",
        "- Model: Qwen2.5-7B-Instruct, L20 last token, same N=100 §3 prompts.",
        "- E, A: same as Evidence Erasure Test.",
        "- D1..D4: results/ocft/per_candidate/{D1_source,D2_action_prior,",
        "  D3_candidate_present,D4_obs_length}/direction.npy (unit norm).",
        "- K=20 random unit directions in R^3584, seed=42, resampled if",
        "  |cos(r,A)|>0.05 or |cos(r,E)|>0.05.",
        "- Erase: h ← h − (h·ê)ê.  Flip: h ← h − 2(h·ê)ê.  No RMS scaling.",
        "- Margin = logsumexp(logits[Action]) − logsumexp(logits[Final]) at last token.",
        "- Outcomes:",
        "  - R1: |Δm_erase_E| ∈ [p5,p95] of random AND |Δm_flip_E| ∈ [p5,p95] of random",
        "    → E indistinguishable from random; A is the unique operative outlier.",
        "  - R2: |Δm_erase_E| > p95(random) OR |Δm_flip_E| > p95(random) → small effect.",
        "  - R3: |Δm_erase_E| < p5(random) → unexpectedly more inert than random.",
        "",
        f"## Sampling diagnostics: K={rs['K']} random dirs accepted, "
        f"{n_resample} resamples needed (orthogonality threshold 0.05).",
        "",
        f"## Outcome: **{outcome}** (N={N})",
        "",
        f"- Random erase |Δm|: mean={rs['abs_dm_erase_mean']:.4f}, "
        f"sd={rs['abs_dm_erase_sd']:.4f}, [p5,p95]=[{p5e:.4f}, {p95e:.4f}]",
        f"- Random flip  |Δm|: mean={rs['abs_dm_flip_mean']:.4f}, "
        f"sd={rs['abs_dm_flip_sd']:.4f}, [p5,p95]=[{p5f:.4f}, {p95f:.4f}]",
        f"- E erase |Δm|={E_row['dm_erase']:.4f}; E flip |Δm|={E_row['dm_flip']:.4f}",
        f"- A erase |Δm|={A_row['dm_erase']:.4f}; A flip |Δm|={A_row['dm_flip']:.4f}",
        "",
        "| direction | type | |cos(·,A)| | |Δm_erase| | |Δm_flip| | ratio to A_flip |",
        "|---|---|---:|---:|---:|---:|",
    ] + rows
    (out / "random_erasure_report.md").write_text("\n".join(md) + "\n")
    print(f"[save] {out}/random_erasure_report.md")
