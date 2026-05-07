#!/usr/bin/env python3
"""Analysis + figure-JSON writer for dose_response_erasure."""
import json
from pathlib import Path
import numpy as np

ALPHAS = [0.00, 0.25, 0.50, 0.75, 1.00, 2.00]
N_BOOT = 2000
SEED = 20260502
COLORS = {"A": "#d62728", "E": "#1f77b4",
          "D3": "#ff7f0e", "D1": "#2ca02c"}
LABELS = {"A": "Action direction", "E": "Evidence direction",
          "D3": "D3 (candidate-presence-derived)",
          "D1": "D1 (source identity)"}


def boot_mean_ci(x, B=N_BOOT, level=95.0, seed=SEED):
    if len(x) == 0 or float(np.abs(x).max()) == 0:
        return float(x.mean()) if len(x) else 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(x.mean()), float(lo), float(hi)


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    if abs(r) >= 1.0 - 1e-12:
        return r, 0.0
    t = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
    # two-sided p via normal approx; conservative for n=4 but fine for note
    from math import erfc, sqrt
    p = float(erfc(abs(t) / sqrt(2)))
    return r, p


def analyse_and_write(cached, new_margins, base, dirs, H,
                      sample_ids, out_test: Path, out: Path):
    out_test = Path(out_test); out = Path(out)
    n = len(sample_ids)

    # All-α margins per direction
    margins = {nm: {0.0: base.copy()} for nm in ["A", "E", "D3", "D1"]}
    for nm in margins:
        for a in (0.25, 0.50, 0.75):
            margins[nm][a] = new_margins[(nm, a)]
        margins[nm][1.0] = cached[(nm, 1.0)]
        margins[nm][2.0] = cached[(nm, 2.0)]

    # Per-direction stats per α
    base_pos = (base > 0).astype(np.int8)
    curves = {}
    for nm in ["A", "E", "D3", "D1"]:
        dm_mean, dm_lo, dm_hi, flip = [], [], [], []
        for a in ALPHAS:
            d = margins[nm][a] - base
            m, lo, hi = boot_mean_ci(d)
            dm_mean.append(m); dm_lo.append(lo); dm_hi.append(hi)
            sign_after = (margins[nm][a] > 0).astype(np.int8)
            flip.append(float((sign_after != base_pos).mean()))
        curves[nm] = {
            "name": LABELS[nm], "color": COLORS[nm],
            "dm_mean": dm_mean, "dm_ci_lo": dm_lo, "dm_ci_hi": dm_hi,
            "flip_rate": flip,
        }

    # Random band (cached α=1,2 from random_control)
    rc = np.load(out_test / "random_control" / "new_margins.npz")
    K = sum(1 for k in rc.files if k.startswith("r_") and k.endswith("_erase"))
    rand_e = np.array([float(np.abs((rc[f"r_{k+1:02d}_erase"] - base).mean())) for k in range(K)])
    rand_f = np.array([float(np.abs((rc[f"r_{k+1:02d}_flip"]  - base).mean())) for k in range(K)])
    rand_e_mean = float(rand_e.mean()); rand_f_mean = float(rand_f.mean())
    rand_e_p5, rand_e_p95 = float(np.percentile(rand_e, 5)), float(np.percentile(rand_e, 95))
    rand_f_p5, rand_f_p95 = float(np.percentile(rand_f, 5)), float(np.percentile(rand_f, 95))

    def interp(end_at_1, end_at_2):
        # Linear at α: 0->0, 0.25, 0.50, 0.75, 1.00->end_at_1, 2.00->end_at_2
        return [0.0,
                0.25 * end_at_1, 0.50 * end_at_1, 0.75 * end_at_1,
                end_at_1, end_at_2]

    rand_band = {
        "method": "K=20 cached at α∈{1.0, 2.0}; linear interpolation at α∈{0.25, 0.50, 0.75}",
        "dm_mean":  interp(rand_e_mean, rand_f_mean),
        "dm_p5":    interp(rand_e_p5,   rand_f_p5),
        "dm_p95":   interp(rand_e_p95,  rand_f_p95),
    }
    fig_dr = {"alpha_values": ALPHAS, "curves": curves, "random_band": rand_band}
    (out_test / "figure_dose_response.json").write_text(json.dumps(fig_dr, indent=2))

    # Projection vs effect (per prompt at α=2 = flip)
    panels = {}
    for nm in ["A", "E", "D3", "D1"]:
        D = dirs[nm]
        proj = (H @ D).astype(np.float64)  # signed natural projection
        dm_flip = (margins[nm][2.0] - base).astype(np.float64)
        r, p = pearson(proj, dm_flip)
        panels[nm] = {
            "pearson_r_flip": r, "pearson_p_flip": p,
            "points": [{"projection": float(proj[i]),
                        "dm_flip":   float(dm_flip[i])} for i in range(len(proj))],
        }
    (out_test / "figure_projection_vs_effect.json").write_text(
        json.dumps({"panels": panels}, indent=2))

    # cos vs effect (named + random)
    spectrum = json.loads((out_test / "figure_spectrum.json").read_text())
    pts = []
    for d in spectrum["directions"]:
        pts.append({"name": d["name"],
                    "cos_abs": float(abs(d["cos_with_A"])),
                    "dm_flip": float(d["dm_flip"]),
                    "type": d["type"]})
    cos_all = np.array([p["cos_abs"] for p in pts])
    eff_all = np.array([p["dm_flip"] for p in pts])
    r_all, p_all = pearson(cos_all, eff_all)
    mask = np.array([p["name"] != "A" for p in pts])
    r_nA, p_nA = pearson(cos_all[mask], eff_all[mask])
    fig_cos = {"points": pts,
               "pearson_r_all_26": r_all, "pearson_p_all_26": p_all,
               "pearson_r_excluding_A": r_nA, "pearson_p_excluding_A": p_nA}
    (out_test / "figure_cos_vs_effect.json").write_text(json.dumps(fig_cos, indent=2))

    # Per-α JSON results
    per_alpha = {nm: {f"alpha={a:.2f}": {
        "dm_mean": curves[nm]["dm_mean"][i],
        "dm_ci": [curves[nm]["dm_ci_lo"][i], curves[nm]["dm_ci_hi"][i]],
        "flip_rate": curves[nm]["flip_rate"][i],
        "source": "cached" if a in (0.0, 1.0, 2.0) else "new",
    } for i, a in enumerate(ALPHAS)} for nm in ["A", "E", "D3", "D1"]}
    (out / "per_alpha_results.json").write_text(json.dumps({
        "per_direction": per_alpha,
        "random_band_alpha_1": {"mean": rand_e_mean, "p5": rand_e_p5, "p95": rand_e_p95},
        "random_band_alpha_2": {"mean": rand_f_mean, "p5": rand_f_p5, "p95": rand_f_p95},
        "n": n, "K_random": int(K),
    }, indent=2))

    write_report(out, curves, panels, fig_cos, n, ALPHAS,
                 rand_e_mean, rand_f_mean, rand_e_p95, rand_f_p95)
    print_table(curves, panels, fig_cos)


def print_table(curves, panels, fig_cos):
    print("\n| α    | Δm_A          | Δm_E          | Δm_D3         | Δm_D1         |")
    print(  "|------|---------------|---------------|---------------|---------------|")
    for i, a in enumerate(ALPHAS):
        tag = "(cached)" if a in (0.0, 1.0, 2.0) else "(new)"
        row = [f"{a:.2f}"]
        for nm in ["A", "E", "D3", "D1"]:
            v = curves[nm]["dm_mean"][i]
            row.append(f"{v:+.4f} {tag}")
        print("| " + " | ".join(row) + " |")
    print("\n[pearson] projection-vs-effect (per direction, n=100):")
    for nm in ["A", "E", "D3", "D1"]:
        print(f"  {nm}: r={panels[nm]['pearson_r_flip']:+.3f}  "
              f"p≈{panels[nm]['pearson_p_flip']:.3g}")
    print(f"\n[pearson] cos-vs-effect, all 26: r={fig_cos['pearson_r_all_26']:+.3f}  "
          f"p≈{fig_cos['pearson_p_all_26']:.3g}")
    print(f"[pearson] cos-vs-effect, excl A (25): "
          f"r={fig_cos['pearson_r_excluding_A']:+.3f}  "
          f"p≈{fig_cos['pearson_p_excluding_A']:.3g}")


def write_report(out, curves, panels, fig_cos, n, alphas,
                 rand_e_mean, rand_f_mean, rand_e_p95, rand_f_p95):
    lines = [
        "# Gradient Erasure Dose-Response — A, E, D3, D1 at L20",
        "",
        "## Pre-registration / provenance",
        "",
        "- Model: Qwen2.5-7B-Instruct, L20 last token, same N=100 §3 prompts.",
        "- Intervention: h ← h − α · (h·D̂) · D̂ (natural scale, no RMS).",
        "- α ∈ {0.00, 0.25, 0.50, 0.75, 1.00, 2.00}.",
        "  - α=0.00, 1.00, 2.00 for A,E reused from per_prompt_margins.npz (cached).",
        "  - α=1.00, 2.00 for D3,D1 reused from random_control/new_margins.npz (cached).",
        "  - α=0.25, 0.50, 0.75 for all four newly run (1,200 forwards).",
        "- Random band: K=20 random unit dirs (seed=42), |cos·{A,E}|<0.05; |Δm| computed",
        "  at α=1 and α=2 from cache and **linearly interpolated** to intermediate α.",
        f"  (random α=1: mean |Δm|={rand_e_mean:.4f}, p95={rand_e_p95:.4f}; "
        f"random α=2: mean |Δm|={rand_f_mean:.4f}, p95={rand_f_p95:.4f}.)",
        "- Pipeline check: baseline recapture margin must equal cached baseline; reported",
        "  in `per_prompt_margins_alpha_new.npz` (max abs diff printed at run time).",
        "",
        f"## Dose-response (N={n})",
        "",
        "| α | Δm_A | Δm_E | Δm_D3 | Δm_D1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for i, a in enumerate(alphas):
        src = "cached" if a in (0.0, 1.0, 2.0) else "new"
        row = [f"{a:.2f}"] + [
            f"{curves[nm]['dm_mean'][i]:+.4f} ({src})" for nm in ["A","E","D3","D1"]
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Behavioral flip rate (sign(margin) change vs baseline)", "",
              "| α | A | E | D3 | D1 |", "|---|---:|---:|---:|---:|"]
    for i, a in enumerate(alphas):
        row = [f"{a:.2f}"] + [f"{curves[nm]['flip_rate'][i]:.2%}"
                              for nm in ["A","E","D3","D1"]]
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Projection-vs-effect Pearson r (per direction, α=2 flip; n=100)", ""]
    for nm in ["A", "E", "D3", "D1"]:
        lines.append(f"- {nm}: r={panels[nm]['pearson_r_flip']:+.3f}, "
                     f"p≈{panels[nm]['pearson_p_flip']:.3g}")
    lines += ["", "## Cosine-vs-effect (geometry attack falsification)", "",
              f"- All 26 directions: r={fig_cos['pearson_r_all_26']:+.3f}, "
              f"p≈{fig_cos['pearson_p_all_26']:.3g}",
              f"- Excluding A (25 dirs): r={fig_cos['pearson_r_excluding_A']:+.3f}, "
              f"p≈{fig_cos['pearson_p_excluding_A']:.3g}", ""]
    (out / "dose_response_report.md").write_text("\n".join(lines) + "\n")
    print(f"[save] {out}/dose_response_report.md")
