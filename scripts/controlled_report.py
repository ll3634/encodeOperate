#!/usr/bin/env python3
"""Render controlled_report.md with pre-reg, spectrum, rotation, verdict."""
from pathlib import Path
import numpy as np


PRE_REG = """## Pre-registered outcomes (declared before any GPU run)

Intervention:
    s_i = sign(h_i . D_hat) ; if zero, treat as +1.
    h'_i = h_i - c * s_i * D_hat
The amount of energy removed along D_hat equals exactly c units, regardless
of |h_i . D_hat|.  This isolates readout gain from representation strength.

Magnitudes (computed from cached H, then locked):
    c_E = mean_i |h_i . E_hat|
    c_A = mean_i |h_i . A_hat|
At c = c_E, controlled removal along E ~= natural erasure along E; controlled
removal along A removes much LESS than natural erasure along A.

Outcome decisions:
- SELECTIVE : at c = c_E,
              |dm_controlled_A| / |dm_controlled_E| >= 5.0
              AND |dm_controlled_A| > 0.05
              AND |dm_controlled_E| ~ mean(|dm_controlled_random|).
              => Readout gain is direction-selective at matched magnitude.
- AGNOSTIC  : at c = c_E,
              |dm_controlled_A| / |dm_controlled_E| < 2.0.
              => Readout is direction-agnostic at small magnitudes.
- MIXED     : ratio < 5.0 at c_E but >= 5.0 at c_A.
              => Selectivity is a large-magnitude phenomenon.

Rotation rescue (E->D3 at fixed c):
- If |dm_controlled| is monotonic up to a peak, the projection-magnitude
  confound from the unmatched-magnitude rotation is removed by construction
  and null-space position drives the effect directly.
- If flat, position alone (at fixed energy) is not the explanatory factor.
"""


def write_report(out_dir: Path, figure: dict, magnitudes: dict,
                 proj_dir: dict, proj_rot: dict) -> None:
    L = []
    L += ["# Controlled-Magnitude Erasure", "",
          "Matched-energy direction-selectivity test "
          "(complement to natural-scale erasure)."]
    L += [PRE_REG]

    c_E = magnitudes["c_E"]; c_A = magnitudes["c_A"]
    L += ["", "## Magnitudes used", "",
          f"- c_E = mean|h.E_hat| = **{c_E:.6f}**",
          f"- c_A = mean|h.A_hat| = **{c_A:.6f}**",
          f"- ratio c_A / c_E    = **{c_A / max(c_E, 1e-12):.2f}x**",
          ""]

    # Spectrum table
    L += ["", "## Direction spectrum (16 directions x 2 magnitudes)", "",
          "| direction | type | mean\\|h.D\\| | \\|dm\\| @ c_E (CI) | flip @ c_E | "
          "\\|dm\\| @ c_A (CI) | flip @ c_A |",
          "|---|---|---:|---|---:|---|---:|"]
    for rec in figure["directions"]:
        de = (f"{rec['abs_dm_at_c_E']:.4f} "
              f"[{rec['abs_dm_at_c_E_ci'][0]:.4f}, {rec['abs_dm_at_c_E_ci'][1]:.4f}]")
        da = (f"{rec['abs_dm_at_c_A']:.4f} "
              f"[{rec['abs_dm_at_c_A_ci'][0]:.4f}, {rec['abs_dm_at_c_A_ci'][1]:.4f}]")
        L.append(f"| {rec['name']} | {rec['type']} | {rec['mean_proj_magnitude']:.4f} | "
                 f"{de} | {rec['flip_rate_c_E']:.3f} | {da} | {rec['flip_rate_c_A']:.3f} |")

    # Selectivity ratios + random-band band
    A_rec = next(r for r in figure["directions"] if r["name"] == "A")
    E_rec = next(r for r in figure["directions"] if r["name"] == "E")
    rand = [r for r in figure["directions"] if r["type"] == "random"]
    rand_cE = np.array([r["abs_dm_at_c_E"] for r in rand])
    rand_cA = np.array([r["abs_dm_at_c_A"] for r in rand])
    p5_cE, p95_cE = float(np.percentile(rand_cE, 5)), float(np.percentile(rand_cE, 95))
    p5_cA, p95_cA = float(np.percentile(rand_cA, 5)), float(np.percentile(rand_cA, 95))
    ratio_AE_cE = A_rec["abs_dm_at_c_E"] / max(E_rec["abs_dm_at_c_E"], 1e-12)
    ratio_AE_cA = A_rec["abs_dm_at_c_A"] / max(E_rec["abs_dm_at_c_A"], 1e-12)

    L += ["", "## Direction-selectivity ratios", "",
          "| metric | c_E | c_A |",
          "|---|---:|---:|",
          f"| \\|dm_A\\| | {A_rec['abs_dm_at_c_E']:.4f} | {A_rec['abs_dm_at_c_A']:.4f} |",
          f"| \\|dm_E\\| | {E_rec['abs_dm_at_c_E']:.4f} | {E_rec['abs_dm_at_c_A']:.4f} |",
          f"| **\\|dm_A\\| / \\|dm_E\\|** | **{ratio_AE_cE:.2f}x** | **{ratio_AE_cA:.2f}x** |",
          f"| random p5..p95 ({len(rand)} dirs) | [{p5_cE:.4f}, {p95_cE:.4f}] | "
          f"[{p5_cA:.4f}, {p95_cA:.4f}] |",
          f"| E in random p5..p95? | "
          f"{'YES' if p5_cE <= E_rec['abs_dm_at_c_E'] <= p95_cE else 'NO'} | "
          f"{'YES' if p5_cA <= E_rec['abs_dm_at_c_A'] <= p95_cA else 'NO'} |",
          f"| A above random p95?  | {'YES' if A_rec['abs_dm_at_c_E'] > p95_cE else 'NO'} | "
          f"{'YES' if A_rec['abs_dm_at_c_A'] > p95_cA else 'NO'} |"]

    # Verdict
    sel_cE = (ratio_AE_cE >= 5.0 and A_rec["abs_dm_at_c_E"] > 0.05
              and p5_cE <= E_rec["abs_dm_at_c_E"] <= p95_cE)
    agn_cE = ratio_AE_cE < 2.0
    sel_cA = ratio_AE_cA >= 5.0
    if sel_cE:
        verdict = "**OUTCOME SELECTIVE** -- readout gain selectivity confirmed at matched magnitude c_E."
    elif sel_cA and not sel_cE:
        verdict = "**OUTCOME MIXED** -- selectivity emerges only at large magnitude (c_A)."
    elif agn_cE:
        verdict = "**OUTCOME AGNOSTIC** -- readout is direction-agnostic at matched magnitude."
    else:
        verdict = (f"**OUTCOME PARTIAL** -- ratio {ratio_AE_cE:.2f}x at c_E "
                   f"(threshold 5.0); see numbers above.")
    L += ["", "## Verdict", "", verdict, ""]

    # Rotation curves
    L += ["", "## Null-space rotation at fixed c (E->D3 path)", "",
          "Same construction as previous experiment, but at fixed energy removal."]
    for c_label in ("c_E", "c_A"):
        block = figure[f"nullspace_rotation_at_{c_label}"]
        c_val = block["constant_c"]
        L += ["", f"### Rotation at c = {c_val:.4f} ({c_label})", "",
              "| theta | proj | \\|dm_controlled\\| (CI) |",
              "|---:|---:|---|"]
        for ang, dm, ci in zip(block["angles_deg"], block["dm_controlled"],
                                block["dm_controlled_ci"]):
            key = f"E_to_D3__theta{ang:02d}"
            proj = proj_rot.get(key, float("nan"))
            L.append(f"| {ang:>3d} | {proj:.4f} | {dm:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] |")
        dms = block["dm_controlled"]
        peak = max(range(len(dms)), key=lambda i: dms[i])
        peak_ang = block["angles_deg"][peak]
        peak_val = dms[peak]
        base_val = dms[0]
        ratio = peak_val / max(base_val, 1e-12)
        monotonic_to_peak = all(dms[i] <= dms[i + 1] + 1e-6 for i in range(peak))
        L += ["",
              f"- peak |dm_controlled| = {peak_val:.4f} at theta={peak_ang} deg "
              f"(theta=0: {base_val:.4f}; ratio {ratio:.2f}x)",
              f"- monotonic 0..peak: {monotonic_to_peak}"]
        if ratio >= 1.5 and peak_val > 0.02:
            L.append("- **RESCUE**: at fixed c, rotation alone increases the effect; "
                     "null-space position drives margin shift.")
        else:
            L.append("- rotation curve does NOT show clear ascent at fixed c; null-"
                     "space position alone (at matched energy) is not the explanatory factor.")

    (out_dir / "controlled_report.md").write_text("\n".join(L) + "\n")
    print(f"[saved] {out_dir / 'controlled_report.md'}")
