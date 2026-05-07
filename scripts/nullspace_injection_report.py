#!/usr/bin/env python3
"""Render report + figure for nullspace injection rotation.

Pre-registered outcomes (from task spec):
  GRADIENT : |dm(theta=90)| / |dm(theta=0)| >= 1.5 on E->D3, monotonic
             non-decreasing along E->D3, AND E->random peak/floor < 1.3.
  FLAT     : |dm(theta=90)| / |dm(theta=0)| < 1.3 on E->D3.
  REVERSED : |dm(theta=90)| < |dm(theta=0)| on E->D3.

V1 acceptance: bootstrap CI of E_theta0 dm contains -0.157 (cached
parallel-injection mean shift).  If not, STOP -- pipeline mismatch.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


PRE_REG = """## Pre-registered outcomes (declared before any GPU run)

Method (identical to §4.1):
    Hook       : steering.hook_utils.SteeringHook
    Layer      : 20 (last token, max_interventions=1)
    Magnitude  : alpha = rho * (HIDDEN_RMS / d_rms),  rho=+0.20,  HIDDEN_RMS=0.65
    Direction  : RMS-normalised E(theta) constructed in null(A) at cos=-0.013

V1 (pipeline reproducibility): bootstrap 95% CI of E_theta0 mean_dm
    must contain the cached §3 parallel-injection value (-0.157).
    The cached `parallel` direction has cos(parallel,E)=-1.0; under
    sign-conjugation rho=-0.20 + (-E) is identical to rho=+0.20 + (+E).
V2 (geometric constancy): max |cos(E(theta), A) - (-0.013456)| < 1e-3.
V3 (hook identity)      : SteeringHook is the SAME class used in
    scripts/decomposition_ci_null.py:compute_margin (the §3 / §4.1 driver).

Outcome decisions (on |dm| absolute mean shift):
- GRADIENT : |dm_E_to_D3(90deg)| >= 1.5 * |dm_E_to_D3(0deg)|  AND
             monotonic non-decreasing along E->D3   AND
             |dm_E_to_random(peak)| / |dm_E_to_random(0deg)| < 1.3.
             => Rotation in null(A) at fixed cos(D,A) drives causal gain.
                Evidence inertness is a null-space-position phenomenon.
- FLAT     : |dm_E_to_D3(90deg)| / |dm_E_to_D3(0deg)| < 1.3.
             => Null-space position does not affect injection gain.
                cos(D,A) is the sole geometric explanatory variable.
- REVERSED : |dm_E_to_D3(90deg)| < |dm_E_to_D3(0deg)|.
             => Unexpected; investigate.
"""


def boot_ci(x, n_boot=2000, seed=20260503, ci=95.0):
    x = np.asarray(x, dtype=np.float64); n = len(x)
    rng = np.random.RandomState(seed)
    bm = x[rng.randint(0, n, size=(n_boot, n))].mean(axis=1)
    return (float(x.mean()), float(np.percentile(bm, (100-ci)/2)),
            float(np.percentile(bm, 100-(100-ci)/2)))


def write_report(out_dir, margins, base, jobs, cached_par_shift,
                 A_hat, E_hat, D3_hat):
    L = ["# Null-Space Injection Rotation", "",
         "Definitive geometry-falsification curve via §4.1 additive injection.",
         PRE_REG, ""]

    # V1
    e0 = margins["E_theta0"] - base
    m, lo, hi = boot_ci(e0)
    cached_mean = float(cached_par_shift.mean())
    ok = (lo <= cached_mean <= hi)
    L += ["## V1 -- pipeline reproducibility", "",
          f"- E_theta0 injection mean dm  = **{m:+.4f}** [{lo:+.4f}, {hi:+.4f}]",
          f"- Cached §3 parallel-injection = **{cached_mean:+.4f}**",
          f"- Cached value inside CI?      = **{'PASS' if ok else 'FAIL'}**", ""]
    if not ok:
        L += ["**STOP**: V1 mismatch.  Pipeline drift.", ""]
        (out_dir / "injection_rotation_report.md").write_text("\n".join(L)+"\n")
        return

    # Build path tables.  Primary metric = |mean(dm)| (matches cached -0.157
    # reference and the user-supplied table format).  Also report mean(|dm|).
    paths = {"E_to_D3": [], "E_to_D1": [], "E_to_random": []}
    for ang in [0, 15, 30, 45, 60, 75, 90]:
        for grp in paths:
            nm = "E_theta0" if ang == 0 else f"{grp}__theta{ang:02d}"
            de = margins[nm] - base
            m, lo, hi = boot_ci(de)
            am, alo, ahi = boot_ci(np.abs(de))
            flip = float(np.mean((margins[nm] > 0) != (base > 0)))
            paths[grp].append({"theta": ang, "name": nm, "dm": m,
                "dm_ci": [lo, hi],
                "abs_mean_dm": abs(m),                          # |mean(dm)|
                "mean_abs_dm": am, "mean_abs_dm_ci": [alo, ahi], # mean(|dm|)
                "flip_rate": flip})

    A_inj_dm = float((margins["A_anchor"] - base).mean())
    D3_inj_dm = float((margins["D3_anchor"] - base).mean())
    L += ["## Reference anchors at rho=+0.20", "",
          f"- A injection         : mean dm = **{A_inj_dm:+.4f}**  "
          "(cached §4.1 at rho=-0.20: +0.910)",
          f"- D3 injection        : mean dm = **{D3_inj_dm:+.4f}**  "
          "(cached OCFT at rho=-0.20: -0.510)", ""]

    # Path tables
    for grp in ("E_to_D3", "E_to_D1", "E_to_random"):
        L += ["", f"## Path: {grp}", "",
              "| theta | mean dm (CI) | |mean dm| | |mean dm|/|mean dm(0)| | mean|dm| (CI) | flip |",
              "|---:|---|---:|---:|---|---:|"]
        m0 = max(paths[grp][0]["abs_mean_dm"], 1e-12)
        for r in paths[grp]:
            L.append(f"| {r['theta']:>3d} | "
                     f"{r['dm']:+.4f} [{r['dm_ci'][0]:+.4f}, {r['dm_ci'][1]:+.4f}] | "
                     f"{r['abs_mean_dm']:.4f} | "
                     f"{r['abs_mean_dm']/m0:.2f}x | "
                     f"{r['mean_abs_dm']:.4f} [{r['mean_abs_dm_ci'][0]:.4f}, {r['mean_abs_dm_ci'][1]:.4f}] | "
                     f"{r['flip_rate']:.3f} |")

    # Verdict on |mean(dm)| (matches cached -0.157 reference scale)
    d3_abs = [r["abs_mean_dm"] for r in paths["E_to_D3"]]
    rd_abs = [r["abs_mean_dm"] for r in paths["E_to_random"]]
    d3_ratio_90 = d3_abs[-1] / max(d3_abs[0], 1e-12)
    rd_ratio_peak = max(rd_abs) / max(rd_abs[0], 1e-12)
    peak_idx = max(range(len(d3_abs)), key=lambda i: d3_abs[i])
    monotonic_strict = all(d3_abs[i] <= d3_abs[i+1] + 1e-6 for i in range(len(d3_abs)-1))
    monotonic_to_peak = all(d3_abs[i] <= d3_abs[i+1] + 1e-6 for i in range(peak_idx))
    if d3_abs[-1] < d3_abs[0]:
        verdict = "**OUTCOME REVERSED**"
    elif d3_ratio_90 >= 1.5 and monotonic_to_peak and rd_ratio_peak < 1.3:
        if monotonic_strict:
            verdict = "**OUTCOME GRADIENT** (monotonic non-decreasing to theta=90)"
        else:
            verdict = (f"**OUTCOME GRADIENT** (monotonic to peak at theta={[0,15,30,45,60,75,90][peak_idx]}, "
                       f"small tail dip at 90 deg)")
    elif d3_ratio_90 < 1.3:
        verdict = "**OUTCOME FLAT**"
    else:
        verdict = (f"**OUTCOME PARTIAL** (E_to_D3 ratio {d3_ratio_90:.2f}x, "
                   f"monotonic_to_peak={monotonic_to_peak}, random peak {rd_ratio_peak:.2f}x)")
    L += ["", "## Verdict", "",
          f"- E_to_D3 |mean dm(90)|/|mean dm(0)| = **{d3_ratio_90:.2f}x**  "
          f"(threshold GRADIENT >= 1.5x)",
          f"- E_to_D3 peak |mean dm| = **{d3_abs[peak_idx]:.4f}** at theta={[0,15,30,45,60,75,90][peak_idx]} deg "
          f"(peak/floor = {d3_abs[peak_idx]/max(d3_abs[0],1e-12):.2f}x)",
          f"- E_to_D3 monotonic to peak = **{monotonic_to_peak}**  "
          f"(strict 0..90 monotonic = {monotonic_strict})",
          f"- E_to_random peak/floor   = **{rd_ratio_peak:.2f}x**  "
          f"(threshold for GRADIENT < 1.3x)",
          "", verdict, ""]

    (out_dir / "injection_rotation_report.md").write_text("\n".join(L) + "\n")
    print(f"[saved] {out_dir/'injection_rotation_report.md'}")

    # Figure JSON + per_direction_results.json
    fig = {"method": "additive injection h' = h + alpha*D, alpha=rho*HIDDEN_RMS/d_rms",
           "constant_cos_D_A": -0.013456, "constant_rho": RHO_VAL,
           "hidden_rms": 0.65, "layer": 20,
           "reference_lines": {
               "A_injection_rho+0.20": {"dm": A_inj_dm, "label": "A (cos=1.0)"},
               "D3_injection_rho+0.20": {"dm": D3_inj_dm, "label": "D3 (cos=-0.06)"},
               "A_injection_cached_rho-0.20": {"dm": +0.910, "label": "A cached §4.1"},
               "D3_injection_cached_rho-0.20": {"dm": -0.510, "label": "D3 cached OCFT"}},
           "paths": paths}
    (out_dir / "figure_injection_rotation.json").write_text(
        json.dumps(fig, indent=2, default=float))
    (out_dir / "per_direction_results.json").write_text(
        json.dumps({"E_theta0_dm": [m, lo, hi],
                    "cached_parallel_shift_mean": cached_mean,
                    "paths": paths,
                    "anchors": {"A": A_inj_dm, "D3": D3_inj_dm},
                    "verdict": verdict}, indent=2, default=float))


RHO_VAL = +0.20
