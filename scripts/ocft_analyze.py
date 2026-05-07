#!/usr/bin/env python3
"""
OCFT — Stage 3: bootstrap CIs, permutation tests, R3 evaluation, and reports.

Reads:
  - results/ocft/probes_summary.json
  - results/ocft/injection_chosen.json
  - results/ocft/per_example_shifts_<DK>.npz  (per chosen candidate)
  - results/decomposition_ci_null/null_distribution.json   (random null K=200)

Writes:
  - results/ocft/ocft_results.json
  - results/ocft/ocft_report.md   (pre-registration block FIRST, then table)
"""

import os, sys, json, argparse
from pathlib import Path
from datetime import datetime
import numpy as np

SEED = 20260502
N_BOOT = 10_000
N_PERM = 10_000

PRE_REGISTRATION = """\
## Pre-registration (audit trail — written BEFORE inspecting any results)

**Question:** Is the §3 evidence-parallel inertness specific to the evidence
direction, or a generic property of any near-orthogonal high-AUROC L20
direction?

**Operating point (identical to §3):**
  Model = Qwen/Qwen2.5-7B-Instruct,  layer = 20,
  injection point = p0 last token (decision-only, max_interventions=1),
  rho = -0.20,  hidden_rms = 0.65,  normalize_rms(direction) = 1.0,
  N = 100 paired prompts (same sample_ids as §3 / §16.3),
  K = 200 random RMS-matched unit directions (cached from §16.3),
  bootstrap = 10 000 resamples,  permutation = 10 000 sign-flips,
  random seed = 20260502 (probes), 20260429 (cached §16.3 stats).

**Candidates (each = independent labelled binary contrast on L20 p0 hidden states):**
  D1 = source dataset      (HotpotQA vs MuSiQue)
  D2 = action prior        (margin_before > 0 on HotpotQA)
  D3 = candidate present   (T0/T1 vs N0/S0 on extractability pairs)
  D4 = observation length  (token_len > median on extractability pairs)

**Pre-registered rules — applied BEFORE looking at injection results:**
  R1  AUROC(D_k) >= 0.75              (the contrast is genuinely encoded)
  R2  |cos(D_k, A_L20)| <= 0.10       (D_k is near-orthogonal to action dir)
  R3  |Δm_parallel_k| / |Δm_full| >= 0.25
       AND  paired-permutation p(parallel_k vs random_per_prompt) < 0.05
       (D_k-parallel injection is operative — non-trivial mediation)

**Verdict logic:**
  • If at least one D_k satisfies R1 ∧ R2 ∧ R3:
      the geometric-triviality attack is FALSIFIED — §3 inertness is
      not a generic property of near-orthogonal high-AUROC directions.
  • If no D_k satisfies R3 while passing R1 ∧ R2:
      the test FAILS to falsify the geometric-triviality attack.
"""


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocft-dir", default="results/ocft")
    ap.add_argument("--null-dist",
                    default="results/decomposition_ci_null/null_distribution.json")
    ap.add_argument("--cached-shifts",
                    default="results/decomposition_ci_null/per_example_shifts.npz")
    return ap.parse_args()


def bootstrap_ci(values, n_boot=N_BOOT, seed=SEED):
    v = np.asarray(values, dtype=np.float64); n = len(v)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_boot, n))
    bm = v[idx].mean(axis=1)
    return {"mean": float(v.mean()),
            "ci_low": float(np.percentile(bm, 2.5)),
            "ci_high": float(np.percentile(bm, 97.5)),
            "n": int(n)}


def perm_test(a, b, n_perm=N_PERM, seed=SEED, two_sided=True):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    diff = a - b
    obs = float(abs(diff.mean())) if two_sided else float(diff.mean())
    rng = np.random.RandomState(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    pm = (signs * diff).mean(axis=1)
    if two_sided:
        ge = int(np.sum(np.abs(pm) >= obs))
    else:
        ge = int(np.sum(pm >= obs))
    return {"mean_diff": float(diff.mean()),
            "abs_mean_diff": float(abs(diff.mean())),
            "p_value": float((ge + 1) / (n_perm + 1)),
            "n_pairs": int(len(diff)), "n_perm": int(n_perm)}


def main():
    args = parse_args()
    out_dir = Path(args.ocft_dir)
    summary = json.load(open(out_dir / "probes_summary.json"))
    inj = json.load(open(out_dir / "injection_chosen.json"))
    cached = np.load(args.cached_shifts, allow_pickle=True)
    sample_ids = list(cached["sample_ids"])
    baseline = cached["baseline"]; full_sh = cached["full"]
    rand_per_prompt = cached["random"].mean(axis=0)
    n = len(sample_ids)

    full_boot = bootstrap_ci(full_sh, seed=SEED)
    full_abs_mean = float(np.abs(full_sh.mean()))

    # Cached random-null context (per-direction mean shifts, K=200)
    null_ctx = json.load(open(args.null_dist))
    null_signed = null_ctx["signed"]
    null_abs    = null_ctx["abs"]

    rows = []
    for c in inj["chosen"]:
        name = c["name"]
        npz = np.load(out_dir / f"per_example_shifts_{name}.npz",
                      allow_pickle=True)
        par_sh = npz["parallel"]; perp_sh = npz["perp"]
        par_boot = bootstrap_ci(par_sh, seed=SEED + 1)
        perp_boot = bootstrap_ci(perp_sh, seed=SEED + 2)
        # R3 stats
        ratio = float(abs(par_sh.mean()) / max(full_abs_mean, 1e-12))
        t_par_vs_rand = perm_test(par_sh, rand_per_prompt,
                                  seed=SEED + 10)
        t_perp_vs_full = perm_test(perp_sh, full_sh, seed=SEED + 11)
        t_perp_vs_par  = perm_test(perp_sh, par_sh,  seed=SEED + 12)
        passed_R3 = (ratio >= 0.25) and (t_par_vs_rand["p_value"] < 0.05)
        rows.append({
            "name": name,
            "contrast": c["contrast"],
            "auroc": c["auroc"],
            "cos_with_action": c["cos_with_action"],
            "var_parallel_fraction": c["files"]["var_parallel_fraction"],
            "parallel": par_boot, "perp": perp_boot,
            "ratio_parallel_over_full": ratio,
            "perm_parallel_vs_random": t_par_vs_rand,
            "perm_perp_vs_full": t_perp_vs_full,
            "perm_perp_vs_parallel": t_perp_vs_par,
            "passed_R1": c["passed_R1"],
            "passed_R2": c["passed_R2"],
            "passed_R3": bool(passed_R3),
            "operative": bool(c["passed_R1"] and c["passed_R2"] and passed_R3),
        })

    falsified = any(r["operative"] for r in rows)

    # Strict-R2 sensitivity (|cos|<=0.05). Re-evaluate operativeness with the
    # tighter near-orthogonality bound.
    rows_strict = []
    for r in rows:
        passed_R2_strict = abs(r["cos_with_action"]) <= 0.05
        rows_strict.append({
            "name": r["name"],
            "passed_R2_strict": bool(passed_R2_strict),
            "operative_strict": bool(passed_R2_strict and r["passed_R3"]),
        })
    falsified_strict = any(rs["operative_strict"] for rs in rows_strict)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {"layer": 20, "rho": -0.20, "n_samples": int(n),
                   "K_random": 200, "n_boot": N_BOOT, "n_perm": N_PERM,
                   "seed_probes": SEED, "seed_inject": 20260429,
                   "auroc_min": 0.75, "cos_max": 0.10, "cos_max_strict": 0.05,
                   "ratio_min": 0.25, "perm_p_max": 0.05},
        "full_action_dir": {
            "mean_shift": full_boot["mean"],
            "ci_low": full_boot["ci_low"], "ci_high": full_boot["ci_high"]},
        "random_null_K200_signed": {
            "mean": null_signed["mean"], "std": null_signed["std"],
            "p2_5": null_signed["p2_5"], "p97_5": null_signed["p97_5"]},
        "random_null_K200_abs": {
            "mean": null_abs["mean"], "std": null_abs["std"],
            "p97_5": null_abs["p97_5"]},
        "candidates_passed_R1R2": rows,
        "candidates_passed_R1R2_strict": rows_strict,
        "candidates_dropped": inj["dropped"],
        "verdict": ("FALSIFIED" if falsified else "NOT_FALSIFIED"),
        "verdict_strict_R2": ("FALSIFIED" if falsified_strict else "NOT_FALSIFIED"),
        "verdict_reason": (
            "At least one D_k passes R1∧R2∧R3" if falsified else
            "No D_k passes R1∧R2∧R3 — §3 evidence-parallel inertness "
            "remains consistent with geometric-triviality at this N"),
    }
    with open(out_dir / "ocft_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Report (pre-reg first) ─────────────────────────────────────────────
    md = ["# OCFT — Operative-Confound Falsification Test\n"]
    md.append(PRE_REGISTRATION)
    md.append("\n---\n\n## Results\n\n")
    md.append(f"**N = {n}**.  Full action-direction shift "
              f"= {full_boot['mean']:+.3f} "
              f"(95% CI [{full_boot['ci_low']:+.3f}, "
              f"{full_boot['ci_high']:+.3f}]).\n\n")
    md.append(
        f"Cached random-null context (K={null_ctx['K']}, RMS-matched, identical to §16.3): "
        f"signed per-direction mean shift = {null_signed['mean']:+.3f} "
        f"(95% range [{null_signed['p2_5']:+.3f}, {null_signed['p97_5']:+.3f}]); "
        f"|·|-mean = {null_abs['mean']:.3f}.\n\n")

    md.append("### Candidates dropped at R1 or R2 (no injection run)\n\n")
    if not inj["dropped"]:
        md.append("_All four candidates passed R1 (AUROC ≥ 0.75) and R2 (|cos| ≤ 0.10); "
                  "no candidate was dropped pre-injection._\n\n")
    else:
        md.append("| Candidate | Contrast | AUROC | cos(D,A) | dropped reason |\n")
        md.append("|---|---|---:|---:|---|\n")
        for d in inj["dropped"]:
            why = []
            if not d["passed_R1"]: why.append("R1 (AUROC<0.75)")
            if not d["passed_R2"]: why.append("R2 (|cos|>0.10)")
            md.append(f"| {d['name']} | {d['contrast']} | "
                      f"{d['auroc']:.3f} | {d['cos_with_action']:+.3f} | "
                      f"{', '.join(why)} |\n")
        md.append("\n")

    md.append("### Candidates passing R1∧R2 — injection results\n\n")
    md.append("| Candidate | AUROC | cos(D,A) | Δm_full | Δm_par (95% CI) | "
              "Δm_perp (95% CI) | ratio &#124;par&#124;/&#124;full&#124; | "
              "perm p (par vs random) | R3 | operative |\n")
    md.append("|---|---:|---:|---:|---|---|---:|---:|---|---|\n")
    for r in rows:
        md.append(
            f"| {r['name']} | {r['auroc']:.3f} | "
            f"{r['cos_with_action']:+.3f} | "
            f"{full_boot['mean']:+.3f} | "
            f"{r['parallel']['mean']:+.3f} "
            f"[{r['parallel']['ci_low']:+.3f}, {r['parallel']['ci_high']:+.3f}] | "
            f"{r['perp']['mean']:+.3f} "
            f"[{r['perp']['ci_low']:+.3f}, {r['perp']['ci_high']:+.3f}] | "
            f"{r['ratio_parallel_over_full']:.2f} | "
            f"{r['perm_parallel_vs_random']['p_value']:.4f} | "
            f"{'PASS' if r['passed_R3'] else 'fail'} | "
            f"{'**YES**' if r['operative'] else 'no'} |\n")

    md.append("\n### Pre-registered verdict\n\n")
    md.append(f"**{results['verdict']}**  —  {results['verdict_reason']}.\n\n")

    md.append("### Sensitivity: strict R2 (|cos| ≤ 0.05)\n\n")
    md.append("| Candidate | cos(D,A) | passed R2_strict | passed R3 | operative_strict |\n")
    md.append("|---|---:|---|---|---|\n")
    for r, rs in zip(rows, rows_strict):
        md.append(f"| {r['name']} | {r['cos_with_action']:+.3f} | "
                  f"{'PASS' if rs['passed_R2_strict'] else 'fail'} | "
                  f"{'PASS' if r['passed_R3'] else 'fail'} | "
                  f"{'**YES**' if rs['operative_strict'] else 'no'} |\n")
    md.append(f"\nStrict-R2 verdict: **{results['verdict_strict_R2']}**.\n\n")

    md.append("### Notes on candidate construction\n\n")
    md.append("- Δm_par for **D3** has the *opposite sign* of Δm_full: the candidate-present"
              " subspace contains a substantial 'anti-action' component along which A_L20 has"
              " a non-trivial scalar projection.  Magnitude (not sign) is what R3 evaluates.\n")
    md.append("- **D2_action_prior** has highly imbalanced labels (475:11 on hotpot p0); the"
              " probe direction may be dominated by question-specific features rather than a"
              " pure action-prior axis.  R2 still passes; readers should treat D2 as a weaker"
              " operativeness probe.\n")
    md.append("- All probes were trained at L20 with stratified 80/20 splits (LR, "
              "class_weight='balanced', C=1.0). Direction = standardised LR coef, L2-normalised.\n")

    with open(out_dir / "ocft_report.md", "w") as f:
        f.writelines(md)
    print(f"[done] {out_dir/'ocft_results.json'}")
    print(f"[done] {out_dir/'ocft_report.md'}")
    print(f"[verdict] {results['verdict']}")


if __name__ == "__main__":
    main()
