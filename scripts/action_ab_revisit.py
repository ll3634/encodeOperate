#!/usr/bin/env python3
"""Action A/B revisited, with corruption protocol now validated by paired-induced E5.

Two questions:

(Q1) Orthogonality robustness.  The paper claims cos(evidence, action) ≈ 0
     using cross-sectional `evidence_dir`.  After `paired_induced_e5/` we now
     have a second evidence direction (paired-induced) that is nearly
     orthogonal to the cross-sectional one.  For the routing thesis, the
     critical test is whether BOTH evidence operationalisations are
     orthogonal to action_dir.  If yes, the orthogonality claim is robust to
     direction choice.  If only one is, the routing story is direction-
     specific and the paper has to commit.

(Q2) Action A/B as an informative test.  E5 with the paired-induced direction
     PASSED for all 5 models — corruption reaches the residual everywhere.
     Re-run action A/B with each model's existing `action_dir`, using the
     same geom-median + bootstrap CI protocol as E5, so the numbers are
     directly comparable.  A persistent null in Llama (or R1) would now mean:
     corruption reaches the residual but does not project onto action_dir.

We additionally report a signal/noise diagnostic for action_dir on the
paired data:
     signal ≈ ⟨|Δh·action_dir|⟩            (group A)
     noise  ≈ ‖Δh‖ · 1/√D                  (random-direction floor)
     S/N    = signal / noise
S/N ≈ 1 → null is uninformative (signal at noise floor).
S/N ≫ 1 → null is informative (signal exists but does not separate A from B).

Inputs : results/cross_model_{model}_v2/per_sample.npz
Outputs: results/action_ab_revisit/{summary.json, README.md}
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "results" / "action_ab_revisit"

MODELS = [
    ("qwen25_7b",   "cross_model_qwen25_v2"),
    ("mistral_7b",  "cross_model_mistral_v2"),
    ("llama31_8b",  "cross_model_llama31_v2"),
    ("gemma2_9b",   "cross_model_gemma2_v2"),
    ("r1distill_7b","cross_model_r1distill_v2"),
]
SEED = 20260503
N_BOOT = 1000


def geom_median(x, n_iter=200, eps=1e-9):
    y = float(np.median(x))
    for _ in range(n_iter):
        d = np.maximum(np.abs(x - y), eps)
        w = 1.0 / d
        y_new = float(np.sum(w * x) / np.sum(w))
        if abs(y_new - y) < eps:
            break
        y = y_new
    return y


def lognormal_boot_ratio_ci(a, b, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    na, nb = len(a), len(b)
    lr = np.empty(n_boot)
    for i in range(n_boot):
        lr[i] = (np.log(geom_median(a[rng.integers(0, na, na)]))
                 - np.log(geom_median(b[rng.integers(0, nb, nb)])))
    return float(np.exp(np.quantile(lr, 0.025))), float(np.exp(np.quantile(lr, 0.975)))


def ab_stats(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    gmA, gmB = geom_median(a), geom_median(b)
    ratio = gmA / gmB if gmB > 0 else float("nan")
    lo, hi = lognormal_boot_ratio_ci(a, b)
    mw = mannwhitneyu(a, b, alternative="two-sided")
    return {"gm_A": float(gmA), "gm_B": float(gmB),
            "AB_ratio": float(ratio), "CI95": [lo, hi],
            "MW_p_two": float(mw.pvalue), "n_A": int(len(a)), "n_B": int(len(b))}


def fit_paired_evi_dir(z, seed=SEED):
    """Train paired-induced evidence direction on full group-A data (used here
    only for cos-triple analysis; orthogonality to action does not need OOF)."""
    ph_c = z["pair_h_clean"].astype(np.float32)
    ph_x = z["pair_h_corrupted"].astype(np.float32)
    groups = list(z["pair_groups"])
    ai = groups.index("A")
    X = np.concatenate([ph_c[ai], ph_x[ai]], axis=0)
    y = np.concatenate([np.ones(ph_c[ai].shape[0]), np.zeros(ph_x[ai].shape[0])])
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs",
                            random_state=seed).fit(Xs, y)
    w = lr.coef_[0] / (sc.scale_ + 1e-12)
    return (w / (np.linalg.norm(w) + 1e-12)).astype(np.float64)


def signal_noise_diagnostic(z, action_dir, evi_paired_dir):
    """For action_dir on the paired data: predict signal vs noise floor."""
    ph_c = z["pair_h_clean"].astype(np.float64)
    ph_x = z["pair_h_corrupted"].astype(np.float64)
    groups = list(z["pair_groups"])
    ai = groups.index("A")
    dh_A = ph_c[ai] - ph_x[ai]
    norms = np.linalg.norm(dh_A, axis=1)
    D = ph_c.shape[-1]
    # observed |proj on action_dir|
    proj_act = np.abs(dh_A @ action_dir)
    proj_paired = np.abs(dh_A @ evi_paired_dir)
    # noise floor: same-norm random direction expected magnitude ≈ ‖Δh‖/√D
    noise_floor = norms / np.sqrt(D)
    return {
        "median_dh_norm": float(np.median(norms)),
        "median_proj_action": float(np.median(proj_act)),
        "median_proj_paired_evi": float(np.median(proj_paired)),
        "median_noise_floor": float(np.median(noise_floor)),
        "SN_action_over_floor": float(np.median(proj_act) / max(np.median(noise_floor), 1e-12)),
        "SN_paired_evi_over_floor": float(np.median(proj_paired) / max(np.median(noise_floor), 1e-12)),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, sub in MODELS:
        npz = ROOT / "results" / sub / "per_sample.npz"
        if not npz.exists():
            print(f"[skip] {label}: missing {npz}"); continue
        z = np.load(npz, allow_pickle=False)
        evi_cross = z["evidence_dir"].astype(np.float64)
        evi_cross /= (np.linalg.norm(evi_cross) + 1e-12)
        action = z["action_dir"].astype(np.float64)
        action /= (np.linalg.norm(action) + 1e-12)
        evi_paired = fit_paired_evi_dir(z)
        # 3-way cos triple
        cos_cross_action = float(evi_cross @ action)
        cos_paired_action = float(evi_paired @ action)
        cos_paired_cross = float(evi_paired @ evi_cross)
        # Action A/B using existing action_dir + geom-median protocol
        d_act = z["pair_d_act"].astype(np.float64)
        groups = list(z["pair_groups"])
        ai, bi, ci = groups.index("A"), groups.index("B"), groups.index("C")
        action_ab = ab_stats(d_act[ai], d_act[bi])
        action_ab["gm_C"] = float(geom_median(d_act[ci].astype(np.float64)))
        # Signal/noise
        sn = signal_noise_diagnostic(z, action, evi_paired)
        rows.append({"model": label,
                     "peak_evi_layer": int(z["peak_evi_layer"]),
                     "peak_act_layer": int(z["peak_act_layer"]),
                     "cos_triple": {"evi_cross__action": cos_cross_action,
                                    "evi_paired__action": cos_paired_action,
                                    "evi_paired__evi_cross": cos_paired_cross},
                     "action_AB_geom": action_ab,
                     "signal_noise": sn})
        print(f"[{label:<14}] cos(cross,act)={cos_cross_action:+.4f}  "
              f"cos(paired,act)={cos_paired_action:+.4f}  "
              f"action AB={action_ab['AB_ratio']:.3f} CI=[{action_ab['CI95'][0]:.2f},{action_ab['CI95'][1]:.2f}] "
              f"p={action_ab['MW_p_two']:.2e}  S/N(act)={sn['SN_action_over_floor']:.2f}  "
              f"S/N(evi_paired)={sn['SN_paired_evi_over_floor']:.2f}")
    summary = {"spec_version": "action-ab-revisit-v1",
               "generated_at": datetime.now(timezone.utc).isoformat(),
               "seed": SEED, "n_bootstrap": N_BOOT, "rows": rows}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    write_readme(rows)
    print(f"\nWrote {OUT_DIR/'summary.json'} and {OUT_DIR/'README.md'}")


def write_readme(rows):
    PASS = lambda e: (e["AB_ratio"] > 1.2 and e["MW_p_two"] < 0.05)
    L = []
    L.append("# Action A/B revisited — orthogonality robustness + informative null test\n")
    L.append("spec_version: action-ab-revisit-v1\n")
    L.append("Companion to `paired_induced_e5/`.  E5 with the paired-induced direction")
    L.append("PASSED for all 5 models, validating the corruption protocol.  This script")
    L.append("addresses two follow-up questions reviewers will raise:\n")
    L.append("- **Q1** Is the orthogonality claim `cos(evidence, action) ≈ 0` robust to")
    L.append("  the choice of evidence direction (cross-sectional vs paired-induced)?")
    L.append("- **Q2** With the corruption protocol now validated, is action A/B")
    L.append("  informative for Llama and R1?  A persistent null means the corruption")
    L.append("  reaches the residual but does not project onto `action_dir`.\n")
    L.append("## Cos triple per model\n")
    L.append("| model | cos(evi_cross, action) | cos(evi_paired, action) | cos(evi_paired, evi_cross) |")
    L.append("|---|---|---|---|")
    for r in rows:
        c = r["cos_triple"]
        L.append(f"| {r['model']} | {c['evi_cross__action']:+.4f} | "
                 f"{c['evi_paired__action']:+.4f} | {c['evi_paired__evi_cross']:+.4f} |")
    L.append("\n**Interpretation (Q1).** If `|cos(evi_paired, action)|` is small for")
    L.append("all models (comparable to `|cos(evi_cross, action)|`), the orthogonality")
    L.append("between evidence and action is robust to direction choice — the two")
    L.append("evidence operationalisations are two near-orthogonal axes of an evidence")
    L.append("subspace, both perpendicular to action.  If `|cos(evi_paired, action)|`")
    L.append("is large for some model, then 'evidence ⊥ action' is direction-specific")
    L.append("and that model's routing claim weakens.\n")
    L.append("## Action A/B (geom-median + bootstrap CI95, MW p two-sided)\n")
    L.append("Same data, same projections as the existing cross-model results, but")
    L.append("computed with the same statistic family as the E5 readout for direct")
    L.append("comparability.\n")
    L.append("| model | AB | CI95 | MW p | gm_A | gm_B | gm_C | verdict |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        a = r["action_AB_geom"]
        L.append(f"| {r['model']} | {a['AB_ratio']:.3f} | "
                 f"[{a['CI95'][0]:.3f}, {a['CI95'][1]:.3f}] | {a['MW_p_two']:.3g} | "
                 f"{a['gm_A']:.3f} | {a['gm_B']:.3f} | {a['gm_C']:.3f} | "
                 f"{'PASS' if PASS(a) else 'FAIL'} |")
    L.append("\n## Signal-vs-noise diagnostic for action_dir on paired data\n")
    L.append("Median per-pair magnitudes for group-A pairs at the evidence layer:\n")
    L.append("| model | ‖Δh‖ | |Δh·action| | |Δh·evi_paired| | noise floor (‖Δh‖/√D) | S/N action | S/N evi_paired |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        s = r["signal_noise"]
        L.append(f"| {r['model']} | {s['median_dh_norm']:.3f} | "
                 f"{s['median_proj_action']:.3f} | {s['median_proj_paired_evi']:.3f} | "
                 f"{s['median_noise_floor']:.3f} | {s['SN_action_over_floor']:.2f} | "
                 f"{s['SN_paired_evi_over_floor']:.2f} |")
    L.append("\n**Interpretation (Q2).**")
    L.append("- S/N(evi_paired) ≫ 1 (sanity): paired-induced evidence direction is")
    L.append("  well above the noise floor — confirms the perturbation is being")
    L.append("  measured.")
    L.append("- S/N(action) ≫ 1 with action AB FAIL → corruption shifts the residual")
    L.append("  along action_dir but the shift does not differentiate evidence-swap")
    L.append("  from distractor-swap.  Routing genuinely absent.")
    L.append("- S/N(action) ~ 1 with action AB FAIL → action signal is at the noise")
    L.append("  floor; null is uninformative for routing (same instrument-calibration")
    L.append("  problem the cross-sectional evi_dir had with E5).\n")
    L.append("## Outputs\n")
    L.append("- `results/action_ab_revisit/summary.json`")
    L.append("- `results/action_ab_revisit/README.md`")
    (OUT_DIR / "README.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
