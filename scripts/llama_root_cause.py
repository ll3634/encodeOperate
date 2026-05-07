#!/usr/bin/env python3
"""Root-cause diagnostic for the Llama-3.1-8B paired-corruption instrument failure.

Five orthogonal tests, all run on existing per_sample.npz + HotpotQA data.
No model loading.

D1 — Δh magnitude:        ‖h_clean − h_corrupted‖₂ per group (median + percentiles)
D2 — Δh alignment:        cos(Δh, evidence_dir) distribution per group
D3 — Δh SVD subspace:     top-3 singular directions of group-A Δh; cos with evi_dir
D4 — Paired-induced AUROC: train probe on (paired clean=1SF) vs (paired A_corr=0SF);
                          if Llama AUROC ≈ 0.5 while Qwen ≫ 0.5, the step1 AUROC=0.861
                          is schema-level (0-doc structural absence), NOT content-level
                          evidence detection.
D5 — Surface-form check:  per-pair |Δlen| of A-swap vs B-swap (chars + tokens), test
                          whether B-swap perturbs more raw text than A-swap (would
                          explain pooled E5 inversion).

Outputs → results/llama_root_cause/{summary.json, README.md}
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from scripts.paired_corruption_analysis import (  # noqa: E402
    select_samples, make_corrupted_obs, get_hotpot_paragraph_text,
)

OUT_DIR = ROOT / "results" / "llama_root_cause"
BASELINE_PATH = ROOT / "results" / "l20_rho020_n500" / "baseline_results.jsonl"
HOTPOTQA_PATH = ROOT / "data" / "hotpotqa" / "hotpot_dev_distractor_v1.json"

MODELS = [
    ("qwen25_7b",   "cross_model_qwen25_v2"),
    ("mistral_7b",  "cross_model_mistral_v2"),
    ("llama31_8b",  "cross_model_llama31_v2"),
    ("gemma2_9b",   "cross_model_gemma2_v2"),
    ("r1distill_7b","cross_model_r1distill_v2"),
]

SEED = 20260503


def percentile_summary(x):
    return {"median": float(np.median(x)), "mean": float(np.mean(x)),
            "p05": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95)),
            "n": int(len(x))}


def cv_auroc(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, baccs = [], []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        p = LogisticRegression(class_weight="balanced", C=1.0,
                               max_iter=2000, solver="lbfgs", random_state=42)
        p.fit(X_tr, y[tr])
        aucs.append(roc_auc_score(y[te], p.predict_proba(X_te)[:, 1]))
        baccs.append(balanced_accuracy_score(y[te], p.predict(X_te)))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(baccs))


def per_model_residual_diagnostics(z, evi_dir):
    """D1 (Δh norm), D2 (cos with evi_dir), D3 (SVD top-3) per group."""
    ph_c = z["pair_h_clean"].astype(np.float64)
    ph_x = z["pair_h_corrupted"].astype(np.float64)
    groups = list(z["pair_groups"])
    out = {"D1_dh_norm": {}, "D2_dh_cos_evi": {}, "D3_svd_top3_cos_evi": {}}
    evi_dir = evi_dir / (np.linalg.norm(evi_dir) + 1e-12)
    for gi, g in enumerate(groups):
        dh = ph_c[gi] - ph_x[gi]                 # (N, D)
        norms = np.linalg.norm(dh, axis=1)       # (N,)
        # cos(Δh, evi_dir) per pair
        dh_unit = dh / (norms[:, None] + 1e-12)
        cos_evi = dh_unit @ evi_dir              # (N,)
        out["D1_dh_norm"][g] = percentile_summary(norms)
        out["D2_dh_cos_evi"][g] = {
            **percentile_summary(np.abs(cos_evi)),
            "signed_mean": float(cos_evi.mean()),
            "signed_median": float(np.median(cos_evi)),
        }
        # SVD on group-A Δh only (others omitted to keep summary compact)
        if g == "A":
            try:
                U, S, Vt = np.linalg.svd(dh, full_matrices=False)
                top3 = Vt[:3]                    # (3, D)
                top3_unit = top3 / (np.linalg.norm(top3, axis=1, keepdims=True) + 1e-12)
                cos_top = (top3_unit @ evi_dir).tolist()
                # Variance fractions
                S2 = S ** 2
                var_frac = (S2 / S2.sum()).tolist()[:3]
                out["D3_svd_top3_cos_evi"]["A"] = {
                    "abs_cos": [float(abs(c)) for c in cos_top],
                    "signed_cos": [float(c) for c in cos_top],
                    "var_frac": [float(v) for v in var_frac],
                }
            except np.linalg.LinAlgError:
                out["D3_svd_top3_cos_evi"]["A"] = {"error": "svd_failed"}
    return out


def paired_induced_probe(z):
    """D4: train probe on (paired clean = label 1) vs (paired A_corrupted = label 0)
    using L24 last-token residuals from per_sample.npz.
    """
    ph_c = z["pair_h_clean"].astype(np.float32)        # (3, N, D)
    ph_x = z["pair_h_corrupted"].astype(np.float32)
    groups = list(z["pair_groups"])
    a_idx = groups.index("A")
    X_pos = ph_c[a_idx]                                 # 1 SF + 1 dist
    X_neg = ph_x[a_idx]                                 # 0 SF + 2 dist
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([np.ones(len(X_pos), np.int32),
                        np.zeros(len(X_neg), np.int32)])
    auroc, std, bacc = cv_auroc(X, y)
    return {"auroc_mean": auroc, "auroc_std": std, "bacc_mean": bacc,
            "n_pos": int(len(X_pos)), "n_neg": int(len(X_neg))}


def surface_form_diagnostics():
    """D5: per-pair character-length deltas for A-swap and B-swap."""
    import random
    samples = select_samples(str(BASELINE_PATH), str(HOTPOTQA_PATH), n=200, seed=42)
    rows = []
    for gi, group in enumerate(("A", "B")):
        for i, s in enumerate(samples):
            rng = random.Random(42 + gi * 10000)
            for j in range(i):
                make_corrupted_obs(samples[j], group, rng)
            clean_obs, corr_obs = make_corrupted_obs(s, group, rng)
            rows.append({"group": group, "len_clean": len(clean_obs),
                         "len_corr": len(corr_obs),
                         "abs_delta": abs(len(clean_obs) - len(corr_obs))})
    A_dlen = np.array([r["abs_delta"] for r in rows if r["group"] == "A"])
    B_dlen = np.array([r["abs_delta"] for r in rows if r["group"] == "B"])
    mw = mannwhitneyu(A_dlen, B_dlen, alternative="two-sided")
    return {
        "A_abs_delta_chars": percentile_summary(A_dlen),
        "B_abs_delta_chars": percentile_summary(B_dlen),
        "MW_p_two_sided": float(mw.pvalue),
        "median_ratio_B_over_A": float(np.median(B_dlen) / max(np.median(A_dlen), 1e-9)),
    }



def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "spec_version": "llama-root-cause-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "models": {},
    }

    # D1/D2/D3 + D4 — per-model
    for label, sub in MODELS:
        npz_path = ROOT / "results" / sub / "per_sample.npz"
        if not npz_path.exists():
            print(f"[skip] {label}: missing {npz_path}")
            continue
        z = np.load(npz_path, allow_pickle=False)
        evi_dir = z["evidence_dir"].astype(np.float64)
        rec = {"hidden_size": int(z["hidden_size"]),
               "peak_evi_layer": int(z["peak_evi_layer"]),
               "peak_act_layer": int(z["peak_act_layer"])}
        rec.update(per_model_residual_diagnostics(z, evi_dir))
        rec["D4_paired_induced_probe"] = paired_induced_probe(z)
        out["models"][label] = rec
        d1 = rec["D1_dh_norm"]
        d2 = rec["D2_dh_cos_evi"]
        d4 = rec["D4_paired_induced_probe"]
        print(f"[{label:14s}] D1 ‖Δh‖ med A={d1['A']['median']:.3f} B={d1['B']['median']:.3f} | "
              f"D2 |cos(Δh,evi)| med A={d2['A']['median']:.4f} B={d2['B']['median']:.4f} | "
              f"D4 paired-AUROC={d4['auroc_mean']:.3f}±{d4['auroc_std']:.3f}")

    # D5 — surface-form (model-independent; based on HotpotQA only)
    print("\n[D5] surface-form |Δlen| analysis ...")
    out["D5_surface_form"] = surface_form_diagnostics()
    d5 = out["D5_surface_form"]
    print(f"[D5] A med |Δchars|={d5['A_abs_delta_chars']['median']:.0f}  "
          f"B med |Δchars|={d5['B_abs_delta_chars']['median']:.0f}  "
          f"ratio B/A={d5['median_ratio_B_over_A']:.2f}  MW p={d5['MW_p_two_sided']:.2e}")

    summary_path = OUT_DIR / "summary.json"
    with summary_path.open("w") as f:
        json.dump(out, f, indent=2)

    # Compact README
    lines = []
    lines.append("# Llama-3.1-8B paired-corruption root-cause diagnostic\n")
    lines.append(f"spec_version: {out['spec_version']}  seed: {SEED}\n")
    lines.append("## D1 — ‖Δh‖₂ at last token, per group (median)\n")
    lines.append("| model | A | B | C |")
    lines.append("|---|---|---|---|")
    for label, _ in MODELS:
        if label not in out["models"]:
            continue
        d1 = out["models"][label]["D1_dh_norm"]
        lines.append(f"| {label} | {d1['A']['median']:.3f} | {d1['B']['median']:.3f} | {d1['C']['median']:.3f} |")
    lines.append("\n## D2 — |cos(Δh, evidence_dir)| (median per group)\n")
    lines.append("| model | A | B | C |")
    lines.append("|---|---|---|---|")
    for label, _ in MODELS:
        if label not in out["models"]:
            continue
        d2 = out["models"][label]["D2_dh_cos_evi"]
        lines.append(f"| {label} | {d2['A']['median']:.4f} | {d2['B']['median']:.4f} | {d2['C']['median']:.4f} |")
    lines.append("\n## D3 — Group-A Δh top-3 SVD directions, |cos| with evidence_dir\n")
    lines.append("| model | top1 | top2 | top3 | var_frac (top3) |")
    lines.append("|---|---|---|---|---|")
    for label, _ in MODELS:
        if label not in out["models"]:
            continue
        s3 = out["models"][label]["D3_svd_top3_cos_evi"].get("A", {})
        if "abs_cos" not in s3:
            lines.append(f"| {label} | n/a | n/a | n/a | n/a |")
            continue
        ac = s3["abs_cos"]
        vf = s3["var_frac"]
        lines.append(f"| {label} | {ac[0]:.3f} | {ac[1]:.3f} | {ac[2]:.3f} | "
                     f"[{vf[0]:.2f}, {vf[1]:.2f}, {vf[2]:.2f}] |")
    lines.append("\n## D4 — Paired-induced probe (clean=1SF vs A_corrupted=0SF), 5-fold CV\n")
    lines.append("| model | AUROC | bacc | step1-AUROC reference |")
    lines.append("|---|---|---|---|")
    step1_ref = {"qwen25_7b": 0.823, "mistral_7b": 0.773, "llama31_8b": 0.861,
                 "gemma2_9b": 0.842, "r1distill_7b": 0.777}
    for label, _ in MODELS:
        if label not in out["models"]:
            continue
        d4 = out["models"][label]["D4_paired_induced_probe"]
        lines.append(f"| {label} | {d4['auroc_mean']:.3f}±{d4['auroc_std']:.3f} | "
                     f"{d4['bacc_mean']:.3f} | {step1_ref.get(label, 'n/a')} |")
    lines.append("\n## D5 — Surface-form (model-independent)\n")
    lines.append(f"- A-swap median |Δchars| = {d5['A_abs_delta_chars']['median']:.0f}")
    lines.append(f"- B-swap median |Δchars| = {d5['B_abs_delta_chars']['median']:.0f}")
    lines.append(f"- Ratio B/A = {d5['median_ratio_B_over_A']:.2f}  (MW p={d5['MW_p_two_sided']:.2e})\n")
    lines.append("## Outputs\n")
    lines.append(f"- {summary_path.relative_to(ROOT)}\n")
    (OUT_DIR / "README.md").write_text("\n".join(lines))
    print(f"\nWrote {summary_path} and {OUT_DIR / 'README.md'}")


if __name__ == "__main__":
    main()
