#!/usr/bin/env python3
"""β-revised — re-train evidence probe with stricter labels on existing step1_h.

Tests phase-α candidate cause #1 (Llama's evidence_dir tracks doc-content
features, not evidence-sufficiency) by training probes with progressively
stricter labels and measuring AUROC drop per model. If Llama's drop is
disproportionately large vs the other 4 families, cause #1 is supported.

Label variants:
  L_loose          : original — 0 if n_sf_retrieved==0 else 1   (97 vs 389)
  L_ans_in_obs     : 1 if gold_answer string in observation (lowercase substring)
                                                              (108 vs 378)
  L_is_correct     : behavioural — 1 if model's eventual answer was correct
                                                              (103 vs 383)
  L_clean_disjoint : 1 if (n_sf>=1 AND ans_in_obs), 0 if (n_sf==0 AND not ans_in_obs);
                     drops ambiguous middle                    (≈105 vs ≈94)

Probe protocol matches cross_model_full.py:train_probe (5-fold StratifiedKFold,
LogisticRegression class_weight='balanced' C=1.0, StandardScaler per-fold,
seed=42).
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "llama_evidence_probe_strict")
LABELS_PATH = os.path.join(ROOT, "results", "phase1_probe", "labels.jsonl")
BASELINE_PATH = os.path.join(ROOT, "results", "l20_rho020_n500", "baseline_results.jsonl")

MODELS = [
    ("qwen25_7b",   "cross_model_qwen25_v2"),
    ("mistral_7b",  "cross_model_mistral_v2"),
    ("llama31_8b",  "cross_model_llama31_v2"),
    ("gemma2_9b",   "cross_model_gemma2_v2"),
    ("r1distill_7b","cross_model_r1distill_v2"),
]


def cv_auroc(X, y):
    """5-fold StratifiedKFold AUROC + BalAcc, matching cross_model_full.py."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, baccs = [], []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        p = LogisticRegression(class_weight="balanced", C=1.0,
                                max_iter=2000, solver="lbfgs", random_state=42)
        p.fit(X_tr, y[tr])
        probs = p.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(y[te], probs))
        baccs.append(balanced_accuracy_score(y[te], p.predict(X_te)))
    return {"auroc_mean": float(np.mean(aucs)),
            "auroc_std":  float(np.std(aucs)),
            "bacc_mean":  float(np.mean(baccs)),
            "fold_aurocs": [float(a) for a in aucs]}


def build_label_universe():
    """Build per-sample-id label dict for all 4 variants, plus n_sf and ans_in_obs flags."""
    recs = [json.loads(l) for l in open(LABELS_PATH)]
    bl = {}
    with open(BASELINE_PATH) as f:
        for line in f:
            ep = json.loads(line)
            bl[ep["sample_id"]] = ep
    out = {}
    for r in recs:
        sid = r["sample_id"]
        ep = bl.get(sid)
        obs = ""
        if ep and ep.get("steps"):
            obs = ep["steps"][0].get("observation", "") or ""
        ans = str(r["gold_answer"]).lower().strip()
        ans_in_obs = bool(ans) and ans in obs.lower()
        n_sf = int(r["n_sf_retrieved"])
        out[sid] = {
            "L_loose":      int(n_sf >= 1),
            "L_ans_in_obs": int(ans_in_obs),
            "L_is_correct": int(bool(r["is_correct"])),
            "L_clean_disjoint":
                1 if (n_sf >= 1 and ans_in_obs) else
                (0 if (n_sf == 0 and not ans_in_obs) else -1),
            "n_sf": n_sf,
            "ans_in_obs": ans_in_obs,
        }
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    labels = build_label_universe()
    label_names = ["L_loose", "L_ans_in_obs", "L_is_correct", "L_clean_disjoint"]

    rows = []
    for short, sub in MODELS:
        z = np.load(os.path.join(ROOT, "results", sub, "per_sample.npz"), allow_pickle=False)
        X = z["step1_h"]
        sids = list(z["step1_sample_ids"])
        peak_evi = int(z["peak_evi_layer"])
        per_label = {}
        for ln in label_names:
            ys = []
            keep_idx = []
            for i, sid in enumerate(sids):
                lab = labels[sid][ln]
                if lab == -1:
                    continue
                keep_idx.append(i); ys.append(lab)
            ys = np.array(ys, dtype=np.int32)
            X_sub = X[keep_idx]
            if len(np.unique(ys)) < 2 or min(np.bincount(ys)) < 5:
                per_label[ln] = {"skipped": True, "n_per_class": [int(c) for c in np.bincount(ys).tolist()]}
                continue
            cv = cv_auroc(X_sub, ys)
            per_label[ln] = {
                "n_total": int(len(ys)),
                "n_class0": int((ys == 0).sum()),
                "n_class1": int((ys == 1).sum()),
                **cv,
            }
        # Drops vs L_loose
        loose = per_label["L_loose"]["auroc_mean"]
        drops = {ln: float(loose - per_label[ln]["auroc_mean"])
                 for ln in label_names
                 if not per_label[ln].get("skipped")}
        rows.append({
            "model_short": short,
            "peak_evi_layer": peak_evi,
            "per_label": per_label,
            "auroc_drops_vs_loose": drops,
        })
        print(f"{short:<14} L{peak_evi}  "
              + "  ".join(f"{ln.replace('L_',''):<14}={per_label[ln].get('auroc_mean','SKIP'):.3f}"
                          if not per_label[ln].get("skipped") else
                          f"{ln.replace('L_',''):<14}=SKIP"
                          for ln in label_names))

    summary = {
        "spec_version": "beta-revised-strict-probe-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "labels_path": os.path.relpath(LABELS_PATH, ROOT),
        "baseline_path": os.path.relpath(BASELINE_PATH, ROOT),
        "label_definitions": {
            "L_loose": "n_sf_retrieved >= 1 vs == 0  (current loose; 389 vs 97)",
            "L_ans_in_obs": "gold_answer lowercase substring in observation  (108 vs 378)",
            "L_is_correct": "is_correct flag (model's eventual answer correct)  (103 vs 383)",
            "L_clean_disjoint": "(n_sf>=1 AND ans_in_obs) vs (n_sf==0 AND NOT ans_in_obs); ambiguous dropped",
        },
        "probe_protocol": "5-fold StratifiedKFold, LogisticRegression class_weight=balanced "
                          "C=1.0 lbfgs max_iter=2000 random_state=42; per-fold StandardScaler",
        "rows": rows,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    write_readme(rows, summary)
    print()
    print("AUROC drops vs L_loose (positive = strict probe AUROC < loose AUROC):")
    print(f"{'model':<14} {'ans_in_obs':<12} {'is_correct':<12} {'clean_disj':<12}")
    for r in rows:
        d = r["auroc_drops_vs_loose"]
        print(f"  {r['model_short']:<12} "
              f"{d.get('L_ans_in_obs', float('nan')):>+11.3f} "
              f"{d.get('L_is_correct', float('nan')):>+11.3f} "
              f"{d.get('L_clean_disjoint', float('nan')):>+11.3f}")


def write_readme(rows, summary):
    md = ["# β-revised — strict-label evidence probe per model\n"]
    md.append(f"spec_version: {summary['spec_version']}\n")
    md.append("## Label variants\n")
    for k, v in summary["label_definitions"].items():
        md.append(f"- **{k}**: {v}")
    md.append("\nProbe protocol: " + summary["probe_protocol"] + "\n")
    md.append("## AUROC table (5-fold CV mean ± std)\n")
    md.append("| model | L_evi | L_loose | L_ans_in_obs | L_is_correct | L_clean_disjoint |")
    md.append("|---|---|---|---|---|---|")
    for r in rows:
        cells = []
        for ln in ["L_loose", "L_ans_in_obs", "L_is_correct", "L_clean_disjoint"]:
            d = r["per_label"][ln]
            if d.get("skipped"):
                cells.append("SKIP")
            else:
                cells.append(f"{d['auroc_mean']:.3f}±{d['auroc_std']:.3f} "
                             f"(n={d['n_total']}, {d['n_class0']}/{d['n_class1']})")
        md.append(f"| {r['model_short']} | L{r['peak_evi_layer']} | " + " | ".join(cells) + " |")
    md.append("\n## AUROC drops vs L_loose (positive = strict label is harder)\n")
    md.append("| model | ans_in_obs | is_correct | clean_disjoint |")
    md.append("|---|---|---|---|")
    for r in rows:
        d = r["auroc_drops_vs_loose"]
        md.append(f"| {r['model_short']} | "
                  f"{d.get('L_ans_in_obs', float('nan')):+.3f} | "
                  f"{d.get('L_is_correct', float('nan')):+.3f} | "
                  f"{d.get('L_clean_disjoint', float('nan')):+.3f} |")
    md.append("\nInterpretation: if Llama's drops are disproportionately large compared to other models, "
              "phase-α candidate cause #1 (probe tracks doc-content features, not evidence sufficiency) is supported. "
              "If drops are similar across models, cause #1 is not supported and cause #2 (multi-position evidence representation) "
              "should be tested next.\n")
    with open(os.path.join(OUT_DIR, "README.md"), "w") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    main()
