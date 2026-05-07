#!/usr/bin/env python3
"""
Probe Confound Test.

Tests whether the L20 evidence-sufficiency probe encodes a genuine
"evidence sufficient" representation, or whether it is detecting one of
three surface-level confounds:
  A. Observation token-count (length).
  B. Sentence-count / structural complexity.
  C. Question-observation lexical (Jaccard) overlap.

For each confound C in {A, B, C}:
  1. Build a binary label from the SAME 486 observations used for the
     evidence probe (median split on the confound statistic).
  2. Train a logistic-regression probe on the SAME L20 hidden states.
  3. Report AUROC (matched 80/20 stratified split, seed=42).
  4. Extract the unit-norm probe direction w_C.
  5. Compute cos(w_C, evidence_dir) and cos(w_C, action_dir).
  6. Decompose action_dir = (action_dir·w_C) w_C + perp; save both as
     direction npz files compatible with run_decomposition_test.py.
  7. Compute analytical Δm proxy = ρ · cos(d_inj, action_dir) · ||action_dir||
     for d_inj ∈ {full, parallel, perp}.

Outputs (results/probe_confound_test/):
  summary.json
  per_confound_results.json
  report.md
  direction_decomp_{conf}_parallel_layer20.npz
  direction_decomp_{conf}_perp_layer20.npz
"""

import argparse, json, re, sys
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))


STOPWORDS = {
    "a","an","the","and","or","but","of","to","in","on","at","for","with","by","from",
    "is","are","was","were","be","been","being","am","do","does","did","have","has","had",
    "this","that","these","those","it","its","as","if","then","than","so","such",
    "what","which","who","whom","whose","when","where","why","how",
    "i","you","he","she","we","they","me","him","her","us","them",
    "my","your","his","its","our","their","not","no","yes",
    "into","over","under","about","between","through","during","before","after",
    "up","down","out","off","again","just","only","also",
}


def tokenize(text: str):
    return re.findall(r"[A-Za-z0-9]+", (text or "").lower())


def sentence_count(text: str) -> int:
    if not text:
        return 0
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return sum(1 for p in parts if p.strip())


def jaccard_overlap(q: str, o: str) -> float:
    q_tokens = {t for t in tokenize(q) if t not in STOPWORDS and len(t) > 1}
    o_tokens = {t for t in tokenize(o) if t not in STOPWORDS and len(t) > 1}
    if not q_tokens and not o_tokens:
        return 0.0
    inter = q_tokens & o_tokens
    union = q_tokens | o_tokens
    return len(inter) / len(union) if union else 0.0


def median_binary(values: np.ndarray) -> np.ndarray:
    """Above-median → 1, at/below-median → 0. Stable when ties exist."""
    med = float(np.median(values))
    return (values > med).astype(np.int32)


def train_probe(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """Match phase1_multilayer_probe.train_probe protocol."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(sss.split(Xs, y))

    clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                             solver="lbfgs", random_state=seed)
    clf.fit(Xs[train_idx], y[train_idx])
    y_pred = clf.predict(Xs[test_idx])
    y_prob = clf.predict_proba(Xs[test_idx])[:, 1]

    bal_acc = float(balanced_accuracy_score(y[test_idx], y_pred))
    auroc = float(roc_auc_score(y[test_idx], y_prob))

    clf_all = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                                 solver="lbfgs", random_state=seed)
    clf_all.fit(Xs, y)
    w_orig = clf_all.coef_[0] / scaler.scale_
    direction = (w_orig / (np.linalg.norm(w_orig) + 1e-12)).astype(np.float32)

    return {
        "balanced_accuracy": bal_acc,
        "auroc": auroc,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_test_label0": int((y[test_idx] == 0).sum()),
        "n_test_label1": int((y[test_idx] == 1).sum()),
        "n_label0": int((y == 0).sum()),
        "n_label1": int((y == 1).sum()),
    }, direction


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def decompose_against(action_raw: np.ndarray, w_unit: np.ndarray):
    """Decompose action_raw into (parallel-to-w, perpendicular-to-w)."""
    proj_scalar = float(np.dot(action_raw, w_unit))
    parallel = proj_scalar * w_unit
    perp = action_raw - parallel
    return parallel.astype(np.float32), perp.astype(np.float32), proj_scalar


def save_decomp_dir(path: Path, direction: np.ndarray, layer: int,
                    label: str, source_action: str, source_confound: str):
    np.savez(str(path),
             decision_direction=direction.astype(np.float32),
             decision_direction_normalized=(direction /
                                            (np.linalg.norm(direction) + 1e-12)).astype(np.float32),
             layer=np.int32(layer),
             method=f"decomp_{label}",
             source_action=source_action,
             source_confound=source_confound)



def load_observation_map(baseline_trace: Path):
    """sample_id → (question, observation_text) from baseline JSONL."""
    obs_map = {}
    with open(baseline_trace) as f:
        for line in f:
            ep = json.loads(line)
            sid = ep["sample_id"]
            steps = ep.get("steps", [])
            if not steps:
                continue
            s0 = steps[0]
            if s0.get("action") != "search" or not s0.get("observation"):
                continue
            obs_map[sid] = (ep.get("question", ""), s0["observation"])
    return obs_map


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--activations",
                   default="results/phase1_probe/activations_multilayer.npz")
    p.add_argument("--baseline-trace",
                   default="results/l20_rho020_n500/baseline_results.jsonl")
    p.add_argument("--evidence-dir",
                   default="results/phase1_probe/probe_direction_l20.npz")
    p.add_argument("--action-dir",
                   default="steering/directions/direction_search_v3_layer20.npz")
    p.add_argument("--out-dir", default="results/probe_confound_test")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--rho", type=float, default=-0.20,
                   help="ρ used to compute analytical Δm proxy.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  PROBE CONFOUND TEST")
    print("=" * 70)
    print(f"\n[1/6] Loading activations from {args.activations}")
    data = np.load(args.activations, allow_pickle=True)
    X = data[f"layer_{args.layer}"].astype(np.float32)
    y_evidence = data["y"].astype(np.int32)
    sample_ids = list(data["sample_ids"])
    n = len(sample_ids)
    print(f"  n_samples={n}  n_label0={(y_evidence==0).sum()}  n_label1={(y_evidence==1).sum()}")
    assert X.shape[0] == n

    print(f"\n[2/6] Loading observation texts from {args.baseline_trace}")
    obs_map = load_observation_map(Path(args.baseline_trace))
    questions, observations, missing = [], [], []
    for sid in sample_ids:
        if sid not in obs_map:
            missing.append(sid)
            questions.append("")
            observations.append("")
        else:
            q, o = obs_map[sid]
            questions.append(q)
            observations.append(o)
    print(f"  matched {n - len(missing)}/{n} samples (missing={len(missing)})")

    print(f"\n[3/6] Building confound labels (median split)")
    tok_counts = np.array([len(tokenize(o)) for o in observations], dtype=np.float32)
    sent_counts = np.array([sentence_count(o) for o in observations], dtype=np.float32)
    overlaps = np.array([jaccard_overlap(q, o) for q, o in zip(questions, observations)],
                        dtype=np.float32)

    confound_stats = {
        "A_length":   {"raw": tok_counts,  "name": "Observation token count"},
        "B_structure":{"raw": sent_counts, "name": "Observation sentence count"},
        "C_overlap":  {"raw": overlaps,    "name": "Question-observation Jaccard overlap"},
    }
    confound_labels = {}
    for key, spec in confound_stats.items():
        lbl = median_binary(spec["raw"])
        confound_labels[key] = lbl
        med = float(np.median(spec["raw"]))
        print(f"  {key:12s}  median={med:.3f}  "
              f"label0={int((lbl==0).sum())}  label1={int((lbl==1).sum())}  "
              f"corr_with_evidence={float(np.corrcoef(lbl, y_evidence)[0,1]):+.3f}")

    print(f"\n[4/6] Loading reference directions")
    ev_data = np.load(args.evidence_dir, allow_pickle=True)
    evidence_dir = ev_data["decision_direction"].astype(np.float32)
    evidence_unit = evidence_dir / (np.linalg.norm(evidence_dir) + 1e-12)
    print(f"  evidence_dir: ||.||={np.linalg.norm(evidence_dir):.4f}  "
          f"AUROC(reported)={float(ev_data['auroc']):.3f}")

    ac_data = np.load(args.action_dir, allow_pickle=True)
    action_raw = ac_data["decision_direction"].astype(np.float32)
    action_norm = float(np.linalg.norm(action_raw))
    action_unit = action_raw / (action_norm + 1e-12)
    print(f"  action_dir:   ||.||={action_norm:.4f}")
    cos_ev_ac = cos(evidence_unit, action_unit)
    print(f"  cos(evidence, action) = {cos_ev_ac:+.4f}")


    # ─── Train each confound probe and run the geometry analysis ──────────
    print(f"\n[5/6] Training confound probes")
    per_confound = {}
    confound_unit_dirs = {}

    # Re-train evidence probe on the SAME 486 activations for a fair AUROC
    ev_metrics, ev_dir_retrained = train_probe(X, y_evidence, seed=args.seed)
    print(f"  evidence (retrain): AUROC={ev_metrics['auroc']:.4f}  "
          f"BalAcc={ev_metrics['balanced_accuracy']:.4f}")

    parallel_full = action_unit @ evidence_unit  # cos
    perp_norm_full = float(np.sqrt(max(0.0, 1.0 - parallel_full ** 2)))
    ev_dm_par = args.rho * parallel_full * action_norm
    ev_dm_perp = args.rho * perp_norm_full * action_norm
    per_confound["evidence_retrain"] = {
        "name": "Evidence-sufficiency (re-trained on 486)",
        "auroc": ev_metrics["auroc"],
        "balanced_accuracy": ev_metrics["balanced_accuracy"],
        "n_label0": ev_metrics["n_label0"],
        "n_label1": ev_metrics["n_label1"],
        "cos_with_evidence_dir_saved": cos(ev_dir_retrained, evidence_unit),
        "cos_with_action_dir": cos(ev_dir_retrained, action_unit),
        "decomp": {
            "parallel_cos_action": float(parallel_full),
            "perp_norm_action": perp_norm_full,
            "delta_m_proxy_parallel": float(ev_dm_par),
            "delta_m_proxy_perp": float(ev_dm_perp),
            "rho": args.rho,
        },
    }

    for key, lbl in confound_labels.items():
        metrics, w_unit = train_probe(X, lbl, seed=args.seed)
        confound_unit_dirs[key] = w_unit

        cos_w_evidence = cos(w_unit, evidence_unit)
        cos_w_action = cos(w_unit, action_unit)

        # Decompose action_raw against w_unit (the confound axis)
        a_par_raw, a_perp_raw, proj_scalar = decompose_against(action_raw, w_unit)

        # Δm proxy: ρ · ⟨d_inj/||d_inj||, action_unit⟩ · ||action_raw||
        # Equivalently: parallel direction has cos with action = sign(proj)·1
        # if w_unit is unit norm. So Δm_par = ρ · (a_par·action_unit)/||a_par||·||action_raw||
        a_par_norm = float(np.linalg.norm(a_par_raw))
        a_perp_norm = float(np.linalg.norm(a_perp_raw))

        if a_par_norm > 1e-9:
            cos_par_action = float(np.dot(a_par_raw, action_unit) / a_par_norm)
            dm_par = args.rho * cos_par_action * action_norm
        else:
            cos_par_action = 0.0
            dm_par = 0.0
        if a_perp_norm > 1e-9:
            cos_perp_action = float(np.dot(a_perp_raw, action_unit) / a_perp_norm)
            dm_perp = args.rho * cos_perp_action * action_norm
        else:
            cos_perp_action = 0.0
            dm_perp = 0.0

        # Save direction npz files for downstream behavioral runs
        par_path = out_dir / f"direction_decomp_{key}_parallel_layer{args.layer}.npz"
        perp_path = out_dir / f"direction_decomp_{key}_perp_layer{args.layer}.npz"
        save_decomp_dir(par_path, a_par_raw, args.layer, "parallel",
                        args.action_dir, key)
        save_decomp_dir(perp_path, a_perp_raw, args.layer, "perp",
                        args.action_dir, key)

        per_confound[key] = {
            "name": confound_stats[key]["name"],
            "auroc": metrics["auroc"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "n_label0": metrics["n_label0"],
            "n_label1": metrics["n_label1"],
            "median_value": float(np.median(confound_stats[key]["raw"])),
            "label_corr_with_evidence": float(np.corrcoef(lbl, y_evidence)[0, 1]),
            "cos_with_evidence_dir_saved": cos_w_evidence,
            "cos_with_action_dir": cos_w_action,
            "decomp": {
                "proj_scalar": float(proj_scalar),
                "parallel_norm": a_par_norm,
                "perp_norm": a_perp_norm,
                "var_parallel_fraction": float(a_par_norm ** 2 / (action_norm ** 2 + 1e-12)),
                "cos_parallel_action": cos_par_action,
                "cos_perp_action": cos_perp_action,
                "delta_m_proxy_parallel": float(dm_par),
                "delta_m_proxy_perp": float(dm_perp),
                "rho": args.rho,
                "parallel_dir_path": str(par_path),
                "perp_dir_path": str(perp_path),
            },
        }
        print(f"  {key:12s}  AUROC={metrics['auroc']:.4f}  BalAcc={metrics['balanced_accuracy']:.4f}  "
              f"cos(evidence)={cos_w_evidence:+.4f}  cos(action)={cos_w_action:+.4f}  "
              f"var(par)/var(act)={per_confound[key]['decomp']['var_parallel_fraction']:.4f}")

    # ─── Cross-confound cosines ───────────────────────────────────────────
    print(f"\n[6/6] Cross-confound direction alignments")
    keys = list(confound_unit_dirs.keys())
    cross_cos = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            c = cos(confound_unit_dirs[k1], confound_unit_dirs[k2])
            cross_cos[f"{k1}__vs__{k2}"] = c
            print(f"  cos({k1}, {k2}) = {c:+.4f}")

    # ─── Build summary table and save outputs ─────────────────────────────
    table_rows = []
    for key in ["evidence_retrain", "A_length", "B_structure", "C_overlap"]:
        r = per_confound[key]
        d = r["decomp"]
        table_rows.append({
            "probe": key,
            "name": r["name"],
            "auroc": r["auroc"],
            "cos_with_evidence": r["cos_with_evidence_dir_saved"],
            "cos_with_action": r["cos_with_action_dir"],
            "delta_m_parallel": d.get("delta_m_proxy_parallel"),
            "delta_m_perp": d.get("delta_m_proxy_perp"),
        })

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "layer": args.layer,
        "n_samples": n,
        "n_matched_observations": n - len(missing),
        "rho_for_delta_m_proxy": args.rho,
        "evidence_dir_source": args.evidence_dir,
        "action_dir_source": args.action_dir,
        "action_dir_norm": action_norm,
        "cos_evidence_action_saved": cos_ev_ac,
        "summary_table": table_rows,
        "cross_confound_cosines": cross_cos,
        "missing_sample_ids": missing,
        "delta_m_proxy_definition": (
            "Δm_proxy = ρ · cos(d_inj, action_unit) · ||action_raw||  "
            "(first-order logit-margin shift induced by injecting ρ·d at L20). "
            "For behavioural Δm replace this with run_decomposition_test.py "
            "using the saved direction_decomp_{conf}_{parallel|perp}_layer20.npz files."
        ),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(out_dir / "per_confound_results.json", "w") as f:
        json.dump(per_confound, f, indent=2, ensure_ascii=False)

    # Markdown report
    md = []
    md.append(f"# Probe Confound Test\n")
    md.append(f"Model: Qwen2.5-7B-Instruct  Layer: L{args.layer}  N: {n} "
              f"(matched obs: {n - len(missing)})  ρ for Δm proxy: {args.rho}\n")
    md.append(f"\nReference: cos(evidence_dir, action_dir) = **{cos_ev_ac:+.4f}**, "
              f"||action_dir|| = {action_norm:.4f}\n")
    md.append("\n## Summary Table\n")
    md.append("| Probe | AUROC | cos(w/ evidence) | cos(w/ action) | "
              "confound-parallel Δm | confound-perp Δm |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for r in table_rows:
        md.append(f"| {r['probe']} | {r['auroc']:.4f} | "
                  f"{r['cos_with_evidence']:+.4f} | {r['cos_with_action']:+.4f} | "
                  f"{r['delta_m_parallel']:+.4f} | {r['delta_m_perp']:+.4f} |")
    md.append("\n*Δm columns use the first-order analytical proxy "
              "ρ·cos(d_inj, action)·||action||. Behavioural Δm requires running "
              "`run_decomposition_test.py` with the saved decomposition direction "
              "npz files.*\n")
    md.append("\n## Cross-confound Direction Cosines\n")
    md.append("| Pair | cos |")
    md.append("|---|---:|")
    for pair, c in cross_cos.items():
        md.append(f"| {pair} | {c:+.4f} |")

    md.append("\n## Confound construction\n")
    for key in ["A_length", "B_structure", "C_overlap"]:
        r = per_confound[key]
        md.append(f"- **{key}** ({r['name']}): median split at "
                  f"{r['median_value']:.3f}; "
                  f"label corr with evidence = {r['label_corr_with_evidence']:+.3f}")

    md.append("\n## Interpretation\n")
    md.append("- **Best (paper strengthens):** confound AUROCs < evidence AUROC AND "
              "|cos(confound, evidence)| < 0.3.")
    md.append("- **Acceptable:** AUROCs comparable but |cos(confound, action)| also "
              "near zero → Δm-parallel near zero regardless of what the probe encodes.")
    md.append("- **Worst:** |cos(confound, evidence)| > 0.5 AND Δm-parallel "
              "significantly positive (same sign as full Δm).")

    with open(out_dir / "report.md", "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"\nSummary table:")
    print(f"  {'probe':<20} {'AUROC':>7} {'cos(ev)':>9} {'cos(act)':>9} "
          f"{'Δm_par':>10} {'Δm_perp':>10}")
    for r in table_rows:
        print(f"  {r['probe']:<20} {r['auroc']:>7.4f} "
              f"{r['cos_with_evidence']:>+9.4f} {r['cos_with_action']:>+9.4f} "
              f"{r['delta_m_parallel']:>+10.4f} {r['delta_m_perp']:>+10.4f}")

    print(f"\nWrote: {out_dir/'summary.json'}")
    print(f"Wrote: {out_dir/'per_confound_results.json'}")
    print(f"Wrote: {out_dir/'report.md'}")
    print("Done.")


if __name__ == "__main__":
    main()

