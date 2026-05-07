#!/usr/bin/env python3
"""
Closure-B Probe — Three experiments to test Link B of the thesis chain:
  (A) Probe encodes evidence insufficiency  [already confirmed, AUROC=0.862]
  (B) This signal causally links to action  [THIS SCRIPT]
  (C) Signal is destroyed by thought gen    [confirmed]
  (D) Decision-point steering fixes it      [confirmed, A3]

Experiment 1: Behavioral Outcome Probe — FULL dataset (LOO-CV, N_pos≈19)
  Label=1: samples triggered AND rescued (via additional search) in A3
  Label=0: all others (not triggered OR triggered but not rescued)
  → Reports AUROC. (Confounded — negatives include non-triggered samples)

Experiment 2: Permutation test for cos(p0_probe, steering)
  Shuffle the evidence labels 1000×, train p0 probe each time,
  compute cos(probe, steering) → p-value for observed cosine

Experiment 3: Within-Triggered Behavioral Probe (CLEAN, LOO-CV)
  Only uses A3 triggered subset (N≈93):
    Label=1: rescued via search (N≈19)
    Label=0: triggered but NOT rescued (N≈74)
  → Reports AUROC, cos(probe, steering), permutation test for cos
"""

import os, json, argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score

POSITIONS = ["p0_input", "p1_25pct", "p2_50pct", "p3_75pct", "p4_100pct"]

# ── Data loading ──────────────────────────────────────────────────────────────
def load_a3_triggered_and_rescued(baseline_path, steered_path):
    """Return (triggered_set, rescued_set) of sample_ids.
    triggered = steered had more tool_calls than baseline.
    rescued   = triggered AND baseline wrong AND steered correct.
    """
    base = {r['sample_id']: r
            for r in (json.loads(l) for l in open(baseline_path))}
    triggered, rescued = set(), set()
    for j in (json.loads(l) for l in open(steered_path)):
        sid = j['sample_id']
        b = base.get(sid)
        if b is None:
            continue
        j_tc = j.get('tool_calls', 0)
        b_tc = b.get('tool_calls', 0)
        if j_tc > b_tc:
            triggered.add(sid)
            em_b = b.get('em_correct', b.get('exact_match', b.get('is_correct', 0)))
            em_j = j.get('em_correct', j.get('exact_match', j.get('is_correct', 0)))
            if em_b == 0 and em_j == 1:
                rescued.add(sid)
    return triggered, rescued


def load_all(erosion_dir, probe_dir, steering_dir, baseline_path, steered_path):
    npz  = np.load(os.path.join(erosion_dir, "raw_erosion_data.npz"))
    meta = [json.loads(l) for l in open(os.path.join(erosion_dir, "raw_erosion_meta.jsonl"))]
    X_p0      = npz["p0_input"].astype(np.float32)
    evid_labs  = np.array([m["evidence_label"] for m in meta])
    sample_ids = [m["sample_id"] for m in meta]

    # BUG FIX: Previously loaded probe_direction_l20.npz (Phase 1 probe) as
    # "steering direction". This was wrong — the actual A3 steering direction
    # is in direction_search_v3_layer20.npz. The old code produced
    # cos(p0_retrained_probe, Phase1_probe)=0.363, which was misinterpreted
    # as cos(evidence_probe, steering_direction).
    sd = np.load(steering_dir)
    steer = sd["decision_direction"].astype(np.float32)
    steer /= np.linalg.norm(steer) + 1e-12
    print(f"  Loaded steering direction from {steering_dir}, norm={np.linalg.norm(sd['decision_direction']):.4f}")

    # Also load Phase 1 probe for cross-reference
    p1 = np.load(os.path.join(probe_dir, "probe_direction_l20.npz"))
    w_phase1 = p1["decision_direction"].astype(np.float32)
    w_phase1 /= np.linalg.norm(w_phase1) + 1e-12
    print(f"  cos(Phase1_probe, steering) = {np.dot(w_phase1, steer):.6f}")

    triggered, rescued = load_a3_triggered_and_rescued(baseline_path, steered_path)
    behav_labs = np.array([1 if sid in rescued else 0 for sid in sample_ids])
    triggered_mask = np.array([sid in triggered for sid in sample_ids])

    return X_p0, evid_labs, behav_labs, triggered_mask, steer, sample_ids


# ── Experiment 1: Behavioral probe LOO-CV ────────────────────────────────────
def behavioral_probe_loo(X, y):
    """LOO-CV AUROC for behavioral label (N_pos very small)."""
    loo = LeaveOneOut()
    n_total = len(y)
    scores = np.zeros(n_total)
    for done, (train_idx, test_idx) in enumerate(loo.split(X), start=1):
        clf = LogisticRegression(max_iter=500, C=1.0, solver="liblinear",
                                  class_weight="balanced", random_state=0)
        clf.fit(X[train_idx], y[train_idx])
        scores[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
        bar = "#" * (done * 40 // n_total)
        print(f"\r  [{bar:<40}] {done}/{n_total}", end="", flush=True)
    print(flush=True)
    auroc = roc_auc_score(y, scores)
    return auroc, scores


def fit_full_probe(X, y):
    """Fit probe on all data, return normalised weight vector."""
    clf = LogisticRegression(max_iter=500, C=1.0, solver="liblinear",
                              class_weight="balanced", random_state=0)
    clf.fit(X, y)
    w = clf.coef_[0].astype(np.float32)
    w /= np.linalg.norm(w) + 1e-12
    return w


# ── Experiment 2: Permutation test for cos(p0_probe, steering) ───────────────
def permutation_test_cos(X, evid_labels, steer, n_perm=1000, seed=0):
    rng = np.random.RandomState(seed)
    # Observed: train evidence probe on p0, measure cos with steer
    w_obs = fit_full_probe(X, evid_labels)
    obs_cos = float(np.dot(w_obs, steer))

    null_cos = []
    for i in range(n_perm):
        perm_y = rng.permutation(evid_labels)
        w_null = fit_full_probe(X, perm_y)
        null_cos.append(float(np.dot(w_null, steer)))
        done = i + 1
        bar = "#" * (done * 40 // n_perm)
        print(f"\r  [{bar:<40}] {done}/{n_perm}", end="", flush=True)
    print(flush=True)

    null_cos = np.array(null_cos)
    # one-sided: how often does |null| >= |obs|?
    p_val = float((np.abs(null_cos) >= np.abs(obs_cos)).mean())
    return obs_cos, null_cos, p_val



# ── Experiment 3: Within-triggered probe + permutation ────────────────────────
def within_triggered_probe(X_trig, y_trig, steer, n_perm=100, seed=0):
    """
    LOO-CV AUROC within triggered subset, plus permutation test for cosine.
    y_trig: 1=rescued, 0=triggered-but-not-rescued
    """
    n_total = len(y_trig)
    n_pos = int(y_trig.sum())
    n_neg = n_total - n_pos
    print(f"  Within-triggered: N={n_total}, N_pos(rescued)={n_pos}, N_neg={n_neg}")

    if n_pos < 3 or n_neg < 3:
        print("  ❌ Too few samples for LOO-CV, skipping")
        return None

    # LOO-CV AUROC
    print("  Running LOO-CV...", flush=True)
    auroc, loo_scores = behavioral_probe_loo(X_trig, y_trig)
    print(f"  AUROC (LOO-CV) = {auroc:.4f}")

    # Full probe and cosine with steering
    w_trig = fit_full_probe(X_trig, y_trig)
    obs_cos = float(np.dot(w_trig, steer))
    print(f"  cos(within_trig_probe, steering) = {obs_cos:+.4f}")

    # Sanity: print first 10 components of both vectors
    print(f"  probe w[0:10] = {w_trig[:10]}")
    print(f"  steer  [0:10] = {steer[:10]}")

    # Permutation test for cosine
    rng = np.random.RandomState(seed)
    null_aurocs = []
    null_cos_vals = []
    print(f"  Permutation test ({n_perm}×)...", flush=True)
    for i in range(n_perm):
        perm_y = rng.permutation(y_trig)
        # LOO-CV for each permutation
        loo = LeaveOneOut()
        scores_perm = np.zeros(n_total)
        for train_idx, test_idx in loo.split(X_trig):
            clf = LogisticRegression(max_iter=500, C=1.0, solver="liblinear",
                                      class_weight="balanced", random_state=0)
            clf.fit(X_trig[train_idx], perm_y[train_idx])
            scores_perm[test_idx] = clf.predict_proba(X_trig[test_idx])[:, 1]
        try:
            null_aurocs.append(roc_auc_score(perm_y, scores_perm))
        except ValueError:
            null_aurocs.append(0.5)

        w_perm = fit_full_probe(X_trig, perm_y)
        null_cos_vals.append(float(np.dot(w_perm, steer)))

        done = i + 1
        bar = "#" * (done * 40 // n_perm)
        print(f"\r  [{bar:<40}] {done}/{n_perm}", end="", flush=True)
    print(flush=True)

    null_aurocs = np.array(null_aurocs)
    null_cos_vals = np.array(null_cos_vals)
    p_auroc = float((null_aurocs >= auroc).mean())
    p_cos = float((np.abs(null_cos_vals) >= np.abs(obs_cos)).mean())

    print(f"  Null AUROC: {null_aurocs.mean():.4f} ± {null_aurocs.std():.4f}, p={p_auroc:.4f}")
    print(f"  Null cos:   {null_cos_vals.mean():+.4f} ± {null_cos_vals.std():.4f}, p={p_cos:.4f}")

    return {
        "n_triggered": n_total, "n_rescued": n_pos, "n_not_rescued": n_neg,
        "auroc_loo_cv": auroc,
        "cos_probe_steering": obs_cos,
        "probe_first10": w_trig[:10].tolist(),
        "steer_first10": steer[:10].tolist(),
        "perm_n": n_perm,
        "perm_auroc_mean": float(null_aurocs.mean()),
        "perm_auroc_std": float(null_aurocs.std()),
        "perm_auroc_p": p_auroc,
        "perm_cos_mean": float(null_cos_vals.mean()),
        "perm_cos_std": float(null_cos_vals.std()),
        "perm_cos_p": p_cos,
        "verdict": ("link_b_closed" if auroc >= 0.65 and p_cos < 0.05
                    else "link_b_auroc_only" if auroc >= 0.65
                    else "link_b_underpowered" if auroc >= 0.55
                    else "link_b_failed"),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--erosion-dir",   default="results/thought_erosion")
    ap.add_argument("--probe-dir",     default="results/phase1_probe")
    ap.add_argument("--steering-dir",  default="steering/directions/direction_search_v3_layer20.npz",
                    help="Path to the actual A3 steering direction .npz file")
    ap.add_argument("--baseline",      default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--steered",       default="results/l20_rho020_n500/v3_L20/jes_tau0.20_mr0.20.jsonl")
    ap.add_argument("--output-dir",    default="results/closure_b_fixed")
    ap.add_argument("--n-perm",        type=int, default=1000)
    ap.add_argument("--n-perm-trig",   type=int, default=100,
                    help="Permutations for within-triggered test (LOO is expensive)")
    ap.add_argument("--seed",          type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading data...", flush=True)
    X_p0, evid_labs, behav_labs, triggered_mask, steer, sids = load_all(
        args.erosion_dir, args.probe_dir, args.steering_dir, args.baseline, args.steered)
    N = len(behav_labs)
    N_pos = int(behav_labs.sum())
    N_trig = int(triggered_mask.sum())
    print(f"  N={N}, N_pos (rescued)={N_pos}, N_neg={N - N_pos}")
    print(f"  N_triggered (in erosion data)={N_trig}")

    # ── Experiment 1: Behavioral probe (full, confounded) ─────────────────────
    print("\n=== Experiment 1: Behavioral Outcome Probe — FULL (LOO-CV) ===", flush=True)
    print("  (Confounded: negatives include ~391 non-triggered samples)", flush=True)
    auroc, loo_scores = behavioral_probe_loo(X_p0, behav_labs)
    print(f"  AUROC (LOO-CV) = {auroc:.4f}")

    behav_probe_cos = None
    if auroc >= 0.70:
        w_behav = fit_full_probe(X_p0, behav_labs)
        behav_probe_cos = float(np.dot(w_behav, steer))
        print(f"  cos(behav_probe, steering) = {behav_probe_cos:+.4f}")

    # ── Experiment 2: Permutation test for cos(evidence_probe, steering) ──────
    print(f"\n=== Experiment 2: Permutation test for cos(p0_probe, steering) ({args.n_perm}×) ===",
          flush=True)
    obs_cos, null_cos, p_cos = permutation_test_cos(
        X_p0, evid_labs, steer, n_perm=args.n_perm, seed=args.seed)

    print(f"\n  Observed cos(p0_probe, steering) = {obs_cos:+.4f}")
    print(f"  Null mean  = {null_cos.mean():+.4f} ± {null_cos.std():.4f}")
    print(f"  p(|null| >= |obs|) = {p_cos:.4f}")
    sig_cos = "YES ✓" if p_cos < 0.05 else "NO ✗"
    print(f"  Significant? {sig_cos}")

    # ── Experiment 3: Within-triggered probe (CLEAN) ──────────────────────────
    print(f"\n=== Experiment 3: Within-Triggered Behavioral Probe (LOO-CV) ===", flush=True)
    print("  Only triggered samples: rescued(1) vs triggered-not-rescued(0)", flush=True)

    X_trig = X_p0[triggered_mask]
    y_trig = behav_labs[triggered_mask]  # 1=rescued, 0=triggered-not-rescued
    exp3 = within_triggered_probe(X_trig, y_trig, steer,
                                   n_perm=args.n_perm_trig, seed=args.seed)

    # ── Save ──────────────────────────────────────────────────────────────────
    result = {
        "experiment1_behavioral_probe_full": {
            "N": N, "N_pos": N_pos,
            "auroc_loo_cv": auroc,
            "behav_probe_cos_steering": behav_probe_cos,
            "note": "Confounded — negatives include non-triggered samples",
        },
        "experiment2_permutation_cos": {
            "n_perm": args.n_perm,
            "observed_cos": obs_cos,
            "null_mean": float(null_cos.mean()),
            "null_std": float(null_cos.std()),
            "p_value": p_cos,
        },
        "experiment3_within_triggered": exp3,
    }
    out = os.path.join(args.output_dir, "closure_b_results.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out}")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("CLOSURE-B RESULTS SUMMARY")
    print("="*70)
    print(f"  Exp1 Full AUROC    = {auroc:.4f}  (confounded)")
    print(f"  Exp2 cos p-value   = {p_cos:.4f}  ({'<0.05 ✓' if p_cos<0.05 else '≥0.05 ✗'})")
    if exp3:
        print(f"  Exp3 Within-trig AUROC = {exp3['auroc_loo_cv']:.4f}  "
              f"({'≥0.65 ✓' if exp3['auroc_loo_cv']>=0.65 else '<0.65 ✗'})")
        print(f"  Exp3 cos(probe,steer)  = {exp3['cos_probe_steering']:+.4f}  "
              f"(perm p={exp3['perm_cos_p']:.4f})")
        print(f"  Exp3 verdict: {exp3['verdict']}")
    print("="*70)


if __name__ == "__main__":
    main()
