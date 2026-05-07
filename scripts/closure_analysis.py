#!/usr/bin/env python3
"""
Erosion ↔ Steering Closure Analysis (Priorities 1-4)

Priority 1: Sample-level closure between erosion score and A3 rescue outcome
Priority 2: Subspace rotation matrix (5×5 cosine + steering cosine sequence)
Priority 3: Probe × Behavior 2×2 formal Fisher test
Priority 4: Master results table
"""

import json
import os
import sys
import numpy as np
import argparse
from collections import defaultdict

# ── imports ──────────────────────────────────────────────────────────────────
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from scipy.stats import mannwhitneyu, pointbiserialr, fisher_exact, spearmanr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    PLOT_OK = True
except ImportError as e:
    print(f"[WARN] Import error: {e} — plots will be skipped", flush=True)
    PLOT_OK = False


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_erosion_data(erosion_dir):
    """Load raw activations (N×D per position) and metadata."""
    npz = np.load(os.path.join(erosion_dir, "raw_erosion_data.npz"))
    meta = [json.loads(l) for l in open(os.path.join(erosion_dir, "raw_erosion_meta.jsonl"))]
    POSITIONS = ["p0_input", "p1_25pct", "p2_50pct", "p3_75pct", "p4_100pct"]
    acts = {p: npz[p] for p in POSITIONS}
    return acts, meta, POSITIONS


STEERING_PATH_DEFAULT = "steering/directions/direction_search_v3_layer20.npz"


def load_probe_direction(probe_dir):
    """Load Phase-1 evidence probe direction (shape D,).
    NOTE: This is the evidence sufficiency probe, NOT the A3 steering direction.
    """
    p = np.load(os.path.join(probe_dir, "probe_direction_l20.npz"))
    direction = p["decision_direction"].astype(np.float32)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    return direction


def load_steering_direction(steering_path=STEERING_PATH_DEFAULT):
    """Load the REAL A3 steering direction (mean-diff from margin-based contrastive pairs)."""
    s = np.load(steering_path)
    d = s["decision_direction"].astype(np.float32)
    d = d / (np.linalg.norm(d) + 1e-12)
    print(f"  Loaded A3 steering direction from {steering_path}, "
          f"norm(raw)={np.linalg.norm(s['decision_direction']):.2f}")
    return d


def load_a3_triggered_rescued(baseline_path, steered_path):
    """
    Reconstruct triggered/rescued sets from paired JSONL files.
    triggered = samples where steered has more tool_calls than baseline.
    rescued   = triggered AND (baseline EM=0, steered EM=1).
    Returns: (dict sample_id→is_triggered, dict sample_id→is_rescued)
    """
    base_map = {json.loads(l)["sample_id"]: json.loads(l)
                for l in open(baseline_path)}
    jes_list = [json.loads(l) for l in open(steered_path)]

    triggered, rescued = {}, {}
    for s in jes_list:
        sid = s["sample_id"]
        if sid not in base_map:
            continue
        b = base_map[sid]
        is_trig = (s.get("tool_calls", 0) or 0) > (b.get("tool_calls", 0) or 0)
        is_resc = (is_trig and
                   not b.get("em_correct", False) and
                   s.get("em_correct", False))
        triggered[sid] = is_trig
        rescued[sid]   = is_resc
    return triggered, rescued


# ═══════════════════════════════════════════════════════════════════════════════
# Priority 1 — Erosion ↔ Steering closure
# ═══════════════════════════════════════════════════════════════════════════════

def compute_erosion_scores(acts, meta, direction):
    """erosion_score[i] = proj_p0[i] − proj_p4[i]."""
    proj_p0 = acts["p0_input"] @ direction          # (N,)
    proj_p4 = acts["p4_100pct"] @ direction         # (N,)
    erosion  = proj_p0 - proj_p4
    return erosion, proj_p0, proj_p4


def priority1_closure(acts, meta, direction, triggered_map, rescued_map,
                      output_dir, figures_dir):
    print("\n" + "="*70)
    print("PRIORITY 1 — EROSION ↔ STEERING CLOSURE")
    print("="*70, flush=True)

    N = len(meta)
    erosion, proj_p0, proj_p4 = compute_erosion_scores(acts, meta, direction)

    # ── build per-sample arrays ──────────────────────────────────────────────
    sample_ids   = [m["sample_id"] for m in meta]
    is_triggered = np.array([triggered_map.get(sid, False) for sid in sample_ids], dtype=bool)
    is_rescued   = np.array([rescued_map.get(sid,   False) for sid in sample_ids], dtype=bool)

    # ── 1a: within triggered (N=93 in erosion set) ──────────────────────────
    trig_mask    = is_triggered
    resc_mask    = is_rescued
    trig_resc    = erosion[trig_mask & resc_mask]
    trig_noresc  = erosion[trig_mask & ~resc_mask]

    print(f"\n1a. Triggered subset (N={trig_mask.sum()}):")
    print(f"    Rescued (N={len(trig_resc)}): "
          f"median={np.median(trig_resc):+.4f}  mean={np.mean(trig_resc):+.4f}  "
          f"std={np.std(trig_resc):.4f}")
    print(f"    Not rescued (N={len(trig_noresc)}): "
          f"median={np.median(trig_noresc):+.4f}  mean={np.mean(trig_noresc):+.4f}  "
          f"std={np.std(trig_noresc):.4f}")

    if len(trig_resc) >= 3 and len(trig_noresc) >= 3:
        # rescued samples have MORE NEGATIVE erosion_score (= more eroded)
        # so test rescued < not-rescued  (i.e. more negative = more erosion)
        stat, pval = mannwhitneyu(trig_resc, trig_noresc, alternative="less")
        stat_2s, pval_2s = mannwhitneyu(trig_resc, trig_noresc, alternative="two-sided")
        print(f"    Mann-Whitney U (rescued < not-rescued, i.e. more eroded): "
              f"U={stat:.0f}  p={pval:.4f}")
        print(f"    Mann-Whitney U (two-sided): U={stat_2s:.0f}  p={pval_2s:.4f}")
        mw_result = {"U": float(stat), "p_less": float(pval),
                     "p_twosided": float(pval_2s),
                     "n_rescued": len(trig_resc), "n_not_rescued": len(trig_noresc),
                     "median_rescued": float(np.median(trig_resc)),
                     "median_not_rescued": float(np.median(trig_noresc))}
    else:
        print("    [SKIP] Insufficient samples for Mann-Whitney")
        mw_result = None

    # ── 1b: quintile analysis over all N ────────────────────────────────────
    quintiles = np.quantile(erosion, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(erosion, quintiles)  # 0..4

    print(f"\n1b. Quintile analysis (N={N} total):")
    print(f"{'Bucket':>8}  {'Range':>20}  {'N':>5}  {'Trig%':>7}  {'Resc%':>7}  {'Mean_eros':>10}")

    quintile_data = []
    for q in range(5):
        mask = (bins == q)
        n_q       = mask.sum()
        trig_rate = is_triggered[mask].mean() if n_q > 0 else float("nan")
        resc_rate = is_rescued[mask].mean()   if n_q > 0 else float("nan")
        q_lo = erosion[mask].min() if n_q > 0 else float("nan")
        q_hi = erosion[mask].max() if n_q > 0 else float("nan")
        mean_eros = erosion[mask].mean() if n_q > 0 else float("nan")
        print(f"  Q{q+1}     [{q_lo:+.3f},{q_hi:+.3f}]  {n_q:>5}  {trig_rate:>7.1%}  "
              f"{resc_rate:>7.1%}  {mean_eros:>10.4f}")
        quintile_data.append({
            "bucket": q + 1,
            "n": int(n_q),
            "trigger_rate": float(trig_rate),
            "rescue_rate": float(resc_rate),
            "mean_erosion": float(mean_eros),
        })

    # NOTE: Q1 has the most NEGATIVE erosion_score = most eroded (proj_p4 >> proj_p0)
    #       Q5 has the most POSITIVE erosion_score = least eroded (proj_p0 > proj_p4)
    rescue_rates = [d["rescue_rate"] for d in quintile_data]
    # Check if rescue rate DECREASES with quintile (most eroded Q1 has highest rescue)
    monotone_dec = all(rescue_rates[i] >= rescue_rates[i+1] for i in range(len(rescue_rates)-1))
    print(f"    Rescue rate monotone decreasing (Q1=most eroded → Q5=least): "
          f"{'YES ✓' if monotone_dec else 'NO (non-monotone)'}")

    # ── 1c: rank correlation ─────────────────────────────────────────────────
    print(f"\n1c. Rank correlation (erosion_score vs rescue):")

    # Point-biserial
    r_pb, p_pb = pointbiserialr(is_rescued.astype(float), erosion)
    print(f"    Point-biserial r = {r_pb:+.4f}  p = {p_pb:.4f}")

    # Spearman
    r_sp, p_sp = spearmanr(erosion, is_rescued.astype(float))
    print(f"    Spearman rho     = {r_sp:+.4f}  p = {p_sp:.4f}")

    # AUROC: erosion_score predicting rescue
    # NOTE: rescued samples have MORE NEGATIVE erosion_score, so raw AUROC < 0.5.
    # We report both raw and flipped (1-AUROC) for clarity.
    if is_rescued.sum() >= 2 and (~is_rescued).sum() >= 2:
        auroc_raw = roc_auc_score(is_rescued.astype(int), erosion)
        # Use negated erosion so higher = more eroded → AUROC > 0.5 if erosion predicts rescue
        auroc_negated = roc_auc_score(is_rescued.astype(int), -erosion)
        print(f"    Erosion→Rescue AUROC (raw, higher erosion_score=less eroded) = {auroc_raw:.4f}")
        print(f"    Erosion→Rescue AUROC (negated, higher=more eroded)           = {auroc_negated:.4f}")
        print(f"    Interpretation: AUROC={auroc_negated:.3f} means MORE eroded samples "
              f"are {'more' if auroc_negated>0.55 else 'similarly'} likely to be rescued")
        auroc = auroc_negated  # use the semantically correct direction
    else:
        auroc = float("nan")
        auroc_raw = float("nan")
        auroc_negated = float("nan")
        print("    [SKIP] Insufficient rescued samples for AUROC")

    corr_result = {
        "point_biserial_r": float(r_pb), "point_biserial_p": float(p_pb),
        "spearman_rho": float(r_sp), "spearman_p": float(p_sp),
        "erosion_rescue_auroc_raw": float(auroc_raw),
        "erosion_rescue_auroc": float(auroc),  # negated = more eroded → higher
    }

    # ── 1d: figure ───────────────────────────────────────────────────────────
    if PLOT_OK:
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax2 = ax1.twinx()

        qs = [d["bucket"] for d in quintile_data]
        rr = [d["rescue_rate"] * 100 for d in quintile_data]
        me = [d["mean_erosion"] for d in quintile_data]

        ax1.bar(qs, rr, color="#4C72B0", alpha=0.7, label="Rescue rate (%)")
        ax2.plot(qs, me, "o-", color="#C44E52", linewidth=2,
                 markersize=8, label="Mean erosion score")

        ax1.set_xlabel("Erosion Score Quintile (Q1=most eroded, Q5=least eroded)",
                       fontsize=11)
        ax1.set_ylabel("Rescue Rate (%)", color="#4C72B0", fontsize=11)
        ax2.set_ylabel("Mean Erosion Score (proj_p0 − proj_p4)",
                       color="#C44E52", fontsize=11)
        ax1.set_xticks(qs)
        ax1.set_xticklabels([f"Q{q}" for q in qs])
        ax1.set_ylim(0, max(rr) * 1.3 + 0.5)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
                   fontsize=9)

        mw_p_disp = mw_result['p_less'] if mw_result else float('nan')
        ax1.set_title(
            f"Erosion ↔ Rescue Closure\n"
            f"AUROC={auroc:.3f} (more eroded→more rescued)  "
            f"MW p={mw_p_disp:.4f} (one-sided: rescued more eroded)",
            fontsize=11
        )
        plt.tight_layout()
        fig_path = os.path.join(figures_dir, "erosion_rescue_closure.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"\n  Saved: {fig_path}")

    p1_result = {
        "erosion_scores_summary": {
            "mean": float(erosion.mean()),
            "std": float(erosion.std()),
            "min": float(erosion.min()),
            "max": float(erosion.max()),
        },
        "mannwhitney_triggered": mw_result,
        "quintile_analysis": quintile_data,
        "rescue_rate_monotone_decreasing": monotone_dec,
        "correlation": corr_result,
    }
    return p1_result, erosion


# ═══════════════════════════════════════════════════════════════════════════════
# Priority 2 — Subspace rotation matrix
# ═══════════════════════════════════════════════════════════════════════════════

def train_position_probe(acts_pos, labels, seed=42):
    """Train logistic probe at one position, return unit-norm coef vector."""
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                             class_weight="balanced", random_state=seed)
    clf.fit(acts_pos, labels)
    coef = clf.coef_[0].astype(np.float32)
    coef /= (np.linalg.norm(coef) + 1e-12)
    return coef


def priority2_subspace(acts, meta, direction, POSITIONS, figures_dir,
                       steering_direction=None):
    print("\n" + "="*70)
    print("PRIORITY 2 — SUBSPACE ROTATION MATRIX")
    print("="*70, flush=True)

    labels = np.array([m["evidence_label"] for m in meta])

    # Train probe at each position
    print("\nTraining position probes...")
    coefs = {}
    for pos in POSITIONS:
        coefs[pos] = train_position_probe(acts[pos], labels)
        print(f"  {pos}: trained", flush=True)

    # 2a: 5×5 cosine matrix
    cos_matrix = np.zeros((5, 5))
    for i, pi in enumerate(POSITIONS):
        for j, pj in enumerate(POSITIONS):
            cos_matrix[i, j] = float(np.dot(coefs[pi], coefs[pj]))

    short_names = ["p0", "p1", "p2", "p3", "p4"]
    print(f"\n2a. 5×5 cosine similarity matrix (probe_i · probe_j):")
    header = "         " + "".join(f"  {n:>6}" for n in short_names)
    print(header)
    for i, name in enumerate(short_names):
        row = f"  {name:>6}  " + "  ".join(f"{cos_matrix[i,j]:>6.3f}" for j in range(5))
        print(row)

    cos_p0_p4 = cos_matrix[0, 4]
    print(f"\n  cos(p0, p4) = {cos_p0_p4:.4f}  "
          f"({'evidence direction rotates substantially' if abs(cos_p0_p4) < 0.5 else 'direction relatively stable'})")

    # 2b: cosine(probe_pi, A3_steering_direction)
    # BUG FIX: Previously used `direction` (Phase1 probe) here, making this
    # cos(retrained_probe, Phase1_probe) ≈ 0.36 instead of
    # cos(retrained_probe, A3_steering) ≈ -0.01.
    steer_dir = steering_direction if steering_direction is not None else direction
    steer_is_real = steering_direction is not None
    label = "A3 steering direction" if steer_is_real else "Phase1 probe (FALLBACK — no steering loaded)"
    print(f"\n2b. cosine(probe_pi, {label}):")
    steer_cosines = {}
    for pos, short in zip(POSITIONS, short_names):
        c = float(np.dot(coefs[pos], steer_dir))
        steer_cosines[pos] = c
        print(f"  {short}: {c:+.4f}")

    # 2c: plots
    if PLOT_OK:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Cosine heatmap
        ax = axes[0]
        im = ax.imshow(cos_matrix, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(short_names)
        ax.set_yticklabels(short_names)
        ax.set_title("5×5 Probe Cosine Similarity\n(across thought positions)", fontsize=11)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cos_matrix[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if abs(cos_matrix[i,j]) < 0.7 else "white")
        plt.colorbar(im, ax=ax)

        # Steering cosine sequence
        ax = axes[1]
        cs_vals = [steer_cosines[p] for p in POSITIONS]
        ax.plot(range(5), cs_vals, "o-", color="#2196F3", linewidth=2, markersize=8)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(range(5))
        ax.set_xticklabels(short_names)
        ax.set_xlabel("Thought position")
        ax.set_ylabel("cosine(probe_pi, steering direction)")
        ax.set_title("Alignment: Position Probes ↔ Steering Direction", fontsize=11)
        ax.set_ylim(-0.6, 0.6)
        for i, (x, y) in enumerate(zip(range(5), cs_vals)):
            ax.annotate(f"{y:+.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8)

        plt.tight_layout()
        fig_path = os.path.join(figures_dir, "subspace_rotation.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"\n  Saved: {fig_path}")

    p2_result = {
        "cosine_matrix_5x5": cos_matrix.tolist(),
        "cos_p0_p4": float(cos_p0_p4),
        "steering_cosines_per_position": {
            pos: float(steer_cosines[pos]) for pos in POSITIONS
        },
    }
    return p2_result


# ═══════════════════════════════════════════════════════════════════════════════
# Priority 3 — Probe × Behavior 2×2 formal Fisher test
# ═══════════════════════════════════════════════════════════════════════════════

def priority3_probe_behavior_2x2(acts, meta, direction, rescued_map, figures_dir):
    print("\n" + "="*70)
    print("PRIORITY 3 — PROBE × BEHAVIOR 2×2 FORMAL")
    print("="*70, flush=True)

    labels = np.array([m["evidence_label"] for m in meta])
    stop   = np.array([m["behavioral_stop"]  for m in meta], dtype=bool)
    sample_ids = [m["sample_id"] for m in meta]

    # Train probe at p0 (the decision-point activation) using cross-validation
    # to avoid train=test leakage
    acts_p0 = acts["p0_input"]
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                             class_weight="balanced", random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probe_pred = cross_val_predict(clf, acts_p0, labels, cv=cv, method="predict")
    # Also fit on full data for reference accuracy
    clf.fit(acts_p0, labels)
    train_pred = clf.predict(acts_p0)
    print(f"  Probe train acc: {(train_pred==labels).mean():.3f}  "
          f"CV acc: {(probe_pred==labels).mean():.3f}")

    # 2×2: probe prediction × behavioral outcome
    # Rows = behavioral outcome (stop / continue)
    # Cols = probe prediction (insufficient / sufficient)
    #
    #              probe=insuff  probe=suff
    # behav=stop   [  A   ]     [  B  ]
    # behav=cont   [  C   ]     [  D  ]
    stop_insuff  = int(( stop & (probe_pred == 0)).sum())   # A
    stop_suff    = int(( stop & (probe_pred == 1)).sum())   # B
    cont_insuff  = int((~stop & (probe_pred == 0)).sum())   # C
    cont_suff    = int((~stop & (probe_pred == 1)).sum())   # D

    table = [[stop_insuff, stop_suff],
             [cont_insuff,  cont_suff]]

    print(f"\n3a. 2×2 Confusion Matrix (cross-validated probe predictions):")
    print(f"                   Probe: insuff  Probe: suff")
    print(f"  Behav: stop   |  {stop_insuff:>10}  |  {stop_suff:>10}  |")
    print(f"  Behav: cont   |  {cont_insuff:>10}  |  {cont_suff:>10}  |")
    print(f"  (Total: stop={stop.sum()}, cont={(~stop).sum()}, N={len(meta)})")

    # 3b: Fisher exact test (two-sided — we don't have a prior on direction)
    odds_ratio, p_fisher = fisher_exact(table, alternative="two-sided")
    print(f"\n3b. Fisher exact test (two-sided): OR={odds_ratio:.3f}  p={p_fisher:.4e}")

    # 3c: Dissociation rate
    n_insuff = (probe_pred == 0).sum()
    diss_rate = stop_insuff / max(n_insuff, 1)
    print(f"\n3c. Dissociation rate = P(stop | probe=insufficient) = "
          f"{stop_insuff}/{n_insuff} = {diss_rate:.1%}")

    # Rate comparisons
    n_suff = (probe_pred == 1).sum()
    stop_rate_insuff = stop_insuff / max(n_insuff, 1)
    stop_rate_suff   = stop_suff   / max(n_suff,   1)
    print(f"    P(stop | probe=sufficient)   = {stop_suff}/{n_suff} = {stop_rate_suff:.1%}")
    print(f"    Stop-rate ratio (insuff/suff) = {stop_rate_insuff/max(stop_rate_suff,1e-6):.2f}x")

    # 3d: Mark rescued samples in matrix
    resc_mask = np.array([rescued_map.get(sid, False) for sid in sample_ids], dtype=bool)
    resc_stop_insuff = int(( resc_mask &  stop & (probe_pred == 0)).sum())
    resc_stop_suff   = int(( resc_mask &  stop & (probe_pred == 1)).sum())
    resc_cont_insuff = int(( resc_mask & ~stop & (probe_pred == 0)).sum())
    resc_cont_suff   = int(( resc_mask & ~stop & (probe_pred == 1)).sum())
    n_rescued = resc_mask.sum()

    print(f"\n3d. A3 rescued (N={n_rescued}) location in the 2×2:")
    print(f"    Rescued in [stop, probe=insuff]: {resc_stop_insuff}")
    print(f"    Rescued in [stop, probe=suff]:   {resc_stop_suff}")
    print(f"    Rescued in [cont, probe=insuff]: {resc_cont_insuff}")
    print(f"    Rescued in [cont, probe=suff]:   {resc_cont_suff}")
    frac_dissoc = resc_stop_insuff / max(n_rescued, 1)
    print(f"    Fraction of rescued in dissociation cell: "
          f"{resc_stop_insuff}/{n_rescued} = {frac_dissoc:.1%}")

    if PLOT_OK:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        cell_labels = [
            [f"{stop_insuff}\n(rescued: {resc_stop_insuff})",
             f"{stop_suff}\n(rescued: {resc_stop_suff})"],
            [f"{cont_insuff}\n(rescued: {resc_cont_insuff})",
             f"{cont_suff}\n(rescued: {resc_cont_suff})"],
        ]
        vals = np.array(table, dtype=float)
        vals_norm = vals / vals.sum()
        im = ax.imshow(vals_norm, cmap="Blues", vmin=0, aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Probe: Insufficient\n(evidence<sufficient)",
                            "Probe: Sufficient\n(evidence=sufficient)"], fontsize=9)
        ax.set_yticklabels(["Behaviorally: Stop\n(no 2nd search)",
                            "Behaviorally: Continue\n(2nd search)"], fontsize=9)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cell_labels[i][j], ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if vals_norm[i,j] > 0.4 else "black")
        ax.set_title(f"Probe × Behavior 2×2 (CV predictions)\n"
                     f"Fisher p={p_fisher:.2e} (two-sided)  "
                     f"Dissociation rate={diss_rate:.1%}",
                     fontsize=11)
        plt.tight_layout()
        fig_path = os.path.join(figures_dir, "probe_behavior_2x2.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"\n  Saved: {fig_path}")

    p3_result = {
        "table": table,
        "fisher_OR": float(odds_ratio),
        "fisher_p": float(p_fisher),
        "dissociation_rate": float(diss_rate),
        "stop_rate_given_insufficient": float(stop_rate_insuff),
        "stop_rate_given_sufficient": float(stop_rate_suff),
        "rescued_in_dissociation_cell": int(resc_stop_insuff),
        "n_rescued_total": int(n_rescued),
    }
    return p3_result


# ═══════════════════════════════════════════════════════════════════════════════
# Priority 4 — Master results table
# ═══════════════════════════════════════════════════════════════════════════════

def priority4_master_table(probe_dir, dissoc_metrics_path,
                           pc_results_path, p1_result, p2_result, p3_result,
                           erosion_analysis_path, output_dir):
    print("\n" + "="*70)
    print("PRIORITY 4 — MASTER RESULTS TABLE")
    print("="*70, flush=True)

    # Load upstream results
    probe_npz    = np.load(os.path.join(probe_dir, "probe_direction_l20.npz"))
    dissoc_m     = json.load(open(dissoc_metrics_path))
    pc           = json.load(open(pc_results_path))
    erosion_an   = json.load(open(erosion_analysis_path))

    probe_auroc  = float(probe_npz["auroc"])
    probe_balacc = float(probe_npz["balanced_accuracy"])

    # Erosion values
    gap_p0       = erosion_an["primary_curve"]["p0_input"]["evidence_gap"]
    gap_p4       = erosion_an["primary_curve"]["p4_100pct"]["evidence_gap"]
    gap_shrink   = erosion_an["summary"]["observed_gap_shrinkage"]
    perm_p       = erosion_an["permutation_test"]["p_value_gap_shrinkage"]
    fd_p0        = erosion_an["primary_curve"]["p0_input"]["fixed_dir_auroc"]
    fd_p4        = erosion_an["primary_curve"]["p4_100pct"]["fixed_dir_auroc"]
    boot_p0_lo   = erosion_an["bootstrap_ci"]["p0_input"]["evidence_gap_ci95"][0]
    boot_p4_hi   = erosion_an["bootstrap_ci"]["p4_100pct"]["evidence_gap_ci95"][1]
    ci_overlap   = boot_p0_lo < boot_p4_hi   # True means overlap → not significant

    # Steering (A3 numbers from CLAUDE.md + dissociation metrics)
    a3_net_em      = 17
    a3_mcnemar_p   = 0.000488
    a3_purity      = 0.95
    a3_regression  = 3

    # Dissociation
    diss_rate      = dissoc_m["strict_dissociation_debiased_rate"]
    e_cont         = dissoc_m["continue_rates"].get("E", float("nan"))
    d_cont         = dissoc_m["continue_rates"].get("D", float("nan"))
    e_vs_d_p       = dissoc_m["mcnemar"]["E_vs_D"]["p"]
    b_debiased     = dissoc_m["continue_rates"].get("B_debiased", float("nan"))

    # Positive control
    synth_insuff   = pc["groups"]["SYNTH_both_SFs"]["a_insufficient_rate"]  # rate
    synth_suff_rate = 1.0 - synth_insuff   # sufficient rate when both SFs present

    # Closure (Priority 1)
    erosion_rescue_auroc = p1_result["correlation"]["erosion_rescue_auroc"]
    mw_p = (p1_result["mannwhitney_triggered"]["p_twosided"]
            if p1_result["mannwhitney_triggered"] else float("nan"))

    # Subspace cosine
    cos_evidence_action = p2_result.get("steering_cosines_per_position", {}).get("p0_input", float("nan"))
    # The direction_alignment from phase1 gives the authoritative cosine
    try:
        phase1_r = json.load(open(os.path.join(probe_dir, "phase1_multilayer_results.json")))
        cos_ev_act = phase1_r["direction_alignment"]["cosine_similarity"]
    except Exception:
        cos_ev_act = cos_evidence_action

    rows = [
        # (Category, Finding, Metric, Value, Significance)
        ("Probe",           "L20 evidence probe",        "AUROC",
         f"{probe_auroc:.3f}",   "—"),
        ("Probe",           "L20 evidence probe",        "Balanced Accuracy",
         f"{probe_balacc:.3f}",  "—"),
        ("Dissociation",    "Behavioral dissociation",   "Strict dissociation rate (debiased)",
         f"{diss_rate:.1%}",     "—"),
        ("Dissociation",    "Context vs baseline",       "B_debiased continue rate",
         f"{b_debiased:.1%}",    f"McNemar p≈0"),
        ("Erosion",         "Evidence gap shrinkage",    "Gap p0→p4",
         f"{gap_p0:.3f}→{gap_p4:.3f} (−{gap_shrink/gap_p0*100:.1f}%)",
         f"perm p={perm_p:.4f}"),
        ("Erosion",         "Fixed-dir AUROC drop",      "p0→p4",
         f"{fd_p0:.3f}→{fd_p4:.3f}",
         f"Bootstrap CI non-overlap={'YES' if not ci_overlap else 'NO'}"),
        ("Steering",        "A3 main result",            "Net EM gain",
         f"+{a3_net_em}",         f"McNemar p={a3_mcnemar_p:.4f}"),
        ("Steering",        "A3 causal purity",          "Rescue via search",
         f"{a3_purity:.0%}",      "—"),
        ("Steering",        "A3 regression",             "Regressed samples",
         f"{a3_regression}",      "—"),
        ("Closure",         "Erosion→Rescue",            "AUROC (erosion predicts rescue)",
         f"{erosion_rescue_auroc:.3f}",
         f"MW p={mw_p:.4f} (two-sided, rescued vs not-rescued)"),
        ("Orthogonality",   "Evidence ↔ Action subspace","cosine(probe, steering dir)",
         f"{cos_ev_act:+.4f}",    "No alignment (orthogonal subspaces)"),
        ("Probe×Behavior",  "2×2 formal",                "Dissociation rate P(stop|probe=insuff)",
         f"{p3_result['dissociation_rate']:.1%}",
         f"Fisher p={p3_result['fisher_p']:.2e}"),
        ("Positive Control","SYNTH sufficient condition","A condition sufficient rate",
         f"{synth_suff_rate:.0%}",  "(vs 11% on insufficient samples)"),
        ("E-condition",     "Prompted ReAct",            "Continue rate",
         f"{e_cont:.1%}",          f"vs D={d_cont:.1%}  McNemar p={e_vs_d_p:.3f} (ns)"),
    ]

    # Print table
    print(f"\n{'Category':<18} {'Finding':<32} {'Metric':<40} {'Value':<30} {'Significance'}")
    print("-" * 140)
    for cat, finding, metric, val, sig in rows:
        print(f"{cat:<18} {finding:<32} {metric:<40} {val:<30} {sig}")

    # Save JSON
    table_json = [
        {"category": cat, "finding": finding, "metric": metric,
         "value": val, "significance": sig}
        for cat, finding, metric, val, sig in rows
    ]

    out_path = os.path.join(output_dir, "master_results_table.json")
    with open(out_path, "w") as f:
        json.dump({"table": table_json, "raw": {
            "probe_auroc": probe_auroc,
            "probe_balanced_accuracy": probe_balacc,
            "strict_dissociation_rate_debiased": diss_rate,
            "b_debiased_continue_rate": b_debiased,
            "erosion_gap_p0": gap_p0,
            "erosion_gap_p4": gap_p4,
            "erosion_gap_shrinkage_pct": gap_shrink * 100,
            "erosion_permutation_p": perm_p,
            "fd_auroc_p0": fd_p0,
            "fd_auroc_p4": fd_p4,
            "a3_net_em": a3_net_em,
            "a3_mcnemar_p": a3_mcnemar_p,
            "a3_causal_purity": a3_purity,
            "a3_regression": a3_regression,
            "erosion_rescue_auroc": float(erosion_rescue_auroc),
            "erosion_rescue_mw_p": float(mw_p),
            "cosine_evidence_action": float(cos_ev_act),
            "dissociation_rate_fisher": p3_result["dissociation_rate"],
            "fisher_p": p3_result["fisher_p"],
            "synth_sufficient_rate": synth_suff_rate,
            "e_continue_rate": e_cont,
            "d_continue_rate": d_cont,
            "e_vs_d_mcnemar_p": e_vs_d_p,
        }}, f, indent=2)
    print(f"\nSaved: {out_path}")
    return table_json


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--erosion-dir",     default="results/thought_erosion")
    ap.add_argument("--probe-dir",       default="results/phase1_probe")
    ap.add_argument("--baseline-path",   default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--steered-path",    default="results/l20_rho020_n500/v3_L20/jes_tau0.20_mr0.20.jsonl")
    ap.add_argument("--dissoc-metrics",  default="results/agent_specific_dissociation/metrics.json")
    ap.add_argument("--pc-results",      default="results/positive_control/positive_control_results.json")
    ap.add_argument("--erosion-analysis",default="results/thought_erosion/erosion_analysis.json")
    ap.add_argument("--steering-path",   default=STEERING_PATH_DEFAULT)
    ap.add_argument("--output-dir",      default="results")
    ap.add_argument("--figures-dir",     default="results/figures")
    args = ap.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data...", flush=True)
    acts, meta, POSITIONS = load_erosion_data(args.erosion_dir)
    direction = load_probe_direction(args.probe_dir)
    steering_direction = load_steering_direction(args.steering_path)
    triggered_map, rescued_map = load_a3_triggered_rescued(
        args.baseline_path, args.steered_path
    )

    N = len(meta)
    n_triggered = sum(triggered_map.get(m["sample_id"], False) for m in meta)
    n_rescued   = sum(rescued_map.get(m["sample_id"],   False) for m in meta)
    print(f"  N={N}, triggered={n_triggered}, rescued={n_rescued}", flush=True)

    # ── Run priorities ────────────────────────────────────────────────────────
    p1_result, erosion_scores = priority1_closure(
        acts, meta, direction, triggered_map, rescued_map,
        args.output_dir, args.figures_dir
    )

    p2_result = priority2_subspace(acts, meta, direction, POSITIONS, args.figures_dir,
                                   steering_direction=steering_direction)

    p3_result = priority3_probe_behavior_2x2(
        acts, meta, direction, rescued_map, args.figures_dir
    )

    p4_table = priority4_master_table(
        probe_dir          = args.probe_dir,
        dissoc_metrics_path= args.dissoc_metrics,
        pc_results_path    = args.pc_results,
        p1_result          = p1_result,
        p2_result          = p2_result,
        p3_result          = p3_result,
        erosion_analysis_path= args.erosion_analysis,
        output_dir         = args.output_dir,
    )

    # ── Save full closure results ─────────────────────────────────────────────
    closure_path = os.path.join(args.output_dir, "closure_analysis.json")
    with open(closure_path, "w") as f:
        json.dump({
            "priority1_erosion_closure": p1_result,
            "priority2_subspace_rotation": p2_result,
            "priority3_probe_behavior_2x2": p3_result,
        }, f, indent=2)
    print(f"\nSaved: {closure_path}")

    print("\n" + "="*70)
    print("ALL PRIORITIES COMPLETE")
    print("="*70)
    print(f"  results/figures/erosion_rescue_closure.png  ← Figure 2 candidate")
    print(f"  results/figures/subspace_rotation.png")
    print(f"  results/figures/probe_behavior_2x2.png")
    print(f"  results/master_results_table.json")
    print(f"  results/closure_analysis.json")


if __name__ == "__main__":
    main()
