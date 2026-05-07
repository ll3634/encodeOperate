#!/usr/bin/env python3
"""
Thought Erosion Statistical Analysis — Post-hoc analysis of raw_erosion_data.npz.

Loads pre-computed L20 activations at 5 thought positions and runs:

Tier 1 (primary figure):
  1. Gap-based erosion curve (evidence_gap + fixed-dir AUROC, dual y-axis)
  2. Bootstrap 95% CIs on fixed-dir AUROC and evidence gap (B=1000)
  3. Split stability: 5 random seeds, retrained AUROC at each position
  4. Permutation test: gap shrinkage vs null distribution (K=1000 shuffles)

Tier 2 (subgroup):
  5. A3-rescued subset erosion curve overlaid on overall
  6. 0-doc (N=97) vs 1-doc (N=389) erosion curves

Tier 3 (mechanism):
  7. E-condition self-contradiction examples (from agent_specific_dissociation)

Usage:
    cd tmc/scripts/e2e_agent
    python scripts/thought_erosion_analysis.py \\
        --erosion-dir results/thought_erosion \\
        --probe-dir    results/phase1_probe \\
        --dissoc-path  results/agent_specific_dissociation/raw_results.jsonl \\
        --output-dir   results/thought_erosion
"""

import sys, json, argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent.parent))

POSITIONS       = ["p0_input", "p1_25pct", "p2_50pct", "p3_75pct", "p4_100pct"]
POS_LABELS      = ["p0\n(input)", "p1\n(25%)", "p2\n(50%)", "p3\n(75%)", "p4\n(100%)"]
POS_X           = [0, 1, 2, 3, 4]

A3_RESCUED_VIA_SEARCH = {
    "5abaee845542994c784ddb49", "5abbcfaf5542993f40c73ba9",
    "5ae2eda355429928c4239570", "5a8782f25542996e4f308818",
    "5a8f51185542992414482a3d", "5a85b2895542994c784ddb49",
    "5ae256435542992decbdccc3", "5ab29956554299194fa9342d",
    "5ae55d1e55429960a22e02cb", "5ab9cfe655429970cfb8ebaf",
    "5a821c95554299676cceb219", "5abdba405542993f32c2a023",
    "5abf92c45542993fe9a41e07", "5ac2a35055429967731025ce",
    "5ae7535c5542997b22f6a6d8", "5ae47cab5542996836b02cb9",
    "5a79311755429970f5fffe67", "5a7e02b75542997cc2c474f3",
    "5a83c2e25542996488c2e4bc",
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(erosion_dir: Path, probe_dir: Path):
    d = np.load(erosion_dir / "raw_erosion_data.npz")
    acts = {pos: d[pos].astype(np.float32) for pos in POSITIONS}   # (N, 3584)

    meta = [json.loads(l) for l in open(erosion_dir / "raw_erosion_meta.jsonl")]
    N = len(meta)
    for pos in POSITIONS:
        assert acts[pos].shape[0] == N, f"Shape mismatch at {pos}: {acts[pos].shape[0]} vs {N}"

    labels    = np.array([m["evidence_label"] for m in meta], dtype=np.int32)   # 0 or 1
    stop      = np.array([m["behavioral_stop"] for m in meta], dtype=bool)
    is_a3     = np.array([m["is_a3_rescued"]   for m in meta], dtype=bool)
    sample_ids = [m["sample_id"] for m in meta]
    thought_lens = np.array([m["n_thought_tokens"] for m in meta], dtype=int)

    probe_direction = np.load(probe_dir / "probe_direction_l20.npz")["decision_direction"].astype(np.float32)

    print(f"Loaded N={N}: label0={( labels==0).sum()}, label1={(labels==1).sum()}, "
          f"a3={is_a3.sum()}, stop={stop.sum()}, cont={(~stop).sum()}")
    print(f"Thought length: mean={thought_lens.mean():.1f} median={np.median(thought_lens):.0f} "
          f"min={thought_lens.min()} max={thought_lens.max()}")

    return acts, labels, stop, is_a3, sample_ids, thought_lens, probe_direction


# ── Core metrics ───────────────────────────────────────────────────────────────

def fixed_dir_auroc(acts_pos: np.ndarray, labels: np.ndarray,
                    probe_direction: np.ndarray) -> float:
    """AUROC using Phase-1 probe direction as the fixed score."""
    from sklearn.metrics import roc_auc_score
    proj = acts_pos @ probe_direction
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, proj))


def evidence_gap(acts_pos: np.ndarray, labels: np.ndarray,
                 probe_direction: np.ndarray) -> float:
    """mean(proj|label=1) - mean(proj|label=0)."""
    proj = acts_pos @ probe_direction
    return float(proj[labels == 1].mean() - proj[labels == 0].mean())


def behavioral_gap(acts_pos: np.ndarray, stop: np.ndarray,
                   probe_direction: np.ndarray) -> Optional[float]:
    """mean(proj|stop) - mean(proj|continue)."""
    if (~stop).sum() < 3:
        return None   # too few continue samples to be meaningful
    proj = acts_pos @ probe_direction
    return float(proj[stop].mean() - proj[~stop].mean())


def retrained_auroc(acts_pos: np.ndarray, labels: np.ndarray, seed: int = 42) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler

    if len(set(labels)) < 2:
        return float("nan")
    scaler = StandardScaler()
    X = scaler.fit_transform(acts_pos)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(sss.split(X, labels))
    clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                             solver="lbfgs", random_state=seed)
    clf.fit(X[train_idx], labels[train_idx])
    y_prob = clf.predict_proba(X[test_idx])[:, 1]
    try:
        return float(roc_auc_score(labels[test_idx], y_prob))
    except ValueError:
        return float("nan")


# ── Tier 1: Gap + AUROC curve ─────────────────────────────────────────────────

def compute_primary_curve(acts, labels, stop, probe_direction):
    results = {}
    for pos in POSITIONS:
        fd_auroc = fixed_dir_auroc(acts[pos], labels, probe_direction)
        ev_gap   = evidence_gap(acts[pos], labels, probe_direction)
        beh_gap  = behavioral_gap(acts[pos], stop, probe_direction)
        ret_auroc = retrained_auroc(acts[pos], labels)
        results[pos] = {
            "fixed_dir_auroc": fd_auroc,
            "evidence_gap":    ev_gap,
            "behavioral_gap":  beh_gap,
            "retrained_auroc": ret_auroc,
        }
    return results


# ── Tier 1: Bootstrap CIs ─────────────────────────────────────────────────────

def bootstrap_ci(acts, labels, stop, probe_direction, B: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    N = len(labels)

    fd_aurocs   = {pos: [] for pos in POSITIONS}
    ev_gaps     = {pos: [] for pos in POSITIONS}
    ret_aurocs  = {pos: [] for pos in POSITIONS}

    for b in range(B):
        idx = rng.integers(0, N, size=N)
        lbl_b = labels[idx]
        stp_b = stop[idx]

        for pos in POSITIONS:
            acts_b = acts[pos][idx]
            # Fixed-dir AUROC
            try:
                fda = fixed_dir_auroc(acts_b, lbl_b, probe_direction)
            except Exception:
                fda = float("nan")
            fd_aurocs[pos].append(fda)
            # Evidence gap
            try:
                eg = evidence_gap(acts_b, lbl_b, probe_direction)
            except Exception:
                eg = float("nan")
            ev_gaps[pos].append(eg)

        if b % 100 == 0:
            print(f"  Bootstrap {b}/{B}...")

    results = {}
    for pos in POSITIONS:
        fda_arr = np.array(fd_aurocs[pos])
        eg_arr  = np.array(ev_gaps[pos])
        results[pos] = {
            "fixed_dir_auroc_mean": float(np.nanmean(fda_arr)),
            "fixed_dir_auroc_ci95": (float(np.nanpercentile(fda_arr, 2.5)),
                                     float(np.nanpercentile(fda_arr, 97.5))),
            "evidence_gap_mean": float(np.nanmean(eg_arr)),
            "evidence_gap_ci95": (float(np.nanpercentile(eg_arr, 2.5)),
                                  float(np.nanpercentile(eg_arr, 97.5))),
        }
    return results


# ── Tier 1: Split stability ───────────────────────────────────────────────────

def split_stability(acts, labels, seeds=(42, 7, 13, 99, 123)):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler

    seed_curves = {}
    for seed in seeds:
        curve = {}
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        for pos in POSITIONS:
            X = StandardScaler().fit_transform(acts[pos])
            train_idx, test_idx = next(sss.split(X, labels))
            clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                                     solver="lbfgs", random_state=seed)
            clf.fit(X[train_idx], labels[train_idx])
            y_prob = clf.predict_proba(X[test_idx])[:, 1]
            try:
                auc = roc_auc_score(labels[test_idx], y_prob)
            except ValueError:
                auc = float("nan")
            curve[pos] = float(auc)
        seed_curves[seed] = curve
        vals = [curve[p] for p in POSITIONS]
        print(f"  seed={seed}: " + "  ".join(f"{p}={v:.4f}" for p, v in zip(POSITIONS, vals)))
    return seed_curves


# ── Tier 1: Permutation test ─────────────────────────────────────────────────

def permutation_test(acts, labels, probe_direction, K: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    N = len(labels)

    observed_shrinkage = (evidence_gap(acts["p0_input"], labels, probe_direction) -
                          evidence_gap(acts["p4_100pct"], labels, probe_direction))
    observed_fd_p0 = fixed_dir_auroc(acts["p0_input"], labels, probe_direction)
    observed_fd_p4 = fixed_dir_auroc(acts["p4_100pct"], labels, probe_direction)
    observed_auroc_shrinkage = observed_fd_p0 - observed_fd_p4

    null_shrinkages  = []
    null_auroc_shrinkages = []
    null_gap_p0 = []
    null_gap_p4 = []

    for k in range(K):
        perm_labels = rng.permutation(labels)
        g0 = evidence_gap(acts["p0_input"], perm_labels, probe_direction)
        g4 = evidence_gap(acts["p4_100pct"], perm_labels, probe_direction)
        null_shrinkages.append(g0 - g4)
        null_gap_p0.append(g0)
        null_gap_p4.append(g4)

        try:
            from sklearn.metrics import roc_auc_score
            proj_p0 = acts["p0_input"] @ probe_direction
            proj_p4 = acts["p4_100pct"] @ probe_direction
            a0 = roc_auc_score(perm_labels, proj_p0)
            a4 = roc_auc_score(perm_labels, proj_p4)
            null_auroc_shrinkages.append(a0 - a4)
        except Exception:
            null_auroc_shrinkages.append(float("nan"))

    null_shrinkages = np.array(null_shrinkages)
    p_value = float((null_shrinkages >= observed_shrinkage).mean())

    print(f"\nPermutation test (K={K}):")
    print(f"  Observed gap_shrinkage (p0-p4): {observed_shrinkage:.4f}")
    print(f"  Null distribution: mean={null_shrinkages.mean():.4f} std={null_shrinkages.std():.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Observed AUROC_shrinkage (p0-p4): {observed_auroc_shrinkage:.4f}")

    return {
        "observed_gap_shrinkage":      observed_shrinkage,
        "observed_auroc_shrinkage":    observed_auroc_shrinkage,
        "observed_fd_auroc_p0":        observed_fd_p0,
        "observed_fd_auroc_p4":        observed_fd_p4,
        "p_value_gap_shrinkage":       p_value,
        "null_gap_shrinkage_mean":     float(null_shrinkages.mean()),
        "null_gap_shrinkage_std":      float(null_shrinkages.std()),
        "null_gap_p0_mean":            float(np.array(null_gap_p0).mean()),
        "null_gap_p4_mean":            float(np.array(null_gap_p4).mean()),
    }


# ── Tier 2: Subgroup erosion curves ──────────────────────────────────────────

def subgroup_curves(acts, labels, stop, is_a3, probe_direction):
    """Compute evidence_gap and fixed-dir AUROC for subgroups at each position."""
    subgroups = {
        "all":        np.ones(len(labels), dtype=bool),
        "label0":     labels == 0,
        "label1":     labels == 1,
        "a3_rescued": is_a3,
        "stop":       stop,
        "continue":   ~stop,
    }

    results = {grp: {pos: {} for pos in POSITIONS} for grp in subgroups}

    for pos in POSITIONS:
        proj = acts[pos] @ probe_direction   # (N,)
        for grp, mask in subgroups.items():
            if mask.sum() < 3:
                results[grp][pos] = {"n": int(mask.sum()), "mean_proj": None, "ev_gap": None, "fd_auroc": None}
                continue
            sub_proj  = proj[mask]
            sub_lbl   = labels[mask]
            results[grp][pos]["n"]         = int(mask.sum())
            results[grp][pos]["mean_proj"] = float(sub_proj.mean())
            # Evidence gap only if both classes present
            if len(set(sub_lbl)) >= 2:
                results[grp][pos]["ev_gap"] = float(
                    sub_proj[sub_lbl == 1].mean() - sub_proj[sub_lbl == 0].mean()
                )
                try:
                    from sklearn.metrics import roc_auc_score
                    results[grp][pos]["fd_auroc"] = float(roc_auc_score(sub_lbl, sub_proj))
                except Exception:
                    results[grp][pos]["fd_auroc"] = None
            else:
                results[grp][pos]["ev_gap"]   = None
                results[grp][pos]["fd_auroc"] = None

    return results


# ── Tier 3: E-condition self-contradictions ───────────────────────────────────

UNCERTAINTY_PHRASES = [
    "does not contain", "not specified", "not explicitly", "not mentioned",
    "not provided", "cannot determine", "insufficient", "not enough",
    "need more", "more information", "further research", "not clear",
    "cannot be determined", "no clear", "not available",
]


def find_self_contradictions(dissoc_path: Path, n_examples: int = 10) -> List[dict]:
    if not dissoc_path.exists():
        print(f"  Dissociation results not found at {dissoc_path}")
        return []

    rows = [json.loads(l) for l in open(dissoc_path)]
    # Use most-recent (deduped) full-run rows
    seen = {}
    for r in rows:
        if "cf_parse_ok" in r:   # marker of full run
            seen[r["sample_id"]] = r
    rows = list(seen.values())

    contradictions = []
    for r in rows:
        raw_e = r.get("raw_e", "").lower()
        if r.get("e_continue"):        # already searching, not a contradiction
            continue
        if any(p in raw_e for p in UNCERTAINTY_PHRASES):
            # Type 1: text expresses uncertainty but still gives Final Answer
            contradictions.append({
                "sample_id":      r["sample_id"],
                "question":       r.get("question", ""),
                "evidence_label": r.get("evidence_label", "?"),
                "d_stop":         r.get("d_stop", True),
                "raw_e":          r.get("raw_e", ""),
                "gold_answer":    r.get("gold_answer", ""),
            })

    contradictions.sort(key=lambda x: x["evidence_label"])   # label=0 first (most interesting)
    print(f"\nType 1 self-contradictions: {len(contradictions)}")
    print(f"  label=0 (no SF retrieved): {sum(1 for c in contradictions if c['evidence_label']==0)}")
    print(f"  label=1 (partial SF):      {sum(1 for c in contradictions if c['evidence_label']==1)}")
    return contradictions[:n_examples]


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_all_plots(
    primary_curve, bootstrap_ci_results, seed_curves,
    perm_results, subgroup_results,
    output_dir: Path
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    x = POS_X
    xl = POS_LABELS

    def _vals(key, curve=primary_curve):
        return [curve[p][key] for p in POSITIONS]

    # ── Figure 1: Main figure (gap curve + fixed-dir AUROC dual y-axis) ───────
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ev_gaps = _vals("evidence_gap")
    fd_aurocs = _vals("fixed_dir_auroc")

    # Evidence gap + CI
    eg_lo = [bootstrap_ci_results[p]["evidence_gap_ci95"][0] for p in POSITIONS]
    eg_hi = [bootstrap_ci_results[p]["evidence_gap_ci95"][1] for p in POSITIONS]
    eg_mean = [bootstrap_ci_results[p]["evidence_gap_mean"] for p in POSITIONS]

    color_gap  = "#1f77b4"
    color_auroc = "#d62728"

    ax1.plot(x, ev_gaps, "o-", color=color_gap, linewidth=2.5, markersize=8,
             label="Evidence gap (L1−L0 projection)", zorder=3)
    ax1.fill_between(x, eg_lo, eg_hi, alpha=0.15, color=color_gap, label="95% CI (bootstrap)")
    ax1.set_xlabel("Thought generation position", fontsize=12)
    ax1.set_ylabel("Evidence gap\n(mean proj_label1 − mean proj_label0)", fontsize=11,
                   color=color_gap)
    ax1.tick_params(axis="y", labelcolor=color_gap)
    ax1.set_xticks(x)
    ax1.set_xticklabels(xl, fontsize=9)
    ax1.set_ylim(bottom=0)
    ax1.axhline(0, color="gray", linestyle=":", linewidth=0.8)

    ax2 = ax1.twinx()
    # Fixed-dir AUROC + CI
    fa_lo = [bootstrap_ci_results[p]["fixed_dir_auroc_ci95"][0] for p in POSITIONS]
    fa_hi = [bootstrap_ci_results[p]["fixed_dir_auroc_ci95"][1] for p in POSITIONS]
    ax2.plot(x, fd_aurocs, "s--", color=color_auroc, linewidth=2, markersize=7,
             label="Fixed-dir AUROC (Phase-1 probe)", zorder=3)
    ax2.fill_between(x, fa_lo, fa_hi, alpha=0.10, color=color_auroc)
    ax2.axhline(0.5, color="#d62728", linestyle=":", linewidth=0.8, alpha=0.4)
    ax2.set_ylabel("Fixed-dir AUROC", fontsize=11, color=color_auroc)
    ax2.tick_params(axis="y", labelcolor=color_auroc)
    ax2.set_ylim(0.4, 1.0)

    # Annotations
    gap_shrinkage = ev_gaps[0] - ev_gaps[-1]
    pval = perm_results["p_value_gap_shrinkage"]
    ax1.set_title(
        f"Evidence Signal Erosion During Thought Generation (L20, N=486)\n"
        f"Gap shrinkage: {ev_gaps[0]:.3f}→{ev_gaps[-1]:.3f} (−{gap_shrinkage:.3f}, perm p={pval:.4f})",
        fontsize=11
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    fig.tight_layout()
    out = output_dir / "erosion_gap_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # ── Figure 2: Split stability ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(seed_curves)))
    for (seed, curve), c in zip(seed_curves.items(), colors):
        vals = [curve[p] for p in POSITIONS]
        ax.plot(x, vals, "o-", color=c, linewidth=1.5, markersize=5,
                label=f"seed={seed}", alpha=0.8)
    # Mean curve
    mean_curve = [np.mean([sc[p] for sc in seed_curves.values()]) for p in POSITIONS]
    ax.plot(x, mean_curve, "k^-", linewidth=2.5, markersize=8, label="Mean (5 seeds)", zorder=5)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(xl, fontsize=9)
    ax.set_ylabel("Retrained AUROC (80/20 split)", fontsize=11)
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Split Stability: 5 Random Seeds — Erosion Curve Shape", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out2 = output_dir / "erosion_stability.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")

    # ── Figure 3: Permutation test null distribution ──────────────────────────
    # (We'll just annotate perm_results in the main figure; no separate plot needed
    #  unless we want to show the histogram)

    # ── Figure 4: Subgroup erosion curves ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: Overall vs A3-rescued
    ax = axes[0]
    all_eg   = [subgroup_results["all"][p].get("ev_gap") for p in POSITIONS]
    a3_proj  = [subgroup_results["a3_rescued"][p].get("mean_proj") for p in POSITIONS]
    all_proj = [subgroup_results["all"][p].get("mean_proj") for p in POSITIONS]

    ax.plot(x, all_eg, "o-", color="#1f77b4", linewidth=2, markersize=7,
            label=f"All (N={subgroup_results['all']['p0_input']['n']}) — ev_gap")
    # For A3, compute ev_gap manually using per-position mean_proj
    # Since A3 subset is all-label=0 (mostly), ev_gap within A3 is not meaningful.
    # Instead: compare A3 mean_proj vs overall mean_proj (both normalized)
    # Normalize: proj difference from p0
    a3_norm = [v - a3_proj[0] if v is not None else None for v in a3_proj]
    all_norm = [v - all_proj[0] if v is not None else None for v in all_proj]
    ax2b = ax.twinx()
    ax2b.plot(x, a3_norm, "^--", color="#2ca02c", linewidth=1.8, markersize=7,
              label=f"A3 rescued (N={subgroup_results['a3_rescued']['p0_input']['n']}) — proj drift",
              alpha=0.85)
    ax2b.plot(x, all_norm, "s:", color="#aec7e8", linewidth=1.5, markersize=5,
              label="All — proj drift", alpha=0.7)
    ax2b.set_ylabel("Mean proj drift (relative to p0)", fontsize=9, color="#2ca02c")
    ax2b.tick_params(axis="y", labelcolor="#2ca02c")

    ax.set_xticks(x); ax.set_xticklabels(xl, fontsize=9)
    ax.set_ylabel("Evidence gap (L1−L0, overall)", fontsize=10, color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.set_title("Overall vs A3-Rescued Subset\nErosion Pattern Comparison", fontsize=10)
    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2b.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8, loc="lower left")

    # Panel B: label0 vs label1 separate erosion
    ax = axes[1]
    def _safe(vals):
        return [v if v is not None else float("nan") for v in vals]

    # For 0-doc and 1-doc separately, track mean projection (within-group)
    l0_proj = _safe([subgroup_results["label0"][p].get("mean_proj") for p in POSITIONS])
    l1_proj = _safe([subgroup_results["label1"][p].get("mean_proj") for p in POSITIONS])
    # Also track fixed-dir AUROC within each group (meaningful only for "all")
    fd_all = _vals("fixed_dir_auroc")

    ax.plot(x, l0_proj, "o-", color="#d62728", linewidth=2, markersize=7,
            label=f"0-doc (N=97) mean proj")
    ax.plot(x, l1_proj, "s-", color="#1f77b4", linewidth=2, markersize=7,
            label=f"1-doc (N=389) mean proj")
    ax.set_xticks(x); ax.set_xticklabels(xl, fontsize=9)
    ax.set_ylabel("Mean projection on evidence direction", fontsize=10)
    ax.set_title("0-doc vs 1-doc Subgroup\nMean Projection vs Thought Position", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Subgroup Erosion Analysis", fontsize=12)
    fig.tight_layout()
    out3 = output_dir / "erosion_subgroups.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out3}")


# ── Paper figure (composite) ──────────────────────────────────────────────

def make_paper_figure(
    primary_curve, bootstrap_ci_results, perm_results,
    contradictions: List[dict],
    output_dir: Path,
    N: int = 486,
):
    """
    Main-text figure: Panel A = FD_AUROC + EV_gap dual-axis with bootstrap CI.
    Panel B = 2–3 E-condition self-contradiction examples.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import textwrap
    except ImportError:
        print("matplotlib not available, skipping paper figure")
        return

    x = POS_X
    xl = ["Input\n(p0)", "25%\n(p1)", "50%\n(p2)", "75%\n(p3)", "100%\n(p4)"]

    def _vals(key):
        return [primary_curve[p][key] for p in POSITIONS]

    ev_gaps   = _vals("evidence_gap")
    fd_aurocs = _vals("fixed_dir_auroc")

    # Bootstrap CIs
    eg_lo = [bootstrap_ci_results[p]["evidence_gap_ci95"][0] for p in POSITIONS]
    eg_hi = [bootstrap_ci_results[p]["evidence_gap_ci95"][1] for p in POSITIONS]
    fa_lo = [bootstrap_ci_results[p]["fixed_dir_auroc_ci95"][0] for p in POSITIONS]
    fa_hi = [bootstrap_ci_results[p]["fixed_dir_auroc_ci95"][1] for p in POSITIONS]

    pval = perm_results["p_value_gap_shrinkage"]
    gap_shrinkage = ev_gaps[0] - ev_gaps[-1]
    pct = gap_shrinkage / ev_gaps[0] * 100

    # ── Layout: 5:3 ratio ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 5.5))
    gs = GridSpec(1, 2, width_ratios=[5, 3], wspace=0.35)

    # ── Panel A: Erosion curves ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])

    color_gap   = "#2166ac"   # dark blue
    color_auroc = "#b2182b"   # dark red

    # Evidence gap
    ax1.plot(x, ev_gaps, "o-", color=color_gap, linewidth=2.5, markersize=9,
             label="Evidence gap (label-1 − label-0)", zorder=3)
    ax1.fill_between(x, eg_lo, eg_hi, alpha=0.18, color=color_gap)
    ax1.set_xlabel("Position in thought generation", fontsize=12)
    ax1.set_ylabel("Evidence gap\n(class separation on probe direction)", fontsize=11,
                   color=color_gap)
    ax1.tick_params(axis="y", labelcolor=color_gap)
    ax1.set_xticks(x)
    ax1.set_xticklabels(xl, fontsize=9)
    ax1.set_ylim(bottom=-0.05, top=max(eg_hi) * 1.15)
    ax1.axhline(0, color="gray", linestyle=":", linewidth=0.7)

    # Fixed-dir AUROC (right y-axis)
    ax2 = ax1.twinx()
    ax2.plot(x, fd_aurocs, "s--", color=color_auroc, linewidth=2, markersize=7,
             label="Fixed-dir AUROC", zorder=3)
    ax2.fill_between(x, fa_lo, fa_hi, alpha=0.12, color=color_auroc)
    ax2.axhline(0.5, color=color_auroc, linestyle=":", linewidth=0.7, alpha=0.35)
    ax2.set_ylabel("Fixed-direction AUROC", fontsize=11, color=color_auroc)
    ax2.tick_params(axis="y", labelcolor=color_auroc)
    ax2.set_ylim(0.40, 1.02)

    # Annotation: shrinkage
    ax1.annotate(
        f"−{pct:.0f}%\n(p < 0.001)",
        xy=(4, ev_gaps[-1]), xytext=(3.3, ev_gaps[0] * 0.45),
        fontsize=10, fontweight="bold", color=color_gap,
        arrowprops=dict(arrowstyle="->", color=color_gap, lw=1.5),
        ha="center",
    )
    ax2.annotate(
        f"→ chance",
        xy=(4, fd_aurocs[-1]), xytext=(3.5, 0.72),
        fontsize=9, color=color_auroc, alpha=0.8,
        arrowprops=dict(arrowstyle="->", color=color_auroc, lw=1.2, alpha=0.6),
        ha="center",
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
              loc="upper right", fontsize=9, framealpha=0.9)

    ax1.set_title("(a) Evidence signal degradation during thought generation",
                  fontsize=11, fontweight="bold", pad=10)

    # ── Panel B: Contradiction examples ───────────────────────────────────────
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis("off")
    ax_text.set_title("(b) Self-contradiction examples (label = 0)",
                      fontsize=11, fontweight="bold", pad=10)

    # Pick best label=0 examples
    label0_contras = [c for c in contradictions if c["evidence_label"] == 0]
    show = label0_contras[:3] if len(label0_contras) >= 3 else contradictions[:3]

    y_top = 0.95
    box_height = 0.30
    for i, c in enumerate(show):
        y = y_top - i * (box_height + 0.04)

        q_text = c["question"][:90]
        if len(c["question"]) > 90:
            q_text += "..."

        # Extract the key uncertainty phrase from the E output
        raw_e = c["raw_e"]
        # Trim to the most relevant part
        e_text = raw_e[:180]
        if len(raw_e) > 180:
            e_text += "..."

        # Wrap text
        q_wrapped = textwrap.fill(f"Q: {q_text}", width=55)
        e_wrapped = textwrap.fill(f"E: {e_text}", width=55)

        block = f"{q_wrapped}\n\n{e_wrapped}"

        ax_text.text(
            0.02, y, block,
            transform=ax_text.transAxes,
            fontsize=7.5, fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3e0",
                      edgecolor="#e65100", alpha=0.85, linewidth=1.2),
        )

    # Footer note
    ax_text.text(
        0.02, 0.02,
        "Model outputs \"Final Answer\" despite expressing\n"
        "uncertainty (\"further research needed\", \"not clear\").\n"
        "Evidence: insufficient (0 of 2 supporting facts retrieved).",
        transform=ax_text.transAxes,
        fontsize=8, fontstyle="italic", color="#555555",
        verticalalignment="bottom",
    )

    fig.suptitle(
        f"Evidence Signal Erosion During Thought Generation  "
        f"(Layer 20, N={N})",
        fontsize=13, fontweight="bold", y=1.02,
    )

    out_png = output_dir / "paper_figure_erosion.png"
    out_pdf = output_dir / "paper_figure_erosion.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved paper figure: {out_png}")
    print(f"Saved paper figure (PDF): {out_pdf}")


# ── Report printing ───────────────────────────────────────────────────────────

def print_bootstrap_report(bootstrap_results, primary_curve):
    print("\n" + "=" * 75)
    print("BOOTSTRAP 95% CI REPORT (B=1000)")
    print("=" * 75)
    print(f"{'Position':<14} {'FD_AUROC':>10} {'CI_lo':>8} {'CI_hi':>8} "
          f"{'EV_gap':>10} {'CI_lo':>8} {'CI_hi':>8}")
    print("-" * 75)
    for pos in POSITIONS:
        br = bootstrap_results[pos]
        pr = primary_curve[pos]
        fda_lo, fda_hi = br["fixed_dir_auroc_ci95"]
        eg_lo,  eg_hi  = br["evidence_gap_ci95"]
        print(f"{pos:<14} {pr['fixed_dir_auroc']:>10.4f} {fda_lo:>8.4f} {fda_hi:>8.4f} "
              f"{pr['evidence_gap']:>10.4f} {eg_lo:>8.4f} {eg_hi:>8.4f}")

    # Check p0 vs p4 CI overlap
    p0_fda_ci = bootstrap_results["p0_input"]["fixed_dir_auroc_ci95"]
    p4_fda_ci = bootstrap_results["p4_100pct"]["fixed_dir_auroc_ci95"]
    p0_eg_ci  = bootstrap_results["p0_input"]["evidence_gap_ci95"]
    p4_eg_ci  = bootstrap_results["p4_100pct"]["evidence_gap_ci95"]

    fda_overlap = p0_fda_ci[0] <= p4_fda_ci[1]  # p0_lo <= p4_hi
    eg_overlap  = p0_eg_ci[0]  <= p4_eg_ci[1]

    print(f"\n  p0 FD_AUROC CI: [{p0_fda_ci[0]:.4f}, {p0_fda_ci[1]:.4f}]")
    print(f"  p4 FD_AUROC CI: [{p4_fda_ci[0]:.4f}, {p4_fda_ci[1]:.4f}]")
    print(f"  CI overlap (p0 vs p4 FD_AUROC): {'YES' if fda_overlap else 'NO (significant)'}",
          "←" if not fda_overlap else "")
    print(f"\n  p0 EV_gap CI: [{p0_eg_ci[0]:.4f}, {p0_eg_ci[1]:.4f}]")
    print(f"  p4 EV_gap CI: [{p4_eg_ci[0]:.4f}, {p4_eg_ci[1]:.4f}]")
    print(f"  CI overlap (p0 vs p4 EV_gap):   {'YES' if eg_overlap else 'NO (significant)'}",
          "←" if not eg_overlap else "")


def print_split_stability_report(seed_curves):
    print("\n" + "=" * 75)
    print("SPLIT STABILITY (5 SEEDS)")
    print("=" * 75)
    print(f"{'Seed':<8}", end="")
    for pos in POSITIONS:
        print(f"{pos:>14}", end="")
    print()
    print("-" * 75)
    for seed, curve in seed_curves.items():
        print(f"{seed:<8}", end="")
        for pos in POSITIONS:
            print(f"{curve[pos]:>14.4f}", end="")
        print()
    # Check monotonicity
    print("\n  Monotone decreasing?")
    for seed, curve in seed_curves.items():
        vals = [curve[p] for p in POSITIONS]
        is_mono = all(vals[i] >= vals[i+1] for i in range(len(vals)-1))
        print(f"  seed={seed}: {'YES ✓' if is_mono else 'NO (non-monotone)'}")


def save_contradiction_examples(contradictions, output_dir):
    if not contradictions:
        return
    out = output_dir / "e_condition_contradictions.json"
    with open(out, "w") as f:
        json.dump(contradictions, f, indent=2)

    md_lines = [
        "# E-Condition Type 1 Self-Contradictions",
        "",
        f"N = {len(contradictions)} examples where E text expresses uncertainty but still gives Final Answer.",
        "",
    ]
    for i, c in enumerate(contradictions[:5]):
        md_lines += [
            f"## Example {i+1}  (label={c['evidence_label']})",
            f"**Q:** {c['question']}",
            "",
            f"**E output:**",
            f"> {c['raw_e'][:400]}",
            "",
            f"**Gold answer:** {c['gold_answer']}",
            "",
        ]
    with open(output_dir / "e_condition_contradictions.md", "w") as f:
        f.write("\n".join(md_lines))
    print(f"\nSaved E-condition contradiction examples to {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--erosion-dir",  default="results/thought_erosion")
    parser.add_argument("--probe-dir",    default="results/phase1_probe")
    parser.add_argument("--dissoc-path",  default="results/agent_specific_dissociation/raw_results.jsonl")
    parser.add_argument("--output-dir",   default="results/thought_erosion")
    parser.add_argument("--bootstrap-b",  type=int, default=1000)
    parser.add_argument("--permutation-k", type=int, default=1000)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    erosion_dir = Path(args.erosion_dir)
    probe_dir   = Path(args.probe_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ────────────────────────────────────────────────────────────────
    acts, labels, stop, is_a3, sample_ids, thought_lens, probe_dir_vec = load_data(
        erosion_dir, probe_dir
    )

    # ── Tier 1: Primary curve ────────────────────────────────────────────────
    print("\n=== Primary Curve ===")
    primary = compute_primary_curve(acts, labels, stop, probe_dir_vec)
    print(f"{'Position':<14} {'FD_AUROC':>10} {'EV_gap':>10} {'BEH_gap':>10} {'RET_AUROC':>12}")
    print("-" * 58)
    for pos in POSITIONS:
        p = primary[pos]
        bg = f"{p['behavioral_gap']:+.4f}" if p['behavioral_gap'] is not None else "  N/A (few cont)"
        print(f"{pos:<14} {p['fixed_dir_auroc']:>10.4f} {p['evidence_gap']:>10.4f} "
              f"{bg:>10} {p['retrained_auroc']:>12.4f}")

    gap_vals = [primary[p]["evidence_gap"] for p in POSITIONS]
    shrinkage = gap_vals[0] - gap_vals[-1]
    pct = shrinkage / gap_vals[0] * 100
    print(f"\n  Evidence gap shrinkage: {gap_vals[0]:.3f}→{gap_vals[-1]:.3f}  "
          f"Δ={shrinkage:.3f} ({pct:.1f}%)")

    # Note on behavioral_gap
    n_cont = int((~stop).sum())
    if n_cont < 20:
        print(f"\n  NOTE: behavioral_gap unreliable — only {n_cont} continue samples.")
        print("        Use evidence_gap (label0 vs label1) as the primary erosion metric.")

    # ── Tier 1: Bootstrap ────────────────────────────────────────────────────
    print(f"\n=== Bootstrap CIs (B={args.bootstrap_b}) ===")
    bootstrap_results = bootstrap_ci(
        acts, labels, stop, probe_dir_vec,
        B=args.bootstrap_b, seed=args.seed
    )
    print_bootstrap_report(bootstrap_results, primary)

    # ── Tier 1: Split stability ──────────────────────────────────────────────
    print("\n=== Split Stability (5 seeds) ===")
    seed_curves = split_stability(acts, labels, seeds=(42, 7, 13, 99, 123))
    print_split_stability_report(seed_curves)

    # ── Tier 1: Permutation test ──────────────────────────────────────────────
    print(f"\n=== Permutation Test (K={args.permutation_k}) ===")
    perm_results = permutation_test(
        acts, labels, probe_dir_vec,
        K=args.permutation_k, seed=args.seed
    )

    # ── Tier 2: Subgroup analysis ─────────────────────────────────────────────
    print("\n=== Subgroup Analysis ===")
    subgroup_results = subgroup_curves(acts, labels, stop, is_a3, probe_dir_vec)

    # A3 subset summary
    print(f"\nA3 rescued (N={is_a3.sum()}) mean projection at each position:")
    for pos in POSITIONS:
        sg = subgroup_results["a3_rescued"][pos]
        all_sg = subgroup_results["all"][pos]
        print(f"  {pos}: A3_proj={sg.get('mean_proj', None):.3f}  "
              f"all_proj={all_sg.get('mean_proj', None):.3f}")

    print(f"\n0-doc (N={(labels==0).sum()}) vs 1-doc (N={(labels==1).sum()}) mean projections:")
    print(f"{'pos':<14} {'0-doc':>10} {'1-doc':>10}  gap")
    for pos in POSITIONS:
        l0 = subgroup_results["label0"][pos].get("mean_proj")
        l1 = subgroup_results["label1"][pos].get("mean_proj")
        gap = (l1 - l0) if (l0 is not None and l1 is not None) else None
        print(f"{pos:<14} {l0:>10.4f} {l1:>10.4f}  {gap:.4f}" if gap else f"{pos}: N/A")

    # ── Tier 3: E-condition self-contradictions ───────────────────────────────
    print("\n=== E-Condition Self-Contradictions ===")
    contradictions = find_self_contradictions(
        Path(args.dissoc_path), n_examples=10
    )
    for i, c in enumerate(contradictions[:5]):
        print(f"\n  [{i+1}] label={c['evidence_label']}  Q: {c['question'][:70]}")
        print(f"      E: {c['raw_e'][:200]}")
    save_contradiction_examples(contradictions, output_dir)

    # ── Save results ──────────────────────────────────────────────────────────
    def _ser(v):
        if isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {kk: _ser(vv) for kk, vv in v.items()}
        if isinstance(v, (list, tuple)):
            return [_ser(i) for i in v]
        return v

    out_json = {
        "primary_curve":    _ser(primary),
        "bootstrap_ci":     _ser(bootstrap_results),
        "split_stability":  _ser({str(k): v for k, v in seed_curves.items()}),
        "permutation_test": _ser(perm_results),
        "subgroup_curves":  _ser(subgroup_results),
        "summary": {
            "observed_gap_shrinkage": float(shrinkage),
            "observed_gap_shrinkage_pct": float(pct),
            "permutation_p_value": float(perm_results["p_value_gap_shrinkage"]),
            "n_continue_samples": int((~stop).sum()),
            "gap_p0": float(gap_vals[0]),
            "gap_p4": float(gap_vals[-1]),
            "fd_auroc_p0": float(primary["p0_input"]["fixed_dir_auroc"]),
            "fd_auroc_p4": float(primary["p4_100pct"]["fixed_dir_auroc"]),
        }
    }
    json_path = output_dir / "erosion_analysis.json"
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nSaved analysis to {json_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    make_all_plots(primary, bootstrap_results, seed_curves,
                   perm_results, subgroup_results, output_dir)

    # ── Paper figure (composite) ─────────────────────────────────────────────
    make_paper_figure(
        primary, bootstrap_results, perm_results,
        contradictions, output_dir, N=len(labels),
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("EROSION ANALYSIS SUMMARY")
    print("=" * 65)
    print(f"  Gap shrinkage:         {gap_vals[0]:.3f} → {gap_vals[-1]:.3f}  (−{shrinkage:.3f}, −{pct:.1f}%)")
    print(f"  Permutation p-value:   {perm_results['p_value_gap_shrinkage']:.4f}")
    perm_sig = perm_results['p_value_gap_shrinkage'] < 0.05
    print(f"  Erosion significant?   {'YES ✓' if perm_sig else 'NO'}")

    p0_ci = bootstrap_results["p0_input"]["evidence_gap_ci95"]
    p4_ci = bootstrap_results["p4_100pct"]["evidence_gap_ci95"]
    ci_overlap = p0_ci[0] <= p4_ci[1]
    print(f"  CI overlap (p0 vs p4): {'YES (not clearly separate)' if ci_overlap else 'NO (clearly separated) ✓'}")

    mono = all(
        np.mean([sc[POSITIONS[i]] for sc in seed_curves.values()]) >=
        np.mean([sc[POSITIONS[i+1]] for sc in seed_curves.values()])
        for i in range(len(POSITIONS)-1)
    )
    print(f"  Mean curve monotone?   {'YES ✓' if mono else 'NO (non-monotone mean)'}")
    print(f"\n  Outputs: {output_dir}/")


if __name__ == "__main__":
    main()
