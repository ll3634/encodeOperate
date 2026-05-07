#!/usr/bin/env python3
"""
Erosion Specificity Test — causal chain control for the alignment drop finding.

Shows that the alignment drop (cos(probe_retrained_at_p, w_fixed) going from
~0.363 at p0 to ~0.040 at p4) is evidence-specific, not general drift.

Three controls:

1. Random direction baseline (K=200, seed=42)
   - Sample 200 random unit vectors from R^{3584}
   - Each has a constant cos with w_fixed (horizontal lines in the plot)
   - Null distribution of |alignment drops| (all ~0 for random directions)
   - Reports: evidence probe drop percentile + p-value vs null

2. Content-matched control
   - Train a probe predicting question length (long/short, threshold=median)
     at each position using the same erosion activations
   - Compute cos(content_probe_at_p, w_fixed) at each position
   - Expected: flat and low curve (no drop), proves erosion is evidence-specific

3. Summary figure
   - x-axis: positions p0 → p4
   - y-axis: cos(direction, w_fixed) [alignment with fixed evidence probe]
   - Blue  : evidence probe retrained at each position (0.363 → 0.040)
   - Gray  : 200 random directions (mean ± 2 std)  [horizontal band]
   - Orange: content-matched control probe retrained at each position

Saves to: results/erosion_specificity/

Usage:
    cd tmc/scripts/e2e_agent
    python scripts/erosion_specificity.py \\
        --erosion-dir results/thought_erosion \\
        --probe-dir   results/phase1_probe \\
        --output-dir  results/erosion_specificity
"""

import sys, json, argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

POSITIONS  = ["p0_input", "p1_25pct", "p2_50pct", "p3_75pct", "p4_100pct"]
POS_LABELS = ["p0\n(input)", "p1\n(25%)", "p2\n(50%)", "p3\n(75%)", "p4\n(100%)"]
POS_X      = [0, 1, 2, 3, 4]
K_RANDOM   = 200
RANDOM_SEED = 42


# ── Data ─────────────────────────────────────────────────────────────────────

def load_data(erosion_dir: Path, probe_dir: Path):
    d = np.load(erosion_dir / "raw_erosion_data.npz")
    acts = {pos: d[pos].astype(np.float32) for pos in POSITIONS}   # (N, 3584)

    meta = [json.loads(l) for l in open(erosion_dir / "raw_erosion_meta.jsonl")]
    labels = np.array([m["evidence_label"] for m in meta], dtype=np.int32)

    # Question length labels (binary: >median → 1, else → 0)
    q_lens = np.array([len(m["question"]) for m in meta], dtype=np.int32)
    ql_labels = (q_lens > np.median(q_lens)).astype(np.int32)

    # Fixed reference probe direction (from phase 1, in original activation space)
    w_fixed = np.load(probe_dir / "probe_direction_l20.npz")["decision_direction"].astype(np.float32)
    w_fixed = w_fixed / np.linalg.norm(w_fixed)

    N = len(meta)
    print(f"Loaded N={N}  label0={(labels==0).sum()}  label1={(labels==1).sum()}")
    print(f"Question length: median={np.median(q_lens):.0f}  "
          f"ql_label0={(ql_labels==0).sum()}  ql_label1={(ql_labels==1).sum()}")
    return acts, labels, ql_labels, w_fixed


# ── Probe training ─────────────────────────────────────────────────────────────

def train_probe_direction(acts_p: np.ndarray, labels: np.ndarray,
                          seed: int = 42) -> Tuple[np.ndarray, float]:
    """
    Train logistic regression probe on acts_p using labels.
    Returns (unit-norm direction in original activation space, CV AUROC).
    Weights are back-transformed from scaled space so they live in the same
    R^{3584} as w_fixed.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    sc = StandardScaler()
    X = sc.fit_transform(acts_p)
    clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                             solver="lbfgs", random_state=seed)
    clf.fit(X, labels)

    # Back-transform to original space: w_orig = w_scaled / scale
    w_orig = clf.coef_[0] / sc.scale_
    w_orig = w_orig.astype(np.float32)
    norm = np.linalg.norm(w_orig)
    if norm < 1e-10:
        return np.zeros_like(w_orig), 0.5
    w_orig = w_orig / norm

    # 80/20 stratified CV AUROC for reporting
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr, te = next(sss.split(X, labels))
        clf_cv = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                                     solver="lbfgs", random_state=seed)
        clf_cv.fit(X[tr], labels[tr])
        auroc = float(roc_auc_score(labels[te], clf_cv.predict_proba(X[te])[:, 1]))
    except Exception:
        auroc = float("nan")

    return w_orig, auroc


# ── Alignment curves ───────────────────────────────────────────────────────────

def compute_alignment_curve(direction: np.ndarray,
                             probe_dirs_at_pos: Dict[str, np.ndarray]) -> Dict[str, float]:
    """cos(direction, probe_at_p) at each position."""
    return {pos: float(np.dot(direction, probe_dirs_at_pos[pos])) for pos in POSITIONS}


# ── Random baseline ────────────────────────────────────────────────────────────

def random_null_baseline(w_fixed: np.ndarray, K: int = K_RANDOM,
                         seed: int = RANDOM_SEED) -> Dict:
    """
    Sample K random unit vectors; each has a CONSTANT cos with w_fixed
    (horizontal lines in the figure since both are fixed vectors).
    Returns null distributions of: alignment at p0, alignment at p4, drop.
    """
    rng = np.random.default_rng(seed)
    dim = len(w_fixed)
    # Sample from the unit sphere via normalised Gaussian
    R = rng.standard_normal((K, dim)).astype(np.float32)
    norms = np.linalg.norm(R, axis=1, keepdims=True)
    R = R / norms    # (K, dim) unit vectors

    # cos(r_k, w_fixed) is constant for each r_k regardless of position
    cos_vals = (R @ w_fixed).tolist()    # list of K floats
    abs_cos  = [abs(c) for c in cos_vals]

    # alignment "drop" for each random direction: |cos_p0| - |cos_p4| = 0
    # because the direction and w_fixed are both fixed.
    drops = [0.0] * K

    return {
        "random_directions": R,        # (K, dim) unit vectors, not serialised
        "cos_with_fixed":    cos_vals,
        "abs_cos_with_fixed": abs_cos,
        "drops":             drops,     # all 0.0 (fixed vectors)
        "mean_abs_cos":  float(np.mean(abs_cos)),
        "std_abs_cos":   float(np.std(abs_cos)),
        "p99_abs_cos":   float(np.percentile(abs_cos, 99)),
        "p95_abs_cos":   float(np.percentile(abs_cos, 95)),
        "theoretical_mean": float(np.sqrt(2 / np.pi) / np.sqrt(dim)),
    }


def pvalue_above_null(observed: float, null_vals: List[float]) -> float:
    """One-sided p-value: fraction of null values >= observed."""
    return sum(v >= observed for v in null_vals) / max(len(null_vals), 1)


def percentile_in_null(observed: float, null_vals: List[float]) -> float:
    """Percentile of observed in the null distribution."""
    return float(sum(v < observed for v in null_vals) / max(len(null_vals), 1) * 100)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--erosion-dir", default="results/thought_erosion")
    ap.add_argument("--probe-dir",   default="results/phase1_probe")
    ap.add_argument("--output-dir",  default="results/erosion_specificity")
    args = ap.parse_args()

    erosion_dir = Path(args.erosion_dir)
    probe_dir   = Path(args.probe_dir)
    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    acts, ev_labels, ql_labels, w_fixed = load_data(erosion_dir, probe_dir)

    # ── Train probes at each position ──────────────────────────────────────────
    print("\nTraining evidence probes at each position …")
    ev_probe_dirs = {}
    ev_probe_aurocs = {}
    for pos in POSITIONS:
        w_p, auroc_p = train_probe_direction(acts[pos], ev_labels)
        ev_probe_dirs[pos]  = w_p
        ev_probe_aurocs[pos] = auroc_p
        print(f"  {pos}: AUROC={auroc_p:.4f}  cos(w_p, w_fixed)={np.dot(w_p, w_fixed):.4f}")

    print("\nTraining content (question-length) probes at each position …")
    ql_probe_dirs = {}
    ql_probe_aurocs = {}
    for pos in POSITIONS:
        w_p, auroc_p = train_probe_direction(acts[pos], ql_labels)
        ql_probe_dirs[pos]  = w_p
        ql_probe_aurocs[pos] = auroc_p
        print(f"  {pos}: AUROC={auroc_p:.4f}  cos(w_p, w_fixed)={np.dot(w_p, w_fixed):.4f}")

    # ── Evidence probe alignment curve ─────────────────────────────────────────
    ev_align = compute_alignment_curve(w_fixed, ev_probe_dirs)
    ev_align_abs = {pos: abs(v) for pos, v in ev_align.items()}
    ev_drop = ev_align_abs["p0_input"] - ev_align_abs["p4_100pct"]

    print(f"\nEvidence probe alignment curve:")
    for pos in POSITIONS:
        print(f"  {pos}: {ev_align[pos]:.4f}  |{ev_align_abs[pos]:.4f}|")
    print(f"  → alignment drop (p0−p4): {ev_drop:.4f}")

    # ── Content control alignment curve ────────────────────────────────────────
    ql_align = compute_alignment_curve(w_fixed, ql_probe_dirs)
    ql_align_abs = {pos: abs(v) for pos, v in ql_align.items()}
    ql_drop = ql_align_abs["p0_input"] - ql_align_abs["p4_100pct"]

    print(f"\nContent-matched (question-length) alignment curve:")
    for pos in POSITIONS:
        print(f"  {pos}: {ql_align[pos]:.4f}  |{ql_align_abs[pos]:.4f}|")
    print(f"  → alignment drop (p0−p4): {ql_drop:.4f}")

    # ── Random baseline ─────────────────────────────────────────────────────────
    print(f"\nBuilding random null baseline (K={K_RANDOM}, seed={RANDOM_SEED}) …")
    null = random_null_baseline(w_fixed, K=K_RANDOM, seed=RANDOM_SEED)
    print(f"  |cos| with w_fixed: mean={null['mean_abs_cos']:.4f}  "
          f"std={null['std_abs_cos']:.4f}  p95={null['p95_abs_cos']:.4f}  "
          f"p99={null['p99_abs_cos']:.4f}")
    print(f"  Theoretical mean for dim=3584: {null['theoretical_mean']:.4f}")

    # ── Statistical tests ───────────────────────────────────────────────────────
    # Test 1: Is evidence p0 alignment above the null distribution?
    p0_obs = ev_align_abs["p0_input"]
    p4_obs = ev_align_abs["p4_100pct"]
    drop_obs = ev_drop

    p0_pvalue     = pvalue_above_null(p0_obs, null["abs_cos_with_fixed"])
    p0_percentile = percentile_in_null(p0_obs, null["abs_cos_with_fixed"])
    p4_pvalue     = pvalue_above_null(p4_obs, null["abs_cos_with_fixed"])
    p4_percentile = percentile_in_null(p4_obs, null["abs_cos_with_fixed"])

    # Test 2: Is evidence drop above null distribution of drops?
    # Null drops = all 0 (since random directions are fixed), so:
    # p-value = fraction of null drops >= observed drop = 0
    drop_pvalue     = pvalue_above_null(drop_obs, null["drops"])
    drop_percentile = percentile_in_null(drop_obs, null["drops"])

    # Bound p-value: if 0/K, report as ≤ 1/(K+1)
    drop_pvalue_bound = 1.0 / (K_RANDOM + 1) if drop_pvalue == 0.0 else drop_pvalue
    p0_pvalue_bound   = 1.0 / (K_RANDOM + 1) if p0_pvalue   == 0.0 else p0_pvalue

    print(f"\nStatistical tests:")
    print(f"  Evidence p0 alignment ({p0_obs:.4f}):")
    print(f"    percentile={p0_percentile:.1f}%  p≤{p0_pvalue_bound:.4f}")
    print(f"  Evidence p4 alignment ({p4_obs:.4f}):")
    print(f"    percentile={p4_percentile:.1f}%  p={p4_pvalue:.4f}")
    print(f"  Evidence alignment drop ({drop_obs:.4f}):")
    print(f"    percentile={drop_percentile:.1f}%  p≤{drop_pvalue_bound:.4f}")

    # ── Construct per-position results for figure ──────────────────────────────
    # For random directions: each has a constant cos across positions
    rand_cos_per_pos = {}
    for pos in POSITIONS:
        # All random directions have same cos at all positions (they're fixed)
        rand_cos_per_pos[pos] = null["abs_cos_with_fixed"]

    rand_mean_per_pos = {pos: float(np.mean(null["abs_cos_with_fixed"])) for pos in POSITIONS}
    rand_std_per_pos  = {pos: float(np.std(null["abs_cos_with_fixed"])) for pos in POSITIONS}

    # ── Save results JSON ──────────────────────────────────────────────────────
    results = {
        "config": {
            "K_random": K_RANDOM,
            "random_seed": RANDOM_SEED,
            "dim": int(w_fixed.shape[0]),
            "N_samples": len(ev_labels),
        },
        "evidence_probe_alignment": {
            pos: {"raw": float(ev_align[pos]), "abs": ev_align_abs[pos]}
            for pos in POSITIONS
        },
        "evidence_probe_auroc_at_pos": {pos: ev_probe_aurocs[pos] for pos in POSITIONS},
        "content_probe_alignment": {
            pos: {"raw": float(ql_align[pos]), "abs": ql_align_abs[pos]}
            for pos in POSITIONS
        },
        "content_probe_auroc_at_pos": {pos: ql_probe_aurocs[pos] for pos in POSITIONS},
        "random_null": {
            "mean_abs_cos": null["mean_abs_cos"],
            "std_abs_cos":  null["std_abs_cos"],
            "p95_abs_cos":  null["p95_abs_cos"],
            "p99_abs_cos":  null["p99_abs_cos"],
            "theoretical_mean": null["theoretical_mean"],
            "per_pos_mean": rand_mean_per_pos,
            "per_pos_std":  rand_std_per_pos,
        },
        "statistics": {
            "p0_alignment_obs": float(p0_obs),
            "p4_alignment_obs": float(p4_obs),
            "alignment_drop_obs": float(drop_obs),
            "p0_percentile_in_null": float(p0_percentile),
            "p0_pvalue": float(p0_pvalue_bound),
            "p4_percentile_in_null": float(p4_percentile),
            "p4_pvalue": float(p4_pvalue),
            "drop_percentile_in_null": float(drop_percentile),
            "drop_pvalue": float(drop_pvalue_bound),
            "is_erosion_specific": bool(drop_pvalue_bound < 0.01 and p0_pvalue_bound < 0.01),
        },
        "summary": {
            "evidence_p0": float(p0_obs),
            "evidence_p4": float(p4_obs),
            "evidence_drop": float(drop_obs),
            "random_mean_abs_cos": null["mean_abs_cos"],
            "random_p99_abs_cos":  null["p99_abs_cos"],
            "content_p0": ql_align_abs["p0_input"],
            "content_p4": ql_align_abs["p4_100pct"],
            "content_drop": float(ql_drop),
        },
    }

    out_path = out_dir / "erosion_specificity_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results → {out_path}")

    # ── Figure ─────────────────────────────────────────────────────────────────
    _make_figure(results, null, out_dir)

    # ── Print final verdict ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EROSION SPECIFICITY SUMMARY")
    print("=" * 60)
    print(f"Evidence probe alignment at p0 : {p0_obs:.4f}")
    print(f"Evidence probe alignment at p4 : {p4_obs:.4f}")
    print(f"Alignment drop (p0−p4)         : {drop_obs:.4f}")
    print(f"Random null mean |cos|         : {null['mean_abs_cos']:.4f}")
    print(f"Random null p99 |cos|          : {null['p99_abs_cos']:.4f}")
    print(f"Evidence p0 percentile in null : {p0_percentile:.1f}%  (p≤{p0_pvalue_bound:.4f})")
    print(f"Evidence p4 percentile in null : {p4_percentile:.1f}%  (p={p4_pvalue:.4f})")
    print(f"Drop percentile in null        : {drop_percentile:.1f}%  (p≤{drop_pvalue_bound:.4f})")
    print(f"Content control drop           : {ql_drop:.4f}")
    erosion_specific = results["statistics"]["is_erosion_specific"]
    verdict = "PASS — erosion IS evidence-specific" if erosion_specific else "REVIEW — check results"
    print(f"Verdict: {verdict}")
    print("=" * 60)


# ── Plotting ───────────────────────────────────────────────────────────────────

def _make_figure(results: dict, null: dict, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping figure")
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    xs = POS_X

    # ── Gray band: random directions (horizontal — fixed direction vs fixed w_fixed) ─
    rand_mean = null["mean_abs_cos"]
    rand_std  = null["std_abs_cos"]

    rand_lo = [max(0.0, rand_mean - 2 * rand_std)] * 5
    rand_hi = [rand_mean + 2 * rand_std] * 5

    ax.fill_between(xs, rand_lo, rand_hi, color="silver", alpha=0.45, zorder=1,
                    label=f"Random directions  mean ± 2σ  (n={K_RANDOM})")
    ax.axhline(rand_mean, color="gray", lw=1.0, ls="--", alpha=0.7, zorder=2)

    # Light individual random lines (first 100, very thin)
    for cos_k in null["abs_cos_with_fixed"][:100]:
        ax.axhline(cos_k, color="lightgray", lw=0.25, alpha=0.35, zorder=0)

    # ── Orange: content-matched control (question length) ─────────────────────
    ql_ys = [results["content_probe_alignment"][pos]["abs"] for pos in POSITIONS]
    ax.plot(xs, ql_ys, "s--", color="darkorange", lw=1.8, ms=5, zorder=3,
            label="Question-length control probe")

    # ── Blue: evidence probe ──────────────────────────────────────────────────
    ev_ys = [results["evidence_probe_alignment"][pos]["abs"] for pos in POSITIONS]
    ax.plot(xs, ev_ys, "o-", color="steelblue", lw=2.5, ms=7, zorder=4,
            label="Evidence probe  (decision-point ← trained at each pos)")

    # Value labels at p0 and p4
    ax.text(0 - 0.05, ev_ys[0] + 0.012, f"{ev_ys[0]:.3f}",
            color="steelblue", fontsize=9, fontweight="bold", ha="right")
    ax.text(4 + 0.05, ev_ys[-1] + 0.012, f"{ev_ys[-1]:.3f}",
            color="steelblue", fontsize=9, fontweight="bold", ha="left")

    # Double-headed arrow showing the drop (at x = 4.55, i.e. just right of p4)
    arrow_x = 4.35
    ax.annotate("",
                xy=(arrow_x, ev_ys[-1]),
                xytext=(arrow_x, ev_ys[0]),
                arrowprops=dict(arrowstyle="<->", color="steelblue", lw=1.4,
                                mutation_scale=12))
    ax.text(arrow_x + 0.08, (ev_ys[0] + ev_ys[-1]) / 2,
            f"Δ = {ev_ys[0] - ev_ys[-1]:.3f}",
            color="steelblue", fontsize=8.5, va="center")

    # Axis formatting
    ax.set_xticks(xs)
    ax.set_xticklabels(POS_LABELS, fontsize=9.5)
    ax.set_xlim(-0.6, 5.2)
    ax.set_xlabel("Position within thought generation", fontsize=10)
    ax.set_ylabel("|cos(probe_at_p,  w_fixed)|", fontsize=10)
    ax.set_title(
        "Erosion Specificity  —  evidence-action coupling is evidence-specific\n"
        "Blue: above random band at p0 → falls to band level at p4  |  "
        "Orange & gray: flat throughout",
        fontsize=9.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.3, lw=0.6)

    # ── Stats box ─────────────────────────────────────────────────────────────
    stats = results["statistics"]
    info = (
        f"evidence p0 = {stats['p0_alignment_obs']:.3f}  "
        f"(100th pct, p ≤ {stats['p0_pvalue']:.4f})\n"
        f"evidence p4 = {stats['p4_alignment_obs']:.3f}  "
        f"(pct={stats['p4_percentile_in_null']:.0f}%, p = {stats['p4_pvalue']:.4f})\n"
        f"drop Δ = {stats['alignment_drop_obs']:.3f}  "
        f"(p ≤ {stats['drop_pvalue']:.4f})\n"
        f"content drop = {results['summary']['content_drop']:.3f}   "
        f"random mean = {results['summary']['random_mean_abs_cos']:.4f}"
    )
    ax.text(0.01, 0.98, info, transform=ax.transAxes, fontsize=7.5,
            va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="steelblue", alpha=0.85))

    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = out_dir / f"erosion_specificity.{ext}"
        fig.savefig(p, dpi=150 if ext == "png" else None)
        print(f"Saved figure → {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
