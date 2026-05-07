#!/usr/bin/env python3
"""
Phase 2 Analysis: Compare Qwen2.5-7B-Base vs Qwen2.5-7B-Instruct

Computes the critical cosine: cosine(evidence_probe_direction, action_direction)
for both models and generates the RLHF Tax comparison table.

Usage:
    cd tmc/scripts/e2e_agent
    python scripts/phase2_compare.py \
        --instruct-activations results/phase1_probe/activations_multilayer.npz \
        --instruct-labels results/phase1_probe/labels.jsonl \
        --base-activations results/phase2_rlhf_tax/base_activations.npz \
        --base-labels results/phase2_rlhf_tax/base_labels.jsonl \
        --base-summary results/phase2_rlhf_tax/base_summary.json \
        --output-dir results/phase2_rlhf_tax
"""

import sys, json, argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Known instruct model baseline values from Phase 1 / A3 ─────────────────
INSTRUCT_KNOWN = {
    "baseline_accuracy": 0.208,
    "second_search_rate_baseline": 0.031,
    "pf_rate_a3": 0.026,
    "dissociation_rate_0doc": 0.938,
    "dissociation_rate_1doc": 0.977,
    "cosine_probe_vs_A3_direction": -0.013,
    "probe_auroc_l20": 0.862,
    "probe_balanced_acc_l20": 0.773,
}


def load_multilayer_npz(path: str) -> tuple:
    """Load activations NPZ. Returns (X_per_layer, y, sample_ids, margins)."""
    data = np.load(path, allow_pickle=True)
    X = {}
    for k in data.files:
        if k.startswith("layer_"):
            l = int(k.replace("layer_", ""))
            X[l] = data[k].astype(np.float32)
    y = data["y"].astype(np.int32)
    sample_ids = list(data["sample_ids"])
    margins = data.get("margins", np.zeros(len(y))).astype(np.float32)
    return X, y, sample_ids, margins


def load_labels_jsonl(path: str) -> list:
    """Load labels.jsonl as list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def train_probe(X: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple:
    """Train logistic regression probe on (X, y).

    Returns:
      (metrics dict, unit-norm probe direction in original space, scaler, clf)
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(sss.split(X_s, y))

    clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                             solver="lbfgs", random_state=seed)
    clf.fit(X_s[train_idx], y[train_idx])

    y_pred = clf.predict(X_s[test_idx])
    y_prob = clf.predict_proba(X_s[test_idx])[:, 1]

    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], y_pred)),
        "auroc": float(roc_auc_score(y[test_idx], y_prob)),
        "precision": float(precision_score(y[test_idx], y_pred, zero_division=0)),
        "recall": float(recall_score(y[test_idx], y_pred, zero_division=0)),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_label0": int((y == 0).sum()),
        "n_label1": int((y == 1).sum()),
    }

    # Retrain on all data for direction extraction
    clf_all = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                                 solver="lbfgs", random_state=seed)
    clf_all.fit(X_s, y)
    w_orig = clf_all.coef_[0] / scaler.scale_
    direction = (w_orig / np.linalg.norm(w_orig)).astype(np.float32)

    return metrics, direction, scaler, clf_all


def compute_action_direction_p20p80(
    X_l20: np.ndarray, margins: np.ndarray,
    low_pct: int = 20, high_pct: int = 80
) -> tuple:
    """Extract action direction using P20/P80 margin-based mean-diff.

    Direction convention: h_low_mean - h_high_mean
    (points toward "stop" / low-search-propensity, matching extract_search_direction_v2.py)

    Returns (direction, metadata dict).
    """
    p_low = np.percentile(margins, low_pct)
    p_high = np.percentile(margins, high_pct)

    low_mask = margins <= p_low
    high_mask = margins >= p_high

    h_low = X_l20[low_mask]
    h_high = X_l20[high_mask]

    print(f"    Action direction: P{low_pct}={p_low:.2f}, P{high_pct}={p_high:.2f}")
    print(f"    Low-margin (stop-propensity) samples: {low_mask.sum()}")
    print(f"    High-margin (search-propensity) samples: {high_mask.sum()}")

    if len(h_low) < 5 or len(h_high) < 5:
        print("    [WARNING] Too few samples for direction. Using P10/P90.")
        p_low = np.percentile(margins, 10)
        p_high = np.percentile(margins, 90)
        low_mask = margins <= p_low
        high_mask = margins >= p_high
        h_low = X_l20[low_mask]
        h_high = X_l20[high_mask]

    # h_low - h_high = "stop" direction (mirrors extract_search_direction_v2.py convention)
    action_dir = h_low.mean(axis=0) - h_high.mean(axis=0)
    norm = np.linalg.norm(action_dir)
    action_dir_normed = (action_dir / norm).astype(np.float32)

    meta = {
        "n_low": int(low_mask.sum()),
        "n_high": int(high_mask.sum()),
        "p_low": float(p_low),
        "p_high": float(p_high),
        "direction_norm": float(norm),
    }
    return action_dir_normed, meta


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two unit or unnormalized vectors."""
    a_n = a / (np.linalg.norm(a) + 1e-12)
    b_n = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_n, b_n))


def compute_dissociation_rates(y: np.ndarray, behavioral_stops: np.ndarray) -> dict:
    """Compute dissociation rates for 0-doc and 1-doc groups."""
    idx0 = (y == 0)
    idx1 = (y == 1)

    n0 = idx0.sum()
    n1 = idx1.sum()

    diss0 = behavioral_stops[idx0].sum() if n0 > 0 else 0
    diss1 = behavioral_stops[idx1].sum() if n1 > 0 else 0

    return {
        "n_label0": int(n0),
        "n_label1": int(n1),
        "dissociation_count_0doc": int(diss0),
        "dissociation_count_1doc": int(diss1),
        "dissociation_rate_0doc": float(diss0 / n0) if n0 > 0 else None,
        "dissociation_rate_1doc": float(diss1 / n1) if n1 > 0 else None,
    }


def analyze_model(
    name: str, X_per_layer: dict, y: np.ndarray,
    labels: list, margins: np.ndarray, seed: int = 42
) -> dict:
    """Full analysis pipeline for one model.

    Returns dict with probe metrics, direction, cosines, dissociation rates.
    """
    print(f"\n{'─'*55}")
    print(f"  Analyzing: {name}")
    print(f"  N={len(y)}, label=0: {(y==0).sum()}, label=1: {(y==1).sum()}")
    print(f"  Margin: mean={np.mean(margins):.2f}, std={np.std(margins):.2f}")

    # ── Probe at each available layer ────────────────────────────────────────
    layer_results = {}
    probe_direction_l20 = None

    for l in sorted(X_per_layer.keys()):
        X = X_per_layer[l]
        m, d, sc, clf = train_probe(X, y, seed=seed)
        layer_results[f"L{l}"] = m
        if l == 20:
            probe_direction_l20 = d
            probe_scaler = sc
            probe_clf = clf

    print(f"\n  Layer probe results:")
    print(f"  {'Layer':>5} | {'BalAcc':>8} | {'AUROC':>8}")
    for l_name, m in layer_results.items():
        flag = " <<<" if m["auroc"] == max(r["auroc"] for r in layer_results.values()) else ""
        print(f"  {l_name:>5} | {m['balanced_accuracy']:>8.3f} | {m['auroc']:>8.3f}{flag}")

    # ── Behavioral analysis ──────────────────────────────────────────────────
    # Truncate labels to match y length (they should be identical but guard against edge cases)
    labels_aligned = labels[:len(y)]
    behavioral_stops = np.array(
        [1 if r.get("behavioral_stop", True) else 0 for r in labels_aligned], dtype=np.int32)

    diss_rates = compute_dissociation_rates(y, behavioral_stops)

    # ── Action direction from step-1 margins (P20/P80) ───────────────────────
    if 20 in X_per_layer and len(margins) == len(y):
        print(f"\n  Computing action direction (L20, P20/P80 of step-1 margins)...")
        action_dir, action_meta = compute_action_direction_p20p80(
            X_per_layer[20], margins)
    else:
        action_dir = None
        action_meta = {}

    # ── Cosine computations ──────────────────────────────────────────────────
    cosines = {}

    if probe_direction_l20 is not None and action_dir is not None:
        cos_val = cosine_similarity(probe_direction_l20, action_dir)
        cosines["probe_vs_step1_action_dir"] = cos_val
        print(f"\n  Cosine(evidence_probe_L20, step1_action_dir_L20) = {cos_val:+.4f}")

        # Interpret sign:
        # - probe_direction points toward label=1 (sufficient evidence)
        # - action_direction = h_low - h_high = "stop" direction
        # POSITIVE cosine: stop-propensity co-aligned with sufficient-evidence rep (CORRECT)
        # NEGATIVE cosine: stop-propensity co-aligned with insufficient-evidence rep (DISSOCIATION)
        if cos_val < -0.10:
            interp = "DISSOCIATION SIGNATURE: stop direction aligns with insufficient evidence"
        elif cos_val > 0.10:
            interp = "HEALTHY BEHAVIOR: stop direction aligns with sufficient evidence"
        else:
            interp = "DECOUPLED: action and evidence representations are orthogonal"
        print(f"    → {interp}")
        cosines["interpretation"] = interp

    result = {
        "model": name,
        "n_samples": int(len(y)),
        "n_label0": int((y == 0).sum()),
        "n_label1": int((y == 1).sum()),
        "margin_stats": {
            "mean": float(np.mean(margins)),
            "std": float(np.std(margins)),
            "min": float(np.min(margins)),
            "max": float(np.max(margins)),
        },
        "probe_per_layer": layer_results,
        "probe_l20": layer_results.get("L20", {}),
        "dissociation": diss_rates,
        "cosines": cosines,
        "action_dir_meta": action_meta,
    }
    return result, probe_direction_l20, action_dir


def print_comparison_table(instruct_result: dict, base_result: dict, base_summary: dict):
    """Print the comparison table in the format specified in CLAUDE.md Phase 2."""
    def fmt(v, pct=False, pm=None):
        if v is None:
            return "?"
        if pct:
            return f"{v:.1%}"
        return f"{v:.3f}"

    i = instruct_result
    b = base_result

    i_l20 = i.get("probe_l20", {})
    b_l20 = b.get("probe_l20", {})
    i_diss = i.get("dissociation", {})
    b_diss = b.get("dissociation", {})
    i_cos = i.get("cosines", {})
    b_cos = b.get("cosines", {})

    rows = [
        ("Probe AUROC (L20)", fmt(i_l20.get("auroc")), fmt(b_l20.get("auroc"))),
        ("Probe BalAcc (L20)", fmt(i_l20.get("balanced_accuracy")), fmt(b_l20.get("balanced_accuracy"))),
        ("Evidence⊥Action cosine (step-1)", fmt(i_cos.get("probe_vs_step1_action_dir")),
         fmt(b_cos.get("probe_vs_step1_action_dir"))),
        ("Known cosine (A3 direction)", fmt(INSTRUCT_KNOWN["cosine_probe_vs_A3_direction"]), "N/A"),
        ("2nd search rate", fmt(INSTRUCT_KNOWN["second_search_rate_baseline"], pct=True),
         fmt(base_summary.get("second_search_rate"), pct=True)),
        ("Overall accuracy (EM)", fmt(INSTRUCT_KNOWN["baseline_accuracy"], pct=True),
         fmt(base_summary.get("accuracy"), pct=True)),
        ("Dissociation rate (0-doc)", fmt(INSTRUCT_KNOWN["dissociation_rate_0doc"], pct=True),
         fmt(b_diss.get("dissociation_rate_0doc"), pct=True)),
        ("Dissociation rate (1-doc)", fmt(INSTRUCT_KNOWN["dissociation_rate_1doc"], pct=True),
         fmt(b_diss.get("dissociation_rate_1doc"), pct=True)),
        ("PF rate", fmt(INSTRUCT_KNOWN["pf_rate_a3"], pct=True),
         fmt(base_summary.get("pf_rate"), pct=True)),
        ("N valid samples", str(i.get("n_samples", 486)), str(b.get("n_samples", "?"))),
    ]

    w = max(len(r[0]) for r in rows) + 2
    print(f"\n{'─'*65}")
    print(f"  COMPARISON TABLE: Instruct vs Base Model")
    print(f"{'─'*65}")
    print(f"  {'Metric':{w}} | {'Qwen2.5-7B-Instruct':>20} | {'Qwen2.5-7B-Base':>18}")
    print(f"  {'-'*w}-+----------------------+--------------------")
    for row in rows:
        print(f"  {row[0]:{w}} | {row[1]:>20} | {row[2]:>18}")
    print(f"{'─'*65}\n")


def interpret_findings(instruct_result: dict, base_result: dict) -> dict:
    """Classify into Scenario A/B/C/D from CLAUDE.md Phase 2 spec."""
    i_cos = instruct_result.get("cosines", {}).get("probe_vs_step1_action_dir", 0.0) or 0.0
    b_cos = base_result.get("cosines", {}).get("probe_vs_step1_action_dir", 0.0) or 0.0
    b_auroc = base_result.get("probe_l20", {}).get("auroc", 0.0)

    delta = b_cos - i_cos

    if b_auroc < 0.65:
        scenario = "D"
        desc = ("Base model probe AUROC < 0.65 — base model does not encode evidence state "
                "at L20. Cannot compare cosines meaningfully. Needs layer sweep for base model.")
    elif abs(delta) < 0.10:
        # Both near-zero or similar
        b_diss0 = base_result.get("dissociation", {}).get("dissociation_rate_0doc", None)
        i_diss0 = INSTRUCT_KNOWN.get("dissociation_rate_0doc", 0.938)
        if b_diss0 is not None and b_diss0 < i_diss0 * 0.7:
            scenario = "B"
            desc = ("Cosines similar but base model has substantially lower dissociation rate. "
                    "Decoupling exists pre-training, but RLHF amplifies behavioral consequence.")
        else:
            scenario = "C"
            desc = ("Both cosines near-zero AND similar dissociation rates. Decoupling is a "
                    "pre-training architectural phenomenon, not purely RLHF-related. "
                    "Pivot: fundamental representational property of transformer tool-use.")
    elif b_cos > i_cos + 0.10:
        scenario = "A"
        desc = (f"Base cosine ({b_cos:+.3f}) substantially more positive than instruct "
                f"({i_cos:+.3f}). Indicates MORE correct behavior in base model — RLHF "
                f"decoupled evidence and action representations (RLHF Tax confirmed).")
    else:
        scenario = "B_neg"
        desc = (f"Base cosine ({b_cos:+.3f}) more negative than instruct ({i_cos:+.3f}). "
                f"Base model's stop direction is MORE correlated with insufficient evidence — "
                f"which could indicate base model is better at recognizing insufficiency but "
                f"lacks the RLHF override to stop anyway.")

    return {
        "scenario": scenario,
        "description": desc,
        "instruct_cosine": i_cos,
        "base_cosine": b_cos,
        "delta": delta,
        "base_probe_auroc": b_auroc,
        "RLHF_tax_confirmed": scenario == "A",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruct-activations",
                        default="results/phase1_probe/activations_multilayer.npz")
    parser.add_argument("--instruct-labels",
                        default="results/phase1_probe/labels.jsonl")
    parser.add_argument("--base-activations",
                        default="results/phase2_rlhf_tax/base_activations.npz")
    parser.add_argument("--base-labels",
                        default="results/phase2_rlhf_tax/base_labels.jsonl")
    parser.add_argument("--base-summary",
                        default="results/phase2_rlhf_tax/base_summary.json")
    parser.add_argument("--output-dir", default="results/phase2_rlhf_tax")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  PHASE 2: RLHF TAX COMPARISON")
    print("=" * 65)

    # ── Load instruct data ────────────────────────────────────────────────────
    print("\nLoading instruct model data...")
    i_X, i_y, i_ids, i_margins_npz = load_multilayer_npz(args.instruct_activations)
    i_labels = load_labels_jsonl(args.instruct_labels)

    # Use margin_before from labels.jsonl (more reliable than zeros in npz)
    i_margin_map = {r["sample_id"]: r.get("margin_before", 0.0) for r in i_labels}
    i_margins = np.array([i_margin_map.get(sid, 0.0) for sid in i_ids], dtype=np.float32)
    print(f"  Instruct: N={len(i_y)}, layers={sorted(i_X.keys())}")

    # ── Load base model data ──────────────────────────────────────────────────
    print("\nLoading base model data...")
    if not Path(args.base_activations).exists():
        print(f"  [ERROR] Base activations not found: {args.base_activations}")
        print("  Run phase2_base_agent.py first.")
        return

    b_X, b_y, b_ids, b_margins = load_multilayer_npz(args.base_activations)
    b_labels = load_labels_jsonl(args.base_labels)
    b_summary = json.loads(Path(args.base_summary).read_text()) if Path(args.base_summary).exists() else {}
    print(f"  Base: N={len(b_y)}, layers={sorted(b_X.keys())}")

    # Align labels to NPZ sample_ids order
    def align_labels(ids, labels):
        id_to_label = {r["sample_id"]: r for r in labels}
        aligned = []
        for sid in ids:
            sid_str = sid.decode() if isinstance(sid, bytes) else str(sid)
            if sid_str in id_to_label:
                aligned.append(id_to_label[sid_str])
            else:
                # Fallback: empty record (behavioral_stop=True by default)
                aligned.append({"sample_id": sid_str, "behavioral_stop": True})
        return aligned

    i_labels_aligned = align_labels(i_ids, i_labels)
    b_labels_aligned = align_labels(b_ids, b_labels)

    # ── Analyze both models ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ANALYZING INSTRUCT MODEL")
    i_result, i_probe_dir, i_action_dir = analyze_model(
        "Qwen2.5-7B-Instruct", i_X, i_y, i_labels_aligned, i_margins, seed=args.seed)

    print("\n" + "=" * 65)
    print("  ANALYZING BASE MODEL")
    b_result, b_probe_dir, b_action_dir = analyze_model(
        "Qwen2.5-7B-Base", b_X, b_y, b_labels_aligned, b_margins, seed=args.seed)

    # ── Print comparison table ────────────────────────────────────────────────
    print_comparison_table(i_result, b_result, b_summary)

    # ── Interpretation ────────────────────────────────────────────────────────
    interpretation = interpret_findings(i_result, b_result)
    print("  INTERPRETATION:")
    print(f"  Scenario: {interpretation['scenario']}")
    print(f"  {interpretation['description']}")
    print(f"\n  RLHF Tax confirmed: {interpretation['RLHF_tax_confirmed']}")

    # ── Save results ──────────────────────────────────────────────────────────
    cosine_comparison = {
        "sign_convention": (
            "probe_direction points toward label=1 (sufficient evidence). "
            "action_direction = h_low_margin - h_high_margin (points toward 'stop'). "
            "Positive cosine = stop aligned with sufficient evidence (correct behavior). "
            "Negative cosine = stop aligned with insufficient evidence (dissociation)."
        ),
        "instruct": i_result,
        "base": b_result,
        "interpretation": interpretation,
        "known_instruct_values": INSTRUCT_KNOWN,
    }

    (out_dir / "cosine_comparison.json").write_text(
        json.dumps(cosine_comparison, indent=2, default=str))

    behavioral = {
        "instruct_baseline_acc": INSTRUCT_KNOWN["baseline_accuracy"],
        "instruct_second_search_rate": INSTRUCT_KNOWN["second_search_rate_baseline"],
        "instruct_pf_rate": INSTRUCT_KNOWN["pf_rate_a3"],
        "base_acc": b_summary.get("accuracy"),
        "base_second_search_rate": b_summary.get("second_search_rate"),
        "base_pf_rate": b_summary.get("pf_rate"),
        "instruct_dissociation_0doc": INSTRUCT_KNOWN["dissociation_rate_0doc"],
        "instruct_dissociation_1doc": INSTRUCT_KNOWN["dissociation_rate_1doc"],
        "base_dissociation_0doc": b_result.get("dissociation", {}).get("dissociation_rate_0doc"),
        "base_dissociation_1doc": b_result.get("dissociation", {}).get("dissociation_rate_1doc"),
    }
    (out_dir / "behavioral_comparison.json").write_text(
        json.dumps(behavioral, indent=2, default=str))

    # Full report
    report = {
        "instruct": i_result,
        "base": b_result,
        "interpretation": interpretation,
        "behavioral": behavioral,
        "known_instruct_values": INSTRUCT_KNOWN,
    }
    (out_dir / "phase2_report.json").write_text(
        json.dumps(report, indent=2, default=str))

    print(f"\nResults saved to {out_dir}/")
    print(f"  cosine_comparison.json")
    print(f"  behavioral_comparison.json")
    print(f"  phase2_report.json")


if __name__ == "__main__":
    main()
