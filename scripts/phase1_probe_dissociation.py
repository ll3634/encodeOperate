#!/usr/bin/env python3
"""
Phase 1: Evidence Sufficiency Probe on L20 Activations

Goal: Prove that L20 hidden states at the decision point encode evidence
sufficiency, even when the model behaviorally commits to stopping.

Label definition (evidence sufficiency, Method A - strict):
  Label 0 (insufficient): 0 supporting paragraph titles retrieved as actual
      documents in step-0 observation. Acc=13.4% — model has nothing to work with.
  Label 1 (some evidence): 1+ supporting paragraph titles retrieved as actual
      documents. Acc=23.4% — model has at least the first hop.

Why NOT Method B (title mentions anywhere in text):
  Both groups have identical accuracy (21.4% vs 21.1%), confirming it's
  measuring noise, not evidence sufficiency.

Decision point (consistent with A3):
  After step 0 (first search), before step 1. Prompt = question + step-0
  scratchpad (action + observation). This is exactly the prompt A3 steered.

Output (results/phase1_probe/):
  activations_l20.npz        — L20 hidden states [N, hidden_dim]
  labels.jsonl               — per-sample label + metadata
  probe_results.json         — probe metrics + dissociation rate
  probe_direction.npz        — trained probe weight vector
  probe_direction_meta.json  — metadata for the direction file

Usage:
  cd tmc/scripts/e2e_agent
  python scripts/phase1_probe_dissociation.py \\
      --baseline-trace results/l20_rho020_n500/baseline_results.jsonl \\
      --hotpotqa-data data/hotpotqa/hotpot_dev_distractor_v1.json \\
      --model /home/featurize/work/models/Qwen2.5-7B-Instruct \\
      --output-dir results/phase1_probe \\
      --steering-dir steering/directions
"""

import os
import sys
import re
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    balanced_accuracy_score, confusion_matrix
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder
from steering.hook_utils import get_model_layers


# ── Label computation ──────────────────────────────────────────────────────────

def extract_retrieved_doc_titles(observation: str) -> list:
    """
    Extract titles of documents actually retrieved by the search tool.
    Format in observation: '[N] Title: text...'
    """
    titles = re.findall(r'\[\d+\]\s*([^:]+):', observation)
    return [t.strip() for t in titles]


def title_match(sf_title: str, retrieved_titles: list) -> bool:
    """Case-insensitive bidirectional substring match."""
    sf_lower = sf_title.lower()
    for rt in retrieved_titles:
        rt_lower = rt.lower()
        if sf_lower in rt_lower or rt_lower in sf_lower:
            return True
    return False


def compute_evidence_label(sf_titles: list, observation: str) -> dict:
    """
    Compute evidence sufficiency label from supporting_facts and observation.

    Method A (strict): counts how many supporting paragraph titles were
    retrieved as actual documents in the observation. Returns label=0 if
    0 were retrieved, label=1 if 1+ were retrieved.

    Returns dict with n_sf_retrieved, label, retrieved_doc_titles.
    """
    retrieved_titles = extract_retrieved_doc_titles(observation)
    n_sf_retrieved = sum(1 for t in sf_titles if title_match(t, retrieved_titles))
    label = 0 if n_sf_retrieved == 0 else 1
    return {
        "n_sf_retrieved": n_sf_retrieved,
        "n_sf_total": len(sf_titles),
        "label": label,
        "retrieved_doc_titles": retrieved_titles,
    }


# ── Data loading ───────────────────────────────────────────────────────────────

def load_episodes(baseline_path: str, hotpotqa_path: str):
    """
    Load baseline traces and HotpotQA annotations.
    Returns list of episode dicts with all fields needed for Phase 1.
    """
    with open(hotpotqa_path) as f:
        raw = json.load(f)
    hotpot_by_id = {s["_id"]: s for s in raw}

    episodes = []
    skipped = 0
    with open(baseline_path) as f:
        for line in f:
            ep = json.loads(line)
            sid = ep["sample_id"]
            hp = hotpot_by_id.get(sid)
            if hp is None:
                skipped += 1
                continue

            steps = ep.get("steps", [])
            s0 = steps[0] if steps else None

            # Decision point requires a valid first search with observation
            if not s0 or s0.get("action") != "search" or not s0.get("observation"):
                skipped += 1
                continue

            obs = s0["observation"]
            sf_titles = list(set(sf[0] for sf in hp.get("supporting_facts", [])))

            ev = compute_evidence_label(sf_titles, obs)

            # Behavioral label: did step 1 choose "search" (continue)?
            s1 = steps[1] if len(steps) > 1 else None
            behavioral_continue = bool(s1 and s1.get("action") == "search")

            episodes.append({
                "sample_id": sid,
                "question": ep["question"],
                "gold_answer": ep.get("gold_answer", ""),
                "is_correct": ep.get("is_correct", False),
                # Step-0 context for prompt reconstruction
                "step0_query": s0["action_input"],
                "step0_obs": obs,
                # Evidence label
                "label": ev["label"],
                "n_sf_retrieved": ev["n_sf_retrieved"],
                "n_sf_total": ev["n_sf_total"],
                "sf_titles": sf_titles,
                "retrieved_doc_titles": ev["retrieved_doc_titles"],
                # Behavioral label
                "behavioral_continue": behavioral_continue,
                "behavioral_stop": not behavioral_continue,
                # margin at decision point (logged by agent)
                "margin_before": s1.get("margin_before") if s1 else None,
            })

    print(f"Loaded {len(episodes)} valid episodes (skipped {skipped})")
    n0 = sum(1 for e in episodes if e["label"] == 0)
    n1 = sum(1 for e in episodes if e["label"] == 1)
    n_stop = sum(1 for e in episodes if e["behavioral_stop"])
    n_cont = sum(1 for e in episodes if e["behavioral_continue"])
    print(f"  Label 0 (insufficient, 0 docs retrieved): {n0}")
    print(f"  Label 1 (some evidence, 1+ docs retrieved): {n1}")
    print(f"  Behavioral stop: {n_stop}  continue: {n_cont}")
    return episodes


# ── Activation extraction ──────────────────────────────────────────────────────

def build_decision_point_prompt(ep: dict, tokenizer, prompt_builder: PromptBuilder):
    """
    Reconstruct the decision-point prompt: question + step-0 scratchpad.
    This is identical to what A3 used for steering.
    """
    steps = [{"action": "search", "action_input": ep["step0_query"],
               "observation": ep["step0_obs"]}]
    messages = prompt_builder.build_full_prompt(ep["question"], steps)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


def extract_l20_activations(model, tokenizer, episodes, layer: int = 20, batch_size: int = 1):
    """
    Extract L20 hidden state at the last token position for each episode.
    This matches the position used in A3 (position=-1 in AgentConfig).
    Returns numpy array [N, hidden_dim].
    """
    pb = PromptBuilder(tools=["search", "calculator"])
    layers = get_model_layers(model)
    n_layers = len(layers)
    actual_layer = layer if layer >= 0 else n_layers + layer
    print(f"Extracting from layer {actual_layer} / {n_layers} (0-indexed)")

    device = next(model.parameters()).device
    hiddens = []
    failed_ids = []

    for i, ep in enumerate(episodes):
        prompt = build_decision_point_prompt(ep, tokenizer, pb)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        captured = {}

        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            # Last token position, matching A3
            captured["hidden"] = h[0, -1, :].detach().float().cpu().numpy()

        handle = layers[actual_layer].register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                model(input_ids)
        except Exception as exc:
            print(f"  [{i+1}] ERROR on {ep['sample_id']}: {exc}")
            failed_ids.append(ep["sample_id"])
            handle.remove()
            continue
        handle.remove()

        if "hidden" not in captured:
            failed_ids.append(ep["sample_id"])
            continue

        hiddens.append(captured["hidden"])

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(episodes)}] extracted")

    print(f"Extracted {len(hiddens)} / {len(episodes)} (failed: {len(failed_ids)})")
    return np.array(hiddens, dtype=np.float32), failed_ids


# ── Probe training ─────────────────────────────────────────────────────────────

def train_probe(X: np.ndarray, y: np.ndarray, C: float = 1.0, n_splits: int = 5):
    """
    Train logistic regression probe with stratified 80/20 split.
    Reports accuracy, AUROC, precision, recall on held-out test set.
    Also runs 5-fold CV for balanced accuracy.
    Returns: test_metrics dict, probe direction (unit norm), scaler.
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # 80/20 stratified split for test metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(
        class_weight="balanced", C=C, max_iter=2000,
        solver="lbfgs", random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    test_metrics = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_test_label0": int((y_test == 0).sum()),
        "n_test_label1": int((y_test == 1).sum()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "auroc": float(roc_auc_score(y_test, y_prob)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    # 5-fold CV balanced accuracy on full dataset
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_s, y, cv=cv, scoring="balanced_accuracy")
    test_metrics["cv_balanced_accuracy_mean"] = float(cv_scores.mean())
    test_metrics["cv_balanced_accuracy_std"] = float(cv_scores.std())
    test_metrics["cv_per_fold"] = [float(s) for s in cv_scores]

    # Retrain on full dataset for the final direction
    clf_full = LogisticRegression(
        class_weight="balanced", C=C, max_iter=2000,
        solver="lbfgs", random_state=42
    )
    clf_full.fit(X_s, y)

    # Probe direction: weight vector in ORIGINAL (unscaled) space
    # Direction points toward label=1 (some evidence) from label=0 (insufficient)
    w_scaled = clf_full.coef_[0]           # weights in scaled space
    w_orig = w_scaled / scaler.scale_     # project back to original space
    direction = (w_orig / np.linalg.norm(w_orig)).astype(np.float32)

    return test_metrics, direction, clf_full, scaler


# ── Dissociation analysis ──────────────────────────────────────────────────────

def compute_dissociation(episodes, valid_episodes, X, y, clf, scaler):
    """
    Compute dissociation rate: probe predicts label=0 (insufficient) AND
    model behaviorally stopped.

    This is the core of Phase 1: the model internally 'knows' evidence is
    insufficient but externally commits to stopping.
    """
    X_s = scaler.transform(X)
    probe_pred = clf.predict(X_s)

    results = []
    for i, ep in enumerate(valid_episodes):
        results.append({
            "sample_id": ep["sample_id"],
            "label_true": int(y[i]),
            "probe_pred": int(probe_pred[i]),
            "behavioral_stop": ep["behavioral_stop"],
            "behavioral_continue": ep["behavioral_continue"],
            "is_correct": ep["is_correct"],
            "n_sf_retrieved": ep["n_sf_retrieved"],
        })

    # 2×2 core table: (label, behavior)
    truly_insufficient = [r for r in results if r["label_true"] == 0]
    truly_sufficient = [r for r in results if r["label_true"] == 1]

    # Dissociation: insufficient + stop
    dissociation_cases = [r for r in truly_insufficient if r["behavioral_stop"]]
    # Probe-confirmed dissociation: probe also says insufficient
    probe_confirmed = [r for r in dissociation_cases if r["probe_pred"] == 0]

    # Behavioral mismatch rate (without probe)
    raw_mismatch_rate = len(dissociation_cases) / len(truly_insufficient) if truly_insufficient else 0.0
    # Probe-confirmed dissociation rate
    probe_dissociation_rate = len(probe_confirmed) / len(truly_insufficient) if truly_insufficient else 0.0

    # Confusion matrix of (label, behavior)
    conf = {
        "insuf_stop": sum(1 for r in results if r["label_true"]==0 and r["behavioral_stop"]),
        "insuf_cont": sum(1 for r in results if r["label_true"]==0 and r["behavioral_continue"]),
        "suf_stop":   sum(1 for r in results if r["label_true"]==1 and r["behavioral_stop"]),
        "suf_cont":   sum(1 for r in results if r["label_true"]==1 and r["behavioral_continue"]),
    }

    diag = {
        "n_total": len(results),
        "n_insufficient": len(truly_insufficient),
        "n_sufficient": len(truly_sufficient),
        "dissociation_cases": len(dissociation_cases),
        "raw_mismatch_rate": raw_mismatch_rate,
        "probe_confirmed_dissociation": len(probe_confirmed),
        "probe_dissociation_rate": probe_dissociation_rate,
        "label_behavior_2x2": conf,
        "accuracy_by_label": {
            "insufficient": sum(r["is_correct"] for r in truly_insufficient) / max(len(truly_insufficient), 1),
            "sufficient": sum(r["is_correct"] for r in truly_sufficient) / max(len(truly_sufficient), 1),
        }
    }
    return diag, results


# ── Steering direction alignment ───────────────────────────────────────────────

def compute_direction_alignment(probe_direction: np.ndarray, steering_dir_path: str):
    """
    Compute cosine similarity between probe direction and A3 steering direction.
    High similarity means the A3 intervention was implicitly correcting the
    representation-level dissociation.
    """
    if not Path(steering_dir_path).exists():
        return None
    try:
        d = np.load(steering_dir_path)
        steering_dir = d["decision_direction"].astype(np.float32)
        cos_sim = float(np.dot(probe_direction, steering_dir) /
                        (np.linalg.norm(probe_direction) * np.linalg.norm(steering_dir) + 1e-10))
        return {
            "steering_direction_path": steering_dir_path,
            "cosine_similarity": cos_sim,
            "abs_cosine_similarity": abs(cos_sim),
        }
    except Exception as e:
        return {"error": str(e)}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Evidence Sufficiency Probe")
    parser.add_argument("--baseline-trace", required=True,
                        help="Path to baseline_results.jsonl from A3")
    parser.add_argument("--hotpotqa-data", required=True,
                        help="Path to hotpot_dev_distractor_v1.json")
    parser.add_argument("--model", default="/home/featurize/work/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", default="results/phase1_probe")
    parser.add_argument("--steering-dir", default="steering/directions",
                        help="Directory containing direction_*.npz files")
    parser.add_argument("--layer", type=int, default=20,
                        help="Layer to extract activations from (A3 uses L20)")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Logistic regression regularization strength")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip model forward pass; load saved activations from output-dir")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  PHASE 1: EVIDENCE SUFFICIENCY PROBE")
    print("=" * 70)
    print(f"  Baseline trace: {args.baseline_trace}")
    print(f"  Layer: {args.layer}")
    print(f"  Output: {out_dir}")
    print()

    # ── Step 1: Load episodes and compute labels ──────────────────────────────
    print("Step 1: Loading episodes and computing evidence sufficiency labels...")
    episodes = load_episodes(args.baseline_trace, args.hotpotqa_data)
    print()

    # Save label file (without activations)
    labels_path = out_dir / "labels.jsonl"
    with open(labels_path, "w") as f:
        for ep in episodes:
            row = {k: ep[k] for k in [
                "sample_id", "question", "gold_answer", "is_correct",
                "label", "n_sf_retrieved", "n_sf_total", "sf_titles",
                "retrieved_doc_titles", "behavioral_continue", "behavioral_stop",
                "margin_before"
            ]}
            f.write(json.dumps(row) + "\n")
    print(f"Labels saved → {labels_path}")
    print()

    # ── Step 2: Extract L20 activations ──────────────────────────────────────
    act_path = out_dir / "activations_l20.npz"

    if args.skip_extraction and act_path.exists():
        print("Step 2: Loading saved activations (--skip-extraction)...")
        d = np.load(act_path, allow_pickle=True)
        X_all = d["activations"]
        saved_ids = list(d["sample_ids"])
        # Filter episodes to match saved IDs
        id_to_ep = {ep["sample_id"]: ep for ep in episodes}
        valid_episodes = [id_to_ep[sid] for sid in saved_ids if sid in id_to_ep]
        print(f"  Loaded {len(X_all)} activations")
    else:
        print(f"Step 2: Extracting L{args.layer} activations (this requires GPU)...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"  Loading model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, attn_implementation="eager"
        )
        model.eval()
        print(f"  Model loaded. Extracting activations...")

        X_all, failed_ids = extract_l20_activations(model, tokenizer, episodes, layer=args.layer)

        # Remove failed episodes
        failed_set = set(failed_ids)
        valid_episodes = [ep for ep in episodes if ep["sample_id"] not in failed_set]

        # Save
        np.savez(
            str(act_path),
            activations=X_all,
            sample_ids=np.array([ep["sample_id"] for ep in valid_episodes]),
            layer=args.layer,
        )
        print(f"  Activations saved → {act_path}  shape={X_all.shape}")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()
    print()

    # ── Step 3: Prepare labels ────────────────────────────────────────────────
    y = np.array([ep["label"] for ep in valid_episodes], dtype=np.int32)
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    print(f"Step 3: Label distribution: Label-0={n0} (insufficient), Label-1={n1} (some evidence)")
    print()

    # ── Kill criterion check ─────────────────────────────────────────────────
    # Check class balance — if one class < 5% of total, probe is unreliable
    min_class = min(n0, n1)
    total = n0 + n1
    if min_class < 0.05 * total:
        print(f"WARNING: Extreme class imbalance ({min_class}/{total}). "
              f"Probe results may be unreliable.")

    # ── Step 4: Train probe ───────────────────────────────────────────────────
    print("Step 4: Training logistic regression probe (stratified 80/20 split + 5-fold CV)...")
    test_metrics, probe_direction, clf, scaler = train_probe(X_all, y, C=args.C)

    print()
    print("  PROBE RESULTS:")
    print(f"    Test accuracy:          {test_metrics['accuracy']:.3f}")
    print(f"    Test balanced accuracy: {test_metrics['balanced_accuracy']:.3f}")
    print(f"    Test AUROC:             {test_metrics['auroc']:.3f}")
    print(f"    Test precision (L0):    {test_metrics['precision']:.3f}")
    print(f"    Test recall (L0):       {test_metrics['recall']:.3f}")
    print(f"    CV balanced acc:        {test_metrics['cv_balanced_accuracy_mean']:.3f} "
          f"± {test_metrics['cv_balanced_accuracy_std']:.3f}")
    print(f"    Confusion matrix:       {test_metrics['confusion_matrix']}")
    print()

    # Kill criterion
    kill = test_metrics["balanced_accuracy"] < 0.65
    if kill:
        print("  *** KILL CRITERION TRIGGERED: balanced_accuracy < 0.65 ***")
        print("  *** Core thesis does not hold at L20. Reassess before proceeding. ***")
    else:
        print("  Probe above kill threshold (balanced_accuracy >= 0.65) ✓")
    print()

    # ── Step 5: Dissociation analysis ────────────────────────────────────────
    print("Step 5: Computing dissociation rate...")
    diag, per_sample_results = compute_dissociation(
        episodes, valid_episodes, X_all, y, clf, scaler
    )

    print()
    print("  DISSOCIATION RESULTS:")
    print(f"    N total valid:              {diag['n_total']}")
    print(f"    N insufficient (label=0):   {diag['n_insufficient']}")
    print(f"    N some evidence (label=1):  {diag['n_sufficient']}")
    print()
    print("  Label × Behavior 2×2 table:")
    c = diag["label_behavior_2x2"]
    print(f"                     STOP   CONTINUE")
    print(f"    Insufficient:     {c['insuf_stop']:4d}    {c['insuf_cont']:4d}")
    print(f"    Some evidence:    {c['suf_stop']:4d}    {c['suf_cont']:4d}")
    print()
    print(f"    Raw mismatch rate (insuf+stop/all_insuf):     "
          f"{diag['dissociation_cases']}/{diag['n_insufficient']} = "
          f"{diag['raw_mismatch_rate']:.1%}")
    print(f"    Probe-confirmed dissociation rate:           "
          f"{diag['probe_confirmed_dissociation']}/{diag['n_insufficient']} = "
          f"{diag['probe_dissociation_rate']:.1%}")
    print()
    acc = diag["accuracy_by_label"]
    print(f"    Accuracy by label (validates label quality):")
    print(f"      Insufficient: {acc['insufficient']:.3f}")
    print(f"      Some evidence: {acc['sufficient']:.3f}")
    print()

    # ── Step 6: Steering direction alignment ─────────────────────────────────
    print("Step 6: Computing alignment with A3 steering direction...")
    # Primary A3 direction: layer-20 version
    steering_candidates = [
        str(Path(args.steering_dir) / "direction_search_v3_layer20.npz"),
        str(Path(args.steering_dir) / "direction_probe_layer20.npz"),
    ]
    alignments = {}
    for sd_path in steering_candidates:
        name = Path(sd_path).stem
        result = compute_direction_alignment(probe_direction, sd_path)
        if result:
            alignments[name] = result
            if "cosine_similarity" in result:
                print(f"    {name}: cos_sim = {result['cosine_similarity']:.4f} "
                      f"(|cos| = {result['abs_cosine_similarity']:.4f})")
    print()

    # ── Step 7: Save probe direction ─────────────────────────────────────────
    probe_dir_path = out_dir / "probe_direction.npz"
    direction_rms = float(np.sqrt(np.mean(probe_direction ** 2)))
    np.savez(
        str(probe_dir_path),
        decision_direction=probe_direction,
        layer=args.layer,
        method="phase1_evidence_sufficiency",
        n_samples=total,
        n_label0=int(n0),
        n_label1=int(n1),
        balanced_accuracy=test_metrics["balanced_accuracy"],
        auroc=test_metrics["auroc"],
        cv_balanced_accuracy=test_metrics["cv_balanced_accuracy_mean"],
    )
    print(f"Probe direction saved → {probe_dir_path}  (RMS={direction_rms:.6f})")

    # Save direction metadata
    meta = {
        "timestamp": datetime.now().isoformat(),
        "method": "phase1_evidence_sufficiency_label_method_a",
        "layer": args.layer,
        "label_definition": {
            "0": "0 supporting paragraph docs retrieved by BM25 in step-0 observation",
            "1": "1+ supporting paragraph docs retrieved",
            "why_method_a": (
                "Method A (strict doc-title match) gives acc gap: label0=13.4% vs label1=23.4%. "
                "Method B (full-text mention) gives no gap: 21.4% vs 21.1%. Method A is valid."
            ),
        },
        "n_samples": total,
        "n_label0": int(n0),
        "n_label1": int(n1),
        "direction_rms": direction_rms,
        "probe_test_metrics": test_metrics,
        "kill_criterion_triggered": kill,
    }
    with open(out_dir / "probe_direction_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── Full results JSON ─────────────────────────────────────────────────────
    results = {
        "config": {
            "timestamp": datetime.now().isoformat(),
            "baseline_trace": args.baseline_trace,
            "hotpotqa_data": args.hotpotqa_data,
            "model": args.model,
            "layer": args.layer,
            "C_regularization": args.C,
        },
        "data_stats": {
            "n_episodes_loaded": len(episodes),
            "n_valid_for_probe": total,
            "n_label0_insufficient": int(n0),
            "n_label1_some_evidence": int(n1),
            "label_fraction_insufficient": float(n0 / total),
        },
        "probe_metrics": test_metrics,
        "kill_criterion_triggered": kill,
        "dissociation": diag,
        "direction_alignment": alignments,
    }
    results_path = out_dir / "probe_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results → {results_path}")

    # Save per-sample results
    per_sample_path = out_dir / "per_sample_results.jsonl"
    with open(per_sample_path, "w") as f:
        for row in per_sample_results:
            f.write(json.dumps(row) + "\n")
    print(f"Per-sample → {per_sample_path}")

    print()
    print("=" * 70)
    print("  PHASE 1 SUMMARY")
    print("=" * 70)
    print(f"  Probe balanced accuracy: {test_metrics['balanced_accuracy']:.3f}")
    print(f"  AUROC:                   {test_metrics['auroc']:.3f}")
    print(f"  CV balanced accuracy:    {test_metrics['cv_balanced_accuracy_mean']:.3f} "
          f"± {test_metrics['cv_balanced_accuracy_std']:.3f}")
    print(f"  Kill criterion:          {'TRIGGERED ✗' if kill else 'NOT triggered ✓'}")
    print(f"  Dissociation rate:       {diag['probe_dissociation_rate']:.1%} "
          f"(probe-confirmed)")
    print(f"  Raw mismatch rate:       {diag['raw_mismatch_rate']:.1%}")
    for name, aln in alignments.items():
        if "cosine_similarity" in aln:
            print(f"  Align with {name}: {aln['cosine_similarity']:.4f}")
    print()


if __name__ == "__main__":
    main()
