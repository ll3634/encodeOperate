#!/usr/bin/env python3
"""
Phase 1 Multi-Layer Probe (Evidence Sufficiency Labels, Label v1)

Steps:
  1. Load labels.jsonl (already computed by phase1_probe_dissociation.py)
  2. Load model and extract L12/L16/L20/L24 activations at decision points
  3. Train logistic regression probe at each layer; report acc, AUROC, precision, recall
  4. Causal validation: run L20 probe on the 19 rescued_via_search samples from A3
  5. Direction alignment: cosine similarity between probe weight vector and steering direction

Usage:
    cd tmc/scripts/e2e_agent
    python scripts/phase1_multilayer_probe.py \
        --labels results/phase1_probe/labels.jsonl \
        --baseline-trace results/l20_rho020_n500/baseline_results.jsonl \
        --hotpotqa-data data/hotpotqa/hotpot_dev_distractor_v1.json \
        --steering-dir steering/directions \
        --output-dir results/phase1_probe \
        --model Qwen/Qwen2.5-7B-Instruct
"""

import os, sys, json, argparse
from pathlib import Path

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             precision_score, recall_score, classification_report)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder
from steering.hook_utils import get_model_layers


# ─── 19 rescued_via_search sample IDs from A3 experiment ───────────────────
A3_RESCUED_VIA_SEARCH = [
    "5abaee845542996606241696",
    "5abbcfaf5542993f40c73ba9",
    "5ae2eda355429928c4239570",
    "5a8782f25542996e4f308818",
    "5a8f51185542992414482a3d",
    "5a85b2895542994c784ddb49",
    "5ae256435542992decbdccc3",
    "5ab29956554299194fa9342d",
    "5ae55d1e55429960a22e02cb",
    "5ab9cfe655429970cfb8ebaf",
    "5a821c95554299676cceb219",
    "5abdba405542993f32c2a023",
    "5abf92c45542993fe9a41e07",
    "5ac2a35055429967731025ce",
    "5ae7535c5542997b22f6a6d8",
    "5ae47cab5542996836b02cb9",
    "5a79311755429970f5fffe67",
    "5a7e02b75542997cc2c474f3",
    "5a83c2e25542996488c2e4bc",
]


def load_labels(labels_path):
    labels = []
    with open(labels_path) as f:
        for line in f:
            labels.append(json.loads(line))
    return labels


def build_decision_prompt(ep, baseline_trace, hotpotqa_data, pb):
    """Reconstruct step-1 decision prompt from baseline trace."""
    steps = ep.get("steps", [])
    if not steps:
        return None
    s0 = steps[0]
    if s0.get("action") != "search" or not s0.get("observation"):
        return None
    history = [{"action": "search", "action_input": s0["action_input"],
                "observation": s0["observation"]}]
    messages = pb.build_full_prompt(ep["question"], history)
    return messages


def collect_activations(model, tokenizer, episodes, label_map, layers_to_capture):
    """Single forward pass per episode, capture hidden states at all target layers."""
    pb = PromptBuilder(tools=["search", "calculator"])
    model_layers = get_model_layers(model)
    device = next(model.parameters()).device

    all_hidden = {l: [] for l in layers_to_capture}
    labels, sample_ids = [], []
    skipped = 0

    for i, ep in enumerate(episodes):
        sid = ep["sample_id"]
        if sid not in label_map:
            skipped += 1
            continue

        steps = ep.get("steps", [])
        if not steps or steps[0].get("action") != "search" or not steps[0].get("observation"):
            skipped += 1
            continue

        history = [{"action": "search", "action_input": steps[0]["action_input"],
                    "observation": steps[0]["observation"]}]
        messages = pb.build_full_prompt(ep["question"], history)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        captured = {}

        def make_hook(l_idx):
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[l_idx] = h[0, -1, :].detach().float().cpu().numpy()
            return hook_fn

        handles = [model_layers[l].register_forward_hook(make_hook(l))
                   for l in layers_to_capture]
        try:
            with torch.no_grad():
                model(input_ids)
        except Exception as e:
            for h in handles:
                h.remove()
            print(f"  [{i+1}] ERROR {sid[:20]}: {e}")
            skipped += 1
            continue
        for h in handles:
            h.remove()

        if len(captured) != len(layers_to_capture):
            skipped += 1
            continue

        for l in layers_to_capture:
            all_hidden[l].append(captured[l])
        labels.append(label_map[sid])
        sample_ids.append(sid)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(episodes)}] collected (skipped={skipped})")

    y = np.array(labels, dtype=np.int32)
    X_per_layer = {l: np.array(vecs, dtype=np.float32) for l, vecs in all_hidden.items()}
    print(f"\nCollected {len(y)} samples: label=0 (insuff): {(y==0).sum()}, label=1 (suff): {(y==1).sum()}")
    return X_per_layer, y, sample_ids


def train_probe(X, y, seed=42):
    """Train logistic regression with 80/20 stratified split."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(sss.split(X_s, y))

    clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                             solver="lbfgs", random_state=seed)
    clf.fit(X_s[train_idx], y[train_idx])

    y_pred = clf.predict(X_s[test_idx])
    y_prob = clf.predict_proba(X_s[test_idx])[:, 1]

    bal_acc = balanced_accuracy_score(y[test_idx], y_pred)
    auroc   = roc_auc_score(y[test_idx], y_prob)
    prec    = precision_score(y[test_idx], y_pred, zero_division=0)
    rec     = recall_score(y[test_idx], y_pred, zero_division=0)

    # Train on all data for direction
    clf_all = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                                 solver="lbfgs", random_state=seed)
    clf_all.fit(X_s, y)
    w_orig = clf_all.coef_[0] / scaler.scale_
    direction = (w_orig / np.linalg.norm(w_orig)).astype(np.float32)

    return {
        "balanced_accuracy": float(bal_acc),
        "auroc": float(auroc),
        "precision": float(prec),
        "recall": float(rec),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_test_label0": int((y[test_idx]==0).sum()),
        "n_test_label1": int((y[test_idx]==1).sum()),
    }, direction, scaler, clf_all


def causal_validation(X_l20, y, sample_ids, rescued_ids, scaler, clf):
    """Check how many A3 rescued_via_search samples are predicted insufficient."""
    rescued_set = set(rescued_ids)
    indices = [i for i, sid in enumerate(sample_ids) if sid in rescued_set]

    if not indices:
        print("  WARNING: none of the 19 rescued IDs found in collected samples!")
        return None

    X_rescued = X_l20[indices]
    y_rescued = y[indices]
    X_rescaled = scaler.transform(X_rescued)
    y_pred = clf.predict(X_rescaled)

    n_found = len(indices)
    n_pred_insuff = int((y_pred == 0).sum())
    n_true_insuff = int((y_rescued == 0).sum())
    print(f"\n=== Causal Validation (A3 rescued_via_search) ===")
    print(f"  Found {n_found}/19 rescued samples in probe dataset")
    print(f"  True label=0 (insufficient): {n_true_insuff}/{n_found}")
    print(f"  Probe predicts label=0:       {n_pred_insuff}/{n_found} = {n_pred_insuff/n_found:.1%}")

    passed = n_pred_insuff / n_found >= 0.80
    print(f"  Causal validation {'PASSED ✓' if passed else 'FAILED ✗'} (threshold: 80%)")

    return {
        "n_found": n_found,
        "n_pred_insufficient": n_pred_insuff,
        "n_true_insufficient": n_true_insuff,
        "pct_pred_insufficient": float(n_pred_insuff / n_found),
        "passed": passed,
        "per_sample": [
            {"sample_id": sample_ids[idx], "true_label": int(y[idx]),
             "pred_label": int(y_pred[j])}
            for j, idx in enumerate(indices)
        ]
    }


def direction_alignment(probe_direction, steering_dir_path, layer):
    """Compute cosine similarity between probe direction and steering direction."""
    if not Path(steering_dir_path).exists():
        print(f"  Steering direction not found: {steering_dir_path}")
        return None

    data = np.load(steering_dir_path)
    keys = list(data.keys())

    # Try common key names (prefer normalized version if available)
    for key in ["decision_direction_normalized", "decision_direction", "direction", "steering_direction"]:
        if key in data:
            steer_dir = data[key].astype(np.float32)
            break
    else:
        print(f"  No recognized direction key in {keys}")
        return None

    steer_layer = int(data.get("layer", -1))

    # Normalize
    probe_n = probe_direction / (np.linalg.norm(probe_direction) + 1e-12)
    steer_n = steer_dir / (np.linalg.norm(steer_dir) + 1e-12)

    cos = float(np.dot(probe_n, steer_n))
    print(f"\n=== Direction Alignment ===")
    print(f"  Steering direction: {steering_dir_path} (layer={steer_layer})")
    print(f"  Probe direction layer: {layer}")
    print(f"  Cosine similarity: {cos:.4f}")

    strong = abs(cos) > 0.5
    print(f"  Alignment {'STRONG ✓ (>0.5)' if strong else 'WEAK (<=0.5)'}")

    return {
        "cosine_similarity": cos,
        "strong_alignment": strong,
        "steering_layer": steer_layer,
        "probe_layer": layer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="results/phase1_probe/labels.jsonl")
    parser.add_argument("--baseline-trace", default="results/l20_rho020_n500/baseline_results.jsonl")
    parser.add_argument("--hotpotqa-data", default="data/hotpotqa/hotpot_dev_distractor_v1.json")
    parser.add_argument("--steering-dir", default="steering/directions")
    parser.add_argument("--output-dir", default="results/phase1_probe")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-path", default=None,
                        help="Optional PEFT adapter dir to merge on top of --model.")
    parser.add_argument("--layers", nargs="+", type=int, default=[12, 16, 20, 24])
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Load saved activations from npz instead of re-running model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  PHASE 1 MULTI-LAYER PROBE (Evidence Sufficiency, Label v1)")
    print("=" * 65)

    # Load labels
    label_records = load_labels(args.labels)
    label_map = {r["sample_id"]: r["label"] for r in label_records}
    print(f"Loaded {len(label_map)} labels: label=0: {sum(1 for v in label_map.values() if v==0)}, "
          f"label=1: {sum(1 for v in label_map.values() if v==1)}")

    # Load or extract activations
    npz_path = out_dir / "activations_multilayer.npz"
    if args.skip_extraction and npz_path.exists():
        print(f"\nLoading saved activations from {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        X_per_layer = {int(k.replace("layer_", "")): data[k] for k in data.files
                       if k.startswith("layer_")}
        y = data["y"]
        sample_ids = list(data["sample_ids"])
        print(f"Loaded {len(y)} samples, layers: {sorted(X_per_layer.keys())}")
    else:
        # Load baseline trace
        episodes = []
        with open(args.baseline_trace) as f:
            for line in f:
                episodes.append(json.loads(line))
        print(f"Loaded {len(episodes)} baseline episodes")

        # Load model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"\nLoading model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, attn_implementation="eager")
        if args.adapter_path:
            from peft import PeftModel
            print(f"Loading adapter: {args.adapter_path}")
            model = PeftModel.from_pretrained(model, args.adapter_path)
            model = model.merge_and_unload()
        model.eval()
        print("Model loaded.")

        print(f"\nExtracting activations for layers {args.layers}...")
        X_per_layer, y, sample_ids = collect_activations(
            model, tokenizer, episodes, label_map, args.layers)

        del model
        torch.cuda.empty_cache()

        # Save
        save_dict = {"y": y, "sample_ids": np.array(sample_ids)}
        for l, X in X_per_layer.items():
            save_dict[f"layer_{l}"] = X
        np.savez(str(npz_path), **save_dict)
        print(f"\nSaved activations to {npz_path}")

    # Train probe at each layer
    print(f"\n{'Layer':>6} | {'BalAcc':>8} | {'AUROC':>8} | {'Prec':>7} | {'Recall':>7} | {'n_test':>7}")
    print("-" * 60)

    layer_results = {}
    best_auroc, best_layer = 0.0, -1
    probe_directions = {}

    for l in sorted(X_per_layer.keys()):
        X = X_per_layer[l]
        metrics, direction, scaler, clf = train_probe(X, y, seed=args.seed)
        layer_results[l] = metrics
        probe_directions[l] = (direction, scaler, clf)

        m = metrics
        flag = " <<< KILL" if m["balanced_accuracy"] < 0.65 else (
               " <<<" if m["auroc"] > best_auroc else "")
        print(f"  L{l:>2} | {m['balanced_accuracy']:>8.3f} | {m['auroc']:>8.3f} | "
              f"{m['precision']:>7.3f} | {m['recall']:>7.3f} |"
              f" {m['n_test']:>6}{flag}")

        if m["auroc"] > best_auroc:
            best_auroc, best_layer = m["auroc"], l

    print(f"\nBest layer by AUROC: L{best_layer} (AUROC={best_auroc:.3f})")

    # Kill criterion
    l20_metrics = layer_results.get(20, {})
    if l20_metrics.get("balanced_accuracy", 0) < 0.65:
        print("\n[KILL CRITERION] L20 balanced accuracy < 0.65 — core thesis may not hold!")
    else:
        print(f"\n[PASS] L20 balanced accuracy = {l20_metrics['balanced_accuracy']:.3f} >= 0.65")

    # Causal validation (L20)
    causal_result = None
    if 20 in X_per_layer:
        direction_l20, scaler_l20, clf_l20 = probe_directions[20]
        causal_result = causal_validation(
            X_per_layer[20], y, sample_ids,
            A3_RESCUED_VIA_SEARCH, scaler_l20, clf_l20)

    # Direction alignment (L20)
    align_result = None
    if 20 in probe_directions:
        direction_l20 = probe_directions[20][0]
        # Check A3 steering direction first, then fallbacks
        for fname in ["direction_search_v3_layer20.npz", "direction_probe_layer20.npz",
                      "direction_probe_vc.npz"]:
            steer_path = Path(args.steering_dir) / fname
            if steer_path.exists():
                align_result = direction_alignment(direction_l20, steer_path, layer=20)
                break

    # Save results
    results = {
        "model": args.model,
        "layers": sorted(X_per_layer.keys()),
        "n_samples": len(y),
        "n_label0": int((y == 0).sum()),
        "n_label1": int((y == 1).sum()),
        "per_layer": {f"L{l}": m for l, m in layer_results.items()},
        "best_layer_by_auroc": best_layer,
        "causal_validation": causal_result,
        "direction_alignment": align_result,
    }

    out_path = out_dir / "phase1_multilayer_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save L20 probe direction
    if 20 in probe_directions:
        dir_path = out_dir / "probe_direction_l20.npz"
        np.savez(str(dir_path),
                 decision_direction=probe_directions[20][0],
                 layer=20,
                 method="evidence_sufficiency_logreg",
                 n_samples=len(y),
                 n_label0=int((y==0).sum()),
                 balanced_accuracy=float(l20_metrics.get("balanced_accuracy", 0)),
                 auroc=float(l20_metrics.get("auroc", 0)))
        print(f"L20 probe direction saved to {dir_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
