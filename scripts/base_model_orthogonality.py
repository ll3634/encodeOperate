#!/usr/bin/env python3
"""
Base vs Instruct Orthogonality Check
=====================================
Compare cos(evidence_dir, action_dir) between Qwen2.5-7B (base) and
Qwen2.5-7B-Instruct to determine if post-training introduces
evidence-action decoupling.

Pipeline:
  1. Load base model
  2. Extract action_dir via margin contrastive on PopQA (same as Instruct)
  3. Extract evidence_dir via probe on 0-doc vs 1+-doc labels (same as Instruct)
  4. Compute cos(evidence_dir, action_dir)
  5. Compare with Instruct's cos = -0.0135

Usage:
  cd tmc/scripts/e2e_agent
  python scripts/base_model_orthogonality.py
"""

import os, sys, json, argparse, random
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import get_model_layers

LAYER = 20  # Same layer as Instruct experiments


def extract_action_direction(model, tokenizer, popqa_path, layer, n_samples=200):
    """Extract action direction via margin-based contrastive pairs on PopQA."""
    print("\n=== PART 1: Extracting action_dir (PopQA margin contrastive) ===")
    layers = get_model_layers(model)
    device = next(model.parameters()).device

    samples = []
    with open(popqa_path) as f:
        for line in f:
            samples.append(json.loads(line))
    random.seed(42)
    random.shuffle(samples)
    samples = samples[:n_samples]

    pb = PromptBuilder(tools=["search"])
    all_data = []
    for i, s in enumerate(samples):
        question = s["question"]
        messages = pb.build_full_prompt(question, [])
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        captured = {}
        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured["h"] = h[0, -1, :].detach().float().cpu().numpy()
        handle = layers[layer].register_forward_hook(hook_fn)

        with torch.no_grad():
            logits = model(input_ids).logits[0, -1, :]
        handle.remove()

        log_probs = torch.log_softmax(logits, dim=-1)
        tool_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
        fin_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]
        margin = (torch.logsumexp(log_probs[tool_ids], 0) - torch.logsumexp(log_probs[fin_ids], 0)).item()

        all_data.append({"margin": margin, "hidden": captured["h"]})
        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{len(samples)}] margins collected")

    margins = [d["margin"] for d in all_data]
    p20, p80 = np.percentile(margins, 20), np.percentile(margins, 80)
    low = [d for d in all_data if d["margin"] <= p20]
    high = [d for d in all_data if d["margin"] >= p80]
    print(f"  Low-margin: {len(low)}, High-margin: {len(high)}")
    print(f"  Margin range: [{min(margins):.2f}, {max(margins):.2f}], mean={np.mean(margins):.2f}")

    h_low = np.mean(np.stack([d["hidden"] for d in low]), axis=0)
    h_high = np.mean(np.stack([d["hidden"] for d in high]), axis=0)
    direction = h_low - h_high  # toward NON-ADOPT (same convention)
    direction = direction / np.linalg.norm(direction)
    print(f"  action_dir extracted (dim={direction.shape[0]})")
    return direction.astype(np.float32), margins


def extract_evidence_direction(model, tokenizer, labels_path, baseline_path, layer):
    """Train probe on 0-doc vs 1+-doc using base model's hidden states."""
    print("\n=== PART 2: Extracting evidence_dir (probe on 0-doc vs 1+-doc) ===")
    layers = get_model_layers(model)
    device = next(model.parameters()).device

    label_data = []
    with open(labels_path) as f:
        for line in f:
            label_data.append(json.loads(line))
    print(f"  Loaded {len(label_data)} labeled samples")

    bl_map = {}
    with open(baseline_path) as f:
        for line in f:
            ep = json.loads(line)
            bl_map[ep["sample_id"]] = ep

    pb = PromptBuilder(tools=["search", "calculator"])
    hidden_states, labels = [], []
    for i, ld in enumerate(label_data):
        sid = ld["sample_id"]
        ep = bl_map.get(sid)
        if not ep or not ep.get("steps") or len(ep["steps"]) < 1:
            continue
        s0 = ep["steps"][0]
        if s0.get("action") != "search" or not s0.get("observation"):
            continue

        steps = [{"action": "search", "action_input": s0["action_input"],
                  "observation": s0["observation"]}]
        messages = pb.build_full_prompt(ld["question"], steps)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        captured = {}
        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured["h"] = h[0, -1, :].detach().float().cpu().numpy()
        handle = layers[layer].register_forward_hook(hook_fn)
        with torch.no_grad():
            model(input_ids)
        handle.remove()

        hidden_states.append(captured["h"])
        labels.append(ld["label"])
        if (i+1) % 100 == 0:
            print(f"  [{i+1}/{len(label_data)}] hidden states collected")

    X = np.array(hidden_states, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"  Collected {len(X)} states: {y.sum()} label=1 (sufficient), {(1-y).sum()} label=0 (insufficient)")

    # Train probe with CV
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                           solver="lbfgs", random_state=42),
        X_scaled, y, cv=cv, scoring="balanced_accuracy")
    print(f"  5-fold CV balanced accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # AUROC via held-out split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X_scaled, y))
    probe_test = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                                solver="lbfgs", random_state=42)
    probe_test.fit(X_scaled[train_idx], y[train_idx])
    auroc = roc_auc_score(y[test_idx], probe_test.predict_proba(X_scaled[test_idx])[:, 1])
    print(f"  Test AUROC: {auroc:.3f}")

    # Final probe on all data for direction extraction
    probe_full = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                                     solver="lbfgs", random_state=42)
    probe_full.fit(X_scaled, y)
    w_original = probe_full.coef_[0] / scaler.scale_
    direction = w_original / np.linalg.norm(w_original)
    print(f"  evidence_dir extracted (dim={direction.shape[0]})")
    return direction.astype(np.float32), {
        "cv_balanced_accuracy": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "auroc": float(auroc),
        "n_samples": len(X),
        "n_label1": int(y.sum()),
        "n_label0": int((1-y).sum()),
    }


def main():
    ap = argparse.ArgumentParser(description="Base vs Instruct Orthogonality Check")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--popqa-path", default="data/popqa/popqa_test.jsonl")
    ap.add_argument("--labels-path", default="results/phase1_probe/labels.jsonl")
    ap.add_argument("--baseline-trace", default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--output-dir", default="results/base_vs_instruct_orthogonality")
    ap.add_argument("--n-popqa", type=int, default=200)
    ap.add_argument("--layer", type=int, default=20)
    args = ap.parse_args()

    global LAYER
    LAYER = args.layer

    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load base model
    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    n_layers = len(get_model_layers(model))
    D = model.config.hidden_size
    print(f"  Layers: {n_layers}, Hidden: {D}")

    # Step 1: Action direction
    action_dir, margins = extract_action_direction(
        model, tokenizer, args.popqa_path, LAYER, n_samples=args.n_popqa)

    # Step 2: Evidence direction
    evidence_dir, probe_metrics = extract_evidence_direction(
        model, tokenizer, args.labels_path, args.baseline_trace, LAYER)

    # Step 3: Compute cosine
    cos_base = float(np.dot(action_dir, evidence_dir))
    cos_instruct = -0.0135  # Known from previous experiments

    print(f"\n{'='*60}")
    print(f"  ★ KEY RESULT: Base vs Instruct Orthogonality")
    print(f"{'='*60}")
    print(f"  Base model cos(action, evidence)    = {cos_base:.4f}")
    print(f"  Instruct model cos(action, evidence) = {cos_instruct:.4f}")
    print(f"  Δcos = {cos_base - cos_instruct:.4f}")
    print(f"")
    if abs(cos_base) > 0.15:
        print(f"  → Base model has SIGNIFICANT coupling (|cos|>{abs(cos_base):.3f})")
        print(f"  → Post-training INTRODUCED evidence-action decoupling")
        print(f"  → Geometric explanation for why RLHF degrades abstention")
    elif abs(cos_base) < 0.05:
        print(f"  → Base model also nearly orthogonal (|cos|={abs(cos_base):.3f})")
        print(f"  → Orthogonality is ARCHITECTURAL, not post-training artifact")
    else:
        print(f"  → Moderate coupling (|cos|={abs(cos_base):.3f})")
        print(f"  → Partial decoupling by post-training")
    print(f"{'='*60}")

    # Save results
    out = {
        "timestamp": datetime.now().isoformat(),
        "base_model": args.base_model,
        "instruct_model": "Qwen/Qwen2.5-7B-Instruct",
        "layer": LAYER,
        "cos_base": cos_base,
        "cos_instruct": cos_instruct,
        "delta_cos": cos_base - cos_instruct,
        "base_probe_metrics": probe_metrics,
        "base_margin_stats": {
            "mean": float(np.mean(margins)),
            "std": float(np.std(margins)),
            "min": float(np.min(margins)),
            "max": float(np.max(margins)),
        },
    }
    out_path = os.path.join(args.output_dir, "base_vs_instruct_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Save directions for potential further analysis
    np.savez(
        os.path.join(args.output_dir, "base_directions.npz"),
        action_dir=action_dir,
        evidence_dir=evidence_dir,
        layer=LAYER,
    )
    print(f"Saved: {args.output_dir}/base_directions.npz")


if __name__ == "__main__":
    main()
