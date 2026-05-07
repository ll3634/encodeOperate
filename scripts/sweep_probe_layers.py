#!/usr/bin/env python3
"""
All-layer probe sweep: collect hidden states from ALL 28 layers in a single
forward pass, then train a logistic regression probe at each layer.

Outputs:
  - Per-layer CV balanced accuracy table
  - Best direction .npz at the optimal layer
  - Mean-difference direction at the optimal layer (as a more robust alternative)

Usage:
    cd tmc/scripts/e2e_agent
    python scripts/sweep_probe_layers.py \
        --baseline-trace results/probe_comparison_n200/baseline_results.jsonl \
        --oracle-trace results/probe_comparison_n200/oracle_results.jsonl \
        --output-dir steering/directions/layer_sweep
"""

import os, sys, json, argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.prompts import PromptBuilder
from steering.hook_utils import get_model_layers


def load_vc_labels(baseline_path, oracle_path, n_samples=None):
    """Load baseline + oracle traces and derive VC labels."""
    bl_eps = [json.loads(l) for l in open(baseline_path)]
    orc_eps = [json.loads(l) for l in open(oracle_path)]
    if n_samples:
        bl_eps, orc_eps = bl_eps[:n_samples], orc_eps[:n_samples]

    orc_map = {ep["sample_id"]: ep["is_correct"] for ep in orc_eps}
    episodes = []
    for ep in bl_eps:
        sid = ep["sample_id"]
        is_vc = (not ep["is_correct"]) and orc_map.get(sid, False)
        if not ep.get("steps") or len(ep["steps"]) < 1:
            continue
        s0 = ep["steps"][0]
        if s0.get("action") != "search" or not s0.get("observation"):
            continue
        episodes.append({
            "sample_id": sid, "question": ep["question"],
            "step0_query": s0["action_input"], "step0_obs": s0["observation"],
            "is_vc": is_vc,
        })
    n_vc = sum(1 for e in episodes if e["is_vc"])
    print(f"Loaded {len(episodes)} episodes: {n_vc} VC, {len(episodes)-n_vc} non-VC")
    return episodes


def collect_all_layers(model, tokenizer, episodes):
    """Single forward pass per episode, capture ALL layers' last-token hidden state."""
    pb = PromptBuilder(tools=["search", "calculator"])
    layers = get_model_layers(model)
    n_layers = len(layers)
    device = next(model.parameters()).device

    # Output: {layer_idx: list of hidden vectors}
    all_hidden = {l: [] for l in range(n_layers)}
    labels, sample_ids = [], []

    hooks = []

    for i, ep in enumerate(episodes):
        steps = [{"action": "search", "action_input": ep["step0_query"],
                  "observation": ep["step0_obs"]}]
        messages = pb.build_full_prompt(ep["question"], steps)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        captured = {}

        def make_hook(layer_idx):
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[layer_idx] = h[0, -1, :].detach().float().cpu().numpy()
            return hook_fn

        handles = []
        for l_idx in range(n_layers):
            handles.append(layers[l_idx].register_forward_hook(make_hook(l_idx)))

        try:
            with torch.no_grad():
                model(input_ids)
        except Exception as e:
            for h in handles:
                h.remove()
            print(f"  [{i+1}] ERROR: {e}")
            continue
        for h in handles:
            h.remove()

        if len(captured) != n_layers:
            continue

        for l_idx in range(n_layers):
            all_hidden[l_idx].append(captured[l_idx])
        labels.append(1 if ep["is_vc"] else 0)
        sample_ids.append(ep["sample_id"])

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(episodes)}] collected")

    y = np.array(labels, dtype=np.int32)
    X_per_layer = {l: np.array(vecs, dtype=np.float32) for l, vecs in all_hidden.items()}
    print(f"\nCollected {len(y)} samples × {n_layers} layers: {y.sum()} VC, {(1-y).sum()} non-VC")
    return X_per_layer, y, sample_ids


def train_probe_for_layer(X, y, C=1.0):
    """Train probe and return CV accuracy + direction."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(class_weight="balanced", C=C, max_iter=2000,
                             solver="lbfgs", random_state=42)
    scores = cross_val_score(clf, X_s, y, cv=cv, scoring="balanced_accuracy")

    # Train final
    clf.fit(X_s, y)
    w_orig = clf.coef_[0] / scaler.scale_
    direction = (w_orig / np.linalg.norm(w_orig)).astype(np.float32)

    # Mean difference direction (simpler, more robust)
    mean_vc = X[y == 1].mean(axis=0)
    mean_non = X[y == 0].mean(axis=0)
    mean_diff = (mean_vc - mean_non).astype(np.float32)
    md_norm = np.linalg.norm(mean_diff)
    if md_norm > 1e-10:
        mean_diff_unit = mean_diff / md_norm
    else:
        mean_diff_unit = mean_diff

    return scores, direction, mean_diff_unit


def main():
    parser = argparse.ArgumentParser(description="Sweep probe across all layers")
    parser.add_argument("--baseline-trace", required=True)
    parser.add_argument("--oracle-trace", required=True)
    parser.add_argument("--output-dir", default="steering/directions/layer_sweep")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--C", type=float, default=1.0, help="LogReg regularization")
    args = parser.parse_args()

    print("=" * 60)
    print("  ALL-LAYER PROBE SWEEP")
    print("=" * 60)

    episodes = load_vc_labels(args.baseline_trace, args.oracle_trace, args.n_samples)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")
    model.eval()

    X_per_layer, y, sample_ids = collect_all_layers(model, tokenizer, episodes)
    del model
    torch.cuda.empty_cache()

    n_layers = len(X_per_layer)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Layer':>5} | {'CV Acc':>8} | {'±Std':>6} | {'MeanDiff cos':>12} | {'Best fold':>9} | {'Worst fold':>10}")
    print("-" * 70)

    results = []
    best_layer, best_acc = -1, 0.0

    for l in range(n_layers):
        X = X_per_layer[l]
        scores, probe_dir, md_dir = train_probe_for_layer(X, y, C=args.C)
        acc = scores.mean()
        cos = float(np.dot(probe_dir, md_dir))

        results.append({
            "layer": l, "cv_acc": float(acc), "cv_std": float(scores.std()),
            "cv_folds": [float(s) for s in scores],
            "best_fold": float(scores.max()), "worst_fold": float(scores.min()),
            "probe_md_cosine": cos,
        })

        marker = " <<<" if acc > best_acc else ""
        print(f"{l:>5} | {acc:>8.3f} | {scores.std():>6.3f} | {cos:>12.3f} | "
              f"{scores.max():>9.3f} | {scores.min():>10.3f}{marker}")

        if acc > best_acc:
            best_acc, best_layer = acc, l

    # Save best directions
    print(f"\n{'='*60}")
    print(f"  BEST LAYER: {best_layer}  (CV acc = {best_acc:.3f})")
    print(f"{'='*60}")

    X_best = X_per_layer[best_layer]
    _, best_probe_dir, best_md_dir = train_probe_for_layer(X_best, y, C=args.C)

    for name, direction in [("probe", best_probe_dir), ("mean_diff", best_md_dir)]:
        rms = float(np.sqrt(np.mean(direction ** 2)))
        out_path = out_dir / f"direction_{name}_layer{best_layer}.npz"
        np.savez(str(out_path),
                 decision_direction=direction,
                 layer=best_layer,
                 method=f"layer_sweep_{name}",
                 n_samples=len(y), n_vc=int(y.sum()),
                 cv_balanced_accuracy=best_acc)
        print(f"  Saved: {out_path}  (RMS={rms:.6f})")

    # Save full sweep results
    sweep_report = {
        "timestamp": datetime.now().isoformat(),
        "n_layers": n_layers, "n_samples": len(y),
        "n_vc": int(y.sum()), "n_non_vc": int((1-y).sum()),
        "best_layer": best_layer, "best_cv_acc": best_acc,
        "per_layer": results,
    }
    with open(out_dir / "layer_sweep_report.json", "w") as f:
        json.dump(sweep_report, f, indent=2)
    print(f"  Report: {out_dir / 'layer_sweep_report.json'}")


if __name__ == "__main__":
    main()

