#!/usr/bin/env python3
"""
Linear Probe Steering Direction (Representation Engineering).

Instead of selecting SAE features based on the model's own margins (circular),
this script trains a logistic regression probe on Layer 11 hidden states using
*oracle* labels (VC = model should have searched but didn't) to find the causal
direction that separates "needs search" from "doesn't need search".

Steps:
  1. Load baseline + oracle traces → derive VC/non-VC labels
  2. Load model, reconstruct step-1 decision prompts
  3. Capture Layer 11 hidden states at the decision token
  4. Train logistic regression probe (with class balancing)
  5. Save probe weight vector as steering direction (.npz)

Usage:
    cd tmc/scripts/e2e_agent
    python scripts/build_probe_direction.py \
        --baseline-trace results/direction_comparison_n200/baseline_results.jsonl \
        --oracle-trace results/direction_comparison_n200/oracle_results.jsonl \
        --output steering/directions/direction_probe_vc.npz
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

from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import get_model_layers


def load_vc_labels(baseline_path, oracle_path, n_samples=None):
    """Load baseline + oracle traces and derive VC labels."""
    bl_eps = []
    with open(baseline_path) as f:
        for line in f:
            bl_eps.append(json.loads(line))
    orc_eps = []
    with open(oracle_path) as f:
        for line in f:
            orc_eps.append(json.loads(line))
    if n_samples:
        bl_eps = bl_eps[:n_samples]
        orc_eps = orc_eps[:n_samples]

    orc_map = {ep["sample_id"]: ep["is_correct"] for ep in orc_eps}

    episodes = []
    for ep in bl_eps:
        sid = ep["sample_id"]
        bl_correct = ep["is_correct"]
        orc_correct = orc_map.get(sid, False)
        # VC: baseline wrong, oracle correct → should have searched
        is_vc = (not bl_correct) and orc_correct
        # Need valid step-0 search for prompt reconstruction
        if not ep.get("steps") or len(ep["steps"]) < 1:
            continue
        s0 = ep["steps"][0]
        if s0.get("action") != "search" or not s0.get("observation"):
            continue
        episodes.append({
            "sample_id": sid,
            "question": ep["question"],
            "step0_query": s0["action_input"],
            "step0_obs": s0["observation"],
            "is_vc": is_vc,
            "bl_correct": bl_correct,
            "orc_correct": orc_correct,
        })

    n_vc = sum(1 for e in episodes if e["is_vc"])
    print(f"Loaded {len(episodes)} valid episodes: {n_vc} VC, {len(episodes)-n_vc} non-VC")
    return episodes


def collect_hidden_states(model, tokenizer, episodes, layer=11):
    """Capture Layer 11 hidden states at step-1 decision point."""
    pb = PromptBuilder(tools=["search", "calculator"])
    layers = get_model_layers(model)
    device = next(model.parameters()).device

    hidden_states, labels, sample_ids = [], [], []

    for i, ep in enumerate(episodes):
        steps = [{"action": "search", "action_input": ep["step0_query"],
                  "observation": ep["step0_obs"]}]
        messages = pb.build_full_prompt(ep["question"], steps)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        captured = {}
        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured["hidden"] = h.detach()

        handle = layers[layer].register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                model(input_ids)
        except Exception as e:
            handle.remove()
            print(f"  [{i+1}] ERROR: {e}")
            continue
        handle.remove()

        if "hidden" not in captured:
            continue

        h_last = captured["hidden"][0, -1, :].float().cpu().numpy()
        hidden_states.append(h_last)
        labels.append(1 if ep["is_vc"] else 0)
        sample_ids.append(ep["sample_id"])

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(episodes)}] collected")

    X = np.array(hidden_states, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"\nCollected {len(X)} hidden states: {y.sum()} VC, {(1-y).sum()} non-VC")
    return X, y, sample_ids


def train_probe(X, y):
    """Train logistic regression probe with cross-validation."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validated accuracy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                           solver="lbfgs", random_state=42),
        X_scaled, y, cv=cv, scoring="balanced_accuracy")
    print(f"\n5-fold CV balanced accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"  Per-fold: {[f'{s:.3f}' for s in scores]}")

    # Train final probe on all data
    probe = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                               solver="lbfgs", random_state=42)
    probe.fit(X_scaled, y)

    # The probe weight in ORIGINAL (unscaled) space
    # w_original = w_scaled / std
    w_scaled = probe.coef_[0]  # [3584]
    w_original = w_scaled / scaler.scale_
    direction = w_original.astype(np.float32)

    # Normalize to unit norm
    norm = np.linalg.norm(direction)
    direction_unit = direction / norm

    print(f"  Probe coef norm (scaled): {np.linalg.norm(w_scaled):.4f}")
    print(f"  Probe coef norm (original): {norm:.4f}")
    print(f"  Direction RMS: {np.sqrt(np.mean(direction_unit**2)):.6f}")
    print(f"  Train accuracy: {probe.score(X_scaled, y):.3f}")

    return direction_unit, probe, scaler, scores


def main():
    parser = argparse.ArgumentParser(description="Build probe-based steering direction")
    parser.add_argument("--baseline-trace", required=True)
    parser.add_argument("--oracle-trace", required=True)
    parser.add_argument("--output", default="steering/directions/direction_probe_vc.npz")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=int, default=12,
                        help="Layer index for hidden-state capture. Must match "
                             "the runtime --layer in run_direction_comparison.py "
                             "(default: 12 = layers[12] resid_post).")
    parser.add_argument("--n-samples", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  LINEAR PROBE STEERING DIRECTION")
    print("=" * 60)
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"  Layer: {args.layer}")
    print()

    # 1. Load VC labels
    episodes = load_vc_labels(args.baseline_trace, args.oracle_trace, args.n_samples)

    # 2. Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")
    model.eval()

    # 3. Collect hidden states
    X, y, sample_ids = collect_hidden_states(model, tokenizer, episodes, args.layer)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # 4. Train probe
    direction, probe, scaler, cv_scores = train_probe(X, y)

    # 5. Save direction
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    rms = float(np.sqrt(np.mean(direction ** 2)))
    norm = float(np.linalg.norm(direction))

    np.savez(
        str(out),
        decision_direction=direction,
        layer=args.layer,
        method="linear_probe_vc",
        n_samples=len(X),
        n_vc=int(y.sum()),
        n_non_vc=int((1 - y).sum()),
        cv_balanced_accuracy=float(cv_scores.mean()),
        cv_std=float(cv_scores.std()),
    )

    print(f"\nSaved probe direction to {out}")
    print(f"  Norm: {norm:.4f}, RMS: {rms:.6f}")
    print(f"  CV balanced acc: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Also save analysis JSON
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "method": "linear_probe_vc",
        "layer": args.layer,
        "n_samples": len(X),
        "n_vc": int(y.sum()),
        "n_non_vc": int((1 - y).sum()),
        "cv_balanced_accuracy": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "cv_per_fold": [float(s) for s in cv_scores],
        "direction_norm": norm,
        "direction_rms": rms,
    }
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved analysis to {out.with_suffix('.json')}")
    print("\nDone!")


if __name__ == "__main__":
    main()
