#!/usr/bin/env python3
"""
Cross-Model Validation: PCA Subspace Alignment Ratio
=====================================================
Validates on Mistral-7B that the action/evidence alignment ratio in the
circuit's PCA subspace is >10x (matching Qwen's 16-18x).

Pipeline:
  1. Extract action_dir via margin contrastive on PopQA
  2. Extract evidence_dir via linear probe on 0-doc vs 1+-doc HotpotQA labels
  3. Compute PCA of clean-corrupt MLP output differences
  4. Report ||P_10(action)||^2 / ||P_10(evidence)||^2

Usage:
  cd tmc/scripts/e2e_agent
  python scripts/cross_model_validation.py
"""

import os, sys, json, re, argparse, random
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import get_model_layers


# ── Layer mapping ────────────────────────────────────────────────────────────
# Qwen: 28 layers. Circuit: attn_L18 (64.3%) → mlp_L20 (71.4%)
# Mistral: 32 layers. Equivalent: attn_L21 → mlp_L23
ATTN_LAYER = 21
MLP_LAYER = 23
PROBE_LAYER = 23  # same as MLP layer (peak evidence layer)


def extract_at_layers(model, tokenizer, prompt, attn_l, mlp_l):
    """Extract attn and mlp outputs at specific layers, last token only."""
    layers = get_model_layers(model)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    captured = {}
    handles = []

    def make_hook(key):
        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[key] = h[0, -1, :].detach().float().cpu().numpy()
        return hook_fn

    handles.append(layers[attn_l].self_attn.register_forward_hook(make_hook(('attn', attn_l))))
    handles.append(layers[mlp_l].mlp.register_forward_hook(make_hook(('mlp', mlp_l))))
    # Also capture full layer output at PROBE_LAYER for probe
    handles.append(layers[PROBE_LAYER].register_forward_hook(make_hook('probe_h')))

    with torch.no_grad():
        model(input_ids)
    for h in handles:
        h.remove()
    return captured


# ── Part 1: Action Direction (margin contrastive on PopQA) ────────────────
def extract_action_direction(model, tokenizer, popqa_path, layer, n_samples=200):
    """Extract action direction via margin-based contrastive pairs."""
    print("\n=== PART 1: Extracting action_dir ===")
    layers = get_model_layers(model)
    device = next(model.parameters()).device

    # Load PopQA
    samples = []
    with open(popqa_path) as f:
        for line in f:
            s = json.loads(line)
            samples.append(s)
    random.seed(42)
    random.shuffle(samples)
    samples = samples[:n_samples]

    # Build prompts and compute margins + hidden states
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
    print(f"  Margin range: [{min(margins):.2f}, {max(margins):.2f}]")

    h_low = np.mean(np.stack([d["hidden"] for d in low]), axis=0)
    h_high = np.mean(np.stack([d["hidden"] for d in high]), axis=0)
    direction = h_low - h_high  # toward NON-ADOPT
    direction /= np.linalg.norm(direction)
    print(f"  action_dir extracted (dim={direction.shape[0]})")
    return direction


# ── Part 2: Evidence Direction (probe on 0-doc vs 1+-doc) ─────────────────
def extract_evidence_direction(model, tokenizer, labels_path, baseline_path,
                                hotpotqa_path, layer):
    """Train probe on 0-doc vs 1+-doc using model's hidden states."""
    print("\n=== PART 2: Extracting evidence_dir ===")
    layers = get_model_layers(model)
    device = next(model.parameters()).device

    # Load labels
    label_data = []
    with open(labels_path) as f:
        for line in f:
            label_data.append(json.loads(line))
    print(f"  Loaded {len(label_data)} labeled samples")

    # Load baseline trace for prompt reconstruction
    bl_map = {}
    with open(baseline_path) as f:
        for line in f:
            ep = json.loads(line)
            bl_map[ep["sample_id"]] = ep

    # Collect hidden states
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
    print(f"  Collected {len(X)} states: {y.sum()} label=1, {(1-y).sum()} label=0")

    # Train probe
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    probe = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                                solver="lbfgs", random_state=42)
    probe.fit(X_scaled, y)
    w_original = probe.coef_[0] / scaler.scale_
    direction = w_original / np.linalg.norm(w_original)
    acc = probe.score(X_scaled, y)
    print(f"  Probe train accuracy: {acc:.3f}")
    print(f"  evidence_dir extracted (dim={direction.shape[0]})")
    return direction.astype(np.float32)


# ── Part 3: PCA Bridge Analysis ──────────────────────────────────────────
OBS_ENTRY_RE = re.compile(r'\[(\d+)\]\s*([^:]+):\s*(.*?)(?=\n\n\[\d+\]|\Z)', re.DOTALL)


def run_pca_bridge(model, tokenizer, action_dir, evidence_dir,
                   baseline_path, hotpotqa_path, n_samples=100, seed=42):
    """Compute PCA of clean-corrupt diffs and report alignment ratios."""
    print("\n=== PART 3: PCA Bridge Analysis ===")
    # Import sample selection from paired_corruption_analysis
    from scripts.paired_corruption_analysis import (
        select_samples, make_corrupted_obs, build_prompt,
    )

    samples = select_samples(baseline_path, hotpotqa_path, n=n_samples, seed=seed)
    print(f"  Selected {len(samples)} samples")

    D = action_dir.shape[0]
    diffs_mlp = []
    for i, sample in enumerate(samples):
        rng_copy = random.Random(seed)
        for j in range(i):
            make_corrupted_obs(samples[j], "A", rng_copy)
        clean_obs, corrupted_obs = make_corrupted_obs(sample, "A", rng_copy)
        prompt_clean = build_prompt(tokenizer, sample["question"],
                                     sample["step0_query"], clean_obs)
        prompt_corrupt = build_prompt(tokenizer, sample["question"],
                                       sample["step0_query"], corrupted_obs)

        cache_clean = extract_at_layers(model, tokenizer, prompt_clean, ATTN_LAYER, MLP_LAYER)
        cache_corrupt = extract_at_layers(model, tokenizer, prompt_corrupt, ATTN_LAYER, MLP_LAYER)
        diff_mlp = (cache_clean[('mlp', MLP_LAYER)] - cache_corrupt[('mlp', MLP_LAYER)]).astype(np.float32)
        diffs_mlp.append(diff_mlp)
        if (i+1) % 20 == 0:
            print(f"  [{i+1}/{len(samples)}] diffs collected")

    mat = np.stack(diffs_mlp, axis=0)
    _, S, Vt = np.linalg.svd(mat - mat.mean(0), full_matrices=False)

    # Projection analysis
    print(f"\n{'='*60}")
    print(f"  mlp_L{MLP_LAYER} PCA Subspace Alignment")
    print(f"{'='*60}")
    print(f"{'k':>5} | {'||P_k(act)||²':>14} {'||P_k(evi)||²':>14} "
          f"{'k/D':>8} | {'act/rand':>9} {'evi/rand':>9} | {'ratio':>8}")

    results = {}
    for k in [1, 2, 3, 5, 10, 20, 50, 100]:
        p_act = sum(float(np.dot(action_dir, Vt[j]))**2 for j in range(min(k, len(Vt))))
        p_evi = sum(float(np.dot(evidence_dir, Vt[j]))**2 for j in range(min(k, len(Vt))))
        baseline = k / D
        r_act = p_act / baseline
        r_evi = p_evi / baseline
        ratio = p_act / p_evi if p_evi > 1e-10 else float('inf')
        results[k] = {"p_act": p_act, "p_evi": p_evi, "r_act": r_act, "r_evi": r_evi, "ratio": ratio}
        print(f"{k:>5} | {p_act:>14.4f} {p_evi:>14.4f} "
              f"{baseline:>8.4f} | {r_act:>9.2f}x {r_evi:>9.2f}x | {ratio:>8.1f}x")

    r10 = results[10]
    print(f"\n{'='*60}")
    print(f"  ★ KEY RESULT: top-10 action/evidence ratio = {r10['ratio']:.1f}x")
    print(f"    (Qwen reference: 16.4x)")
    print(f"    action_dir alignment: {r10['r_act']:.1f}x random")
    print(f"    evidence_dir alignment: {r10['r_evi']:.1f}x random")
    if r10['ratio'] > 10:
        print(f"    ✅ GENERALIZATION CONFIRMED (ratio > 10x)")
    else:
        print(f"    ⚠️  ratio < 10x, may not generalize")
    print(f"{'='*60}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--popqa-path", default="data/popqa/popqa_test.jsonl")
    ap.add_argument("--labels-path", default="results/phase1_probe/labels.jsonl")
    ap.add_argument("--baseline-trace", default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--hotpotqa-data", default="data/hotpotqa/hotpot_dev_distractor_v1.json")
    ap.add_argument("--output-dir", default="results/cross_model_mistral")
    ap.add_argument("--n-popqa", type=int, default=200)
    ap.add_argument("--n-pca-samples", type=int, default=100)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    n_layers = len(get_model_layers(model))
    D = model.config.hidden_size
    print(f"  Layers: {n_layers}, Hidden: {D}")
    print(f"  Circuit layers: attn_L{ATTN_LAYER}, mlp_L{MLP_LAYER}")

    # Step 1: Action direction
    action_dir = extract_action_direction(model, tokenizer, args.popqa_path,
                                           PROBE_LAYER, n_samples=args.n_popqa)

    # Step 2: Evidence direction
    evidence_dir = extract_evidence_direction(model, tokenizer, args.labels_path,
                                               args.baseline_trace,
                                               args.hotpotqa_data, PROBE_LAYER)

    cos = float(np.dot(action_dir, evidence_dir))
    print(f"\n  cos(action_dir, evidence_dir) = {cos:.4f}")

    # Step 3: PCA bridge analysis
    results = run_pca_bridge(model, tokenizer, action_dir, evidence_dir,
                              args.baseline_trace, args.hotpotqa_data,
                              n_samples=args.n_pca_samples)

    # Save results
    out = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "attn_layer": ATTN_LAYER,
        "mlp_layer": MLP_LAYER,
        "cos_action_evidence": cos,
        "pca_alignment": {str(k): v for k, v in results.items()},
    }
    out_path = os.path.join(args.output_dir, "cross_model_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

