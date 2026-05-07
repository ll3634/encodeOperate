#!/usr/bin/env python3
"""Quick layer sweep to find the peak MLP layer for Mistral-7B."""

import os, sys, json, re, random
import numpy as np
from pathlib import Path
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import get_model_layers
from scripts.paired_corruption_analysis import (
    select_samples, make_corrupted_obs, build_prompt,
)


def extract_mlp_diff(model, tokenizer, prompt_clean, prompt_corrupt, mlp_layer):
    """Extract MLP output diff at a specific layer."""
    layers = get_model_layers(model)
    device = next(model.parameters()).device
    
    results = {}
    for label, prompt in [("clean", prompt_clean), ("corrupt", prompt_corrupt)]:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        captured = {}
        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured["h"] = h[0, -1, :].detach().float().cpu().numpy()
        handle = layers[mlp_layer].mlp.register_forward_hook(hook_fn)
        with torch.no_grad():
            model(input_ids)
        handle.remove()
        results[label] = captured["h"]
    return (results["clean"] - results["corrupt"]).astype(np.float32)


def extract_direction_at_layer(model, tokenizer, popqa_path, layer, n=200):
    """Quick action direction extraction at a given layer."""
    layers = get_model_layers(model)
    device = next(model.parameters()).device
    samples = []
    with open(popqa_path) as f:
        for line in f:
            samples.append(json.loads(line))
    random.seed(42)
    random.shuffle(samples)
    samples = samples[:n]

    pb = PromptBuilder(tools=["search"])
    all_data = []
    for s in samples:
        messages = pb.build_full_prompt(s["question"], [])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

    margins = [d["margin"] for d in all_data]
    p20, p80 = np.percentile(margins, 20), np.percentile(margins, 80)
    low = [d for d in all_data if d["margin"] <= p20]
    high = [d for d in all_data if d["margin"] >= p80]
    h_low = np.mean(np.stack([d["hidden"] for d in low]), axis=0)
    h_high = np.mean(np.stack([d["hidden"] for d in high]), axis=0)
    d = h_low - h_high
    return (d / np.linalg.norm(d)).astype(np.float32)


def extract_evidence_at_layer(model, tokenizer, labels_path, baseline_path, layer):
    """Quick evidence probe at a given layer."""
    layers = get_model_layers(model)
    device = next(model.parameters()).device
    label_data = []
    with open(labels_path) as f:
        for line in f:
            label_data.append(json.loads(line))
    bl_map = {}
    with open(baseline_path) as f:
        for line in f:
            ep = json.loads(line)
            bl_map[ep["sample_id"]] = ep

    pb = PromptBuilder(tools=["search", "calculator"])
    hs, ys = [], []
    for ld in label_data:
        ep = bl_map.get(ld["sample_id"])
        if not ep or not ep.get("steps") or len(ep["steps"]) < 1:
            continue
        s0 = ep["steps"][0]
        if s0.get("action") != "search" or not s0.get("observation"):
            continue
        steps = [{"action": "search", "action_input": s0["action_input"],
                  "observation": s0["observation"]}]
        messages = pb.build_full_prompt(ld["question"], steps)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        captured = {}
        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured["h"] = h[0, -1, :].detach().float().cpu().numpy()
        handle = layers[layer].register_forward_hook(hook_fn)
        with torch.no_grad():
            model(input_ids)
        handle.remove()
        hs.append(captured["h"])
        ys.append(ld["label"])

    X, y = np.array(hs, dtype=np.float32), np.array(ys, dtype=np.int32)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    probe = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, solver="lbfgs", random_state=42)
    probe.fit(X_s, y)
    w = probe.coef_[0] / scaler.scale_
    return (w / np.linalg.norm(w)).astype(np.float32)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--layers", default="18,20,21,22,23,24,25,26")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = args.model
    print(f"Loading {model_name}...")
    hf_token = os.environ.get("HF_TOKEN", None)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, token=hf_token)
    model.eval()
    D = model.config.hidden_size
    n_layers = len(get_model_layers(model))
    print(f"  {n_layers} layers, D={D}")

    baseline_path = "results/l20_rho020_n500/baseline_results.jsonl"
    hotpotqa_path = "data/hotpotqa/hotpot_dev_distractor_v1.json"
    popqa_path = "data/popqa/popqa_test.jsonl"
    labels_path = "results/phase1_probe/labels.jsonl"

    samples = select_samples(baseline_path, hotpotqa_path, n=60, seed=42)
    print(f"  {len(samples)} PCA samples")

    # Sweep layers
    layer_list = [int(x) for x in args.layers.split(",")]
    for mlp_l in layer_list:
        print(f"\n--- Layer {mlp_l} ---")
        act_dir = extract_direction_at_layer(model, tokenizer, popqa_path, mlp_l, n=200)
        evi_dir = extract_evidence_at_layer(model, tokenizer, labels_path, baseline_path, mlp_l)
        cos_ae = float(np.dot(act_dir, evi_dir))

        diffs = []
        for i, sample in enumerate(samples):
            rng_copy = random.Random(42)
            for j in range(i):
                make_corrupted_obs(samples[j], "A", rng_copy)
            clean_obs, corrupted_obs = make_corrupted_obs(sample, "A", rng_copy)
            prompt_c = build_prompt(tokenizer, sample["question"], sample["step0_query"], clean_obs)
            prompt_x = build_prompt(tokenizer, sample["question"], sample["step0_query"], corrupted_obs)
            diffs.append(extract_mlp_diff(model, tokenizer, prompt_c, prompt_x, mlp_l))

        mat = np.stack(diffs)
        _, S, Vt = np.linalg.svd(mat - mat.mean(0), full_matrices=False)
        k = 10
        p_act = sum(float(np.dot(act_dir, Vt[j]))**2 for j in range(min(k, len(Vt))))
        p_evi = sum(float(np.dot(evi_dir, Vt[j]))**2 for j in range(min(k, len(Vt))))
        ratio = p_act / p_evi if p_evi > 1e-10 else float('inf')
        print(f"  cos(act,evi)={cos_ae:.4f}  P10_act={p_act:.4f}  P10_evi={p_evi:.4f}  ratio={ratio:.1f}x")


if __name__ == "__main__":
    main()

