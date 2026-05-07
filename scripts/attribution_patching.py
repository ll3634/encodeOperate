#!/usr/bin/env python3
"""Step 2: Attribution patching — coarse scan across all layers/components.

For each (clean, corrupted) pair from minimal corruption dataset:
  1. Run clean forward: cache attn & mlp outputs at last token for all layers
  2. Run corrupt forward: get baseline margin
  3. For each component: run corrupt forward but patch that component with clean value
     → measure how much margin recovers

This identifies which components mediate the evidence→action effect.
"""

import json, sys, argparse, time
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import get_model_layers
from scripts.minimal_corruption import build_prompt


def get_margin(model, tokenizer, input_ids):
    """Compute search-stop logit margin."""
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    tool_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]
    tool_lp = torch.logsumexp(log_probs[tool_ids], 0).item()
    fin_lp = torch.logsumexp(log_probs[fin_ids], 0).item()
    return tool_lp - fin_lp


def cache_component_outputs(model, tokenizer, prompt, n_layers):
    """Run forward pass, cache attn and mlp outputs at last token for all layers."""
    layers = get_model_layers(model)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    cache = {}
    handles = []

    for l in range(n_layers):
        def make_hook(comp, layer_idx):
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                cache[(comp, layer_idx)] = h[0, -1, :].detach().clone()
            return hook_fn
        handles.append(layers[l].self_attn.register_forward_hook(make_hook('attn', l)))
        handles.append(layers[l].mlp.register_forward_hook(make_hook('mlp', l)))

    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :]
    for h in handles:
        h.remove()

    # Also compute margin
    log_probs = torch.log_softmax(logits, dim=-1)
    tool_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]
    margin = (torch.logsumexp(log_probs[tool_ids], 0) - torch.logsumexp(log_probs[fin_ids], 0)).item()

    return cache, margin, input_ids


def patch_and_get_margin(model, tokenizer, input_ids_corrupt, clean_cache,
                          comp, layer_idx):
    """Run corrupt forward but patch one component from clean cache. Return margin."""
    layers = get_model_layers(model)
    device = next(model.parameters()).device
    clean_vec = clean_cache[(comp, layer_idx)]

    target = layers[layer_idx].self_attn if comp == 'attn' else layers[layer_idx].mlp

    def patch_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h[0, -1, :] = clean_vec.to(h.dtype)
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h

    handle = target.register_forward_hook(patch_hook)
    with torch.no_grad():
        logits = model(input_ids_corrupt).logits[0, -1, :]
    handle.remove()

    log_probs = torch.log_softmax(logits, dim=-1)
    tool_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]
    return (torch.logsumexp(log_probs[tool_ids], 0) - torch.logsumexp(log_probs[fin_ids], 0)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/minimal_corruption/full_corruption_data.jsonl")
    parser.add_argument("--n", type=int, default=50, help="Number of samples to use")
    parser.add_argument("--output-dir", default="results/attribution_patching")
    parser.add_argument("--layers", default="0-27", help="Layer range to scan")
    args = parser.parse_args()

    # Parse layer range
    parts = args.layers.split("-")
    layer_start, layer_end = int(parts[0]), int(parts[1])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    samples = []
    with open(args.data) as f:
        for line in f:
            samples.append(json.loads(line))
    samples = samples[:args.n]
    print(f"Loaded {len(samples)} samples")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    n_layers = len(get_model_layers(model))
    print(f"  {n_layers} layers")

    # Run attribution patching
    results = defaultdict(list)  # (comp, layer) -> [recovery_fractions]
    t0 = time.time()

    for si, s in enumerate(samples):
        prompt_clean = build_prompt(tokenizer, s["question"], s["step0_query"], s["obs_clean"])
        prompt_corrupt_A = build_prompt(tokenizer, s["question"], s["step0_query"], s["obs_corrupt_A"])

        # Cache clean outputs and get margins
        clean_cache, margin_clean, _ = cache_component_outputs(
            model, tokenizer, prompt_clean, n_layers)
        _, margin_corrupt, input_ids_corrupt = cache_component_outputs(
            model, tokenizer, prompt_corrupt_A, n_layers)

        total_effect = margin_clean - margin_corrupt
        if abs(total_effect) < 0.01:
            continue

        # Patch each component and measure recovery
        for l in range(layer_start, layer_end + 1):
            for comp in ['attn', 'mlp']:
                margin_patched = patch_and_get_margin(
                    model, tokenizer, input_ids_corrupt, clean_cache, comp, l)
                recovery = (margin_patched - margin_corrupt) / total_effect
                results[(comp, l)].append(recovery)

        if (si + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{si+1}/{len(samples)}] {elapsed:.0f}s")

    # Print results table
    print(f"\n{'='*70}")
    print(f"Attribution Patching Results (N={len(samples)}, layers {layer_start}-{layer_end})")
    print(f"{'='*70}")
    print(f"{'Component':<15} {'median_rec':>10} {'mean_rec':>10} {'p(>0)':>8} {'n':>5}")
    print("-" * 50)

    all_results = []
    for l in range(layer_start, layer_end + 1):
        for comp in ['attn', 'mlp']:
            key = (comp, l)
            if key not in results:
                continue
            vals = results[key]
            med = np.median(vals)
            mean = np.mean(vals)
            n = len(vals)
            # Wilcoxon sign test for > 0
            from scipy.stats import wilcoxon
            if n > 5:
                _, p = wilcoxon(vals, alternative='greater')
            else:
                p = 1.0
            name = f"{comp}_L{l}"
            all_results.append({"component": name, "layer": l, "type": comp,
                                "median": float(med), "mean": float(mean),
                                "p": float(p), "n": n, "values": [float(v) for v in vals]})
            if med > 0.05 or p < 0.01:
                marker = " ***" if p < 0.001 else (" **" if p < 0.01 else (" *" if p < 0.05 else ""))
                print(f"{name:<15} {med:>10.3f} {mean:>10.3f} {p:>8.4f} {n:>5}{marker}")

    # Top-10 by median
    all_results.sort(key=lambda x: x["median"], reverse=True)
    print(f"\n{'='*50}")
    print("Top-10 components by median recovery:")
    for i, r in enumerate(all_results[:10]):
        print(f"  {i+1}. {r['component']:<15} med={r['median']:.3f} mean={r['mean']:.3f} p={r['p']:.4f}")

    # Save
    save_data = {
        "n_samples": len(samples),
        "layer_range": [layer_start, layer_end],
        "components": all_results,
    }
    with open(out_dir / "attribution_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()

