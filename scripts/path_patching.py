#!/usr/bin/env python3
"""Step 3: Exact path patching — sufficiency, necessity, and additivity.

Tests:
  A. Sufficiency: patch top-k components clean→corrupt, measure combined recovery
  B. Necessity: in clean run, replace components with corrupt values, measure margin drop
  C. Additivity: compare sum-of-individuals vs joint patch
  D. Control: patch bottom-k components (expected ~0 recovery)
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
from scripts.attribution_patching import cache_component_outputs


def compute_margin_from_logits(logits, tokenizer):
    """Compute search-stop margin from logits tensor."""
    log_probs = torch.log_softmax(logits, dim=-1)
    tool_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]
    return (torch.logsumexp(log_probs[tool_ids], 0) - torch.logsumexp(log_probs[fin_ids], 0)).item()


def multi_patch_margin(model, tokenizer, input_ids, source_cache, patch_specs):
    """Run forward with multiple components patched from source_cache.

    patch_specs: list of (comp_type, layer_idx) tuples
    source_cache: dict mapping (comp_type, layer_idx) -> tensor
    """
    layers = get_model_layers(model)
    handles = []

    for comp, l in patch_specs:
        src_vec = source_cache[(comp, l)]
        target = layers[l].self_attn if comp == 'attn' else layers[l].mlp

        def make_hook(vec):
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h[0, -1, :] = vec.to(h.dtype)
                if isinstance(out, tuple):
                    return (h,) + out[1:]
                return h
            return hook_fn

        handles.append(target.register_forward_hook(make_hook(src_vec)))

    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :]
    for h in handles:
        h.remove()
    return compute_margin_from_logits(logits, tokenizer)


# Component sets to test
TOP_COMPONENTS = [
    ('attn', 22), ('attn', 18), ('mlp', 21), ('attn', 19),
    ('mlp', 18), ('mlp', 20), ('attn', 23),
]

BOTTOM_COMPONENTS = [
    ('attn', 14), ('mlp', 14), ('attn', 15), ('mlp', 16),
    ('attn', 16), ('mlp', 15), ('attn', 24),
]

# The originally hypothesized circuit
CIRCUIT_COMPONENTS = [('attn', 18), ('mlp', 20)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/minimal_corruption/full_corruption_data.jsonl")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--output-dir", default="results/path_patching")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    with open(args.data) as f:
        for line in f:
            samples.append(json.loads(line))
    samples = samples[:args.n]
    print(f"Loaded {len(samples)} samples")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    n_layers = len(get_model_layers(model))

    # Define test configurations
    configs = {
        # Sufficiency: patch clean→corrupt (in corrupt run, restore clean values)
        "suff_top1": TOP_COMPONENTS[:1],
        "suff_top2": TOP_COMPONENTS[:2],
        "suff_top3": TOP_COMPONENTS[:3],
        "suff_top5": TOP_COMPONENTS[:5],
        "suff_top7": TOP_COMPONENTS[:7],
        "suff_circuit": CIRCUIT_COMPONENTS,
        "suff_bottom7": BOTTOM_COMPONENTS[:7],
        # Individual top components for additivity check
        "suff_attn18_only": [('attn', 18)],
        "suff_mlp20_only": [('mlp', 20)],
        "suff_attn22_only": [('attn', 22)],
    }

    results = defaultdict(list)
    t0 = time.time()

    for si, s in enumerate(samples):
        prompt_clean = build_prompt(tokenizer, s["question"], s["step0_query"], s["obs_clean"])
        prompt_corrupt = build_prompt(tokenizer, s["question"], s["step0_query"], s["obs_corrupt_A"])

        clean_cache, margin_clean, input_ids_clean = cache_component_outputs(
            model, tokenizer, prompt_clean, n_layers)
        corrupt_cache, margin_corrupt, input_ids_corrupt = cache_component_outputs(
            model, tokenizer, prompt_corrupt, n_layers)

        total_effect = margin_clean - margin_corrupt
        if abs(total_effect) < 0.01:
            continue

        # A. Sufficiency tests: run corrupt, patch in clean values
        for name, specs in configs.items():
            if name.startswith("suff_"):
                m = multi_patch_margin(model, tokenizer, input_ids_corrupt,
                                       clean_cache, specs)
                recovery = (m - margin_corrupt) / total_effect
                results[name].append(recovery)

        # B. Necessity tests: run clean, patch in corrupt values
        for nec_name, specs in [("nec_top7", TOP_COMPONENTS[:7]),
                                 ("nec_circuit", CIRCUIT_COMPONENTS),
                                 ("nec_bottom7", BOTTOM_COMPONENTS[:7])]:
            m = multi_patch_margin(model, tokenizer, input_ids_clean,
                                   corrupt_cache, specs)
            drop = (margin_clean - m) / total_effect
            results[nec_name].append(drop)

        if (si + 1) % 10 == 0:
            print(f"  [{si+1}/{len(samples)}] {time.time()-t0:.0f}s")

    # Print results
    print(f"\n{'='*70}")
    print(f"Path Patching Results (N={len(samples)})")
    print(f"{'='*70}")

    print("\n--- A. SUFFICIENCY (clean→corrupt recovery) ---")
    print(f"{'Config':<25} {'median':>8} {'mean':>8} {'std':>8} {'n':>5}")
    print("-" * 55)
    for name in sorted(results.keys()):
        if not name.startswith("suff_"):
            continue
        vals = results[name]
        print(f"{name:<25} {np.median(vals):>8.3f} {np.mean(vals):>8.3f} {np.std(vals):>8.3f} {len(vals):>5}")

    print("\n--- B. NECESSITY (clean margin drop when patching corrupt) ---")
    print(f"{'Config':<25} {'median':>8} {'mean':>8} {'std':>8} {'n':>5}")
    print("-" * 55)
    for name in sorted(results.keys()):
        if not name.startswith("nec_"):
            continue
        vals = results[name]
        print(f"{name:<25} {np.median(vals):>8.3f} {np.mean(vals):>8.3f} {np.std(vals):>8.3f} {len(vals):>5}")

    # Additivity check
    print("\n--- C. ADDITIVITY CHECK ---")
    n_valid = min(len(results["suff_attn18_only"]), len(results["suff_mlp20_only"]),
                  len(results["suff_circuit"]))
    if n_valid > 0:
        sum_indiv = [results["suff_attn18_only"][i] + results["suff_mlp20_only"][i]
                     for i in range(n_valid)]
        joint = results["suff_circuit"][:n_valid]
        print(f"  attn_L18 alone:     median={np.median(results['suff_attn18_only'][:n_valid]):.3f}")
        print(f"  mlp_L20 alone:      median={np.median(results['suff_mlp20_only'][:n_valid]):.3f}")
        print(f"  Sum of individuals: median={np.median(sum_indiv):.3f}")
        print(f"  Joint patch:        median={np.median(joint):.3f}")
        ratio = np.median(joint) / (np.median(sum_indiv) + 1e-9)
        print(f"  Joint/Sum ratio:    {ratio:.3f} (>1 = synergy, <1 = redundancy)")

    # Save
    save_data = {"n_samples": len(samples)}
    for name, vals in results.items():
        save_data[name] = {
            "median": float(np.median(vals)), "mean": float(np.mean(vals)),
            "std": float(np.std(vals)), "n": len(vals),
            "values": [float(v) for v in vals],
        }
    with open(out_dir / "path_patching_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()

