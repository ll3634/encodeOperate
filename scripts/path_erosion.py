#!/usr/bin/env python3
"""Step 4: Path erosion — does mediation effect decay during thought generation?

For each (clean, corrupted) pair:
  1. Generate Thought tokens from clean input (teacher-forced)
  2. At each position p0-p4, measure:
     a. Total effect: action_dir · (h_clean - h_corrupt) at that position
     b. Patched effect: patch top components clean→corrupt, measure recovery
  3. If recovery(p0) >> recovery(p4), mediation path erodes during thought

Uses action_dir projection (not margin) since intermediate positions
don't have meaningful next-token action predictions.
"""

import json, sys, argparse, time, random
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, REACT_THOUGHT_SYSTEM_PROMPT, ACTION_TOKENS
from steering.hook_utils import get_model_layers
from scripts.minimal_corruption import build_prompt
from scripts.path_patching import TOP_COMPONENTS, CIRCUIT_COMPONENTS, BOTTOM_COMPONENTS

POSITION_NAMES = ["p0_input", "p1_25pct", "p2_50pct", "p3_75pct", "p4_100pct"]
MIN_THOUGHT_TOKENS = 4


def generate_thought(model, tokenizer, input_ids, max_new_tokens=120):
    """Generate thought tokens, stopping at Action/Final boundary."""
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = output_ids[0][input_ids.shape[1]:].tolist()

    boundary_tok_idx = len(gen_ids)
    accumulated = ""
    for tok_idx, tid in enumerate(gen_ids):
        tok_text = tokenizer.decode([tid], skip_special_tokens=True)
        accumulated += tok_text
        action_pos = accumulated.find("\nAction")
        final_pos = accumulated.find("\nFinal")
        if action_pos >= 0 or final_pos >= 0:
            cut_char = min(
                action_pos if action_pos >= 0 else len(accumulated),
                final_pos if final_pos >= 0 else len(accumulated),
            )
            prefix_len = 0
            for j, t in enumerate(gen_ids[:tok_idx + 1]):
                prefix_len += len(tokenizer.decode([t], skip_special_tokens=True))
                if prefix_len > cut_char:
                    boundary_tok_idx = j
                    break
            else:
                boundary_tok_idx = tok_idx
            break
    return gen_ids[:boundary_tok_idx]


def build_thought_prompt(tokenizer, question, query, observation):
    """Build prompt with REACT_THOUGHT_SYSTEM_PROMPT for thought generation."""
    pb = PromptBuilder(tools=["search", "calculator"],
                       system_template=REACT_THOUGHT_SYSTEM_PROMPT)
    steps = [{"action": "search", "action_input": query,
              "observation": observation[:1500]}]
    messages = pb.build_full_prompt(question, steps)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


def get_position_indices(input_len, n_thought):
    """Compute token indices for p0-p4."""
    return {
        "p0_input":  input_len - 1,
        "p1_25pct":  input_len + max(0, int(round(0.25 * n_thought)) - 1),
        "p2_50pct":  input_len + max(0, int(round(0.50 * n_thought)) - 1),
        "p3_75pct":  input_len + max(0, int(round(0.75 * n_thought)) - 1),
        "p4_100pct": input_len + n_thought - 1,
    }


def capture_at_positions(model, layers, full_ids, pos_indices, layer_idx,
                          patch_specs=None, source_cache=None):
    """Run forward, capture residual at specified positions. Optionally patch components.

    Returns dict {pos_name: activation_vector (numpy)} and per-component caches.
    If patch_specs is provided, patches those components' outputs at ALL positions.
    source_cache maps (comp, layer, pos_name) -> tensor.
    """
    captured = {}
    comp_outputs = {}  # (comp, layer, pos_name) -> tensor
    handles = []

    # Capture hook at target layer
    def capture_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        seq = h[0].detach().float().cpu()
        for name, idx in pos_indices.items():
            if idx < seq.shape[0]:
                captured[name] = seq[idx].numpy()

    handles.append(layers[layer_idx].register_forward_hook(capture_hook))

    # Component capture hooks (for caching)
    if source_cache is None:
        # Cache mode: save component outputs at all positions
        n_layers = len(layers)
        for l in range(n_layers):
            for comp in ['attn', 'mlp']:
                target = layers[l].self_attn if comp == 'attn' else layers[l].mlp
                def make_cache_hook(c, li):
                    def hook_fn(module, inp, out):
                        h = out[0] if isinstance(out, tuple) else out
                        seq = h[0].detach()
                        for name, idx in pos_indices.items():
                            if idx < seq.shape[0]:
                                comp_outputs[(c, li, name)] = seq[idx].clone()
                    return hook_fn
                handles.append(target.register_forward_hook(make_cache_hook(comp, l)))

    # Patch hooks
    if patch_specs and source_cache:
        for comp, l in patch_specs:
            target = layers[l].self_attn if comp == 'attn' else layers[l].mlp
            def make_patch_hook(c, li):
                def hook_fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    for name, idx in pos_indices.items():
                        key = (c, li, name)
                        if key in source_cache and idx < h.shape[1]:
                            h[0, idx, :] = source_cache[key].to(h.dtype)
                    if isinstance(out, tuple):
                        return (h,) + out[1:]
                    return h
                return hook_fn
            handles.append(target.register_forward_hook(make_patch_hook(comp, l)))

    with torch.no_grad():
        model(full_ids)
    for h in handles:
        h.remove()

    return captured, comp_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/minimal_corruption/full_corruption_data.jsonl")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--output-dir", default="results/path_erosion")
    parser.add_argument("--layer", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    action_dir = np.load("steering/directions/direction_search_v3_layer20.npz")["decision_direction_normalized"]
    action_dir = action_dir / np.linalg.norm(action_dir)

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
    layers = get_model_layers(model)
    n_layers = len(layers)
    device = next(model.parameters()).device

    # Test configs for patching
    patch_configs = {
        "circuit": CIRCUIT_COMPONENTS,
        "top7": TOP_COMPONENTS[:7],
        "bottom7": BOTTOM_COMPONENTS[:7],
    }

    # Results: position -> config -> [recovery values]
    results = {pos: {cfg: [] for cfg in list(patch_configs.keys()) + ["total_effect"]}
               for pos in POSITION_NAMES}
    skipped = 0
    t0 = time.time()

    for si, s in enumerate(samples):
        # Build thought prompts
        prompt_clean = build_thought_prompt(
            tokenizer, s["question"], s["step0_query"], s["obs_clean"])
        prompt_corrupt = build_thought_prompt(
            tokenizer, s["question"], s["step0_query"], s["obs_corrupt_A"])

        input_clean = tokenizer.encode(prompt_clean, return_tensors="pt").to(device)
        input_corrupt = tokenizer.encode(prompt_corrupt, return_tensors="pt").to(device)

        # Generate thought from CLEAN (teacher-forced for both)
        try:
            thought_ids = generate_thought(model, tokenizer, input_clean)
        except Exception:
            skipped += 1
            continue
        if len(thought_ids) < MIN_THOUGHT_TOKENS:
            skipped += 1
            continue

        # Build full sequences
        thought_tensor = torch.tensor([thought_ids], dtype=torch.long, device=device)
        full_clean = torch.cat([input_clean, thought_tensor], dim=1)
        full_corrupt = torch.cat([input_corrupt, thought_tensor], dim=1)

        # IMPORTANT: separate position indices for each sequence (different input lengths)
        pos_indices_clean = get_position_indices(input_clean.shape[1], len(thought_ids))
        pos_indices_corrupt = get_position_indices(input_corrupt.shape[1], len(thought_ids))

        # Run clean: capture residuals + component outputs
        acts_clean, cache_clean = capture_at_positions(
            model, layers, full_clean, pos_indices_clean, args.layer)
        # Run corrupt: capture residuals (using CORRUPT position indices)
        acts_corrupt, _ = capture_at_positions(
            model, layers, full_corrupt, pos_indices_corrupt, args.layer)

        if len(acts_clean) != 5 or len(acts_corrupt) != 5:
            skipped += 1
            continue

        # Compute total effect at each position
        valid = True
        for pos in POSITION_NAMES:
            te = float(np.dot(acts_clean[pos] - acts_corrupt[pos], action_dir))
            results[pos]["total_effect"].append(te)
            if abs(te) < 0.01:
                valid = False

        if not valid:
            # Still recorded total_effect, but skip patching
            for pos in POSITION_NAMES:
                for cfg in patch_configs:
                    results[pos][cfg].append(float('nan'))
            continue

        # Run patched conditions (corrupt input + clean component patches)
        for cfg_name, specs in patch_configs.items():
            acts_patched, _ = capture_at_positions(
                model, layers, full_corrupt, pos_indices_corrupt, args.layer,
                patch_specs=specs, source_cache=cache_clean)
            for pos in POSITION_NAMES:
                if pos in acts_patched:
                    te = results[pos]["total_effect"][-1]
                    patched_proj = float(np.dot(acts_patched[pos], action_dir))
                    corrupt_proj = float(np.dot(acts_corrupt[pos], action_dir))
                    recovery = (patched_proj - corrupt_proj) / te if abs(te) > 0.01 else float('nan')
                    results[pos][cfg_name].append(recovery)
                else:
                    results[pos][cfg_name].append(float('nan'))

        if (si + 1) % 10 == 0:
            print(f"  [{si+1}/{len(samples)}] {time.time()-t0:.0f}s, skipped={skipped}")

    # Print results
    print(f"\n{'='*70}")
    print(f"Path Erosion Results (N={len(samples)}, skipped={skipped})")
    print(f"{'='*70}")

    print(f"\n{'Position':<12}", end="")
    print(f"{'total_eff':>10}", end="")
    for cfg in patch_configs:
        print(f"  {cfg+'_rec':>12}", end="")
    print()
    print("-" * 60)

    for pos in POSITION_NAMES:
        short = pos.split("_")[0]
        te_vals = [v for v in results[pos]["total_effect"] if not np.isnan(v)]
        print(f"{short:<12}", end="")
        print(f"{np.median(te_vals):>10.3f}", end="")
        for cfg in patch_configs:
            vals = [v for v in results[pos][cfg] if not np.isnan(v)]
            if vals:
                print(f"  {np.median(vals):>12.3f}", end="")
            else:
                print(f"  {'N/A':>12}", end="")
        print()

    # Erosion ratio: p0 recovery / p4 recovery
    print(f"\n--- Erosion Ratios (p0/p4) ---")
    for cfg in patch_configs:
        p0_vals = [v for v in results["p0_input"][cfg] if not np.isnan(v)]
        p4_vals = [v for v in results["p4_100pct"][cfg] if not np.isnan(v)]
        if p0_vals and p4_vals:
            ratio = np.median(p0_vals) / (np.median(p4_vals) + 1e-9)
            from scipy.stats import mannwhitneyu
            _, p = mannwhitneyu(p0_vals, p4_vals, alternative='greater')
            print(f"  {cfg}: p0_med={np.median(p0_vals):.3f}, p4_med={np.median(p4_vals):.3f}, "
                  f"ratio={ratio:.2f}, MW p={p:.4f}")

    # Save
    save_data = {"n_samples": len(samples), "skipped": skipped, "results": {}}
    for pos in POSITION_NAMES:
        save_data["results"][pos] = {}
        for cfg in list(patch_configs.keys()) + ["total_effect"]:
            vals = results[pos][cfg]
            clean_vals = [v for v in vals if not np.isnan(v)]
            save_data["results"][pos][cfg] = {
                "median": float(np.median(clean_vals)) if clean_vals else None,
                "mean": float(np.mean(clean_vals)) if clean_vals else None,
                "n": len(clean_vals),
                "values": [float(v) for v in vals],
            }
    with open(out_dir / "path_erosion_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()


