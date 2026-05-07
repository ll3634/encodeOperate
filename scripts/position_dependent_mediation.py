#!/usr/bin/env python3
"""
Position-Dependent Mediation (追加1)
=====================================
For Group A (evidence corruption) and Group B (distractor swap) pairs,
extract L20 activations at p0-p4 (5 positions through generated thought).

Measure "evidence-specific delta_action" = mean_A(|Δaction|) - mean_B(|Δaction|)
at each position, with bootstrap 95% CI.

Output → results/paired_corruption/position_mediation_results.json
         results/figures/position_mediation.png
"""

import os, sys, json, argparse, random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import torch
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, REACT_THOUGHT_SYSTEM_PROMPT
from steering.hook_utils import get_model_layers

# Reuse helpers from paired_corruption_analysis
from scripts.paired_corruption_analysis import (
    select_samples, make_corrupted_obs, rebuild_observation,
)

POSITION_NAMES = ["p0_input", "p1_25pct", "p2_50pct", "p3_75pct", "p4_100pct"]
SHORT_POS = ["p0", "p1", "p2", "p3", "p4"]
MIN_THOUGHT_TOKENS = 6


# ── Thought generation (from thought_erosion_probe) ─────────────────────────

def generate_thought(model, tokenizer, input_ids, max_new_tokens=120):
    """Generate thought tokens before Action/Final boundary."""
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

    thought_ids = gen_ids[:boundary_tok_idx]
    return thought_ids


# ── Activation extraction at 5 positions ────────────────────────────────────

def extract_at_positions(model, model_layers, input_ids, thought_ids, layer_idx=20):
    """Extract L{layer_idx} activation at p0-p4. Returns dict or None."""
    n_thought = len(thought_ids)
    if n_thought == 0:
        return None
    input_len = input_ids.shape[1]

    pos_idx = {}
    pos_idx["p0_input"]  = input_len - 1
    pos_idx["p1_25pct"]  = input_len + max(0, int(round(0.25 * n_thought)) - 1)
    pos_idx["p2_50pct"]  = input_len + max(0, int(round(0.50 * n_thought)) - 1)
    pos_idx["p3_75pct"]  = input_len + max(0, int(round(0.75 * n_thought)) - 1)
    pos_idx["p4_100pct"] = input_len + n_thought - 1

    thought_tensor = torch.tensor([thought_ids], dtype=torch.long, device=input_ids.device)
    full_ids = torch.cat([input_ids, thought_tensor], dim=1)

    captured = {}
    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        seq = h[0].detach().float().cpu()
        for name, idx in pos_idx.items():
            if idx < seq.shape[0]:
                captured[name] = seq[idx].numpy()

    handle = model_layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(full_ids)
    except Exception as e:
        handle.remove()
        return None
    handle.remove()

    if len(captured) != len(POSITION_NAMES):
        return None
    return captured


# ── Prompt building (REACT_THOUGHT_SYSTEM_PROMPT for thought generation) ────

def build_thought_prompt(tokenizer, question, query, observation):
    pb = PromptBuilder(tools=["search", "calculator"],
                       system_template=REACT_THOUGHT_SYSTEM_PROMPT)
    steps = [{"action": "search", "action_input": query,
              "observation": observation[:1500]}]
    messages = pb.build_full_prompt(question, steps)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


# ── Bootstrap CI ────────────────────────────────────────────────────────────

def bootstrap_ci(data_a, data_b, n_boot=2000, ci=0.95, seed=42):
    """Bootstrap CI for mean(data_a) - mean(data_b)."""
    rng = np.random.RandomState(seed)
    n = len(data_a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        diffs.append(data_a[idx].mean() - data_b[idx].mean())
    diffs = np.sort(diffs)
    lo = np.percentile(diffs, (1 - ci) / 2 * 100)
    hi = np.percentile(diffs, (1 + ci) / 2 * 100)
    return float(np.mean(diffs)), float(lo), float(hi)


# ── Main experiment ──────────────────────────────────────────────────────────

def run_experiment(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    model_layers = get_model_layers(model)
    device = next(model.parameters()).device

    # Load directions
    ev_data = np.load(args.evidence_dir)
    evidence_dir = ev_data["decision_direction"].astype(np.float32)
    evidence_dir /= np.linalg.norm(evidence_dir) + 1e-12

    act_data = np.load(args.action_dir)
    action_dir = act_data["decision_direction"].astype(np.float32)
    action_dir /= np.linalg.norm(action_dir) + 1e-12

    print(f"cos(evidence, action) = {np.dot(evidence_dir, action_dir):.4f}")

    # Select samples (same as original experiment)
    samples = select_samples(args.baseline_trace, args.hotpotqa_data,
                             n=args.n_samples, seed=args.seed)

    rng = random.Random(args.seed)

    # Storage: per-position, per-group arrays of |Δaction|
    delta_action = {pos: {"A": [], "B": []} for pos in POSITION_NAMES}
    delta_evidence = {pos: {"A": [], "B": []} for pos in POSITION_NAMES}
    skipped = 0
    per_sample = []

    teacher_forced = getattr(args, 'teacher_forced', False)
    mode_str = "TEACHER-FORCED" if teacher_forced else "FREE-GEN"
    print(f"\nMode: {mode_str}")

    for group in ["A", "B"]:
        print(f"\n=== Group {group} ({len(samples)} pairs) ===", flush=True)
        for i, sample in enumerate(samples):
            clean_obs, corrupted_obs = make_corrupted_obs(sample, group, rng)

            prompt_clean = build_thought_prompt(
                tokenizer, sample["question"], sample["step0_query"], clean_obs)
            prompt_corrupt = build_thought_prompt(
                tokenizer, sample["question"], sample["step0_query"], corrupted_obs)

            input_clean = tokenizer.encode(prompt_clean, return_tensors="pt").to(device)
            input_corrupt = tokenizer.encode(prompt_corrupt, return_tensors="pt").to(device)

            if teacher_forced:
                # Teacher-forced: generate thought from CLEAN only,
                # use same tokens for both conditions
                try:
                    thought_clean = generate_thought(model, tokenizer, input_clean)
                except Exception as e:
                    print(f"  SKIP {sample['sample_id'][:20]} gen error: {e}")
                    skipped += 1
                    continue

                if len(thought_clean) < MIN_THOUGHT_TOKENS:
                    print(f"  SKIP {sample['sample_id'][:20]} short thought "
                          f"(len={len(thought_clean)})")
                    skipped += 1
                    continue

                # Same thought tokens for both conditions
                thought_corrupt = thought_clean
            else:
                # Free-gen: generate independent thoughts for both
                try:
                    thought_clean = generate_thought(model, tokenizer, input_clean)
                    thought_corrupt = generate_thought(model, tokenizer, input_corrupt)
                except Exception as e:
                    print(f"  SKIP {sample['sample_id'][:20]} gen error: {e}")
                    skipped += 1
                    continue

                if len(thought_clean) < MIN_THOUGHT_TOKENS or len(thought_corrupt) < MIN_THOUGHT_TOKENS:
                    print(f"  SKIP {sample['sample_id'][:20]} short thought "
                          f"(clean={len(thought_clean)}, corrupt={len(thought_corrupt)})")
                    skipped += 1
                    continue

            # Extract at 5 positions
            acts_clean = extract_at_positions(
                model, model_layers, input_clean, thought_clean, layer_idx=args.layer)
            acts_corrupt = extract_at_positions(
                model, model_layers, input_corrupt, thought_corrupt, layer_idx=args.layer)

            if acts_clean is None or acts_corrupt is None:
                skipped += 1
                continue

            sample_row = {"sample_id": sample["sample_id"], "group": group}
            for pos in POSITION_NAMES:
                dh = acts_clean[pos] - acts_corrupt[pos]
                da = abs(float(np.dot(dh, action_dir)))
                de = abs(float(np.dot(dh, evidence_dir)))
                delta_action[pos][group].append(da)
                delta_evidence[pos][group].append(de)
                sample_row[f"da_{pos}"] = da
                sample_row[f"de_{pos}"] = de
            per_sample.append(sample_row)

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(samples)}]", flush=True)

    print(f"\nSkipped: {skipped}")

    # ── Compute evidence-specific delta at each position ─────────────────────
    print("\n" + "=" * 70)
    print("POSITION-DEPENDENT MEDIATION RESULTS")
    print("=" * 70)

    position_results = {}
    for pos, short in zip(POSITION_NAMES, SHORT_POS):
        arr_a = np.array(delta_action[pos]["A"])
        arr_b = np.array(delta_action[pos]["B"])
        n = min(len(arr_a), len(arr_b))
        arr_a, arr_b = arr_a[:n], arr_b[:n]

        mean_diff, ci_lo, ci_hi = bootstrap_ci(arr_a, arr_b, n_boot=2000)
        u, p = mannwhitneyu(arr_a, arr_b, alternative="greater")

        position_results[pos] = {
            "mean_A": float(arr_a.mean()),
            "mean_B": float(arr_b.mean()),
            "evidence_specific_shift": mean_diff,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "mann_whitney_U": float(u), "mann_whitney_p": float(p),
            "n": n,
        }
        print(f"{short}: shift={mean_diff:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]  "
              f"A={arr_a.mean():.4f}  B={arr_b.mean():.4f}  MW p={p:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = "_tf" if teacher_forced else ""
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_samples": args.n_samples, "layer": args.layer, "seed": args.seed,
            "model": args.model, "skipped": skipped,
            "teacher_forced": teacher_forced,
        },
        "position_results": position_results,
        "per_sample": per_sample,
    }
    out_path = os.path.join(args.output_dir, f"position_mediation_results{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shifts = [position_results[p]["evidence_specific_shift"] for p in POSITION_NAMES]
        ci_los = [position_results[p]["ci_lo"] for p in POSITION_NAMES]
        ci_his = [position_results[p]["ci_hi"] for p in POSITION_NAMES]
        errs_lo = [s - l for s, l in zip(shifts, ci_los)]
        errs_hi = [h - s for s, h in zip(shifts, ci_his)]

        fig, ax = plt.subplots(figsize=(7, 4))
        xs = range(5)
        ax.errorbar(xs, shifts, yerr=[errs_lo, errs_hi],
                     fmt='o-', capsize=5, color='#2196F3', linewidth=2, markersize=8)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(SHORT_POS)
        ax.set_xlabel("Position through Thought generation")
        ax.set_ylabel("Evidence-specific Δaction\n(mean_A − mean_B)")
        ax.set_title("Position-Dependent Mediation: Evidence → Action Causal Shift")
        fig.tight_layout()

        fig_dir = os.path.join(os.path.dirname(args.output_dir), "figures")
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, f"position_mediation{suffix}.png")
        fig.savefig(fig_path, dpi=150)
        print(f"Figure saved: {fig_path}")
        plt.close(fig)
    except Exception as e:
        print(f"Plot failed: {e}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--baseline-trace",
                    default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--hotpotqa-data",
                    default="data/hotpotqa/hotpot_dev_distractor_v1.json")
    ap.add_argument("--evidence-dir",
                    default="results/phase1_probe/probe_direction_l20.npz")
    ap.add_argument("--action-dir",
                    default="steering/directions/direction_search_v3_layer20.npz")
    ap.add_argument("--output-dir", default="results/paired_corruption")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--teacher-forced", action="store_true",
                    help="Generate thought from clean only, teacher-force onto both conditions")
    args = ap.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
