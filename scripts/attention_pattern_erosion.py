#!/usr/bin/env python3
"""
Attention Pattern Erosion During Thought Generation
====================================================
Measures whether attn_L18 KV Group 2's attention mass on observation tokens
decays as the model generates thought tokens (p0 → p4).

For each sample (1-doc, with observation):
  1. Build REACT_THOUGHT prompt (question + observation)
  2. Generate thought tokens (greedy)
  3. Teacher-force the full sequence [prompt | thought_tokens]
  4. At each of 5 normalized positions (p0=input, p1=25%, ..., p4=100%),
     extract L18 attention weights from the query at that position
  5. Compute attention mass on observation token range, per KV group

Expected result: KV2 obs attention decays p0→p4 while other KV groups stay flat.

Output → results/attention_pattern_erosion/
"""

import os, sys, json, argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, REACT_THOUGHT_SYSTEM_PROMPT
from steering.hook_utils import get_model_layers

POSITION_NAMES = ["p0_input", "p1_25pct", "p2_50pct", "p3_75pct", "p4_100pct"]
MIN_THOUGHT_TOKENS = 6

# Qwen2.5-7B GQA config
N_Q_HEADS = 28
N_KV_HEADS = 4
HEAD_DIM = 128
HEADS_PER_KV = N_Q_HEADS // N_KV_HEADS  # 7

KV_GROUPS = {
    "KV0": list(range(0, 7)),
    "KV1": list(range(7, 14)),
    "KV2": list(range(14, 21)),
    "KV3": list(range(21, 28)),
}


# ── Prompt building ──────────────────────────────────────────────────────────

def build_thought_prompt(tokenizer, question, query, observation):
    """Build prompt with REACT_THOUGHT_SYSTEM_PROMPT (generates Thought first)."""
    pb = PromptBuilder(tools=["search", "calculator"],
                       system_template=REACT_THOUGHT_SYSTEM_PROMPT)
    steps = [{"action": "search", "action_input": query,
              "observation": observation[:1500]}]
    messages = pb.build_full_prompt(question, steps)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


def generate_thought(model, tokenizer, prompt, max_tokens=200):
    """Generate thought tokens greedily. Returns list of token ids."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    thought_ids = out[0, input_ids.shape[1]:].tolist()
    # Truncate at EOS
    eos_ids = [tokenizer.eos_token_id]
    if hasattr(tokenizer, 'additional_special_tokens_ids'):
        eos_ids.extend(tokenizer.additional_special_tokens_ids)
    for i, tid in enumerate(thought_ids):
        if tid in eos_ids:
            thought_ids = thought_ids[:i]
            break
    return thought_ids


# ── Observation token range ──────────────────────────────────────────────────

def find_obs_token_range(tokenizer, prompt):
    """Find the token range of the observation text in the tokenized prompt.
    Returns (obs_start, obs_end) as token indices, or None.
    """
    obs_marker = "Observation:"
    obs_pos = prompt.find(obs_marker)
    if obs_pos < 0:
        return None

    obs_text_start = obs_pos  # include "Observation:" label

    # End of observation: next section boundary
    end_markers = ["\nThought:", "\nAction:", "<|im_end|>"]
    obs_end_char = len(prompt)
    for m in end_markers:
        idx = prompt.find(m, obs_text_start + len(obs_marker))
        if 0 <= idx < obs_end_char:
            obs_end_char = idx

    # Use offset_mapping for accurate token→char alignment
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]

    tok_start = None
    tok_end = None
    for i, (cs, ce) in enumerate(offsets):
        if tok_start is None and ce > obs_text_start:
            tok_start = i
        if cs >= obs_end_char:
            tok_end = i
            break
    if tok_end is None:
        tok_end = len(offsets)
    if tok_start is None:
        return None
    return tok_start, tok_end


# ── Attention extraction ─────────────────────────────────────────────────────

def extract_attention_at_positions(model, input_ids, thought_ids,
                                   layer_idx, obs_range):
    """
    Teacher-forced forward pass on [input_ids | thought_ids].
    At p0-p4 query positions, compute POST-RoPE attention weights and return
    obs-token attention mass per KV group.
    """
    n_thought = len(thought_ids)
    if n_thought < MIN_THOUGHT_TOKENS:
        return None

    input_len = input_ids.shape[1]
    obs_start, obs_end = obs_range

    # Position indices (absolute, in full sequence)
    pos_idx = {
        "p0_input":  input_len - 1,
        "p1_25pct":  input_len + max(0, int(round(0.25 * n_thought)) - 1),
        "p2_50pct":  input_len + max(0, int(round(0.50 * n_thought)) - 1),
        "p3_75pct":  input_len + max(0, int(round(0.75 * n_thought)) - 1),
        "p4_100pct": input_len + n_thought - 1,
    }

    # Full sequence
    thought_tensor = torch.tensor([thought_ids], dtype=torch.long,
                                  device=input_ids.device)
    full_ids = torch.cat([input_ids, thought_tensor], dim=1)
    seq_len = full_ids.shape[1]

    # Hook Q and K projections (pre-RoPE)
    layers = get_model_layers(model)
    attn_mod = layers[layer_idx].self_attn
    captured = {}

    def hook_q(module, inp, out):
        captured['q'] = out.detach()

    def hook_k(module, inp, out):
        captured['k'] = out.detach()

    handles = [
        attn_mod.q_proj.register_forward_hook(hook_q),
        attn_mod.k_proj.register_forward_hook(hook_k),
    ]

    try:
        with torch.no_grad():
            model(full_ids)
    finally:
        for h in handles:
            h.remove()

    # Reshape: (1, seq, heads*dim) → (1, heads, seq, dim)
    q = captured['q'].float().view(1, seq_len, N_Q_HEADS, HEAD_DIM).transpose(1, 2)
    k = captured['k'].float().view(1, seq_len, N_KV_HEADS, HEAD_DIM).transpose(1, 2)

    # Apply RoPE for position-accurate attention
    # rotary_emb lives on the base model, not on individual attention modules
    base_model = model.model if hasattr(model, 'model') else model
    rotary_emb = base_model.rotary_emb
    position_ids = torch.arange(seq_len, device=full_ids.device).unsqueeze(0)
    cos, sin = rotary_emb(k, position_ids)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    # Repeat K for GQA: (1, 4, seq, 128) → (1, 28, seq, 128)
    k = k.repeat_interleave(HEADS_PER_KV, dim=1)

    # For each position, compute attention weights
    results = {}
    for pos_name, qi in pos_idx.items():
        q_pos = q[:, :, qi:qi+1, :]        # (1, 28, 1, 128)
        # Causal mask: can only attend to positions ≤ qi
        k_causal = k[:, :, :qi+1, :]       # (1, 28, qi+1, 128)
        scores = torch.matmul(q_pos, k_causal.transpose(-2, -1)) / (HEAD_DIM ** 0.5)
        attn = torch.softmax(scores, dim=-1)  # (1, 28, 1, qi+1)
        attn = attn[0, :, 0, :].cpu().numpy()  # (28, qi+1)

        # Compute obs attention mass per KV group
        obs_end_clamped = min(obs_end, qi + 1)
        if obs_start >= obs_end_clamped:
            # Query position is before obs tokens — skip
            obs_mass = {gn: 0.0 for gn in KV_GROUPS}
        else:
            obs_mass = {}
            for gn, heads in KV_GROUPS.items():
                mass = attn[heads, obs_start:obs_end_clamped].sum(axis=1).mean()
                obs_mass[gn] = float(mass)

        results[pos_name] = {
            "obs_mass": obs_mass,
            "total_heads_attn_on_obs": {
                gn: attn[heads, obs_start:obs_end_clamped].sum(axis=1).tolist()
                for gn, heads in KV_GROUPS.items()
            } if obs_start < obs_end_clamped else {},
        }

    return results



# ── Main experiment ──────────────────────────────────────────────────────────

def run_experiment(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Load dataset: join labels with baseline trace (for query/observation)
    print("Loading dataset...", flush=True)
    labels_path = Path(args.labels)
    with open(labels_path) as f:
        label_data = [json.loads(line) for line in f]

    bl_map = {}
    with open(args.baseline_trace) as f:
        for line in f:
            ep = json.loads(line)
            bl_map[ep["sample_id"]] = ep

    # Filter: label=1, has valid step-1 observation
    valid = []
    for ld in label_data:
        if ld["label"] != 1:
            continue
        sid = ld["sample_id"]
        ep = bl_map.get(sid)
        if not ep or not ep.get("steps"):
            continue
        s0 = ep["steps"][0]
        if s0.get("action") != "search" or not s0.get("observation"):
            continue
        valid.append({
            "question": ld["question"],
            "query": s0["action_input"],
            "observation": s0["observation"],
            "sample_id": sid,
        })
    if args.max_samples > 0:
        valid = valid[:args.max_samples]
    print(f"Samples: {len(valid)} (label=1 with obs, from {len(label_data)} total)",
          flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-sample results
    all_results = []
    skipped = 0

    for i, sample in enumerate(valid):
        question = sample["question"]
        query = sample["query"]
        observation = sample["observation"]

        # Build prompt
        prompt = build_thought_prompt(tokenizer, question, query, observation)

        # Find observation tokens
        obs_range = find_obs_token_range(tokenizer, prompt)
        if obs_range is None:
            skipped += 1
            continue

        obs_start, obs_end = obs_range
        n_obs_tokens = obs_end - obs_start
        if n_obs_tokens < 5:
            skipped += 1
            continue

        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        # Generate thought tokens
        thought_ids = generate_thought(model, tokenizer, prompt,
                                        max_tokens=args.max_thought_tokens)
        if len(thought_ids) < MIN_THOUGHT_TOKENS:
            skipped += 1
            continue

        # Extract attention at p0-p4
        result = extract_attention_at_positions(
            model, input_ids, thought_ids,
            layer_idx=args.layer, obs_range=obs_range
        )
        if result is None:
            skipped += 1
            continue

        result["sample_idx"] = i
        result["n_obs_tokens"] = n_obs_tokens
        result["n_thought_tokens"] = len(thought_ids)
        result["input_len"] = input_ids.shape[1]
        all_results.append(result)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(valid)}] processed={len(all_results)}, "
                  f"skipped={skipped}", flush=True)

    print(f"\nDone: {len(all_results)} samples, {skipped} skipped", flush=True)

    # ── Analysis ─────────────────────────────────────────────────────────────

    # Build arrays: (n_samples, 5_positions, 4_kv_groups)
    n = len(all_results)
    kv_names = sorted(KV_GROUPS.keys())  # KV0, KV1, KV2, KV3
    obs_mass_arr = np.zeros((n, len(POSITION_NAMES), len(kv_names)))

    for si, r in enumerate(all_results):
        for pi, pname in enumerate(POSITION_NAMES):
            for ki, kvn in enumerate(kv_names):
                obs_mass_arr[si, pi, ki] = r[pname]["obs_mass"][kvn]

    # Summary statistics
    summary = {
        "n_samples": n,
        "skipped": skipped,
        "layer": args.layer,
        "timestamp": datetime.now().isoformat(),
    }

    # Per KV group: mean obs attention at each position
    group_trends = {}
    for ki, kvn in enumerate(kv_names):
        trend = {}
        for pi, pname in enumerate(POSITION_NAMES):
            vals = obs_mass_arr[:, pi, ki]
            trend[pname] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "median": float(np.median(vals)),
            }
        # Spearman correlation: position (0-4) vs obs_mass
        per_sample_rhos = []
        for si in range(n):
            rho, _ = spearmanr([0, 1, 2, 3, 4], obs_mass_arr[si, :, ki])
            if not np.isnan(rho):
                per_sample_rhos.append(rho)
        # Also: paired test p0 vs p4
        p0_vals = obs_mass_arr[:, 0, ki]
        p4_vals = obs_mass_arr[:, 4, ki]
        if len(p0_vals) > 1:
            mw_stat, mw_p = mannwhitneyu(p0_vals, p4_vals, alternative="two-sided")
        else:
            mw_stat, mw_p = 0, 1.0

        decay_pct = 0
        if np.mean(p0_vals) > 0:
            decay_pct = (np.mean(p0_vals) - np.mean(p4_vals)) / np.mean(p0_vals) * 100

        group_trends[kvn] = {
            "positions": trend,
            "decay_p0_to_p4": {
                "p0_mean": float(np.mean(p0_vals)),
                "p4_mean": float(np.mean(p4_vals)),
                "decay_pct": float(decay_pct),
                "MW_U": float(mw_stat),
                "MW_p": float(mw_p),
            },
            "per_sample_spearman_rho": {
                "mean": float(np.mean(per_sample_rhos)) if per_sample_rhos else 0,
                "median": float(np.median(per_sample_rhos)) if per_sample_rhos else 0,
                "n_negative": sum(1 for r in per_sample_rhos if r < 0),
                "n_total": len(per_sample_rhos),
            },
        }

    summary["group_trends"] = group_trends

    # Print summary table
    print("\n" + "=" * 70)
    print(f"ATTENTION PATTERN EROSION — Layer {args.layer}, N={n}")
    print("=" * 70)
    print(f"\n{'KV Group':<10}", end="")
    for pname in POSITION_NAMES:
        print(f"  {pname:>10}", end="")
    print(f"  {'decay%':>8}  {'MW p':>8}")
    print("-" * 80)
    for kvn in kv_names:
        gt = group_trends[kvn]
        print(f"{kvn:<10}", end="")
        for pname in POSITION_NAMES:
            print(f"  {gt['positions'][pname]['mean']:>10.4f}", end="")
        d = gt["decay_p0_to_p4"]
        print(f"  {d['decay_pct']:>7.1f}%  {d['MW_p']:>8.4f}")
    print()

    # Save
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save per-sample data (compact)
    per_sample_compact = []
    for r in all_results:
        entry = {"idx": r["sample_idx"], "n_obs": r["n_obs_tokens"],
                 "n_thought": r["n_thought_tokens"]}
        for pname in POSITION_NAMES:
            entry[pname] = r[pname]["obs_mass"]
        per_sample_compact.append(entry)
    with open(out_dir / "per_sample.jsonl", "w") as f:
        for entry in per_sample_compact:
            f.write(json.dumps(entry) + "\n")

    print(f"\nSaved to {out_dir}/")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--labels",
                        default="results/phase1_probe/labels.jsonl")
    parser.add_argument("--baseline-trace",
                        default="results/l20_rho020_n500/baseline_results.jsonl")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Max samples (label=1). 0=all")
    parser.add_argument("--max-thought-tokens", type=int, default=200)
    parser.add_argument("--output-dir",
                        default="results/attention_pattern_erosion")
    args = parser.parse_args()
    run_experiment(args)