#!/usr/bin/env python3
"""
Extract Search-specific Direction Vector V2 - Based on Actual Margin Distribution.

Key insight: Popularity doesn't correlate with model's decision to use search.
The model wants to use search for ALL questions (margin > 0).

New approach:
1. Compute margin for a large set of samples
2. Find samples where model BARELY wants to use search (low margin, 0 < m < threshold)
3. Find samples where model STRONGLY wants to use search (high margin, m > threshold)
4. Compute direction from these empirical decision boundaries

This is analogous to the boundary-mining approach used for calculator direction.
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
import random

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.popqa import PopQADataset
from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import get_model_layers


def load_model(model_id: str, adapter_path: str = None):
    """Load model and tokenizer; optionally merge a PEFT adapter on top."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        from peft import PeftModel
        print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def build_react_prompt(tokenizer, question: str) -> str:
    """Build a ReAct-style prompt for a question."""
    # Keep prompt EXACTLY consistent with the agent prompt used during evaluation.
    pb = PromptBuilder()
    messages = [
        {"role": "system", "content": pb.build_system_prompt()},
        {"role": "user", "content": question},
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


def compute_margin_and_hidden(
    model, tokenizer, question: str, 
    layer: int = 12, position: int = -1
) -> Tuple[float, np.ndarray]:
    """Compute margin and extract hidden state at the *steering intervention site*.

    Best-practice: capture the hidden state via a forward hook on the transformer block
    (same layer semantics as SteeringHook / get_model_layers), rather than relying on
    `outputs.hidden_states[...]` indexing conventions.
    """
    prompt = build_react_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Capture hidden at the output of transformer block `layer`.
    layers = get_model_layers(model)
    num_layers = len(layers)
    actual_layer = layer if layer >= 0 else num_layers + layer
    if actual_layer < 0 or actual_layer >= num_layers:
        raise ValueError(f"Layer {layer} out of range [0, {num_layers})")

    captured = {}

    def capture_hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden

    handle = layers[actual_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            outputs = model(**inputs)
    finally:
        handle.remove()

    if "hidden" not in captured:
        raise RuntimeError("Failed to capture hidden state via forward hook")

    hs = captured["hidden"]
    seq_len = hs.shape[1]
    pos = (seq_len + position) if position < 0 else min(position, seq_len - 1)
    pos = max(0, min(int(pos), seq_len - 1))
    hidden = hs[0, pos, :].float().cpu().numpy().astype(np.float32)
    
    # Get margin
    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    # Token IDs for the *decision* token.
    # With DEFAULT_SYSTEM_PROMPT, the first generated token should be either "Action" or "Final".
    tool_tokens: List[int] = []
    finish_tokens: List[int] = []
    for token_str in ACTION_TOKENS["tool_call"]:
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        if ids:
            tool_tokens.append(ids[0])
    for token_str in ACTION_TOKENS["finish"]:
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        if ids:
            finish_tokens.append(ids[0])

    tool_logprob = torch.logsumexp(log_probs[tool_tokens], dim=0).item() if tool_tokens else -100.0
    finish_logprob = torch.logsumexp(log_probs[finish_tokens], dim=0).item() if finish_tokens else -100.0
    margin = tool_logprob - finish_logprob
    
    return margin, hidden


def compute_rms(arr: np.ndarray) -> float:
    """Compute root mean square."""
    return np.sqrt(np.mean(arr ** 2))


def main():
    parser = argparse.ArgumentParser(description="Extract Search Direction V2")
    parser.add_argument("--data-path", required=True, help="Path to PopQA JSONL")
    parser.add_argument("--output", required=True, help="Output NPZ file")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-path", default=None,
                        help="Optional PEFT adapter dir to merge on top of --model.")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--n-samples", type=int, default=200, help="Total samples to analyze")
    parser.add_argument("--n-boundary", type=int, default=50, help="Samples per boundary")
    parser.add_argument("--low-margin-max", type=float, default=8.0, help="Max margin for low group")
    parser.add_argument("--high-margin-min", type=float, default=10.0, help="Min margin for high group")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    rng = random.Random(args.seed)
    
    # Load dataset
    dataset = PopQADataset(args.data_path)
    samples = rng.sample(dataset.samples, min(args.n_samples, len(dataset.samples)))
    print(f"Analyzing {len(samples)} samples...")
    
    # Load model
    model, tokenizer = load_model(args.model, adapter_path=args.adapter_path)

    # Phase 1: Collect margins and hidden states for all samples
    print(f"\nPhase 1: Computing margins and hidden states...")
    all_data = []
    for sample in tqdm(samples, desc="Computing margins"):
        margin, hidden = compute_margin_and_hidden(
            model, tokenizer, sample.question, args.layer, args.position
        )
        all_data.append({
            "id": sample.id,
            "question": sample.question,
            "margin": margin,
            "hidden": hidden,
            "s_pop": sample.s_pop,
        })

    # Analyze margin distribution
    margins = [d["margin"] for d in all_data]
    print(f"\n=== Margin Distribution ===")
    print(f"  Mean: {np.mean(margins):.2f}")
    print(f"  Std: {np.std(margins):.2f}")
    print(f"  Min: {np.min(margins):.2f}")
    print(f"  Max: {np.max(margins):.2f}")
    print(f"  Percentiles: 10%={np.percentile(margins, 10):.2f}, "
          f"50%={np.percentile(margins, 50):.2f}, "
          f"90%={np.percentile(margins, 90):.2f}")

    # Phase 2: Use percentile-based boundaries
    p10 = np.percentile(margins, 20)  # Low margin boundary
    p90 = np.percentile(margins, 80)  # High margin boundary

    print(f"\nUsing adaptive boundaries: low_max={p10:.2f}, high_min={p90:.2f}")

    # Split into low and high margin groups
    low_margin_data = [d for d in all_data if d["margin"] <= p10]
    high_margin_data = [d for d in all_data if d["margin"] >= p90]

    print(f"Low-margin samples (<=P20): {len(low_margin_data)}")
    print(f"High-margin samples (>=P80): {len(high_margin_data)}")

    # Sample if needed
    n = args.n_boundary
    if len(low_margin_data) > n:
        low_margin_data = rng.sample(low_margin_data, n)
    if len(high_margin_data) > n:
        high_margin_data = rng.sample(high_margin_data, n)

    print(f"\nUsing {len(low_margin_data)} low-margin and {len(high_margin_data)} high-margin samples")

    # Phase 3: Compute direction
    h_low = np.stack([d["hidden"] for d in low_margin_data])
    h_high = np.stack([d["hidden"] for d in high_margin_data])

    h_low_mean = np.mean(h_low, axis=0)
    h_high_mean = np.mean(h_high, axis=0)

    # Direction: h_low - h_high points toward NON-ADOPT (less search)
    # This is consistent with existing code where +alpha pushes toward reject
    decision_direction = h_low_mean - h_high_mean
    direction_norm = np.linalg.norm(decision_direction)
    decision_direction_normalized = decision_direction / direction_norm
    direction_rms = compute_rms(decision_direction)

    print(f"\n=== Direction Results ===")
    print(f"Direction norm: {direction_norm:.4f}")
    print(f"Direction RMS: {direction_rms:.6f}")
    print(f"Low-margin mean: {np.mean([d['margin'] for d in low_margin_data]):.2f}")
    print(f"High-margin mean: {np.mean([d['margin'] for d in high_margin_data]):.2f}")

    # Generate random direction for control
    np.random.seed(args.seed)
    random_direction = np.random.randn(decision_direction.shape[0]).astype(np.float32)
    random_direction = random_direction / np.linalg.norm(random_direction) * direction_norm

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        decision_direction=decision_direction.astype(np.float16),
        decision_direction_normalized=decision_direction_normalized.astype(np.float16),
        random_direction=random_direction,
        h_low_margin_mean=h_low_mean.astype(np.float16),
        h_high_margin_mean=h_high_mean.astype(np.float16),
        direction_norm=np.float16(direction_norm),
        direction_rms=np.float16(direction_rms),
        layer=args.layer,
        n_low_margin=len(low_margin_data),
        n_high_margin=len(high_margin_data),
        low_margin_range=np.array([np.min([d["margin"] for d in low_margin_data]),
                                   np.max([d["margin"] for d in low_margin_data])]),
        high_margin_range=np.array([np.min([d["margin"] for d in high_margin_data]),
                                    np.max([d["margin"] for d in high_margin_data])]),
        all_margins=np.array(margins),
    )
    print(f"\nSaved to: {args.output}")

    # Save metadata
    meta_path = str(args.output).replace(".npz", "_meta.json")
    meta = {
        "layer": args.layer,
        "position": args.position,
        "n_samples_analyzed": len(samples),
        "n_low_margin": len(low_margin_data),
        "n_high_margin": len(high_margin_data),
        "low_margin_percentile": 20,
        "high_margin_percentile": 80,
        "low_margin_threshold": float(p10),
        "high_margin_threshold": float(p90),
        "direction_norm": float(direction_norm),
        "direction_rms": float(direction_rms),
        "margin_stats": {
            "mean": float(np.mean(margins)),
            "std": float(np.std(margins)),
            "min": float(np.min(margins)),
            "max": float(np.max(margins)),
        }
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()

