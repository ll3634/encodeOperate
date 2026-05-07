#!/usr/bin/env python3
"""
Extract Search-specific Direction Vector for Tool Adoption.

This script extracts a direction vector that encodes the decision boundary
for "use search tool" vs "don't use search tool" in a ReAct agent setting.

Methodology:
1. Collect LOW-pop PopQA samples (must search - model can't answer from memory)
2. Collect HIGH-pop PopQA samples (don't need search - model knows the answer)
3. For each sample, run the agent prompt and extract hidden states at Layer 12
4. Compute direction: h_must_search - h_no_need_search

This gives us a direction that points toward "use search tool".

NOTE: The existing calculator direction uses h_low - h_high which points toward NON-ADOPT.
For consistency with existing code, we compute h_high_pop - h_low_pop which also points
toward NON-ADOPT (don't use search).
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
import random

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.popqa import PopQADataset
from agent.prompts import PromptBuilder, ACTION_TOKENS


def load_model(model_id: str):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def build_react_prompt(tokenizer, question: str) -> str:
    """Build a ReAct-style prompt for a question."""
    pb = PromptBuilder()
    messages = [
        {"role": "system", "content": pb.build_system_prompt()},
        {"role": "user", "content": question},
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


def extract_hidden_state(
    model, tokenizer, question: str, 
    layer: int = 12, position: int = -1
) -> np.ndarray:
    """Extract hidden state at the decision point (before generating action)."""
    prompt = build_react_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Get hidden state at specified layer and position
    hidden_states = outputs.hidden_states[layer]  # (1, seq_len, hidden_dim)
    seq_len = hidden_states.shape[1]
    
    if position < 0:
        actual_pos = seq_len + position
    else:
        actual_pos = min(position, seq_len - 1)
    
    hidden = hidden_states[0, actual_pos, :].float().cpu().numpy().astype(np.float32)
    return hidden


def compute_margin(model, tokenizer, question: str) -> float:
    """Compute decision margin for a question (log P(Action) - log P(Final))."""
    prompt = build_react_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    tool_tokens = []
    finish_tokens = []
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
    return margin


def compute_rms(arr: np.ndarray) -> float:
    """Compute root mean square."""
    return np.sqrt(np.mean(arr ** 2))


def main():
    parser = argparse.ArgumentParser(description="Extract Search-specific Direction Vector")
    parser.add_argument("--data-path", required=True, help="Path to PopQA JSONL file")
    parser.add_argument("--output", required=True, help="Output NPZ file path")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=int, default=12, help="Layer to extract from")
    parser.add_argument("--position", type=int, default=-1, help="Token position")
    parser.add_argument("--n-low-pop", type=int, default=50, help="Number of low-pop samples")
    parser.add_argument("--n-high-pop", type=int, default=50, help="Number of high-pop samples")
    parser.add_argument("--low-pop-max", type=int, default=100, help="Max s_pop for low-pop")
    parser.add_argument("--high-pop-min", type=int, default=10000, help="Min s_pop for high-pop")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load dataset
    dataset = PopQADataset(args.data_path)
    all_samples = dataset.samples

    # Split by popularity
    low_pop_samples = [s for s in all_samples if s.s_pop <= args.low_pop_max]
    high_pop_samples = [s for s in all_samples if s.s_pop >= args.high_pop_min]

    print(f"Found {len(low_pop_samples)} samples with s_pop <= {args.low_pop_max}")
    print(f"Found {len(high_pop_samples)} samples with s_pop >= {args.high_pop_min}")

    # Sample subsets
    low_pop_subset = rng.sample(low_pop_samples, min(args.n_low_pop, len(low_pop_samples)))
    high_pop_subset = rng.sample(high_pop_samples, min(args.n_high_pop, len(high_pop_samples)))

    print(f"\nUsing {len(low_pop_subset)} low-pop (must search) samples")
    print(f"Using {len(high_pop_subset)} high-pop (no need search) samples")

    # Load model
    model, tokenizer = load_model(args.model)

    # Extract hidden states for low-pop samples (must search)
    print(f"\nExtracting hidden states from layer {args.layer}, position {args.position}...")

    h_low_pop = []  # Must search (adopt)
    margins_low = []
    for sample in tqdm(low_pop_subset, desc="Low-pop (must search)"):
        h = extract_hidden_state(model, tokenizer, sample.question, args.layer, args.position)
        m = compute_margin(model, tokenizer, sample.question)
        h_low_pop.append(h)
        margins_low.append(m)

    h_high_pop = []  # No need search (reject)
    margins_high = []
    for sample in tqdm(high_pop_subset, desc="High-pop (no need search)"):
        h = extract_hidden_state(model, tokenizer, sample.question, args.layer, args.position)
        m = compute_margin(model, tokenizer, sample.question)
        h_high_pop.append(h)
        margins_high.append(m)

    h_low_pop = np.stack(h_low_pop)
    h_high_pop = np.stack(h_high_pop)

    print(f"\nLow-pop hidden states: {h_low_pop.shape}")
    print(f"High-pop hidden states: {h_high_pop.shape}")

    # Compute direction: h_high_pop - h_low_pop
    # This points from "must search" to "no need search" (toward NON-ADOPT)
    # Compatible with existing code where +alpha pushes toward reject
    h_low_mean = np.mean(h_low_pop, axis=0)
    h_high_mean = np.mean(h_high_pop, axis=0)

    # Direction points toward NON-ADOPT (for compatibility with existing steering code)
    decision_direction = h_high_mean - h_low_mean
    direction_norm = np.linalg.norm(decision_direction)
    decision_direction_normalized = decision_direction / direction_norm

    # Compute RMS
    direction_rms = compute_rms(decision_direction)

    print(f"\n=== Results ===")
    print(f"Direction norm: {direction_norm:.4f}")
    print(f"Direction RMS: {direction_rms:.6f}")
    print(f"Direction shape: {decision_direction.shape}")

    print(f"\nMargin statistics:")
    print(f"  Low-pop (must search): mean={np.mean(margins_low):.2f}, std={np.std(margins_low):.2f}")
    print(f"  High-pop (no need search): mean={np.mean(margins_high):.2f}, std={np.std(margins_high):.2f}")

    # Also generate a random direction for control
    random_direction = np.random.randn(decision_direction.shape[0]).astype(np.float32)
    random_direction = random_direction / np.linalg.norm(random_direction) * direction_norm

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        decision_direction=decision_direction.astype(np.float16),
        decision_direction_normalized=decision_direction_normalized.astype(np.float16),
        random_direction=random_direction,
        h_low_pop_mean=h_low_mean.astype(np.float16),
        h_high_pop_mean=h_high_mean.astype(np.float16),
        direction_norm=np.float16(direction_norm),
        direction_rms=np.float16(direction_rms),
        layer=args.layer,
        n_low_pop=len(low_pop_subset),
        n_high_pop=len(high_pop_subset),
        margins_low=np.array(margins_low),
        margins_high=np.array(margins_high),
        low_pop_max=args.low_pop_max,
        high_pop_min=args.high_pop_min,
    )

    print(f"\nSaved to: {args.output}")

    # Also save metadata
    meta_path = str(args.output).replace(".npz", "_meta.json")
    meta = {
        "layer": args.layer,
        "position": args.position,
        "n_low_pop": len(low_pop_subset),
        "n_high_pop": len(high_pop_subset),
        "low_pop_max": args.low_pop_max,
        "high_pop_min": args.high_pop_min,
        "direction_norm": float(direction_norm),
        "direction_rms": float(direction_rms),
        "margin_low_mean": float(np.mean(margins_low)),
        "margin_low_std": float(np.std(margins_low)),
        "margin_high_mean": float(np.mean(margins_high)),
        "margin_high_std": float(np.std(margins_high)),
        "low_pop_samples": [s.id for s in low_pop_subset],
        "high_pop_samples": [s.id for s in high_pop_subset],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()

