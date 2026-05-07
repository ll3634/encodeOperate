#!/usr/bin/env python3
"""
Validate Search Direction Vector via Causal Intervention.

This script tests whether the new search-specific direction can actually
change the model's margin (log P(search) - log P(finish)) when applied
as a steering intervention.

Success criteria:
- Positive rho (push toward NON-ADOPT) should DECREASE margin
- Negative rho (push toward ADOPT) should INCREASE margin
- The relationship should be monotonic and consistent
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import random

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.popqa import PopQADataset
from steering.hook_utils import SteeringHook
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
    """Build ReAct prompt."""
    pb = PromptBuilder()
    messages = [
        {"role": "system", "content": pb.build_system_prompt()},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def compute_margin_with_steering(model, tokenizer, question, direction, rho, layer=12, position=-1):
    """Compute margin with steering applied."""
    prompt = build_react_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Compute alpha from rho
    direction_rms = np.sqrt(np.mean(direction ** 2))
    hidden_rms = 0.65  # Approximate
    alpha = rho * (hidden_rms / direction_rms)
    
    # Forward with steering
    if abs(alpha) > 1e-6:
        with SteeringHook(model, direction, alpha, layer, position):
            with torch.no_grad():
                outputs = model(**inputs)
    else:
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
    
    return tool_logprob - finish_logprob


def main():
    parser = argparse.ArgumentParser(description="Validate Search Direction")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--direction-path", required=True, help="Path to direction NPZ")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    rng = random.Random(args.seed)

    # Load direction
    print(f"Loading direction from: {args.direction_path}")
    data = np.load(args.direction_path)
    direction = data["decision_direction"].astype(np.float32)
    direction_rms = np.sqrt(np.mean(direction ** 2))
    print(f"Direction RMS: {direction_rms:.6f}")

    # Load dataset
    dataset = PopQADataset(args.data_path)
    samples = rng.sample(dataset.samples, min(args.n_samples, len(dataset.samples)))

    # Load model
    model, tokenizer = load_model(args.model)

    # Test rho values (0.0 first to establish baseline)
    rho_values = [0.0, -0.1, -0.2, -0.3, 0.1, 0.2, 0.3]

    print(f"\n=== Validation Results (N={len(samples)}) ===")
    print(f"{'rho':>8} | {'Mean Margin':>12} | {'Std':>8} | {'Change vs 0':>12}")
    print("-" * 50)

    results = {}
    baseline_margins = None

    for rho in rho_values:
        margins = []
        for sample in tqdm(samples, desc=f"rho={rho:+.1f}", leave=False):
            m = compute_margin_with_steering(
                model, tokenizer, sample.question, direction, rho, args.layer
            )
            margins.append(m)

        mean_m = np.mean(margins)
        std_m = np.std(margins)

        if rho == 0.0:
            baseline_margins = margins
            change = 0.0
        else:
            change = mean_m - np.mean(baseline_margins)

        results[rho] = {"margins": margins, "mean": mean_m, "std": std_m, "change": change}
        print(f"{rho:>+8.2f} | {mean_m:>12.2f} | {std_m:>8.2f} | {change:>+12.2f}")

    # Check monotonicity
    print("\n=== Analysis ===")

    # Positive rho should decrease margin (push toward reject)
    # Negative rho should increase margin (push toward adopt)
    pos_rhos = [r for r in rho_values if r > 0]
    neg_rhos = [r for r in rho_values if r < 0]

    pos_changes = [results[r]["change"] for r in pos_rhos]
    neg_changes = [results[r]["change"] for r in neg_rhos]

    all_pos_decrease = all(c < 0 for c in pos_changes)
    all_neg_increase = all(c > 0 for c in neg_changes)

    print(f"Positive rho changes: {[f'{c:+.2f}' for c in pos_changes]}")
    print(f"Negative rho changes: {[f'{c:+.2f}' for c in neg_changes]}")
    print(f"✓ All positive rho decrease margin: {all_pos_decrease}")
    print(f"✓ All negative rho increase margin: {all_neg_increase}")

    # Compute slope (dm/drho)
    rho_array = np.array(rho_values)
    mean_array = np.array([results[r]["mean"] for r in rho_values])
    slope = np.polyfit(rho_array, mean_array, 1)[0]

    print(f"\nLinear slope dm/dρ: {slope:.2f}")
    print(f"Interpretation: Each unit of ρ changes margin by {slope:.2f}")

    if all_pos_decrease and all_neg_increase:
        print("\n✅ DIRECTION VALIDATION: SUCCESS")
        print("The search-specific direction correctly controls margin.")
    else:
        print("\n❌ DIRECTION VALIDATION: FAILED")
        print("The direction does not monotonically affect margin.")

    # Save results
    output_path = str(args.direction_path).replace(".npz", "_validation.json")
    with open(output_path, "w") as f:
        json.dump({
            "n_samples": len(samples),
            "rho_values": rho_values,
            "results": {str(k): {"mean": v["mean"], "std": v["std"], "change": v["change"]}
                       for k, v in results.items()},
            "slope": float(slope),
            "validation_passed": bool(all_pos_decrease and all_neg_increase),
        }, f, indent=2)
    print(f"\nSaved validation results to: {output_path}")


if __name__ == "__main__":
    main()

