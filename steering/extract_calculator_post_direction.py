#!/usr/bin/env python3
"""Extract calculator-domain post-tool direction via paired observations.

This is the calculator analogue of ``extract_search_post_direction.py``:
- same question in both conditions
- same prior ``Action: calculator`` and ``Action Input``
- only the ``Observation`` changes (correct vs incorrect result)

The extracted ``decision_direction`` points toward NON-ADOPT, i.e. toward
continuing with another tool call instead of trusting the current calculator
observation.
"""

import argparse
import json
import random
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.prompts import PromptBuilder, ACTION_TOKENS
from datasets.gsm8k import GSM8KDataset
from steering.hook_utils import compute_rms, get_model_layers


def load_model(model_id: str, use_4bit: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_id}")
    kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16, "trust_remote_code": True}
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            print("  Using 4-bit quantization")
        except ImportError:
            print("  bitsandbytes not available, using bf16")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tok


def normalize_numeric_string(value: str) -> str:
    cleaned = str(value).strip().replace(",", "")
    if not cleaned:
        raise ValueError("Empty numeric answer")
    num = Decimal(cleaned)
    if num == num.to_integral():
        return str(int(num))
    normalized = format(num.normalize(), "f")
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


def build_calculator_post_pair(answer_text: str) -> Tuple[str, str, str]:
    """Return (expression, correct_observation, incorrect_observation)."""
    correct_observation = normalize_numeric_string(answer_text)
    value = Decimal(correct_observation)

    if value == value.to_integral():
        magnitude = abs(int(value))
        step = max(1, int(round(magnitude * 0.05))) if magnitude >= 20 else 1
        incorrect_value = value + Decimal(step)
    else:
        decimal_places = max(1, -value.as_tuple().exponent)
        step = Decimal(1).scaleb(-decimal_places)
        incorrect_value = value + step

    if incorrect_value == value:
        incorrect_value = value + Decimal(1)

    incorrect_observation = normalize_numeric_string(str(incorrect_value))
    if incorrect_observation == correct_observation:
        raise ValueError("Failed to create distinct incorrect calculator observation")

    calculator_expression = correct_observation
    return calculator_expression, correct_observation, incorrect_observation


def build_step1_prompt(
    tokenizer,
    question: str,
    calculator_expression: str,
    observation: str,
    max_obs_chars: int = 200,
) -> str:
    pb = PromptBuilder(tools=["calculator"])
    steps = [{
        "action": "calculator",
        "action_input": calculator_expression,
        "observation": observation[:max_obs_chars],
    }]
    messages = pb.build_full_prompt(question, steps)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def compute_margin_and_hidden(model, tokenizer, prompt: str, layer: int = 12) -> Tuple[float, np.ndarray]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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

    hidden = captured["hidden"][0, -1, :].float().cpu().numpy().astype(np.float32)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    action_ids = []
    final_ids = []
    for token_str in ACTION_TOKENS["tool_call"]:
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        if ids:
            action_ids.append(ids[0])
    for token_str in ACTION_TOKENS["finish"]:
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        if ids:
            final_ids.append(ids[0])

    action_lp = torch.logsumexp(log_probs[action_ids], dim=0).item() if action_ids else -100.0
    final_lp = torch.logsumexp(log_probs[final_ids], dim=0).item() if final_ids else -100.0
    return float(action_lp - final_lp), hidden


def select_eligible_samples(dataset: GSM8KDataset, n_pairs: int, seed: int):
    eligible = []
    skipped = []
    for sample in dataset.samples:
        try:
            build_calculator_post_pair(sample.answer)
            eligible.append(sample)
        except (InvalidOperation, ValueError):
            skipped.append(sample.id)
    rng = random.Random(seed)
    if n_pairs is None or n_pairs < 0 or n_pairs >= len(eligible):
        selected = list(eligible)
    else:
        selected = rng.sample(eligible, n_pairs)
    return selected, eligible, skipped


def main():
    parser = argparse.ArgumentParser(description="Extract calculator-domain post-tool direction via paired observations")
    parser.add_argument("--output", required=True, help="Output NPZ path")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data-path", default=None, help="Optional local GSM8K JSONL")
    parser.add_argument("--split", default="train", choices=["train", "test"],
                        help="GSM8K split (train default to avoid test leakage)")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--n-pairs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-obs-chars", type=int, default=200)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    dataset = GSM8KDataset(args.data_path, split=args.split)
    samples, eligible_samples, skipped_ids = select_eligible_samples(dataset, args.n_pairs, args.seed)
    if len(samples) < 2:
        raise ValueError(
            f"Need at least 2 eligible GSM8K samples, got {len(samples)} "
            f"(eligible_pool={len(eligible_samples)})"
        )

    print(
        f"Selected {len(samples)} calculator post-tool pairs "
        f"(split={args.split}, eligible_pool={len(eligible_samples)}, skipped={len(skipped_ids)})"
    )

    model, tok = load_model(args.model, use_4bit=not args.no_4bit)

    correct_hiddens = []
    incorrect_hiddens = []
    pair_info = []

    for sample in tqdm(samples, desc="Extracting calculator post-tool pairs"):
        expression, obs_correct, obs_incorrect = build_calculator_post_pair(sample.answer)
        prompt_correct = build_step1_prompt(
            tok, sample.question, expression, obs_correct, max_obs_chars=args.max_obs_chars
        )
        prompt_incorrect = build_step1_prompt(
            tok, sample.question, expression, obs_incorrect, max_obs_chars=args.max_obs_chars
        )

        m_correct, h_correct = compute_margin_and_hidden(model, tok, prompt_correct, layer=args.layer)
        m_incorrect, h_incorrect = compute_margin_and_hidden(model, tok, prompt_incorrect, layer=args.layer)

        correct_hiddens.append(h_correct)
        incorrect_hiddens.append(h_incorrect)
        pair_info.append({
            "question_id": sample.id,
            "question": sample.question,
            "gold_answer": sample.answer,
            "calculator_expression": expression,
            "observation_correct": obs_correct,
            "observation_incorrect": obs_incorrect,
            "margin_correct": round(m_correct, 4),
            "margin_incorrect": round(m_incorrect, 4),
            "margin_diff": round(m_incorrect - m_correct, 4),
        })

    correct_mean = np.mean(correct_hiddens, axis=0)
    incorrect_mean = np.mean(incorrect_hiddens, axis=0)
    adopt_direction = correct_mean - incorrect_mean
    decision_direction = incorrect_mean - correct_mean

    direction_norm = float(np.linalg.norm(adopt_direction))
    direction_rms = compute_rms(adopt_direction)

    margin_correct = [row["margin_correct"] for row in pair_info]
    margin_incorrect = [row["margin_incorrect"] for row in pair_info]
    margin_diffs = [row["margin_diff"] for row in pair_info]
    n_correct_sign = sum(1 for diff in margin_diffs if diff > 0)

    print(f"\n{'=' * 60}")
    print("=== Calculator Post-tool Direction Extracted ===")
    print(f"{'=' * 60}")
    print(f"N pairs:                {len(samples)}")
    print(f"Layer:                  {args.layer}")
    print(f"Direction norm:         {direction_norm:.4f}")
    print(f"Direction RMS:          {direction_rms:.6f}")
    print("Margin diagnostics:")
    print(f"  Correct mean:         {np.mean(margin_correct):.4f}")
    print(f"  Incorrect mean:       {np.mean(margin_incorrect):.4f}")
    print(f"  Diff (inc-cor) mean:  {np.mean(margin_diffs):.4f}")
    print(f"  Correct sign:         {n_correct_sign}/{len(samples)} ({100*n_correct_sign/len(samples):.1f}%)")

    for ref_name, ref_path in [
        ("calculator_v1", "steering/directions/direction_calculator_v1.npz"),
        ("v12_post", "steering/directions/direction_v12_post_scaled.npz"),
    ]:
        ref_full = Path(__file__).resolve().parent.parent / ref_path
        if ref_full.exists():
            ref_data = np.load(ref_full)
            ref_direction = ref_data["decision_direction"].astype(np.float64)
            cos = float(np.dot(decision_direction.flatten(), ref_direction.flatten()) /
                        (np.linalg.norm(decision_direction) * np.linalg.norm(ref_direction) + 1e-10))
            print(f"  cosine(this, {ref_name}): {cos:.6f}")

    np.random.seed(args.seed)
    random_direction = np.random.randn(decision_direction.shape[0]).astype(np.float32)
    random_direction = random_direction / (np.linalg.norm(random_direction) + 1e-12) * direction_norm

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        decision_direction=decision_direction.astype(np.float32),
        adopt_direction=adopt_direction.astype(np.float32),
        random_direction=random_direction.astype(np.float32),
        layer=args.layer,
        n_pairs=len(samples),
        seed=args.seed,
        split=args.split,
        method="paired_observation",
        domain="calculator",
        context="post_tool",
    )

    info_path = str(out).replace(".npz", "_pair_info.jsonl")
    with open(info_path, "w", encoding="utf-8") as f:
        for info in pair_info:
            f.write(json.dumps(info, ensure_ascii=False) + "\n")

    summary = {
        "n_pairs": len(samples),
        "layer": args.layer,
        "seed": args.seed,
        "split": args.split,
        "eligible_pool_size": len(eligible_samples),
        "skipped_non_numeric_count": len(skipped_ids),
        "direction_norm": direction_norm,
        "direction_rms": direction_rms,
        "method": "paired_observation",
        "domain": "calculator",
        "context": "post_tool",
        "margin_correct_mean": float(np.mean(margin_correct)),
        "margin_incorrect_mean": float(np.mean(margin_incorrect)),
        "margin_diff_mean": float(np.mean(margin_diffs)),
        "correct_sign_fraction": n_correct_sign / len(samples),
        "description": (
            "d = h_incorrect - h_correct (decision_direction, toward NON-ADOPT). "
            "Paired observation method: same GSM8K question, same calculator action, "
            "only observation changes from correct to incorrect numeric result."
        ),
    }
    summary_path = str(out).replace(".npz", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDirection saved to: {out}")
    print(f"Pair info saved to: {info_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()