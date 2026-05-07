#!/usr/bin/env python3
"""Validate calculator direction on GSM8K: does rho monotonically change margin?

We validate the *sign convention*:
- Direction files in this repo are intended to be NON-ADOPT directions.
- Therefore, positive rho (=> positive alpha) should DECREASE margin
  (less likely to produce "Action" vs "Final").

Prompt MUST match GSM8K/MATH evaluation: tools=["calculator"].
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.gsm8k import GSM8KDataset
from steering.hook_utils import SteeringHook, get_model_layers
from agent.prompts import PromptBuilder, ACTION_TOKENS


def load_model(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def calibrate_hidden_rms(model, tokenizer, layer: int) -> float:
    """Match ReActAgent._calibrate_hidden_rms: last-token RMS on a short user-only prompt."""
    calibration_text = "The quick brown fox jumps over the lazy dog."
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": calibration_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(next(model.parameters()).device)

    layers = get_model_layers(model)
    num_layers = len(layers)
    actual_layer = layer if layer >= 0 else num_layers + layer
    captured = {}

    def hook_fn(_module, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        captured["hidden"] = hidden
        return out

    handle = layers[actual_layer].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        handle.remove()

    h_last = captured["hidden"][0, -1, :].float()
    rms = float(h_last.pow(2).mean().sqrt().item())
    return rms


def build_prompt(tokenizer, question: str) -> str:
    pb = PromptBuilder(tools=["calculator"])
    msgs = [
        {"role": "system", "content": pb.build_system_prompt()},
        {"role": "user", "content": pb.build_user_prompt(question)},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def compute_margin(model, tokenizer, question: str, direction: np.ndarray, alpha: float, layer: int, position: int) -> float:
    prompt = build_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

    if abs(alpha) > 1e-9:
        with SteeringHook(model, direction, alpha, layer=layer, position=position):
            with torch.no_grad():
                outputs = model(**inputs)
    else:
        with torch.no_grad():
            outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    tool_ids = []
    fin_ids = []
    for s in ACTION_TOKENS["tool_call"]:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            tool_ids.append(ids[0])
    for s in ACTION_TOKENS["finish"]:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            fin_ids.append(ids[0])

    tool_lp = torch.logsumexp(log_probs[tool_ids], dim=0).item() if tool_ids else -100.0
    fin_lp = torch.logsumexp(log_probs[fin_ids], dim=0).item() if fin_ids else -100.0
    return float(tool_lp - fin_lp)


def main():
    p = argparse.ArgumentParser(description="Validate calculator direction on GSM8K")
    p.add_argument("--direction-path", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--data-path", default=None)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--n-samples", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--position", type=int, default=-1)
    p.add_argument("--rho-values", default="0,-0.1,-0.25,-0.5,0.1,0.25,0.5")
    p.add_argument("--alpha-max", type=float, default=2000.0)
    args = p.parse_args()

    rho_values = [float(x.strip()) for x in args.rho_values.split(",") if x.strip()]

    data = np.load(args.direction_path, allow_pickle=True)
    direction = data["decision_direction"].astype(np.float32)
    direction_rms = float(np.sqrt(np.mean(direction ** 2)))
    print(f"Direction RMS: {direction_rms:.6f}")

    ds = GSM8KDataset(args.data_path, split=args.split)
    rng = random.Random(args.seed)
    samples = rng.sample(ds.samples, min(args.n_samples, len(ds.samples)))
    print(f"Validation samples: {len(samples)} (split={args.split})")

    model, tok = load_model(args.model)
    hidden_rms = calibrate_hidden_rms(model, tok, args.layer)
    print(f"Calibrated hidden_rms(layer={args.layer}) = {hidden_rms:.6f}")

    def rho_to_alpha(rho: float) -> float:
        alpha = rho * (hidden_rms / (direction_rms + 1e-12))
        return float(np.clip(alpha, -args.alpha_max, args.alpha_max))

    print("\n=== Validation Results ===")
    print(f"{'rho':>8} | {'alpha':>10} | {'mean_m':>10} | {'std_m':>8} | {'Δvs0':>8}")
    print("-" * 60)

    baseline_mean = None
    results = {}
    for rho in rho_values:
        alpha = rho_to_alpha(rho)
        ms = []
        for s in tqdm(samples, desc=f"rho={rho:+.2f}", leave=False):
            ms.append(compute_margin(model, tok, s.question, direction, alpha, args.layer, args.position))
        mean_m = float(np.mean(ms))
        std_m = float(np.std(ms))
        if rho == 0.0:
            baseline_mean = mean_m
        delta = float(mean_m - (baseline_mean if baseline_mean is not None else mean_m))
        results[str(rho)] = {"alpha": alpha, "mean": mean_m, "std": std_m, "delta": delta}
        print(f"{rho:>+8.2f} | {alpha:>10.2f} | {mean_m:>10.3f} | {std_m:>8.3f} | {delta:>+8.3f}")

    pos = [float(r) for r in rho_values if r > 0]
    neg = [float(r) for r in rho_values if r < 0]
    pos_ok = all(results[str(r)]["delta"] < 0 for r in pos)
    neg_ok = all(results[str(r)]["delta"] > 0 for r in neg)

    print("\n=== Sign Check ===")
    print(f"Positive rho decreases margin: {pos_ok}")
    print(f"Negative rho increases margin: {neg_ok}")
    passed = bool(pos_ok and neg_ok)
    print(f"PASSED: {passed}")

    out_json = str(args.direction_path).replace(".npz", "_gsm8k_validation.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "direction_path": args.direction_path,
                "split": args.split,
                "n_samples": len(samples),
                "layer": args.layer,
                "position": args.position,
                "hidden_rms": hidden_rms,
                "direction_rms": direction_rms,
                "rho_values": rho_values,
                "results": results,
                "passed": passed,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
