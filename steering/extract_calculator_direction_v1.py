#!/usr/bin/env python3
"""Extract calculator-specific tool-call direction (GSM8K) for E2E ReAct agent.

Best-practice notes:
- This extracts a direction at the *tool_call vs finish* decision token ("Action" vs "Final").
- Prompt MUST match evaluation. GSM8K/MATH muscle uses tools={"calculator": ...},
  so the system prompt lists ONLY the calculator tool.
- To avoid test leakage, default split is GSM8K *train* (evaluation uses test).

Method (boundary mining, analogous to extract_search_direction_v2.py):
1) Sample N GSM8K questions, compute margin m = logP(Action) - logP(Final)
2) Select low-margin group (<= P20) and high-margin group (>= P80)
3) Direction = mean(h_low) - mean(h_high), which points toward NON-ADOPT
   (so +alpha along this direction should reduce margin, i.e., discourage tool-call)
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.gsm8k import GSM8KDataset
from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import get_model_layers


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


def build_react_prompt(tokenizer, question: str, tools: List[str]) -> str:
    pb = PromptBuilder(tools=tools)
    messages = [
        {"role": "system", "content": pb.build_system_prompt()},
        {"role": "user", "content": pb.build_user_prompt(question)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def compute_margin_and_hidden(
    model,
    tokenizer,
    question: str,
    tools: List[str],
    layer: int = 12,
    position: int = -1,
) -> Tuple[float, np.ndarray]:
    """Single forward pass to get decision margin and hidden at intervention site.

    Best-practice: capture the hidden state via a forward hook on the transformer block
    (same layer semantics as SteeringHook / get_model_layers), rather than relying on
    `outputs.hidden_states[...]` indexing conventions.
    """
    prompt = build_react_prompt(tokenizer, question, tools)
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

    hs = captured["hidden"]
    seq_len = hs.shape[1]
    pos = (seq_len + position) if position < 0 else min(position, seq_len - 1)
    pos = max(0, min(int(pos), seq_len - 1))
    hidden = hs[0, pos, :].float().cpu().numpy().astype(np.float32)

    # Decision margin at next token: log P(Action) - log P(Final)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

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

    tool_lp = torch.logsumexp(log_probs[tool_tokens], dim=0).item() if tool_tokens else -100.0
    fin_lp = torch.logsumexp(log_probs[finish_tokens], dim=0).item() if finish_tokens else -100.0
    margin = float(tool_lp - fin_lp)
    return margin, hidden


def compute_rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr ** 2)))


def main():
    p = argparse.ArgumentParser(description="Extract calculator tool-call direction (GSM8K)")
    p.add_argument("--output", required=True, help="Output NPZ path")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--data-path", default=None, help="Optional local GSM8K JSONL")
    p.add_argument("--split", default="train", choices=["train", "test"], help="HF split when --data-path is not provided")
    p.add_argument("--tools", default="calculator", help='Comma-separated tools list for system prompt (default: "calculator")')
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--position", type=int, default=-1)
    p.add_argument("--n-samples", type=int, default=400)
    p.add_argument("--n-boundary", type=int, default=60, help="Samples per boundary group")
    p.add_argument("--low-percentile", type=float, default=20.0)
    p.add_argument("--high-percentile", type=float, default=80.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    if not tools:
        raise ValueError("--tools cannot be empty")

    rng = random.Random(args.seed)

    ds = GSM8KDataset(args.data_path, split=args.split)
    samples = rng.sample(ds.samples, min(args.n_samples, len(ds.samples)))
    print(f"Analyzing {len(samples)} GSM8K samples (split={args.split}, tools={tools})")

    model, tok = load_model(args.model)

    all_data = []
    for s in tqdm(samples, desc="Computing margins+hidden"):
        m, h = compute_margin_and_hidden(model, tok, s.question, tools, args.layer, args.position)
        all_data.append({"id": s.id, "margin": m, "hidden": h})

    margins = np.array([d["margin"] for d in all_data], dtype=np.float32)
    p_lo = float(np.percentile(margins, args.low_percentile))
    p_hi = float(np.percentile(margins, args.high_percentile))
    print("\n=== Margin Distribution ===")
    print(f"  mean={float(margins.mean()):.3f} std={float(margins.std()):.3f} min={float(margins.min()):.3f} max={float(margins.max()):.3f}")
    print(f"  low_threshold(P{args.low_percentile:g})={p_lo:.3f} high_threshold(P{args.high_percentile:g})={p_hi:.3f}")

    low = [d for d in all_data if d["margin"] <= p_lo]
    high = [d for d in all_data if d["margin"] >= p_hi]
    if len(low) > args.n_boundary:
        low = rng.sample(low, args.n_boundary)
    if len(high) > args.n_boundary:
        high = rng.sample(high, args.n_boundary)
    print(f"Low-margin group:  {len(low)}")
    print(f"High-margin group: {len(high)}")

    h_low = np.stack([d["hidden"] for d in low])
    h_high = np.stack([d["hidden"] for d in high])
    h_low_mean = h_low.mean(axis=0)
    h_high_mean = h_high.mean(axis=0)

    # NON-ADOPT direction (compatible with +alpha pushing toward reject)
    decision_direction = (h_low_mean - h_high_mean).astype(np.float32)
    direction_norm = float(np.linalg.norm(decision_direction))
    if direction_norm < 1e-12:
        raise ValueError("Extracted near-zero direction norm; increase n-samples or check prompt consistency")
    decision_direction_normalized = (decision_direction / direction_norm).astype(np.float32)
    direction_rms = compute_rms(decision_direction)

    # Random direction (same norm) for control
    np.random.seed(args.seed)
    random_direction = np.random.randn(decision_direction.shape[0]).astype(np.float32)
    random_direction = random_direction / (np.linalg.norm(random_direction) + 1e-12) * direction_norm

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        decision_direction=decision_direction.astype(np.float16),
        decision_direction_normalized=decision_direction_normalized.astype(np.float16),
        random_direction=random_direction.astype(np.float16),
        h_low_margin_mean=h_low_mean.astype(np.float16),
        h_high_margin_mean=h_high_mean.astype(np.float16),
        direction_norm=np.float16(direction_norm),
        direction_rms=np.float16(direction_rms),
        layer=args.layer,
        position=args.position,
        tools=np.array(tools),
        split=args.split,
        n_samples_analyzed=len(samples),
        n_low_margin=len(low),
        n_high_margin=len(high),
        low_margin_threshold=np.float16(p_lo),
        high_margin_threshold=np.float16(p_hi),
        all_margins=margins.astype(np.float16),
    )

    meta_path = str(out).replace(".npz", "_meta.json")
    meta = {
        "model": args.model,
        "layer": args.layer,
        "position": args.position,
        "tools": tools,
        "split": args.split,
        "seed": args.seed,
        "n_samples_analyzed": len(samples),
        "n_low_margin": len(low),
        "n_high_margin": len(high),
        "low_percentile": args.low_percentile,
        "high_percentile": args.high_percentile,
        "low_margin_threshold": p_lo,
        "high_margin_threshold": p_hi,
        "direction_norm": direction_norm,
        "direction_rms": direction_rms,
        "margin_stats": {
            "mean": float(margins.mean()),
            "std": float(margins.std()),
            "min": float(margins.min()),
            "max": float(margins.max()),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n=== Direction Results ===")
    print(f"Saved: {out}")
    print(f"Meta:  {meta_path}")
    print(f"Direction norm={direction_norm:.4f} rms={direction_rms:.6f}")
    print(f"Low-mean margin={float(np.mean([d['margin'] for d in low])):.3f}")
    print(f"High-mean margin={float(np.mean([d['margin'] for d in high])):.3f}")


if __name__ == "__main__":
    main()
