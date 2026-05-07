#!/usr/bin/env python3
"""Re-extract per-sample L20 hidden states + margins for the action direction.

Mirrors steering/extract_search_direction_v2.py exactly (same seed, n_samples=200,
same prompt, same hook semantics) but saves the per-sample matrix instead of just
the (h_low_mean, h_high_mean) pair, so that bootstrap resampling at the
sample level becomes possible.

Sanity check: after extraction, recompute h_low_mean - h_high_mean using the
20/80 percentile thresholds; the resulting direction must match
direction_search_v3_layer20.npz['decision_direction'] up to float16 precision.

Usage:
    cd tmc/scripts/e2e_agent
    python scripts/extract_action_persample_l20.py \
        --data data/popqa/popqa_test.jsonl \
        --out  results/cos_evidence_action_ci/popqa_l20_persample.npz
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from datasets.popqa import PopQADataset
from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import get_model_layers


def build_react_prompt(tokenizer, question):
    pb = PromptBuilder()
    messages = [
        {"role": "system", "content": pb.build_system_prompt()},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def compute_margin_and_hidden(model, tokenizer, question, layer, position=-1):
    prompt = build_react_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    layers = get_model_layers(model)
    captured = {}
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h
    handle = layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            outputs = model(**inputs)
    finally:
        handle.remove()
    hs = captured["h"]
    seq_len = hs.shape[1]
    pos = (seq_len + position) if position < 0 else min(position, seq_len - 1)
    pos = max(0, min(int(pos), seq_len - 1))
    hidden = hs[0, pos, :].float().cpu().numpy().astype(np.float32)

    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    tool_tokens, finish_tokens = [], []
    for s in ACTION_TOKENS["tool_call"]:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids: tool_tokens.append(ids[0])
    for s in ACTION_TOKENS["finish"]:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids: finish_tokens.append(ids[0])
    tool_lp   = torch.logsumexp(log_probs[tool_tokens], dim=0).item() if tool_tokens else -100.0
    finish_lp = torch.logsumexp(log_probs[finish_tokens], dim=0).item() if finish_tokens else -100.0
    return tool_lp - finish_lp, hidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/popqa/popqa_test.jsonl")
    ap.add_argument("--out",  default="results/cos_evidence_action_ci/popqa_l20_persample.npz")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--position", type=int, default=-1)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ds = PopQADataset(args.data)
    samples = rng.sample(ds.samples, min(args.n_samples, len(ds.samples)))
    print(f"[info] {len(samples)} PopQA samples; layer L{args.layer}; seed={args.seed}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()

    ids, margins, hidden = [], [], []
    for s in tqdm(samples, desc="extract"):
        m, h = compute_margin_and_hidden(model, tok, s.question, args.layer, args.position)
        ids.append(s.id); margins.append(m); hidden.append(h)

    ids = np.array(ids)
    margins = np.array(margins, dtype=np.float64)
    H = np.stack(hidden, axis=0).astype(np.float32)

    p20 = float(np.percentile(margins, 20))
    p80 = float(np.percentile(margins, 80))
    low_mask  = margins <= p20
    high_mask = margins >= p80
    h_low_mean  = H[low_mask ].mean(axis=0)
    h_high_mean = H[high_mask].mean(axis=0)
    md = h_low_mean - h_high_mean

    ref = np.load("steering/directions/direction_search_v3_layer20.npz")
    md_ref = ref["decision_direction"].astype(np.float32)
    cos_check = float(np.dot(md / np.linalg.norm(md), md_ref / np.linalg.norm(md_ref)))
    print(f"\n[sanity] reproduced direction vs stored: cos = {cos_check:+.6f}")
    print(f"         n_low = {int(low_mask.sum())} (stored 40)  n_high = {int(high_mask.sum())} (stored 44)")
    print(f"         p20 = {p20:.4f} (stored 6.974)  p80 = {p80:.4f} (stored 10.500)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, ids=ids, margins=margins, hidden=H,
             layer=args.layer, position=args.position, seed=args.seed,
             p20=p20, p80=p80, sanity_cos=cos_check)
    print(f"[wrote] {args.out}  shape={H.shape}")


if __name__ == "__main__":
    main()
