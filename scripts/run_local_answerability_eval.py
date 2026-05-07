#!/usr/bin/env python3
"""
Evaluate paired Low-L / High-L observations from build_local_answerability_pairs.py.

For each pair, runs both conditions through Qwen2.5-7B-Instruct at the post-tool
decision point and records (margin, action_type, final_answer, em, parse_failure,
raw_output). Writes per-row JSONL.

Paired analysis (McNemar on 2ndSR, Wilcoxon on margin) is done in a separate
analyze script.
"""
import argparse, json, sys, time, re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS, parse_action
from eval.scorers import answer_scorer


def compute_margin(logits, tokenizer):
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    tool_ids = [tokenizer.encode(t, add_special_tokens=False)[0]
                for t in ACTION_TOKENS["tool_call"] if tokenizer.encode(t, add_special_tokens=False)]
    fin_ids = [tokenizer.encode(t, add_special_tokens=False)[0]
               for t in ACTION_TOKENS["finish"] if tokenizer.encode(t, add_special_tokens=False)]
    tool_lp = torch.logsumexp(log_probs[tool_ids], 0).item() if tool_ids else -100.0
    fin_lp = torch.logsumexp(log_probs[fin_ids], 0).item() if fin_ids else -100.0
    return tool_lp - fin_lp


def run_one(pair, condition, model, tokenizer, builder, device, max_new_tokens=256):
    obs = pair["obs_low"] if condition == "low" else pair["obs_high"]
    query = f"about: {pair['question'][:80]}"
    steps = [{"action": "search", "action_input": query, "observation": obs}]
    messages = builder.build_full_prompt(pair["question"], steps)
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt_str, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    attn_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model(input_ids, attention_mask=attn_mask)
    margin = compute_margin(out.logits[0, -1, :], tokenizer)

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids, attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            do_sample=False,
        )
    completion_ids = gen_ids[0, prompt_len:]
    raw_output = tokenizer.decode(completion_ids, skip_special_tokens=True)

    parsed = parse_action(raw_output)
    action2 = parsed["action"]
    final_answer = parsed["final_answer"]
    parse_failure = (action2 is None and final_answer is None)
    if action2 and action2.lower() in ("search", "calculator"):
        action_type = "search"
    elif final_answer is not None:
        action_type = "stop"
    else:
        action_type = None

    em = None
    if final_answer is not None and pair.get("gold_answer"):
        gold = pair.get("gold_answers") or [pair["gold_answer"]]
        em = int(answer_scorer(final_answer, gold, mode="exact")["matched"])

    return {
        "sample_id": pair["sample_id"],
        "condition": condition,
        "margin": margin,
        "action_type": action_type,
        "action2": action2,
        "final_answer": final_answer,
        "em": em,
        "parse_failure": parse_failure,
        "raw_output": raw_output[:400],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/local_answerability/pairs.jsonl")
    ap.add_argument("--out", default="results/local_answerability/eval_results.jsonl")
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    pairs = [json.loads(l) for l in open(args.pairs)]
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"[info] loaded {len(pairs)} pairs")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(f"[info] loading model {args.model_path} ({args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    builder = PromptBuilder()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_pf = {"low": 0, "high": 0}
    n_search = {"low": 0, "high": 0}
    with open(out_path, "w") as f:
        for i, pair in enumerate(pairs):
            for cond in ("low", "high"):
                row = run_one(pair, cond, model, tokenizer, builder, device,
                              max_new_tokens=args.max_new_tokens)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                n_pf[cond] += int(row["parse_failure"])
                n_search[cond] += int(row["action_type"] == "search")
            if (i + 1) % 5 == 0 or i + 1 == len(pairs):
                dt = time.time() - t0
                print(f"  [{i+1}/{len(pairs)}] {dt:.1f}s "
                      f"low_search={n_search['low']} high_search={n_search['high']} "
                      f"low_pf={n_pf['low']} high_pf={n_pf['high']}")
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
