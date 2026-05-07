#!/usr/bin/env python3
"""Single-observation eval runner for anti_cue_specificity.

Reads pairs.jsonl where each line is one (sample_id, target, cue) record with
a single `obs` field. Runs Qwen2.5-7B-Instruct at the post-tool decision point
and records per-record (margin, action_type, final_answer, em, parse_failure,
raw_output). Same margin / parsing conventions as run_local_answerability_eval.
"""
import argparse, json, sys, time
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
    fin_lp  = torch.logsumexp(log_probs[fin_ids],  0).item() if fin_ids  else -100.0
    return tool_lp - fin_lp


def run_one(rec, model, tokenizer, builder, device, max_new_tokens=256):
    obs = rec["obs"]
    query = f"about: {rec['question'][:80]}"
    steps = [{"action": "search", "action_input": query, "observation": obs}]
    messages = builder.build_full_prompt(rec["question"], steps)
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
    raw = tokenizer.decode(gen_ids[0, prompt_len:], skip_special_tokens=True)
    parsed = parse_action(raw)
    action2, final_answer = parsed["action"], parsed["final_answer"]
    parse_failure = (action2 is None and final_answer is None)
    if action2 and action2.lower() in ("search", "calculator"):
        action_type = "search"
    elif final_answer is not None:
        action_type = "stop"
    else:
        action_type = None

    em = None
    if final_answer is not None and rec.get("gold_answer"):
        gold = rec.get("gold_answers") or [rec["gold_answer"]]
        em = int(answer_scorer(final_answer, gold, mode="exact")["matched"])

    return {
        "sample_id": rec["sample_id"],
        "target": rec["target"], "cue": rec["cue"],
        "condition_id": rec["condition_id"],
        "margin": margin,
        "action_type": action_type, "action2": action2,
        "final_answer": final_answer, "em": em,
        "parse_failure": parse_failure,
        "raw_output": raw[:400],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/anti_cue_specificity/pairs.jsonl")
    ap.add_argument("--out",   default="results/anti_cue_specificity/eval_results.jsonl")
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.pairs)]
    if args.limit:
        records = records[:args.limit]
    print(f"[info] loaded {len(records)} records")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(f"[info] loading model {args.model_path} ({args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    builder = PromptBuilder()
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    counts = {}
    with open(out_path, "w") as f:
        for i, rec in enumerate(records):
            row = run_one(rec, model, tokenizer, builder, device,
                          max_new_tokens=args.max_new_tokens)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            k = rec["condition_id"]
            c = counts.setdefault(k, {"n": 0, "search": 0, "stop": 0, "pf": 0})
            c["n"] += 1
            c["pf"] += int(row["parse_failure"])
            if row["action_type"] == "search":
                c["search"] += 1
            elif row["action_type"] == "stop":
                c["stop"] += 1
            if (i + 1) % 25 == 0 or i + 1 == len(records):
                dt = time.time() - t0
                summ = " ".join(f"{k}:s{v['search']}/st{v['stop']}/pf{v['pf']}"
                                for k, v in sorted(counts.items()))
                print(f"  [{i+1}/{len(records)}] {dt:.1f}s  {summ}")
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
