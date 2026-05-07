#!/usr/bin/env python3
"""
SF Counterfactual Behavioral Experiment — DeepSeek-R1 variant
==============================================================
Differs from sf_counterfactual.py in two ways:

1. Model generates <think>...</think> chain before the action decision.
   Margin is measured AFTER </think>, not at the observation boundary.

2. max_new_tokens is larger (~1200) to allow full think chain + action.

Margin measurement protocol:
  a. Generate up to max_new_tokens until </think> appears (or budget exhausted).
  b. At the </think> position, run ONE more forward pass to get action logits.
  c. margin = logit(Action) - logit(Final) at that position.
  d. Continue parsing the full generation for behavioral outcome (search/stop/unknown).

Usage:
  cd tmc/scripts/e2e_agent
  python scripts/sf_counterfactual_r1.py --n 10 --smoke
  python scripts/sf_counterfactual_r1.py --n 300 --output-dir results/sf_counterfactual_r1
"""

import os, sys, re, json, argparse
import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS, parse_action
from eval.scorers import answer_scorer


MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


def compute_margin_from_logits(logits, tokenizer):
    lp = torch.log_softmax(logits.float(), dim=-1)
    tool_ids = [tokenizer.encode(t, add_special_tokens=False)[0]
                for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tokenizer.encode(t, add_special_tokens=False)[0]
                for t in ACTION_TOKENS["finish"]]
    tool_lp = torch.logsumexp(lp[tool_ids], 0).item() if tool_ids else -100.0
    fin_lp  = torch.logsumexp(lp[fin_ids],  0).item() if fin_ids  else -100.0
    return tool_lp - fin_lp


def run_one_r1(sample, condition, model, tokenizer, builder, device,
               max_new_tokens=1200):
    """
    Run one sample under one condition for R1-style reasoning model.

    Returns dict with: sample_id, condition, margin_pre (at obs boundary),
    margin_post (after </think>), think_len, action_type, final_answer, em,
    parse_failure, raw_output.
    """
    obs = sample["obs_1sf"] if condition == "1sf" else sample["obs_2sf"]
    steps = [{"action": "search", "action_input": sample["query"], "observation": obs}]
    msgs = builder.build_full_prompt(sample["question"], steps)
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    attn = torch.ones_like(input_ids)

    # ── Margin at observation boundary (pre-think) ──
    # NOTE: R1's apply_chat_template appends '<|Assistant|><think>\n' to the prompt.
    # The last token is '\n' (inside the think tag), so the model is already in
    # reasoning mode.  logit("Action") vs logit("Final") at this position does NOT
    # measure action preference — those tokens are not the natural continuation of
    # a think chain.  We compute it for completeness / auditing but it should NOT
    # be interpreted as a meaningful behavioral signal.
    with torch.no_grad():
        pre_out = model(input_ids, attention_mask=attn)
    margin_pre = compute_margin_from_logits(pre_out.logits[0, -1, :], tokenizer)
    # ⚠️ margin_pre is NOT a valid action-preference measure for R1 (see note above).

    # ── Generate full think chain + action ──
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        attention_mask=attn,
        do_sample=False,
    )
    with torch.no_grad():
        gen_ids = model.generate(input_ids, **gen_kwargs)

    gen_token_ids = gen_ids[0, prompt_len:].tolist()
    raw_output = tokenizer.decode(gen_ids[0, prompt_len:], skip_special_tokens=False)
    raw_output_clean = tokenizer.decode(gen_ids[0, prompt_len:], skip_special_tokens=True)

    # ── Find </think> position ──
    think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
    think_pos = None
    for i, tid in enumerate(gen_token_ids):
        if tid == think_end_id:
            think_pos = i
            break

    think_len = think_pos if think_pos is not None else -1

    # ── Margin after </think> ──
    margin_post = None
    if think_pos is not None:
        prefix = gen_ids[0, :prompt_len + think_pos + 1].unsqueeze(0)
        with torch.no_grad():
            post_out = model(prefix, attention_mask=torch.ones_like(prefix))
        margin_post = compute_margin_from_logits(post_out.logits[0, -1, :], tokenizer)

    # ── Parse action from generated text (strip think tags) ──
    # Use raw_output_clean (skip_special_tokens=True) to avoid <|im_end|> etc.
    # being matched by parse_action's angle-bracket placeholder filter.
    text_for_parse = re.sub(r'<think>.*?</think>', '', raw_output_clean, flags=re.DOTALL)
    text_for_parse = text_for_parse.replace('<think>', '').replace('</think>', '')
    parsed = parse_action(text_for_parse)
    action2 = parsed["action"]
    final_answer = parsed["final_answer"]

    parse_failure = (action2 is None and final_answer is None)
    if action2 and action2.lower() in ("search", "calculator"):
        action_type = "search"
    elif final_answer is not None:
        action_type = "stop"
    else:
        action_type = None  # budget exhausted or ambiguous

    # ── EM ──
    em = None
    if final_answer is not None and sample.get("answer"):
        em = int(answer_scorer(final_answer, sample["answer"], mode="exact")["matched"])

    return {
        "sample_id": sample["sample_id"],
        "condition": condition,
        "margin_pre": margin_pre,         # ⚠️ NOT a valid action-pref measure (see comment above)
        "margin_post": margin_post,       # after </think> — valid decision-point margin
        "think_len": think_len,           # -1 if </think> not found within budget
        "action_type": action_type,
        "action2": action2,
        "final_answer": final_answer,
        "em": em,
        "parse_failure": parse_failure,
        "raw_output": raw_output_clean[:400],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_ID)
    parser.add_argument("--meta-path",
                        default="results/probe_sufficiency_v2/meta.jsonl")
    parser.add_argument("--hotpotqa-path",
                        default="data/hotpotqa/hotpot_dev_distractor_v1.json")
    parser.add_argument("--output-dir", default="results/sf_counterfactual_r1")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    args = parser.parse_args()
    if args.smoke:
        args.n = 10

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    samples = []
    with open(args.meta_path) as f:
        for line in f:
            samples.append(json.loads(line))
    samples = samples[:args.n]
    print(f"Loaded {len(samples)} samples")

    with open(args.hotpotqa_path) as f:
        hpqa = json.load(f)
    id2ans = {item["_id"]: item["answer"] for item in hpqa}
    for s in samples:
        s["answer"] = id2ans.get(s["sample_id"])
    print(f"Answers matched: {sum(1 for s in samples if s['answer'])}/{len(samples)}")

    # ── Load model ──
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"\nLoading {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  model on {device}")

    builder = PromptBuilder(tools=["search"])

    # ── Run ──
    results_1sf, results_2sf = [], []
    for i, sample in enumerate(samples):
        if i % 10 == 0:
            print(f"  [{i}/{len(samples)}] {sample['sample_id']}")
        for cond, store in [("1sf", results_1sf), ("2sf", results_2sf)]:
            r = run_one_r1(sample, cond, model, tokenizer, builder, device,
                           max_new_tokens=args.max_new_tokens)
            store.append(r)

    # ── Save ──
    for cond, store in [("1sf", results_1sf), ("2sf", results_2sf)]:
        out_path = outdir / f"r1_{cond}_trajectories.jsonl"
        with open(out_path, "w") as f:
            for r in store:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {len(store)} → {out_path}")

    # ── Quick summary ──
    print("\n" + "=" * 60)
    for cond, store in [("1sf", results_1sf), ("2sf", results_2sf)]:
        n = len(store)
        n_search = sum(1 for r in store if r["action_type"] == "search")
        n_stop   = sum(1 for r in store if r["action_type"] == "stop")
        n_budget = sum(1 for r in store if r["action_type"] is None)
        margins_pre  = [r["margin_pre"] for r in store]
        margins_post = [r["margin_post"] for r in store if r["margin_post"] is not None]
        think_lens   = [r["think_len"] for r in store if r["think_len"] >= 0]
        em_vals = [r["em"] for r in store if r["em"] is not None]
        print(f"\n  {cond.upper()} (N={n}):")
        print(f"    2ndSR:         {n_search}/{n} = {n_search/n*100:.1f}%")
        print(f"    Stop:          {n_stop}/{n} = {n_stop/n*100:.1f}%")
        print(f"    Budget out:    {n_budget}/{n} = {n_budget/n*100:.1f}%")
        print(f"    Think len:     mean={np.mean(think_lens):.0f} ± {np.std(think_lens):.0f} tokens"
              if think_lens else "    Think len: N/A")
        print(f"    Margin pre:    {np.mean(margins_pre):.3f} ± {np.std(margins_pre):.3f}")
        if margins_post:
            print(f"    Margin post:   {np.mean(margins_post):.3f} ± {np.std(margins_post):.3f}")
        if em_vals:
            print(f"    EM (if stop):  {np.mean(em_vals)*100:.1f}% (N={len(em_vals)})")

    print("\nDone.")


if __name__ == "__main__":
    main()
