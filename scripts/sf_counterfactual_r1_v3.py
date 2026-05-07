#!/usr/bin/env python3
"""
SF Counterfactual Behavioral Experiment — DeepSeek-R1 v3 (margin position fix)
================================================================================
Fixes two bugs in v2:

  BUG 1: margin_post was measured at end of </think> token, where top-1 is
         always '\n\n' and Action/Final are both ~1e-12.  True decision is at
         </think>\n\n + 1 (pos B).

  BUG 2: action_type used parse_action on the FULL generation (which can
         include a hallucinated observation + Final Answer after an initial
         Action: search).  True first-decision should be read from the token
         actually emitted at pos B.

v3 protocol:
  - Single generate() with output_scores=True captures all step logits.
  - margin_A  = logp(Action) - logp(Final) at scores[think_pos+1]  (predicts \n\n) — v2-equivalent
  - margin_B  = same at scores[think_pos+2]  (predicts Action/Final) — TRUE decision
  - first_action_token  = decoded gen_token_ids[think_pos+2]  (what model actually emitted)
  - first_action_top1   = argmax of scores[think_pos+2]  (should equal above under greedy)
  - action_type_v2      = v2-style parse_action classification (Final-priority)
  - action_type_first   = classification from first_action_token only
"""

import os, sys, re, json, argparse
import numpy as np
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS, parse_action
from eval.scorers import answer_scorer

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


def margin_from_logits(logits, tool_ids, fin_ids):
    lp = torch.log_softmax(logits.float(), dim=-1)
    tool_lp = torch.logsumexp(lp[tool_ids], 0).item()
    fin_lp  = torch.logsumexp(lp[fin_ids],  0).item()
    return tool_lp - fin_lp, tool_lp, fin_lp


def run_one(sample, condition, model, tokenizer, builder, device,
            tool_ids, fin_ids, max_new_tokens=1200):
    obs = sample["obs_1sf"] if condition == "1sf" else sample["obs_2sf"]
    steps = [{"action": "search", "action_input": sample["query"], "observation": obs}]
    msgs = builder.build_full_prompt(sample["question"], steps)
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    attn = torch.ones_like(input_ids)

    with torch.no_grad():
        gen = model.generate(
            input_ids, attention_mask=attn, max_new_tokens=max_new_tokens,
            do_sample=False, return_dict_in_generate=True, output_scores=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    gen_ids = gen.sequences
    gen_token_ids = gen_ids[0, prompt_len:].tolist()
    raw_output = tokenizer.decode(gen_ids[0, prompt_len:], skip_special_tokens=True)

    think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
    think_pos = next((i for i, t in enumerate(gen_token_ids) if t == think_end_id), None)

    margin_A = margin_B = margin_C = None
    first_action_token = first_action_top1 = None
    top1_lp_A = top1_lp_B = None
    scores = gen.scores  # tuple of len == len(gen_token_ids); scores[i] produced gen_token_ids[i]
    if think_pos is not None and think_pos + 2 < len(scores):
        mA, _, _ = margin_from_logits(scores[think_pos + 1][0], tool_ids, fin_ids)
        mB, _, _ = margin_from_logits(scores[think_pos + 2][0], tool_ids, fin_ids)
        margin_A, margin_B = mA, mB
        if think_pos + 3 < len(scores):
            mC, _, _ = margin_from_logits(scores[think_pos + 3][0], tool_ids, fin_ids)
            margin_C = mC
        top1_A_id = torch.argmax(scores[think_pos + 1][0]).item()
        top1_B_id = torch.argmax(scores[think_pos + 2][0]).item()
        top1_lp_A = torch.log_softmax(scores[think_pos + 1][0].float(), dim=-1)[top1_A_id].item()
        top1_lp_B = torch.log_softmax(scores[think_pos + 2][0].float(), dim=-1)[top1_B_id].item()
        first_action_top1 = tokenizer.decode([top1_B_id])
        first_action_token = tokenizer.decode([gen_token_ids[think_pos + 2]])

    # Classification from first_action_token (pos-B view)
    if first_action_token is None:
        action_type_first = None
    else:
        tok = first_action_token.strip().lower()
        if tok.startswith("action") or tok == "action":
            action_type_first = "search"
        elif tok.startswith("final") or tok == "final":
            action_type_first = "stop"
        else:
            action_type_first = "other"

    # v2-style action_type (full-gen parse, final-priority)
    text_for_parse = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL)
    text_for_parse = text_for_parse.replace('<think>', '').replace('</think>', '')
    parsed = parse_action(text_for_parse)
    action2 = parsed["action"]; final_answer = parsed["final_answer"]
    if action2 and action2.lower() in ("search", "calculator"):
        action_type_v2 = "search"
    elif final_answer is not None:
        action_type_v2 = "stop"
    else:
        action_type_v2 = None

    em = int(answer_scorer(final_answer, sample["answer"], mode="exact")["matched"]) \
         if (final_answer is not None and sample.get("answer")) else None

    return {
        "sample_id": sample["sample_id"], "condition": condition,
        "think_len": think_pos if think_pos is not None else -1,
        "margin_A_v2eq": margin_A, "margin_B_true": margin_B, "margin_C": margin_C,
        "top1_A_tok": None if top1_lp_A is None else tokenizer.decode([torch.argmax(scores[think_pos+1][0]).item()]),
        "top1_A_lp": top1_lp_A, "top1_B_lp": top1_lp_B,
        "first_action_token": first_action_token, "first_action_top1": first_action_top1,
        "action_type_first": action_type_first,
        "action_type_v2": action_type_v2, "action2": action2, "final_answer": final_answer,
        "em": em, "raw_output": raw_output[:2000],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-path", default="results/probe_sufficiency_v2/meta.jsonl")
    ap.add_argument("--hotpotqa-path", default="data/hotpotqa/hotpot_dev_distractor_v1.json")
    ap.add_argument("--output-dir", default="results/sf_counterfactual_r1_v3")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    args = ap.parse_args()
    if args.smoke: args.n = 6

    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(l) for l in open(args.meta_path)][:args.n]
    hpqa = json.load(open(args.hotpotqa_path))
    id2ans = {i["_id"]: i["answer"] for i in hpqa}
    for s in samples: s["answer"] = id2ans.get(s["sample_id"])

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tool_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    model.eval(); device = next(model.parameters()).device
    builder = PromptBuilder(tools=["search"])

    res = {"1sf": [], "2sf": []}
    for i, s in enumerate(samples):
        if i % 10 == 0: print(f"  [{i}/{len(samples)}] {s['sample_id']}", flush=True)
        for cond in ("1sf", "2sf"):
            res[cond].append(run_one(s, cond, model, tok, builder, device,
                                     tool_ids, fin_ids, args.max_new_tokens))
    for cond, store in res.items():
        p = outdir / f"r1_{cond}_trajectories_v3.jsonl"
        with open(p, "w") as f:
            for r in store: f.write(json.dumps(r) + "\n")
        print(f"Saved {len(store)} → {p}")

    print("\n" + "=" * 60)
    for cond, store in res.items():
        n = len(store)
        valid = [r for r in store if r["margin_B_true"] is not None]
        n_first_search = sum(1 for r in store if r["action_type_first"] == "search")
        n_first_stop   = sum(1 for r in store if r["action_type_first"] == "stop")
        n_v2_search    = sum(1 for r in store if r["action_type_v2"] == "search")
        n_v2_stop      = sum(1 for r in store if r["action_type_v2"] == "stop")
        mA = np.mean([r["margin_A_v2eq"] for r in valid])
        mB = np.mean([r["margin_B_true"] for r in valid])
        print(f"\n  {cond.upper()} (N={n}, valid={len(valid)}):")
        print(f"    First-token 2ndSR:  {n_first_search}/{n} = {n_first_search/n*100:.1f}%  stop={n_first_stop}")
        print(f"    v2-parse 2ndSR:     {n_v2_search}/{n} = {n_v2_search/n*100:.1f}%  stop={n_v2_stop}")
        print(f"    Margin A (v2-eq):   {mA:+.3f}   Margin B (true): {mB:+.3f}")
        # Confusion: margin_B sign vs first_action
        tp = sum(1 for r in valid if r["margin_B_true"] > 0 and r["action_type_first"] == "search")
        tn = sum(1 for r in valid if r["margin_B_true"] < 0 and r["action_type_first"] == "stop")
        fp = sum(1 for r in valid if r["margin_B_true"] > 0 and r["action_type_first"] == "stop")
        fn = sum(1 for r in valid if r["margin_B_true"] < 0 and r["action_type_first"] == "search")
        print(f"    Confusion (margin_B sign vs first_action): TP={tp} TN={tn} FP={fp} FN={fn}")


if __name__ == "__main__":
    main()
