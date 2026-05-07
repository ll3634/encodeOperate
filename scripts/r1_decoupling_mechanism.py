#!/usr/bin/env python3
"""
R1 margin-action decoupling mechanism probe.

For a subset of samples, regenerate the R1 trajectory but keep:
  - full raw_output (not truncated)
  - post-</think> text
  - top-10 tokens with logits at the position immediately after </think>
  - logit("Action"), logit("Final"), top-1 token + its logit

Input: pre-selected sample_ids (split into 'decoupled' and 'coupled_stop' groups).
Output: per-sample record with the extra diagnostic fields.
"""

import os, sys, re, json, argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS, parse_action
from eval.scorers import answer_scorer


MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


def run_one(sample, condition, model, tokenizer, builder, device, max_new_tokens=1200):
    obs = sample["obs_1sf"] if condition == "1sf" else sample["obs_2sf"]
    steps = [{"action": "search", "action_input": sample["query"], "observation": obs}]
    msgs = builder.build_full_prompt(sample["question"], steps)
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    attn = torch.ones_like(input_ids)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        attention_mask=attn,
        do_sample=False,
    )
    with torch.no_grad():
        gen_ids = model.generate(input_ids, **gen_kwargs)

    gen_token_ids = gen_ids[0, prompt_len:].tolist()
    raw_output = tokenizer.decode(gen_ids[0, prompt_len:], skip_special_tokens=True)

    think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
    think_pos = next((i for i, t in enumerate(gen_token_ids) if t == think_end_id), None)
    if think_pos is None:
        return None

    # text split
    think_text = tokenizer.decode(gen_ids[0, prompt_len:prompt_len+think_pos+1], skip_special_tokens=True)
    post_text  = tokenizer.decode(gen_ids[0, prompt_len+think_pos+1:], skip_special_tokens=True)

    # logits at position right after </think>
    prefix = gen_ids[0, :prompt_len + think_pos + 1].unsqueeze(0)
    with torch.no_grad():
        post_out = model(prefix, attention_mask=torch.ones_like(prefix))
    logits = post_out.logits[0, -1, :].float()
    lp = torch.log_softmax(logits, dim=-1)

    tool_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]
    action_lp = torch.logsumexp(lp[tool_ids], 0).item()
    final_lp  = torch.logsumexp(lp[fin_ids], 0).item()
    margin = action_lp - final_lp

    # top-10 tokens at post-</think> position
    topk = torch.topk(lp, k=10)
    top_list = [{"tok": tokenizer.decode([tid.item()]).replace("\n", "\\n"),
                 "tok_id": tid.item(),
                 "logit": logits[tid].item(),
                 "logp": tl.item()} for tid, tl in zip(topk.indices, topk.values)]
    top1 = top_list[0]

    # parse action from post-text
    text_for_parse = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL)
    text_for_parse = text_for_parse.replace('<think>', '').replace('</think>', '')
    parsed = parse_action(text_for_parse)
    action2 = parsed["action"]
    final_answer = parsed["final_answer"]
    if action2 and action2.lower() in ("search", "calculator"):
        action_type = "search"
    elif final_answer is not None:
        action_type = "stop"
    else:
        action_type = None

    em = None
    if final_answer is not None and sample.get("answer"):
        em = int(answer_scorer(final_answer, sample["answer"], mode="exact")["matched"])

    return {
        "sample_id": sample["sample_id"],
        "condition": condition,
        "think_len_tok": think_pos,
        "post_len_tok": len(gen_token_ids) - think_pos - 1,
        "margin_post": margin,
        "action_logp": action_lp,
        "final_logp": final_lp,
        "top1_tok": top1["tok"],
        "top1_logp": top1["logp"],
        "top10": top_list,
        "action_type": action_type,
        "action2": action2,
        "final_answer": final_answer,
        "em": em,
        "think_text": think_text,
        "post_text": post_text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True,
                    help="Path to r1_{cond}_trajectories.jsonl")
    ap.add_argument("--condition", required=True, choices=["1sf", "2sf"])
    ap.add_argument("--meta-path", default="results/probe_sufficiency_v2/meta.jsonl")
    ap.add_argument("--hotpotqa-path", default="data/hotpotqa/hotpot_dev_distractor_v1.json")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-decoupled", type=int, default=40)
    ap.add_argument("--n-coupled", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input_jsonl)]
    decoup_ids = [x["sample_id"] for x in rows
                  if x.get("margin_post") is not None and x["margin_post"] < 0
                  and x.get("action_type") == "search"]
    coup_ids = [x["sample_id"] for x in rows
                if x.get("margin_post") is not None and x["margin_post"] < 0
                and x.get("action_type") == "stop"]
    rng = np.random.RandomState(args.seed)
    decoup_pick = list(rng.choice(decoup_ids, size=min(args.n_decoupled, len(decoup_ids)), replace=False))
    coup_pick   = list(rng.choice(coup_ids,   size=min(args.n_coupled, len(coup_ids)),   replace=False))
    print(f"Selected decoupled={len(decoup_pick)}  coupled={len(coup_pick)}")

    samples_all = [json.loads(l) for l in open(args.meta_path)]
    by_id = {s["sample_id"]: s for s in samples_all}
    hpqa = json.load(open(args.hotpotqa_path))
    id2ans = {item["_id"]: item["answer"] for item in hpqa}

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    device = next(model.parameters()).device
    builder = PromptBuilder(tools=["search"])

    out = []
    for group, ids in [("decoupled", decoup_pick), ("coupled_stop", coup_pick)]:
        for i, sid in enumerate(ids):
            s = dict(by_id[sid])
            s["answer"] = id2ans.get(sid)
            r = run_one(s, args.condition, model, tokenizer, builder, device)
            if r is None:
                continue
            r["group"] = group
            out.append(r)
            if i % 10 == 0:
                print(f"  [{group} {i}/{len(ids)}] sid={sid} margin={r['margin_post']:+.2f} top1={r['top1_tok']!r} act={r['action_type']}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(out)} → {args.output}")


if __name__ == "__main__":
    main()
