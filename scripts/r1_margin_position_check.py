#!/usr/bin/env python3
"""
Verify the correct margin measurement position for R1.

For each sample, measure logit("Action") vs logit("Final") at THREE positions:
  pos A: right after </think>   (current margin_post — likely wrong)
  pos B: right after </think>\n\n   (true decision position — candidate)
  pos C: right after </think>\n\n + next token   (sanity check)

If pos B gives top-1 = "Action" for search-action samples and top-1 = "Final"/narrative
for stop-action samples, then margin_post is off-by-one and needs to be re-measured.
"""

import sys, json, re
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


def run_check(sample, condition, model, tokenizer, builder, device, max_new_tokens=1200):
    obs = sample["obs_1sf"] if condition == "1sf" else sample["obs_2sf"]
    steps = [{"action": "search", "action_input": sample["query"], "observation": obs}]
    msgs = builder.build_full_prompt(sample["question"], steps)
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        gen_ids = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    gen_token_ids = gen_ids[0, prompt_len:].tolist()

    think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
    think_pos = next((i for i, t in enumerate(gen_token_ids) if t == think_end_id), None)
    if think_pos is None or think_pos + 3 >= len(gen_token_ids):
        return None

    tool_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]

    def logits_at(end_idx):
        prefix = gen_ids[0, :prompt_len + end_idx + 1].unsqueeze(0)
        with torch.no_grad():
            out = model(prefix, attention_mask=torch.ones_like(prefix))
        return out.logits[0, -1, :].float()

    def margin_and_top(logits, tokenizer):
        lp = torch.log_softmax(logits, dim=-1)
        a = torch.logsumexp(lp[tool_ids], 0).item()
        f = torch.logsumexp(lp[fin_ids],  0).item()
        top1_id = lp.argmax().item()
        top1_tok = tokenizer.decode([top1_id])
        top1_lp = lp[top1_id].item()
        return {"action_lp": a, "final_lp": f, "margin": a - f,
                "top1_tok": top1_tok, "top1_lp": top1_lp}

    # pos A: right after </think>
    A = margin_and_top(logits_at(think_pos), tokenizer)
    # pos B: right after </think> + 1 more token
    B = margin_and_top(logits_at(think_pos + 1), tokenizer)
    # pos C: + 2 tokens
    C = margin_and_top(logits_at(think_pos + 2), tokenizer)

    # what are the actual tokens in that stretch
    sep_tokens = [tokenizer.decode([t]) for t in gen_token_ids[think_pos:think_pos+5]]

    return {
        "sample_id": sample["sample_id"],
        "condition": condition,
        "sep_tokens": sep_tokens,   # actual tokens at positions [think, +1, +2, +3, +4]
        "A_after_endthink": A,
        "B_after_endthink_plus1": B,
        "C_after_endthink_plus2": C,
    }


def main():
    # Pick 6 samples — 3 decoupled + 3 coupled_stop — from 1sf v2 data
    v2 = [json.loads(l) for l in open("results/sf_counterfactual_r1_v2/r1_1sf_trajectories.jsonl")]
    decoup = [x for x in v2 if x.get('margin_post') is not None and x['margin_post'] < 0 and x['action_type']=='search'][:3]
    coupled = [x for x in v2 if x.get('margin_post') is not None and x['margin_post'] < 0 and x['action_type']=='stop'][:3]

    meta = {json.loads(l)["sample_id"]: json.loads(l)
            for l in open("results/probe_sufficiency_v2/meta.jsonl")}
    hpqa = json.load(open("data/hotpotqa/hotpot_dev_distractor_v1.json"))
    id2ans = {i["_id"]: i["answer"] for i in hpqa}

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    device = next(model.parameters()).device
    builder = PromptBuilder(tools=["search"])

    print(f"\n{'GROUP':<14}{'act':<8}{'sep_tokens (5 after </think>)':<45}"
          f"{'A_top1':<12}{'A_margin':>10}{'B_top1':<12}{'B_margin':>10}{'C_top1':<12}{'C_margin':>10}")
    print("-" * 140)
    for group, grp in [("decoupled", decoup), ("coupled_stop", coupled)]:
        for x in grp:
            sid = x['sample_id']
            s = dict(meta[sid]); s["answer"] = id2ans.get(sid)
            act = x['action_type']
            r = run_check(s, "1sf", model, tok, builder, device)
            if r is None:
                print(f"{group:<14}{act:<8}SKIP (no </think>)"); continue
            seps = ' '.join(repr(t) for t in r['sep_tokens'])[:42]
            A, B, C = r['A_after_endthink'], r['B_after_endthink_plus1'], r['C_after_endthink_plus2']
            print(f"{group:<14}{act:<8}{seps:<45}"
                  f"{A['top1_tok']!r:<12}{A['margin']:>+10.2f}"
                  f"{B['top1_tok']!r:<12}{B['margin']:>+10.2f}"
                  f"{C['top1_tok']!r:<12}{C['margin']:>+10.2f}")


if __name__ == "__main__":
    main()
