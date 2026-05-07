#!/usr/bin/env python3
"""Post-tool decision eval for the Extractability-Support-Missingness toggle.

No steering; records per (sample_id, condition):
  - parsed action (search / stop), commit-W, commit-V, parse failure
  - teacher-forced label margin (ml): logP(search | Action:) - logP({Final, Action: Final Answer} | Action:)
  - first-token margin (mft): logP(Action) - logP(Final) at the initial decode step
  - greedy raw output

Reuses run_steering_trap_eval.compute_label_margin / setup_label_tokens /
PromptBuilder conventions.
"""
import argparse, json, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent.prompts import PromptBuilder, parse_action                # noqa: E402
from eval.scorers import answer_scorer                               # noqa: E402
from run_steering_trap_eval import (                                 # noqa: E402
    setup_label_tokens, compute_label_margin,
)


def run_one(rec, model, tokenizer, builder, device, label_tokens,
            max_new_tokens=256):
    obs = rec["obs"]
    query = f"about: {rec['question'][:80]}"
    steps = [{"action": "search", "action_input": query, "observation": obs}]
    messages = builder.build_full_prompt(rec["question"], steps)
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt_str, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    attn = torch.ones_like(input_ids)

    margins = compute_label_margin(model, input_ids, prompt_len, None, 0.0,
                                   label_tokens, device)

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids, attention_mask=attn, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            do_sample=False,
        )
    raw = tokenizer.decode(gen_ids[0, prompt_len:], skip_special_tokens=True)

    parsed = parse_action(raw)
    a2, fa = parsed["action"], parsed["final_answer"]
    pf = (a2 is None and fa is None)
    if a2 and a2.lower() in ("search", "calculator"):
        action_type = "search"
    elif fa is not None:
        action_type = "stop"
    else:
        action_type = None

    W = rec.get("candidate_W") or rec.get("W") or ""
    fa_low = (fa or "").lower()
    contains_W = int(bool(W) and W.lower() in fa_low)

    em = None
    if fa is not None and rec.get("gold_answer"):
        gold = rec.get("gold_answers") or [rec["gold_answer"]]
        em = int(answer_scorer(fa, gold, mode="exact")["matched"])

    return {
        "sample_id":    rec["sample_id"],
        "schema_type":  rec.get("schema_type") or rec.get("schema"),
        "condition":    rec.get("condition") or rec.get("condition_id"),
        "condition_id": rec.get("condition_id") or rec.get("condition"),
        "candidate_W":  W,
        "E_intended":   rec.get("E_intended"),
        "S_intended":   rec.get("S_intended"),
        "M_intended":   rec.get("M_intended"),
        "margin_label":      margins["margin_label"],
        "margin_first_token": margins["margin_first_token"],
        "lp_Action":         margins["lp_Action"],
        "lp_Final":          margins["lp_Final"],
        "lp_search_after":   margins["lp_search_after"],
        "lp_Final_after":    margins["lp_Final_after"],
        "action_type":  action_type,
        "action2":      a2,
        "final_answer": fa,
        "em":           em,
        "contains_W":   contains_W,
        "parse_failure": pf,
        "raw_output":   raw[:400],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/extractability_support_toggle/pairs.jsonl")
    ap.add_argument("--out",   default="results/extractability_support_toggle/eval_results.jsonl")
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--conditions", nargs="+",
                    default=["N0", "T0", "T1", "S0"])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.pairs)]
    records = [r for r in records
               if (r.get("condition") or r.get("condition_id")) in args.conditions]
    if args.limit:
        records = records[:args.limit]
    print(f"[info] loaded {len(records)} records; conditions={args.conditions}")

    print(f"[info] loading model {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    ); model.eval()
    device = next(model.parameters()).device

    builder = PromptBuilder()
    label_tokens = setup_label_tokens(tok)

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out_path, "w") as f:
        for i, rec in enumerate(records, 1):
            row = run_one(rec, model, tok, builder, device, label_tokens,
                          max_new_tokens=args.max_new_tokens)
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            if i % 25 == 0 or i == len(records):
                print(f"  [{i}/{len(records)}] {time.time()-t0:.1f}s")
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
