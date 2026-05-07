#!/usr/bin/env python3
"""LLM-judge precision audit for the extractability heuristic.

For each (question, observation, candidate_W) we ask Qwen2.5-7B (in a
neutral judge role, NOT the agent role) whether the observation provides
complete logical support for `W` as the answer to the question.

We then compute heuristic precision (= fraction the judge calls
'NOT supported') for several subsets:
  - all v1-extractable wrong-stop (n up to 50)
  - all v2-extractable wrong-stop (n up to ~25)
  - v1-only wrong-stop (extractable under v1 but supported under v2)
  - all v1-extractable correct-stop (n up to 50; expected to be mostly
    'supported' by judge => false-positive rate of heuristic)
"""
import argparse, json, random, re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


JUDGE_SYS = (
    "You are an expert reading-comprehension judge. You will be given a "
    "question, several retrieved passages, and a candidate answer. Decide "
    "whether the passages provide COMPLETE LOGICAL SUPPORT for the candidate "
    "answer being correct. Be strict: if the passages only mention the "
    "candidate string in passing, or only support a related claim, that is "
    "NOT complete support. Reply with exactly one line of the form:\n"
    "VERDICT: <YES|NO|UNCLEAR>\n"
    "REASON: <one short sentence>"
)


def build_prompt(question, observation, W):
    return (
        f"Question: {question}\n\n"
        f"Passages:\n{observation}\n\n"
        f"Candidate answer: {W}\n\n"
        "Does the passage set above provide complete logical support for "
        "this candidate answer?"
    )


def parse_verdict(text):
    m = re.search(r"verdict\s*:\s*(yes|no|unclear)", text, re.IGNORECASE)
    if not m:
        return None, text.strip()[:200]
    v = m.group(1).lower()
    rm = re.search(r"reason\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    reason = rm.group(1).strip()[:300] if rm else ""
    return v, reason


def judge(model, tok, question, observation, W, max_new_tokens=80):
    msgs = [
        {"role": "system", "content": JUDGE_SYS},
        {"role": "user", "content": build_prompt(question, observation, W)},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=None, top_p=None, top_k=None,
            pad_token_id=tok.eos_token_id,
        )
    gen = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    v, reason = parse_verdict(gen)
    return v, reason, gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--audit", default="results/natural_extractability_audit/natural_audit_raw_v2.jsonl")
    ap.add_argument("--out",   default="results/natural_extractability_audit/llm_judge_results.jsonl")
    ap.add_argument("--n-per-subset", type=int, default=50)
    ap.add_argument("--seed",  type=int, default=0)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.audit)]
    by_id = {r["sample_id"]: r for r in records}

    pools = {
        "wrong_v1": [r for r in records if r["category"] == "step1_stop_wrong"
                     and r.get("extractable_unsupported")],
        "wrong_v2": [r for r in records if r["category"] == "step1_stop_wrong"
                     and r.get("extractable_unsupported_v2")],
        "wrong_v1_only": [r for r in records if r["category"] == "step1_stop_wrong"
                          and r.get("extractable_unsupported")
                          and not r.get("extractable_unsupported_v2")],
        "correct_v1": [r for r in records if r["category"] == "step1_stop_correct"
                       and r.get("extractable_unsupported")],
    }
    print({k: len(v) for k, v in pools.items()})

    rng = random.Random(args.seed)
    targets = {}
    for k, pool in pools.items():
        rng.shuffle(pool)
        targets[k] = pool[:args.n_per_subset]
    # union of unique sample_ids actually to judge
    seen = {}
    for k, items in targets.items():
        for r in items:
            seen.setdefault(r["sample_id"], set()).add(k)
    print(f"Unique samples to judge: {len(seen)}")

    print("Loading model …")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(seen); done = 0
    with open(out_path, "w") as f:
        for sid, subset_tags in seen.items():
            r = by_id[sid]
            v, reason, gen = judge(
                model, tok, r["question"], r["observation_full"],
                r["emitted_answer_W"],
            )
            rec = {
                "sample_id": sid,
                "subset_tags": sorted(subset_tags),
                "category": r["category"],
                "question": r["question"],
                "gold_answer": r["gold_answer"],
                "candidate_W": r["emitted_answer_W"],
                "extractable_v1": bool(r.get("extractable_unsupported")),
                "extractable_v2": bool(r.get("extractable_unsupported_v2")),
                "judge_verdict": v,
                "judge_reason": reason,
                "judge_raw": gen[:400],
            }
            f.write(json.dumps(rec) + "\n"); f.flush()
            done += 1
            if done % 10 == 0 or done == n:
                print(f"  [{done}/{n}]")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
