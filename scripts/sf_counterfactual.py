#!/usr/bin/env python3
"""
SF Counterfactual Behavioral Experiment
========================================
Compare model behavior after 1-SF vs 2-SF observations.

For each of 300 paired samples, construct teacher-forced prefix:
  [system] ... [user] question [assistant] Action: search / Action Input: query
  Observation: {obs_1sf OR obs_2sf}

Then free-generate and measure:
  - 2ndSR: does model issue a second search (Action) or stop (Final Answer)?
  - margin: logit(Action) - logit(Final) at decision point
  - EM: if Final Answer, exact match with ground truth

Usage:
  cd tmc/scripts/e2e_agent
  python scripts/sf_counterfactual.py --model qwen --n 300 --output-dir results/sf_counterfactual
  python scripts/sf_counterfactual.py --model qwen --n 10 --smoke   # quick test
"""

import os, sys, re, json, argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS, DEFAULT_SYSTEM_PROMPT, parse_action
from eval.scorers import answer_scorer


# ── Margin computation ────────────────────────────────────────────────────────

def compute_margin(logits, tokenizer):
    """logit(Action tokens) - logit(Final tokens) at last position."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    tool_ids = [tokenizer.encode(t, add_special_tokens=False)[0]
                for t in ACTION_TOKENS["tool_call"] if tokenizer.encode(t, add_special_tokens=False)]
    fin_ids  = [tokenizer.encode(t, add_special_tokens=False)[0]
                for t in ACTION_TOKENS["finish"] if tokenizer.encode(t, add_special_tokens=False)]
    tool_lp = torch.logsumexp(log_probs[tool_ids], 0).item() if tool_ids else -100.0
    fin_lp  = torch.logsumexp(log_probs[fin_ids],  0).item() if fin_ids  else -100.0
    return tool_lp - fin_lp


# ── Prompt building ───────────────────────────────────────────────────────────

def build_prefix_messages(question: str, query: str, observation: str,
                           builder: PromptBuilder) -> list:
    """Build teacher-forced prefix messages for Qwen (uses apply_chat_template)."""
    steps = [{"action": "search", "action_input": query, "observation": observation}]
    return builder.build_full_prompt(question, steps)


def build_mistral_raw_prompt(question: str, query: str, observation: str,
                              tool_desc: str = "- search(query): Search for information about a topic") -> str:
    """
    Build raw Mistral prompt string.
    Mistral [INST]...[/INST] format: assistant turn is NOT closed with </s>
    so the model continues generating in the same turn.
    """
    sys_prompt = DEFAULT_SYSTEM_PROMPT.format(tool_descriptions=tool_desc)
    return (
        f"<s>[INST] {sys_prompt}\n\n{question} [/INST] "
        f"Action: search\nAction Input: {query}\nObservation: {observation}\n"
    )


# ── Single-sample runner ──────────────────────────────────────────────────────

def run_one(sample: dict, condition: str, model, tokenizer,
            builder: PromptBuilder, device, max_new_tokens: int = 512,
            use_mistral_format: bool = False) -> dict:
    """
    Run one sample under one condition (1sf or 2sf).

    Returns a dict with: sample_id, condition, margin, action2, final_answer,
    em, parse_failure, raw_output
    """
    obs = sample["obs_1sf"] if condition == "1sf" else sample["obs_2sf"]

    # ── Build prompt string ──
    if use_mistral_format:
        prompt_str = build_mistral_raw_prompt(sample["question"], sample["query"], obs)
        input_ids = tokenizer.encode(prompt_str, return_tensors="pt",
                                     add_special_tokens=False).to(device)
    else:
        messages = build_prefix_messages(sample["question"], sample["query"], obs, builder)
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer.encode(prompt_str, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    attn_mask = torch.ones_like(input_ids)

    # ── Compute margin (one forward pass, no grad) ──
    with torch.no_grad():
        out = model(input_ids, attention_mask=attn_mask)
    margin = compute_margin(out.logits[0, -1, :], tokenizer)

    # ── Free-generate continuation ──
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        attention_mask=attn_mask,
        do_sample=False,
    )
    with torch.no_grad():
        gen_ids = model.generate(input_ids, **gen_kwargs)

    completion_ids = gen_ids[0, prompt_len:]
    raw_output = tokenizer.decode(completion_ids, skip_special_tokens=True)

    # ── Parse action ──
    parsed = parse_action(raw_output)
    action2 = parsed["action"]          # "search" / None
    final_answer = parsed["final_answer"]

    parse_failure = (action2 is None and final_answer is None)

    # ── Determine action type ──
    # "search" if action2 is search-type, "stop" if final_answer, None if failure
    if action2 and action2.lower() in ("search", "calculator"):
        action_type = "search"
    elif final_answer is not None:
        action_type = "stop"
    elif use_mistral_format and re.search(r'^\s*\[\d+\]', raw_output):
        # Mistral sometimes generates "[N] Title: ..." — it hallucinates additional
        # search results rather than issuing an actual search Action.
        # This is an implicit STOP / hallucination, NOT a genuine second search.
        # Classify as "hallucinated_obs" and treat as stop for McNemar purposes.
        action_type = "hallucinated_obs"
        parse_failure = False   # behavior is identifiable; not a format failure
    else:
        action_type = None

    # ── EM scoring ──
    em = None
    if final_answer is not None and sample.get("answer"):
        result = answer_scorer(final_answer, sample["answer"], mode="exact")
        em = int(result["matched"])

    return {
        "sample_id": sample["sample_id"],
        "condition": condition,
        "margin": margin,
        "action_type": action_type,   # "search" / "stop" / None
        "action2": action2,
        "final_answer": final_answer,
        "em": em,
        "parse_failure": parse_failure,
        "raw_output": raw_output[:300],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen",
                        choices=["qwen", "mistral"],
                        help="Which model to run")
    parser.add_argument("--model-path", default=None,
                        help="Override model path/name")
    parser.add_argument("--meta-path",
                        default="results/probe_sufficiency_v2/meta.jsonl",
                        help="Path to paired meta.jsonl")
    parser.add_argument("--hotpotqa-path",
                        default="data/hotpotqa/hotpot_dev_distractor_v1.json",
                        help="Path to HotpotQA JSON for answer lookup")
    parser.add_argument("--output-dir", default="results/sf_counterfactual")
    parser.add_argument("--n", type=int, default=300,
                        help="Max number of paired samples to run")
    parser.add_argument("--smoke", action="store_true",
                        help="Run only 10 samples for a quick sanity check")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    if args.smoke:
        args.n = 10

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load paired samples ──
    print(f"Loading paired samples from {args.meta_path} ...")
    samples = []
    with open(args.meta_path) as f:
        for line in f:
            samples.append(json.loads(line))
    samples = samples[: args.n]
    print(f"  {len(samples)} samples loaded")

    # ── Load answers from HotpotQA ──
    print(f"Loading answers from {args.hotpotqa_path} ...")
    with open(args.hotpotqa_path) as f:
        hpqa = json.load(f)
    id2ans = {item["_id"]: item["answer"] for item in hpqa}
    for s in samples:
        s["answer"] = id2ans.get(s["sample_id"])
    print(f"  answers matched: {sum(1 for s in samples if s['answer'])}/{len(samples)}")

    # ── Load model ──
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if args.model_path:
        model_id = args.model_path
    elif args.model == "qwen":
        model_id = "Qwen/Qwen2.5-7B-Instruct"
    else:
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    print(f"\nLoading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  model on {device}, dtype={next(model.parameters()).dtype}")

    # ── Build prompt builder ──
    builder = PromptBuilder(tools=["search"])
    use_mistral_format = (args.model == "mistral")

    # ── Run experiment ──
    results_1sf = []
    results_2sf = []
    total = len(samples)

    print(f"\nRunning {total} pairs × 2 conditions ...")
    for i, sample in enumerate(samples):
        if i % 20 == 0:
            print(f"  [{i}/{total}] processing sample {sample['sample_id']}")
        for cond, store in [("1sf", results_1sf), ("2sf", results_2sf)]:
            r = run_one(sample, cond, model, tokenizer, builder, device,
                        max_new_tokens=args.max_new_tokens,
                        use_mistral_format=use_mistral_format)
            store.append(r)

    # ── Save trajectories ──
    model_tag = args.model
    for cond, store in [("1sf", results_1sf), ("2sf", results_2sf)]:
        out_path = outdir / f"{model_tag}_{cond}_trajectories.jsonl"
        with open(out_path, "w") as f:
            for r in store:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {len(store)} records to {out_path}")

    # ── Quick summary ──
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    for cond, store in [("1sf", results_1sf), ("2sf", results_2sf)]:
        n = len(store)
        n_search = sum(1 for r in store if r["action_type"] == "search")
        n_stop   = sum(1 for r in store if r["action_type"] == "stop")
        n_pf     = sum(1 for r in store if r["parse_failure"])
        margins  = [r["margin"] for r in store]
        em_vals  = [r["em"] for r in store if r["em"] is not None]
        print(f"\n  {cond.upper()} (N={n}):")
        print(f"    2ndSR (search): {n_search}/{n} = {n_search/n*100:.1f}%")
        print(f"    Stop:           {n_stop}/{n} = {n_stop/n*100:.1f}%")
        print(f"    Parse failure:  {n_pf}/{n} = {n_pf/n*100:.1f}%")
        print(f"    Mean margin:    {np.mean(margins):.3f} ± {np.std(margins):.3f}")
        if em_vals:
            print(f"    EM (if stop):   {np.mean(em_vals)*100:.1f}% (N={len(em_vals)})")

    print("\nDone. Run analysis with: python scripts/analyze_sf_counterfactual.py")


if __name__ == "__main__":
    main()
