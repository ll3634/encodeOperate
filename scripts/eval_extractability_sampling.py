#!/usr/bin/env python3
"""Decoding robustness check for the N0/T0 Extractability-Support toggle.

Runs Qwen2.5-7B (or any HF causal LM) on the existing controlled pairs with
sampling decoding (temperature, top_p) across multiple seeds. Same conditions,
same prompts, same parser as the corrected cross-model behaviour experiments,
but with a *naturalistic* full-ReAct system prompt that does NOT contain the
"first word must be Action or Final" hard constraint (per user spec).

Per generation we log: sample_id, condition, seed, raw_output, parsed action,
final answer, contains_W, contains_gold, EM, parse_failure, malformed_both,
n_generated_tokens.
"""
import argparse, json, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.prompts import TOOL_DESCRIPTIONS, parse_action  # noqa: E402
from eval.scorers import answer_scorer                     # noqa: E402
from scripts.eval_extractability_cross_model import (      # noqa: E402
    apply_chat_template_safe, first_action_token, to_natural_snippet,
)


# Naturalistic full-ReAct system prompt: full two-shape format, no
# "first word must be Action or Final" restriction (per spec).
SYSTEM_PROMPT = """You are a helpful assistant that answers questions using available tools.

Available tools:
{tool_descriptions}

Respond in exactly ONE of the following two formats.

If you need to use a tool:
Action: search
Action Input: <your next query>

If you can answer directly:
Final Answer: <your answer>"""


def build_messages(question, observation, obs_style="factcard"):
    sys_p = SYSTEM_PROMPT.format(
        tool_descriptions="- " + TOOL_DESCRIPTIONS["search"]
    )
    if obs_style == "snippet":
        observation = to_natural_snippet(observation)
    user = (
        f"{question}\n\n"
        f"I have already run a search for you.\n"
        f"Tool: search\n"
        f"Tool input: about: {question[:80]}\n"
        f"Tool result:\n{observation}\n"
    )
    return [{"role": "system", "content": sys_p},
            {"role": "user",   "content": user}]


def has_both_markers(text: str) -> bool:
    tl = text.lower()
    return ("action:" in tl) and (("final answer:" in tl) or ("final:" in tl))


def run_one(rec, model, tokenizer, device, seed, max_new_tokens,
            temperature, top_p, obs_style, greedy=False):
    set_seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    messages = build_messages(rec["question"], rec["obs"], obs_style=obs_style)
    prompt_str = apply_chat_template_safe(tokenizer, messages,
                                          add_generation_prompt=True)
    p_ids = tokenizer.encode(prompt_str, return_tensors="pt",
                             add_special_tokens=False).to(device)
    attn = torch.ones_like(p_ids)
    gen_kwargs = dict(max_new_tokens=max_new_tokens,
                      pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    if greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))
    with torch.no_grad():
        gen = model.generate(p_ids, attention_mask=attn, **gen_kwargs)
    raw = tokenizer.decode(gen[0, p_ids.shape[1]:], skip_special_tokens=True)
    n_gen = int(gen.shape[1] - p_ids.shape[1])

    parsed = parse_action(raw)
    a2, fa = parsed["action"], parsed["final_answer"]
    pf = (a2 is None and fa is None)
    if a2 and a2.lower() in ("search", "calculator"):
        action_type = "search"
    elif fa is not None:
        action_type = "stop"
    else:
        action_type = None

    fa_first = first_action_token(raw)
    malformed_both = has_both_markers(raw)

    W = rec.get("candidate_W") or rec.get("W") or ""
    contains_W = int(bool(W) and W.lower() in (fa or "").lower())
    gold = rec.get("gold_answers") or ([rec["gold_answer"]] if rec.get("gold_answer") else [])
    contains_gold = int(any(g and g.lower() in (fa or "").lower() for g in gold))
    em = None
    if fa is not None and gold:
        em = int(answer_scorer(fa, gold, mode="exact")["matched"])

    return {
        "sample_id": rec["sample_id"],
        "schema_type": rec.get("schema_type"),
        "condition": rec.get("condition") or rec.get("condition_id"),
        "seed": int(seed),
        "candidate_W": W,
        "action": a2, "final_answer": fa,
        "action_type": action_type,
        "first_action_token": fa_first,
        "contains_W": int(contains_W),
        "contains_gold": int(contains_gold),
        "em": em,
        "parse_failure": bool(pf),
        "malformed_both": bool(malformed_both),
        "n_generated_tokens": n_gen,
        "raw_output": raw[:600],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/extractability_support_toggle/pairs.jsonl")
    ap.add_argument("--out", required=True, help="Output JSONL (per-generation rows).")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--conditions", nargs="+", default=["N0", "T0", "S0"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 17, 23, 42, 99])
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--obs-style", choices=["factcard", "snippet"], default="factcard")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit on number of unique sample_ids per condition.")
    ap.add_argument("--greedy", action="store_true",
                    help="Greedy decoding (do_sample=False). Ignores --temperature/--top-p. "
                         "Per-(sample,condition) the run is deterministic; --seeds list is "
                         "honoured only as a label and to avoid trivial duplicate generations.")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.pairs)]
    rows = [r for r in rows
            if (r.get("condition") or r.get("condition_id")) in args.conditions]
    if args.limit:
        keep_ids, kept = set(), []
        for r in rows:
            if r["sample_id"] not in keep_ids and len(keep_ids) < args.limit:
                keep_ids.add(r["sample_id"])
            if r["sample_id"] in keep_ids:
                kept.append(r)
        rows = kept

    n_total = len(rows) * len(args.seeds)
    mode = "greedy" if args.greedy else f"sampling(temp={args.temperature}, top_p={args.top_p})"
    print(f"[info] model={args.model_path}; pairs_rows={len(rows)}; "
          f"seeds={args.seeds}; total_generations={n_total}; mode={mode} "
          f"max_new={args.max_new_tokens} obs_style={args.obs_style}")

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.eval()
    device = next(model.parameters()).device

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); n_done = 0
    with open(out, "w") as f:
        for rec in rows:
            for seed in args.seeds:
                row = run_one(rec, model, tok, device,
                              seed=seed, max_new_tokens=args.max_new_tokens,
                              temperature=args.temperature, top_p=args.top_p,
                              obs_style=args.obs_style, greedy=args.greedy)
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
                n_done += 1
                if n_done % 25 == 0 or n_done == n_total:
                    el = time.time() - t0
                    eta = el * (n_total - n_done) / max(1, n_done)
                    print(f"  [{n_done}/{n_total}] elapsed={el:.1f}s eta={eta:.1f}s")
    print(f"[done] -> {out}  total_time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
