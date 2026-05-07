#!/usr/bin/env python3
"""Cross-model behavioural replication of the Extractability-Support toggle.

Conditions: N0, T0, S0 (T1 optional via --include-T1).
Per (sample, condition) records:
  - parsed action / final answer / commit-W / EM / parse-failure
  - teacher-forced label margin (model-agnostic, sequence logp):
        margin_label = sum logp(" search\\nAction Input:" | prompt + "Action:")
                     - sum logp(" Final Answer:"          | prompt + "Action:")
  - first-token margin (logP(first tok of "Action") - logP(first tok of "Final"))
  - greedy raw output

Prompt format: single-user-message ReAct (system + user-with-tool-trace).
This unified format is the only structure that survives Mistral's strict
[INST]/[/INST] alternation, Phi's two-turn schema, and Qwen's chatml.
"""
import argparse, json, re, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.prompts import DEFAULT_SYSTEM_PROMPT, TOOL_DESCRIPTIONS, parse_action  # noqa: E402
from eval.scorers import answer_scorer                                            # noqa: E402


SEARCH_CONT = " search\nAction Input:"
FINAL_CONT  = " Final Answer:"


def is_r1_model(path: str) -> bool:
    p = path.lower()
    return "deepseek-r1" in p or "r1-distill" in p


def first_action_token(text: str) -> str:
    """Which of Action: vs Final Answer: appears FIRST in `text`.
    Returns 'search', 'stop', or 'parse_fail'. Mirrors R1 v3 fix
    (CLAUDE.md §4.9 #8): the parser must reflect the FIRST decision the
    model commits to, not whichever marker appears anywhere in a longer
    hallucinated continuation."""
    if not text:
        return "parse_fail"
    tl = text.lower()
    ai = tl.find("action:")
    fi = tl.find("final answer:")
    if fi == -1:
        fi = tl.find("final:")
    if ai == -1 and fi == -1:
        return "parse_fail"
    if ai == -1:
        return "stop"
    if fi == -1:
        return "search"
    return "search" if ai < fi else "stop"


def strip_think(raw: str) -> str:
    """For R1: chat template puts opening <think> in the prompt, so raw_output
    only contains the think body followed by </think>. Drop everything up to
    and including </think>. Also handle the (rare) case where both tags appear."""
    s = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    if "</think>" in s:
        s = s.split("</think>", 1)[1]
    return s.lstrip()


# --- Prompt paraphrase variants (Robustness A) ---------------------------
# v1: terse two-shape template (the post-bug-fix default)
# v2: verbose decision-framed, Final Answer listed first (sufficiency framing)
# v3: minimal-formatting reverse-order template
_TAIL_V1 = (
    "Now produce a complete response in one of these two shapes:\n"
    "  (a)  Action: search\n"
    "       Action Input: <your next query>\n"
    "  (b)  Final Answer: <your answer>\n"
)
_TAIL_V2 = (
    "Decide your next step.\n\n"
    "If the tool result above contains enough information to answer the question, respond:\n\n"
    "Final Answer: <your answer>\n\n"
    "Otherwise, request another search by responding:\n\n"
    "Action: search\n"
    "Action Input: <your next query>\n"
)
_TAIL_V3 = (
    "Reply with EXACTLY one of these formats:\n\n"
    "Action: search\n"
    "Action Input: <next query>\n\n"
    "OR\n\n"
    "Final Answer: <answer>\n"
)
PROMPT_TAILS = {"v1": _TAIL_V1, "v2": _TAIL_V2, "v3": _TAIL_V3}


# --- Observation style rewriter (Robustness B) ---------------------------
def to_natural_snippet(obs: str) -> str:
    """Convert fact-card observation ('[1] Title: ... [2] Title: ...') into
    a single natural retrieval-snippet paragraph. Removes paragraph numbering,
    section titles, and the 'retrieved passage' boilerplate, keeping all
    factual content."""
    s = obs
    s = re.sub(r"^\s*\[\d+\]\s*[^:\n]{0,40}:\s*", "", s, flags=re.MULTILINE)
    s = s.replace("The retrieved text includes the following passage.", "")
    s = s.replace("End of the retrieved passage.", "")
    s = re.sub(r"\s+", " ", s).strip()
    return f"Search results:\n\n{s}"


def build_messages(question, observation, prompt_variant="v1", obs_style="factcard"):
    sys_p = DEFAULT_SYSTEM_PROMPT.format(
        tool_descriptions="- " + TOOL_DESCRIPTIONS["search"]
    )
    if obs_style == "snippet":
        observation = to_natural_snippet(observation)
    tail = PROMPT_TAILS[prompt_variant]
    user = (
        f"{question}\n\n"
        f"I have already run a search for you.\n"
        f"Tool: search\n"
        f"Tool input: about: {question[:80]}\n"
        f"Tool result:\n{observation}\n\n"
        f"{tail}"
    )
    return [{"role": "system", "content": sys_p},
            {"role": "user",   "content": user}]


def apply_chat_template_safe(tokenizer, messages, add_generation_prompt=True):
    """Apply chat template, handling:
    - Qwen3: enable_thinking=False to suppress <think> tokens and stay in
      standard action-generation mode (required for margin computation).
    - Gemma: merge system→user if model rejects system role.
    """
    chat_template = getattr(tokenizer, "chat_template", "") or ""
    # Auto-detect Qwen3 by presence of enable_thinking param in its template
    extra_kwargs = {"enable_thinking": False} if "enable_thinking" in chat_template else {}
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
            **extra_kwargs)
    except Exception as e:
        if "system" not in str(e).lower() and "System" not in str(e):
            raise
        merged, sys_text = [], ""
        for msg in messages:
            if msg["role"] == "system":
                sys_text = msg["content"]
            elif msg["role"] == "user" and sys_text:
                merged.append({"role": "user",
                               "content": sys_text + "\n\n" + msg["content"]})
                sys_text = ""
            else:
                merged.append(msg)
        return tokenizer.apply_chat_template(
            merged, tokenize=False, add_generation_prompt=add_generation_prompt,
            **extra_kwargs)


def _seq_logp(model, ids_full, prefix_len, device):
    """Sum logp of tokens ids_full[prefix_len:] under teacher forcing."""
    with torch.no_grad():
        out = model(ids_full.to(device), attention_mask=torch.ones_like(ids_full).to(device))
    logits = out.logits[0]
    # logits[t-1] predicts ids_full[t]; we want sum over t in [prefix_len, len)
    lp = torch.log_softmax(logits[prefix_len - 1: -1, :].float(), dim=-1)
    tgt = ids_full[0, prefix_len:].to(lp.device)
    return lp.gather(1, tgt.unsqueeze(1)).sum().item()


def label_margin(model, tokenizer, prompt_str, device):
    base_str = prompt_str + "Action:"
    base_ids = tokenizer.encode(base_str, return_tensors="pt", add_special_tokens=False)
    n_base   = base_ids.shape[1]

    s_str = base_str + SEARCH_CONT
    f_str = base_str + FINAL_CONT
    s_ids = tokenizer.encode(s_str, return_tensors="pt", add_special_tokens=False)
    f_ids = tokenizer.encode(f_str, return_tensors="pt", add_special_tokens=False)
    lp_s = _seq_logp(model, s_ids, n_base, device)
    lp_f = _seq_logp(model, f_ids, n_base, device)

    # First-token margin: logP(first tok of "Action" | prompt) - logP(first tok of "Final" | prompt).
    a_first = tokenizer.encode("Action", add_special_tokens=False)[0]
    f_first = tokenizer.encode("Final",  add_special_tokens=False)[0]
    p_ids = tokenizer.encode(prompt_str, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        out = model(p_ids, attention_mask=torch.ones_like(p_ids))
    lp0 = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
    return {
        "margin_label":       lp_s - lp_f,
        "lp_search_seq":      lp_s,
        "lp_final_seq":       lp_f,
        "margin_first_token": lp0[a_first].item() - lp0[f_first].item(),
        "lp_Action":          lp0[a_first].item(),
        "lp_Final":           lp0[f_first].item(),
    }


def run_one(rec, model, tokenizer, device, max_new_tokens=256, is_r1=False,
            prompt_variant="v1", obs_style="factcard"):
    messages = build_messages(rec["question"], rec["obs"],
                              prompt_variant=prompt_variant, obs_style=obs_style)
    prompt_str = apply_chat_template_safe(tokenizer, messages,
                                          add_generation_prompt=True)

    # margin_label / margin_first_token are measured at the prompt boundary.
    # For R1 the prompt ends inside <think>, so these are NOT a valid action-
    # preference signal there (see CLAUDE.md §4.9 #7). margin_post (computed
    # below for R1 only) is the correct decision-point margin.
    margins = label_margin(model, tokenizer, prompt_str, device)

    p_ids = tokenizer.encode(prompt_str, return_tensors="pt",
                             add_special_tokens=False).to(device)
    attn = torch.ones_like(p_ids)
    with torch.no_grad():
        gen = model.generate(p_ids, attention_mask=attn,
                             max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    gen_token_ids = gen[0, p_ids.shape[1]:].tolist()
    raw = tokenizer.decode(gen[0, p_ids.shape[1]:], skip_special_tokens=True)

    margin_post = None
    think_len = -1
    text_for_parse = raw
    if is_r1:
        think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
        a_first = tokenizer.encode(" Action", add_special_tokens=False)[0]
        f_first = tokenizer.encode(" Final",  add_special_tokens=False)[0]
        a_first_nl = tokenizer.encode("Action", add_special_tokens=False)[0]
        f_first_nl = tokenizer.encode("Final",  add_special_tokens=False)[0]
        for i, tid in enumerate(gen_token_ids):
            if tid == think_end_id:
                think_len = i
                # CLAUDE.md \u00a74.9 #7: token at think_pos+1 is \n / \n\n (noise).
                # Scan forward for the first non-whitespace token; that is the
                # decision token. Compute logits at the position generating it.
                dec_pos = None
                for j in range(i + 1, len(gen_token_ids)):
                    tok_str = tokenizer.decode([gen_token_ids[j]])
                    if tok_str.strip() != "":
                        dec_pos = j
                        break
                if dec_pos is not None:
                    prefix = gen[0, :p_ids.shape[1] + dec_pos].unsqueeze(0)
                    with torch.no_grad():
                        out = model(prefix, attention_mask=torch.ones_like(prefix))
                    lp = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
                    a_lp = max(lp[a_first].item(), lp[a_first_nl].item())
                    f_lp = max(lp[f_first].item(), lp[f_first_nl].item())
                    margin_post = a_lp - f_lp
                break
        text_for_parse = strip_think(raw)

    parsed = parse_action(text_for_parse)
    a2, fa = parsed["action"], parsed["final_answer"]
    pf = (a2 is None and fa is None)
    if a2 and a2.lower() in ("search", "calculator"):
        action_type = "search"
    elif fa is not None:
        action_type = "stop"
    else:
        action_type = None

    fa_first = first_action_token(text_for_parse)

    W = rec.get("candidate_W") or rec.get("W") or ""
    contains_W = int(bool(W) and W.lower() in (fa or "").lower())
    em = None
    if fa is not None and rec.get("gold_answer"):
        gold = rec.get("gold_answers") or [rec["gold_answer"]]
        em = int(answer_scorer(fa, gold, mode="exact")["matched"])

    raw_cap = 4000 if is_r1 else 400
    return {
        "sample_id": rec["sample_id"], "schema_type": rec.get("schema_type"),
        "condition": rec.get("condition"), "candidate_W": W,
        "prompt_variant": prompt_variant, "obs_style": obs_style,
        **margins,
        "margin_post": margin_post, "think_len": think_len,
        "action_type": action_type, "first_action_token": fa_first,
        "action2": a2, "final_answer": fa,
        "em": em, "contains_W": contains_W, "parse_failure": pf,
        "raw_output": raw[:raw_cap],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs",      default="results/extractability_support_toggle/pairs.jsonl")
    ap.add_argument("--out",        default=None,
                    help="Output JSONL path (single-config mode).")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--conditions", nargs="+", default=["N0", "T0", "S0"])
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="Default 256 for non-R1, 1200 for R1.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompt-variant", choices=["v1", "v2", "v3"], default="v1",
                    help="Robustness A: paraphrase of the user-message tail.")
    ap.add_argument("--obs-style", choices=["factcard", "snippet"], default="factcard",
                    help="Robustness B: 'factcard' (default) or 'snippet' (natural prose).")
    ap.add_argument("--multi-configs", nargs="+", default=None,
                    help="Run multiple (variant,style) configs in one model load. "
                         "Format: 'variant:style:OUT_PATH'. Overrides --out / "
                         "--prompt-variant / --obs-style.")
    args = ap.parse_args()

    is_r1 = is_r1_model(args.model_path)
    if args.max_new_tokens is None:
        args.max_new_tokens = 1200 if is_r1 else 256

    rows = [json.loads(l) for l in open(args.pairs)]
    rows = [r for r in rows if (r.get("condition") or r.get("condition_id")) in args.conditions]
    if args.limit: rows = rows[:args.limit]

    if args.multi_configs:
        configs = []
        for spec in args.multi_configs:
            v, s, p = spec.split(":", 2)
            configs.append((v, s, p))
    else:
        if not args.out:
            ap.error("--out required when --multi-configs not given")
        configs = [(args.prompt_variant, args.obs_style, args.out)]

    print(f"[info] loading {args.model_path}; {len(rows)} records; conds={args.conditions}; "
          f"is_r1={is_r1}; max_new_tokens={args.max_new_tokens}; "
          f"n_configs={len(configs)}")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True); model.eval()
    device = next(model.parameters()).device

    for variant, style, out_path in configs:
        out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True); t0 = time.time()
        print(f"[cfg] variant={variant} style={style} -> {out_path}")
        with open(out, "w") as f:
            for i, rec in enumerate(rows, 1):
                row = run_one(rec, model, tok, device, args.max_new_tokens, is_r1=is_r1,
                              prompt_variant=variant, obs_style=style)
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
                if i % 10 == 0 or i == len(rows):
                    print(f"  [{i}/{len(rows)}] {time.time()-t0:.1f}s")
        print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
