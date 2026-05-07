#!/usr/bin/env python3
"""
Multi-turn ReAct meeting-scheduling sanity check.

Mirrors the HotpotQA p0 prompt structure exactly: system + user(question) +
prefilled assistant scratchpad ("Action:...\nAction Input:...\nObservation:...")
+ open assistant turn (via add_generation_prompt=True). The model's first
generated decision is what we measure.

This is the structural fix for the prior 8 single-shot nulls: in single-shot
the model treats the user message as a complete spec and runs a checklist; in
the prefilled-scratchpad form (the same form the QA experiments use) the
observation reads as ambient state and the model decides whether to issue
another Action or commit Final Answer.

Outputs (one timestamped run dir):
  config.json            run config + model meta
  item_definitions.json  generated items (deterministic from seed)
  prompts.jsonl          one row per (item_id, condition) with W_str + messages
  raw_generations.jsonl  raw model output per (item_id, condition)
  parsed_outputs.jsonl   parsed structured fields per (item_id, condition)
  summary.json           condition-level rates
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent))

from nonqa_react_meeting_items import build_items, MeetingItem
from parse_nonqa_react_meeting_outputs import parse_output


SYSTEM_PROMPT = """You are a helpful assistant that answers questions using available tools.

Available tools:
- calendar(person, time-window): Look up calendar availability for a person within a time window

You MUST respond in exactly one of the following formats.

If you need to use a tool:
Action: <tool_name>
Action Input: <input>

If you can answer directly:
Final Answer: <answer>

Do NOT write "Thought:" and do NOT output any other text before the first word of your response.
Your first word must be either "Action" or "Final"."""


USER_PROMPT_TEMPLATE = (
    "Schedule a {duration_min}-minute meeting between {person_a} and "
    "{person_b} this {day} between {window}. Reply with the chosen slot."
)


CONDITIONS = ["N0", "T0", "IC", "S0"]


def w_for_condition(item: MeetingItem, condition: str) -> str:
    # Negative conditions: trap_slot is the premature-commit target.
    # S0: joint_slot is the correct commit target.
    return item.joint_slot if condition == "S0" else item.trap_slot


def build_user(item: MeetingItem) -> str:
    return USER_PROMPT_TEMPLATE.format(
        duration_min=item.duration_min,
        person_a=item.person_a,
        person_b=item.person_b,
        day=item.day,
        window=item.window,
    )


def build_assistant_scratchpad(item: MeetingItem, condition: str) -> str:
    obs_attr = {"N0": item.obs_N0, "T0": item.obs_T0,
                "IC": item.obs_IC, "S0": item.obs_S0}
    return (
        f"Action: calendar\n"
        f"Action Input: {item.person_a}, {item.day} {item.window}\n"
        f"Observation: {obs_attr[condition]}"
    )


def build_messages(item: MeetingItem, condition: str, mode: str = "prefilled"):
    """
    mode='prefilled'   : true multi-turn ReAct (system, user, prefilled assistant
                         scratchpad), then add_generation_prompt opens a new
                         assistant turn. Mirrors the QA p0 setup exactly.
    mode='single_shot' : structural ablation matching the 8 prior failed pilots
                         (the observation is collapsed into the user message,
                         no prefilled assistant turn). Used to demonstrate that
                         the multi-turn prefill is the structural ingredient
                         that unlocks the trap.
    """
    if mode == "prefilled":
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user(item)},
            {"role": "assistant",
             "content": build_assistant_scratchpad(item, condition)},
        ]
    if mode == "single_shot":
        scratch = build_assistant_scratchpad(item, condition)
        user_with_obs = (
            f"{build_user(item)}\n\n"
            f"You have already issued one tool call:\n{scratch}\n\n"
            f"Decide the next step."
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_with_obs},
        ]
    raise ValueError(f"unknown mode {mode!r}")


def write_jsonl(path: Path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _attn_impl_for(model_path: str) -> str:
    p = model_path.lower()
    if "gemma" in p:
        return "eager"
    return "sdpa"


def normalize_messages_for_model(messages, model_path: str):
    """Concat system role into first user message for non-Qwen chat templates.

    Gemma-2's chat template raises on system role; Mistral folds system into
    [INST] but interacts unpredictably with prefilled assistant scaffolds.
    Returns a new list; does not mutate input.
    """
    p = (model_path or "").lower()
    needs_concat = ("gemma" in p) or ("mistral" in p)
    if not needs_concat:
        return list(messages)
    out = []
    pending_system = None
    for m in messages:
        if m["role"] == "system":
            pending_system = m["content"]
            continue
        if m["role"] == "user" and pending_system is not None:
            merged = pending_system.rstrip() + "\n\n" + m["content"]
            out.append({"role": "user", "content": merged})
            pending_system = None
        else:
            out.append(dict(m))
    if pending_system is not None:
        out.append({"role": "user", "content": pending_system})
    return out


def load_model(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"[load_model] {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    attn_impl = _attn_impl_for(model_name)
    print(f"[load_model] attn_implementation={attn_impl}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model.eval()
    return model, tok


def apply_template_for_completion(tokenizer, messages, model_name: str) -> str:
    """Mistral's chat template closes assistant turns with </s> and has no
    semantics for "open a new assistant turn", so add_generation_prompt=True
    after a prefilled assistant message yields a prompt the model immediately
    completes with EOS. Fall back to continue_final_message=True for Mistral
    when the last role is assistant. Qwen / Gemma keep add_generation_prompt.
    """
    last_role = messages[-1]["role"] if messages else "user"
    p = (model_name or "").lower()
    if "mistral" in p and last_role == "assistant":
        return tokenizer.apply_chat_template(
            messages, tokenize=False, continue_final_message=True
        )
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_one(model, tokenizer, messages, max_new_tokens: int,
                 model_name: str = "") -> str:
    import torch
    text = apply_template_for_completion(tokenizer, messages, model_name)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def make_run_dir(base: Path, n_items: int, pilot: bool, tag: str = "") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "pilot" if pilot else f"n{n_items}"
    name = f"{ts}_react_meeting_{suffix}"
    if tag:
        name += f"_{tag}"
    out = base / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", action="store_true", help="Use first 20 items only.")
    p.add_argument("--n_items", type=int, default=60)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--mode", choices=["prefilled", "single_shot"], default="prefilled",
                   help="prefilled = multi-turn ReAct (default); "
                        "single_shot = collapse observation into user prompt "
                        "(structural ablation of the prior 8 failed pilots).")
    p.add_argument("--seed", type=int, default=20260501)
    p.add_argument("--reuse_raw", type=str, default=None,
                   help="Path to existing raw_generations.jsonl to skip generation.")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--dry_run", action="store_true",
                   help="Print 3 prompts from each condition (incl. assembled chat) and exit.")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    base_results = Path(__file__).parent.parent / "results" / "nonqa_react_meeting"
    base_results.mkdir(parents=True, exist_ok=True)

    n_items = 20 if args.pilot else args.n_items
    if args.dry_run:
        n_items = max(n_items, 3)

    pool = build_items(n_items=max(60, n_items), seed=args.seed)
    items = pool[:n_items]

    if args.dry_run:
        # Show full assembled chat for one item per condition by applying chat template.
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        print(f"[dry_run] {len(items)} items; mode={args.mode}; "
              f"showing 1 assembled chat per condition\n")
        for cond in CONDITIONS:
            it = items[0]
            msgs = build_messages(it, cond, mode=args.mode)
            msgs = normalize_messages_for_model(msgs, args.model)
            text = apply_template_for_completion(tok, msgs, args.model)
            W_s = w_for_condition(it, cond)
            print(f"========= condition {cond}  W={W_s} =========")
            print(text)
            print()
        return

    # Auto-suffix the run dir with the mode tag so prefilled and single_shot
    # runs are never confused.
    auto_tag = args.tag
    if args.mode != "prefilled":
        auto_tag = f"{args.tag}_{args.mode}" if args.tag else args.mode
    out_dir = Path(args.output_dir) if args.output_dir else \
        make_run_dir(base_results, n_items, args.pilot, auto_tag)
    print(f"[run] output dir: {out_dir}  (mode={args.mode})")

    config = {
        "scenario": "react_meeting",
        "mode": args.mode,
        "pilot": args.pilot,
        "n_items": n_items,
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": args.seed,
        "conditions": CONDITIONS,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    item_defs = [it.to_dict() for it in items]
    (out_dir / "item_definitions.json").write_text(json.dumps(item_defs, indent=2))

    prompt_rows = []
    for it in items:
        for cond in CONDITIONS:
            msgs = build_messages(it, cond, mode=args.mode)
            msgs = normalize_messages_for_model(msgs, args.model)
            W_s = w_for_condition(it, cond)
            prompt_rows.append({
                "item_id": it.item_id,
                "condition": cond,
                "W_str": W_s,
                "trap_slot": it.trap_slot,
                "joint_slot": it.joint_slot,
                "person_a": it.person_a,
                "person_b": it.person_b,
                "messages": msgs,
            })
    write_jsonl(out_dir / "prompts.jsonl", prompt_rows)

    raw_rows: List[dict] = []
    if args.reuse_raw:
        print(f"[run] reusing raw generations from {args.reuse_raw}")
        with open(args.reuse_raw) as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_rows.append(json.loads(line))
    else:
        model, tok = load_model(args.model)
        t0 = time.time()
        with open(out_dir / "raw_generations.jsonl", "w") as fout:
            for i, row in enumerate(prompt_rows):
                raw = generate_one(model, tok, row["messages"], args.max_new_tokens, args.model)
                rec = {
                    "item_id": row["item_id"],
                    "condition": row["condition"],
                    "W_str": row["W_str"],
                    "trap_slot": row["trap_slot"],
                    "joint_slot": row["joint_slot"],
                    "raw": raw,
                }
                raw_rows.append(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                if (i + 1) % 20 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / max(elapsed, 1e-6)
                    print(f"[run] {i+1}/{len(prompt_rows)} ({rate:.1f}/s, {elapsed:.0f}s)")
        print(f"[run] generation done in {time.time()-t0:.0f}s")

    parsed_rows = []
    for r in raw_rows:
        po = parse_output(r["raw"], r["W_str"])
        d = po.to_dict()
        d.update({
            "item_id": r["item_id"],
            "condition": r["condition"],
            "W_str": r["W_str"],
            "trap_slot": r.get("trap_slot"),
            "joint_slot": r.get("joint_slot"),
        })
        parsed_rows.append(d)
    write_jsonl(out_dir / "parsed_outputs.jsonl", parsed_rows)

    by_cond = {c: [] for c in CONDITIONS}
    for d in parsed_rows:
        by_cond[d["condition"]].append(d)

    summary = {"n_items": n_items, "model": args.model, "by_condition": {}}
    for c in CONDITIONS:
        rows = by_cond[c]
        n = len(rows)
        commit = sum(r["commit_W"] for r in rows)
        commit_any = sum(r["commit_W_anywhere"] for r in rows)
        act = sum(r["first_is_action"] for r in rows)
        fin = sum(r["first_is_final"] for r in rows)
        fin_any = sum(r["final_present"] for r in rows)
        hallu = sum(r["hallucinated_observation"] for r in rows)
        pf = sum(r["parse_failure"] for r in rows)
        summary["by_condition"][c] = {
            "n": n,
            "commit_W_rate": commit / n if n else 0.0,
            "commit_W_anywhere_rate": commit_any / n if n else 0.0,
            "first_is_action_rate": act / n if n else 0.0,
            "first_is_final_rate": fin / n if n else 0.0,
            "final_present_rate": fin_any / n if n else 0.0,
            "hallucinated_observation_rate": hallu / n if n else 0.0,
            "parse_failure_rate": pf / n if n else 0.0,
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[summary]")
    for c, s in summary["by_condition"].items():
        print(f"  {c}: commit_W_any={s['commit_W_anywhere_rate']:.1%}  "
              f"hallu_obs={s['hallucinated_observation_rate']:.1%}  "
              f"final_any={s['final_present_rate']:.1%}  "
              f"action(p0)={s['first_is_action_rate']:.1%}  "
              f"pf={s['parse_failure_rate']:.1%}")

    print(f"\n[run] artifacts at: {out_dir}")


if __name__ == "__main__":
    main()
