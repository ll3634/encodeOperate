#!/usr/bin/env python3
"""
Runner for the meeting-scheduling sanity check.

Each item has 4 conditions: N0 (no anchor), T0 (prior-meeting anchor),
IC (length-matched irrelevant note), S0 (sufficient: both calendars).
The model receives one tool observation and must respond with exactly one of:

  Action: GetCalendar[<person>]
  Final: SLOT=<DAY HH:MM>

System prompt is intentionally minimal: no "check both calendars" meta-rule.

Outputs (all in one timestamped run dir):
  config.json            run config + model meta
  item_definitions.json  generated items (deterministic from seed)
  prompts.jsonl          one row per (item_id, condition) with full prompt
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

from nonqa_meeting_items import build_items, MeetingItem
from parse_nonqa_meeting_outputs import parse_output


SYSTEM_PROMPT = "You are a scheduling assistant. Decide the next step."


USER_PROMPT_TEMPLATE = """Task: schedule a 30-minute 1:1 between {A} and {B} next week (Mon-Fri, 9 AM-5 PM, in their shared timezone).

Tool observation:
{observation}

Allowed output formats:
Action: GetCalendar[<person>]
Final: SLOT=<DAY HH:MM>

Respond with one line only."""


CONDITIONS = ["N0", "T0", "IC", "S0"]


def build_user_prompt(item: MeetingItem, condition: str) -> str:
    obs_attr = {"N0": item.obs_N0, "T0": item.obs_T0,
                "IC": item.obs_IC, "S0": item.obs_S0}
    return USER_PROMPT_TEMPLATE.format(
        A=item.person_A, B=item.person_B,
        observation=obs_attr[condition],
    )


def build_messages(item: MeetingItem, condition: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(item, condition)},
    ]


def write_jsonl(path: Path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_model(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"[load_model] {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def generate_one(model, tokenizer, messages, max_new_tokens: int) -> str:
    import torch
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
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


def make_run_dir(base: Path, scenario: str, n_items: int, pilot: bool, tag: str = "") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "pilot" if pilot else f"n{n_items}"
    name = f"{ts}_{scenario}_{suffix}"
    if tag:
        name += f"_{tag}"
    out = base / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="meeting", choices=["meeting"])
    p.add_argument("--pilot", action="store_true",
                   help="Use first 20 items only.")
    p.add_argument("--n_items", type=int, default=120)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260429)
    p.add_argument("--reuse_raw", type=str, default=None,
                   help="Path to existing raw_generations.jsonl to skip generation.")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--dry_run", action="store_true",
                   help="Print 3 prompts from each condition and exit.")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    base_results = Path(__file__).parent.parent / "results" / "nonqa_meeting_scheduling_sanity"
    base_results.mkdir(parents=True, exist_ok=True)

    n_items = 20 if args.pilot else args.n_items
    if args.dry_run:
        n_items = max(n_items, 3)

    pool = build_items(n_items=max(120, n_items), seed=args.seed)
    items = pool[:n_items]

    if args.dry_run:
        print(f"[dry_run] showing 3 prompts per condition (of {len(items)} items)\n")
        for cond in CONDITIONS:
            print(f"========= condition {cond} =========")
            for it in items[:3]:
                msgs = build_messages(it, cond)
                print(f"--- {it.item_id}: {it.person_A} & {it.person_B} — W={it.W_str}")
                print(f"[user]\n{msgs[1]['content']}\n")
        return

    out_dir = Path(args.output_dir) if args.output_dir else \
        make_run_dir(base_results, args.scenario, n_items, args.pilot, args.tag)
    print(f"[run] output dir: {out_dir}")

    config = {
        "scenario": args.scenario,
        "pilot": args.pilot,
        "n_items": n_items,
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
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
            msgs = build_messages(it, cond)
            prompt_rows.append({
                "item_id": it.item_id,
                "condition": cond,
                "W_str": it.W_str,
                "person_A": it.person_A,
                "person_B": it.person_B,
                "system": msgs[0]["content"],
                "user": msgs[1]["content"],
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
                msgs = [{"role": "system", "content": row["system"]},
                        {"role": "user", "content": row["user"]}]
                raw = generate_one(model, tok, msgs, args.max_new_tokens)
                rec = {
                    "item_id": row["item_id"],
                    "condition": row["condition"],
                    "W_str": row["W_str"],
                    "raw": raw,
                }
                raw_rows.append(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                if (i + 1) % 20 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / max(elapsed, 1e-6)
                    print(f"[run] {i+1}/{len(prompt_rows)} "
                          f"({rate:.1f}/s, {elapsed:.0f}s)")
        print(f"[run] generation done in {time.time()-t0:.0f}s")

    parsed_rows = []
    for r in raw_rows:
        po = parse_output(r["raw"], r["W_str"])
        d = po.to_dict()
        d.update({
            "item_id": r["item_id"],
            "condition": r["condition"],
            "W_str": r["W_str"],
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
        srch = sum(r["first_is_search"] for r in rows)
        fin = sum(r["first_is_final"] for r in rows)
        pf = sum(r["parse_failure"] for r in rows)
        summary["by_condition"][c] = {
            "n": n,
            "commit_W_rate": commit / n if n else 0.0,
            "first_is_search_rate": srch / n if n else 0.0,
            "first_is_final_rate": fin / n if n else 0.0,
            "parse_failure_rate": pf / n if n else 0.0,
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[summary]")
    for c, s in summary["by_condition"].items():
        print(f"  {c}: commit_W={s['commit_W_rate']:.1%}  "
              f"search={s['first_is_search_rate']:.1%}  "
              f"final={s['first_is_final_rate']:.1%}  "
              f"pf={s['parse_failure_rate']:.1%}")

    print(f"\n[run] artifacts at: {out_dir}")


if __name__ == "__main__":
    main()
