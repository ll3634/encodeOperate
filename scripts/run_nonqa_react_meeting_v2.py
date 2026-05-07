#!/usr/bin/env python3
"""
v2 runner for the multi-turn ReAct meeting-scheduling sanity check.

v2 isolates **premature stop at p0** (mirroring the QA main mechanism). The
prefilled scratchpad already executed a single joint-availability query whose
Observation contains a SURFACE-SUFFICIENT joint slot; the user prompt states a
constraint that the trap_slot violates and the joint_slot respects.

Primary metric: first_is_final at p0 (analog of the QA p0 stop-vs-search vote).
Secondary:      commit_W (first_is_final AND commits to the constraint-violating
                surface slot in T0; commits to the constraint-respecting slot in S0).

Outputs identical to v1 (config, item_definitions, prompts, raw_generations,
parsed_outputs, summary) under .../nonqa_react_meeting_v2/.
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

from nonqa_react_meeting_v2_items import build_items, MeetingItemV2
from parse_nonqa_react_meeting_outputs import parse_output
# Reuse generic helpers (model loading, generation, jsonl IO).
from run_nonqa_react_meeting import load_model, generate_one, write_jsonl


SYSTEM_PROMPT = """You are a helpful assistant that answers questions using available tools.

Available tools:
- find_joint_availability(person_a, person_b, day, time-window): Look up overlapping availability for two people on a given day inside a time window

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
    "{person_b} this {day} between {window}. {constraint_text} "
    "Reply with the chosen slot."
)


CONDITIONS = ["N0", "T0", "T_partial", "T_edge", "IC", "S0"]


def w_for_condition(item: MeetingItemV2, condition: str) -> str:
    return {
        "N0": item.trap_slot,        # W is the trap; commit_W=0 expected (no slot in obs)
        "T0": item.trap_slot,
        "T_partial": item.partial_slot,
        "T_edge": item.edge_slot,
        "IC": item.trap_slot,        # W is the trap; commit_W=0 expected
        "S0": item.joint_slot,
    }[condition]


def build_user(item: MeetingItemV2) -> str:
    return USER_PROMPT_TEMPLATE.format(
        duration_min=item.duration_min,
        person_a=item.person_a,
        person_b=item.person_b,
        day=item.day,
        window=item.window,
        constraint_text=item.constraint_text,
    )


def build_assistant_scratchpad(item: MeetingItemV2, condition: str) -> str:
    obs_attr = {"N0": item.obs_N0, "T0": item.obs_T0,
                "T_partial": item.obs_T_partial,
                "T_edge": item.obs_T_edge,
                "IC": item.obs_IC, "S0": item.obs_S0}
    return (
        f"Action: find_joint_availability\n"
        f"Action Input: {item.person_a}, {item.person_b}, {item.day}, {item.window}\n"
        f"Observation: {obs_attr[condition]}"
    )


def build_messages(item: MeetingItemV2, condition: str, mode: str = "prefilled"):
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


def make_run_dir(base: Path, n_items: int, pilot: bool, tag: str = "") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "pilot" if pilot else f"n{n_items}"
    name = f"{ts}_react_meeting_v2_{suffix}"
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
    p.add_argument("--mode", choices=["prefilled", "single_shot"], default="prefilled")
    p.add_argument("--seed", type=int, default=20260501)
    p.add_argument("--reuse_raw", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    base_results = Path(__file__).parent.parent / "results" / "nonqa_react_meeting_v2"
    base_results.mkdir(parents=True, exist_ok=True)

    n_items = 20 if args.pilot else args.n_items
    if args.dry_run:
        n_items = max(n_items, 3)

    pool = build_items(n_items=max(60, n_items), seed=args.seed)
    items = pool[:n_items]

    if args.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        print(f"[dry_run] {len(items)} items; mode={args.mode}; "
              f"showing 1 assembled chat per condition\n")
        for cond in CONDITIONS:
            it = items[0]
            msgs = build_messages(it, cond, mode=args.mode)
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            W_s = w_for_condition(it, cond)
            print(f"========= condition {cond}  W={W_s} =========")
            print(text)
            print()
        return


    auto_tag = args.tag
    if args.mode != "prefilled":
        auto_tag = f"{args.tag}_{args.mode}" if args.tag else args.mode
    out_dir = Path(args.output_dir) if args.output_dir else \
        make_run_dir(base_results, n_items, args.pilot, auto_tag)
    print(f"[run] output dir: {out_dir}  (mode={args.mode})")

    config = {
        "scenario": "react_meeting_v2",
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
                raw = generate_one(model, tok, row["messages"], args.max_new_tokens)
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

    summary = {"n_items": n_items, "model": args.model, "mode": args.mode,
               "by_condition": {}}
    for c in CONDITIONS:
        rows = by_cond[c]
        n = len(rows)
        summary["by_condition"][c] = {
            "n": n,
            "first_is_final_rate": sum(r["first_is_final"] for r in rows) / n if n else 0.0,
            "first_is_action_rate": sum(r["first_is_action"] for r in rows) / n if n else 0.0,
            "commit_W_rate": sum(r["commit_W"] for r in rows) / n if n else 0.0,
            "final_present_rate": sum(r["final_present"] for r in rows) / n if n else 0.0,
            "commit_W_anywhere_rate": sum(r["commit_W_anywhere"] for r in rows) / n if n else 0.0,
            "hallucinated_observation_rate": sum(r["hallucinated_observation"] for r in rows) / n if n else 0.0,
            "parse_failure_rate": sum(r["parse_failure"] for r in rows) / n if n else 0.0,
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[summary]")
    for c, s in summary["by_condition"].items():
        print(f"  {c}: first_is_final={s['first_is_final_rate']:.1%}  "
              f"commit_W={s['commit_W_rate']:.1%}  "
              f"action(p0)={s['first_is_action_rate']:.1%}  "
              f"hallu_obs={s['hallucinated_observation_rate']:.1%}  "
              f"pf={s['parse_failure_rate']:.1%}")
    print(f"\n[run] artifacts at: {out_dir}")


if __name__ == "__main__":
    main()
