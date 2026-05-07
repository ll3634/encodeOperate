#!/usr/bin/env python3
"""
Multi-turn ReAct SQL data-analysis runner.

Mirrors run_nonqa_react_codesearch.py for a SQL data-analysis surface where
the agent uses `execute_query` (a computation tool, not a search tool) to
answer a business analytics question. The prefilled scratchpad already
issued one SQL query; in T0 the prefilled SQL computes the WRONG metric and
the result table has a plausible rank-1 entity W; the correct answer is a
different entity reachable only by a different query.

Decision point (p0):
  first_is_action  : continues (refines query / runs another SQL)
  first_is_final   : commits Final Answer (entity name)
  commit_W         : commits to W (the trap entity in T0; correct in S0)
  commit_correct   : commits to the true answer (correct entity)
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
sys.path.insert(0, str(Path(__file__).parent.parent))

from nonqa_react_sql_items import build_items, SQLItem
from parse_nonqa_react_sql_outputs import parse_output
from run_nonqa_react_meeting import (
    load_model, generate_one, write_jsonl, normalize_messages_for_model,
    apply_template_for_completion,
)

import numpy as np


def generate_one_steered(model, tokenizer, messages, max_new_tokens: int,
                         direction, alpha: float, layer: int,
                         model_name: str = "") -> str:
    """Generate with decision-only A3 steering at position=-1, max_interventions=1."""
    import torch
    from steering.hook_utils import SteeringHook
    text = apply_template_for_completion(tokenizer, messages, model_name)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with SteeringHook(model, direction, alpha, layer=layer,
                      position=-1, mode="addition", max_interventions=1):
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


def generate_one_kvgroup(model, tokenizer, messages, max_new_tokens,
                         layer: int, kv_group: int, alpha: float,
                         model_name: str = ""):
    """Generate with decision-only KV-group scaling on attn_L{layer}."""
    import torch
    from steering.hook_utils import KVGroupScalingHook
    text = apply_template_for_completion(tokenizer, messages, model_name)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with KVGroupScalingHook(model, layer=layer, kv_group=kv_group,
                            alpha=alpha, max_interventions=1):
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


SYSTEM_PROMPT = """You are a data analyst assistant. You answer business questions by writing and executing SQL queries against company databases.

Available tools:
- execute_query(sql): Run a SQL query against the analytics database and return the result table.

You MUST respond in exactly one of the following formats.

If you need to use a tool:
Action: execute_query
Action Input: <SQL>

If you can answer directly:
Final Answer: <answer>

Important:
- Make sure your query computes exactly the metric asked for.
- If the results don't directly answer the question, refine your query.
- Do not guess. If the data is insufficient or the metric is wrong, query again.

Do NOT write "Thought:" and do NOT output any other text before the first word of your response.
Your first word must be either "Action" or "Final"."""


USER_PROMPT_TEMPLATE = "{question}\n\n{schema}"


CONDITIONS = ["N0", "T0", "IC", "S0"]


def w_for_condition(item: SQLItem, condition: str) -> str:
    return {
        "N0": item.W_entity,
        "T0": item.W_entity,
        "IC": item.W_entity,
        "S0": item.correct_entity,
    }[condition]


def build_user(item: SQLItem) -> str:
    return USER_PROMPT_TEMPLATE.format(question=item.question, schema=item.schema)


def build_assistant_scratchpad(item: SQLItem, condition: str) -> str:
    obs_attr = {"N0": item.obs_N0, "T0": item.obs_T0,
                "IC": item.obs_IC, "S0": item.obs_S0}
    # In S0 the prefilled SQL is the correct one; in N0/T0/IC it is the trap.
    sql = item.correct_sql if condition == "S0" else item.trap_sql
    return (
        f"Action: execute_query\n"
        f"Action Input: {sql}\n"
        f"Observation: {obs_attr[condition]}"
    )



def build_messages(item: SQLItem, condition: str, mode: str = "prefilled"):
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
    name = f"{ts}_react_sql_{suffix}"
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
    p.add_argument("--max_new_tokens", type=int, default=220)
    p.add_argument("--mode", choices=["prefilled", "single_shot"], default="prefilled")
    p.add_argument("--seed", type=int, default=20260501)
    p.add_argument("--reuse_raw", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--tag", default="")
    p.add_argument("--steer_direction", type=str, default=None)
    p.add_argument("--steer_layer", type=int, default=20)
    p.add_argument("--steer_rho", type=float, default=0.0)
    p.add_argument("--steer_key", type=str, default="decision_direction")
    p.add_argument("--hidden_rms", type=float, default=0.65)
    p.add_argument("--kv_layer", type=int, default=18)
    p.add_argument("--kv_group", type=int, default=2)
    p.add_argument("--kv_alpha", type=float, default=1.0)
    args = p.parse_args()

    base_results = Path(__file__).parent.parent / "results" / "nonqa_react_sql"
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
            msgs = normalize_messages_for_model(msgs, args.model)
            text = apply_template_for_completion(tok, msgs, args.model)
            W_s = w_for_condition(it, cond)
            print(f"========= condition {cond}  W={W_s} =========")
            print(text)
            print()
        return

    use_steering = args.steer_direction is not None and abs(args.steer_rho) > 1e-8
    use_kv_scaling = abs(args.kv_alpha - 1.0) > 1e-8
    if use_steering and use_kv_scaling:
        raise ValueError("Use either --steer_rho or --kv_alpha, not both.")
    direction = None
    alpha = 0.0
    direction_rms = None
    if use_steering:
        d = np.load(args.steer_direction, allow_pickle=True)
        direction = np.asarray(d[args.steer_key], dtype=np.float32)
        direction_rms = float(np.sqrt(np.mean(direction ** 2)))
        alpha = float(args.steer_rho) * (args.hidden_rms / direction_rms)
        print(f"[steer] dir={args.steer_direction} key={args.steer_key} "
              f"layer={args.steer_layer} rho={args.steer_rho:+.3f} "
              f"alpha={alpha:+.4f} dir_rms={direction_rms:.4f}")
    if use_kv_scaling:
        print(f"[kv] L{args.kv_layer} kv_group={args.kv_group} "
              f"alpha={args.kv_alpha:.3f} (decision-only, max_interventions=1)")

    auto_tag = args.tag
    if args.mode != "prefilled":
        auto_tag = f"{args.tag}_{args.mode}" if args.tag else args.mode
    if use_steering:
        sign = "neg" if args.steer_rho < 0 else "pos"
        rho_tag = f"steer_L{args.steer_layer}_{sign}{abs(args.steer_rho):.3f}".replace(".", "p")
        auto_tag = f"{auto_tag}_{rho_tag}" if auto_tag else rho_tag
    if use_kv_scaling:
        kv_tag = f"kvL{args.kv_layer}_g{args.kv_group}_a{args.kv_alpha:.2f}".replace(".", "p")
        auto_tag = f"{auto_tag}_{kv_tag}" if auto_tag else kv_tag
    out_dir = Path(args.output_dir) if args.output_dir else \
        make_run_dir(base_results, n_items, args.pilot, auto_tag)
    print(f"[run] output dir: {out_dir}  (mode={args.mode})")

    config = {
        "scenario": "react_sql",
        "mode": args.mode, "pilot": args.pilot,
        "n_items": n_items, "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0, "top_p": 1.0, "seed": args.seed,
        "conditions": CONDITIONS,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "steer_direction": args.steer_direction, "steer_key": args.steer_key,
        "steer_layer": args.steer_layer, "steer_rho": args.steer_rho,
        "steer_alpha": alpha, "steer_direction_rms": direction_rms,
        "steer_hidden_rms": args.hidden_rms,
        "kv_layer": args.kv_layer, "kv_group": args.kv_group,
        "kv_alpha": args.kv_alpha, "use_kv_scaling": use_kv_scaling,
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
                "W_entity": it.W_entity,
                "correct_entity": it.correct_entity,
                "template_key": it.template_key,
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
                if use_steering:
                    raw = generate_one_steered(
                        model, tok, row["messages"], args.max_new_tokens,
                        direction=direction, alpha=alpha, layer=args.steer_layer,
                        model_name=args.model,
                    )
                elif use_kv_scaling:
                    raw = generate_one_kvgroup(
                        model, tok, row["messages"], args.max_new_tokens,
                        layer=args.kv_layer, kv_group=args.kv_group, alpha=args.kv_alpha,
                        model_name=args.model,
                    )
                else:
                    raw = generate_one(model, tok, row["messages"], args.max_new_tokens, args.model)
                rec = {
                    "item_id": row["item_id"],
                    "condition": row["condition"],
                    "W_str": row["W_str"],
                    "W_entity": row["W_entity"],
                    "correct_entity": row["correct_entity"],
                    "template_key": row["template_key"],
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
        po = parse_output(r["raw"], r["W_entity"], r.get("correct_entity"))
        d = po.to_dict()
        d.update({
            "item_id": r["item_id"],
            "condition": r["condition"],
            "W_str": r["W_str"],
            "W_entity": r.get("W_entity"),
            "correct_entity": r.get("correct_entity"),
            "template_key": r.get("template_key"),
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
        def _frac(field):
            return sum(r[field] for r in rows) / n if n else 0.0
        summary["by_condition"][c] = {
            "n": n,
            "first_is_final_rate": _frac("first_is_final"),
            "first_is_action_rate": _frac("first_is_action"),
            "commit_W_rate": _frac("commit_W"),
            "commit_correct_rate": _frac("commit_correct"),
            "final_present_rate": _frac("final_present"),
            "commit_W_anywhere_rate": _frac("commit_W_anywhere"),
            "commit_correct_anywhere_rate": _frac("commit_correct_anywhere"),
            "hallucinated_observation_rate": _frac("hallucinated_observation"),
            "parse_failure_rate": _frac("parse_failure"),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[summary]")
    for c, s in summary["by_condition"].items():
        print(f"  {c}: first_is_final={s['first_is_final_rate']:.1%}  "
              f"commit_W={s['commit_W_rate']:.1%}  "
              f"commit_correct={s['commit_correct_rate']:.1%}  "
              f"action(p0)={s['first_is_action_rate']:.1%}  "
              f"hallu={s['hallucinated_observation_rate']:.1%}  "
              f"pf={s['parse_failure_rate']:.1%}")
    print(f"\n[run] artifacts at: {out_dir}")


if __name__ == "__main__":
    main()
