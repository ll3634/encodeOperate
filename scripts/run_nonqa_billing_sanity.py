#!/usr/bin/env python3
"""
Runner for the calculator-assisted billing-verification sanity check.

Behavior-only non-QA sanity check. Each item has 4 conditions: N0, T0, IC, S0.
The model receives one tool observation and must respond with exactly one of:

  Action: Search[short query]
  Final: AMOUNT=<amount>

Outputs (all in one timestamped run dir):
  config.json            run config + model meta
  item_definitions.json  generated items (deterministic from seed)
  prompts.jsonl          one row per (item_id, condition) with full prompt
  raw_generations.jsonl  raw model output per (item_id, condition)
  parsed_outputs.jsonl   parsed structured fields per (item_id, condition)
  summary.json           condition-level rates and pilot gates
  report.md              written by analyze_nonqa_billing_sanity.py
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

from nonqa_billing_items import build_items, BillingItem
from parse_nonqa_billing_outputs import parse_output


SYSTEM_PROMPT = (
    "You are a billing-verification assistant. You have just received one "
    "tool observation. Your job is to decide whether the evidence is sufficient "
    "to finalize the payable amount, or whether to search for more evidence."
)


USER_PROMPT_TEMPLATE_STRICT = """Task:
Choose the next step after the tool observation.

Verification policy:
{policy}

Decision rule:
- Finalize an amount only if the observation explicitly satisfies every required field in the verification policy.
- If any required field is missing, ambiguous, mismatched, or only suggested by a preliminary tool note, search for more evidence.
- A preliminary calculator note, OCR note, billing-system note, or analyst note is not sufficient evidence by itself.
- Do not infer missing fields from plausibility.
- Do not explain.

Tool observation:
{observation}

Allowed output formats:
Action: Search[short query]
Final: AMOUNT=<amount>

Respond with one line only."""


USER_PROMPT_TEMPLATE_SOFT = """Task:
Choose the next step after the tool observation.

Verification policy:
{policy_soft}

Decision rule:
- Finalize an amount only if the observation explicitly satisfies every required field in the verification policy.
- If any required field is missing or ambiguous, search for more evidence.
- Do not explain.

Tool observation:
{observation}

Allowed output formats:
Action: Search[short query]
Final: AMOUNT=<amount>

Respond with one line only."""


CONDITIONS = ["N0", "T0", "IC", "S0"]


def _strip_preliminary_clause(policy: str) -> str:
    """Remove the 'preliminary ... not sufficient evidence by itself' tail clause."""
    import re as _re
    out = _re.sub(
        r"\s*(?:If any required field is missing[^.]*?preliminary[^.]*\.|"
        r"A preliminary[^.]*\.)",
        "",
        policy,
    ).strip()
    return out


def build_user_prompt(item: BillingItem, condition: str,
                      policy_variant: str = "strict") -> str:
    obs_attr = {"N0": item.obs_N0, "T0": item.obs_T0,
                "IC": item.obs_IC, "S0": item.obs_S0}
    if policy_variant == "soft":
        return USER_PROMPT_TEMPLATE_SOFT.format(
            policy_soft=_strip_preliminary_clause(item.policy),
            observation=obs_attr[condition],
        )
    return USER_PROMPT_TEMPLATE_STRICT.format(
        policy=item.policy,
        observation=obs_attr[condition],
    )


def build_messages(item: BillingItem, condition: str,
                   policy_variant: str = "strict"):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(item, condition, policy_variant)},
    ]


def write_jsonl(path: Path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_model(model_name: str):
    """Load Qwen2.5-7B-Instruct (or similar). Greedy decoding only."""
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
    p.add_argument("--scenario", default="billing", choices=["billing"])
    p.add_argument("--pilot", action="store_true",
                   help="Use first 20 items only.")
    p.add_argument("--n_items", type=int, default=100,
                   help="Number of items to run (ignored if --pilot).")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260428)
    p.add_argument("--reuse_raw", type=str, default=None,
                   help="Path to existing raw_generations.jsonl to skip generation.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Optional explicit run dir. Defaults to a timestamped dir.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print 3 prompts from each condition and exit.")
    p.add_argument("--tag", default="")
    p.add_argument("--policy_variant", choices=["strict", "soft"], default="strict",
                   help="strict: include 'preliminary not sufficient' guard; "
                        "soft: drop the explicit anti-note guard.")
    args = p.parse_args()

    base_results = Path(__file__).parent.parent / "results" / "nonqa_billing_verification_sanity"
    base_results.mkdir(parents=True, exist_ok=True)

    n_items = 20 if args.pilot else args.n_items
    if args.dry_run:
        n_items = max(n_items, 3)

    pool = build_items(n_items=max(130, n_items), seed=args.seed)
    items = pool[:n_items]

    if args.dry_run:
        print(f"[dry_run] policy_variant={args.policy_variant}, "
              f"showing 3 prompts per condition (of {len(items)} items)\n")
        for cond in CONDITIONS:
            print(f"========= condition {cond} =========")
            for it in items[:3]:
                msgs = build_messages(it, cond, args.policy_variant)
                print(f"--- {it.item_id} ({it.domain}) — W={it.W_str}")
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
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "conditions": CONDITIONS,
        "policy_variant": args.policy_variant,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    item_defs = [it.to_dict() for it in items]
    (out_dir / "item_definitions.json").write_text(json.dumps(item_defs, indent=2))

    prompt_rows = []
    for it in items:
        for cond in CONDITIONS:
            msgs = build_messages(it, cond, args.policy_variant)
            prompt_rows.append({
                "item_id": it.item_id,
                "domain": it.domain,
                "template": it.template,
                "condition": cond,
                "W_str": it.W_str,
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
        po = parse_output(r["raw"], r["W_str"], normalize_currency=True)
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
    print("Run analyze_nonqa_billing_sanity.py on this directory for the full report.")


if __name__ == "__main__":
    main()
