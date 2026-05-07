#!/usr/bin/env python3
"""Path B: Mistral targeted re-run with canonical §5.4 hook logging
+ margin-stratified design. Tests A4 saturation hypothesis.

One model load. Pass A: collect baseline T0 margins on 60 SQL + 60 CS.
Pass B: ρ=-0.60 sweep on lowest-|margin| 20 items × 4 conditions per task,
with PATH_B_LOG_HOOK=1 enabling §5.4 cos/norm capture.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from steering.hook_utils import SteeringHook, get_model_layers
from agent.prompts import ACTION_TOKENS
from run_nonqa_react_meeting import (
    load_model, normalize_messages_for_model, apply_template_for_completion,
)
from scripts.patch_L20_localise_full_residual import margin_from_logits
from scripts.attack3_calibrate_a3 import make_margin_ids, hidden_rms_at_last

import nonqa_react_sql_items as sql_items
import nonqa_react_codesearch_items as cs_items
import run_nonqa_react_sql as sql_runner
import run_nonqa_react_codesearch as cs_runner
from parse_nonqa_react_sql_outputs import parse_output as parse_sql_output
from parse_nonqa_react_codesearch_outputs import parse_output as parse_cs_output


CONDITIONS = ["N0", "T0", "IC", "S0"]


def w_for_condition_sql(it, cond):
    return sql_runner.w_for_condition(it, cond)


def w_for_condition_cs(it, cond):
    return {"N0": it.legacy_path, "T0": it.legacy_path,
            "IC": it.legacy_path, "S0": it.canonical_path}[cond]


def build_msgs_sql(it, cond, model_path):
    msgs = sql_runner.build_messages(it, cond, mode="prefilled")
    return normalize_messages_for_model(msgs, model_path)


def build_msgs_cs(it, cond, model_path):
    msgs = cs_runner.build_messages(it, cond, mode="prefilled")
    return normalize_messages_for_model(msgs, model_path)


def parse_sql(raw, it, cond):
    W = w_for_condition_sql(it, cond)
    return parse_sql_output(raw, W, it.correct_entity).to_dict()


def parse_cs(raw, it, cond):
    W = w_for_condition_cs(it, cond)
    return parse_cs_output(raw, W).to_dict()


def margin_no_hook(model, tok, prompt, tool_ids, fin_ids):
    ids = tok.encode(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0, -1, :]
    return margin_from_logits(logits, tool_ids, fin_ids)


def steered_generate(model, tok, prompt, tool_ids, fin_ids,
                     direction, alpha, layer, max_new_tokens, item_id):
    ids = tok.encode(prompt, return_tensors="pt").to(model.device)
    with SteeringHook(model, direction, alpha, layer=layer,
                      position=-1, mode="addition", max_interventions=1) as hook:
        hook._current_item_id = item_id
        with torch.no_grad():
            out = model.generate(
                input_ids=ids,
                max_new_tokens=max_new_tokens, do_sample=False,
                temperature=1.0, top_p=1.0,
                pad_token_id=tok.eos_token_id,
                output_scores=True, return_dict_in_generate=True,
            )
        log_entries = list(hook._path_b_log)
    first_logits = out.scores[0][0]  # (vocab,)
    post_margin = margin_from_logits(first_logits, tool_ids, fin_ids)
    new_tokens = out.sequences[0][ids.shape[1]:]
    text = tok.decode(new_tokens, skip_special_tokens=True)
    return post_margin, text, log_entries


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def stratify(rows, n=20):
    """Pick lowest-|margin| n items; return list of item_ids and their margins."""
    sorted_rows = sorted(rows, key=lambda r: abs(r["baseline_margin"]))
    top = sorted_rows[:n]
    return top


def pass_a_for_task(model, tok, model_path, task, items, tool_ids, fin_ids, out_dir):
    """Collect baseline T0 margin for each item. No hook. Single forward."""
    print(f"[pass A] {task} N={len(items)} ...", flush=True)
    rows = []
    t0 = time.time()
    builder = build_msgs_sql if task == "sql" else build_msgs_cs
    for i, it in enumerate(items):
        msgs = builder(it, "T0", model_path)
        prompt = apply_template_for_completion(tok, msgs, model_path)
        m = margin_no_hook(model, tok, prompt, tool_ids, fin_ids)
        rows.append({
            "item_id": it.item_id,
            "task": task,
            "baseline_margin": m,
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(items)}] mean_so_far={np.mean([r['baseline_margin'] for r in rows]):+.3f}", flush=True)
    write_jsonl(out_dir / f"baseline_margins_{task}.jsonl", rows)
    margins = np.array([r["baseline_margin"] for r in rows])
    print(f"[pass A] {task} done in {time.time()-t0:.1f}s  mean={margins.mean():+.3f} std={margins.std():.3f} "
          f"min={margins.min():+.3f} max={margins.max():+.3f}", flush=True)
    return rows



def pass_b_for_task(model, tok, model_path, task, items_full, baseline_rows,
                    tool_ids, fin_ids, direction, hidden_rms, direction_rms,
                    rho, layer, max_new_tokens, out_dir, n_strat=20):
    """Sweep ρ on lowest-|margin| n_strat items × 4 conditions. Hook + log."""
    strat = stratify(baseline_rows, n=n_strat)
    strat_ids = {r["item_id"] for r in strat}
    strat_meta = [{"item_id": r["item_id"], "baseline_margin": r["baseline_margin"]}
                  for r in strat]
    n_low = sum(1 for r in strat if abs(r["baseline_margin"]) < 6)
    print(f"[stratify] {task} chose {len(strat)} items; "
          f"|m|<6: {n_low}/{len(strat)}", flush=True)
    json.dump({"task": task, "n_strat": len(strat), "n_low_margin": n_low,
               "items": strat_meta},
              open(out_dir / f"stratified_items_{task}.json", "w"), indent=2)

    items_by_id = {it.item_id: it for it in items_full}
    builder = build_msgs_sql if task == "sql" else build_msgs_cs
    parser = parse_sql if task == "sql" else parse_cs
    base_margin_by_id = {r["item_id"]: r["baseline_margin"] for r in baseline_rows}

    alpha = float(rho) * (hidden_rms / direction_rms)
    print(f"[pass B] {task} rho={rho:+.3f} alpha={alpha:+.4f} layer={layer}", flush=True)

    sweep_rows, hook_rows = [], []
    t0 = time.time()
    n_done = 0
    for r in strat:
        it = items_by_id[r["item_id"]]
        for cond in CONDITIONS:
            msgs = builder(it, cond, model_path)
            prompt = apply_template_for_completion(tok, msgs, model_path)
            tag = f"{it.item_id}|{cond}"
            post_margin, text, log_entries = steered_generate(
                model, tok, prompt, tool_ids, fin_ids,
                direction, alpha, layer, max_new_tokens, tag,
            )
            parsed = parser(text, it, cond)
            sweep_rows.append({
                "item_id": it.item_id,
                "condition": cond,
                "baseline_margin_T0": base_margin_by_id.get(it.item_id),
                "post_margin": post_margin,
                "shift": post_margin - (base_margin_by_id.get(it.item_id) or 0.0),
                "first_is_action": int(parsed.get("first_is_action", 0)),
                "first_is_final": int(parsed.get("first_is_final", 0)),
                "parse_failure": int(parsed.get("parse_failure", 0)),
                "first_line": parsed.get("first_line", ""),
                "raw": text,
            })
            for le in log_entries:
                le["task"] = task
                le["condition"] = cond
                le["item_id"] = it.item_id
                hook_rows.append(le)
            n_done += 1
            if n_done % 10 == 0:
                print(f"  [{n_done}/{len(strat)*len(CONDITIONS)}] "
                      f"last shift={sweep_rows[-1]['shift']:+.3f} "
                      f"post_m={post_margin:+.3f} "
                      f"flip={1 if sweep_rows[-1]['first_is_action'] else 0}",
                      flush=True)
    write_jsonl(out_dir / f"sweep_{task}_rho{rho:+.2f}.jsonl", sweep_rows)
    write_jsonl(out_dir / f"hook_verification_{task}.jsonl", hook_rows)
    print(f"[pass B] {task} done in {time.time()-t0:.1f}s "
          f"({len(sweep_rows)} rows, {len(hook_rows)} hook logs)", flush=True)
    return sweep_rows, hook_rows, strat_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--direction-npz",
                    default="results/mistral_circuit_sanity/exp2_steering/directions.npz")
    ap.add_argument("--direction-key", default="action_dir")
    ap.add_argument("--act-layer", type=int, default=28)
    ap.add_argument("--rho", type=float, default=-0.60)
    ap.add_argument("--n-baseline", type=int, default=60)
    ap.add_argument("--n-strat", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--seed", type=int, default=20260501)
    ap.add_argument("--out-dir",
                    default="results/attack3_closure/_audit/B_mistral_canonical")
    ap.add_argument("--skip-pass-a", action="store_true",
                    help="Reuse baseline_margins_*.jsonl from out-dir")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out_dir / "config.json", "w"), indent=2)

    print(f"[run] {args.model}  layer={args.act_layer}  rho={args.rho}  "
          f"out={out_dir}", flush=True)

    model, tok = load_model(args.model)
    device = next(model.parameters()).device
    print(f"[run] device={device} dtype={next(model.parameters()).dtype} "
          f"attn={model.config._attn_implementation}", flush=True)
    tool_ids, fin_ids = make_margin_ids(tok)
    print(f"[run] tool_ids={tool_ids} fin_ids={fin_ids}", flush=True)

    d = np.load(args.direction_npz, allow_pickle=True)
    direction = np.asarray(d[args.direction_key], dtype=np.float32)
    direction_rms = float(np.sqrt(np.mean(direction ** 2)))
    print(f"[dir] {args.direction_npz}::{args.direction_key} "
          f"rms={direction_rms:.6f} dim={direction.shape}", flush=True)

    sql_pool = sql_items.build_items(n_items=args.n_baseline, seed=args.seed)
    cs_pool = cs_items.build_items(n_items=args.n_baseline, seed=args.seed)
    print(f"[items] sql={len(sql_pool)} cs={len(cs_pool)}", flush=True)

    # ---- Pass A ----
    if args.skip_pass_a:
        sql_base = [json.loads(l) for l in open(out_dir / "baseline_margins_sql.jsonl")]
        cs_base = [json.loads(l) for l in open(out_dir / "baseline_margins_codesearch.jsonl")]
        print(f"[pass A] skipped; reused {len(sql_base)} sql, {len(cs_base)} cs", flush=True)
    else:
        sql_base = pass_a_for_task(model, tok, args.model, "sql", sql_pool,
                                   tool_ids, fin_ids, out_dir)
        cs_base = pass_a_for_task(model, tok, args.model, "codesearch", cs_pool,
                                  tool_ids, fin_ids, out_dir)

    # Sanity: mean baseline margin should approximately match A4's −4.74
    sql_mean = float(np.mean([r["baseline_margin"] for r in sql_base]))
    cs_mean = float(np.mean([r["baseline_margin"] for r in cs_base]))
    a4_ref = -4.7437
    sql_match = abs(sql_mean - a4_ref) <= 2.0
    cs_match = abs(cs_mean - a4_ref) <= 2.0
    print(f"[sanity] sql_mean={sql_mean:+.3f} cs_mean={cs_mean:+.3f} "
          f"a4_ref={a4_ref:+.3f}  sql_match={sql_match} cs_match={cs_match}",
          flush=True)
    if not sql_match:
        print(f"[sanity] WARN: SQL baseline mean diverges >2.0 from A4. Continuing.", flush=True)
    # Use measured Mistral hidden_rms from calibration (avoids re-measuring)
    cal_path = ROOT / "results/attack3_closure/calibration/mistral_l28.json"
    cal = json.load(open(cal_path))
    hidden_rms = float(cal["hidden_rms_mean"])
    print(f"[hidden_rms] from calibration: {hidden_rms:.6f}", flush=True)

    os.environ["PATH_B_LOG_HOOK"] = "1"
    sql_sweep, sql_hook, sql_strat = pass_b_for_task(
        model, tok, args.model, "sql", sql_pool, sql_base,
        tool_ids, fin_ids, direction, hidden_rms, direction_rms,
        args.rho, args.act_layer, args.max_new_tokens, out_dir, args.n_strat,
    )
    cs_sweep, cs_hook, cs_strat = pass_b_for_task(
        model, tok, args.model, "codesearch", cs_pool, cs_base,
        tool_ids, fin_ids, direction, hidden_rms, direction_rms,
        args.rho, args.act_layer, args.max_new_tokens, out_dir, args.n_strat,
    )
    os.environ.pop("PATH_B_LOG_HOOK", None)
    print("[done] Pass A+B complete. Run path_b_analyze.py for verdicts.", flush=True)


if __name__ == "__main__":
    main()
