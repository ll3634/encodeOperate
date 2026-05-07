#!/usr/bin/env python3
"""Steering-on-Trap Probe.

For each (record, intervention) runs:
  - margin under intervention
  - greedy generation under intervention (max_interventions=1 => first new token only)

Interventions: baseline, a3 (direction_search_v3_layer20), evidence_parallel
(direction_decomp_parallel_layer20), random (direction_random_control).

Reuses prompt / margin / parse conventions from run_anti_cue_eval.py.
"""
import argparse, json, sys, time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS, parse_action
from eval.scorers import answer_scorer
from steering.hook_utils import SteeringHook


HIDDEN_RMS = 0.65  # approximate residual RMS at L20 for Qwen-7B; consistent with existing scripts

INTERVENTIONS = {
    "baseline":          None,
    "a3":                "steering/directions/direction_search_v3_layer20.npz",
    "evidence_parallel": "steering/directions/direction_decomp_parallel_layer20.npz",
    "random":            "steering/directions/direction_random_control.npz",
    "random_s17":        "steering/directions/direction_random_seed17.npz",
    "random_s42":        "steering/directions/direction_random_seed42.npz",
    "random_s99":        "steering/directions/direction_random_seed99.npz",
}


def load_direction(path):
    d = np.load(path, allow_pickle=True)
    v = d["decision_direction"].astype(np.float32)
    return v


def compute_margin(logits, tool_ids, fin_ids):
    lp = torch.log_softmax(logits.float(), dim=-1)
    tl = torch.logsumexp(lp[tool_ids], 0).item() if tool_ids else -100.0
    fl = torch.logsumexp(lp[fin_ids],  0).item() if fin_ids  else -100.0
    return float(tl - fl)


def setup_label_tokens(tokenizer):
    """Token ids for label-score margin.

    First-position decision: Action (2512) vs Final (19357).
    After 'Action:' teacher-forced, next token: ' search' (2711) vs ' Final' (13023).
    The drift-path 'Action: Final Answer' is behaviourally a STOP, so the true
    behavioural margin must account for both standard and drift stop formats.
    """
    return {
        "id_Action_first": tokenizer.encode("Action", add_special_tokens=False)[0],
        "id_Final_first":  tokenizer.encode("Final",  add_special_tokens=False)[0],
        "action_colon_ids": tokenizer.encode("Action:", add_special_tokens=False),
        "id_search_after": tokenizer.encode(" search", add_special_tokens=False)[0],
        "id_Final_after":  tokenizer.encode(" Final",  add_special_tokens=False)[0],
    }


def compute_label_margin(model, input_ids, prompt_len, direction, alpha, lt,
                         device, layer=20):
    """Compute both first-token and behavioural (label-score) margins.

    Teacher-forces 'Action:' after the prompt so we can read the post-colon
    distribution. Steering hook fires once at position=prompt_len-1 (last prompt
    token), matching decision-only semantics used during generation.
    """
    act_ids = torch.tensor([lt["action_colon_ids"]], device=device,
                           dtype=input_ids.dtype)
    extended = torch.cat([input_ids, act_ids], dim=1)
    attn = torch.ones_like(extended)

    if direction is not None:
        ctx = SteeringHook(model, direction, alpha, layer=layer,
                           position=prompt_len - 1, max_interventions=1)
    else:
        ctx = nullcontext()
    with ctx:
        with torch.no_grad():
            out = model(extended, attention_mask=attn)

    logits = out.logits[0]
    lp0 = torch.log_softmax(logits[prompt_len - 1, :].float(), dim=-1)
    lp1 = torch.log_softmax(logits[-1, :].float(), dim=-1)

    lp_Action       = lp0[lt["id_Action_first"]].item()
    lp_Final        = lp0[lt["id_Final_first"]].item()
    lp_search_after = lp1[lt["id_search_after"]].item()
    lp_Final_after  = lp1[lt["id_Final_after"]].item()

    margin_first = lp_Action - lp_Final
    log_P_search = lp_Action + lp_search_after
    log_P_stop = float(torch.logsumexp(
        torch.tensor([lp_Final, lp_Action + lp_Final_after]), dim=0).item())
    margin_label = log_P_search - log_P_stop

    return {
        "margin_first_token": margin_first,
        "margin_label":       margin_label,
        "lp_Action":          lp_Action,
        "lp_Final":           lp_Final,
        "lp_search_after":    lp_search_after,
        "lp_Final_after":     lp_Final_after,
    }


def run_one(rec, intervention, direction, alpha, model, tokenizer, builder, device,
            tool_ids, fin_ids, label_tokens, max_new_tokens=256, skip_gen=False):
    obs = rec["obs"]
    query = f"about: {rec['question'][:80]}"
    steps = [{"action": "search", "action_input": query, "observation": obs}]
    messages = builder.build_full_prompt(rec["question"], steps)
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt_str, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    attn = torch.ones_like(input_ids)

    def _generate():
        if direction is None:
            ctx = nullcontext()
        else:
            ctx = SteeringHook(model, direction, alpha, layer=20, position=-1,
                               max_interventions=1)
        with ctx:
            with torch.no_grad():
                ids = model.generate(
                    input_ids, attention_mask=attn,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    do_sample=False,
                )
        return tokenizer.decode(ids[0, prompt_len:], skip_special_tokens=True)

    margins = compute_label_margin(model, input_ids, prompt_len, direction, alpha,
                                   label_tokens, device)

    if skip_gen:
        raw = ""
        a2, fa = None, None
        pf = False
        action_type = None
        em = None
        contains_W, contains_V = 0, 0
    else:
        raw = _generate()
        parsed = parse_action(raw)
        a2, fa = parsed["action"], parsed["final_answer"]
        pf = (a2 is None and fa is None)
        if a2 and a2.lower() in ("search", "calculator"):
            action_type = "search"
        elif fa is not None:
            action_type = "stop"
        else:
            action_type = None
        em = None
        if fa is not None and rec.get("gold_answer"):
            gold = rec.get("gold_answers") or [rec["gold_answer"]]
            em = int(answer_scorer(fa, gold, mode="exact")["matched"])
        contains_W = int(rec.get("W") is not None and rec["W"].lower() in (fa or "").lower())
        contains_V = int(rec.get("V") is not None and rec["V"].lower() in (fa or "").lower())
    return {
        "sample_id": rec["sample_id"],
        "condition_id": rec["condition_id"],
        "intervention": intervention,
        "margin": margins["margin_first_token"],          # kept for backward compat
        "margin_first_token": margins["margin_first_token"],
        "margin_label": margins["margin_label"],
        "lp_Action": margins["lp_Action"],
        "lp_Final": margins["lp_Final"],
        "lp_search_after": margins["lp_search_after"],
        "lp_Final_after": margins["lp_Final_after"],
        "action_type": action_type, "action2": a2,
        "final_answer": fa, "em": em,
        "contains_W": contains_W, "contains_V": contains_V,
        "parse_failure": pf,
        "raw_output": raw[:400],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/unsupported_trap/pairs.jsonl")
    ap.add_argument("--out",   default="results/steering_trap/eval_results.jsonl")
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--rho", type=float, default=-0.20)
    ap.add_argument("--conditions", nargs="+",
                    default=["Trap-B0", "True-D0"])
    ap.add_argument("--interventions", nargs="+",
                    default=["baseline", "a3", "evidence_parallel", "random"])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-gen", action="store_true",
                    help="Skip generation; only compute margin_first_token and margin_label.")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.pairs)]
    records = [r for r in records if r["condition_id"] in args.conditions]
    if args.limit:
        records = records[:args.limit]
    print(f"[info] loaded {len(records)} records; interventions={args.interventions}")

    print(f"[info] loading model {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    ); model.eval()
    device = next(model.parameters()).device

    tool_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]
                if tok.encode(t, add_special_tokens=False)]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]
                if tok.encode(t, add_special_tokens=False)]
    builder = PromptBuilder()
    label_tokens = setup_label_tokens(tok)

    dirs = {}
    for name in args.interventions:
        p = INTERVENTIONS[name]
        if p is None:
            dirs[name] = (None, 0.0)
            continue
        v = load_direction(p)
        drms = float(np.sqrt(np.mean(v ** 2)))
        alpha = args.rho * (HIDDEN_RMS / drms)
        dirs[name] = (v, alpha)
        print(f"[dir] {name:<18s} path={p} direction_rms={drms:.4f} alpha={alpha:+.4f}")

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out_path, "w") as f:
        total = len(records) * len(args.interventions); done = 0
        for rec in records:
            for name in args.interventions:
                d, a = dirs[name]
                row = run_one(rec, name, d, a, model, tok, builder, device,
                              tool_ids, fin_ids, label_tokens,
                              max_new_tokens=args.max_new_tokens,
                              skip_gen=args.skip_gen)
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
                done += 1
                if done % 25 == 0 or done == total:
                    print(f"  [{done}/{total}] {time.time()-t0:.1f}s")
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
