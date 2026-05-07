#!/usr/bin/env python3
"""Capability audit for an FT LoRA adapter (default: qwen_balanced_v1).

Five checks; emits per_check_details.json, summary.json, report.md to --out-dir.

Check 1  Supported-evidence (S0) performance         final_rate(adapter) >= base - 0.05
Check 2  General QA on PopQA (contains)              em(adapter) >= base - 0.05  AND  >= 0.05
Check 3  Output-length distribution KS test          KS p-value >= 0.05  OR  median ratio in [0.5, 2.0]
Check 4  Parse failure rate                          |pf(adapter) - pf(base)| <= 0.05
Check 5  Adapter reversibility (L20 cosine)          mean cos(adapter_on, adapter_off) > 0.999

Loads model + adapter once. Uses model.disable_adapter() for the base side.
"""
from __future__ import annotations
import argparse, json, sys, time, math, re
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent.prompts import PromptBuilder, parse_action  # noqa: E402
from eval.scorers import answer_scorer                  # noqa: E402
from run_steering_trap_eval import (                    # noqa: E402
    setup_label_tokens, compute_label_margin,
)

MODEL_PATH    = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_PAIRS = "results/extractability_support_toggle_v200/pairs.jsonl"
DEFAULT_POPQA = "data/popqa/popqa_test.jsonl"
DEFAULT_ADAPT = "adapters/qwen_balanced_v1"
DEFAULT_OUT   = "results/ft_capability_audit"
LAYER         = 20

POPQA_SYSTEM = ("You are a helpful assistant. Answer the user's question "
                "with the answer only, in as few words as possible.")


def _load_adapter_model(model_path: str, adapter_path: str):
    print(f"[load] model={model_path}")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    ); model.eval()
    print(f"[load] adapter={adapter_path}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_path); model.eval()
    return model, tok


# ---------- Check 1: S0 supported-evidence ---------------------------------

def _run_eval_one_s0(rec, model, tok, builder, device, label_tokens, max_new=256):
    obs = rec["obs"]
    query = f"about: {rec['question'][:80]}"
    steps = [{"action": "search", "action_input": query, "observation": obs}]
    messages = builder.build_full_prompt(rec["question"], steps)
    prompt_str = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    input_ids = tok.encode(prompt_str, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    attn = torch.ones_like(input_ids)
    margins = compute_label_margin(model, input_ids, prompt_len, None, 0.0,
                                   label_tokens, device)
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids, attention_mask=attn, max_new_tokens=max_new,
            pad_token_id=tok.pad_token_id or tok.eos_token_id, do_sample=False,
        )
    raw = tok.decode(gen_ids[0, prompt_len:], skip_special_tokens=True)
    out_len_tok = int(gen_ids.shape[1] - prompt_len)
    parsed = parse_action(raw)
    a2, fa = parsed["action"], parsed["final_answer"]
    pf = (a2 is None and fa is None)
    if a2 and a2.lower() in ("search", "calculator"):
        atype = "search"
    elif fa is not None:
        atype = "stop"
    else:
        atype = None
    em = None
    if fa is not None and rec.get("gold_answer"):
        gold = rec.get("gold_answers") or [rec["gold_answer"]]
        em = int(answer_scorer(fa, gold, mode="exact")["matched"])
    return {
        "sample_id": rec["sample_id"], "schema_type": rec.get("schema_type"),
        "condition": rec.get("condition") or rec.get("condition_id"),
        "margin_label": margins["margin_label"],
        "margin_first_token": margins["margin_first_token"],
        "action_type": atype, "final_answer": fa, "em": em,
        "parse_failure": bool(pf), "output_len_tok": out_len_tok,
        "raw_output": raw[:300],
    }


def check1_s0(model, tok, args):
    print(f"\n[check1] S0 supported-evidence  N={args.n_s0}")
    builder = PromptBuilder()
    device = next(model.parameters()).device
    label_tokens = setup_label_tokens(tok)
    pairs = [json.loads(l) for l in open(args.pairs_path)]
    s0 = [r for r in pairs
          if (r.get("condition") or r.get("condition_id")) == "S0"]
    if args.n_s0:
        s0 = s0[: args.n_s0]
    print(f"[check1]   loaded {len(s0)} S0 records")
    out = {"adapter_on": [], "adapter_off": []}
    for tag, ctx_fn in (("adapter_off", model.disable_adapter),
                        ("adapter_on", nullcontext)):
        t0 = time.time()
        with ctx_fn():
            for i, rec in enumerate(s0, 1):
                row = _run_eval_one_s0(rec, model, tok, builder, device,
                                       label_tokens)
                row["adapter"] = (tag == "adapter_on")
                out[tag].append(row)
                if i % 25 == 0 or i == len(s0):
                    print(f"  [{tag} {i}/{len(s0)}] {time.time()-t0:.1f}s")
    return out


def _summ_s0(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "final_rate": sum(1 for r in rows if r["action_type"] == "stop") / n,
        "search_rate": sum(1 for r in rows if r["action_type"] == "search") / n,
        "em_rate": sum(1 for r in rows if r["em"] == 1) / n,
        "parse_fail_rate": sum(1 for r in rows if r["parse_failure"]) / n,
        "output_len_tok_mean": float(np.mean([r["output_len_tok"] for r in rows])),
        "output_len_tok_median": float(np.median([r["output_len_tok"] for r in rows])),
        "margin_label_mean": float(np.mean([r["margin_label"] for r in rows])),
    }



# ---------- Check 2: PopQA general QA -------------------------------------

def _popqa_prompt(tok, question: str) -> torch.Tensor:
    msgs = [
        {"role": "system", "content": POPQA_SYSTEM},
        {"role": "user", "content": question},
    ]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok.encode(s, return_tensors="pt")


def _run_popqa_one(rec, model, tok, device, max_new=64):
    raw_ans = rec.get("possible_answers")
    if isinstance(raw_ans, str):
        try:
            golds = json.loads(raw_ans)
        except Exception:
            golds = [raw_ans]
    else:
        golds = list(raw_ans) if raw_ans else []
    if rec.get("obj") and rec["obj"] not in golds:
        golds = [rec["obj"]] + golds

    input_ids = _popqa_prompt(tok, rec["question"]).to(device)
    prompt_len = input_ids.shape[1]
    attn = torch.ones_like(input_ids)
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids, attention_mask=attn, max_new_tokens=max_new,
            pad_token_id=tok.pad_token_id or tok.eos_token_id, do_sample=False,
        )
    raw = tok.decode(gen_ids[0, prompt_len:], skip_special_tokens=True).strip()
    out_len_tok = int(gen_ids.shape[1] - prompt_len)
    em = int(answer_scorer(raw, golds, mode="contains")["matched"])
    pf = bool(re.search(r"^\s*(Action|Final Answer)\s*:", raw))
    return {"qid": rec.get("id"), "question": rec["question"][:140],
            "gold": golds[:5], "prediction": raw[:160],
            "em": em, "parse_failure": pf, "output_len_tok": out_len_tok}


def check2_popqa(model, tok, args):
    print(f"\n[check2] PopQA general QA  N={args.n_popqa}")
    device = next(model.parameters()).device
    pop = [json.loads(l) for l in open(args.popqa_path)]
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(pop))[: args.n_popqa]
    pop = [pop[int(i)] for i in idx]
    out = {"adapter_on": [], "adapter_off": []}
    for tag, ctx_fn in (("adapter_off", model.disable_adapter),
                        ("adapter_on", nullcontext)):
        t0 = time.time()
        with ctx_fn():
            for i, rec in enumerate(pop, 1):
                row = _run_popqa_one(rec, model, tok, device)
                row["adapter"] = (tag == "adapter_on")
                out[tag].append(row)
                if i % 25 == 0 or i == len(pop):
                    print(f"  [{tag} {i}/{len(pop)}] {time.time()-t0:.1f}s")
    return out


def _summ_popqa(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "em_rate": sum(r["em"] for r in rows) / n,
        "parse_fail_rate": sum(1 for r in rows if r["parse_failure"]) / n,
        "output_len_tok_mean": float(np.mean([r["output_len_tok"] for r in rows])),
        "output_len_tok_median": float(np.median([r["output_len_tok"] for r in rows])),
    }


# ---------- Check 5: Adapter reversibility (L20 cosine) -------------------

def check5_reversibility(model, tok, args):
    print(f"\n[check5] L{LAYER} reversibility  N={args.n_revers}")
    device = next(model.parameters()).device
    builder = PromptBuilder()
    pairs = [json.loads(l) for l in open(args.pairs_path)]
    s0 = [r for r in pairs
          if (r.get("condition") or r.get("condition_id")) == "S0"][: args.n_revers]
    rows, cos_vals = [], []
    for i, rec in enumerate(s0, 1):
        obs = rec["obs"]
        query = f"about: {rec['question'][:80]}"
        steps = [{"action": "search", "action_input": query, "observation": obs}]
        msgs = builder.build_full_prompt(rec["question"], steps)
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tok.encode(s, return_tensors="pt").to(device)
        attn = torch.ones_like(input_ids)
        with torch.no_grad():
            with model.disable_adapter():
                h_off = model(input_ids, attention_mask=attn,
                              output_hidden_states=True).hidden_states[LAYER][0, -1].float()
            h_on = model(input_ids, attention_mask=attn,
                         output_hidden_states=True).hidden_states[LAYER][0, -1].float()
        cos = float(torch.nn.functional.cosine_similarity(
            h_on.unsqueeze(0), h_off.unsqueeze(0)).item())
        l2 = float((h_on - h_off).norm().item() / max(h_off.norm().item(), 1e-8))
        rows.append({"sample_id": rec["sample_id"], "cos_l20": cos,
                     "rel_l2_l20": l2})
        cos_vals.append(cos)
        if i % 10 == 0 or i == len(s0):
            print(f"  [{i}/{len(s0)}] mean_cos={np.mean(cos_vals):.6f}")
    return {"per_sample": rows,
            "summary": {"n": len(rows),
                        "mean_cos_l20": float(np.mean(cos_vals)),
                        "median_cos_l20": float(np.median(cos_vals)),
                        "min_cos_l20": float(np.min(cos_vals)),
                        "max_cos_l20": float(np.max(cos_vals))}}


# ---------- Check 3 & 4: derived from check 1 + check 2 outputs -----------

def _ks_two_sample(a, b):
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    cdf1 = np.searchsorted(a, np.concatenate([a, b]), side="right") / n1
    cdf2 = np.searchsorted(b, np.concatenate([a, b]), side="right") / n2
    d = float(np.max(np.abs(cdf1 - cdf2)))
    en = math.sqrt(n1 * n2 / (n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * d
    j = np.arange(1, 101)
    p = 2.0 * float(np.sum((-1) ** (j - 1) * np.exp(-2.0 * lam ** 2 * j ** 2)))
    p = max(min(p, 1.0), 0.0)
    return d, p


def _verdict(summary):
    s1 = summary["check1_s0"]; s2 = summary["check2_popqa"]
    s3 = summary["check3_length"]; s4 = summary["check4_parse_fail"]
    s5 = summary["check5_reversibility"]
    v = {}
    # Check 1
    delta1 = s1["adapter"]["final_rate"] - s1["base"]["final_rate"]
    v["check1"] = {
        "metric": "S0 final_rate (adapter) >= base - 0.05",
        "base_final_rate": s1["base"]["final_rate"],
        "adapter_final_rate": s1["adapter"]["final_rate"],
        "delta": delta1,
        "pass": bool(delta1 >= -0.05),
    }
    # Check 2
    delta2 = s2["adapter"]["em_rate"] - s2["base"]["em_rate"]
    v["check2"] = {
        "metric": "PopQA em_rate (adapter) >= base - 0.05  AND  >= 0.05",
        "base_em": s2["base"]["em_rate"],
        "adapter_em": s2["adapter"]["em_rate"],
        "delta": delta2,
        "pass": bool(delta2 >= -0.05 and s2["adapter"]["em_rate"] >= 0.05),
    }
    # Check 3
    med_b = s3["base_median"]; med_a = s3["adapter_median"]
    ratio = med_a / med_b if med_b > 0 else float("inf")
    v["check3"] = {
        "metric": "KS p>=0.05  OR  median ratio in [0.5, 2.0]",
        "ks_d": s3["ks_d"], "ks_p": s3["ks_p"],
        "base_median": med_b, "adapter_median": med_a, "median_ratio": ratio,
        "pass": bool(s3["ks_p"] >= 0.05 or 0.5 <= ratio <= 2.0),
    }
    # Check 4
    delta4 = s4["adapter_rate"] - s4["base_rate"]
    v["check4"] = {
        "metric": "|pf(adapter) - pf(base)| <= 0.05",
        "base_rate": s4["base_rate"], "adapter_rate": s4["adapter_rate"],
        "delta": delta4, "pass": bool(abs(delta4) <= 0.05),
    }
    # Check 5
    v["check5"] = {
        "metric": "mean cos(adapter_on, adapter_off) > 0.999",
        "mean_cos_l20": s5["mean_cos_l20"],
        "min_cos_l20": s5["min_cos_l20"],
        "pass": bool(s5["mean_cos_l20"] > 0.999),
    }
    v["all_pass"] = all(v[k]["pass"] for k in ("check1","check2","check3","check4","check5"))
    v["recommendation"] = ("Discussion section" if v["all_pass"]
                            else "Appendix-only or omit")
    return v


def _write_report(out_dir: Path, summary, verdict, args):
    lines = []
    lines.append(f"# FT Capability Audit — `{args.adapter_path}`")
    lines.append("")
    lines.append(f"- Base model: `{args.model_path}`")
    lines.append(f"- Layer (Check 5): L{LAYER}")
    lines.append(f"- Pairs: `{args.pairs_path}`")
    lines.append(f"- PopQA: `{args.popqa_path}`")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**ALL PASS = {verdict['all_pass']}** → {verdict['recommendation']}")
    lines.append("")
    lines.append("| # | Check | PASS | Detail |")
    lines.append("|---|---|---|---|")
    c1 = verdict["check1"]; c2 = verdict["check2"]; c3 = verdict["check3"]
    c4 = verdict["check4"]; c5 = verdict["check5"]
    lines.append(f"| 1 | S0 final_rate (Δ ≥ -0.05) | {c1['pass']} | "
                 f"base={c1['base_final_rate']:.3f}  adapter={c1['adapter_final_rate']:.3f}  Δ={c1['delta']:+.3f} |")
    lines.append(f"| 2 | PopQA EM (Δ ≥ -0.05 ∧ ≥ 0.05) | {c2['pass']} | "
                 f"base={c2['base_em']:.3f}  adapter={c2['adapter_em']:.3f}  Δ={c2['delta']:+.3f} |")
    lines.append(f"| 3 | Length KS / median ratio | {c3['pass']} | "
                 f"KS_d={c3['ks_d']:.3f}  p={c3['ks_p']:.3f}  med_b={c3['base_median']:.0f}  med_a={c3['adapter_median']:.0f}  ratio={c3['median_ratio']:.2f} |")
    lines.append(f"| 4 | Parse-fail Δ ≤ 0.05 | {c4['pass']} | "
                 f"base={c4['base_rate']:.3f}  adapter={c4['adapter_rate']:.3f}  Δ={c4['delta']:+.3f} |")
    lines.append(f"| 5 | L{LAYER} cos > 0.999 | {c5['pass']} | "
                 f"mean={c5['mean_cos_l20']:.6f}  min={c5['min_cos_l20']:.6f} |")
    lines.append("")
    lines.append("## Per-check details")
    lines.append("")
    lines.append("### Check 1 — S0 supported-evidence (decision-point behavior)")
    s1 = summary["check1_s0"]
    lines.append(f"- Base: n={s1['base']['n']}, final_rate={s1['base']['final_rate']:.3f}, "
                 f"search_rate={s1['base']['search_rate']:.3f}, em={s1['base']['em_rate']:.3f}, "
                 f"pf={s1['base']['parse_fail_rate']:.3f}, "
                 f"out_len(med)={s1['base']['output_len_tok_median']:.0f}")
    lines.append(f"- Adapter: n={s1['adapter']['n']}, final_rate={s1['adapter']['final_rate']:.3f}, "
                 f"search_rate={s1['adapter']['search_rate']:.3f}, em={s1['adapter']['em_rate']:.3f}, "
                 f"pf={s1['adapter']['parse_fail_rate']:.3f}, "
                 f"out_len(med)={s1['adapter']['output_len_tok_median']:.0f}")
    lines.append("")
    lines.append("### Check 2 — PopQA general QA (zero-shot, contains-match)")
    s2 = summary["check2_popqa"]
    lines.append(f"- Base: n={s2['base']['n']}, em={s2['base']['em_rate']:.3f}, "
                 f"pf={s2['base']['parse_fail_rate']:.3f}, "
                 f"out_len(med)={s2['base']['output_len_tok_median']:.0f}")
    lines.append(f"- Adapter: n={s2['adapter']['n']}, em={s2['adapter']['em_rate']:.3f}, "
                 f"pf={s2['adapter']['parse_fail_rate']:.3f}, "
                 f"out_len(med)={s2['adapter']['output_len_tok_median']:.0f}")
    lines.append("")
    lines.append("### Check 3 — Output-length distribution")
    lines.append(f"- Pooled S0+PopQA tokens. KS_d={c3['ks_d']:.4f}, p={c3['ks_p']:.4f}")
    lines.append(f"- Base median={c3['base_median']:.1f}, Adapter median={c3['adapter_median']:.1f}, ratio={c3['median_ratio']:.3f}")
    lines.append("")
    lines.append("### Check 4 — Parse-failure rate (pooled)")
    lines.append(f"- Base pf={c4['base_rate']:.4f}, Adapter pf={c4['adapter_rate']:.4f}, Δ={c4['delta']:+.4f}")
    lines.append("")
    lines.append(f"### Check 5 — Adapter reversibility (L{LAYER} hidden state, last token)")
    lines.append(f"- n={summary['check5_reversibility']['n']}, mean cos={c5['mean_cos_l20']:.6f}, "
                 f"min={c5['min_cos_l20']:.6f}")
    lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path",  default=MODEL_PATH)
    ap.add_argument("--adapter-path", default=DEFAULT_ADAPT)
    ap.add_argument("--pairs-path",  default=DEFAULT_PAIRS)
    ap.add_argument("--popqa-path",  default=DEFAULT_POPQA)
    ap.add_argument("--out-dir",     default=DEFAULT_OUT)
    ap.add_argument("--n-s0",      type=int, default=200)
    ap.add_argument("--n-popqa",   type=int, default=300)
    ap.add_argument("--n-revers",  type=int, default=50)
    ap.add_argument("--seed",      type=int, default=20260429)
    ap.add_argument("--skip-check1", action="store_true")
    ap.add_argument("--skip-check2", action="store_true")
    ap.add_argument("--skip-check5", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model, tok = _load_adapter_model(args.model_path, args.adapter_path)

    details = {}
    summary = {}

    if not args.skip_check1:
        c1 = check1_s0(model, tok, args)
        details["check1_s0"] = c1
        summary["check1_s0"] = {"base":    _summ_s0(c1["adapter_off"]),
                                "adapter": _summ_s0(c1["adapter_on"])}

    if not args.skip_check2:
        c2 = check2_popqa(model, tok, args)
        details["check2_popqa"] = c2
        summary["check2_popqa"] = {"base":    _summ_popqa(c2["adapter_off"]),
                                   "adapter": _summ_popqa(c2["adapter_on"])}

    if not args.skip_check5:
        c5 = check5_reversibility(model, tok, args)
        details["check5_reversibility"] = c5
        summary["check5_reversibility"] = c5["summary"]

    # Check 3 (length distribution) and Check 4 (parse-fail) — aggregated
    base_lens, adap_lens = [], []
    base_pf,   adap_pf   = [], []
    for key in ("check1_s0", "check2_popqa"):
        if key not in details:
            continue
        for r in details[key]["adapter_off"]:
            base_lens.append(r["output_len_tok"])
            base_pf.append(int(r["parse_failure"]))
        for r in details[key]["adapter_on"]:
            adap_lens.append(r["output_len_tok"])
            adap_pf.append(int(r["parse_failure"]))
    ks_d, ks_p = _ks_two_sample(base_lens, adap_lens)
    summary["check3_length"] = {
        "n_base": len(base_lens), "n_adapter": len(adap_lens),
        "base_median": float(np.median(base_lens)) if base_lens else 0.0,
        "adapter_median": float(np.median(adap_lens)) if adap_lens else 0.0,
        "base_mean": float(np.mean(base_lens)) if base_lens else 0.0,
        "adapter_mean": float(np.mean(adap_lens)) if adap_lens else 0.0,
        "ks_d": ks_d, "ks_p": ks_p,
    }
    summary["check4_parse_fail"] = {
        "n_base": len(base_pf), "n_adapter": len(adap_pf),
        "base_rate": float(np.mean(base_pf)) if base_pf else 0.0,
        "adapter_rate": float(np.mean(adap_pf)) if adap_pf else 0.0,
        "base_count": int(np.sum(base_pf)),
        "adapter_count": int(np.sum(adap_pf)),
    }

    summary["config"] = {
        "model_path": args.model_path, "adapter_path": args.adapter_path,
        "pairs_path": args.pairs_path, "popqa_path": args.popqa_path,
        "n_s0": args.n_s0, "n_popqa": args.n_popqa, "n_revers": args.n_revers,
        "seed": args.seed, "layer": LAYER,
    }
    verdict = _verdict(summary)
    summary["verdict"] = verdict

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "per_check_details.json").write_text(json.dumps(details, indent=2))
    _write_report(out_dir, summary, verdict, args)
    print("\n=== VERDICT ===")
    for k in ("check1","check2","check3","check4","check5"):
        print(f"  {k}: PASS={verdict[k]['pass']}  ({verdict[k]['metric']})")
    print(f"  ALL_PASS={verdict['all_pass']}  → {verdict['recommendation']}")
    print(f"\nWrote: {out_dir}/{{summary.json, per_check_details.json, report.md}}")


if __name__ == "__main__":
    main()
