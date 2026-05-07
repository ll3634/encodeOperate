#!/usr/bin/env python3
"""Scaling-law falsification audits.

Audit 1 (PopQA difficulty):
  Per model, free-gen greedy on N PopQA samples. Report EM (contains) and the
  per-sample "top-1 logit margin" = logp(top1) - logp(top2) at the first
  generated-token position (zero-shot QA prompt).

Audit 2 (Second-probe quality):
  Per model, derive an action_dir at --peak-layer using HotpotQA dev questions
  (no observations) as the contrastive source instead of PopQA. Direction =
  mean(low-10% margin) - mean(high-10% margin). Report quality =
  |Spearman(margin, h.direction)| across all samples (same metric as the
  PopQA-derived action_dir_quality reported in cross_model_full).

Both audits load the model once.
"""
from __future__ import annotations
import argparse, json, sys, time, re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))
from agent.prompts import PromptBuilder, ACTION_TOKENS  # noqa
from eval.scorers import answer_scorer                  # noqa
from cross_model_full import compute_margin, apply_chat_template_safe  # noqa
from steering.hook_utils import get_model_layers        # noqa

POPQA_SYSTEM = ("You are a helpful assistant. Answer the user's question "
                "with the answer only, in as few words as possible.")


def _stats(vals):
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)),
            "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)),
            "min": float(a.min()), "max": float(a.max())}


# ───────── Audit 1 ───────────────────────────────────────────────────────────
def audit1_popqa(model, tok, device, popqa_path, n, seed, out_dir):
    print(f"\n=== Audit 1: PopQA difficulty  N={n} ===", flush=True)
    rows = [json.loads(l) for l in open(popqa_path)]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))[:n]
    rows = [rows[int(i)] for i in idx]
    out_jsonl = out_dir / "audit1_popqa.jsonl"
    em_list, marg_list, pf_list = [], [], []
    t0 = time.time()
    with open(out_jsonl, "w") as f, torch.no_grad():
        for i, rec in enumerate(rows, 1):
            raw_ans = rec.get("possible_answers")
            if isinstance(raw_ans, str):
                try: golds = json.loads(raw_ans)
                except Exception: golds = [raw_ans]
            else:
                golds = list(raw_ans) if raw_ans else []
            if rec.get("obj") and rec["obj"] not in golds:
                golds = [rec["obj"]] + golds
            msgs = [{"role": "system", "content": POPQA_SYSTEM},
                    {"role": "user", "content": rec["question"]}]
            prompt = tok.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
            ids = tok.encode(prompt, return_tensors="pt").to(device)
            attn = torch.ones_like(ids)
            out = model(ids, attention_mask=attn, use_cache=False)
            logits = out.logits[0, -1, :].float()
            lp = torch.log_softmax(logits, dim=-1)
            top2 = torch.topk(lp, 2)
            margin = float(top2.values[0] - top2.values[1])
            top1_id = int(top2.indices[0])
            top1_str = tok.decode([top1_id])
            gen = model.generate(ids, attention_mask=attn, max_new_tokens=64,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id,
                                 do_sample=False)
            raw = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip()
            em = int(answer_scorer(raw, golds, mode="contains")["matched"])
            pf = bool(re.search(r"^\s*(Action|Final Answer)\s*:", raw))
            row = {"qid": rec.get("id"),
                   "question": rec["question"][:200],
                   "gold": golds[:5], "prediction": raw[:200],
                   "em": em, "parse_failure": pf,
                   "first_token_top1": top1_str,
                   "first_token_top1_margin": margin}
            em_list.append(em); marg_list.append(margin); pf_list.append(int(pf))
            f.write(json.dumps(row) + "\n"); f.flush()
            if i % 25 == 0 or i == len(rows):
                print(f"  [{i}/{len(rows)}] em={np.mean(em_list):.3f} "
                      f"marg_med={np.median(marg_list):.2f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    summ = {"n": len(em_list),
            "em_rate": float(np.mean(em_list)),
            "parse_fail_rate": float(np.mean(pf_list)),
            "first_token_top1_margin": _stats(marg_list)}
    summ["em_count"] = int(np.sum(em_list))
    rule = ("ceiling" if summ["em_rate"] >= 0.95
            else "floor" if summ["em_rate"] <= 0.60
            else ("weak" if 0.70 <= summ["em_rate"] <= 0.85 else "mid"))
    summ["decision_rule_zone"] = rule
    return summ


# ───────── Audit 2 ───────────────────────────────────────────────────────────
def audit2_second_probe(model, tok, device, hotpot_path, peak_layer, n, seed, out_dir):
    print(f"\n=== Audit 2: Second probe (HotpotQA), L{peak_layer}  N={n} ===",
          flush=True)
    raw = json.load(open(hotpot_path))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(raw))[:n]
    rows = [raw[int(i)] for i in idx]
    pb = PromptBuilder(tools=["search"])
    layers = get_model_layers(model)
    captured = {}
    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h[0, -1, :].detach().float().cpu().numpy()
    handle = layers[peak_layer].register_forward_hook(hook_fn)
    margins, hiddens = [], []
    t0 = time.time()
    out_jsonl = out_dir / "audit2_hotpot.jsonl"
    try:
        with open(out_jsonl, "w") as f, torch.no_grad():
            for i, s in enumerate(rows, 1):
                msgs = pb.build_full_prompt(s["question"], [])
                prompt = apply_chat_template_safe(tok, msgs)
                ids = tok.encode(prompt, return_tensors="pt").to(device)
                out = model(ids, use_cache=False)
                m = compute_margin(out.logits[0, -1, :], tok)
                margins.append(m); hiddens.append(captured["h"])
                f.write(json.dumps({"qid": s.get("_id"),
                                    "margin": m}) + "\n")
                if i % 50 == 0 or i == len(rows):
                    print(f"  [{i}/{len(rows)}] m=[{min(margins):.2f},"
                          f"{max(margins):.2f}] ({time.time()-t0:.0f}s)",
                          flush=True)
    finally:
        handle.remove()
    margins = np.array(margins); H = np.array(hiddens, dtype=np.float32)
    p_lo, p_hi = np.percentile(margins, 10), np.percentile(margins, 90)
    lo_mask, hi_mask = margins <= p_lo, margins >= p_hi
    direction = H[lo_mask].mean(0) - H[hi_mask].mean(0)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return {"n": int(margins.size), "quality": 0.0,
                "note": "degenerate direction"}
    direction /= norm
    proj = H @ direction
    rho, p = spearmanr(margins, proj)
    quality = float(abs(rho))
    np.savez(out_dir / "audit2_direction.npz", direction=direction,
             margins=margins, proj=proj, peak_layer=peak_layer)
    return {"n": int(margins.size), "peak_layer": int(peak_layer),
            "quality": quality, "spearman_r": float(rho),
            "spearman_p": float(p),
            "margin_stats": _stats(margins.tolist()),
            "n_low": int(lo_mask.sum()), "n_high": int(hi_mask.sum())}


# ───────── main ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--peak-layer", type=int, required=True,
                    help="Action-peak layer for this model (e.g., 7B=20, 14B=46, 32B=52).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--popqa-path", default="tmc/scripts/e2e_agent/data/popqa/popqa_test.jsonl")
    ap.add_argument("--hotpot-path",
                    default="tmc/scripts/e2e_agent/data/hotpotqa/hotpot_dev_distractor_v1.json")
    ap.add_argument("--n-popqa", type=int, default=200)
    ap.add_argument("--n-hotpot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260502)
    ap.add_argument("--skip-audit1", action="store_true")
    ap.add_argument("--skip-audit2", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[load] {args.model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True); model.eval()
    device = next(model.parameters()).device
    print(f"[ok] device={device}", flush=True)

    summary = {"model_path": args.model_path,
               "peak_layer": args.peak_layer,
               "n_popqa": args.n_popqa, "n_hotpot": args.n_hotpot,
               "seed": args.seed}
    if not args.skip_audit1:
        summary["audit1_popqa"] = audit1_popqa(
            model, tok, device, args.popqa_path,
            args.n_popqa, args.seed, out_dir)
    if not args.skip_audit2:
        summary["audit2_second_probe"] = audit2_second_probe(
            model, tok, device, args.hotpot_path,
            args.peak_layer, args.n_hotpot, args.seed, out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] -> {out_dir}/summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
