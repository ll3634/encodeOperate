#!/usr/bin/env python3
"""Diagnostics for patch_mlp20_task_missingness_locality.py.

Three checks on a handful of samples:
  (D1) Self-patch:  source = target's OWN mlp_out_last     -> Δmargin == 0 (exact).
  (D2) Last-token identity: verify the last input-token id is the same across
       all four cells (sf_tm, dist_tm, sf_neutral, dist_neutral) for each sample.
       This is what the chat template guarantees, but we should verify it.
  (D3) Full-residual patch: instead of mlp_L20 output, patch the whole L20
       layer output at the last position (sf_tm -> dist_tm). If this DOES move
       the margin substantially, it proves the pipeline is wired correctly and
       the mlp-only null is real (the attention/pre-mlp pathway carries the
       signal, not mlp_L20 output). Uses the SAME hook infrastructure.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import get_model_layers


def make_ids(tok):
    tool = [tok.encode(t, add_special_tokens=False)[0]
            for t in ACTION_TOKENS["tool_call"] if tok.encode(t, add_special_tokens=False)]
    fin  = [tok.encode(t, add_special_tokens=False)[0]
            for t in ACTION_TOKENS["finish"]    if tok.encode(t, add_special_tokens=False)]
    return tool, fin


def margin(logits_last, tool_ids, fin_ids):
    lp = torch.log_softmax(logits_last.float(), dim=-1)
    tl = torch.logsumexp(lp[tool_ids], 0).item()
    fl = torch.logsumexp(lp[fin_ids],  0).item()
    return float(tl - fl)


def build_prompt(builder, tok, rec):
    steps = [{"action": "search", "action_input": f"about: {rec['question'][:80]}",
              "observation": rec["obs"]}]
    msgs = builder.build_full_prompt(rec["question"], steps)
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def capture_and_margin(model, tok, prompt, device, tool_ids, fin_ids, module):
    cap = {}
    def h(m, i, o):
        x = o[0] if isinstance(o, tuple) else o
        cap["last"] = x[0, -1, :].detach().float().cpu().numpy().copy()
    handle = module.register_forward_hook(h)
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(ids)
    handle.remove()
    return cap["last"], margin(out.logits[0, -1, :], tool_ids, fin_ids), int(ids[0, -1].item())


def patched_margin(model, tok, prompt, device, tool_ids, fin_ids, module, vec_np):
    vec_t = torch.from_numpy(vec_np)
    captured = {"orig_last": None, "fired": 0}
    def h(m, i, o):
        x = o[0] if isinstance(o, tuple) else o
        captured["orig_last"] = x[0, -1, :].detach().float().cpu().numpy().copy()
        captured["fired"] += 1
        pv = vec_t.to(device=x.device, dtype=x.dtype)
        x2 = x.clone(); x2[0, -1, :] = pv
        if isinstance(o, tuple):
            return (x2,) + o[1:]
        return x2
    handle = module.register_forward_hook(h)
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(ids)
    handle.remove()
    return margin(out.logits[0, -1, :], tool_ids, fin_ids), captured["orig_last"], captured["fired"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/anti_cue_tm_n100/pairs.jsonl")
    ap.add_argument("--out", default="results/task_missingness_mlp20_patch/diagnostics.json")
    ap.add_argument("--n-samples", type=int, default=8)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.pairs)]
    sids = sorted(set(r["sample_id"] for r in records))[:args.n_samples]
    by_sid = {s: {} for s in sids}
    for r in records:
        if r["sample_id"] in by_sid:
            by_sid[r["sample_id"]][(r["target"], r["cue"])] = r

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval(); device = next(model.parameters()).device
    layers = get_model_layers(model)
    mlp20 = layers[20].mlp; layer20 = layers[20]
    tool_ids, fin_ids = make_ids(tok)
    builder = PromptBuilder()

    results = {"D1_self_patch": [], "D2_last_token": [], "D3_full_residual_patch": []}
    CELLS = [("sf","task_missingness"), ("distractor","task_missingness"),
             ("sf","neutral"), ("distractor","neutral")]
    for s in sids:
        natural = {}
        for cell in CELLS:
            prompt = build_prompt(builder, tok, by_sid[s][cell])
            mlp_last, m, last_id = capture_and_margin(model, tok, prompt, device, tool_ids, fin_ids, mlp20)
            lay_last, _, _       = capture_and_margin(model, tok, prompt, device, tool_ids, fin_ids, layer20)
            natural[cell] = {"prompt": prompt, "mlp_last": mlp_last, "lay_last": lay_last,
                             "margin": m, "last_token_id": last_id}
        results["D2_last_token"].append({
            "sample_id": s,
            "last_token_ids": {f"{c[0]}_{c[1]}": natural[c]["last_token_id"] for c in CELLS},
            "all_equal": len(set(natural[c]["last_token_id"] for c in CELLS)) == 1,
        })
        # D1: self-patch dist_tm -> dist_tm
        tgt = natural[("distractor","task_missingness")]
        m_sp, orig, fired = patched_margin(model, tok, tgt["prompt"], device, tool_ids, fin_ids, mlp20, tgt["mlp_last"])
        results["D1_self_patch"].append({
            "sample_id": s, "margin_natural": tgt["margin"], "margin_self_patched": m_sp,
            "delta": m_sp - tgt["margin"], "component_abs_diff_mean": float(np.mean(np.abs(tgt["mlp_last"] - orig))),
            "fired": fired,
        })
        # D3: full layer-output patch sf_tm -> dist_tm
        src = natural[("sf","task_missingness")]
        m_fp, orig3, fired3 = patched_margin(model, tok, tgt["prompt"], device, tool_ids, fin_ids, layer20, src["lay_last"])
        results["D3_full_residual_patch"].append({
            "sample_id": s, "margin_dist_tm_natural": tgt["margin"], "margin_sf_tm_natural": src["margin"],
            "margin_full_patched": m_fp, "delta_vs_dist_tm": m_fp - tgt["margin"],
            "locality_gap": src["margin"] - tgt["margin"],
            "component_abs_diff_mean": float(np.mean(np.abs(src["lay_last"] - orig3))),
            "fired": fired3,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    # Console summary
    d1 = results["D1_self_patch"]; d2 = results["D2_last_token"]; d3 = results["D3_full_residual_patch"]
    print(f"\n=== D1 self-patch (should be delta==0 exactly) ===")
    for r in d1: print(f"  {r['sample_id'][:10]}  nat={r['margin_natural']:+.4f}  sp={r['margin_self_patched']:+.4f}  Δ={r['delta']:+.6f}  comp_abs={r['component_abs_diff_mean']:.6f}  fired={r['fired']}")
    print(f"\n=== D2 last-token identity ===")
    for r in d2: print(f"  {r['sample_id'][:10]}  all_equal={r['all_equal']}  ids={list(r['last_token_ids'].values())}")
    print(f"\n=== D3 full-layer-output patch (sf_tm -> dist_tm) ===")
    print(f"  {'sample':12s} {'dist_tm_nat':>11s} {'sf_tm_nat':>10s} {'patched':>8s} {'Δ':>7s} {'gap':>7s} {'recovery':>8s}")
    for r in d3:
        rec = r['delta_vs_dist_tm'] / r['locality_gap'] if abs(r['locality_gap']) > 0.01 else float('nan')
        print(f"  {r['sample_id'][:10]:12s} {r['margin_dist_tm_natural']:+11.3f} {r['margin_sf_tm_natural']:+10.3f} {r['margin_full_patched']:+8.3f} {r['delta_vs_dist_tm']:+7.3f} {r['locality_gap']:+7.3f} {rec:+8.3f}")
    print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
