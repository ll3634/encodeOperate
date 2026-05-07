#!/usr/bin/env python3
"""Check 5b — TRUE reversibility: cos(L20_disable_adapter, L20_base_no_peft).

Verifies that PEFT's `disable_adapter()` mathematically restores base behavior.
In bf16 this should be cos ~= 1.0 (limited only by floating-point determinism).

Two passes over the same N S0 prompts:
  pass A : load Qwen base WITHOUT any PEFT wrapping     -> H_base   [N, d]
  pass B : load Qwen base + adapter, with disable_adapter() context
           when computing forward                       -> H_dis    [N, d]
Reports per-sample cosine and L2-rel between H_base and H_dis at the last
prompt token of layer 20.

Also re-emits the original cos(adapter_on, adapter_off) numbers from the main
audit for side-by-side comparison, then patches summary.json + report.md so
verdict reflects the true reversibility metric.
"""
from __future__ import annotations
import argparse, json, sys, gc, time
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent.prompts import PromptBuilder  # noqa: E402

MODEL_PATH    = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_PAIRS = "results/extractability_support_toggle_v200/pairs.jsonl"
DEFAULT_ADAPT = "adapters/qwen_balanced_v1"
DEFAULT_OUT   = "results/ft_capability_audit"
LAYER         = 20


def _build_inputs(records, tok):
    builder = PromptBuilder()
    out = []
    for rec in records:
        obs = rec["obs"]
        query = f"about: {rec['question'][:80]}"
        steps = [{"action": "search", "action_input": query, "observation": obs}]
        msgs = builder.build_full_prompt(rec["question"], steps)
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok.encode(s, return_tensors="pt")
        out.append((rec["sample_id"], ids))
    return out


def _last_l20(model, ids, device):
    ids = ids.to(device)
    attn = torch.ones_like(ids)
    with torch.no_grad():
        h = model(ids, attention_mask=attn,
                  output_hidden_states=True).hidden_states[LAYER][0, -1].float().cpu()
    return h


def _pass_base(records, tok, model_path):
    print(f"[pass A] base (no PEFT) load {model_path}")
    m = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    ); m.eval()
    device = next(m.parameters()).device
    inputs = _build_inputs(records, tok)
    H = []
    t0 = time.time()
    for i, (sid, ids) in enumerate(inputs, 1):
        H.append((sid, _last_l20(m, ids, device)))
        if i % 10 == 0 or i == len(inputs):
            print(f"  base [{i}/{len(inputs)}] {time.time()-t0:.1f}s")
    del m; gc.collect(); torch.cuda.empty_cache()
    return H


def _pass_wrapped_disabled(records, tok, model_path, adapter_path):
    print(f"[pass B] base + adapter with disable_adapter()")
    m = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    ); m.eval()
    from peft import PeftModel
    m = PeftModel.from_pretrained(m, adapter_path); m.eval()
    device = next(m.parameters()).device
    inputs = _build_inputs(records, tok)
    H = []
    t0 = time.time()
    for i, (sid, ids) in enumerate(inputs, 1):
        with m.disable_adapter():
            H.append((sid, _last_l20(m, ids, device)))
        if i % 10 == 0 or i == len(inputs):
            print(f"  wrapped+disabled [{i}/{len(inputs)}] {time.time()-t0:.1f}s")
    del m; gc.collect(); torch.cuda.empty_cache()
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path",   default=MODEL_PATH)
    ap.add_argument("--adapter-path", default=DEFAULT_ADAPT)
    ap.add_argument("--pairs-path",   default=DEFAULT_PAIRS)
    ap.add_argument("--out-dir",      default=DEFAULT_OUT)
    ap.add_argument("--n",            type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    pairs = [json.loads(l) for l in open(args.pairs_path)]
    s0 = [r for r in pairs
          if (r.get("condition") or r.get("condition_id")) == "S0"][: args.n]
    print(f"[init] N={len(s0)} S0 prompts, layer L{LAYER}")

    H_base = _pass_base(s0, tok, args.model_path)
    H_dis  = _pass_wrapped_disabled(s0, tok, args.model_path, args.adapter_path)

    assert [a for a, _ in H_base] == [a for a, _ in H_dis], "id order mismatch"
    rows, cosines = [], []
    for (sid, hb), (_, hd) in zip(H_base, H_dis):
        c = float(torch.nn.functional.cosine_similarity(
            hb.unsqueeze(0), hd.unsqueeze(0)).item())
        l2 = float((hb - hd).norm().item() / max(hb.norm().item(), 1e-8))
        bit_eq = bool(torch.equal(hb.to(torch.bfloat16), hd.to(torch.bfloat16)))
        rows.append({"sample_id": sid, "cos_l20_true_revers": c,
                     "rel_l2": l2, "bf16_bitwise_equal": bit_eq})
        cosines.append(c)

    summary = {
        "metric": "cos(L20[base, no PEFT], L20[base+adapter, disable_adapter()])",
        "n": len(rows),
        "mean_cos_l20": float(np.mean(cosines)),
        "median_cos_l20": float(np.median(cosines)),
        "min_cos_l20": float(np.min(cosines)),
        "max_cos_l20": float(np.max(cosines)),
        "n_bf16_bitwise_equal": int(sum(r["bf16_bitwise_equal"] for r in rows)),
        "pass_threshold_0p999": bool(np.mean(cosines) > 0.999),
    }
    out = {"summary": summary, "per_sample": rows,
           "config": {"model_path": args.model_path,
                      "adapter_path": args.adapter_path,
                      "pairs_path": args.pairs_path,
                      "n": args.n, "layer": LAYER}}
    (out_dir / "check5b_true_reversibility.json").write_text(json.dumps(out, indent=2))
    print("\n=== Check 5b — TRUE reversibility ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {out_dir/'check5b_true_reversibility.json'}")


if __name__ == "__main__":
    main()
