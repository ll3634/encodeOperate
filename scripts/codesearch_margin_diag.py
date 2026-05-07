#!/usr/bin/env python3
"""60-item p0 margin diagnostic on the code-search T0 surface.

Mirrors the per-item portion of verify_sql_steering_hook.py so the two
distributions can be plotted side-by-side.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nonqa_react_codesearch_items import build_items
from run_nonqa_react_codesearch import build_messages

MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _next_logits(model, input_ids):
    with torch.no_grad():
        out = model(input_ids)
    return out.logits[0, -1, :].detach().float().cpu().numpy()


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    ).eval()
    action_id = tok.encode("Action", add_special_tokens=False)[0]
    final_id = tok.encode("Final", add_special_tokens=False)[0]

    items = build_items(n_items=60, seed=20260501)
    rows = []
    print(f"{'item':>10s} | {'scenario':>14s} | {'argmax':>10s} | {'margin':>9s}")
    for it in items:
        msgs = build_messages(it, "T0", mode="prefilled")
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tok(text, return_tensors="pt").to(model.device)["input_ids"]
        logits = _next_logits(model, input_ids)
        m = float(logits[action_id] - logits[final_id])
        top = int(np.argmax(logits))
        scen = getattr(it, "category", getattr(it, "scenario", getattr(it, "template_key", "?")))
        rows.append({"item_id": it.item_id, "scenario": scen, "margin": m,
                     "argmax_token": tok.decode([top]),
                     "logit_action": float(logits[action_id]),
                     "logit_final": float(logits[final_id])})
        print(f"{it.item_id:>10s} | {scen:>14s} | {tok.decode([top])!r:>10s} | {m:>+9.3f}")

    arr = np.array([r["margin"] for r in rows])
    summary = {
        "n": int(arr.size), "mean": float(arr.mean()), "std": float(arr.std()),
        "min": float(arr.min()), "max": float(arr.max()),
        "median": float(np.median(arr)),
        "n_gt_minus8": int((arr > -8).sum()),
        "n_gt_minus6": int((arr > -6).sum()),
        "n_gt_minus4": int((arr > -4).sum()),
        "n_positive": int((arr > 0).sum()),
    }
    print("\n=== Summary across 60 T0 items (codesearch) ===")
    for k, v in summary.items():
        print(f"  {k:>14s}: {v:+.3f}" if isinstance(v, float) else f"  {k:>14s}: {v}")

    out_dir = Path("tmc/scripts/e2e_agent/results/nonqa_react_codesearch/margin_diag_T0")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_item.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[diag] wrote {out_dir}")


if __name__ == "__main__":
    main()
