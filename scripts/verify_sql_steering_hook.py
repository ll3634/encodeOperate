#!/usr/bin/env python3
"""
SQL margin diagnostic + (optional) steering-hook verification.

Mirrors verify_codesearch_steering_hook.py for the SQL data-analysis surface.
Default mode iterates over all 60 T0 prompts and reports the per-item logit
margin (Action - Final) at p0; this tells us whether the QA A3 direction
(extracted at L20 with effective alpha ~0.20*hidden_rms/dir_rms) has any
chance of crossing the decision boundary on this surface.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering.hook_utils import SteeringHook
from nonqa_react_sql_items import build_items
from run_nonqa_react_sql import build_messages, w_for_condition


MODEL = "Qwen/Qwen2.5-7B-Instruct"
DIR_PATH = "tmc/scripts/e2e_agent/steering/directions/direction_search_v3_layer20.npz"
LAYER = 20
HIDDEN_RMS = 0.65


def _next_logits(model, input_ids):
    with torch.no_grad():
        out = model(input_ids)
    return out.logits[0, -1, :].detach().float().cpu().numpy()


def _capture_l20_last(model, input_ids, layer_idx):
    cap = {}
    def h(m, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        cap["h"] = x[0, -1, :].detach().float().cpu().numpy().copy()
    handle = model.model.layers[layer_idx].register_forward_hook(h)
    with torch.no_grad():
        out = model(input_ids)
    handle.remove()
    return cap["h"], out.logits[0, -1, :].detach().float().cpu().numpy()


def main():
    print(f"[load] {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    ).eval()

    d = np.load(DIR_PATH, allow_pickle=True)
    direction = np.asarray(d["decision_direction"], dtype=np.float32)
    dir_rms = float(np.sqrt(np.mean(direction ** 2)))
    print(f"[dir] shape={direction.shape}  rms={dir_rms:.4f}  "
          f"layer_meta={int(d['layer'])}")

    action_id = tok.encode("Action", add_special_tokens=False)[0]
    final_id = tok.encode("Final", add_special_tokens=False)[0]
    print(f"[tok] Action_id={action_id} ({tok.decode([action_id])!r})  "
          f"Final_id={final_id} ({tok.decode([final_id])!r})")

    items = build_items(n_items=60, seed=20260501)

    print("\n=== Per-item p0 logit margin (Action - Final) on T0 ===")
    print(f"{'item':>8s} | {'tmpl':>20s} | {'argmax':>14s} | "
          f"{'margin':>9s} | {'logit_A':>9s} | {'logit_F':>9s}")
    margins = []
    rows = []
    for it in items:
        msgs = build_messages(it, "T0", mode="prefilled")
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tok(text, return_tensors="pt").to(model.device)["input_ids"]
        logits = _next_logits(model, input_ids)
        m = float(logits[action_id] - logits[final_id])
        top = int(np.argmax(logits))
        margins.append(m)
        rows.append({"item_id": it.item_id, "template_key": it.template_key,
                     "margin": m, "argmax_token": tok.decode([top]),
                     "logit_action": float(logits[action_id]),
                     "logit_final": float(logits[final_id])})
        print(f"{it.item_id:>8s} | {it.template_key:>20s} | "
              f"{tok.decode([top])!r:>14s} | {m:>+9.3f} | "
              f"{float(logits[action_id]):>+9.3f} | {float(logits[final_id]):>+9.3f}")

    arr = np.array(margins)
    summary = {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "n_gt_minus8": int((arr > -8).sum()),
        "n_gt_minus6": int((arr > -6).sum()),
        "n_gt_minus4": int((arr > -4).sum()),
        "n_positive": int((arr > 0).sum()),
    }
    print("\n=== Summary across 60 T0 items ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:>14s}: {v:+.3f}")
        else:
            print(f"  {k:>14s}: {v}")

    out_dir = Path("tmc/scripts/e2e_agent/results/nonqa_react_sql/margin_diag_T0")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_item.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[diag] wrote per-item + summary to {out_dir}")

    # Sanity-check the steering hook on a representative T0 prompt.
    near_median = sorted(rows, key=lambda r: abs(r["margin"] - summary["median"]))[0]
    print(f"\n=== Hook sanity-check on item {near_median['item_id']} "
          f"(margin={near_median['margin']:+.3f}, near median) ===")
    item = next(it for it in items if it.item_id == near_median["item_id"])
    msgs = build_messages(item, "T0", mode="prefilled")
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    input_ids = tok(text, return_tensors="pt").to(model.device)["input_ids"]
    h0, logits0 = _capture_l20_last(model, input_ids, LAYER)
    margin0 = float(logits0[action_id] - logits0[final_id])
    print(f"{'rho':>6s} | {'top1':>14s} | {'margin':>9s} | {'Δmargin':>9s} | "
          f"{'cos(Δh,α·dir)':>14s}")
    for rho in [+0.20, 0.0, -0.20, -0.30, -0.60]:
        if abs(rho) < 1e-8:
            h_s, logits_s = h0, logits0
        else:
            alpha = rho * (HIDDEN_RMS / dir_rms)
            with SteeringHook(model, direction, alpha, layer=LAYER, position=-1,
                              mode="addition", max_interventions=1):
                h_s, logits_s = _capture_l20_last(model, input_ids, LAYER)
        m_s = float(logits_s[action_id] - logits_s[final_id])
        top_s = int(np.argmax(logits_s))
        delta = h_s - h0
        if abs(rho) < 1e-8:
            cos_str = "    n/a"
        else:
            expected = (rho * (HIDDEN_RMS / dir_rms)) * direction
            cos = float(np.dot(delta, expected) /
                        (np.linalg.norm(delta) * np.linalg.norm(expected) + 1e-12))
            cos_str = f"{cos:+.4f}"
        print(f"{rho:>+6.2f} | {tok.decode([top_s])!r:>14s} | "
              f"{m_s:>+9.3f} | {m_s - margin0:>+9.3f} | {cos_str:>14s}")


if __name__ == "__main__":
    main()
