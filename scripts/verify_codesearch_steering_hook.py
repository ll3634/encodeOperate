#!/usr/bin/env python3
"""
Standalone verification that SteeringHook is actually modifying the residual
stream at L20 position=-1 on a code-search T0 prompt, and that the modification
moves the next-token logit margin (Action vs Final).

Three checks:
  (A) Residual delta at L20 position=-1 matches alpha * direction (exact)
  (B) First-token top-1 distribution (and Action vs Final logit margin) shifts
      monotonically with rho.
  (C) Confirm that hook on T0 prompts produces a logit change even though
      argmax token does not flip — i.e. the hook IS modifying logits, just not
      enough to cross the decision boundary.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering.hook_utils import SteeringHook
from nonqa_react_codesearch_items import build_items
from run_nonqa_react_codesearch import build_messages, w_for_condition


MODEL = "Qwen/Qwen2.5-7B-Instruct"
DIR_PATH = "tmc/scripts/e2e_agent/steering/directions/direction_search_v3_layer20.npz"
LAYER = 20
HIDDEN_RMS = 0.65


def _capture_l20_last(model, input_ids, layer_idx):
    """Return (hidden_at_layer_output[-1], next_token_logits)."""
    cap = {}
    def h(m, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        cap["h"] = x[0, -1, :].detach().float().cpu().numpy().copy()
    layers = model.model.layers
    handle = layers[layer_idx].register_forward_hook(h)
    with torch.no_grad():
        out = model(input_ids)
    handle.remove()
    logits = out.logits[0, -1, :].detach().float().cpu().numpy()
    return cap["h"], logits


def main():
    print(f"[load] {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    ).eval()

    d = np.load(DIR_PATH, allow_pickle=True)
    direction = np.asarray(d["decision_direction"], dtype=np.float32)
    dir_rms = float(np.sqrt(np.mean(direction ** 2)))
    print(f"[dir] shape={direction.shape}  rms={dir_rms:.4f}  norm={np.linalg.norm(direction):.4f}  layer_meta={int(d['layer'])}")

    items = build_items(n_items=60, seed=20260501)
    item = items[0]                                      # cs_001
    msgs = build_messages(item, "T0", mode="prefilled")
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    input_ids = tok(text, return_tensors="pt").to(model.device)["input_ids"]
    print(f"[prompt] item={item.item_id}  cond=T0  "
          f"prompt_len={input_ids.shape[1]}  W={w_for_condition(item, 'T0')}")

    # Action / Final token IDs (first BPE piece)
    action_id = tok.encode("Action", add_special_tokens=False)[0]
    final_id  = tok.encode("Final",  add_special_tokens=False)[0]
    print(f"[tok] Action_id={action_id} ({tok.decode([action_id])!r})  "
          f"Final_id={final_id} ({tok.decode([final_id])!r})")

    h0, logits0 = _capture_l20_last(model, input_ids, LAYER)
    margin0 = float(logits0[action_id] - logits0[final_id])
    top0 = int(np.argmax(logits0))
    print(f"\n[baseline rho=0] L20-last-h sample={h0[:4]}")
    print(f"  argmax={tok.decode([top0])!r}  margin(Action-Final)={margin0:+.4f}")

    print(f"\n=== Check (A): residual delta == alpha * direction ===")
    for rho in [-1.0, -0.20, +0.20, -2.0]:
        alpha = rho * (HIDDEN_RMS / dir_rms)
        with SteeringHook(model, direction, alpha, layer=LAYER, position=-1,
                          mode="addition", max_interventions=1):
            h_s, logits_s = _capture_l20_last(model, input_ids, LAYER)
        expected = alpha * direction
        actual_delta = h_s - h0
        cos = float(np.dot(actual_delta, expected) /
                    (np.linalg.norm(actual_delta) * np.linalg.norm(expected) + 1e-12))
        rel_err = float(np.linalg.norm(actual_delta - expected) /
                        (np.linalg.norm(expected) + 1e-12))
        margin_s = float(logits_s[action_id] - logits_s[final_id])
        top_s = int(np.argmax(logits_s))
        print(f"  rho={rho:+.3f}  alpha={alpha:+.4f}  "
              f"||delta||={np.linalg.norm(actual_delta):.4f}  "
              f"cos(delta,alpha*dir)={cos:.4f}  rel_err={rel_err:.4e}  "
              f"argmax={tok.decode([top_s])!r:>10s}  "
              f"margin(A-F)={margin_s:+.4f}  Δmargin={margin_s-margin0:+.4f}")

    print(f"\n=== Check (B): scan rho for next-token argmax + Action/Final margin ===")
    print(f"{'rho':>6s} | {'top1_token':>14s} | {'margin(A-F)':>12s} | "
          f"{'logp(Action)':>13s} | {'logp(Final)':>12s}")
    for rho in [+0.50, +0.20, 0.0, -0.10, -0.20, -0.30, -0.50, -0.60, -1.00, -1.50, -2.00]:
        if abs(rho) < 1e-8:
            _, logits_r = _capture_l20_last(model, input_ids, LAYER)
        else:
            alpha = rho * (HIDDEN_RMS / dir_rms)
            with SteeringHook(model, direction, alpha, layer=LAYER, position=-1,
                              mode="addition", max_interventions=1):
                _, logits_r = _capture_l20_last(model, input_ids, LAYER)
        margin_r = float(logits_r[action_id] - logits_r[final_id])
        top_r = int(np.argmax(logits_r))
        log_softmax = logits_r - np.log(np.exp(logits_r - logits_r.max()).sum()) - logits_r.max()
        print(f"{rho:>+6.2f} | {tok.decode([top_r])!r:>14s} | "
              f"{margin_r:>+12.4f} | {log_softmax[action_id]:>+13.4f} | "
              f"{log_softmax[final_id]:>+12.4f}")


if __name__ == "__main__":
    main()
