#!/usr/bin/env python3
"""Calibration + sign-convention diagnostic for cross-family A3 transfer.

For (model, direction, action_layer): on the first N SQL T0 prompts (matched to
the closure behavioral run), compute:
  - hidden_rms at action_layer (last-token RMS averaged across prompts)
  - baseline margin (Action - Final) at decision point
  - margin shift at rho=+0.20 and rho=-0.20  -> sign of "continue" (action+)

The sign convention used in run_nonqa_react_sql.py is alpha = rho * hidden_rms /
direction_rms. We report which sign of rho moves the margin in the
"more action / less final" direction. By the QA convention rho<0 == continue,
i.e. mean(margin) at rho=-0.20 should be > mean(margin) at rho=+0.20 if the
direction has the "search-more" polarity.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from steering.hook_utils import get_model_layers, SteeringHook
from agent.prompts import ACTION_TOKENS
from nonqa_react_sql_items import build_items
from run_nonqa_react_sql import build_messages
from run_nonqa_react_meeting import (
    normalize_messages_for_model, apply_template_for_completion,
)
from scripts.patch_L20_localise_full_residual import margin_from_logits


def make_margin_ids(tok):
    tool = [tok.encode(t, add_special_tokens=False)[0]
            for t in ACTION_TOKENS["tool_call"] if tok.encode(t, add_special_tokens=False)]
    fin = [tok.encode(t, add_special_tokens=False)[0]
           for t in ACTION_TOKENS["finish"] if tok.encode(t, add_special_tokens=False)]
    return tool, fin


def hidden_rms_at_last(model, tok, prompt, layer_idx, device):
    layers = get_model_layers(model)
    cap = {}
    def hook(_m, _i, o):
        h = o[0] if isinstance(o, tuple) else o
        cap["v"] = h[0, -1, :].detach().float().cpu().numpy()
    handle = layers[layer_idx].register_forward_hook(hook)
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model(ids)
    handle.remove()
    v = cap["v"]
    return float(np.sqrt(np.mean(v * v)))


def steered_margin(model, tok, prompt, device, tool_ids, fin_ids,
                   direction, alpha, layer_idx):
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    if direction is None or abs(alpha) < 1e-12:
        with torch.no_grad():
            logits = model(ids).logits[0, -1, :]
    else:
        with SteeringHook(model, direction, alpha, layer=layer_idx,
                          position=-1, mode="addition", max_interventions=1):
            with torch.no_grad():
                logits = model(ids).logits[0, -1, :]
    return margin_from_logits(logits, tool_ids, fin_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--direction-npz", required=True)
    ap.add_argument("--direction-key", default="action_dir")
    ap.add_argument("--act-layer", type=int, required=True)
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260501)
    args = ap.parse_args()

    print(f"[load] {args.model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    p = args.model_path.lower()
    attn_impl = "eager" if "gemma" in p else "sdpa"
    print(f"[load] attn_implementation={attn_impl}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation=attn_impl,
    )
    model.eval()
    device = next(model.parameters()).device
    actual = model.config._attn_implementation
    assert actual == attn_impl, f"attn rolled back: {actual}"

    d = np.load(args.direction_npz, allow_pickle=True)
    direction = np.asarray(d[args.direction_key], dtype=np.float32)
    direction_rms = float(np.sqrt(np.mean(direction ** 2)))
    print(f"[dir] {args.direction_npz}::{args.direction_key} rms={direction_rms:.6f}")

    items = build_items(n_items=max(60, args.n_prompts), seed=args.seed)[: args.n_prompts]
    tool_ids, fin_ids = make_margin_ids(tok)
    rows = []
    for i, it in enumerate(items):
        msgs = build_messages(it, "T0", mode="prefilled")
        msgs = normalize_messages_for_model(msgs, args.model_path)
        prompt = apply_template_for_completion(tok, msgs, args.model_path)
        rms = hidden_rms_at_last(model, tok, prompt, args.act_layer, device)
        m_base = steered_margin(model, tok, prompt, device, tool_ids, fin_ids,
                                None, 0.0, args.act_layer)
        # rho = +/- 0.20 sign check
        out = {"item_id": it.item_id, "hidden_rms": rms, "margin_base": m_base}
        for rho in (+0.20, -0.20):
            alpha = rho * (rms / direction_rms)
            m_st = steered_margin(model, tok, prompt, device, tool_ids, fin_ids,
                                  direction, alpha, args.act_layer)
            out[f"margin_rho{rho:+.2f}"] = m_st
            out[f"alpha_rho{rho:+.2f}"] = alpha
        rows.append(out)
        print(f"  [{i+1}/{len(items)}] rms={rms:.4f} m={m_base:+.3f} "
              f"m(+0.20)={out['margin_rho+0.20']:+.3f} m(-0.20)={out['margin_rho-0.20']:+.3f}", flush=True)

    rms_arr = np.array([r["hidden_rms"] for r in rows])
    base = np.array([r["margin_base"] for r in rows])
    pos = np.array([r["margin_rho+0.20"] for r in rows])
    neg = np.array([r["margin_rho-0.20"] for r in rows])
    summary = {
        "model_path": args.model_path,
        "direction_npz": args.direction_npz,
        "direction_key": args.direction_key,
        "act_layer": int(args.act_layer),
        "n_prompts": int(len(rows)),
        "hidden_rms_mean": float(rms_arr.mean()),
        "hidden_rms_std": float(rms_arr.std()),
        "direction_rms": direction_rms,
        "margin_base_mean": float(base.mean()),
        "margin_base_std": float(base.std()),
        "margin_rho_pos020_mean": float(pos.mean()),
        "margin_rho_neg020_mean": float(neg.mean()),
        "delta_pos020": float((pos - base).mean()),
        "delta_neg020": float((neg - base).mean()),
        "rows": rows,
    }
    summary["sign_continue"] = "neg" if summary["delta_neg020"] > summary["delta_pos020"] else "pos"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\n[wrote] {args.out}")
    print(f"hidden_rms@L{args.act_layer} = {summary['hidden_rms_mean']:.4f} (std {summary['hidden_rms_std']:.4f})")
    print(f"baseline margin = {summary['margin_base_mean']:+.3f}")
    print(f"delta_margin(rho=+0.20) = {summary['delta_pos020']:+.3f}")
    print(f"delta_margin(rho=-0.20) = {summary['delta_neg020']:+.3f}")
    print(f"continue sign = {summary['sign_continue']}  (neg => use --steer_rho NEGATIVE)")


if __name__ == "__main__":
    main()
