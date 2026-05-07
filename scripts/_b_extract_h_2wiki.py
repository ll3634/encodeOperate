#!/usr/bin/env python3
"""B: extract residual stream at the peak layer for 2WikiMultiHop pairs.

Single-layer / single-dataset variant of _extract_h_n200_multilayer.py.
Reads results/third_benchmark_extractability/pairs.jsonl (50 sids x 3 conds),
captures the last-token residual at --peak-layer, writes
<out_dir>/L<peak>/raw_h_2wiki.npz with (H, sids, conds, L) so it slots into the
existing fit_direction / project_cells pipeline in _robustness_st_contrast.py.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent
_E2E_ROOT = _HERE.parent
sys.path.insert(0, str(_E2E_ROOT))
sys.path.insert(0, str(_HERE))
from eval_extractability_cross_model import build_messages, apply_chat_template_safe  # noqa

DEFAULT_PAIRS = str(_E2E_ROOT / "results/third_benchmark_extractability/pairs.jsonl")
CONDITIONS = ("N0", "T0", "S0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--peak-layer", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pairs", default=DEFAULT_PAIRS)
    args = ap.parse_args()

    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    print(f"[load] {args.model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True); model.eval()
    device = next(model.parameters()).device
    n_layers_total = model.config.num_hidden_layers
    print(f"[ok] device={device}  peak_layer={args.peak_layer}  "
          f"(model has {n_layers_total} layers)", flush=True)
    if not (0 <= args.peak_layer <= n_layers_total):
        raise SystemExit(f"layer {args.peak_layer} out of range")

    rows = [json.loads(l) for l in open(args.pairs)]
    rows = [r for r in rows
            if (r.get("condition") or r.get("condition_id")) in CONDITIONS]
    print(f"[2wiki] N={len(rows)} from {args.pairs}", flush=True)

    sids, conds, H_list = [], [], []
    with torch.no_grad():
        for i, rec in enumerate(rows, 1):
            msgs = build_messages(rec["question"], rec["obs"],
                                  prompt_variant="v1", obs_style="factcard")
            prompt_str = apply_chat_template_safe(tok, msgs,
                                                 add_generation_prompt=True)
            ids = tok.encode(prompt_str, return_tensors="pt",
                             add_special_tokens=False).to(device)
            attn = torch.ones_like(ids)
            out = model(ids, attention_mask=attn,
                        output_hidden_states=True, use_cache=False)
            h = out.hidden_states[args.peak_layer][0, -1, :].float().cpu().numpy()
            H_list.append(h)
            sids.append(rec["sample_id"])
            conds.append(rec.get("condition") or rec.get("condition_id"))
            if i % 25 == 0 or i == len(rows):
                print(f"  [2wiki {i}/{len(rows)}]", flush=True)

    sids_a = np.array(sids); conds_a = np.array(conds)
    sub_dir = out_root / f"L{args.peak_layer}"
    sub_dir.mkdir(parents=True, exist_ok=True)
    H = np.stack(H_list).astype(np.float32)
    out_path = sub_dir / "raw_h_2wiki.npz"
    np.savez(out_path, H=H, sids=sids_a, conds=conds_a, L=int(args.peak_layer))
    print(f"[save] {out_path}  H={H.shape}", flush=True)


if __name__ == "__main__":
    main()
