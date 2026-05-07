#!/usr/bin/env python3
"""N=200 multi-layer extension of _extract_h_for_st_contrast.py.

For each model, captures the residual stream at p0 across multiple layers
(L_peak ± k) for the N=200 paired cells in:
  results/extractability_support_toggle_n200/pairs.jsonl
  results/second_benchmark_extractability_n200/pairs.jsonl

Writes one .npz per (dataset, layer) at:
  <out_dir>/L<layer>/raw_h_<dataset>.npz

so the existing _robustness_st_contrast.py can be pointed at any
per-layer subdir for the N=200 follow-up.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))
from eval_extractability_cross_model import build_messages, apply_chat_template_safe  # noqa

DEFAULT_PAIRS = {
    "hotpotqa": "results/extractability_support_toggle_n200/pairs.jsonl",
    "musique":  "results/second_benchmark_extractability_n200/pairs.jsonl",
}
CONDITIONS = ("N0", "T0", "S0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--peak-layer", type=int, required=True,
                    help="Center of the layer sweep.")
    ap.add_argument("--layer-offsets", type=int, nargs="+",
                    default=[-2, -1, 0, 1, 2],
                    help="Offsets relative to --peak-layer (default L\u00b12).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hotpotqa-pairs", default=DEFAULT_PAIRS["hotpotqa"])
    ap.add_argument("--musique-pairs",  default=DEFAULT_PAIRS["musique"])
    args = ap.parse_args()

    layers = sorted({args.peak_layer + o for o in args.layer_offsets})
    pairs_paths = {"hotpotqa": args.hotpotqa_pairs, "musique": args.musique_pairs}

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
    print(f"[ok] device={device}  layers={layers}  (model has {n_layers_total})",
          flush=True)
    for L in layers:
        if not (0 <= L <= n_layers_total):
            raise SystemExit(f"layer {L} out of range [0, {n_layers_total}]")

    for ds_name, pairs_path in pairs_paths.items():
        rows = [json.loads(l) for l in open(pairs_path)]
        rows = [r for r in rows
                if (r.get("condition") or r.get("condition_id")) in CONDITIONS]
        print(f"[{ds_name}] N={len(rows)} from {pairs_path}", flush=True)

        sids, conds = [], []
        H_per_layer = {L: [] for L in layers}
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
                for L in layers:
                    h = out.hidden_states[L][0, -1, :].float().cpu().numpy()
                    H_per_layer[L].append(h)
                sids.append(rec["sample_id"])
                conds.append(rec.get("condition") or rec.get("condition_id"))
                if i % 50 == 0 or i == len(rows):
                    print(f"  [{ds_name} {i}/{len(rows)}]", flush=True)

        sids_a = np.array(sids); conds_a = np.array(conds)
        for L in layers:
            sub_dir = out_root / f"L{L}"; sub_dir.mkdir(parents=True, exist_ok=True)
            H = np.stack(H_per_layer[L]).astype(np.float32)
            out_path = sub_dir / f"raw_h_{ds_name}.npz"
            np.savez(out_path, H=H, sids=sids_a, conds=conds_a, L=int(L))
            print(f"[save] {out_path}  H={H.shape}", flush=True)


if __name__ == "__main__":
    main()
