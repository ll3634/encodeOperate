#!/usr/bin/env python3
"""Capture raw residual-stream h at L_peak, p0, for all (sid, cond) records
across the c2 paired-cell datasets. Writes a single .npz per (model, dataset)
holding sample_ids, conditions, and the raw h matrix.

Used to derive supporting-vs-distractor (S0 − T0) contrastive directions
from doc-conditioned hidden states, then re-project N0/T0/S0 cells onto that
direction. See scaling_law_summary.md "S0–T0 contrastive recompute".
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

DATASETS = {
    "hotpotqa": "results/extractability_support_toggle/pairs.jsonl",
    "musique":  "results/second_benchmark_extractability/pairs.jsonl",
}
CONDITIONS = ("N0", "T0", "S0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--peak-layer", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    L = args.peak_layer
    print(f"[load] {args.model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True); model.eval()
    device = next(model.parameters()).device
    print(f"[ok] device={device}  L={L}")

    for ds_name, pairs_path in DATASETS.items():
        rows = [json.loads(l) for l in open(pairs_path)]
        rows = [r for r in rows
                if (r.get("condition") or r.get("condition_id")) in CONDITIONS]
        print(f"[{ds_name}] N={len(rows)}", flush=True)

        sids, conds, Hs = [], [], []
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
                h = out.hidden_states[L][0, -1, :].float().cpu().numpy()
                sids.append(rec["sample_id"])
                conds.append(rec.get("condition") or rec.get("condition_id"))
                Hs.append(h)
                if i % 25 == 0 or i == len(rows):
                    print(f"  [{ds_name} {i}/{len(rows)}]", flush=True)

        H = np.stack(Hs).astype(np.float32)
        out_path = out_dir / f"raw_h_{ds_name}.npz"
        np.savez(out_path, H=H, sids=np.array(sids), conds=np.array(conds),
                 L=int(L))
        print(f"[save] {out_path}  H={H.shape}", flush=True)


if __name__ == "__main__":
    main()
