#!/usr/bin/env python3
"""C2: Qwen3-32B step-1 action-direction projection at L52, p0.

For each (dataset, condition) pair, capture the residual-stream activation at
layer 52, position p0 (the last input-token position — i.e., the slot whose
hidden state generates the first new token), project onto action_dir, and
report distributions per condition.

Also reports the corresponding logit-level label margin (read from the C1 eval
jsonls) so the residual-projection signal can be compared on the same samples
to the same-prompt logit-decision signal.

p0 is the natural "step-1 decision" residual: it is the activation immediately
before the model emits its first action token.
"""
import argparse, json, sys
from collections import defaultdict
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


def load_directions(path):
    d = np.load(path, allow_pickle=True)
    return {
        "action_dir":   torch.tensor(d["action_dir"],   dtype=torch.float32),
        "evidence_dir": torch.tensor(d["evidence_dir"], dtype=torch.float32),
        "L_act":  int(d["L_act"]),
        "L_evi":  int(d["L_evi"]),
        "cos_action_evidence": float(d["cos_action_evidence"]),
    }


def cell_stats(vals):
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {
        "n":     int(a.size),
        "mean":  float(a.mean()),
        "std":   float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median":float(np.median(a)),
        "p10":   float(np.percentile(a, 10)),
        "p90":   float(np.percentile(a, 90)),
        "min":   float(a.min()),
        "max":   float(a.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/home/featurize/work/models/Qwen3-32B")
    ap.add_argument("--directions", default="results/qwen3_32b_scale_check/directions.npz")
    ap.add_argument("--c1-dir", default="results/qwen3_32b_scale_check/c1",
                    help="Reads logit-level margins from c1/eval_<ds>.jsonl.")
    ap.add_argument("--out-dir", default="results/qwen3_32b_scale_check/c2")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dirs = load_directions(args.directions)
    L_act, L_evi = dirs["L_act"], dirs["L_evi"]
    print(f"[dirs] L_act={L_act} L_evi={L_evi} "
          f"cos(act,evi)={dirs['cos_action_evidence']:+.4f}")

    print(f"[load] {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True); model.eval()
    device = next(model.parameters()).device
    a_dir = dirs["action_dir"].to(device=device, dtype=torch.float32)
    e_dir = dirs["evidence_dir"].to(device=device, dtype=torch.float32)
    print(f"[ok] device={device}")

    per_dataset = {}
    for ds_name, pairs_path in DATASETS.items():
        rows = [json.loads(l) for l in open(pairs_path)]
        rows = [r for r in rows
                if (r.get("condition") or r.get("condition_id")) in CONDITIONS]
        if args.limit:
            kept, ct = [], defaultdict(int)
            for r in rows:
                c = r.get("condition") or r.get("condition_id")
                if ct[c] < args.limit:
                    kept.append(r); ct[c] += 1
            rows = kept

        # logit-level reference (from C1 eval jsonl, same prompts)
        c1_path = Path(args.c1_dir) / f"eval_{ds_name}.jsonl"
        c1_by_id = {}
        if c1_path.exists():
            for l in open(c1_path):
                r = json.loads(l)
                c1_by_id[(r["sample_id"], r.get("condition"))] = r

        out_jsonl = out_dir / f"step1_{ds_name}.jsonl"
        per_records = []
        with open(out_jsonl, "w") as f, torch.no_grad():
            for i, rec in enumerate(rows, 1):
                msgs = build_messages(rec["question"], rec["obs"],
                                      prompt_variant="v1", obs_style="factcard")
                prompt_str = apply_chat_template_safe(tok, msgs,
                                                      add_generation_prompt=True)
                ids = tok.encode(prompt_str, return_tensors="pt",
                                 add_special_tokens=False).to(device)
                attn = torch.ones_like(ids)
                out = model(ids, attention_mask=attn, output_hidden_states=True,
                            use_cache=False)
                # hidden_states is tuple(len = n_layers + 1); index L_act gives
                # post-layer L_act-1 residual; convention here matches the
                # L52 (1-indexed) used in directions extraction.
                h_act = out.hidden_states[L_act][0, -1, :].float()
                h_evi = out.hidden_states[L_evi][0, -1, :].float()
                proj_act = float((h_act @ a_dir).item())
                proj_evi = float((h_evi @ e_dir).item())
                cross    = float((h_act @ e_dir).item())  # action-layer evi
                row = {
                    "sample_id": rec["sample_id"],
                    "condition": rec.get("condition") or rec.get("condition_id"),
                    "schema_type": rec.get("schema_type"),
                    "proj_action_at_Lact": proj_act,
                    "proj_evidence_at_Levi": proj_evi,
                    "proj_evidence_at_Lact": cross,
                    "norm_h_act": float(h_act.norm().item()),
                    "norm_h_evi": float(h_evi.norm().item()),
                }
                c1 = c1_by_id.get((rec["sample_id"], row["condition"]))
                if c1:
                    row.update({
                        "margin_label_logit":      c1.get("margin_label"),
                        "margin_first_token_logit": c1.get("margin_first_token"),
                        "first_action_token":      c1.get("first_action_token"),
                    })
                per_records.append(row)
                f.write(json.dumps(row) + "\n"); f.flush()
                if i % 25 == 0 or i == len(rows):
                    print(f"  [{ds_name} {i}/{len(rows)}]")

        # per-cell stats
        by_cond = defaultdict(list)
        for r in per_records:
            by_cond[r["condition"]].append(r)
        cells = {}
        for c in CONDITIONS:
            recs = by_cond.get(c, [])
            cells[c] = {
                "n": len(recs),
                "proj_action": cell_stats([r["proj_action_at_Lact"] for r in recs]),
                "proj_evidence_at_Lact": cell_stats(
                    [r["proj_evidence_at_Lact"] for r in recs]),
                "proj_evidence_at_Levi": cell_stats(
                    [r["proj_evidence_at_Levi"] for r in recs]),
                "margin_label_logit": cell_stats(
                    [r["margin_label_logit"] for r in recs
                     if r.get("margin_label_logit") is not None]),
            }
        # contrasts T0-N0 and S0-T0 on the action-projection
        contrasts = {}
        for a, b in [("T0", "N0"), ("S0", "T0"), ("S0", "N0")]:
            xa = np.array([r["proj_action_at_Lact"] for r in by_cond.get(a, [])])
            xb = np.array([r["proj_action_at_Lact"] for r in by_cond.get(b, [])])
            if xa.size and xb.size and xa.size == xb.size:
                contrasts[f"{a}_minus_{b}"] = {
                    "n_pairs": int(xa.size),
                    "mean_delta_proj_action": float((xa - xb).mean()),
                    "median_delta_proj_action": float(np.median(xa - xb)),
                }
        per_dataset[ds_name] = {
            "jsonl": str(out_jsonl),
            "n_records": len(per_records),
            "cells_proj_action_at_Lact": cells,
            "contrasts_proj_action": contrasts,
        }

    out = {
        "model_32b": args.model_path,
        "L_act": L_act, "L_evi": L_evi,
        "cos_action_evidence": dirs["cos_action_evidence"],
        "datasets": per_dataset,
        "note": ("p0 = last input token position of the chat-templated prompt "
                 "(decision-point activation). proj_* are dot products against "
                 "unit-norm action_dir / evidence_dir saved in directions.npz. "
                 "Reference 7B step-1 numbers in this codebase are logit-level "
                 "(margin_label) only; residual-stream comparison would require "
                 "re-running 7B, which is out of C2 scope."),
    }
    (out_dir / "step1_action_margins.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] -> {out_dir}/step1_action_margins.json")


if __name__ == "__main__":
    main()
