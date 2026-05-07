#!/usr/bin/env python3
"""Split attn_L18's +0.471 contribution by KV group.

Patches sf_tm -> dist_tm at the LAST TOKEN, replacing the o_proj input columns
of one KV group at a time:
  KV g -> columns [g*7*128 : (g+1)*7*128]   (Qwen2.5-7B GQA: 28 Q-heads, 4 KV groups of 7)

Conditions:
  KV0_only / KV1_only / KV2_only / KV3_only : replace one group's slice
  all_KV  : replace all 28 heads' columns  (= attn_only sanity, should match +0.471)

Captures via forward_pre_hook on layers[18].self_attn.o_proj.

Output:
  results/task_missingness_L18_kv_groups/patch_results.jsonl
  results/task_missingness_L18_kv_groups/summary.json
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.prompts import PromptBuilder
from steering.hook_utils import get_model_layers
from scripts.patch_L20_localise_full_residual import (
    make_margin_ids, build_prompt, perm_p_paired, boot_ci,
)


N_HEADS, N_KV, HEAD_DIM = 28, 4, 128
HEADS_PER_KV = N_HEADS // N_KV       # 7
SLICE = HEADS_PER_KV * HEAD_DIM      # 896


class OProjKVPatcher:
    """Pre-hook on layers[L].self_attn.o_proj. Captures last-token input
    (= concat of per-head Q-attended values). On patch, replaces last-token
    input columns for the chosen KV group(s) with stored donor values."""
    def __init__(self, model, layer):
        self.o_proj = get_model_layers(model)[layer].self_attn.o_proj
        self.captured = None
        self.fire_count = 0
        self.patch_vec = None        # full [hidden_dim] donor concat
        self.patch_groups = None     # iterable of group ids to overwrite
        self.handle = None

    def _hook(self, mod, inp):
        x = inp[0]                   # (B, T, n_heads*head_dim)
        self.captured = x[0, -1, :].detach().float().cpu().numpy().copy()
        self.fire_count += 1
        if self.patch_vec is not None and self.patch_groups:
            x2 = x.clone()
            pv = torch.from_numpy(self.patch_vec).to(device=x.device, dtype=x.dtype)
            for g in self.patch_groups:
                s = g * SLICE; e = s + SLICE
                x2[0, -1, s:e] = pv[s:e]
            return (x2,) + inp[1:]
        return inp

    def __enter__(self):
        self.handle = self.o_proj.register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, *a):
        self.handle.remove(); self.handle = None

    def reset(self):
        self.captured = None; self.fire_count = 0
        self.patch_vec = None; self.patch_groups = None


def margin_from_logits(logits_last, tool_ids, fin_ids):
    lp = torch.log_softmax(logits_last.float(), dim=-1)
    return float(torch.logsumexp(lp[tool_ids], 0).item()
                 - torch.logsumexp(lp[fin_ids], 0).item())


def fwd_margin(model, tok, prompt, device, tool_ids, fin_ids):
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(ids)
    return margin_from_logits(out.logits[0, -1, :], tool_ids, fin_ids)


CONDS = [
    ("KV0_only", [0]),
    ("KV1_only", [1]),
    ("KV2_only", [2]),
    ("KV3_only", [3]),
    ("all_KV",   [0, 1, 2, 3]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/anti_cue_tm_n100/pairs.jsonl")
    ap.add_argument("--out-dir", default="results/task_missingness_L18_kv_groups")
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--mismatched-perm", default=None,
                    help="JSON with {'map': {sid: donor_sid}}; if set, donor sf_tm "
                         "comes from map[sid] instead of from sid (specificity control)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(l) for l in open(args.pairs)]
    sids = sorted(set(r["sample_id"] for r in records))
    if args.limit: sids = sids[:args.limit]
    need = {("sf","task_missingness"), ("distractor","task_missingness")}
    by_sid = {s: {} for s in sids}
    for r in records:
        if r["sample_id"] in by_sid and (r["target"], r["cue"]) in need:
            by_sid[r["sample_id"]][(r["target"], r["cue"])] = r
    print(f"[info] {len(sids)} samples, layer={args.layer}")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype,
                                                 device_map="auto", trust_remote_code=True)
    model.eval(); device = next(model.parameters()).device
    tool_ids, fin_ids = make_margin_ids(tok)
    builder = PromptBuilder()
    patcher = OProjKVPatcher(model, args.layer)

    natural = {s: {} for s in sids}
    t0 = time.time()
    for i, s in enumerate(sids):
        for cell in [("sf", "task_missingness"), ("distractor", "task_missingness")]:
            prompt = build_prompt(builder, tok, by_sid[s][cell])
            patcher.reset()
            with patcher:
                m = fwd_margin(model, tok, prompt, device, tool_ids, fin_ids)
            natural[s][cell] = {"margin": m, "prompt": prompt,
                                "kv_concat": patcher.captured}
        if (i + 1) % 25 == 0 or i + 1 == len(sids):
            print(f"  [stage1 {i+1}/{len(sids)}] {time.time()-t0:.1f}s")

    donor_map = None
    if args.mismatched_perm:
        with open(args.mismatched_perm) as fp:
            donor_map = json.load(fp).get("map", {})
        missing = [s for s in sids if s not in donor_map or donor_map[s] not in natural]
        if missing:
            raise SystemExit(f"mismatched-perm missing donors for {len(missing)} sids "
                             f"(first: {missing[:3]})")
        print(f"[info] mismatched-donor mode: {len(donor_map)} entries from {args.mismatched_perm}")

    rows_path = out_dir / "patch_results.jsonl"
    n_written = 0; t0 = time.time()
    with open(rows_path, "w") as f:
        for i, s in enumerate(sids):
            donor_sid = donor_map[s] if donor_map else s
            src = natural[donor_sid][("sf", "task_missingness")]
            tgt = natural[s][("distractor", "task_missingness")]
            for cname, groups in CONDS:
                patcher.reset()
                patcher.patch_vec = src["kv_concat"]; patcher.patch_groups = groups
                with patcher:
                    m_p = fwd_margin(model, tok, tgt["prompt"], device, tool_ids, fin_ids)
                row = {
                    "sample_id": s, "donor_sid": donor_sid,
                    "condition": cname, "groups": groups,
                    "margin_source_sf_tm": src["margin"],
                    "margin_target_dist_tm": tgt["margin"],
                    "margin_patched": m_p,
                    "delta_margin": m_p - tgt["margin"],
                    "locality_gap": src["margin"] - tgt["margin"],
                    "action_target_natural": "search" if tgt["margin"] > 0 else "stop",
                    "action_patched":        "search" if m_p > 0 else "stop",
                }
                f.write(json.dumps(row) + "\n"); f.flush(); n_written += 1
            if (i + 1) % 20 == 0 or i + 1 == len(sids):
                print(f"  [stage2 {i+1}/{len(sids)}] {time.time()-t0:.1f}s  rows={n_written}")
    print(f"[wrote] {rows_path}  ({n_written} rows)")
    summarize(rows_path, out_dir / "summary.json")


def summarize(in_path, out_json):
    rows = [json.loads(l) for l in open(in_path)]
    by_cond = {}
    for r in rows: by_cond.setdefault(r["condition"], []).append(r)
    summary = {"conditions": {}}
    for cond, rs in by_cond.items():
        d = np.array([r["delta_margin"] for r in rs])
        gaps = np.array([r["locality_gap"] for r in rs])
        mask = gaps > 0.5
        rec = np.array([r["delta_margin"] / r["locality_gap"]
                        if abs(r["locality_gap"]) > 0.01 else np.nan for r in rs])
        rec_pos = rec[mask & np.isfinite(rec)]
        lo, hi = boot_ci(d)
        rlo, rhi = (boot_ci(rec_pos) if len(rec_pos) >= 5 else (float("nan"), float("nan")))
        flips_s2s = sum(1 for r in rs if r["action_target_natural"] == "stop"   and r["action_patched"] == "search")
        flips_se2st = sum(1 for r in rs if r["action_target_natural"] == "search" and r["action_patched"] == "stop")
        summary["conditions"][cond] = {
            "n": len(rs), "groups": rs[0]["groups"],
            "delta_margin_mean": float(d.mean()),
            "delta_margin_median": float(np.median(d)),
            "delta_margin_ci95": [lo, hi],
            "perm_p_paired_two_sided": perm_p_paired(d),
            "recovery_pos_gap_mean":   float(rec_pos.mean())   if len(rec_pos) else float("nan"),
            "recovery_pos_gap_median": float(np.median(rec_pos)) if len(rec_pos) else float("nan"),
            "recovery_pos_gap_ci95":   [rlo, rhi],
            "flip_stop_to_search": flips_s2s,
            "flip_search_to_stop": flips_se2st,
        }
    base_rows = by_cond[list(by_cond)[0]]
    summary["natural_baseline"] = {
        "margin_sf_tm_mean":   float(np.mean([r["margin_source_sf_tm"]   for r in base_rows])),
        "margin_dist_tm_mean": float(np.mean([r["margin_target_dist_tm"] for r in base_rows])),
        "locality_gap_mean":   float(np.mean([r["locality_gap"]          for r in base_rows])),
    }
    json.dump(summary, open(out_json, "w"), indent=2)
    print(f"[wrote] {out_json}")
    print(f"\n=== L18 KV-group split (sf_tm -> dist_tm, N={len(base_rows)}) ===")
    print(f"natural gap = {summary['natural_baseline']['locality_gap_mean']:+.3f}")
    print(f"{'cond':12s} {'Δmargin':>8s}  {'CI95':>22s}  {'perm_p':>7s}  {'recovery':>8s}  flips(s→S/S→s)")
    for cond, s in summary["conditions"].items():
        ci = s["delta_margin_ci95"]
        print(f"{cond:12s} {s['delta_margin_mean']:+8.3f}  "
              f"[{ci[0]:+7.3f},{ci[1]:+7.3f}]  {s['perm_p_paired_two_sided']:.4f}  "
              f"{s['recovery_pos_gap_mean']:+8.3f}  "
              f"{s['flip_stop_to_search']:>2d}/{s['flip_search_to_stop']:>2d}")


if __name__ == "__main__":
    main()
