#!/usr/bin/env python3
"""Split the 91% full-L20 patch recovery into components.

On the same N=100 (sf_tm, dist_tm) pairs, patch sf_tm -> dist_tm at the last
token, varying the site:
  C1 pre_only       : layers[19] output        (= pre-L20 residual)
  C2 attn_only      : layers[20].self_attn out (= L20 attention output)
  C3 pre_plus_attn  : C1 + C2 simultaneously   (= L20 post-attn residual
                                                  AND, since MLP is pointwise,
                                                  equivalent to full L20 output
                                                  -- used as additivity check)
  C4 mlp_only       : layers[20].mlp out       (negative control, known ~0%)
  C5 full_layer     : layers[20] output        (positive control, known ~91%)

Primary metric: per-sample recovery = (margin_patched - margin_dist_tm) /
                                      (margin_sf_tm - margin_dist_tm)

Output:
  results/task_missingness_L20_split/patch_results.jsonl
  results/task_missingness_L20_split/summary.json
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import get_model_layers


def make_margin_ids(tok):
    tool = [tok.encode(t, add_special_tokens=False)[0]
            for t in ACTION_TOKENS["tool_call"] if tok.encode(t, add_special_tokens=False)]
    fin  = [tok.encode(t, add_special_tokens=False)[0]
            for t in ACTION_TOKENS["finish"]    if tok.encode(t, add_special_tokens=False)]
    return tool, fin


def margin_from_logits(logits_last, tool_ids, fin_ids):
    lp = torch.log_softmax(logits_last.float(), dim=-1)
    tl = torch.logsumexp(lp[tool_ids], 0).item()
    fl = torch.logsumexp(lp[fin_ids],  0).item()
    return float(tl - fl)


def _normalize_msgs_for_template(messages, model_path: str):
    """Mirror of run_nonqa_react_meeting.normalize_messages_for_model, copied
    here to keep this circuit-patching script free of cross-package imports."""
    p = (model_path or "").lower()
    if "gemma" in p or "mistral" in p:
        sys_chunks = [m["content"] for m in messages if m.get("role") == "system"]
        sys_text = "\n\n".join(c for c in sys_chunks if c).strip()
        out = []
        injected = False
        for m in messages:
            if m.get("role") == "system":
                continue
            if not injected and m.get("role") == "user":
                content = m.get("content", "")
                content = (sys_text + "\n\n" + content) if sys_text else content
                out.append({"role": "user", "content": content})
                injected = True
            else:
                out.append(m)
        if not injected and sys_text:
            out.insert(0, {"role": "user", "content": sys_text})
        return out
    return messages


def build_prompt(builder, tok, rec, model_path: str = ""):
    steps = [{"action": "search", "action_input": f"about: {rec['question'][:80]}",
              "observation": rec["obs"]}]
    msgs = builder.build_full_prompt(rec["question"], steps)
    msgs = _normalize_msgs_for_template(msgs, model_path)
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class MultiSitePatcher:
    """Forward hooks on N modules. For each site, capture last-token activation
    on every call; if a patch_vec is set for that site, overwrite last token
    with patch_vec before returning output."""
    def __init__(self, sites):
        self.sites = sites                      # {name: module}
        self.captured = {}                      # {name: np.ndarray}
        self.fire_counts = {n: 0 for n in sites}
        self.patch_vecs = {}                    # {name: np.ndarray} if set
        self.handles = []

    def _make_hook(self, name):
        def hook(mod, inp, out):
            is_tuple = isinstance(out, tuple)
            h = out[0] if is_tuple else out
            self.captured[name] = h[0, -1, :].detach().float().cpu().numpy().copy()
            self.fire_counts[name] += 1
            if name in self.patch_vecs and self.patch_vecs[name] is not None:
                pv = torch.from_numpy(self.patch_vecs[name]).to(device=h.device, dtype=h.dtype)
                h2 = h.clone(); h2[0, -1, :] = pv
                return (h2,) + out[1:] if is_tuple else h2
            return out
        return hook

    def __enter__(self):
        for name, module in self.sites.items():
            self.handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, *a):
        for h in self.handles: h.remove()
        self.handles = []

    def reset_run(self):
        self.captured = {}
        self.fire_counts = {n: 0 for n in self.sites}
        self.patch_vecs = {}


def do_forward(model, tok, prompt, device, tool_ids, fin_ids, patcher):
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(ids)
    return margin_from_logits(out.logits[0, -1, :], tool_ids, fin_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/anti_cue_tm_n100/pairs.jsonl")
    ap.add_argument("--out-dir", default="results/task_missingness_L20_split")
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in open(args.pairs)]
    sids = sorted(set(r["sample_id"] for r in records))
    if args.limit: sids = sids[:args.limit]
    # keep only sf_tm and dist_tm cells
    need = {("sf","task_missingness"), ("distractor","task_missingness")}
    by_sid = {s: {} for s in sids}
    for r in records:
        if r["sample_id"] in by_sid and (r["target"], r["cue"]) in need:
            by_sid[r["sample_id"]][(r["target"], r["cue"])] = r
    print(f"[info] {len(sids)} samples")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    p = args.model_path.lower()
    attn_impl = "eager" if "gemma" in p else "sdpa"
    print(f"[info] attn_implementation={attn_impl}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype,
                                                 device_map="auto", trust_remote_code=True,
                                                 attn_implementation=attn_impl)
    model.eval(); device = next(model.parameters()).device
    layers = get_model_layers(model)
    SITES = {
        "resid_pre_L20":  layers[19],                # pre-L20 residual
        "attn_out_L20":   layers[20].self_attn,      # L20 attention output
        "mlp_out_L20":    layers[20].mlp,            # L20 MLP output
        "layer_out_L20":  layers[20],                # full L20 output
    }
    tool_ids, fin_ids = make_margin_ids(tok)
    builder = PromptBuilder()

    run_all(sids, by_sid, model, tok, builder, device, tool_ids, fin_ids,
            SITES, out_dir, model_path=args.model_path)


CONDS = [
    ("pre_only",      ["resid_pre_L20"]),
    ("attn_only",     ["attn_out_L20"]),
    ("pre_plus_attn", ["resid_pre_L20", "attn_out_L20"]),
    ("mlp_only",      ["mlp_out_L20"]),
    ("full_layer",    ["layer_out_L20"]),
]


def run_all(sids, by_sid, model, tok, builder, device, tool_ids, fin_ids,
            SITES, out_dir, model_path: str = ""):
    patcher = MultiSitePatcher(SITES)
    natural = {s: {} for s in sids}                                     # sid -> cell -> {...}

    # Stage 1: natural captures for sf_tm and dist_tm (multi-site)
    t0 = time.time()
    for i, s in enumerate(sids):
        for cell in [("sf", "task_missingness"), ("distractor", "task_missingness")]:
            prompt = build_prompt(builder, tok, by_sid[s][cell], model_path)
            patcher.reset_run()
            with patcher:
                m = do_forward(model, tok, prompt, device, tool_ids, fin_ids, patcher)
            natural[s][cell] = {"margin": m, "prompt": prompt,
                                "activations": dict(patcher.captured),
                                "fired": dict(patcher.fire_counts)}
        if (i + 1) % 25 == 0 or i + 1 == len(sids):
            print(f"  [stage1 {i+1}/{len(sids)}] {time.time()-t0:.1f}s")

    # Stage 2: patches sf_tm -> dist_tm for each condition
    rows_path = out_dir / "patch_results.jsonl"
    n_written = 0; t0 = time.time()
    with open(rows_path, "w") as f:
        for i, s in enumerate(sids):
            src_cell = ("sf", "task_missingness")
            tgt_cell = ("distractor", "task_missingness")
            src = natural[s][src_cell]; tgt = natural[s][tgt_cell]
            for cname, site_list in CONDS:
                patcher.reset_run()
                for site in site_list:
                    patcher.patch_vecs[site] = src["activations"][site]
                with patcher:
                    m_p = do_forward(model, tok, tgt["prompt"], device, tool_ids, fin_ids, patcher)
                comp_abs = {site: float(np.mean(np.abs(
                    src["activations"][site] - patcher.captured[site])))
                    for site in site_list}
                row = {
                    "sample_id": s, "condition": cname, "sites": site_list,
                    "margin_source_sf_tm": src["margin"],
                    "margin_target_dist_tm": tgt["margin"],
                    "margin_patched": m_p,
                    "delta_margin": m_p - tgt["margin"],
                    "locality_gap": src["margin"] - tgt["margin"],
                    "component_abs_diff_mean": comp_abs,
                    "fire_counts": dict(patcher.fire_counts),
                    "action_target_natural": "search" if tgt["margin"] > 0 else "stop",
                    "action_patched":        "search" if m_p > 0 else "stop",
                }
                f.write(json.dumps(row) + "\n"); f.flush(); n_written += 1
            if (i + 1) % 20 == 0 or i + 1 == len(sids):
                print(f"  [stage2 {i+1}/{len(sids)}] {time.time()-t0:.1f}s  rows={n_written}")
    print(f"[wrote] {rows_path}  ({n_written} rows)")

    summarize(rows_path, out_dir / "summary.json")


def perm_p_paired(d, n=20000, seed=7):
    d = np.asarray(d, dtype=np.float64); rng = np.random.default_rng(seed)
    null = (rng.choice([-1.0, 1.0], size=(n, len(d))) * d).mean(axis=1)
    return float((np.abs(null) >= abs(d.mean())).mean())


def boot_ci(x, n=20000, seed=1):
    x = np.asarray(x, dtype=np.float64); rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


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
            "n": len(rs), "sites": rs[0]["sites"],
            "delta_margin_mean": float(d.mean()),
            "delta_margin_median": float(np.median(d)),
            "delta_margin_ci95": [lo, hi],
            "perm_p_paired_two_sided": perm_p_paired(d),
            "recovery_pos_gap_n": int(mask.sum()),
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
    print(f"\n=== L20 split recovery (sf_tm -> dist_tm, N={len(base_rows)}) ===")
    print(f"natural gap = {summary['natural_baseline']['locality_gap_mean']:+.3f}")
    print(f"{'cond':16s} {'N':>3s}  {'Δmargin':>8s}  {'CI95':>22s}  {'perm_p':>7s}  {'recovery':>8s}  {'rec_CI':>22s}  flips(s→S/S→s)")
    for cond, s in summary["conditions"].items():
        ci = s["delta_margin_ci95"]; rci = s["recovery_pos_gap_ci95"]
        print(f"{cond:16s} {s['n']:>3d}  {s['delta_margin_mean']:+8.3f}  "
              f"[{ci[0]:+7.3f},{ci[1]:+7.3f}]  {s['perm_p_paired_two_sided']:.4f}  "
              f"{s['recovery_pos_gap_mean']:+8.3f}  [{rci[0]:+7.3f},{rci[1]:+7.3f}]  "
              f"{s['flip_stop_to_search']:>2d}/{s['flip_search_to_stop']:>2d}")


if __name__ == "__main__":
    main()
