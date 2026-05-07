#!/usr/bin/env python3
"""Minimal causal patch: is mlp_L20 output sufficient/necessary to carry the
task_missingness locality effect?

Four conditions per sample:
  1. suff    : source=sf_task_missingness      -> target=dist_task_missingness
  2. reverse : source=dist_task_missingness    -> target=sf_task_missingness
  3. neutral : source=sf_neutral               -> target=dist_neutral
  4. mism    : source=sf_task_missingness[perm] -> target=dist_task_missingness
Patch location: layers[20].mlp output, last-token position only.
Metric: first-token search-vs-stop margin.

Output:
  results/task_missingness_mlp20_patch/patch_results.jsonl
  results/task_missingness_mlp20_patch/summary.json
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
    tl = torch.logsumexp(lp[tool_ids], 0).item() if tool_ids else -100.0
    fl = torch.logsumexp(lp[fin_ids],  0).item() if fin_ids  else -100.0
    return float(tl - fl)


def build_prompt(builder, tok, rec):
    steps = [{"action": "search", "action_input": f"about: {rec['question'][:80]}",
              "observation": rec["obs"]}]
    msgs = builder.build_full_prompt(rec["question"], steps)
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class MLPPatcher:
    """Overwrite layers[20].mlp output at the last token with patch_vec."""
    def __init__(self, mlp_module):
        self.mlp = mlp_module
        self.patch_vec = None
        self.captured_original_last = None
        self.fire_count = 0
        self.handle = None

    def _hook(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        self.captured_original_last = h[0, -1, :].detach().float().cpu().numpy().copy()
        self.fire_count += 1
        if self.patch_vec is None:
            return out
        pv = self.patch_vec.to(device=h.device, dtype=h.dtype)
        if isinstance(out, tuple):
            h2 = out[0].clone(); h2[0, -1, :] = pv
            return (h2,) + out[1:]
        h2 = out.clone(); h2[0, -1, :] = pv
        return h2

    def __enter__(self):
        self.handle = self.mlp.register_forward_hook(self._hook); return self

    def __exit__(self, *a):
        self.handle.remove(); self.handle = None


def natural_capture(model, tok, prompt, device, tool_ids, fin_ids, mlp_module):
    cap = {"mlp_out_last": None, "margin": None}
    def h(m, i, o):
        x = o[0] if isinstance(o, tuple) else o
        cap["mlp_out_last"] = x[0, -1, :].detach().float().cpu().numpy().copy()
    handle = mlp_module.register_forward_hook(h)
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(ids)
    handle.remove()
    cap["margin"] = margin_from_logits(out.logits[0, -1, :], tool_ids, fin_ids)
    return cap


def patched_forward(model, tok, prompt, device, tool_ids, fin_ids, patcher, patch_vec_np):
    patcher.patch_vec = torch.from_numpy(patch_vec_np)
    patcher.fire_count = 0
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    with patcher:
        with torch.no_grad():
            out = model(ids)
    margin = margin_from_logits(out.logits[0, -1, :], tool_ids, fin_ids)
    orig_last = patcher.captured_original_last
    patched_last = patch_vec_np
    delta_abs_mean = float(np.mean(np.abs(patched_last - orig_last)))
    patcher.patch_vec = None
    return {"margin_patched": margin,
            "hook_fired": int(patcher.fire_count),
            "component_delta_abs_mean": delta_abs_mean,
            "source_norm": float(np.linalg.norm(patch_vec_np)),
            "target_orig_norm": float(np.linalg.norm(orig_last))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/anti_cue_tm_n100/pairs.jsonl")
    ap.add_argument("--out-dir", default="results/task_missingness_mlp20_patch")
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--perm-seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in open(args.pairs)]
    sids = sorted(set(r["sample_id"] for r in records))
    if args.limit:
        sids = sids[:args.limit]
        records = [r for r in records if r["sample_id"] in set(sids)]
    print(f"[info] {len(sids)} samples, {len(records)} records")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype,
                                                 device_map="auto", trust_remote_code=True)
    model.eval(); device = next(model.parameters()).device
    layers = get_model_layers(model); mlp = layers[args.layer].mlp
    tool_ids, fin_ids = make_margin_ids(tok)
    builder = PromptBuilder()

    run_all(records, sids, model, tok, builder, device, mlp, tool_ids, fin_ids,
            out_dir, args.perm_seed)


def run_all(records, sids, model, tok, builder, device, mlp, tool_ids, fin_ids,
            out_dir, perm_seed):
    # Stage 1: natural capture for all 4 cells of every sample
    by_sid = {s: {} for s in sids}
    for r in records:
        by_sid[r["sample_id"]][(r["target"], r["cue"])] = r
    CELLS = [("sf", "task_missingness"), ("distractor", "task_missingness"),
             ("sf", "neutral"),         ("distractor", "neutral")]
    natural = {s: {} for s in sids}
    t0 = time.time()
    for i, s in enumerate(sids):
        for cell in CELLS:
            rec = by_sid[s][cell]
            prompt = build_prompt(builder, tok, rec)
            cap = natural_capture(model, tok, prompt, device, tool_ids, fin_ids, mlp)
            natural[s][cell] = {"margin": cap["margin"],
                                "mlp_out_last": cap["mlp_out_last"],
                                "prompt": prompt}
        if (i + 1) % 25 == 0 or i + 1 == len(sids):
            print(f"  [stage1 {i+1}/{len(sids)}] {time.time()-t0:.1f}s")

    # Fixed permutation for mismatched donor: perm[i] != i
    rng = np.random.default_rng(perm_seed)
    perm = np.arange(len(sids))
    while True:
        rng.shuffle(perm)
        if not any(perm[i] == i for i in range(len(sids))):
            break
    donor = {sids[i]: sids[perm[i]] for i in range(len(sids))}
    json.dump({"seed": perm_seed, "map": donor},
              open(out_dir / "mismatched_donor_map.json", "w"), indent=2)

    # Stage 2: patches
    patcher = MLPPatcher(mlp)
    CONDS = [
        ("suff",    ("sf", "task_missingness"),     ("distractor", "task_missingness")),
        ("reverse", ("distractor", "task_missingness"), ("sf", "task_missingness")),
        ("neutral", ("sf", "neutral"),              ("distractor", "neutral")),
        ("mism",    ("sf", "task_missingness"),     ("distractor", "task_missingness")),
    ]
    out_path = out_dir / "patch_results.jsonl"
    n_written = 0; t0 = time.time()
    with open(out_path, "w") as f:
        for i, s in enumerate(sids):
            for name, src_cell, tgt_cell in CONDS:
                donor_sid = donor[s] if name == "mism" else s
                src = natural[donor_sid][src_cell]
                tgt = natural[s][tgt_cell]
                patch_res = patched_forward(model, tok, tgt["prompt"], device,
                                            tool_ids, fin_ids, patcher,
                                            src["mlp_out_last"])
                margin_src = natural[s][src_cell]["margin"]
                margin_tgt = tgt["margin"]
                m_p = patch_res["margin_patched"]
                row = {
                    "sample_id": s,
                    "condition": name,
                    "source_sample_id": donor_sid,
                    "source_cell": f"{src_cell[0]}_{src_cell[1]}",
                    "target_cell": f"{tgt_cell[0]}_{tgt_cell[1]}",
                    "margin_source": margin_src,
                    "margin_target_natural": margin_tgt,
                    "margin_patched": m_p,
                    "delta_margin": m_p - margin_tgt,
                    "action_source":   "search" if margin_src > 0 else "stop",
                    "action_target_natural": "search" if margin_tgt > 0 else "stop",
                    "action_patched":  "search" if m_p > 0 else "stop",
                    "margin_sf_tm":    natural[s][("sf", "task_missingness")]["margin"],
                    "margin_dist_tm":  natural[s][("distractor", "task_missingness")]["margin"],
                    "margin_sf_neutral":   natural[s][("sf", "neutral")]["margin"],
                    "margin_dist_neutral": natural[s][("distractor", "neutral")]["margin"],
                    "hook_fired":      patch_res["hook_fired"],
                    "component_delta_abs_mean": patch_res["component_delta_abs_mean"],
                    "source_norm":     patch_res["source_norm"],
                    "target_orig_norm": patch_res["target_orig_norm"],
                    "parse_status":    "ok",
                }
                f.write(json.dumps(row) + "\n"); f.flush(); n_written += 1
            if (i + 1) % 10 == 0 or i + 1 == len(sids):
                print(f"  [stage2 {i+1}/{len(sids)}] {time.time()-t0:.1f}s  rows={n_written}")
    print(f"[wrote] {out_path}  ({n_written} rows)")

    # Summary
    summarize_patches(out_path, out_dir / "summary.json")


def perm_p_paired(d, n=20000, seed=7):
    d = np.asarray(d, dtype=np.float64); rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    null = (rng.choice([-1.0, 1.0], size=(n, len(d))) * d).mean(axis=1)
    return float((np.abs(null) >= obs).mean())


def boot_ci(x, n=20000, seed=1):
    x = np.asarray(x, dtype=np.float64); rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def wilcoxon_p(x):
    try:
        from scipy.stats import wilcoxon
        x = np.asarray(x); x = x[x != 0]
        if len(x) < 3: return float("nan")
        return float(wilcoxon(x, alternative="two-sided", zero_method="wilcox").pvalue)
    except Exception:
        return float("nan")


def sign_test_p(x):
    try:
        from scipy.stats import binomtest
        x = np.asarray(x); x = x[x != 0]
        n = len(x); k = int((x > 0).sum())
        if n == 0: return float("nan")
        return float(binomtest(k, n, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def flip_counts(actions_before, actions_after):
    s2s = sum(1 for a, b in zip(actions_before, actions_after) if a == "stop"   and b == "search")
    se2st = sum(1 for a, b in zip(actions_before, actions_after) if a == "search" and b == "stop")
    return {"stop_to_search": s2s, "search_to_stop": se2st,
            "ndSR_before": sum(1 for a in actions_before if a == "search") / len(actions_before),
            "ndSR_after":  sum(1 for a in actions_after  if a == "search") / len(actions_after)}


def summarize_patches(in_path, out_json):
    rows = [json.loads(l) for l in open(in_path)]
    by_cond = {}
    for r in rows:
        by_cond.setdefault(r["condition"], []).append(r)

    summary = {"conditions": {}}
    for cond, rs in by_cond.items():
        d = np.array([r["delta_margin"] for r in rs])
        m_tgt = np.array([r["margin_target_natural"] for r in rs])
        m_p   = np.array([r["margin_patched"] for r in rs])
        gaps  = np.array([r["margin_sf_tm"] - r["margin_dist_tm"] for r in rs])
        hook_ok = all(r["hook_fired"] >= 1 for r in rs)
        comp_dam = [r["component_delta_abs_mean"] for r in rs]
        # recovery / loss computed per-sample with robust handling
        if cond == "suff" or cond == "mism":
            rec = np.array([(r["margin_patched"] - r["margin_dist_tm"]) /
                            (r["margin_sf_tm"] - r["margin_dist_tm"]) for r in rs])
            rec_label = "sufficiency_recovery"
        elif cond == "reverse":
            rec = np.array([(r["margin_sf_tm"] - r["margin_patched"]) /
                            (r["margin_sf_tm"] - r["margin_dist_tm"]) for r in rs])
            rec_label = "reverse_loss"
        else:
            rec = np.full(len(rs), np.nan); rec_label = "n/a"
        mask_pos_gap = gaps > 0.5
        rec_pos = rec[mask_pos_gap & np.isfinite(rec)]
        rec_all_finite = rec[np.isfinite(rec)]
        flips = flip_counts([r["action_target_natural"] for r in rs],
                            [r["action_patched"] for r in rs])
        lo, hi = boot_ci(d)
        summary["conditions"][cond] = {
            "n": len(rs),
            "source_cell": rs[0]["source_cell"], "target_cell": rs[0]["target_cell"],
            "margin_target_mean_before": float(m_tgt.mean()),
            "margin_target_mean_after":  float(m_p.mean()),
            "delta_margin_mean": float(d.mean()),
            "delta_margin_median": float(np.median(d)),
            "delta_margin_ci95": [lo, hi],
            "perm_p_paired_two_sided": perm_p_paired(d),
            "wilcoxon_p_two_sided": wilcoxon_p(d),
            "sign_test_p_two_sided": sign_test_p(d),
            "n_pos_delta": int((d > 0).sum()), "n_neg_delta": int((d < 0).sum()),
            "flip_counts": flips,
            "recovery_label": rec_label,
            "recovery_all_finite_mean":    float(np.nanmean(rec_all_finite)) if len(rec_all_finite) else float("nan"),
            "recovery_all_finite_median":  float(np.nanmedian(rec_all_finite)) if len(rec_all_finite) else float("nan"),
            "recovery_all_finite_ci95":    list(boot_ci(rec_all_finite)) if len(rec_all_finite) >= 5 else [float("nan"), float("nan")],
            "recovery_pos_gap_n":          int(mask_pos_gap.sum()),
            "recovery_pos_gap_mean":       float(rec_pos.mean()) if len(rec_pos) else float("nan"),
            "recovery_pos_gap_median":     float(np.median(rec_pos)) if len(rec_pos) else float("nan"),
            "recovery_pos_gap_ci95":       list(boot_ci(rec_pos)) if len(rec_pos) >= 5 else [float("nan"), float("nan")],
            "hook_all_fired": bool(hook_ok),
            "component_delta_abs_mean_mean": float(np.mean(comp_dam)),
        }
    # Natural baselines from first row per sample of any condition
    base = {}
    ref_rows = by_cond[list(by_cond)[0]]
    base["margin_sf_tm_mean"]        = float(np.mean([r["margin_sf_tm"] for r in ref_rows]))
    base["margin_dist_tm_mean"]      = float(np.mean([r["margin_dist_tm"] for r in ref_rows]))
    base["margin_sf_neutral_mean"]   = float(np.mean([r["margin_sf_neutral"] for r in ref_rows]))
    base["margin_dist_neutral_mean"] = float(np.mean([r["margin_dist_neutral"] for r in ref_rows]))
    base["natural_locality_gap_mean"] = base["margin_sf_tm_mean"] - base["margin_dist_tm_mean"]
    summary["natural_baseline"] = base
    json.dump(summary, open(out_json, "w"), indent=2)
    print(f"[wrote] {out_json}")
    # brief console print
    print("\n=== mlp_L20 patch summary ===")
    print(f"natural locality gap (sf_tm - dist_tm) = {base['natural_locality_gap_mean']:+.3f}")
    for cond, s in summary["conditions"].items():
        print(f"  {cond:8s} n={s['n']:3d} src={s['source_cell']:20s} tgt={s['target_cell']:20s}"
              f"  Δmargin={s['delta_margin_mean']:+.3f} CI[{s['delta_margin_ci95'][0]:+.3f},{s['delta_margin_ci95'][1]:+.3f}]"
              f"  permp={s['perm_p_paired_two_sided']:.4g}"
              f"  {s['recovery_label']}(pos_gap)={s['recovery_pos_gap_mean']:+.3f}")


if __name__ == "__main__":
    main()
