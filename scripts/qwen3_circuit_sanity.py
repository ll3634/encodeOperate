#!/usr/bin/env python3
"""Qwen3-32B single-layer circuit sanity check.

Mirror of Gemma/Mistral circuit sanity experiments (no KV decomposition):
  Exp 1: Sparse residual formation sweep (mirror Gemma Exp 1)
  Exp 2: Same-layer action vs evidence gain (mirror Mistral Exp 2)

Single model load; runs L_peak identification, then both experiments.
BF16 only, no quantization. No per-head or KV splits.

Output: results/cross_model_qwen3_32b/circuit_sanity/
"""
import argparse, json, random, sys, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS          # noqa: E402
from steering.hook_utils import get_model_layers, SteeringHook, compute_rms  # noqa: E402
from scripts.cross_model_full import (                           # noqa: E402
    collect_step1_states, collect_popqa_multilayer,
    extract_action_dir_from_popqa, train_probe, compute_margin,
)
from scripts.patch_L20_localise_full_residual import (           # noqa: E402
    MultiSitePatcher, make_margin_ids, margin_from_logits,
    perm_p_paired, boot_ci,
)

# ── Qwen3-32B architecture constants (64 layers) ─────────────────────────────
EVI_CANDIDATES = [16, 24, 32, 40]          # ~25–63 % depth
ACT_CANDIDATES = [40, 48, 52, 56, 60, 62]  # ~63–97 % depth
SWEEP_LAYERS   = [0, 8, 16, 24, 32, 40, 48, 52, 56, 60, 63]  # 11 sparse layers


# ── Chat template (Qwen3 disable-thinking variant) ───────────────────────────
def apply_qwen3_template(tok, messages, add_generation_prompt=True):
    """Apply chat template with thinking disabled for Qwen3."""
    try:
        return tok.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False)
    except TypeError:
        # Older transformers without enable_thinking; fall back
        return tok.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=add_generation_prompt)


def do_forward(model, tok, prompt, device, tool_ids, fin_ids):
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(ids)
    return margin_from_logits(out.logits[0, -1, :], tool_ids, fin_ids)


# ── L_peak identification ─────────────────────────────────────────────────────
def find_lpeak(model, tok, popqa_path, n_popqa, evi_cands, act_cands):
    """Sweep evidence + action candidate layers; return (L_evi, L_act)."""
    all_cands = sorted(set(evi_cands + act_cands))
    print(f"\n=== L_peak sweep: {len(all_cands)} layers, N={n_popqa} ===")
    t0 = time.time()
    popqa_by_layer = collect_popqa_multilayer(
        model, tok, popqa_path, all_cands, n=n_popqa)
    print(f"  popqa sweep done in {time.time()-t0:.1f}s")

    act_scores = {}
    for L in act_cands:
        d, q, _ = extract_action_dir_from_popqa(popqa_by_layer[L])
        act_scores[L] = float(q)
        print(f"  ACT L{L:>3d}  Spearman q={q:.4f}")

    L_act = max(act_scores, key=act_scores.get)
    print(f"  => L_peak (action)   = {L_act}  (q={act_scores[L_act]:.4f})")
    return all_cands, popqa_by_layer, act_scores, L_act


# ── Evidence direction ────────────────────────────────────────────────────────
def find_evi_layer(model, tok, labels_path, baseline_path, evi_cands, popqa_by_layer):
    """Train probe at each evidence candidate; return (L_evi, evidence_dir, cv)."""
    print(f"\n=== Evidence probe sweep: {evi_cands} ===")
    step1_data = collect_step1_states(model, tok, labels_path, baseline_path, evi_cands)
    print(f"  step1 N={len(step1_data)}")

    best_auroc, best_L, best_dir, best_cv = -1, evi_cands[0], None, None
    for L in evi_cands:
        X = np.array([d["hidden"][L] for d in step1_data], dtype=np.float32)
        y = np.array([d["label"] for d in step1_data], dtype=np.int32)
        direction, cv = train_probe(X, y, return_cv=True)
        print(f"  EVI L{L:>3d}  AUROC={cv['auroc_mean']:.3f}±{cv['auroc_std']:.3f}")
        if cv["auroc_mean"] > best_auroc:
            best_auroc, best_L, best_dir, best_cv = cv["auroc_mean"], L, direction, cv

    print(f"  => L_peak (evidence) = {best_L}  (AUROC={best_auroc:.3f})")
    return best_L, best_dir, best_cv, step1_data


# ── Exp 2: Same-layer steering ────────────────────────────────────────────────
def run_steering(model, tok, device, pairs_path, out_dir, args,
                 action_dir, evidence_dir, L_act, L_evi,
                 evi_auroc, act_quality, cos_ae, D):
    rhos = [float(x) for x in args.rhos.split(",")]
    rng  = np.random.default_rng(args.seed)
    rand_dir = rng.standard_normal(D).astype(np.float32)
    rand_dir /= np.linalg.norm(rand_dir)

    DIR_MAP = {"action":      (action_dir,  L_act),
               "evidence":    (evidence_dir, L_evi),
               "random_act":  (rand_dir,     L_act),
               "random_evi":  (rand_dir,     L_evi)}

    records_all = [json.loads(l) for l in open(pairs_path)]
    records = [r for r in records_all if r.get("condition") == args.steering_cond][:args.limit]
    print(f"\n=== Exp 2 Steering: cond={args.steering_cond} N={len(records)} ===")
    builder  = PromptBuilder()
    tool_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]

    layers_for_rms = list({L_act, L_evi})
    model_layers   = get_model_layers(model)

    def get_rms(prompt, layer_idx):
        cap = {}
        def h(m, i, o):
            x = o[0] if isinstance(o, tuple) else o
            cap["v"] = x[0, -1, :].detach().float().cpu().numpy()
        hdl = model_layers[layer_idx].register_forward_hook(h)
        ids = tok.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad(): model(ids)
        hdl.remove()
        return compute_rms(cap["v"])

    def steered_margin(prompt, vec, alpha, layer):
        if vec is None:
            return do_forward(model, tok, prompt, device, tool_ids, fin_ids)
        with SteeringHook(model, vec, alpha, layer=layer,
                          position=-1, mode="addition", max_interventions=1):
            return do_forward(model, tok, prompt, device, tool_ids, fin_ids)

    rows_path = out_dir / "exp2_steering_results.jsonl"
    n_written = 0; t0 = time.time()
    with open(rows_path, "w") as f:
        for i, rec in enumerate(records):
            steps   = [{"action": "search",
                        "action_input": f"about: {rec['question'][:80]}",
                        "observation": rec["obs"]}]
            msgs    = builder.build_full_prompt(rec["question"], steps)
            prompt  = apply_qwen3_template(tok, msgs, add_generation_prompt=True)
            rms_act = get_rms(prompt, L_act)
            rms_evi = get_rms(prompt, L_evi)
            m_base  = steered_margin(prompt, None, 0.0, L_act)
            f.write(json.dumps({
                "sample_id": rec["sample_id"], "cond": "baseline",
                "direction": None, "layer": None, "rho": 0.0,
                "margin_baseline": m_base, "margin_steered": m_base, "delta_margin": 0.0,
            }) + "\n"); n_written += 1
            for dname, (vec, layer) in DIR_MAP.items():
                rms   = rms_act if layer == L_act else rms_evi
                d_rms = compute_rms(vec)
                for rho in rhos:
                    for sign in [+1.0, -1.0]:
                        alpha = sign * rho * (rms / d_rms)
                        m_st  = steered_margin(prompt, vec, alpha, layer)
                        f.write(json.dumps({
                            "sample_id": rec["sample_id"],
                            "cond": f"{dname}_L{layer}",
                            "direction": dname, "layer": layer,
                            "rho": sign * rho, "alpha": alpha,
                            "margin_baseline": m_base, "margin_steered": m_st,
                            "delta_margin": m_st - m_base,
                        }) + "\n"); n_written += 1
            f.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(records):
                print(f"  [{i+1}/{len(records)}] {time.time()-t0:.1f}s rows={n_written}")

    summary = summarize_steering(rows_path, L_act, L_evi, evi_auroc, act_quality, cos_ae, args)
    json.dump(summary, open(out_dir / "exp2_summary.json", "w"), indent=2)
    print(f"[wrote] {out_dir/'exp2_summary.json'}")
    return summary


def summarize_steering(rows_path, L_act, L_evi, evi_auroc, act_quality, cos_ae, args):
    rows = [json.loads(l) for l in open(rows_path)]
    baselines = [r["margin_baseline"] for r in rows if r["cond"] == "baseline"]
    by_cond = {}
    for r in rows:
        if r["cond"] == "baseline":
            continue
        key = (r["direction"], r["layer"], round(r["rho"], 3))
        by_cond.setdefault(key, []).append(r)
    sm = {
        "model": args.model_path, "n_samples": len(baselines),
        "action_layer": L_act, "evidence_layer": L_evi,
        "evidence_auroc": evi_auroc, "action_quality": act_quality,
        "cos_action_evidence": cos_ae,
        "baseline_margin_mean": float(np.mean(baselines)),
        "baseline_margin_std":  float(np.std(baselines)),
        "conditions": {},
    }
    for (dname, layer, rho), rs in by_cond.items():
        d = np.array([r["delta_margin"] for r in rs])
        sm["conditions"][f"{dname}_L{layer}_rho{rho:+.2f}"] = {
            "direction": dname, "layer": layer, "rho": rho, "n": len(rs),
            "delta_margin_mean":   float(d.mean()),
            "delta_margin_median": float(np.median(d)),
            "delta_margin_std":    float(d.std()),
            "abs_delta_mean":      float(np.mean(np.abs(d))),
            "flip_search_to_stop": sum(1 for r in rs
                                       if r["margin_baseline"] > 0 and r["margin_steered"] < 0),
            "flip_stop_to_search": sum(1 for r in rs
                                       if r["margin_baseline"] < 0 and r["margin_steered"] > 0),
        }
    print(f"\nbaseline margin = {sm['baseline_margin_mean']:+.3f} ± {sm['baseline_margin_std']:.3f}")
    print(f"cos(action,evidence) = {cos_ae:+.4f}")
    print(f"{'condition':32s} {'n':>3s} {'Δm_mean':>8s} {'|Δm|':>7s}")
    for k in sorted(sm["conditions"]):
        c = sm["conditions"][k]
        print(f"{k:32s} {c['n']:>3d} {c['delta_margin_mean']:+8.3f} {c['abs_delta_mean']:7.3f}")
    return sm


# ── Exp 1: Sparse residual sweep ─────────────────────────────────────────────
def run_residual_sweep(model, tok, device, pairs_path, out_dir, args,
                       sweep_layers, n_layers):
    model_layers = get_model_layers(model)
    bad = [L for L in sweep_layers if L < 0 or L >= n_layers]
    if bad:
        raise SystemExit(f"Layers out of range [0,{n_layers}): {bad}")

    records = [json.loads(l) for l in open(pairs_path)]
    sids_all = sorted(set(r["sample_id"] for r in records))
    sids = sids_all[:args.sweep_limit] if args.sweep_limit else sids_all
    need = {("sf", "task_missingness"), ("distractor", "task_missingness")}
    by_sid = {s: {} for s in sids}
    for r in records:
        if r["sample_id"] in by_sid and (r["target"], r["cue"]) in need:
            by_sid[r["sample_id"]][(r["target"], r["cue"])] = r
    sids = [s for s in sids if len(by_sid[s]) == 2]
    print(f"\n=== Exp 1 Residual Sweep: N={len(sids)} samples, layers={sweep_layers} ===")

    SITES = {f"L{L}": model_layers[L] for L in sweep_layers}
    tool_ids, fin_ids = make_margin_ids(tok)
    builder = PromptBuilder()
    patcher = MultiSitePatcher(SITES)

    def build_prompt(rec):
        steps = [{"action": "search",
                  "action_input": f"about: {rec['question'][:80]}",
                  "observation": rec["obs"]}]
        msgs = builder.build_full_prompt(rec["question"], steps)
        return apply_qwen3_template(tok, msgs, add_generation_prompt=True)

    # Stage 1: natural captures
    natural = {}; t0 = time.time()
    for i, s in enumerate(sids):
        natural[s] = {}
        for cell in [("sf", "task_missingness"), ("distractor", "task_missingness")]:
            prompt = build_prompt(by_sid[s][cell])
            patcher.reset_run()
            with patcher:
                m = do_forward(model, tok, prompt, device, tool_ids, fin_ids)
            natural[s][cell] = {"margin": m, "prompt": prompt,
                                 "activations": dict(patcher.captured)}
        if (i + 1) % 10 == 0 or i + 1 == len(sids):
            print(f"  [stage1 {i+1}/{len(sids)}] {time.time()-t0:.1f}s")

    # Mismatched permutation (circular shift of sids)
    donor_maps = {"matched": {s: s for s in sids},
                  "mismatched": {sids[i]: sids[(i + 1) % len(sids)] for i in range(len(sids))}}

    rows_path = out_dir / "exp1_patch_results.jsonl"
    n_written = 0; t0 = time.time()
    with open(rows_path, "w") as f:
        for mode, dmap in donor_maps.items():
            for li, L in enumerate(sweep_layers):
                site = f"L{L}"
                for s in sids:
                    donor = dmap[s]
                    src = natural[donor][("sf", "task_missingness")]
                    tgt = natural[s][("distractor", "task_missingness")]
                    patcher.reset_run()
                    patcher.patch_vecs[site] = src["activations"][site]
                    with patcher:
                        m_p = do_forward(model, tok, tgt["prompt"], device, tool_ids, fin_ids)
                    f.write(json.dumps({
                        "sample_id": s, "donor_sid": donor, "mode": mode,
                        "layer": L, "site": site,
                        "margin_source_sf_tm":   src["margin"],
                        "margin_target_dist_tm": tgt["margin"],
                        "margin_patched": m_p,
                        "delta_margin":   m_p - tgt["margin"],
                        "locality_gap":   src["margin"] - tgt["margin"],
                        "action_target_natural": "search" if tgt["margin"] > 0 else "stop",
                        "action_patched":        "search" if m_p > 0 else "stop",
                    }) + "\n"); f.flush(); n_written += 1
                print(f"  [stage2 {mode} L{L} ({li+1}/{len(sweep_layers)})] "
                      f"{time.time()-t0:.1f}s rows={n_written}")

    summary = summarize_sweep(rows_path, sweep_layers, args)
    json.dump(summary, open(out_dir / "exp1_summary.json", "w"), indent=2)
    print(f"[wrote] {out_dir/'exp1_summary.json'}")
    return summary


def summarize_sweep(rows_path, sweep_layers, args):
    rows = [json.loads(l) for l in open(rows_path)]
    summary = {"model": args.model_path, "sweep_layers": sweep_layers, "modes": {}}
    for mode in ["matched", "mismatched"]:
        by_L = {L: [] for L in sweep_layers}
        for r in rows:
            if r["mode"] == mode:
                by_L[r["layer"]].append(r)
        layer_stats = {}
        for L, rs in by_L.items():
            if not rs:
                continue
            d = np.array([r["delta_margin"] for r in rs])
            gaps = np.array([r["locality_gap"] for r in rs])
            mask = gaps > 0.5
            rec = np.array([r["delta_margin"] / r["locality_gap"]
                            if abs(r["locality_gap"]) > 0.01 else np.nan for r in rs])
            rec_pos = rec[mask & np.isfinite(rec)]
            lo_d, hi_d = boot_ci(d)
            layer_stats[L] = {
                "n": len(rs),
                "delta_margin_mean":   float(d.mean()),
                "delta_margin_ci95":   [lo_d, hi_d],
                "perm_p":              perm_p_paired(d),
                "recovery_mean":       float(rec_pos.mean()) if len(rec_pos) else float("nan"),
                "flip_stop_to_search": sum(1 for r in rs if r["action_target_natural"] == "stop"
                                           and r["action_patched"] == "search"),
                "flip_search_to_stop": sum(1 for r in rs if r["action_target_natural"] == "search"
                                           and r["action_patched"] == "stop"),
            }
        summary["modes"][mode] = layer_stats
        print(f"\n{mode}:  {'L':>3s} {'Δmarg':>8s} {'CI95':>22s} {'perm_p':>7s} {'rec':>8s}")
        for L in sweep_layers:
            s = layer_stats.get(L)
            if not s:
                continue
            ci = s["delta_margin_ci95"]
            print(f"  {L:>3d} {s['delta_margin_mean']:+8.3f} "
                  f"[{ci[0]:+7.3f},{ci[1]:+7.3f}] {s['perm_p']:.4f} "
                  f"{s['recovery_mean']:+8.3f}")
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/home/featurize/work/models/Qwen3-32B")
    ap.add_argument("--dtype",  default="bfloat16")
    ap.add_argument("--labels-path",   default="results/phase1_probe/labels.jsonl")
    ap.add_argument("--baseline-trace",
                    default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--popqa-path",    default="data/popqa/popqa_test.jsonl")
    ap.add_argument("--n-popqa",       type=int, default=300)
    ap.add_argument("--steering-pairs",
                    default="results/extractability_support_toggle/pairs.jsonl")
    ap.add_argument("--sweep-pairs",   default="results/anti_cue_tm_n100/pairs.jsonl")
    ap.add_argument("--steering-cond", default="N0")
    ap.add_argument("--limit",         type=int, default=50,
                    help="max steering samples (Exp 2)")
    ap.add_argument("--sweep-limit",   type=int, default=None,
                    help="max residual sweep samples (Exp 1)")
    ap.add_argument("--rhos",          default="0.10,0.20")
    ap.add_argument("--out-dir",
                    default="results/cross_model_qwen3_32b/circuit_sanity")
    ap.add_argument("--seed",          type=int, default=20260426)
    ap.add_argument("--skip-exp1",     action="store_true")
    ap.add_argument("--skip-exp2",     action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"[info] Loading {args.model_path}  dtype={args.dtype}")
    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype,
        device_map="auto", trust_remote_code=True)
    model.eval()
    device = next(model.parameters()).device
    n_layers = len(get_model_layers(model))
    D = model.config.hidden_size
    print(f"[info] loaded in {time.time()-t_load:.1f}s  n_layers={n_layers}  D={D}")

    # Validate sweep layers against actual n_layers
    sweep_layers = [L for L in SWEEP_LAYERS if L < n_layers]
    evi_cands    = [L for L in EVI_CANDIDATES if L < n_layers]
    act_cands    = [L for L in ACT_CANDIDATES if L < n_layers]

    # ── L_peak identification ────────────────────────────────────────────────
    all_cands, popqa_by_layer, act_scores, L_act = find_lpeak(
        model, tok, args.popqa_path, args.n_popqa, evi_cands, act_cands)

    L_evi, evidence_dir, evi_cv, step1_data = find_evi_layer(
        model, tok, args.labels_path, args.baseline_trace, evi_cands, popqa_by_layer)

    action_dir, act_quality, _ = extract_action_dir_from_popqa(popqa_by_layer[L_act])
    cos_ae = float(np.dot(action_dir, evidence_dir))
    evi_auroc = evi_cv["auroc_mean"]

    np.savez(out_dir / "directions.npz",
             evidence_dir=evidence_dir, action_dir=action_dir,
             L_evi=L_evi, L_act=L_act,
             cos_action_evidence=cos_ae,
             evidence_auroc=evi_auroc, action_quality=act_quality)

    lpeak_summary = {
        "model": args.model_path, "n_layers": n_layers, "hidden_size": D,
        "L_act": L_act, "L_evi": L_evi,
        "act_scores": {str(L): v for L, v in act_scores.items()},
        "evidence_auroc": evi_auroc, "cos_action_evidence": cos_ae,
    }
    json.dump(lpeak_summary, open(out_dir / "lpeak_summary.json", "w"), indent=2)
    print(f"\n[L_peak] L_act={L_act}  L_evi={L_evi}  "
          f"evi_AUROC={evi_auroc:.3f}  act_q={act_quality:.3f}  cos={cos_ae:+.4f}")

    # ── Exp 2: Same-layer steering ───────────────────────────────────────────
    if not args.skip_exp2:
        run_steering(model, tok, device,
                     args.steering_pairs, out_dir, args,
                     action_dir, evidence_dir, L_act, L_evi,
                     evi_auroc, act_quality, cos_ae, D)

    # ── Exp 1: Residual formation sweep ─────────────────────────────────────
    if not args.skip_exp1:
        run_residual_sweep(model, tok, device,
                           args.sweep_pairs, out_dir, args,
                           sweep_layers, n_layers)

    print("\n=== qwen3_circuit_sanity.py DONE ===")


if __name__ == "__main__":
    main()

