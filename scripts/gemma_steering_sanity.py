#!/usr/bin/env python3
"""Gemma-2-9B-it action-vs-evidence gain sanity check.

Single-load script:
  1. Extract evidence_dir at L23 (probe on labels.jsonl, label = evidence-sufficient)
  2. Extract action_dir at L37 (PopQA p10/p90 contrast on margin)
  3. Save directions as npz (reproducibility)
  4. Steering at decision token on N=50 N0 samples (extractability_support_toggle)
     across {action_dir, evidence_dir, random_dir} × rho ∈ {0.10, 0.20} × signs
     applied at the layer where the direction was extracted.
  5. Measure first-token margin shift; aggregate; save JSONL + summary.

Direction sign convention (matches cross_model_full.py):
  evidence_dir = probe coef toward label=1 (sufficient)  → adding it ⇒ margin DOWN (more "stop")
  action_dir   = h_low_margin − h_high_margin            → adding it ⇒ margin DOWN (more "stop")
  random_dir   = unit gaussian (matched alpha via hidden_rms)
"""
import argparse, json, sys, time, random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS  # noqa: E402
from steering.hook_utils import get_model_layers, SteeringHook, compute_rms  # noqa: E402
from scripts.cross_model_full import (  # noqa: E402
    apply_chat_template_safe, collect_step1_states,
    collect_popqa_multilayer, extract_action_dir_from_popqa,
    train_probe, compute_margin,
)

EVI_LAYER = 23  # default: peak from cross_model_gemma2_v2 (overridden by --evi-layer)
ACT_LAYER = 37  # default: peak from cross_model_gemma2_v2 (overridden by --act-layer)


def margin_first_token(logits_last, tool_ids, fin_ids):
    lp = torch.log_softmax(logits_last.float(), dim=-1)
    tl = torch.logsumexp(lp[tool_ids], 0).item()
    fl = torch.logsumexp(lp[fin_ids],  0).item()
    return float(tl - fl)


def get_hidden_rms_at_layer(model, tok, prompt, layer_idx, device):
    layers = get_model_layers(model)
    cap = {}
    def h(m, i, o):
        x = o[0] if isinstance(o, tuple) else o
        cap["v"] = x[0, -1, :].detach().float().cpu().numpy()
    handle = layers[layer_idx].register_forward_hook(h)
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model(ids)
    handle.remove()
    return compute_rms(cap["v"]), cap["v"]


def steered_margin(model, tok, prompt, device, tool_ids, fin_ids,
                   direction, alpha, layer):
    if direction is None:
        ids = tok.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(ids)
        return margin_first_token(out.logits[0, -1, :], tool_ids, fin_ids)
    with SteeringHook(model, direction, alpha, layer=layer,
                      position=-1, mode="addition", max_interventions=1):
        ids = tok.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(ids)
    return margin_first_token(out.logits[0, -1, :], tool_ids, fin_ids)


def main():
    global EVI_LAYER, ACT_LAYER
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="unsloth/gemma-2-9b-it")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--labels-path", default="results/phase1_probe/labels.jsonl")
    ap.add_argument("--baseline-trace", default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--popqa-path", default="data/popqa/popqa_test.jsonl")
    ap.add_argument("--n-popqa", type=int, default=200)
    ap.add_argument("--steering-pairs", default="results/extractability_support_toggle/pairs.jsonl")
    ap.add_argument("--steering-cond", default="N0",
                    help="condition cell to steer on (default N0 = clean search-leaning baseline)")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--rhos", default="0.10,0.20")
    ap.add_argument("--out-dir", default="results/gemma_circuit_sanity/exp2_steering")
    ap.add_argument("--seed", type=int, default=20260426)
    ap.add_argument("--evi-layer", type=int, default=EVI_LAYER)
    ap.add_argument("--act-layer", type=int, default=ACT_LAYER)
    args = ap.parse_args()

    # Shadow module-level defaults so downstream uses arg values
    EVI_LAYER = args.evi_layer
    ACT_LAYER = args.act_layer

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rhos = [float(x) for x in args.rhos.split(",")]

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"[info] loading {args.model_path} dtype={args.dtype}")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True)
    model.eval(); device = next(model.parameters()).device
    n_layers = len(get_model_layers(model))
    D = model.config.hidden_size
    print(f"[info] n_layers={n_layers}  D={D}")

    # ── Direction extraction ─────────────────────────────────────────────
    layers_needed = [EVI_LAYER, ACT_LAYER]

    print(f"\n=== Step 1: Collecting step1 hidden states at L{EVI_LAYER}, L{ACT_LAYER} ===")
    t0 = time.time()
    step1_data = collect_step1_states(model, tok, args.labels_path,
                                      args.baseline_trace, layers_needed)
    print(f"  step1 N={len(step1_data)} in {time.time()-t0:.1f}s")

    print(f"\n=== Step 2: Collecting PopQA hidden states (n={args.n_popqa}) ===")
    t0 = time.time()
    popqa_by_layer = collect_popqa_multilayer(model, tok, args.popqa_path,
                                              layers_needed, n=args.n_popqa)
    print(f"  popqa done in {time.time()-t0:.1f}s")

    # Evidence dir from probe @ L23
    X = np.array([d["hidden"][EVI_LAYER] for d in step1_data], dtype=np.float32)
    y = np.array([d["label"] for d in step1_data], dtype=np.int32)
    evidence_dir, cv = train_probe(X, y, return_cv=True)
    print(f"\n  evidence_dir@L{EVI_LAYER} CV-AUROC={cv['auroc_mean']:.3f}±{cv['auroc_std']:.3f}")

    # Action dir from PopQA p10/p90 @ L37
    action_dir, act_quality, _ = extract_action_dir_from_popqa(popqa_by_layer[ACT_LAYER])
    print(f"  action_dir@L{ACT_LAYER}   Spearman quality={act_quality:.3f}")
    cos_ae = float(np.dot(action_dir, evidence_dir))
    print(f"  cos(action_L{ACT_LAYER}, evidence_L{EVI_LAYER}) = {cos_ae:+.4f}")

    # Random direction (unit-norm)
    rng = np.random.default_rng(args.seed)
    rand_dir = rng.standard_normal(D).astype(np.float32)
    rand_dir = rand_dir / np.linalg.norm(rand_dir)

    np.savez(out_dir / "directions.npz",
             evidence_dir=evidence_dir, action_dir=action_dir,
             random_dir=rand_dir,
             evi_layer=EVI_LAYER, act_layer=ACT_LAYER,
             cos_action_evidence=cos_ae,
             evidence_auroc=cv["auroc_mean"],
             action_quality=act_quality)
    print(f"  [wrote] {out_dir/'directions.npz'}")

    # ── Steering ─────────────────────────────────────────────────────────
    records_all = [json.loads(l) for l in open(args.steering_pairs)]
    records = [r for r in records_all if r.get("condition") == args.steering_cond][:args.limit]
    print(f"\n=== Steering: cond={args.steering_cond} N={len(records)} ===")
    builder = PromptBuilder()
    tool_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]

    DIR_MAP = {"action": (action_dir, ACT_LAYER),
               "evidence": (evidence_dir, EVI_LAYER),
               "random_act": (rand_dir, ACT_LAYER),
               "random_evi": (rand_dir, EVI_LAYER)}
    rows_path = out_dir / "steering_results.jsonl"
    n_written = 0; t0 = time.time()
    with open(rows_path, "w") as f:
        for i, rec in enumerate(records):
            steps = [{"action": "search",
                      "action_input": f"about: {rec['question'][:80]}",
                      "observation": rec["obs"]}]
            msgs = builder.build_full_prompt(rec["question"], steps)
            prompt = apply_chat_template_safe(tok, msgs, add_generation_prompt=True)
            # Compute hidden_rms at each target layer (used for rho→alpha scaling)
            rms_evi, _ = get_hidden_rms_at_layer(model, tok, prompt, EVI_LAYER, device)
            rms_act, _ = get_hidden_rms_at_layer(model, tok, prompt, ACT_LAYER, device)
            m_base = steered_margin(model, tok, prompt, device, tool_ids, fin_ids,
                                    None, 0.0, ACT_LAYER)
            f.write(json.dumps({
                "sample_id": rec["sample_id"], "cond": "baseline",
                "direction": None, "layer": None, "rho": 0.0, "alpha": 0.0,
                "margin_baseline": m_base, "margin_steered": m_base,
                "delta_margin": 0.0,
                "hidden_rms_evi": rms_evi, "hidden_rms_act": rms_act,
            }) + "\n")
            n_written += 1
            for dname, (vec, layer) in DIR_MAP.items():
                rms = rms_act if layer == ACT_LAYER else rms_evi
                # alpha = rho * hidden_rms / direction_rms; direction is unit-norm so direction_rms = 1/sqrt(D)
                d_rms = compute_rms(vec)
                for rho in rhos:
                    for sign in [+1.0, -1.0]:
                        alpha = sign * rho * (rms / d_rms)
                        m_st = steered_margin(model, tok, prompt, device, tool_ids, fin_ids,
                                              vec, alpha, layer)
                        f.write(json.dumps({
                            "sample_id": rec["sample_id"], "cond": f"{dname}_L{layer}",
                            "direction": dname, "layer": layer,
                            "rho": sign * rho, "alpha": alpha,
                            "margin_baseline": m_base, "margin_steered": m_st,
                            "delta_margin": m_st - m_base,
                            "hidden_rms": rms,
                        }) + "\n"); n_written += 1
            f.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(records):
                print(f"  [{i+1}/{len(records)}] {time.time()-t0:.1f}s rows={n_written}")
    print(f"[wrote] {rows_path}  ({n_written} rows)")
    summarize(rows_path, out_dir / "summary.json", args, cos_ae,
              cv["auroc_mean"], act_quality)


def summarize(in_path, out_json, args, cos_ae, evi_auroc, act_quality):
    rows = [json.loads(l) for l in open(in_path)]
    by_cond_rho = {}
    baselines = []
    for r in rows:
        if r["cond"] == "baseline":
            baselines.append(r["margin_baseline"])
            continue
        key = (r["direction"], r["layer"], round(r["rho"], 3))
        by_cond_rho.setdefault(key, []).append(r)

    sm = {"model": args.model_path, "n_samples": len(baselines),
          "steering_cond": args.steering_cond,
          "evidence_layer": EVI_LAYER, "action_layer": ACT_LAYER,
          "cos_action_evidence": cos_ae,
          "evidence_auroc": evi_auroc, "action_quality": act_quality,
          "baseline_margin_mean": float(np.mean(baselines)),
          "baseline_margin_std":  float(np.std(baselines)),
          "conditions": {}}
    for (dname, layer, rho), rs in by_cond_rho.items():
        d = np.array([r["delta_margin"] for r in rs])
        m = np.array([r["margin_steered"] for r in rs])
        flips_se2st = sum(1 for r in rs if r["margin_baseline"] > 0 and r["margin_steered"] < 0)
        flips_st2se = sum(1 for r in rs if r["margin_baseline"] < 0 and r["margin_steered"] > 0)
        sm["conditions"][f"{dname}_L{layer}_rho{rho:+.2f}"] = {
            "direction": dname, "layer": layer, "rho": rho, "n": len(rs),
            "delta_margin_mean": float(d.mean()),
            "delta_margin_median": float(np.median(d)),
            "delta_margin_std": float(d.std()),
            "abs_delta_mean": float(np.mean(np.abs(d))),
            "margin_steered_mean": float(m.mean()),
            "flip_search_to_stop": flips_se2st,
            "flip_stop_to_search": flips_st2se,
        }
    json.dump(sm, open(out_json, "w"), indent=2)
    print(f"[wrote] {out_json}\n")
    print(f"baseline margin = {sm['baseline_margin_mean']:+.3f} ± {sm['baseline_margin_std']:.3f}")
    print(f"cos(action,evidence) = {cos_ae:+.4f}  evi_AUROC = {evi_auroc:.3f}  act_q = {act_quality:.3f}\n")
    print(f"{'condition':28s} {'n':>3s} {'Δm_mean':>8s} {'Δm_med':>8s} {'|Δm|':>7s} {'flips(s→S/S→s)':>16s}")
    keys_sorted = sorted(sm["conditions"].keys())
    for k in keys_sorted:
        c = sm["conditions"][k]
        print(f"{k:28s} {c['n']:>3d} {c['delta_margin_mean']:+8.3f} {c['delta_margin_median']:+8.3f} "
              f"{c['abs_delta_mean']:7.3f} {c['flip_search_to_stop']:>5d}/{c['flip_stop_to_search']:<5d}")


if __name__ == "__main__":
    main()
