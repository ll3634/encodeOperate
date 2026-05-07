#!/usr/bin/env python3
"""Same-layer control for Gemma-2-9B-it Exp 2.

Original Exp 2 compared action_dir@L37 vs evidence_dir@L23 — a confound
because L23 is 14 layers further from the readout. This script re-extracts
evidence_dir at L37 (same-layer probe), then runs the gain comparison
at a single layer (L37) so action vs evidence is fair.

Reuses action_dir / random_dir / N0 prompts from the prior run.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS  # noqa: E402
from steering.hook_utils import get_model_layers, SteeringHook, compute_rms  # noqa: E402
from scripts.cross_model_full import (  # noqa: E402
    apply_chat_template_safe, collect_step1_states, train_probe,
)
from scripts.gemma_steering_sanity import (  # noqa: E402
    margin_first_token, get_hidden_rms_at_layer, steered_margin,
)

ACT_LAYER = 37  # default: peak action layer (overridden by --act-layer)


def main():
    global ACT_LAYER
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="unsloth/gemma-2-9b-it")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--labels-path", default="results/phase1_probe/labels.jsonl")
    ap.add_argument("--baseline-trace", default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--steering-pairs", default="results/extractability_support_toggle/pairs.jsonl")
    ap.add_argument("--steering-cond", default="N0")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--rhos", default="0.10,0.20")
    ap.add_argument("--prior-dirs", default="results/gemma_circuit_sanity/exp2_steering/directions.npz")
    ap.add_argument("--out-dir", default="results/gemma_circuit_sanity/exp2_samelayer")
    ap.add_argument("--act-layer", type=int, default=ACT_LAYER)
    args = ap.parse_args()
    ACT_LAYER = args.act_layer

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rhos = [float(x) for x in args.rhos.split(",")]

    prior = np.load(args.prior_dirs)
    action_dir = prior["action_dir"].astype(np.float32)
    random_dir = prior["random_dir"].astype(np.float32)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"[info] loading {args.model_path} dtype={args.dtype}")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True)
    model.eval(); device = next(model.parameters()).device
    n_layers = len(get_model_layers(model)); D = model.config.hidden_size
    print(f"[info] n_layers={n_layers} D={D}")

    # === Same-layer evidence_dir @ L37 ===
    print(f"\n=== Collecting step1 hidden states at L{ACT_LAYER} ===")
    t0 = time.time()
    step1_data = collect_step1_states(model, tok, args.labels_path,
                                      args.baseline_trace, [ACT_LAYER])
    print(f"  step1 N={len(step1_data)} in {time.time()-t0:.1f}s")
    X = np.array([d["hidden"][ACT_LAYER] for d in step1_data], dtype=np.float32)
    y = np.array([d["label"] for d in step1_data], dtype=np.int32)
    evidence_dir_L37, cv = train_probe(X, y, return_cv=True)
    print(f"  evidence_dir@L{ACT_LAYER} CV-AUROC={cv['auroc_mean']:.3f}±{cv['auroc_std']:.3f}")

    cos_ae_samelayer = float(np.dot(action_dir, evidence_dir_L37))
    cos_ar_samelayer = float(np.dot(action_dir, random_dir))
    cos_er_samelayer = float(np.dot(evidence_dir_L37, random_dir))
    print(f"  cos(action,  evidence_L37) = {cos_ae_samelayer:+.4f}")
    print(f"  cos(action,  random)        = {cos_ar_samelayer:+.4f}")
    print(f"  cos(evi_L37, random)        = {cos_er_samelayer:+.4f}")

    np.savez(out_dir / "directions.npz",
             evidence_dir_L37=evidence_dir_L37,
             action_dir=action_dir, random_dir=random_dir,
             evi_layer=ACT_LAYER, act_layer=ACT_LAYER,
             cos_action_evidence_L37=cos_ae_samelayer,
             evidence_L37_auroc=cv["auroc_mean"])

    # === Steering ===
    records_all = [json.loads(l) for l in open(args.steering_pairs)]
    records = [r for r in records_all if r.get("condition") == args.steering_cond][:args.limit]
    print(f"\n=== Same-layer steering at L{ACT_LAYER}: cond={args.steering_cond} N={len(records)} ===")
    builder = PromptBuilder()
    tool_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]

    DIR_MAP = {"action": action_dir,
               "evidence_L37": evidence_dir_L37,
               "random": random_dir}
    rows_path = out_dir / "steering_results.jsonl"
    n_written = 0; t0 = time.time()
    with open(rows_path, "w") as f:
        for i, rec in enumerate(records):
            steps = [{"action": "search",
                      "action_input": f"about: {rec['question'][:80]}",
                      "observation": rec["obs"]}]
            msgs = builder.build_full_prompt(rec["question"], steps)
            prompt = apply_chat_template_safe(tok, msgs, add_generation_prompt=True)
            rms_act, _ = get_hidden_rms_at_layer(model, tok, prompt, ACT_LAYER, device)
            m_base = steered_margin(model, tok, prompt, device, tool_ids, fin_ids,
                                    None, 0.0, ACT_LAYER)
            f.write(json.dumps({
                "sample_id": rec["sample_id"], "cond": "baseline",
                "direction": None, "layer": None, "rho": 0.0, "alpha": 0.0,
                "margin_baseline": m_base, "margin_steered": m_base,
                "delta_margin": 0.0, "hidden_rms": rms_act,
            }) + "\n"); n_written += 1
            for dname, vec in DIR_MAP.items():
                d_rms = compute_rms(vec)
                for rho in rhos:
                    for sign in [+1.0, -1.0]:
                        alpha = sign * rho * (rms_act / d_rms)
                        m_st = steered_margin(model, tok, prompt, device,
                                              tool_ids, fin_ids, vec, alpha, ACT_LAYER)
                        f.write(json.dumps({
                            "sample_id": rec["sample_id"],
                            "cond": f"{dname}_L{ACT_LAYER}",
                            "direction": dname, "layer": ACT_LAYER,
                            "rho": sign * rho, "alpha": alpha,
                            "margin_baseline": m_base, "margin_steered": m_st,
                            "delta_margin": m_st - m_base,
                            "hidden_rms": rms_act,
                        }) + "\n"); n_written += 1
            f.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(records):
                print(f"  [{i+1}/{len(records)}] {time.time()-t0:.1f}s rows={n_written}")
    print(f"[wrote] {rows_path}  ({n_written} rows)")
    summarize(rows_path, out_dir / "summary.json", args,
              cos_ae_samelayer, cv["auroc_mean"])


def summarize(in_path, out_json, args, cos_ae, evi_auroc):
    rows = [json.loads(l) for l in open(in_path)]
    by_key = {}; baselines = []
    for r in rows:
        if r["cond"] == "baseline":
            baselines.append(r["margin_baseline"]); continue
        k = (r["direction"], r["layer"], round(r["rho"], 3))
        by_key.setdefault(k, []).append(r)
    sm = {"model": args.model_path, "n_samples": len(baselines),
          "steering_cond": args.steering_cond, "layer": ACT_LAYER,
          "cos_action_evidence_L37": cos_ae,
          "evidence_L37_auroc": evi_auroc,
          "baseline_margin_mean": float(np.mean(baselines)),
          "baseline_margin_std": float(np.std(baselines)),
          "conditions": {}}
    for (dname, layer, rho), rs in by_key.items():
        d = np.array([r["delta_margin"] for r in rs])
        m = np.array([r["margin_steered"] for r in rs])
        f1 = sum(1 for r in rs if r["margin_baseline"] > 0 and r["margin_steered"] < 0)
        f2 = sum(1 for r in rs if r["margin_baseline"] < 0 and r["margin_steered"] > 0)
        sm["conditions"][f"{dname}_L{layer}_rho{rho:+.2f}"] = {
            "direction": dname, "layer": layer, "rho": rho, "n": len(rs),
            "delta_margin_mean": float(d.mean()),
            "delta_margin_median": float(np.median(d)),
            "abs_delta_mean": float(np.mean(np.abs(d))),
            "margin_steered_mean": float(m.mean()),
            "flip_search_to_stop": f1, "flip_stop_to_search": f2,
        }
    json.dump(sm, open(out_json, "w"), indent=2)
    print(f"\nbaseline margin = {sm['baseline_margin_mean']:+.3f}±{sm['baseline_margin_std']:.3f}")
    print(f"cos(action, evidence_L37) = {cos_ae:+.4f}  evi_L37_AUROC = {evi_auroc:.3f}\n")
    print(f"{'condition':28s} {'n':>3s} {'Δm_mean':>8s} {'Δm_med':>8s} {'|Δm|':>7s} {'flips(s→S/S→s)':>16s}")
    for k in sorted(sm["conditions"].keys()):
        c = sm["conditions"][k]
        print(f"{k:28s} {c['n']:>3d} {c['delta_margin_mean']:+8.3f} {c['delta_margin_median']:+8.3f} "
              f"{c['abs_delta_mean']:7.3f} {c['flip_search_to_stop']:>5d}/{c['flip_stop_to_search']:<5d}")


if __name__ == "__main__":
    main()
