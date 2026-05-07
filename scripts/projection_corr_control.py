#!/usr/bin/env python3
"""
Random direction null for projection correlations.
Tests: is corr(h·evi, h·act) significantly different from corr(h·rand, h·act)?

This controls for the possibility that data covariance structure creates
spurious correlations between any two projections.
"""

import os, sys, json, argparse, random
import numpy as np
from pathlib import Path
import torch
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import get_model_layers

LAYER = 20


def extract_hidden_states(model, tokenizer, label_data, bl_map, layer,
                          do_step0=False):
    """Extract L20 hidden states for all samples at step 1 (and optionally step 0)."""
    layers = get_model_layers(model)
    device = next(model.parameters()).device
    pb = PromptBuilder(tools=["search", "calculator"])
    records = []

    for i, ld in enumerate(label_data):
        sid = ld["sample_id"]
        ep = bl_map.get(sid)
        if not ep or not ep.get("steps") or len(ep["steps"]) < 1:
            continue
        s0 = ep["steps"][0]
        if s0.get("action") != "search" or not s0.get("observation"):
            continue

        # Step 1
        steps = [{"action": "search", "action_input": s0["action_input"],
                  "observation": s0["observation"]}]
        msgs = pb.build_full_prompt(ld["question"], steps)
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        captured = {}
        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured["h"] = h[0, -1, :].detach().float().cpu().numpy()
        handle = layers[layer].register_forward_hook(hook_fn)
        with torch.no_grad():
            model(input_ids)
        handle.remove()

        rec = {"label": ld["label"], "h1": captured["h"]}

        if do_step0:
            msgs0 = pb.build_full_prompt(ld["question"], [])
            p0 = tokenizer.apply_chat_template(
                msgs0, tokenize=False, add_generation_prompt=True)
            ids0 = tokenizer.encode(p0, return_tensors="pt").to(device)
            cap0 = {}
            def hook0(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                cap0["h"] = h[0, -1, :].detach().float().cpu().numpy()
            handle0 = layers[layer].register_forward_hook(hook0)
            with torch.no_grad():
                model(ids0)
            handle0.remove()
            rec["h0"] = cap0["h"]

        records.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(label_data)}]")

    return records


def projection_corr_with_null(X, dir_a, dir_b, n_null=1000, seed=42):
    """Compute corr(X·dir_a, X·dir_b) and compare with random null."""
    rng = np.random.RandomState(seed)
    proj_a = X @ dir_a
    proj_b = X @ dir_b
    obs_r, obs_p = pearsonr(proj_a, proj_b)

    # Null: replace dir_a with random directions
    null_rs = []
    d = X.shape[1]
    for _ in range(n_null):
        rand_dir = rng.randn(d).astype(np.float32)
        rand_dir /= np.linalg.norm(rand_dir)
        proj_rand = X @ rand_dir
        r, _ = pearsonr(proj_rand, proj_b)
        null_rs.append(r)
    null_rs = np.array(null_rs)

    p_val = np.mean(np.abs(null_rs) >= abs(obs_r))
    return obs_r, obs_p, null_rs, p_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--labels-path", default="results/phase1_probe/labels.jsonl")
    ap.add_argument("--baseline-trace",
                    default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--instruct-action-dir",
                    default="steering/directions/direction_search_v3_layer20.npz")
    ap.add_argument("--instruct-evidence-dir",
                    default="results/phase1_probe/probe_direction_l20.npz")
    ap.add_argument("--output-dir", default="results/orthogonality_fixed")
    ap.add_argument("--do-step0", action="store_true")
    ap.add_argument("--n-null", type=int, default=1000)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load directions
    v3 = np.load(args.instruct_action_dir)
    act_dir = v3['decision_direction_normalized']
    probe_f = np.load(args.instruct_evidence_dir)
    evi_dir = probe_f['decision_direction']
    evi_dir = (evi_dir / np.linalg.norm(evi_dir)).astype(np.float32)

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    # Load data
    label_data = []
    with open(args.labels_path) as f:
        for line in f:
            label_data.append(json.loads(line))
    bl_map = {}
    with open(args.baseline_trace) as f:
        for line in f:
            ep = json.loads(line)
            bl_map[ep["sample_id"]] = ep

    print("Extracting hidden states...")
    records = extract_hidden_states(
        model, tokenizer, label_data, bl_map, LAYER, do_step0=args.do_step0)
    print(f"  {len(records)} samples")

    X1 = np.array([r["h1"] for r in records], dtype=np.float32)

    # Step 1 control
    print(f"\n=== Step 1: corr(h·evi, h·act) with random null (n={args.n_null}) ===")
    obs_r, obs_p, null_rs, p_null = projection_corr_with_null(
        X1, evi_dir, act_dir, n_null=args.n_null)
    print(f"  Observed: r={obs_r:.4f} (p={obs_p:.4f})")
    print(f"  Null |r|: mean={np.mean(np.abs(null_rs)):.4f}, "
          f"std={np.std(null_rs):.4f}, max={np.max(np.abs(null_rs)):.4f}")
    print(f"  p(|null| >= |obs|) = {p_null:.4f}")

    results = {"model": args.model, "n_samples": len(records),
               "step1": {"obs_r": float(obs_r), "obs_p": float(obs_p),
                         "null_mean_abs": float(np.mean(np.abs(null_rs))),
                         "null_std": float(np.std(null_rs)),
                         "null_max_abs": float(np.max(np.abs(null_rs))),
                         "p_vs_null": float(p_null)}}

    if args.do_step0 and "h0" in records[0]:
        X0 = np.array([r["h0"] for r in records], dtype=np.float32)
        print(f"\n=== Step 0: corr(h·evi, h·act) with random null ===")
        obs_r0, obs_p0, null_rs0, p_null0 = projection_corr_with_null(
            X0, evi_dir, act_dir, n_null=args.n_null)
        print(f"  Observed: r={obs_r0:.4f} (p={obs_p0:.4f})")
        print(f"  Null |r|: mean={np.mean(np.abs(null_rs0)):.4f}, "
              f"std={np.std(null_rs0):.4f}")
        print(f"  p(|null| >= |obs|) = {p_null0:.4f}")
        results["step0"] = {"obs_r": float(obs_r0), "obs_p": float(obs_p0),
                            "null_mean_abs": float(np.mean(np.abs(null_rs0))),
                            "null_std": float(np.std(null_rs0)),
                            "p_vs_null": float(p_null0)}

    tag = args.model.split("/")[-1].lower().replace("-", "_")
    out_path = os.path.join(args.output_dir, f"{tag}_projection_control.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
