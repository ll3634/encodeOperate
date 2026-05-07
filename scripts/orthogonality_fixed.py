#!/usr/bin/env python3
"""
Fixed orthogonality analysis: Base vs Instruct + Step 0 vs Step 1
=================================================================
Fixes two methodological issues from the original scripts:

1. Base model action_dir was noise (margin [1.12, 4.0], no search/stop separation).
   FIX: Use instruct action_dir as shared reference coordinate. Validate by checking
   whether instruct directions are meaningful in base model space (projection AUROC).

2. Step 0 evidence probe AUROC=0.471 (random), so cos(noise, action) ≈ 0 is trivial.
   FIX: Drop invalid step-0 cosine. Instead measure projection correlation:
   corr(h·evidence_dir, h·action_dir) across samples at each step.

New approach:
  A. Project base model hidden states onto instruct directions → check AUROC
  B. Train native base evidence_dir → cos(base_evi, instruct_act) + permutation test
  C. Step 0 vs 1: projection-based correlation analysis

Usage:
  cd tmc/scripts/e2e_agent
  python scripts/orthogonality_fixed.py
"""

import os, sys, json, argparse, random
import numpy as np
from pathlib import Path

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr, mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import get_model_layers

LAYER = 20


def get_margin(logits, tokenizer):
    log_probs = torch.log_softmax(logits, dim=-1)
    tool_ids = [tokenizer.encode(t, add_special_tokens=False)[0]
                for t in ACTION_TOKENS["tool_call"]]
    fin_ids = [tokenizer.encode(t, add_special_tokens=False)[0]
               for t in ACTION_TOKENS["finish"]]
    return (torch.logsumexp(log_probs[tool_ids], 0) -
            torch.logsumexp(log_probs[fin_ids], 0)).item()


def extract_hidden_and_margin(model, tokenizer, prompt, layer):
    layers = get_model_layers(model)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    captured = {}

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h[0, -1, :].detach().float().cpu().numpy()
    handle = layers[layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :]
    handle.remove()
    return captured["h"], get_margin(logits, tokenizer)


def train_probe(X, y, tag=""):
    """Train probe, return direction + metrics. Returns None direction if AUROC < 0.6."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                           solver="lbfgs", random_state=42),
        X_s, y, cv=cv, scoring="balanced_accuracy")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(sss.split(X_s, y))
    p = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                           solver="lbfgs", random_state=42)
    p.fit(X_s[tr], y[tr])
    auroc = roc_auc_score(y[te], p.predict_proba(X_s[te])[:, 1])
    # Full probe for direction
    pf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                            solver="lbfgs", random_state=42)
    pf.fit(X_s, y)
    w = pf.coef_[0] / scaler.scale_
    d = (w / np.linalg.norm(w)).astype(np.float32)
    metrics = {"cv_bal_acc": float(cv_scores.mean()), "cv_std": float(cv_scores.std()),
               "auroc": float(auroc), "n": len(y), "n_pos": int(y.sum())}
    print(f"  [{tag}] AUROC={auroc:.3f} CV={cv_scores.mean():.3f}±{cv_scores.std():.3f} "
          f"N={len(y)} ({y.sum()}/{(1-y).sum()})")
    return d, metrics


def _train_probe_silent(X, y):
    """Train probe without printing. Returns (direction, metrics)."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    pf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                            solver="lbfgs", random_state=42)
    pf.fit(X_s, y)
    w = pf.coef_[0] / scaler.scale_
    d = (w / np.linalg.norm(w)).astype(np.float32)
    return d, {}


def permutation_test_cosine(X, y, ref_dir, n_perm=500, seed=42):
    """Permutation test: how often does a shuffled-label probe produce
    |cos(probe_dir, ref_dir)| >= observed?"""
    rng = np.random.RandomState(seed)
    # Observed
    obs_dir, _ = train_probe(X, y, tag="observed")
    obs_cos = abs(float(np.dot(obs_dir, ref_dir)))

    null_cos = []
    for i in range(n_perm):
        y_shuf = rng.permutation(y)
        try:
            d_shuf, _ = _train_probe_silent(X, y_shuf)
            null_cos.append(abs(float(np.dot(d_shuf, ref_dir))))
        except Exception:
            continue
        if (i + 1) % 100 == 0:
            print(f"    perm [{i+1}/{n_perm}] null_mean={np.mean(null_cos):.4f}")
    null_cos = np.array(null_cos)
    p_val = (np.sum(null_cos >= obs_cos) + 1) / (len(null_cos) + 1)
    return obs_cos, null_cos, p_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B",
                    help="Model to analyze (base or instruct)")
    ap.add_argument("--labels-path", default="results/phase1_probe/labels.jsonl")
    ap.add_argument("--baseline-trace",
                    default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--instruct-action-dir",
                    default="steering/directions/direction_search_v3_layer20.npz")
    ap.add_argument("--instruct-evidence-dir",
                    default="results/phase1_probe/probe_direction_l20.npz")
    ap.add_argument("--output-dir", default="results/orthogonality_fixed")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--do-step0", action="store_true",
                    help="Also extract step-0 (question-only) hidden states")
    args = ap.parse_args()

    global LAYER
    LAYER = args.layer
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load instruct reference directions ──
    v3 = np.load(args.instruct_action_dir)
    instruct_act = v3['decision_direction_normalized']
    probe_f = np.load(args.instruct_evidence_dir)
    instruct_evi = probe_f['decision_direction']
    instruct_evi = instruct_evi / np.linalg.norm(instruct_evi)
    print(f"Instruct directions loaded. cos(evi,act)={np.dot(instruct_evi, instruct_act):.4f}")

    # ── Load model ──
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    # ── Load labels + baseline ──
    label_data = []
    with open(args.labels_path) as f:
        for line in f:
            label_data.append(json.loads(line))
    bl_map = {}
    with open(args.baseline_trace) as f:
        for line in f:
            ep = json.loads(line)
            bl_map[ep["sample_id"]] = ep

    # ── Extract hidden states ──
    pb = PromptBuilder(tools=["search", "calculator"])
    records = []
    print(f"\nExtracting L{LAYER} hidden states...")
    for i, ld in enumerate(label_data):
        sid = ld["sample_id"]
        ep = bl_map.get(sid)
        if not ep or not ep.get("steps") or len(ep["steps"]) < 1:
            continue
        s0 = ep["steps"][0]
        if s0.get("action") != "search" or not s0.get("observation"):
            continue

        # Step 1 prompt (question + observation)
        steps = [{"action": "search", "action_input": s0["action_input"],
                  "observation": s0["observation"]}]
        msgs = pb.build_full_prompt(ld["question"], steps)
        prompt_s1 = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        h1, m1 = extract_hidden_and_margin(model, tokenizer, prompt_s1, LAYER)

        rec = {"sid": sid, "label": ld["label"], "h1": h1, "margin1": m1}

        if args.do_step0:
            msgs0 = pb.build_full_prompt(ld["question"], [])
            prompt_s0 = tokenizer.apply_chat_template(
                msgs0, tokenize=False, add_generation_prompt=True)
            h0, m0 = extract_hidden_and_margin(model, tokenizer, prompt_s0, LAYER)
            rec["h0"] = h0
            rec["margin0"] = m0

        records.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(label_data)}]")

    print(f"  Total: {len(records)} samples")

    # ── Analysis ──
    X1 = np.array([r["h1"] for r in records], dtype=np.float32)
    y = np.array([r["label"] for r in records], dtype=np.int32)
    margins1 = np.array([r["margin1"] for r in records])

    # A. Validate instruct directions in this model's space
    print(f"\n=== A. Instruct direction validity in {args.model} space ===")
    proj_evi = X1 @ instruct_evi
    proj_act = X1 @ instruct_act
    auroc_evi_transfer = roc_auc_score(y, proj_evi)
    r_act_margin, p_act = pearsonr(proj_act, margins1)
    print(f"  Instruct evidence_dir → this model: transfer AUROC = {auroc_evi_transfer:.3f}")
    print(f"  Instruct action_dir → this model: corr(proj, margin) = {r_act_margin:.3f} (p={p_act:.4f})")

    # B. Train native evidence probe + cosine with instruct_act
    print(f"\n=== B. Native evidence probe ===")
    native_evi, evi_metrics = train_probe(X1, y, tag="native_evidence")
    cos_native = float(np.dot(native_evi, instruct_act))
    print(f"  cos(native_evidence, instruct_action) = {cos_native:.4f}")

    # C. Permutation test
    print(f"\n=== C. Permutation test (n={args.n_perm}) ===")
    obs_cos, null_dist, p_perm = permutation_test_cosine(
        X1, y, instruct_act, n_perm=args.n_perm)
    print(f"  Observed |cos| = {obs_cos:.4f}")
    print(f"  Null mean = {np.mean(null_dist):.4f}, std = {np.std(null_dist):.4f}")
    print(f"  p-value = {p_perm:.4f}")

    # D. Projection correlation
    print(f"\n=== D. Projection correlation ===")
    r_corr, p_corr = pearsonr(proj_evi, proj_act)
    rho_corr, p_rho = spearmanr(proj_evi, proj_act)
    print(f"  Pearson corr(h·evi, h·act) = {r_corr:.4f} (p={p_corr:.4f})")
    print(f"  Spearman rho = {rho_corr:.4f} (p={p_rho:.4f})")

    results = {
        "model": args.model, "layer": LAYER, "n_samples": len(records),
        "instruct_direction_transfer": {
            "evidence_auroc": float(auroc_evi_transfer),
            "action_margin_corr": float(r_act_margin),
            "action_margin_p": float(p_act),
        },
        "native_probe": evi_metrics,
        "cos_native_evi_instruct_act": cos_native,
        "permutation_test": {
            "observed_abs_cos": float(obs_cos),
            "null_mean": float(np.mean(null_dist)),
            "null_std": float(np.std(null_dist)),
            "p_value": p_perm,
            "n_perm": args.n_perm,
        },
        "projection_correlation": {
            "pearson_r": float(r_corr), "pearson_p": float(p_corr),
            "spearman_rho": float(rho_corr), "spearman_p": float(p_rho),
        },
        "margin_stats": {
            "mean": float(np.mean(margins1)),
            "std": float(np.std(margins1)),
            "min": float(np.min(margins1)),
            "max": float(np.max(margins1)),
        },
    }

    # E. Step 0 analysis (if requested)
    if args.do_step0 and "h0" in records[0]:
        print(f"\n=== E. Step 0 vs Step 1 comparison ===")
        X0 = np.array([r["h0"] for r in records], dtype=np.float32)
        margins0 = np.array([r["margin0"] for r in records])

        proj_evi_s0 = X0 @ instruct_evi
        proj_act_s0 = X0 @ instruct_act
        auroc_s0 = roc_auc_score(y, proj_evi_s0)
        r_s0, p_s0 = pearsonr(proj_evi_s0, proj_act_s0)
        r_s1, p_s1 = pearsonr(proj_evi, proj_act)

        print(f"  Step 0: instruct_evi transfer AUROC = {auroc_s0:.3f}")
        print(f"  Step 1: instruct_evi transfer AUROC = {auroc_evi_transfer:.3f}")
        print(f"  Step 0: corr(h·evi, h·act) = {r_s0:.4f} (p={p_s0:.4f})")
        print(f"  Step 1: corr(h·evi, h·act) = {r_s1:.4f} (p={p_s1:.4f})")
        print(f"  Step 0 margin: mean={np.mean(margins0):.2f} median={np.median(margins0):.2f}")
        print(f"  Step 1 margin: mean={np.mean(margins1):.2f} median={np.median(margins1):.2f}")

        results["step0"] = {
            "instruct_evi_transfer_auroc": float(auroc_s0),
            "projection_corr": {"pearson_r": float(r_s0), "p": float(p_s0)},
            "margin_mean": float(np.mean(margins0)),
            "margin_median": float(np.median(margins0)),
        }
        results["step1_projection_corr"] = {
            "pearson_r": float(r_s1), "p": float(p_s1)
        }

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  ★ SUMMARY: {args.model} (L{LAYER})")
    print(f"{'='*70}")
    print(f"  Native evidence probe AUROC: {evi_metrics['auroc']:.3f}")
    print(f"  Instruct evidence_dir transfer AUROC: {auroc_evi_transfer:.3f}")
    print(f"  cos(native_evi, instruct_act): {cos_native:.4f}")
    print(f"  Permutation p-value: {p_perm:.4f}")
    print(f"  Projection corr(h·evi, h·act): {r_corr:.4f} (p={p_corr:.4f})")
    d = X1.shape[1]
    random_cos = 1 / np.sqrt(d)
    print(f"  Random |cos| at d={d}: ~{random_cos:.4f}")
    print(f"  Observed / random = {obs_cos / random_cos:.1f}x")
    print(f"{'='*70}")

    tag = args.model.split("/")[-1].lower().replace("-", "_")
    out_path = os.path.join(args.output_dir, f"{tag}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
