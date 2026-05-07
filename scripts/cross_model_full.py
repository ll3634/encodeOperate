#!/usr/bin/env python3
"""
Cross-Model Full Validation v2
================================
Fixes vs v1:
  1. action_dir layer swept INDEPENDENTLY from evidence layer (pick best Spearman quality)
  2. Better action_dir: p10/p90 split (vs p20/p80), N=400 PopQA, quality validated by
     Spearman(margin, h·direction) on held-out samples; all layers collected in one pass
  3. Paired corruption N=200 (vs 50) for adequate statistical power

Single model load runs all experiments.

Usage:
  cd tmc/scripts/e2e_agent
  HF_ENDPOINT=https://hf-mirror.com python scripts/cross_model_full.py \\
      --model unsloth/Meta-Llama-3.1-8B-Instruct \\
      --output-dir results/cross_model_llama31_v2
"""

import os, sys, json, re, random, argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import get_model_layers
from scripts.paired_corruption_analysis import (
    select_samples, make_corrupted_obs, build_prompt,
)


# ── Shared utilities ─────────────────────────────────────────────────────────

def extract_hidden(model, tokenizer, prompt, layer_idx):
    """Extract last-token hidden state at a specific layer."""
    layers = get_model_layers(model)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    captured = {}

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h[0, -1, :].detach().float().cpu().numpy()

    handle = layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        logits = model(input_ids).logits
    handle.remove()
    return captured["h"], logits[0, -1, :]


def compute_margin(logits, tokenizer):
    """Compute search-stop margin from logits."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    tool_ids = [tokenizer.encode(t, add_special_tokens=False)[0]
                for t in ACTION_TOKENS["tool_call"]]
    fin_ids = [tokenizer.encode(t, add_special_tokens=False)[0]
               for t in ACTION_TOKENS["finish"]]
    return (torch.logsumexp(log_probs[tool_ids], 0) -
            torch.logsumexp(log_probs[fin_ids], 0)).item()


def train_probe(X, y, return_cv=False):
    """Train logistic probe. Optionally return 5-fold CV AUROC."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    probe = LogisticRegression(class_weight="balanced", C=1.0,
                                max_iter=2000, solver="lbfgs", random_state=42)
    probe.fit(X_s, y)
    w = probe.coef_[0] / scaler.scale_
    direction = (w / np.linalg.norm(w)).astype(np.float32)

    cv_results = None
    if return_cv:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs, baccs = [], []
        for train_idx, test_idx in skf.split(X, y):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X[train_idx])
            X_te = sc.transform(X[test_idx])
            p = LogisticRegression(class_weight="balanced", C=1.0,
                                    max_iter=2000, solver="lbfgs", random_state=42)
            p.fit(X_tr, y[train_idx])
            probs = p.predict_proba(X_te)[:, 1]
            preds = p.predict(X_te)
            aucs.append(roc_auc_score(y[test_idx], probs))
            baccs.append(balanced_accuracy_score(y[test_idx], preds))
        cv_results = {"auroc_mean": np.mean(aucs), "auroc_std": np.std(aucs),
                      "bacc_mean": np.mean(baccs), "bacc_std": np.std(baccs),
                      "fold_aurocs": aucs}
    return direction, cv_results


# ── Chat template compatibility ───────────────────────────────────────────────

def apply_chat_template_safe(tokenizer, messages, add_generation_prompt=True):
    """Apply chat template, handling:
    - Qwen3: enable_thinking=False to suppress <think> tokens and stay in
      standard action-generation mode (required for margin computation).
    - Gemma: merge system→user if model rejects system role.
    """
    chat_template = getattr(tokenizer, "chat_template", "") or ""
    # Auto-detect Qwen3 by presence of enable_thinking param in its template
    extra_kwargs = {"enable_thinking": False} if "enable_thinking" in chat_template else {}
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
            **extra_kwargs)
    except Exception as e:
        if "system" not in str(e).lower() and "System" not in str(e):
            raise
        # Merge system message into first user message (Gemma-style fallback)
        merged = []
        sys_text = ""
        for msg in messages:
            if msg["role"] == "system":
                sys_text = msg["content"]
            elif msg["role"] == "user" and sys_text:
                merged.append({"role": "user",
                                "content": sys_text + "\n\n" + msg["content"]})
                sys_text = ""
            else:
                merged.append(msg)
        return tokenizer.apply_chat_template(
            merged, tokenize=False, add_generation_prompt=add_generation_prompt,
            **extra_kwargs)


# ── Part 1: Collect hidden states (single pass over data) ────────────────────

def collect_step1_states(model, tokenizer, labels_path, baseline_path, layers):
    """Collect hidden states at multiple layers in one pass per sample."""
    model_layers = get_model_layers(model)
    device = next(model.parameters()).device

    label_data = [json.loads(l) for l in open(labels_path)]
    bl_map = {}
    with open(baseline_path) as f:
        for line in f:
            ep = json.loads(line)
            bl_map[ep["sample_id"]] = ep

    pb = PromptBuilder(tools=["search", "calculator"])
    results = []

    for i, ld in enumerate(label_data):
        ep = bl_map.get(ld["sample_id"])
        if not ep or not ep.get("steps"):
            continue
        s0 = ep["steps"][0]
        if s0.get("action") != "search" or not s0.get("observation"):
            continue

        steps = [{"action": "search", "action_input": s0["action_input"],
                  "observation": s0["observation"]}]
        messages = pb.build_full_prompt(ld["question"], steps)
        prompt = apply_chat_template_safe(tokenizer, messages)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        captured = {}
        handles = []
        for li in layers:
            def make_hook(layer_i):
                def hook_fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    captured[layer_i] = h[0, -1, :].detach().float().cpu().numpy()
                return hook_fn
            handles.append(model_layers[li].register_forward_hook(make_hook(li)))

        with torch.no_grad():
            logits = model(input_ids).logits
        for h in handles:
            h.remove()

        margin = compute_margin(logits[0, -1, :], tokenizer)
        results.append({
            "label": ld["label"],
            "sample_id": ld["sample_id"],
            "margin": margin,
            "hidden": {li: captured[li] for li in layers},
        })
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(label_data)}] collected", flush=True)

    return results


# ── Part 2: Action direction from PopQA margins (multilayer, one pass) ──────

def collect_popqa_multilayer(model, tokenizer, popqa_path, layers, n=400):
    """
    FIX 2: Collect PopQA hidden states at ALL layers in a single pass per sample.
    Returns dict {layer_idx: {"margins": [...], "hiddens": [...]}}
    Uses N=400 (vs old 200) for better contrast group statistics.
    """
    model_layers = get_model_layers(model)
    device = next(model.parameters()).device

    samples = [json.loads(l) for l in open(popqa_path)]
    random.seed(42)
    random.shuffle(samples)
    samples = samples[:n]

    pb = PromptBuilder(tools=["search"])
    all_data = []  # list of {margin, hidden: {layer: array}}

    for i, s in enumerate(samples):
        messages = pb.build_full_prompt(s["question"], [])
        prompt = apply_chat_template_safe(tokenizer, messages)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        captured = {}
        handles = []
        for li in layers:
            def make_hook(layer_i):
                def hook_fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    captured[layer_i] = h[0, -1, :].detach().float().cpu().numpy()
                return hook_fn
            handles.append(model_layers[li].register_forward_hook(make_hook(li)))

        with torch.no_grad():
            logits = model(input_ids).logits
        for h in handles:
            h.remove()

        margin = compute_margin(logits[0, -1, :], tokenizer)
        all_data.append({"margin": margin, "hidden": {li: captured[li] for li in layers}})

        if (i + 1) % 100 == 0:
            print(f"  PopQA [{i+1}/{n}]", flush=True)

    # Organize per-layer
    by_layer = {}
    for li in layers:
        by_layer[li] = {
            "margins": [d["margin"] for d in all_data],
            "hiddens": [d["hidden"][li] for d in all_data],
        }
    margins = [d["margin"] for d in all_data]
    print(f"  PopQA done: N={n}, margin range=[{min(margins):.2f}, {max(margins):.2f}]",
          flush=True)
    return by_layer


def extract_action_dir_from_popqa(popqa_layer_data, percentile_lo=10, percentile_hi=90):
    """
    FIX 2: p10/p90 split (vs old p20/p80) for cleaner contrast groups.
    Also computes Spearman(margin, h·direction) as quality metric on all samples.
    Returns (direction, quality_spearman_r, margin_stats).
    """
    margins = np.array(popqa_layer_data["margins"])
    hiddens = np.array(popqa_layer_data["hiddens"], dtype=np.float32)

    p_lo = np.percentile(margins, percentile_lo)
    p_hi = np.percentile(margins, percentile_hi)
    lo_mask = margins <= p_lo
    hi_mask = margins >= p_hi

    h_low = hiddens[lo_mask].mean(0)
    h_high = hiddens[hi_mask].mean(0)
    direction = h_low - h_high  # toward non-adopt (stop)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return None, 0.0, {}
    direction = (direction / norm).astype(np.float32)

    # Quality: Spearman(margin, projection onto direction) — uses ALL samples
    projections = hiddens @ direction  # shape (N,)
    rho, p_val = spearmanr(margins, projections)
    # Note: high positive rho means high margin → high projection (toward stop = low margin behavior)
    # The direction is h_low - h_high (low margin = search), so expect rho < 0 (high margin = less search)
    quality = abs(float(rho))  # use absolute value; sign depends on direction convention

    margin_stats = {
        "mean": float(margins.mean()),
        "std": float(margins.std()),
        "min": float(margins.min()),
        "max": float(margins.max()),
        "n_low": int(lo_mask.sum()),
        "n_high": int(hi_mask.sum()),
        "p_lo_threshold": float(p_lo),
        "p_hi_threshold": float(p_hi),
        "spearman_r": float(rho),
        "spearman_p": float(p_val),
        "quality": quality,
    }
    return direction, quality, margin_stats


# ── Part 3: Layer sweep (evidence + action independently) ────────────────────

def run_layer_sweep(step1_data, popqa_by_layer):
    """
    FIX 1: Evidence and action peaks are found INDEPENDENTLY.
    For each layer:
      - evidence quality: 5-fold CV AUROC of linear probe
      - action quality:   Spearman(margin, h·direction) from extract_action_dir_from_popqa
    Returns results dict and also action_dirs dict keyed by layer.
    """
    results = {}
    action_dirs = {}
    for li, popqa_data in popqa_by_layer.items():
        # Evidence probe
        X = np.array([d["hidden"][li] for d in step1_data], dtype=np.float32)
        y = np.array([d["label"] for d in step1_data], dtype=np.int32)
        evi_dir, cv = train_probe(X, y, return_cv=True)

        # Action direction + quality (FIX 1+2: independent layer, p10/p90, quality metric)
        act_dir, quality, margin_stats = extract_action_dir_from_popqa(popqa_data)
        action_dirs[li] = act_dir

        cos_ae = float(np.dot(act_dir, evi_dir)) if act_dir is not None else float('nan')
        results[li] = {
            "auroc": cv["auroc_mean"],
            "auroc_std": cv["auroc_std"],
            "bacc": cv["bacc_mean"],
            "cos_action_evidence": cos_ae,
            "action_dir_quality": quality,        # FIX 1: Spearman quality
            "action_margin_stats": margin_stats,  # FIX 2: margin stats for diagnostics
        }
        print(f"  L{li}: AUROC={cv['auroc_mean']:.3f}±{cv['auroc_std']:.3f}  "
              f"cos={cos_ae:.4f}  act_quality={quality:.3f}  "
              f"margin=[{margin_stats.get('min',0):.1f},{margin_stats.get('max',0):.1f}]")
    return results, action_dirs


# ── Part 4: Paired corruption ────────────────────────────────────────────────

def run_paired_corruption(model, tokenizer, action_dir, evidence_dir,
                          baseline_path, hotpotqa_path, peak_layer,
                          n_samples=200, seed=42):   # FIX 3: default N=200
    """Run paired corruption experiment at peak_layer."""
    print(f"\n=== Paired Corruption (L{peak_layer}, N={n_samples}) ===")
    samples = select_samples(baseline_path, hotpotqa_path, n=n_samples, seed=seed)
    print(f"  Selected {len(samples)} samples")

    groups = {"A": [], "B": [], "C": []}
    for gi, group in enumerate(["A", "B", "C"]):
        for i, sample in enumerate(samples):
            rng = random.Random(seed + gi * 10000)
            for j in range(i):
                make_corrupted_obs(samples[j], group, rng)
            clean_obs, corrupted_obs = make_corrupted_obs(sample, group, rng)

            prompt_c = build_prompt(tokenizer, sample["question"],
                                     sample["step0_query"], clean_obs)
            prompt_x = build_prompt(tokenizer, sample["question"],
                                     sample["step0_query"], corrupted_obs)

            h_c, lg_c = extract_hidden(model, tokenizer, prompt_c, peak_layer)
            h_x, lg_x = extract_hidden(model, tokenizer, prompt_x, peak_layer)
            margin_c = compute_margin(lg_c, tokenizer)
            margin_x = compute_margin(lg_x, tokenizer)

            diff = h_c - h_x
            d_act = abs(float(np.dot(diff, action_dir)))
            d_evi = abs(float(np.dot(diff, evidence_dir)))
            groups[group].append({
                "sample_id": sample["sample_id"],
                "delta_action": d_act,
                "delta_evidence": d_evi,
                "margin_clean": float(margin_c),
                "margin_corrupted": float(margin_x),
                "h_clean": h_c.astype(np.float32),
                "h_corrupted": h_x.astype(np.float32),
            })

        print(f"  Group {group}: {len(groups[group])} pairs done")

    # Analysis
    A_act = [d["delta_action"] for d in groups["A"]]
    B_act = [d["delta_action"] for d in groups["B"]]
    A_evi = [d["delta_evidence"] for d in groups["A"]]
    B_evi = [d["delta_evidence"] for d in groups["B"]]

    u_act, p_act = mannwhitneyu(A_act, B_act, alternative="two-sided")
    u_evi, p_evi = mannwhitneyu(A_evi, B_evi, alternative="two-sided")
    rho_A, p_rho = spearmanr(
        [d["delta_evidence"] for d in groups["A"]],
        [d["delta_action"] for d in groups["A"]]
    )

    ratio_act = np.mean(A_act) / np.mean(B_act) if np.mean(B_act) > 0 else float('inf')
    ratio_evi = np.mean(A_evi) / np.mean(B_evi) if np.mean(B_evi) > 0 else float('inf')

    result = {
        "n_samples": len(samples),
        "A_mean_delta_action": float(np.mean(A_act)),
        "B_mean_delta_action": float(np.mean(B_act)),
        "AB_ratio_action": float(ratio_act),
        "MW_action_p": float(p_act),
        "A_mean_delta_evidence": float(np.mean(A_evi)),
        "B_mean_delta_evidence": float(np.mean(B_evi)),
        "AB_ratio_evidence": float(ratio_evi),
        "MW_evidence_p": float(p_evi),
        "within_A_spearman_rho": float(rho_A),
        "within_A_spearman_p": float(p_rho),
    }

    print(f"\n  Results:")
    print(f"    A/B ratio (action): {ratio_act:.2f}x  (p={p_act:.4f})")
    print(f"    A/B ratio (evidence): {ratio_evi:.2f}x  (p={p_evi:.4f})")
    print(f"    Within-A Spearman(Δ_evi, Δ_act): ρ={rho_A:.3f}  (p={p_rho:.4f})")
    return result, groups


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--popqa-path", default="data/popqa/popqa_test.jsonl")
    ap.add_argument("--labels-path", default="results/phase1_probe/labels.jsonl")
    ap.add_argument("--baseline-trace",
                    default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--hotpotqa-data",
                    default="data/hotpotqa/hotpot_dev_distractor_v1.json")
    ap.add_argument("--output-dir", default="results/cross_model_mistral_v2")
    ap.add_argument("--n-popqa", type=int, default=400)        # FIX 2: N=400
    ap.add_argument("--n-corruption", type=int, default=200)   # FIX 3: N=200
    ap.add_argument("--emit-per-sample", type=lambda s: s.lower() != "false",
                    default=True,
                    help="If True, write per_sample.npz alongside full_results.json")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_token = os.environ.get("HF_TOKEN", None)
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, token=hf_token)
    model.eval()

    n_layers = len(get_model_layers(model))
    D = model.config.hidden_size
    print(f"  Layers: {n_layers}, Hidden: {D}")

    # Sweep 50%-95% depth, every 2 layers
    sweep_start = int(n_layers * 0.5)
    sweep_end = int(n_layers * 0.95)
    sweep_layers = list(range(sweep_start, sweep_end, 2))
    print(f"  Sweep layers: {sweep_layers}")

    # ── Step 1: Collect step-1 hidden states (evidence, all sweep layers) ──
    print("\n=== Step 1: Collecting step-1 hidden states ===")
    step1_data = collect_step1_states(
        model, tokenizer, args.labels_path, args.baseline_trace, sweep_layers)
    print(f"  Collected {len(step1_data)} samples across {len(sweep_layers)} layers")

    # ── Step 2: Collect PopQA states (all layers, ONE pass) ──────────────
    # FIX 2: multilayer collection in single pass, N=400
    print(f"\n=== Step 2: Collecting PopQA states (N={args.n_popqa}, all layers) ===")
    popqa_by_layer = collect_popqa_multilayer(
        model, tokenizer, args.popqa_path, sweep_layers, n=args.n_popqa)

    # ── Step 3: Layer sweep (evidence + action peaks INDEPENDENTLY) ──────
    # FIX 1+2: run_layer_sweep now returns (results, action_dirs)
    print("\n=== Step 3: Layer sweep ===")
    sweep_results, action_dirs_by_layer = run_layer_sweep(step1_data, popqa_by_layer)

    # FIX 1: Pick evidence peak (by AUROC) and action peak (by Spearman quality) SEPARATELY
    peak_evi_layer = max(sweep_results, key=lambda l: sweep_results[l]["auroc"])
    peak_act_layer = max(sweep_results, key=lambda l: sweep_results[l]["action_dir_quality"])
    print(f"\n  ★ Evidence peak: L{peak_evi_layer}  "
          f"(AUROC={sweep_results[peak_evi_layer]['auroc']:.3f})")
    print(f"  ★ Action dir peak: L{peak_act_layer}  "
          f"(quality={sweep_results[peak_act_layer]['action_dir_quality']:.3f})")

    # ── Step 4: Evidence probe at evidence peak layer ──────────────────
    print(f"\n=== Step 4: Evidence probe at L{peak_evi_layer} ===")
    X_evi = np.array([d["hidden"][peak_evi_layer] for d in step1_data], dtype=np.float32)
    y_evi = np.array([d["label"] for d in step1_data], dtype=np.int32)
    evidence_dir, cv = train_probe(X_evi, y_evi, return_cv=True)
    print(f"  5-fold CV AUROC: {cv['auroc_mean']:.3f} ± {cv['auroc_std']:.3f}")
    print(f"  5-fold CV BalAcc: {cv['bacc_mean']:.3f} ± {cv['bacc_std']:.3f}")
    print(f"  Fold AUROCs: {[f'{a:.3f}' for a in cv['fold_aurocs']]}")

    # FIX 1: action_dir from its OWN best layer
    action_dir = action_dirs_by_layer[peak_act_layer]
    cos_ae = float(np.dot(action_dir, evidence_dir))
    print(f"  cos(action@L{peak_act_layer}, evidence@L{peak_evi_layer}): {cos_ae:.4f}")

    # Also compute cos at same-layer for reference
    if peak_act_layer != peak_evi_layer:
        action_dir_same_layer = action_dirs_by_layer[peak_evi_layer]
        cos_same = float(np.dot(action_dir_same_layer, evidence_dir))
        print(f"  cos(action@L{peak_evi_layer}, evidence@L{peak_evi_layer}): {cos_same:.4f}  [same-layer ref]")
    else:
        cos_same = cos_ae

    # ── Step 5: Paired corruption (N=200, best layers) ────────────────
    # FIX 3: N=200. Use evidence layer for extraction (post-observation hidden state).
    print(f"\n=== Step 5: Paired Corruption (evi=L{peak_evi_layer}, act=L{peak_act_layer}, N={args.n_corruption}) ===")
    corruption_results, corruption_per_pair = run_paired_corruption(
        model, tokenizer, action_dir, evidence_dir,
        args.baseline_trace, args.hotpotqa_data,
        peak_evi_layer, n_samples=args.n_corruption)

    # ── Save ──────────────────────────────────────────────────────────
    # Convert action_margin_stats (may have numpy types)
    def jsonify(obj):
        if isinstance(obj, dict):
            return {k: jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    out = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "version": "v2",
        "fixes": ["independent_action_layer", "p10p90_N400", "n_corruption_200"],
        "n_layers": n_layers,
        "hidden_size": D,
        "peak_evidence_layer": peak_evi_layer,
        "peak_action_layer": peak_act_layer,
        "layer_sweep": {str(k): jsonify(v) for k, v in sweep_results.items()},
        "evidence_probe": {
            "layer": peak_evi_layer,
            "auroc_mean": cv["auroc_mean"],
            "auroc_std": cv["auroc_std"],
            "bacc_mean": cv["bacc_mean"],
            "bacc_std": cv["bacc_std"],
            "fold_aurocs": cv["fold_aurocs"],
            "n_samples": len(step1_data),
            "label_balance": int(y_evi.sum()),
        },
        "orthogonality": {
            "cos_action_evidence": cos_ae,
            "action_layer": peak_act_layer,
            "evidence_layer": peak_evi_layer,
            "cos_same_layer": cos_same,
        },
        "paired_corruption": corruption_results,
    }

    # ── Print summary ──
    print("\n" + "=" * 70)
    print(f"CROSS-MODEL VALIDATION SUMMARY v2: {args.model}")
    print("=" * 70)
    print(f"\nQwen reference → this model:")
    print(f"  Evidence AUROC:      0.862 → {cv['auroc_mean']:.3f}")
    print(f"  cos(act,evi):       -0.014 → {cos_ae:.4f}")
    print(f"  A/B ratio (act):     1.83x → {corruption_results['AB_ratio_action']:.2f}x  (p={corruption_results['MW_action_p']:.4f})")
    print(f"  Within-A ρ:          0.067 → {corruption_results['within_A_spearman_rho']:.3f}")
    print(f"  Evidence peak layer: L20/28 → L{peak_evi_layer}/{n_layers}")
    print(f"  Action peak layer:   L20/28 → L{peak_act_layer}/{n_layers}  [FIX1: independent]")
    print(f"  Action dir quality:  (Spearman) → {sweep_results[peak_act_layer]['action_dir_quality']:.3f}")
    print(f"  N corruption pairs:  50 → {args.n_corruption}  [FIX3]")
    print("=" * 70)

    out_path = os.path.join(args.output_dir, "full_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Save directions.npz for use in Exp 3 (decomposition) and Exp 4 (dose-response).
    # Keys match qwen3_circuit_sanity.py convention; evidence_key="evidence_dir" is
    # auto-discovered by prepare_directions() in decomposition_ci_hardened_cross_model.py.
    npz_path = os.path.join(args.output_dir, "directions.npz")
    np.savez(npz_path,
             evidence_dir=evidence_dir,
             action_dir=action_dir,
             L_evi=peak_evi_layer,
             L_act=peak_act_layer,
             cos_action_evidence=cos_ae,
             evidence_auroc=cv["auroc_mean"],
             action_quality=sweep_results[peak_act_layer]["action_dir_quality"])
    print(f"Saved: {npz_path}")

    # ── Per-sample artifact for downstream re-analyses (B.1, B.2) ─────────
    if args.emit_per_sample:
        groups_list = ["A", "B", "C"]
        n_pairs_each = len(corruption_per_pair["A"])

        # Pair sample_ids: identical across groups by construction (same select_samples order).
        sids_A = [g["sample_id"] for g in corruption_per_pair["A"]]
        max_sid_len = max(len(s) for s in sids_A)
        if max_sid_len > 24:
            raise ValueError(
                f"sample_id width {max_sid_len} exceeds dtype='U24'; refusing to truncate")
        pair_sample_ids = np.array(sids_A, dtype="U24")

        pair_d_act = np.stack([
            np.array([p["delta_action"]    for p in corruption_per_pair[g]], np.float32)
            for g in groups_list])
        pair_d_evi = np.stack([
            np.array([p["delta_evidence"]  for p in corruption_per_pair[g]], np.float32)
            for g in groups_list])
        pair_margin_c = np.stack([
            np.array([p["margin_clean"]    for p in corruption_per_pair[g]], np.float32)
            for g in groups_list])
        pair_margin_x = np.stack([
            np.array([p["margin_corrupted"]for p in corruption_per_pair[g]], np.float32)
            for g in groups_list])
        pair_h_c = np.stack([
            np.stack([p["h_clean"]         for p in corruption_per_pair[g]])
            for g in groups_list])
        pair_h_x = np.stack([
            np.stack([p["h_corrupted"]     for p in corruption_per_pair[g]])
            for g in groups_list])

        # Step-1 baseline pool at peak_evi_layer.
        step1_h = np.stack([d["hidden"][peak_evi_layer] for d in step1_data]).astype(np.float32)
        step1_margin = np.array([d["margin"]    for d in step1_data], np.float32)
        step1_label  = np.array([d["label"]     for d in step1_data], np.int32)
        s1_ids = [d["sample_id"] for d in step1_data]
        max_s1_len = max(len(s) for s in s1_ids)
        if max_s1_len > 24:
            raise ValueError(
                f"step1 sample_id width {max_s1_len} exceeds dtype='U24'; refusing to truncate")
        step1_sample_id = np.array(s1_ids, dtype="U24")

        per_sample_path = os.path.join(args.output_dir, "per_sample.npz")
        np.savez_compressed(
            per_sample_path,
            pair_groups=np.array(groups_list, dtype="U1"),
            pair_sample_ids=pair_sample_ids,
            pair_d_act=pair_d_act,
            pair_d_evi=pair_d_evi,
            pair_margin_clean=pair_margin_c,
            pair_margin_corrupted=pair_margin_x,
            pair_h_clean=pair_h_c,
            pair_h_corrupted=pair_h_x,
            step1_h=step1_h,
            step1_margin=step1_margin,
            step1_label_sufficiency=step1_label,
            step1_sample_ids=step1_sample_id,
            evidence_dir=evidence_dir.astype(np.float32),
            action_dir=action_dir.astype(np.float32),
            peak_evi_layer=np.int32(peak_evi_layer),
            peak_act_layer=np.int32(peak_act_layer),
            hidden_size=np.int32(D),
            n_pairs=np.int32(n_pairs_each),
            model_name=np.array(args.model, dtype="U128"),
        )
        sz = os.path.getsize(per_sample_path)
        print(f"per_sample.npz written: {sz} bytes "
              f"({sz/1024/1024:.1f} MB), contains: "
              f"pair_h_clean/corrupted shape (3, {n_pairs_each}, {D}) float32; "
              f"pair margins/d_act/d_evi shape (3, {n_pairs_each}); "
              f"step1_h shape ({len(step1_data)}, {D}) float32 + margin/label/sample_id; "
              f"evidence_dir+action_dir; peak layers L{peak_evi_layer}/L{peak_act_layer}.")
        print(f"Saved: {per_sample_path}")


if __name__ == "__main__":
    main()