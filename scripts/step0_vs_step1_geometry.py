#!/usr/bin/env python3
"""
Step 0 (pre-first-tool) vs Step 1 (post-first-tool) Representational Geometry
==============================================================================
Compare how observation intake changes the evidence-action geometry at L20.

Step 0: Model sees only the question (decides whether to call search)
Step 1: Model sees question + first search observation (decides whether to
        search again or stop)

Measurements:
  1. cos(evidence_dir, action_dir) at each step
  2. Evidence probe AUROC at each step
  3. Margin distribution by label at each step (TE direction proxy)
  4. Per-component evidence projection decomposition

Usage:
  cd tmc/scripts/e2e_agent
  python scripts/step0_vs_step1_geometry.py
"""

import os, sys, json, argparse, random, time
import numpy as np
from pathlib import Path
from collections import defaultdict

# Critical: flush every print immediately so tailing the log works
sys.stdout.reconfigure(line_buffering=True)

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import get_model_layers

# Default single-layer for backward compat; overridden by --layers
LAYER = 20
TARGET_LAYERS = [4, 8, 12, 16, 18, 20, 22, 24]


def get_margin(logits, tokenizer):
    """Compute search-stop logit margin from logits."""
    log_probs = torch.log_softmax(logits, dim=-1)
    tool_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]
    return (torch.logsumexp(log_probs[tool_ids], 0) - torch.logsumexp(log_probs[fin_ids], 0)).item()


def extract_multilayer(model, tokenizer, prompt, layers_list):
    """Single forward pass → hidden states at all target layers + margin."""
    all_layers = get_model_layers(model)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    captured = {}
    def make_hook(L):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[L] = h[0, -1, :].detach().float().cpu().numpy()
        return fn

    handles = [all_layers[L].register_forward_hook(make_hook(L)) for L in layers_list]
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :]
    for h in handles:
        h.remove()

    margin = get_margin(logits, tokenizer)
    return captured, margin  # dict {L: hidden_vec}, scalar


def extract_hidden_and_margin(model, tokenizer, prompt, layer):
    """Single-layer backward-compat wrapper."""
    caps, margin = extract_multilayer(model, tokenizer, prompt, [layer])
    return caps[layer], margin


def build_step0_prompt(tokenizer, question):
    """Step 0: question only, no prior steps."""
    pb = PromptBuilder(tools=["search", "calculator"])
    messages = pb.build_full_prompt(question, [])
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


def build_step1_prompt(tokenizer, question, query, observation):
    """Step 1: question + first search + observation."""
    pb = PromptBuilder(tools=["search", "calculator"])
    steps = [{"action": "search", "action_input": query,
              "observation": observation[:1500]}]
    messages = pb.build_full_prompt(question, steps)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


def train_probe_and_get_direction(X, y, tag=""):
    """Train evidence probe, return direction and metrics."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                           solver="lbfgs", random_state=42),
        X_scaled, y, cv=cv, scoring="balanced_accuracy")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X_scaled, y))
    probe = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                                solver="lbfgs", random_state=42)
    probe.fit(X_scaled[train_idx], y[train_idx])
    auroc = roc_auc_score(y[test_idx], probe.predict_proba(X_scaled[test_idx])[:, 1])

    # Full probe for direction
    probe_full = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                                     solver="lbfgs", random_state=42)
    probe_full.fit(X_scaled, y)
    w = probe_full.coef_[0] / scaler.scale_
    direction = (w / np.linalg.norm(w)).astype(np.float32)

    print(f"  [{tag}] CV BalAcc: {cv_scores.mean():.3f}±{cv_scores.std():.3f}, "
          f"AUROC: {auroc:.3f}, N={len(y)} ({y.sum()} pos, {(1-y).sum()} neg)")
    return direction, {
        "cv_balanced_accuracy": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "auroc": float(auroc),
        "n": len(y), "n_pos": int(y.sum()), "n_neg": int((1-y).sum()),
    }


def extract_action_direction(model, tokenizer, popqa_path, layer, n_samples=200):
    """Same as base_model_orthogonality.py: PopQA margin contrastive."""
    print("\n=== Extracting action_dir (PopQA margin contrastive) ===")
    layers = get_model_layers(model)
    device = next(model.parameters()).device

    samples = []
    with open(popqa_path) as f:
        for line in f:
            samples.append(json.loads(line))
    random.seed(42)
    random.shuffle(samples)
    samples = samples[:n_samples]

    pb = PromptBuilder(tools=["search"])
    all_data = []
    for i, s in enumerate(samples):
        messages = pb.build_full_prompt(s["question"], [])
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        h, margin = extract_hidden_and_margin(model, tokenizer, prompt, layer)
        all_data.append({"margin": margin, "hidden": h})
        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{n_samples}]")

    margins = [d["margin"] for d in all_data]
    p20, p80 = np.percentile(margins, 20), np.percentile(margins, 80)
    low = [d for d in all_data if d["margin"] <= p20]
    high = [d for d in all_data if d["margin"] >= p80]
    h_low = np.mean(np.stack([d["hidden"] for d in low]), axis=0)
    h_high = np.mean(np.stack([d["hidden"] for d in high]), axis=0)
    direction = h_low - h_high
    direction = (direction / np.linalg.norm(direction)).astype(np.float32)
    print(f"  action_dir extracted. Margin: [{min(margins):.2f}, {max(margins):.2f}], "
          f"mean={np.mean(margins):.2f}")
    return direction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-path", default="results/phase1_probe/labels.jsonl")
    ap.add_argument("--baseline-trace", default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--popqa-path", default="data/popqa/popqa_test.jsonl")
    ap.add_argument("--output-dir", default="results/step0_vs_step1_geometry")
    ap.add_argument("--layer", type=int, default=None,
                    help="Single layer (legacy). If omitted, sweeps all TARGET_LAYERS.")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="Explicit list of layers to sweep.")
    ap.add_argument("--n-popqa", type=int, default=200)
    ap.add_argument("--no-popqa", action="store_true",
                    help="Skip PopQA pass; derive action_dir from margin_before in labels.")
    args = ap.parse_args()

    layers_to_run = args.layers or TARGET_LAYERS
    if args.layer is not None:
        layers_to_run = [args.layer]   # single-layer legacy mode
    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    device = next(model.parameters()).device
    print(f"  Model on: {device}  |  GPU mem: "
          f"{torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
    print(f"  Sweeping layers: {layers_to_run}", flush=True)

    # ── Load labels and baseline trace ────────────────────────────────
    print("\nLoading labels and trace...", flush=True)
    label_data = []
    with open(args.labels_path) as f:
        for line in f:
            label_data.append(json.loads(line))

    bl_map = {}
    with open(args.baseline_trace) as f:
        for line in f:
            ep = json.loads(line)
            bl_map[ep["sample_id"]] = ep

    # ── Filter samples with valid step-1 observation ───────────────────
    valid_samples = []
    for ld in label_data:
        sid = ld["sample_id"]
        ep = bl_map.get(sid)
        if not ep or not ep.get("steps"):
            continue
        s0 = ep["steps"][0]
        if s0.get("action") != "search" or not s0.get("observation"):
            continue
        valid_samples.append((ld, s0))

    N = len(valid_samples)
    print(f"  {N} valid samples with step-1 observation", flush=True)
    y = np.array([ld["label"] for ld, _ in valid_samples], dtype=np.int32)
    # margin_before from labels (action margin at step-1 without intervention)
    margins_s1_ref = np.array([float(ld.get("margin_before", float("nan")))
                                for ld, _ in valid_samples])

    # ── GPU extraction: ALL target layers in a single pass per sample ──
    print(f"\n=== Extracting {N} × 2 forward passes at layers {layers_to_run} ===",
          flush=True)
    step0_H = {L: [] for L in layers_to_run}
    step1_H = {L: [] for L in layers_to_run}
    step0_margins, step1_margins = [], []

    pb = PromptBuilder(tools=["search", "calculator"])
    t0 = time.time()
    for i, (ld, s0) in enumerate(valid_samples):
        question = ld["question"]

        # Step 0: question only
        msgs0 = pb.build_full_prompt(question, [])
        p0 = tokenizer.apply_chat_template(msgs0, tokenize=False,
                                           add_generation_prompt=True)
        caps0, m0 = extract_multilayer(model, tokenizer, p0, layers_to_run)
        for L in layers_to_run:
            step0_H[L].append(caps0[L])
        step0_margins.append(m0)

        # Step 1: question + first observation
        steps = [{"action": "search", "action_input": s0["action_input"],
                  "observation": s0["observation"][:1500]}]
        msgs1 = pb.build_full_prompt(question, steps)
        p1 = tokenizer.apply_chat_template(msgs1, tokenize=False,
                                           add_generation_prompt=True)
        caps1, m1 = extract_multilayer(model, tokenizer, p1, layers_to_run)
        for L in layers_to_run:
            step1_H[L].append(caps1[L])
        step1_margins.append(m1)

        if (i + 1) % 50 == 0 or (i + 1) == N:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N - i - 1)
            print(f"  [{i+1}/{N}]  {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining",
                  flush=True)

    # Convert to arrays
    for L in layers_to_run:
        step0_H[L] = np.array(step0_H[L], dtype=np.float32)
        step1_H[L] = np.array(step1_H[L], dtype=np.float32)
    step0_margins = np.array(step0_margins)
    step1_margins = np.array(step1_margins)

    # ── Per-layer analysis ─────────────────────────────────────────────
    print(f"\n=== Per-layer analysis ===", flush=True)
    from scipy.stats import mannwhitneyu

    PERM_N = 100
    rng = np.random.RandomState(42)

    def action_dir_from_H(H, margins, pct=20):
        lo = np.percentile(margins, pct)
        hi = np.percentile(margins, 100 - pct)
        d = H[margins >= hi].mean(0) - H[margins <= lo].mean(0)
        return (d / np.linalg.norm(d)).astype(np.float32)

    def cross_r_and_p(H, evi_dir, margins):
        scores = H @ evi_dir
        r, _ = pearsonr(scores, margins)
        null = [pearsonr(H @ rng.permutation(evi_dir), margins)[0]
                for _ in range(PERM_N)]
        null = np.array(null)
        p = (np.sum(np.abs(null) >= abs(r)) + 1) / (PERM_N + 1)
        return float(r), float(p)

    all_results = {}
    for L in layers_to_run:
        print(f"\n  --- L{L} ---", flush=True)
        Hs0 = step0_H[L]
        Hs1 = step1_H[L]

        evi_s0, m_s0 = train_probe_and_get_direction(Hs0, y, tag=f"L{L}-s0")
        evi_s1, m_s1 = train_probe_and_get_direction(Hs1, y, tag=f"L{L}-s1")

        # Action direction (from step-1 margins measured during extraction)
        act = action_dir_from_H(Hs1, step1_margins)
        cos_s0 = float(np.dot(evi_s0, act))
        cos_s1 = float(np.dot(evi_s1, act))

        # Cross-prediction: does evi_dir predict action margin?
        r_s0, p_r_s0 = cross_r_and_p(Hs0, evi_s0, step0_margins)
        r_s1, p_r_s1 = cross_r_and_p(Hs1, evi_s1, step1_margins)

        # Margin by label MW test (step 1 only)
        m1_pos = step1_margins[y == 1]
        m1_neg = step1_margins[y == 0]
        _, mw_p = mannwhitneyu(m1_pos, m1_neg, alternative='two-sided')

        print(f"    AUROC s0={m_s0['auroc']:.4f}  s1={m_s1['auroc']:.4f}  "
              f"Δ={m_s1['auroc']-m_s0['auroc']:+.4f}", flush=True)
        print(f"    cos(evi,act): s0={cos_s0:+.4f}  s1={cos_s1:+.4f}", flush=True)
        print(f"    cross-r: s0={r_s0:+.4f}(p={p_r_s0:.3f})  "
              f"s1={r_s1:+.4f}(p={p_r_s1:.3f})", flush=True)

        all_results[L] = dict(
            auroc_s0=m_s0["auroc"], auroc_s1=m_s1["auroc"],
            cos_s0=cos_s0, cos_s1=cos_s1,
            cross_r_s0=r_s0, cross_r_p_s0=p_r_s0,
            cross_r_s1=r_s1, cross_r_p_s1=p_r_s1,
            margin_mw_p=float(mw_p),
        )

    # ── Summary table ──────────────────────────────────────────────────
    def sig(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "  "

    print(f"\n{'='*95}", flush=True)
    print("  ★ MULTI-LAYER GEOMETRY SWEEP — SUMMARY", flush=True)
    print(f"{'='*95}", flush=True)
    print(f"  {'L':>3}  {'AUROC_s0':>9} {'AUROC_s1':>9} {'ΔAUROC':>7}"
          f"  {'cos_s0':>7} {'cos_s1':>7}"
          f"  {'xr_s0':>7} {'xr_s1':>7}", flush=True)
    print("-" * 95, flush=True)
    for L in layers_to_run:
        r = all_results[L]
        da = r['auroc_s1'] - r['auroc_s0']
        print(f"  L{L:>2}  {r['auroc_s0']:>9.4f} {r['auroc_s1']:>9.4f} {da:>+7.4f}"
              f"  {r['cos_s0']:>+7.4f} {r['cos_s1']:>+7.4f}"
              f"  {r['cross_r_s0']:>+7.4f}{sig(r['cross_r_p_s0'])}"
              f" {r['cross_r_s1']:>+7.4f}{sig(r['cross_r_p_s1'])}",
              flush=True)
    print(f"{'='*95}", flush=True)

    # Interpretation
    print("\n  Q1 — Progressive decoupling?", flush=True)
    early_xr = np.mean([abs(all_results[L]['cross_r_s1'])
                         for L in layers_to_run if L <= 8])
    late_xr  = np.mean([abs(all_results[L]['cross_r_s1'])
                         for L in layers_to_run if L >= 20])
    if early_xr and late_xr:
        print(f"     |cross-r| early(≤L8)={early_xr:.4f}  late(≥L20)={late_xr:.4f}",
              flush=True)
        if early_xr > late_xr * 1.5:
            print("     → PROGRESSIVE: evi-action coupling decays → computational decoupling",
                  flush=True)
        elif abs(early_xr - late_xr) < 0.02:
            print("     → FLAT: coupling absent at all layers → architectural (trivial)",
                  flush=True)
        else:
            print(f"     → MIXED pattern", flush=True)

    print("\n  Q2 — Does observation change geometry?", flush=True)
    for L in [L for L in layers_to_run if L in [4, 12, 20]]:
        r = all_results[L]
        da = r['auroc_s1'] - r['auroc_s0']
        print(f"     L{L}: ΔAUROC={da:+.4f}  "
              f"Δcos={r['cos_s1']-r['cos_s0']:+.4f}  "
              f"Δxr={r['cross_r_s1']-r['cross_r_s0']:+.4f}", flush=True)

    # ── Save ──────────────────────────────────────────────────────────
    out = os.path.join(args.output_dir, "multilayer_results.json")
    with open(out, "w") as f:
        json.dump({"layers": layers_to_run, "n_samples": N,
                   "per_layer": {str(L): v for L, v in all_results.items()}},
                  f, indent=2)
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
