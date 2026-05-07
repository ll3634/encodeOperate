#!/usr/bin/env python3
"""
Experiment C: KV2 Ablation → L20 Evidence Probe AUROC
======================================================
Compare evidence probe AUROC at L20 under three conditions:
  - baseline   : no ablation (cached from activations_multilayer.npz)
  - kv2_ablate : zero out H14-H20 contribution in attn_L18 at ALL positions
  - kv0_ablate : zero out H0-H6  contribution in attn_L18 at ALL positions (control)

If AUROC drops under kv2_ablate  → KV2 participates in building evidence representation.
If AUROC stays the same          → KV2 is purely action mediator (evidence built elsewhere).

Output → results/kv2_ablation_probe_auroc/results.json
"""

import os, sys, json, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder
from steering.hook_utils import get_model_layers

# ── GQA constants for Qwen2.5-7B-Instruct ──────────────────────────────────
N_HEADS    = 28
N_KV       = 4
HEAD_DIM   = 128
HEADS_PER_KV = N_HEADS // N_KV   # 7
ABLATE_LAYER = 18
MEASURE_LAYER = 20


def kv_slice(kv_group: int):
    s = kv_group * HEADS_PER_KV * HEAD_DIM
    e = s + HEADS_PER_KV * HEAD_DIM
    return s, e


def make_ablation_hook(o_proj, kv_group: int):
    """Return a forward hook for o_proj that zeroes KV group's contribution at ALL positions."""
    s, e = kv_slice(kv_group)

    def hook_fn(module, inp, out):
        x = inp[0]                            # (batch, seq, n_heads*head_dim)
        W_slice = module.weight[:, s:e]       # (d_model, kv_dim)  float16
        kv_contribs = (x[0, :, s:e] @ W_slice.T)  # (seq, d_model)
        out_mod = out.clone()
        out_mod[0] -= kv_contribs
        return out_mod

    return hook_fn


def collect_activations_with_hook(model, tokenizer, episodes, label_map, hook_fn=None):
    """
    Forward all episodes through model (optionally with a hook on attn_L18 o_proj).
    Returns X (N, d), y (N,), sample_ids list.
    """
    pb = PromptBuilder(tools=["search", "calculator"])
    layers = get_model_layers(model)
    device = next(model.parameters()).device

    hidden_list, labels, sample_ids = [], [], []

    # Register hook if requested
    handle = None
    if hook_fn is not None:
        handle = layers[ABLATE_LAYER].self_attn.o_proj.register_forward_hook(hook_fn)

    # Capture hook for L20
    l20_cap = {}

    def l20_hook(m, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        l20_cap['h'] = h[0, -1, :].detach().float().cpu().numpy()

    h20 = layers[MEASURE_LAYER].register_forward_hook(l20_hook)

    skipped = 0
    for i, ep in enumerate(episodes):
        sid = ep["sample_id"]
        if sid not in label_map:
            skipped += 1
            continue
        steps = ep.get("steps", [])
        if not steps or steps[0].get("action") != "search" or not steps[0].get("observation"):
            skipped += 1
            continue

        history = [{"action": "search",
                    "action_input": steps[0]["action_input"],
                    "observation":  steps[0]["observation"]}]
        messages = pb.build_full_prompt(ep["question"], history)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        try:
            with torch.no_grad():
                model(input_ids)
        except Exception as exc:
            print(f"  [{i+1}] ERROR {sid[:20]}: {exc}")
            skipped += 1
            continue

        hidden_list.append(l20_cap['h'])
        labels.append(label_map[sid])
        sample_ids.append(sid)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(episodes)}] collected={len(labels)} skipped={skipped}",
                  flush=True)

    h20.remove()
    if handle is not None:
        handle.remove()

    X = np.array(hidden_list, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"  Done: N={len(y)}, label=0: {(y==0).sum()}, label=1: {(y==1).sum()}, "
          f"skipped={skipped}")
    return X, y, sample_ids


def train_probe(X, y, seed=42):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(sss.split(X_s, y))
    clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                             solver="lbfgs", random_state=seed)
    clf.fit(X_s[tr], y[tr])
    y_prob = clf.predict_proba(X_s[te])[:, 1]
    y_pred = clf.predict(X_s[te])
    return {
        "auroc": float(roc_auc_score(y[te], y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y[te], y_pred)),
        "n_train": len(tr),
        "n_test": len(te),
        "n_test_label0": int((y[te] == 0).sum()),
        "n_test_label1": int((y[te] == 1).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--labels", default="results/phase1_probe/labels.jsonl")
    ap.add_argument("--baseline-trace", default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--cached-acts", default="results/phase1_probe/activations_multilayer.npz",
                    help="If exists, load baseline L20 acts from here (skip model for baseline)")
    ap.add_argument("--output-dir", default="results/kv2_ablation_probe_auroc")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    SEP = "=" * 65
    print(SEP)
    print("  Experiment C: KV2 Ablation → Evidence Probe AUROC at L20")
    print(SEP)

    # Load labels
    label_map = {}
    with open(args.labels) as f:
        for line in f:
            r = json.loads(line)
            label_map[r["sample_id"]] = r["label"]
    print(f"Labels: {len(label_map)} samples, "
          f"label=0: {sum(1 for v in label_map.values() if v==0)}, "
          f"label=1: {sum(1 for v in label_map.values() if v==1)}")

    # Load episodes
    episodes = []
    with open(args.baseline_trace) as f:
        for line in f:
            episodes.append(json.loads(line))
    print(f"Episodes: {len(episodes)}")

    # ── Baseline: try cached activations first ───────────────────────────────
    results = {}
    cached_path = Path(args.cached_acts)
    if cached_path.exists():
        print(f"\n[Baseline] Loading cached L20 acts from {cached_path}")
        data = np.load(cached_path, allow_pickle=True)
        X_bl = data["layer_20"].astype(np.float32)
        y_bl = data["y"].astype(np.int32)
        m_bl = train_probe(X_bl, y_bl, seed=args.seed)
        results["baseline_cached"] = m_bl
        print(f"  Baseline AUROC = {m_bl['auroc']:.4f}  BalAcc = {m_bl['balanced_accuracy']:.4f}")
    else:
        print(f"  No cached acts found at {cached_path}; will run model for baseline too.")

    # ── Load model (bfloat16 to match cached baseline) ───────────────────────
    print("\nLoading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")
    model.eval()
    print("Model loaded.")

    layers = get_model_layers(model)
    o_proj = layers[ABLATE_LAYER].self_attn.o_proj

    # Always run fresh no-hook baseline for apples-to-apples comparison
    conditions = [
        ("baseline_fresh", None, "No ablation (fresh, bfloat16, same run as ablated)"),
        ("kv2_ablate", 2, "KV2 ablation (H14-H20 zeroed, attn_L18, all positions)"),
        ("kv0_ablate", 0, "KV0 ablation (H0-H6  zeroed, attn_L18, all positions) [control]"),
    ]

    X_baseline_fresh = None
    for cond_name, kv_group, desc in conditions:
        print(f"\n{SEP}\n[{cond_name}] {desc}\n{SEP}")
        hook_fn = make_ablation_hook(o_proj, kv_group) if kv_group is not None else None
        X, y, sids = collect_activations_with_hook(
            model, tokenizer, episodes, label_map, hook_fn=hook_fn)
        m = train_probe(X, y, seed=args.seed)
        results[cond_name] = m
        print(f"  AUROC = {m['auroc']:.4f}  BalAcc = {m['balanced_accuracy']:.4f}")

        # Sanity check: compare activation norms to fresh baseline
        if cond_name == "baseline_fresh":
            X_baseline_fresh = X.copy()
        elif X_baseline_fresh is not None:
            norm_diff = float(np.mean(np.linalg.norm(X - X_baseline_fresh, axis=1)))
            rel_diff  = norm_diff / float(np.mean(np.linalg.norm(X_baseline_fresh, axis=1)))
            print(f"  [sanity] mean |X_ablate - X_baseline| = {norm_diff:.4f}  "
                  f"relative = {rel_diff:.4f}  (should be > 0 if hook fired)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SUMMARY")
    print(SEP)
    bl_auroc = results.get("baseline_cached", results.get("baseline", {})).get("auroc", None)
    for k, m in results.items():
        delta = f"  Δ={m['auroc'] - bl_auroc:+.4f}" if (bl_auroc and k != "baseline_cached") else ""
        print(f"  {k:<22} AUROC={m['auroc']:.4f}  BalAcc={m['balanced_accuracy']:.4f}{delta}")

    # Save
    out = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "ablate_layer": ABLATE_LAYER,
        "measure_layer": MEASURE_LAYER,
        "kv_group_size": HEADS_PER_KV,
        "seed": args.seed,
        "results": results,
    }
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
