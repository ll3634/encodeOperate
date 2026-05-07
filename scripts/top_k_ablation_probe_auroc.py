#!/usr/bin/env python3
"""
Experiment C-ext: Top-k Component Ablation → L20 Evidence Probe AUROC
======================================================================
Extends kv2_ablation_probe_auroc.py to ablate full components (attn/mlp) at
all positions and measure evidence probe AUROC at L20.

Best-practice design:
  * Ablation mode (--mode {zero,mean}, default=mean):
      - zero : set component output to 0 at all positions (deprecated; OOD)
      - mean : replace component output with channel-wise global mean
               (collected over baseline forward pass, all samples/positions)
  * Fixed train/test split (same seed across conditions)
  * Dual evaluation:
      - fresh_auroc      : retrain probe on ablated activations
      - transferred_auroc: apply baseline-trained probe to ablated activations
  * Sanity: mean |X_ablate - X_baseline_fresh| per condition
  * All conditions run in the same model load / precision (bfloat16)

Component sets (chosen from attribution_patching top-7 restricted to ≤ L20):
  attn_L18_full   : [(attn,18)]                                 -- full-layer check vs KV2-only
  top4_pre_L20    : [(attn,18),(mlp,18),(attn,19),(mlp,20)]     -- top-7 ∩ {≤ L20}
  top6_L18_to_L20 : [(attn,18..20),(mlp,18..20)]                -- nuclear
  bottom_ctrl     : [(attn,14),(mlp,14),(attn,15),(mlp,15)]     -- size-matched low-recovery control

Output → results/top_k_ablation_probe_auroc/results.json
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder
from steering.hook_utils import get_model_layers

MEASURE_LAYER = 20

ABLATION_SETS = {
    "attn_L18_full":   [("attn", 18)],
    "top4_pre_L20":    [("attn", 18), ("mlp", 18), ("attn", 19), ("mlp", 20)],
    "top6_L18_to_L20": [("attn", 18), ("mlp", 18), ("attn", 19),
                        ("mlp", 19), ("attn", 20), ("mlp", 20)],
    "bottom_ctrl":     [("attn", 14), ("mlp", 14), ("attn", 15), ("mlp", 15)],
}


def make_zero_hook():
    """Zero the component's output at all positions."""
    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            return (torch.zeros_like(out[0]),) + out[1:]
        return torch.zeros_like(out)
    return hook_fn


def make_mean_hook(mean_vec: torch.Tensor):
    """Replace component output with channel-wise mean broadcast over all positions."""
    def hook_fn(module, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        mv = mean_vec.to(device=t.device, dtype=t.dtype)
        repl = mv.view(1, 1, -1).expand_as(t).contiguous()
        if isinstance(out, tuple):
            return (repl,) + out[1:]
        return repl
    return hook_fn


def make_mean_collector(store: dict, key):
    """Accumulate channel-wise sum and count of a component's output."""
    def hook_fn(module, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        s = t.detach().float().sum(dim=(0, 1)).cpu().numpy()
        c = int(t.shape[0] * t.shape[1])
        if key in store:
            store[key]['sum'] += s
            store[key]['count'] += c
        else:
            store[key] = {'sum': s, 'count': c}
    return hook_fn


def register_ablation(layers, spec, mode="zero", mean_cache=None):
    handles = []
    for ctype, layer_idx in spec:
        target = layers[layer_idx].self_attn if ctype == "attn" else layers[layer_idx].mlp
        if mode == "zero":
            handles.append(target.register_forward_hook(make_zero_hook()))
        elif mode == "mean":
            if mean_cache is None or (ctype, layer_idx) not in mean_cache:
                raise ValueError(f"mean_cache missing for ({ctype}, L{layer_idx})")
            mv = torch.as_tensor(mean_cache[(ctype, layer_idx)], dtype=torch.float32)
            handles.append(target.register_forward_hook(make_mean_hook(mv)))
        else:
            raise ValueError(f"unknown mode={mode}")
    return handles


def collect_l20(model, tokenizer, episodes, label_map,
                ablation_spec=None, mode="zero", mean_cache=None, collect_spec=None):
    """Forward all episodes (optionally with ablation hooks) and collect L20 last-token acts.

    If collect_spec is provided, also accumulate channel-wise mean of each listed
    component's output across all (sample, position) tokens, and return as
    {(ctype, layer): np.ndarray(hidden_dim,)}.
    """
    pb = PromptBuilder(tools=["search", "calculator"])
    layers = get_model_layers(model)
    device = next(model.parameters()).device

    abl_handles = (register_ablation(layers, ablation_spec, mode=mode, mean_cache=mean_cache)
                   if ablation_spec else [])
    cap = {}

    def l20_hook(m, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        cap['h'] = h[0, -1, :].detach().float().cpu().numpy()

    h20 = layers[MEASURE_LAYER].register_forward_hook(l20_hook)

    mean_store = {}
    mean_handles = []
    if collect_spec:
        for ctype, lidx in collect_spec:
            target = layers[lidx].self_attn if ctype == "attn" else layers[lidx].mlp
            mean_handles.append(target.register_forward_hook(
                make_mean_collector(mean_store, (ctype, lidx))))

    hidden_list, labels, sids_out, skipped = [], [], [], 0
    for i, ep in enumerate(episodes):
        sid = ep["sample_id"]
        if sid not in label_map:
            skipped += 1; continue
        steps = ep.get("steps", [])
        if not steps or steps[0].get("action") != "search" or not steps[0].get("observation"):
            skipped += 1; continue
        history = [{"action": "search",
                    "action_input": steps[0]["action_input"],
                    "observation":  steps[0]["observation"]}]
        messages = pb.build_full_prompt(ep["question"], history)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        try:
            with torch.no_grad():
                model(input_ids)
        except Exception as exc:
            print(f"  [{i+1}] ERROR {sid[:20]}: {exc}"); skipped += 1; continue
        hidden_list.append(cap['h'])
        labels.append(label_map[sid])
        sids_out.append(sid)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(episodes)}] collected={len(labels)} skipped={skipped}", flush=True)

    h20.remove()
    for h in abl_handles:
        h.remove()
    for h in mean_handles:
        h.remove()

    X = np.array(hidden_list, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"  Done: N={len(y)}, label=0: {(y==0).sum()}, label=1: {(y==1).sum()}, skipped={skipped}")
    means = {k: (v['sum'] / max(1, v['count'])).astype(np.float32)
             for k, v in mean_store.items()}
    return X, y, sids_out, means


def fit_probe(X_tr, y_tr, seed=42):
    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                             solver="lbfgs", random_state=seed)
    clf.fit(scaler.transform(X_tr), y_tr)
    return scaler, clf


def eval_probe(scaler, clf, X_te, y_te):
    Xs = scaler.transform(X_te)
    y_prob = clf.predict_proba(Xs)[:, 1]
    y_pred = clf.predict(Xs)
    return {
        "auroc": float(roc_auc_score(y_te, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred)),
        "n_test": int(len(y_te)),
        "n_test_label0": int((y_te == 0).sum()),
        "n_test_label1": int((y_te == 1).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--labels", default="results/phase1_probe/labels.jsonl")
    ap.add_argument("--baseline-trace", default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--output-dir", default=None,
                    help="defaults to results/top_k_ablation_probe_auroc_{mode}")
    ap.add_argument("--conditions", default="", help="comma-separated subset; default = all")
    ap.add_argument("--mode", default="mean", choices=["zero", "mean"],
                    help="ablation method (default=mean; zero is deprecated, OOD)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.output_dir is None:
        args.output_dir = f"results/top_k_ablation_probe_auroc_{args.mode}"
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    SEP = "=" * 65
    print(SEP); print("  Experiment C-ext: Top-k Component Ablation → L20 Probe AUROC"); print(SEP)

    # labels & episodes
    label_map = {json.loads(l)["sample_id"]: json.loads(l)["label"]
                 for l in open(args.labels)}
    print(f"Labels: {len(label_map)} samples, "
          f"label=0: {sum(1 for v in label_map.values() if v==0)}, "
          f"label=1: {sum(1 for v in label_map.values() if v==1)}")
    episodes = [json.loads(l) for l in open(args.baseline_trace)]
    print(f"Episodes: {len(episodes)}")

    # model
    print("\nLoading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")
    model.eval()
    print("Model loaded.")

    # Select conditions first (so we only collect means for components we'll use)
    all_names = list(ABLATION_SETS.keys())
    sel = [n for n in (args.conditions.split(",") if args.conditions else all_names)
           if n in ABLATION_SETS]
    union_spec = sorted({(c, l) for n in sel for c, l in ABLATION_SETS[n]})

    # ── Baseline (no hook) + channel-mean collection for all candidate components ─
    print(f"\n{SEP}\n[baseline_fresh] No ablation  (mode={args.mode})\n{SEP}")
    collect_spec = union_spec if args.mode == "mean" else None
    X_bl, y_bl, _, mean_cache_np = collect_l20(
        model, tokenizer, episodes, label_map,
        ablation_spec=None, collect_spec=collect_spec)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    tr, te = next(sss.split(X_bl, y_bl))
    scaler_bl, probe_bl = fit_probe(X_bl[tr], y_bl[tr], seed=args.seed)
    bl_metrics = eval_probe(scaler_bl, probe_bl, X_bl[te], y_bl[te])
    bl_metrics["n_train"] = int(len(tr))
    print(f"  Baseline AUROC = {bl_metrics['auroc']:.4f}  BalAcc = {bl_metrics['balanced_accuracy']:.4f}")
    if collect_spec:
        print(f"  Collected channel-means for {len(mean_cache_np)} components "
              f"(dim={next(iter(mean_cache_np.values())).shape[0] if mean_cache_np else 0})")

    results = {"baseline_fresh": {**bl_metrics, "fresh_auroc": bl_metrics["auroc"],
                                  "transferred_auroc": bl_metrics["auroc"],
                                  "mean_abs_delta_X": 0.0, "ablated": []}}

    print(f"\nConditions to run: {sel}")

    # ── Ablation conditions ──────────────────────────────────────────────────
    bl_norm = float(np.mean(np.linalg.norm(X_bl, axis=1)))
    for name in sel:
        spec = ABLATION_SETS[name]
        desc = ", ".join(f"{c}_L{l}" for c, l in spec)
        print(f"\n{SEP}\n[{name}] Ablate ({args.mode}): {desc}\n{SEP}")
        X, y, _, _ = collect_l20(model, tokenizer, episodes, label_map,
                                  ablation_spec=spec, mode=args.mode,
                                  mean_cache=mean_cache_np)

        # align N (in principle identical, but guard)
        if len(y) != len(y_bl) or not np.array_equal(y, y_bl):
            print(f"  WARN: y mismatch (baseline N={len(y_bl)}, cond N={len(y)}); using cond's own split")
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
            tr_c, te_c = next(sss2.split(X, y))
        else:
            tr_c, te_c = tr, te

        # Fresh probe (retrain on ablated train split)
        scaler_f, probe_f = fit_probe(X[tr_c], y[tr_c], seed=args.seed)
        m_fresh = eval_probe(scaler_f, probe_f, X[te_c], y[te_c])

        # Transferred probe (baseline probe on ablated test split)
        m_trans = eval_probe(scaler_bl, probe_bl, X[te_c], y[te_c])

        # Sanity
        delta = float(np.mean(np.linalg.norm(X - X_bl, axis=1)))
        rel = delta / bl_norm

        print(f"  fresh      AUROC={m_fresh['auroc']:.4f}  BalAcc={m_fresh['balanced_accuracy']:.4f}  Δ={m_fresh['auroc']-bl_metrics['auroc']:+.4f}")
        print(f"  transferred AUROC={m_trans['auroc']:.4f}  BalAcc={m_trans['balanced_accuracy']:.4f}  Δ={m_trans['auroc']-bl_metrics['auroc']:+.4f}")
        print(f"  [sanity] mean |ΔX| = {delta:.4f}  rel = {rel:.4f}")

        results[name] = {
            "ablated": [{"type": c, "layer": l} for c, l in spec],
            "fresh_auroc": m_fresh["auroc"],
            "fresh_balanced_accuracy": m_fresh["balanced_accuracy"],
            "transferred_auroc": m_trans["auroc"],
            "transferred_balanced_accuracy": m_trans["balanced_accuracy"],
            "n_train": int(len(tr_c)), "n_test": int(len(te_c)),
            "n_test_label0": int((y[te_c] == 0).sum()),
            "n_test_label1": int((y[te_c] == 1).sum()),
            "mean_abs_delta_X": delta,
            "relative_delta_X": rel,
        }

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{SEP}\n  SUMMARY (reference: baseline_fresh AUROC={bl_metrics['auroc']:.4f})\n{SEP}")
    print(f"  {'condition':<22} {'fresh':>8} {'transferred':>13} {'|ΔX|':>10}")
    for n, r in results.items():
        dx = r.get("mean_abs_delta_X", 0.0)
        print(f"  {n:<22} {r['fresh_auroc']:>8.4f} {r['transferred_auroc']:>13.4f} {dx:>10.2f}")

    out = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "measure_layer": MEASURE_LAYER,
        "seed": args.seed,
        "ablation_method": f"{args.mode}_all_positions",
        "probe": "LogisticRegression(class_weight=balanced, C=1, StandardScaler)",
        "split": f"StratifiedShuffleSplit(test_size=0.2, random_state={args.seed})",
        "N": int(len(y_bl)),
        "results": results,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

