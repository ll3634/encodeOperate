#!/usr/bin/env python3
"""
Behavioral Activation Patching
================================
Clean-to-corrupted activation patching with BEHAVIORAL readout (logits).

For each paired corruption sample (Group A: evidence swap):
  1. Clean fwd → cache component activations + get logits
  2. Corrupt fwd → get baseline logits
  3. Corrupt + patch component → get patched logits
  4. Measure: margin recovery = (patched_margin - corrupt_margin) / (clean_margin - corrupt_margin)

Components patched:
  - KV2 @ L18  (target circuit node)
  - KV0 @ L18  (negative control)
  - mlp @ L20  (downstream amplifier)
  - attn @ L18 (full attention, upper bound)

Margin = logit(action_token=2512) - logit(final_token=19357)
"""

import os, sys, json, argparse, random
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
from scipy.stats import wilcoxon, mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent))
from steering.hook_utils import get_model_layers
from scripts.paired_corruption_analysis import (
    select_samples, make_corrupted_obs, build_prompt,
)
from scripts.activation_patching import (
    capture_attn_pre_oproj, extract_component_outputs,
)

ACTION_TOKEN = 2512   # "Thought" / "Action"
FINAL_TOKEN  = 19357  # "Final"


def get_logit_margin(model, tokenizer, prompt):
    """Forward pass, return margin = logit(ACTION) - logit(FINAL) at last token."""
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :]  # (vocab,)
    return float(logits[ACTION_TOKEN] - logits[FINAL_TOKEN])


def _project_onto_subspace(vec, basis):
    """Project vec onto the subspace spanned by basis vectors.

    Args:
        vec: 1D array (D,)
        basis: list of 1D arrays, each (D,). Must be orthonormal.

    Returns:
        projection of vec onto the subspace
    """
    proj = np.zeros_like(vec)
    for b in basis:
        proj += np.dot(vec, b) * b
    return proj


def _orthonormalize(directions):
    """Gram-Schmidt orthonormalization of a list of direction vectors."""
    basis = []
    for d in directions:
        v = d.copy().astype(np.float64)
        for b in basis:
            v -= np.dot(v, b) * b
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            basis.append(v / norm)
    return [b.astype(np.float32) for b in basis]


def patched_logit_margin_subspace(model, tokenizer, prompt_corrupt,
                                   clean_cache, corrupt_cache,
                                   comp_type, comp_layer, basis,
                                   mode="parallel"):
    """Patch the subspace-aligned (or orthogonal) component of the difference.

    Args:
        basis: list of orthonormal direction vectors defining the subspace
        mode: 'parallel' = patch only subspace component; 'orthogonal' = everything else
    """
    layers = get_model_layers(model)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt_corrupt, return_tensors="pt").to(device)

    clean_np = clean_cache[(comp_type, comp_layer)]
    corrupt_np = corrupt_cache[(comp_type, comp_layer)]
    delta = (clean_np - corrupt_np).astype(np.float32)
    delta_proj = _project_onto_subspace(delta, basis)

    if mode == "parallel":
        patched_np = corrupt_np + delta_proj
    else:  # orthogonal
        patched_np = corrupt_np + (delta - delta_proj)

    patched_vec = torch.tensor(patched_np, dtype=torch.bfloat16, device=device)
    target = (layers[comp_layer].self_attn if comp_type == 'attn'
              else layers[comp_layer].mlp)

    def patch_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h[0, -1, :] = patched_vec
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h

    handle = target.register_forward_hook(patch_hook)
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :]
    handle.remove()
    return float(logits[ACTION_TOKEN] - logits[FINAL_TOKEN])


def patched_logit_margin_kv_group(model, tokenizer, prompt_corrupt,
                                   clean_pre_oproj, kv_group,
                                   patch_layer_idx,
                                   n_heads=28, head_dim=128, n_kv_heads=4):
    """Patch KV group's pre-o_proj output at last token, return logit margin."""
    layers = get_model_layers(model)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt_corrupt, return_tensors="pt").to(device)

    heads_per_kv = n_heads // n_kv_heads
    s = kv_group * heads_per_kv * head_dim
    e = s + heads_per_kv * head_dim
    clean_slice = torch.tensor(clean_pre_oproj[s:e],
                                dtype=torch.bfloat16, device=device)

    def patch_hook(module, inp, out):
        x = inp[0]
        x_mod = x.clone()
        x_mod[0, -1, s:e] = clean_slice
        patched_out = module.weight @ x_mod[0, -1, :]
        if module.bias is not None:
            patched_out = patched_out + module.bias
        out_mod = out.clone()
        out_mod[0, -1, :] = patched_out
        return out_mod

    handle = layers[patch_layer_idx].self_attn.o_proj.register_forward_hook(patch_hook)
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :]
    handle.remove()
    return float(logits[ACTION_TOKEN] - logits[FINAL_TOKEN])


def patched_logit_margin_component(model, tokenizer, prompt_corrupt,
                                    clean_cache, comp_type, comp_layer):
    """Patch a full component (attn or mlp) output at last token, return logit margin."""
    layers = get_model_layers(model)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt_corrupt, return_tensors="pt").to(device)

    clean_vec = torch.tensor(clean_cache[(comp_type, comp_layer)],
                              dtype=torch.bfloat16, device=device)
    target = (layers[comp_layer].self_attn if comp_type == 'attn'
              else layers[comp_layer].mlp)

    def patch_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h[0, -1, :] = clean_vec
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h

    handle = target.register_forward_hook(patch_hook)
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :]
    handle.remove()
    return float(logits[ACTION_TOKEN] - logits[FINAL_TOKEN])


def run_experiment(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    n_layers = len(get_model_layers(model))
    print(f"  {n_layers} layers", flush=True)

    # Load interpretable directions for comparison
    act_data = np.load(args.action_dir)
    action_dir = act_data["decision_direction"].astype(np.float32)
    action_dir /= np.linalg.norm(action_dir) + 1e-12
    evi_data = np.load(args.evidence_dir)
    evidence_dir = evi_data["decision_direction"].astype(np.float32)
    evidence_dir /= np.linalg.norm(evidence_dir) + 1e-12
    print(f"  cos(evidence, action) = {float(np.dot(evidence_dir, action_dir)):.4f}")

    samples = select_samples(args.baseline_trace, args.hotpotqa_data,
                              n=args.n_samples, seed=args.seed)
    print(f"Using {len(samples)} samples", flush=True)

    PATCH_LAYER = 18
    K_VALUES = [1, 2, 5, 10, 20, 50, 100, 200, 500]

    # ── PHASE 1: Collect difference vectors + margins ─────────────────────────
    print("\n=== PHASE 1: Collecting difference vectors ===", flush=True)
    all_diffs_attn = []  # (N, D)
    all_diffs_mlp = []   # (N, D)
    sample_data = []     # margins + caches per sample

    for i, sample in enumerate(samples):
        rng_copy = random.Random(args.seed)
        for j in range(i):
            make_corrupted_obs(samples[j], "A", rng_copy)
        clean_obs, corrupted_obs = make_corrupted_obs(sample, "A", rng_copy)

        prompt_clean = build_prompt(tokenizer, sample["question"],
                                     sample["step0_query"], clean_obs)
        prompt_corrupt = build_prompt(tokenizer, sample["question"],
                                       sample["step0_query"], corrupted_obs)

        margin_clean = get_logit_margin(model, tokenizer, prompt_clean)
        margin_corrupt = get_logit_margin(model, tokenizer, prompt_corrupt)

        clean_cache = extract_component_outputs(
            model, tokenizer, prompt_clean, n_layers=n_layers)
        corrupt_cache = extract_component_outputs(
            model, tokenizer, prompt_corrupt, n_layers=n_layers)

        diff_attn = (clean_cache[('attn', PATCH_LAYER)]
                     - corrupt_cache[('attn', PATCH_LAYER)]).astype(np.float32)
        diff_mlp = (clean_cache[('mlp', 20)]
                    - corrupt_cache[('mlp', 20)]).astype(np.float32)

        all_diffs_attn.append(diff_attn)
        all_diffs_mlp.append(diff_mlp)
        sample_data.append({
            "sample_id": sample["sample_id"],
            "margin_clean": margin_clean,
            "margin_corrupt": margin_corrupt,
            "delta": margin_clean - margin_corrupt,
            "prompt_corrupt": prompt_corrupt,
            "clean_cache": clean_cache,
            "corrupt_cache": corrupt_cache,
        })

        if (i + 1) % 10 == 0:
            print(f"  Phase 1: [{i+1}/{len(samples)}]", flush=True)

    # ── PHASE 2: PCA on difference vectors ────────────────────────────────────
    print("\n=== PHASE 2: Computing PCA ===", flush=True)
    diff_matrix_attn = np.stack(all_diffs_attn, axis=0)  # (N, D)
    diff_matrix_mlp = np.stack(all_diffs_mlp, axis=0)

    # Center and SVD
    mean_attn = diff_matrix_attn.mean(axis=0)
    mean_mlp = diff_matrix_mlp.mean(axis=0)
    U_attn, S_attn, Vt_attn = np.linalg.svd(
        diff_matrix_attn - mean_attn, full_matrices=False)
    U_mlp, S_mlp, Vt_mlp = np.linalg.svd(
        diff_matrix_mlp - mean_mlp, full_matrices=False)

    # Variance explained
    var_attn = S_attn ** 2 / (S_attn ** 2).sum()
    var_mlp = S_mlp ** 2 / (S_mlp ** 2).sum()
    print(f"  attn_L18 top-k cumulative variance:")
    for k in K_VALUES:
        if k <= len(var_attn):
            print(f"    k={k:>4}: {var_attn[:k].sum():.4f}")
    print(f"  mlp_L20 top-k cumulative variance:")
    for k in K_VALUES:
        if k <= len(var_mlp):
            print(f"    k={k:>4}: {var_mlp[:k].sum():.4f}")

    # Check alignment of PCA directions with interpretable directions
    print(f"\n  PCA-direction alignment with action_dir / evidence_dir:")
    for name, Vt in [("attn_L18", Vt_attn), ("mlp_L20", Vt_mlp)]:
        cos_act = [abs(float(np.dot(Vt[j], action_dir))) for j in range(min(10, len(Vt)))]
        cos_evi = [abs(float(np.dot(Vt[j], evidence_dir))) for j in range(min(10, len(Vt)))]
        print(f"    {name} top-10 PCs |cos| with action_dir: "
              f"{', '.join(f'{c:.3f}' for c in cos_act)}")
        print(f"    {name} top-10 PCs |cos| with evidence_dir: "
              f"{', '.join(f'{c:.3f}' for c in cos_evi)}")

    # ── PHASE 3: PCA-subspace patching sweep ──────────────────────────────────
    print("\n=== PHASE 3: PCA subspace patching ===", flush=True)

    # Build configs: full + PCA-k for each component
    configs_pca = ["attn_L18_full", "mlp_L20_full"]
    for k in K_VALUES:
        configs_pca.append(f"attn_L18_pca{k}")
        configs_pca.append(f"mlp_L20_pca{k}")
    results = {c: [] for c in configs_pca}

    for i, sd in enumerate(sample_data):
        prompt_corrupt = sd["prompt_corrupt"]
        clean_cache = sd["clean_cache"]
        corrupt_cache = sd["corrupt_cache"]

        # Full patches (recompute for consistency)
        m_attn_full = patched_logit_margin_component(
            model, tokenizer, prompt_corrupt, clean_cache, 'attn', PATCH_LAYER)
        m_mlp_full = patched_logit_margin_component(
            model, tokenizer, prompt_corrupt, clean_cache, 'mlp', 20)

        row_base = {
            "sample_id": sd["sample_id"],
            "margin_clean": sd["margin_clean"],
            "margin_corrupt": sd["margin_corrupt"],
            "delta": sd["delta"],
        }

        def add_result(cfg, m_patched):
            recovery = ((m_patched - sd["margin_corrupt"]) / sd["delta"]
                        if abs(sd["delta"]) > 1e-6 else 0.0)
            results[cfg].append({**row_base, "margin_patched": m_patched,
                                  "recovery": recovery})

        add_result("attn_L18_full", m_attn_full)
        add_result("mlp_L20_full", m_mlp_full)

        # PCA-k patches
        for k in K_VALUES:
            # attn_L18
            basis_attn_k = [Vt_attn[j].astype(np.float32) for j in range(min(k, len(Vt_attn)))]
            m = patched_logit_margin_subspace(
                model, tokenizer, prompt_corrupt, clean_cache, corrupt_cache,
                'attn', PATCH_LAYER, basis_attn_k, mode="parallel")
            add_result(f"attn_L18_pca{k}", m)

            # mlp_L20
            basis_mlp_k = [Vt_mlp[j].astype(np.float32) for j in range(min(k, len(Vt_mlp)))]
            m = patched_logit_margin_subspace(
                model, tokenizer, prompt_corrupt, clean_cache, corrupt_cache,
                'mlp', 20, basis_mlp_k, mode="parallel")
            add_result(f"mlp_L20_pca{k}", m)

        if (i + 1) % 5 == 0:
            print(f"  Phase 3: [{i+1}/{len(sample_data)}]", flush=True)

    # ── Statistics ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PCA SUBSPACE PATCHING RESULTS")
    print(f"  Recovery = (patched - corrupt) / (clean - corrupt)")
    print(f"{'='*70}")
    print(f"{'Config':<25} {'mean_recov':>11} {'median_recov':>13} "
          f"{'n_toward_clean':>15} {'n':>5} {'p(>0)':>10}")

    summary = {}
    for cfg in configs_pca:
        rows = results[cfg]
        recoveries = np.array([r["recovery"] for r in rows])
        n_toward = sum(1 for r in rows
                       if abs(r["delta"]) > 1e-6 and
                       np.sign(r["margin_patched"] - r["margin_corrupt"]) == np.sign(r["delta"]))
        try:
            w, p_val = wilcoxon(recoveries, alternative='greater')
        except Exception:
            w, p_val = 0, 1.0
        summary[cfg] = {
            "mean_recovery": float(recoveries.mean()),
            "median_recovery": float(np.median(recoveries)),
            "std_recovery": float(recoveries.std()),
            "n_toward_clean": int(n_toward),
            "n": len(rows),
            "wilcoxon_p": float(p_val),
        }
        s = summary[cfg]
        print(f"  {cfg:<25} {s['mean_recovery']:>+11.4f} {s['median_recovery']:>+13.4f} "
              f"{s['n_toward_clean']:>15} {s['n']:>5} {s['wilcoxon_p']:>10.4f}")

    # Recovery curve: fraction of full recovery captured by PCA-k
    print(f"\n  Recovery as fraction of full:")
    for comp, comp_full in [("attn_L18", "attn_L18_full"), ("mlp_L20", "mlp_L20_full")]:
        full_med = summary[comp_full]["median_recovery"]
        print(f"    {comp} (full median={full_med:.4f}):")
        for k in K_VALUES:
            cfg_k = f"{comp}_pca{k}"
            if cfg_k in summary:
                med_k = summary[cfg_k]["median_recovery"]
                frac = med_k / full_med if abs(full_med) > 1e-6 else 0.0
                print(f"      k={k:>4}: median={med_k:+.4f}  "
                      f"({frac:>6.1%} of full)")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    pca_info = {
        "var_explained_attn": var_attn[:50].tolist(),
        "var_explained_mlp": var_mlp[:50].tolist(),
    }
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": args.model, "n_samples": len(samples),
            "seed": args.seed, "k_values": K_VALUES,
            "action_token": ACTION_TOKEN, "final_token": FINAL_TOKEN,
        },
        "pca_info": pca_info,
        "summary": summary,
        "per_sample": {cfg: results[cfg] for cfg in configs_pca},
    }
    out_path = os.path.join(args.output_dir, "pca_patching_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--baseline-trace",
                    default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--hotpotqa-data",
                    default="data/hotpotqa/hotpot_dev_distractor_v1.json")
    ap.add_argument("--output-dir", default="results/behavioral_patching_directional")
    ap.add_argument("--action-dir",
                    default="steering/directions/direction_search_v3_layer20.npz")
    ap.add_argument("--evidence-dir",
                    default="results/phase1_probe/probe_direction_l20.npz")
    ap.add_argument("--n-samples", type=int, default=9999,
                    help="Max samples; will use all available if > candidates")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

