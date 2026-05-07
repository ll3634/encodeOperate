#!/usr/bin/env python3
"""
Activation Patching: Evidence→Action Pathway Identification
============================================================
Two phases:
  Phase A – Residual Decomposition: decompose h_clean-h_corrupt into
            per-component (attn_l, mlp_l) contributions projected onto
            action_dir. Exact linear decomposition. 2 fwd passes / sample.
  Phase B – Activation Patching: for top-K components from Phase A,
            patch clean component output into corrupted run and measure
            recovery of action-direction shift. Captures non-linear effects.

Uses the same Group A (evidence corruption) samples from paired_corruption.

Output → results/paired_corruption/activation_patching_results.json
"""

import os, sys, json, argparse, random
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
from scipy.stats import mannwhitneyu, wilcoxon

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder
from steering.hook_utils import get_model_layers
from scripts.paired_corruption_analysis import (
    select_samples, make_corrupted_obs, build_prompt,
)

# ── Phase A: Residual Decomposition ─────────────────────────────────────────

def extract_component_outputs(model, tokenizer, prompt, n_layers=28):
    """Run forward pass, capture each attn and mlp output at last token.

    Returns dict: {('attn', l): np.array, ('mlp', l): np.array, 'residual_L20': np.array}
    """
    layers = get_model_layers(model)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    captured = {}
    handles = []

    for l in range(n_layers):
        layer = layers[l]

        def make_attn_hook(layer_idx):
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[('attn', layer_idx)] = h[0, -1, :].detach().float().cpu().numpy()
            return hook_fn

        def make_mlp_hook(layer_idx):
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[('mlp', layer_idx)] = h[0, -1, :].detach().float().cpu().numpy()
            return hook_fn

        handles.append(layer.self_attn.register_forward_hook(make_attn_hook(l)))
        handles.append(layer.mlp.register_forward_hook(make_mlp_hook(l)))

    # Also capture L20 residual stream (layer output) for verification
    def l20_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured['residual_L20'] = h[0, -1, :].detach().float().cpu().numpy()
    handles.append(layers[20].register_forward_hook(l20_hook))

    with torch.no_grad():
        model(input_ids)

    for h in handles:
        h.remove()

    return captured


def decompose_residual(clean_components, corrupt_components, action_dir,
                       n_layers=28, measure_layer=20):
    """Decompose h_clean - h_corrupt into per-component action projections.

    Returns list of (component_name, projection_value) sorted by |projection|.
    """
    contributions = []
    total = 0.0

    for l in range(measure_layer + 1):  # layers 0..20 contribute to L20 residual
        for comp_type in ['attn', 'mlp']:
            key = (comp_type, l)
            if key not in clean_components or key not in corrupt_components:
                continue
            delta = clean_components[key] - corrupt_components[key]
            proj = float(np.dot(delta, action_dir))
            contributions.append((f"{comp_type}_L{l}", proj))
            total += proj

    # Verification: total should ≈ action projection of full residual delta
    if 'residual_L20' in clean_components and 'residual_L20' in corrupt_components:
        full_delta = clean_components['residual_L20'] - corrupt_components['residual_L20']
        full_proj = float(np.dot(full_delta, action_dir))
    else:
        full_proj = None

    return contributions, total, full_proj


# ── Phase C helper: MLP I/O Routing ─────────────────────────────────────────

def extract_mlp_io(model, tokenizer, prompt, target_layers, n_layers=28):
    """Extract MLP input (post_attention_layernorm output) and MLP output
    at last token for specified layers.

    Returns dict: {layer_idx: {'mlp_input': np.array, 'mlp_output': np.array}}
    """
    layers = get_model_layers(model)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    captured = {}
    handles = []

    for l in target_layers:
        layer = layers[l]
        captured[l] = {}

        def make_ln_hook(layer_idx):
            """Hook post_attention_layernorm to capture MLP input."""
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[layer_idx]['mlp_input'] = h[0, -1, :].detach().float().cpu().numpy()
            return hook_fn

        def make_mlp_hook(layer_idx):
            """Hook MLP to capture MLP output (raw, before residual add)."""
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[layer_idx]['mlp_output'] = h[0, -1, :].detach().float().cpu().numpy()
            return hook_fn

        handles.append(layer.post_attention_layernorm.register_forward_hook(make_ln_hook(l)))
        handles.append(layer.mlp.register_forward_hook(make_mlp_hook(l)))

    with torch.no_grad():
        model(input_ids)

    for h in handles:
        h.remove()

    return captured


# ── Phase E helper: Cross-position attention patching ─────────────────────

def find_divergent_range(ids_a, ids_b):
    """Find the token index range that differs between two token sequences.
    Returns (start, end_a, end_b) where ids_a[start:end_a] != ids_b[start:end_b].
    Tokens before start and after end are identical (common prefix/suffix).
    """
    min_len = min(len(ids_a), len(ids_b))
    # Common prefix
    prefix = 0
    while prefix < min_len and ids_a[prefix] == ids_b[prefix]:
        prefix += 1
    # Common suffix (search from end)
    suffix_a, suffix_b = len(ids_a), len(ids_b)
    while suffix_a > prefix and suffix_b > prefix and ids_a[suffix_a-1] == ids_b[suffix_b-1]:
        suffix_a -= 1
        suffix_b -= 1
    return prefix, suffix_a, suffix_b


def capture_ln_output(model, tokenizer, prompt, target_layer_idx):
    """Run forward pass, capture input_layernorm output (= attn input) at ALL positions
    for the target layer. Returns tensor on GPU (seq_len, d_model)."""
    layers = get_model_layers(model)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    captured = {}

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured['h'] = h[0].detach().clone()  # (seq_len, d_model)

    handle = layers[target_layer_idx].input_layernorm.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(input_ids)
    handle.remove()
    return captured['h'], input_ids[0]  # also return token ids for alignment


def cross_position_patch_and_measure(model, tokenizer, prompt_corrupt,
                                      clean_ln_output, obs_start_clean, obs_end_clean,
                                      obs_start_corrupt, obs_end_corrupt,
                                      patch_layer_idx, action_dir, measure_layer=20):
    """Run corrupt prompt, but at patch_layer_idx's attention input (input_layernorm output),
    replace observation-position hidden states with clean values.
    Measure action projection at last token of L{measure_layer} residual.
    """
    layers = get_model_layers(model)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt_corrupt, return_tensors="pt").to(device)

    n_patch = min(obs_end_corrupt - obs_start_corrupt, obs_end_clean - obs_start_clean)
    clean_slice = clean_ln_output[obs_start_clean:obs_start_clean + n_patch]  # (n_patch, d)

    captured = {}
    handles = []

    # Patch hook on input_layernorm output: replace observation positions
    def patch_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h[0, obs_start_corrupt:obs_start_corrupt + n_patch, :] = clean_slice.to(h.dtype)
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h

    handles.append(layers[patch_layer_idx].input_layernorm.register_forward_hook(patch_hook))

    # Measure hook at L{measure_layer}
    def measure_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured['h'] = h[0, -1, :].detach().float().cpu().numpy()
    handles.append(layers[measure_layer].register_forward_hook(measure_hook))

    with torch.no_grad():
        model(input_ids)

    for h in handles:
        h.remove()

    return float(np.dot(captured['h'], action_dir))


# ── Phase F helper: Per-head decomposition of attention output ──────────────

def capture_attn_pre_oproj(model, tokenizer, prompt, target_layer_idx):
    """Capture the attention output BEFORE o_proj at the last token position.
    Returns: numpy array of shape (n_heads * head_dim,) = (3584,)
    """
    layers = get_model_layers(model)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    captured = {}

    # Hook on o_proj to capture its input (= pre-projection attn output)
    def hook_fn(module, inp, out):
        # inp is a tuple; inp[0] has shape (batch, seq, n_heads*head_dim)
        captured['pre_oproj'] = inp[0][0, -1, :].detach().float().cpu().numpy()

    handle = layers[target_layer_idx].self_attn.o_proj.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(input_ids)
    handle.remove()
    return captured['pre_oproj']


def decompose_attn_by_head(pre_oproj_clean, pre_oproj_corrupt, o_proj_weight,
                           action_dir, n_heads=28, head_dim=128):
    """Decompose the attn output difference into per-head contributions to action_dir.

    pre_oproj_clean/corrupt: (n_heads*head_dim,) vectors before o_proj
    o_proj_weight: (d_model, n_heads*head_dim) weight matrix
    action_dir: (d_model,) unit vector

    Returns list of dicts with per-head attribution.
    """
    diff = pre_oproj_clean - pre_oproj_corrupt  # (n_heads*head_dim,)
    results = []
    total_proj = 0.0
    for h in range(n_heads):
        s, e = h * head_dim, (h + 1) * head_dim
        head_diff = diff[s:e]  # (head_dim,)
        # head's contribution to residual stream: W_o[:, s:e] @ head_diff
        head_contribution = o_proj_weight[:, s:e] @ head_diff  # (d_model,)
        proj = float(np.dot(head_contribution, action_dir))
        total_proj += proj
        results.append({
            "head": h,
            "action_proj": proj,
            "abs_action_proj": abs(proj),
            "head_diff_norm": float(np.linalg.norm(head_diff)),
        })
    # Also compute total for verification
    full_contribution = o_proj_weight @ diff
    full_proj = float(np.dot(full_contribution, action_dir))
    return results, full_proj, total_proj


def per_head_output_patch_and_measure(model, tokenizer, prompt_corrupt,
                                       clean_pre_oproj, head_idx,
                                       patch_layer_idx, action_dir,
                                       measure_layer=20, n_heads=28, head_dim=128):
    """Patch a single head's pre-o_proj output at the last token from clean into corrupt.
    Measure action projection recovery at L{measure_layer} residual.
    """
    layers = get_model_layers(model)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt_corrupt, return_tensors="pt").to(device)

    s, e = head_idx * head_dim, (head_idx + 1) * head_dim
    clean_head = torch.tensor(clean_pre_oproj[s:e], dtype=torch.bfloat16, device=device)

    captured = {}
    handles = []

    # Patch the o_proj input at last token for this head
    def patch_oproj_input(module, inp, out):
        x = inp[0]  # (batch, seq, n_heads*head_dim)
        x[0, -1, s:e] = clean_head
        # Recompute output with patched input
        patched_out = module.weight @ x[0, -1, :] + (module.bias if module.bias is not None else 0)
        out[0, -1, :] = patched_out
        return out

    handles.append(layers[patch_layer_idx].self_attn.o_proj.register_forward_hook(patch_oproj_input))

    # Measure hook
    def measure_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured['h'] = h[0, -1, :].detach().float().cpu().numpy()
    handles.append(layers[measure_layer].register_forward_hook(measure_hook))

    with torch.no_grad():
        model(input_ids)

    for h in handles:
        h.remove()

    return float(np.dot(captured['h'], action_dir))


def per_kv_group_output_patch_and_measure(model, tokenizer, prompt_corrupt,
                                          clean_pre_oproj, kv_group,
                                          patch_layer_idx, action_dir,
                                          measure_layer=20, n_heads=28,
                                          head_dim=128, n_kv_heads=4):
    """Patch all Q-heads in a KV group's pre-o_proj output at last token.
    kv_group: 0..3, each covering 7 Q-heads.
    """
    layers = get_model_layers(model)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt_corrupt, return_tensors="pt").to(device)

    heads_per_kv = n_heads // n_kv_heads  # 7
    h_start = kv_group * heads_per_kv
    h_end = h_start + heads_per_kv
    s = h_start * head_dim
    e = h_end * head_dim
    clean_slice = torch.tensor(clean_pre_oproj[s:e],
                               dtype=torch.bfloat16, device=device)

    captured = {}
    handles = []

    def patch_hook(module, inp, out):
        x = inp[0]  # (batch, seq, n_heads*head_dim)
        x[0, -1, s:e] = clean_slice
        patched_out = module.weight @ x[0, -1, :]
        out[0, -1, :] = patched_out
        return out

    handles.append(layers[patch_layer_idx].self_attn.o_proj.register_forward_hook(patch_hook))

    def measure_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured['h'] = h[0, -1, :].detach().float().cpu().numpy()
    handles.append(layers[measure_layer].register_forward_hook(measure_hook))

    with torch.no_grad():
        model(input_ids)

    for h in handles:
        h.remove()

    return float(np.dot(captured['h'], action_dir))


def capture_attention_weights(model, tokenizer, prompt, target_layer_idx):
    """Capture attention weights at target_layer by temporarily using eager attention.
    Returns: attn_weights at last query position, shape (n_heads, seq_len) as numpy.
    Also returns token_ids for alignment.
    """
    layers = get_model_layers(model)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    captured = {}
    handles = []

    # Hook q_proj and k_proj to get their outputs, then compute weights manually
    def hook_q(module, inp, out):
        captured['q'] = out.detach()  # (batch, seq, n_heads*head_dim)

    def hook_k(module, inp, out):
        captured['k'] = out.detach()  # (batch, seq, n_kv_heads*head_dim)

    attn_mod = layers[target_layer_idx].self_attn
    handles.append(attn_mod.q_proj.register_forward_hook(hook_q))
    handles.append(attn_mod.k_proj.register_forward_hook(hook_k))

    with torch.no_grad():
        model(input_ids)

    for h in handles:
        h.remove()

    # Now compute attention weights manually from Q and K
    # Q: (1, seq, 28*128) -> (1, 28, seq, 128)
    # K: (1, seq, 4*128) -> (1, 4, seq, 128) -> repeat to (1, 28, seq, 128)
    n_heads = 28
    n_kv_heads = 4
    head_dim = 128
    heads_per_kv = n_heads // n_kv_heads  # 7
    seq_len = input_ids.shape[1]

    q = captured['q'].float().view(1, seq_len, n_heads, head_dim).transpose(1, 2)
    k = captured['k'].float().view(1, seq_len, n_kv_heads, head_dim).transpose(1, 2)

    # Note: these are PRE-RoPE. Attention weights computed without RoPE
    # will be approximate but sufficient for identifying which positions
    # each head attends to (relative patterns).
    # For exact weights we'd need post-RoPE Q,K, but the position-level
    # attention mass distribution is still informative.

    # Repeat K for GQA: (1, 4, seq, 128) -> (1, 28, seq, 128)
    k = k.repeat_interleave(heads_per_kv, dim=1)

    # Attention scores: Q @ K^T / sqrt(d)
    # Only compute for last query position
    q_last = q[:, :, -1:, :]  # (1, 28, 1, 128)
    scores = torch.matmul(q_last, k.transpose(-2, -1)) / (head_dim ** 0.5)
    # (1, 28, 1, seq_len)

    # Apply causal mask (last position can attend to all)
    # No masking needed since last position sees everything

    attn_weights = torch.softmax(scores, dim=-1)  # (1, 28, 1, seq_len)
    attn_weights = attn_weights[0, :, 0, :].cpu().numpy()  # (28, seq_len)

    return attn_weights, input_ids[0].cpu().tolist()


# ── Phase B: Activation Patching ────────────────────────────────────────────

def patch_and_measure(model, tokenizer, prompt_corrupt, clean_cache,
                      patch_component, patch_layer, action_dir,
                      measure_layer=20):
    """Run corrupted prompt but patch one component from clean cache.

    patch_component: 'attn' or 'mlp'
    Returns: action projection of L{measure_layer} residual after patching.
    """
    layers = get_model_layers(model)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt_corrupt, return_tensors="pt").to(device)

    captured = {}
    handles = []

    # Patch hook: replace the specified component's output with clean version
    clean_vec = torch.tensor(
        clean_cache[(patch_component, patch_layer)],
        dtype=torch.bfloat16, device=device
    )

    target_module = (layers[patch_layer].self_attn if patch_component == 'attn'
                     else layers[patch_layer].mlp)

    def patch_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h[0, -1, :] = clean_vec
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h

    handles.append(target_module.register_forward_hook(patch_hook))

    # Measure hook at L{measure_layer}
    def measure_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured['h'] = h[0, -1, :].detach().float().cpu().numpy()
    handles.append(layers[measure_layer].register_forward_hook(measure_hook))

    with torch.no_grad():
        model(input_ids)

    for h in handles:
        h.remove()

    return float(np.dot(captured['h'], action_dir))


# ── Main Experiment ─────────────────────────────────────────────────────────

def run_experiment(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    n_layers = len(get_model_layers(model))
    print(f"  {n_layers} layers")

    # Load action direction
    act_data = np.load(args.action_dir)
    action_dir = act_data["decision_direction"].astype(np.float32)
    action_dir /= np.linalg.norm(action_dir) + 1e-12

    # Load evidence direction (for Phase C)
    ev_data = np.load(args.evidence_dir)
    evidence_dir = ev_data["decision_direction"].astype(np.float32)
    evidence_dir /= np.linalg.norm(evidence_dir) + 1e-12
    cos_ev_act = float(np.dot(evidence_dir, action_dir))
    print(f"  cos(evidence_dir, action_dir) = {cos_ev_act:.4f}")

    # Select Group A samples
    samples = select_samples(args.baseline_trace, args.hotpotqa_data,
                             n=args.n_samples, seed=args.seed)
    rng = random.Random(args.seed)
    print(f"  {len(samples)} samples selected")

    # ── Phase A: Residual Decomposition ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE A: RESIDUAL DECOMPOSITION")
    print("=" * 70)

    # Accumulate per-component contributions across samples
    comp_projections = defaultdict(list)  # comp_name -> [proj_per_sample]
    verification_errors = []
    per_sample_data = []
    # For corrected Phase C (residual-space routing, no LayerNorm confound)
    routing_residual_rows = []

    for i, sample in enumerate(samples):
        clean_obs, corrupted_obs = make_corrupted_obs(sample, "A", rng)
        prompt_clean = build_prompt(tokenizer, sample["question"],
                                    sample["step0_query"], clean_obs)
        prompt_corrupt = build_prompt(tokenizer, sample["question"],
                                      sample["step0_query"], corrupted_obs)

        clean_comp = extract_component_outputs(model, tokenizer, prompt_clean,
                                                n_layers=n_layers)
        corrupt_comp = extract_component_outputs(model, tokenizer, prompt_corrupt,
                                                  n_layers=n_layers)

        contribs, decomp_total, full_proj = decompose_residual(
            clean_comp, corrupt_comp, action_dir,
            n_layers=n_layers, measure_layer=args.measure_layer)

        sample_row = {"sample_id": sample["sample_id"]}
        for name, proj in contribs:
            comp_projections[name].append(proj)
            sample_row[name] = proj

        if full_proj is not None:
            err = abs(decomp_total - full_proj)
            verification_errors.append(err)
            sample_row["decomp_total"] = decomp_total
            sample_row["full_proj"] = full_proj
            sample_row["verification_err"] = err

        per_sample_data.append(sample_row)

        # ── Corrected routing: both Δin and Δout in residual-stream space ──
        # residual_pre_mlp_L20 = residual_L20 - mlp_L20_output
        # This avoids the LayerNorm-space confound of the original Phase C.
        ml = args.measure_layer
        if ('residual_L20' in clean_comp and ('mlp', ml) in clean_comp and
                'residual_L20' in corrupt_comp and ('mlp', ml) in corrupt_comp):
            pre_mlp_clean = (clean_comp['residual_L20']
                             - clean_comp[('mlp', ml)])
            pre_mlp_corrupt = (corrupt_comp['residual_L20']
                               - corrupt_comp[('mlp', ml)])
            d_in = pre_mlp_clean - pre_mlp_corrupt           # residual space
            d_out = (clean_comp[('mlp', ml)]
                     - corrupt_comp[('mlp', ml)])             # residual space
            routing_residual_rows.append({
                "sample_id": sample["sample_id"],
                "din_action":   float(np.dot(d_in, action_dir)),
                "din_evidence": float(np.dot(d_in, evidence_dir)),
                "dout_action":  float(np.dot(d_out, action_dir)),
                "dout_evidence": float(np.dot(d_out, evidence_dir)),
                "din_norm":  float(np.linalg.norm(d_in)),
                "dout_norm": float(np.linalg.norm(d_out)),
            })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}]", flush=True)

    # Advance RNG for groups B and C to keep determinism with paired_corruption
    for _ in samples:
        make_corrupted_obs(sample, "B", rng)

    # ── Summarize Phase A ────────────────────────────────────────────────────
    print(f"\nVerification: mean decomposition error = "
          f"{np.mean(verification_errors):.6f}")

    # Sort by mean |projection|
    comp_summary = []
    for name, projs in comp_projections.items():
        arr = np.array(projs)
        comp_summary.append({
            "component": name,
            "mean_proj": float(arr.mean()),
            "mean_abs_proj": float(np.abs(arr).mean()),
            "std_proj": float(arr.std()),
            "frac_positive": float((arr > 0).mean()),
        })

    comp_summary.sort(key=lambda x: x["mean_abs_proj"], reverse=True)

    print("\nTop 15 components by mean |projection onto action_dir|:")
    print(f"{'Component':<12} {'mean_proj':>10} {'|mean_proj|':>12} "
          f"{'std':>8} {'frac>0':>8}")
    for c in comp_summary[:15]:
        print(f"  {c['component']:<12} {c['mean_proj']:>+10.4f} "
              f"{c['mean_abs_proj']:>12.4f} {c['std_proj']:>8.4f} "
              f"{c['frac_positive']:>8.1%}")

    # Total across all components
    total_proj = sum(c["mean_proj"] for c in comp_summary)
    print(f"\n  Sum of all mean_proj: {total_proj:+.4f}")

    # ── Phase C (corrected): Residual-space MLP routing at measure_layer ─────
    routing_corrected_summary = {}
    if routing_residual_rows:
        din_act  = np.array([r["din_action"]   for r in routing_residual_rows])
        din_ev   = np.array([r["din_evidence"] for r in routing_residual_rows])
        dout_act = np.array([r["dout_action"]  for r in routing_residual_rows])
        dout_ev  = np.array([r["dout_evidence"] for r in routing_residual_rows])
        din_norm = np.array([r["din_norm"]  for r in routing_residual_rows])
        dout_norm = np.array([r["dout_norm"] for r in routing_residual_rows])

        # Routing gain in residual space: no LayerNorm confound
        abs_din_act = np.abs(din_act)
        valid = abs_din_act > 0.01
        gains = np.where(valid, np.abs(dout_act) / abs_din_act, np.nan)
        norm_ratio = dout_norm / (din_norm + 1e-8)

        routing_corrected_summary = {
            "measure_layer": args.measure_layer,
            "note": ("Δin = residual_L20 - mlp_L20 (pre-MLP residual stream). "
                     "Δout = mlp_L20 raw output. Both in residual-stream space. "
                     "No LayerNorm confound."),
            "din_action_mean":    float(din_act.mean()),
            "din_action_abs":     float(abs_din_act.mean()),
            "din_evidence_mean":  float(din_ev.mean()),
            "din_evidence_abs":   float(np.abs(din_ev).mean()),
            "dout_action_mean":   float(dout_act.mean()),
            "dout_action_abs":    float(np.abs(dout_act).mean()),
            "dout_evidence_mean": float(dout_ev.mean()),
            "dout_evidence_abs":  float(np.abs(dout_ev).mean()),
            "norm_ratio_mean":    float(norm_ratio.mean()),
            "routing_gain_median": float(np.nanmedian(gains)),
            "routing_gain_mean":   float(np.nanmean(gains)),
            "action_specific_gain": float(np.nanmedian(gains) / norm_ratio.mean()),
            "n_valid": int(valid.sum()),
            "n": len(routing_residual_rows),
        }

        print(f"\n{'='*70}")
        print(f"PHASE C (CORRECTED): RESIDUAL-SPACE MLP ROUTING  [L{args.measure_layer}]")
        print("  Δin  = residual_L20 - mlp_L20  (pre-MLP, residual-stream space)")
        print("  Δout = mlp_L20 raw output       (residual-stream space)")
        print("  Both projected onto action_dir and evidence_dir — no LN confound.")
        print("=" * 70)
        print(f"  Δin  · action:   mean={din_act.mean():+.4f}   |mean|={abs_din_act.mean():.4f}")
        print(f"  Δin  · evidence: mean={din_ev.mean():+.4f}   |mean|={np.abs(din_ev).mean():.4f}")
        print(f"  Δout · action:   mean={dout_act.mean():+.4f}   |mean|={np.abs(dout_act).mean():.4f}")
        print(f"  Δout · evidence: mean={dout_ev.mean():+.4f}   |mean|={np.abs(dout_ev).mean():.4f}")
        print(f"  ||Δout||/||Δin|| (uniform scaling): mean={norm_ratio.mean():.3f}")
        print(f"  Routing gain |Δout·act|/|Δin·act|: "
              f"median={np.nanmedian(gains):.3f}  mean={np.nanmean(gains):.3f}  "
              f"(n={valid.sum()}/{len(routing_residual_rows)})")
        print(f"  Action-specific gain (routing/uniform): "
              f"{np.nanmedian(gains)/norm_ratio.mean():.3f}x")

    # ── Phase D: MLP Jacobian Routing Test ──────────────────────────────────
    # Direct test: does mlp_L20 map evidence_dir → action_dir?
    # Compute J · v for v ∈ {evidence_dir, action_dir, random_dirs}
    # then project result onto action_dir.
    # If J·evidence_dir has significant action_dir component → routing is real.
    print(f"\n{'='*70}")
    print("PHASE D: MLP JACOBIAN ROUTING TEST")
    print("  For each sample, compute J·v where J = ∂mlp_output/∂mlp_input")
    print("  at the actual decision-point activation.")
    print("=" * 70)

    layers = get_model_layers(model)
    target_layer = layers[args.measure_layer]
    device = next(model.parameters()).device

    # Prepare direction tensors
    ev_t = torch.tensor(evidence_dir, dtype=torch.float32, device=device)
    act_t = torch.tensor(action_dir, dtype=torch.float32, device=device)
    n_random = 10
    rng_jac = np.random.RandomState(args.seed)
    random_dirs = []
    for _ in range(n_random):
        rd = rng_jac.randn(len(action_dir)).astype(np.float32)
        rd /= np.linalg.norm(rd) + 1e-12
        random_dirs.append(torch.tensor(rd, dtype=torch.float32, device=device))

    jacobian_rows = []

    for i, sample in enumerate(samples):
        rng_copy = random.Random(args.seed)
        for j in range(i):
            make_corrupted_obs(samples[j], "A", rng_copy)
        clean_obs, _ = make_corrupted_obs(sample, "A", rng_copy)

        prompt = build_prompt(tokenizer, sample["question"],
                              sample["step0_query"], clean_obs)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # 1) Get MLP input (post_attention_layernorm output) at last token
        captured_input = {}
        def ln_hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured_input['h'] = h[0, -1, :].detach().clone()  # keep on GPU
        handle = target_layer.post_attention_layernorm.register_forward_hook(ln_hook)
        with torch.no_grad():
            model(input_ids)
        handle.remove()

        mlp_input = captured_input['h'].float()  # (d_model,)

        # 2) Define mlp as a function of its input (single vector)
        mlp_module = target_layer.mlp

        def mlp_fn(x):
            return mlp_module(x.unsqueeze(0).unsqueeze(0).to(
                next(mlp_module.parameters()).dtype))[0, 0, :].float()

        # 3) JVP: J · evidence_dir, J · action_dir, J · random_dirs
        mlp_input_req = mlp_input.detach().requires_grad_(False)

        def jvp_project(tangent_vec):
            """Compute (J · tangent_vec) · action_dir using forward-mode AD."""
            x = mlp_input_req.detach().clone().requires_grad_(True)
            out = mlp_fn(x)
            # Use backward to get J^T · action_dir, then dot with tangent
            # (more memory-efficient than full JVP for single direction)
            out_weighted = (out * act_t).sum()
            out_weighted.backward()
            # x.grad = J^T · action_dir
            jt_action = x.grad.float()
            # (J · tangent_vec) · action_dir = tangent_vec · (J^T · action_dir)
            return float(torch.dot(tangent_vec, jt_action).item())

        j_ev_to_act = jvp_project(ev_t)
        j_act_to_act = jvp_project(act_t)
        j_rand_to_act = [jvp_project(rd) for rd in random_dirs]

        jacobian_rows.append({
            "sample_id": sample["sample_id"],
            "J_evidence_to_action": j_ev_to_act,
            "J_action_to_action": j_act_to_act,
            "J_random_to_action_mean": float(np.mean([abs(x) for x in j_rand_to_act])),
            "J_random_to_action_vals": j_rand_to_act,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}]", flush=True)

    # Summarize Phase D
    j_ev  = np.array([r["J_evidence_to_action"] for r in jacobian_rows])
    j_act = np.array([r["J_action_to_action"] for r in jacobian_rows])
    j_rnd = np.array([r["J_random_to_action_mean"] for r in jacobian_rows])

    jacobian_summary = {
        "J_evidence_to_action_mean": float(j_ev.mean()),
        "J_evidence_to_action_abs":  float(np.abs(j_ev).mean()),
        "J_evidence_to_action_std":  float(j_ev.std()),
        "J_action_to_action_mean":   float(j_act.mean()),
        "J_action_to_action_abs":    float(np.abs(j_act).mean()),
        "J_random_to_action_abs":    float(j_rnd.mean()),
        "ratio_evidence_vs_random":  float(np.abs(j_ev).mean() / (j_rnd.mean() + 1e-12)),
        "n": len(jacobian_rows),
        "n_random_dirs": n_random,
    }

    print(f"\n  (J · evidence_dir) · action_dir:  mean={j_ev.mean():+.4f}  "
          f"|mean|={np.abs(j_ev).mean():.4f}")
    print(f"  (J · action_dir)   · action_dir:  mean={j_act.mean():+.4f}  "
          f"|mean|={np.abs(j_act).mean():.4f}")
    print(f"  (J · random_dir)   · action_dir:  |mean|={j_rnd.mean():.4f}")
    print(f"  evidence/random ratio: {np.abs(j_ev).mean() / (j_rnd.mean()+1e-12):.2f}x")

    # ── Phase B: Activation Patching (top-K components) ──────────────────────
    top_k = args.top_k
    top_components = comp_summary[:top_k]
    print(f"\n{'='*70}")
    print(f"PHASE B: ACTIVATION PATCHING (top {top_k} components)")
    print("=" * 70)

    # For each sample, measure:
    #   action_clean = action projection of clean L20 residual
    #   action_corrupt = action projection of corrupted L20 residual
    #   action_patched_L = action projection after patching component L
    #   recovery_L = (action_patched_L - action_corrupt) / (action_clean - action_corrupt)

    patch_results = {c["component"]: [] for c in top_components}

    for i, sample in enumerate(samples):
        # Re-generate corruption (need to reset rng for reproducibility)
        rng_copy = random.Random(args.seed)
        # Advance to sample i for group A
        for j in range(i):
            make_corrupted_obs(samples[j], "A", rng_copy)
        clean_obs, corrupted_obs = make_corrupted_obs(sample, "A", rng_copy)

        prompt_clean = build_prompt(tokenizer, sample["question"],
                                    sample["step0_query"], clean_obs)
        prompt_corrupt = build_prompt(tokenizer, sample["question"],
                                      sample["step0_query"], corrupted_obs)

        # Get clean component cache
        clean_cache = extract_component_outputs(model, tokenizer, prompt_clean,
                                                 n_layers=n_layers)
        # Get baseline action projections
        corrupt_cache = extract_component_outputs(model, tokenizer, prompt_corrupt,
                                                   n_layers=n_layers)

        action_clean = float(np.dot(clean_cache['residual_L20'], action_dir))
        action_corrupt = float(np.dot(corrupt_cache['residual_L20'], action_dir))
        delta = action_clean - action_corrupt

        if abs(delta) < 1e-6:
            continue  # skip zero-effect samples

        for comp_info in top_components:
            comp_name = comp_info["component"]
            # Parse comp name: "attn_L5" -> ('attn', 5)
            parts = comp_name.split("_L")
            comp_type = parts[0]
            layer_idx = int(parts[1])

            action_patched = patch_and_measure(
                model, tokenizer, prompt_corrupt, clean_cache,
                comp_type, layer_idx, action_dir,
                measure_layer=args.measure_layer)

            recovery = (action_patched - action_corrupt) / delta
            patch_results[comp_name].append({
                "sample_id": sample["sample_id"],
                "action_clean": action_clean,
                "action_corrupt": action_corrupt,
                "action_patched": action_patched,
                "recovery": recovery,
            })

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(samples)}]", flush=True)

    # Summarize Phase B
    print(f"\nPhase B Results (recovery = fraction of effect restored):")
    print(f"{'Component':<12} {'mean_recov':>12} {'median_recov':>14} {'n':>5}")
    patch_summary = []
    for comp_info in top_components:
        name = comp_info["component"]
        recoveries = [r["recovery"] for r in patch_results[name]]
        if len(recoveries) == 0:
            continue
        arr = np.array(recoveries)
        row = {
            "component": name,
            "mean_recovery": float(arr.mean()),
            "median_recovery": float(np.median(arr)),
            "std_recovery": float(arr.std()),
            "n": len(arr),
        }
        patch_summary.append(row)
        print(f"  {name:<12} {row['mean_recovery']:>+12.4f} "
              f"{row['median_recovery']:>14.4f} {row['n']:>5}")

    # ── Phase E: Cross-position attention patching ─────────────────────────
    # For attn_L{18,19,20}: patch observation-position hidden states
    # from clean into corrupt at the attention input (input_layernorm output).
    # This tests: which attention layer reads evidence from observation tokens
    # and routes it to the action subspace at the last (decision) token?
    phase_e_layers = [16, 18, 19, 20]
    print(f"\n{'='*70}")
    print(f"PHASE E: CROSS-POSITION ATTENTION PATCHING")
    print(f"  Layers tested: {phase_e_layers}")
    print(f"  Patch observation-position attn input from clean → corrupt run")
    print(f"  Measure action projection recovery at last token of L{args.measure_layer}")
    print("=" * 70)

    phase_e_results = {l: [] for l in phase_e_layers}

    for i, sample in enumerate(samples):
        rng_copy = random.Random(args.seed)
        for j in range(i):
            make_corrupted_obs(samples[j], "A", rng_copy)
        clean_obs, corrupted_obs = make_corrupted_obs(sample, "A", rng_copy)

        prompt_clean = build_prompt(tokenizer, sample["question"],
                                    sample["step0_query"], clean_obs)
        prompt_corrupt = build_prompt(tokenizer, sample["question"],
                                      sample["step0_query"], corrupted_obs)

        # Tokenize both to find divergent range
        ids_clean = tokenizer.encode(prompt_clean)
        ids_corrupt = tokenizer.encode(prompt_corrupt)
        obs_start, obs_end_c, obs_end_x = find_divergent_range(ids_clean, ids_corrupt)

        if obs_end_c <= obs_start or obs_end_x <= obs_start:
            continue  # no divergence found (shouldn't happen)

        # Get baseline action projections by extracting L20 residuals
        clean_comp = extract_component_outputs(model, tokenizer, prompt_clean,
                                                n_layers=n_layers)
        corrupt_comp = extract_component_outputs(model, tokenizer, prompt_corrupt,
                                                  n_layers=n_layers)
        action_clean = float(np.dot(clean_comp['residual_L20'], action_dir))
        action_corrupt = float(np.dot(corrupt_comp['residual_L20'], action_dir))
        delta = action_clean - action_corrupt
        if abs(delta) < 1e-6:
            continue

        # For each target layer, capture clean LN output and patch
        for target_l in phase_e_layers:
            # Capture clean input_layernorm output at all positions
            clean_ln, _ = capture_ln_output(model, tokenizer, prompt_clean, target_l)

            # Patch and measure
            action_patched = cross_position_patch_and_measure(
                model, tokenizer, prompt_corrupt,
                clean_ln, obs_start, obs_end_c,
                obs_start, obs_end_x,
                target_l, action_dir, measure_layer=args.measure_layer)

            recovery = (action_patched - action_corrupt) / delta
            phase_e_results[target_l].append({
                "sample_id": sample["sample_id"],
                "obs_tokens_clean": obs_end_c - obs_start,
                "obs_tokens_corrupt": obs_end_x - obs_start,
                "action_clean": action_clean,
                "action_corrupt": action_corrupt,
                "action_patched": action_patched,
                "recovery": recovery,
            })

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(samples)}]", flush=True)

    # Summarize Phase E
    print(f"\nPhase E Results (recovery = fraction of action effect restored):")
    print(f"{'Layer':<12} {'mean_recov':>12} {'median_recov':>14} {'n':>5}")
    phase_e_summary = []
    for target_l in phase_e_layers:
        rows = phase_e_results[target_l]
        if not rows:
            continue
        recoveries = np.array([r["recovery"] for r in rows])
        row = {
            "layer": target_l,
            "mean_recovery": float(recoveries.mean()),
            "median_recovery": float(np.median(recoveries)),
            "std_recovery": float(recoveries.std()),
            "n": len(recoveries),
        }
        phase_e_summary.append(row)
        print(f"  attn_L{target_l:<6} {row['mean_recovery']:>+12.4f} "
              f"{row['median_recovery']:>14.4f} {row['n']:>5}")

    # ── Phase F: L18 attention head decomposition ─────────────────────────
    phase_f_layer = 18
    n_heads = 28
    head_dim = 128
    print(f"\n{'='*70}")
    print(f"PHASE F: ATTENTION HEAD DECOMPOSITION (L{phase_f_layer})")
    print(f"  F1: Linear attribution — per-head contribution to action_dir")
    print(f"  F2: Causal patching — per-head output patch at last token")
    print(f"  {n_heads} query heads, {head_dim} head_dim, 4 KV heads (GQA)")
    print("=" * 70)

    # Get o_proj weight matrix for head decomposition
    layers = get_model_layers(model)
    o_proj_weight = (layers[phase_f_layer].self_attn.o_proj.weight
                     .detach().float().cpu().numpy())  # (d_model, n_heads*head_dim)

    # F1: Linear decomposition
    head_attributions = defaultdict(list)  # head_idx -> [action_proj per sample]
    # F2: Causal patching per head (only for top heads after F1)
    # We'll collect clean pre-o_proj for all samples first, then decide which heads to patch

    f1_per_sample = []

    for i, sample in enumerate(samples):
        rng_copy = random.Random(args.seed)
        for j in range(i):
            make_corrupted_obs(samples[j], "A", rng_copy)
        clean_obs, corrupted_obs = make_corrupted_obs(sample, "A", rng_copy)

        prompt_clean = build_prompt(tokenizer, sample["question"],
                                    sample["step0_query"], clean_obs)
        prompt_corrupt = build_prompt(tokenizer, sample["question"],
                                      sample["step0_query"], corrupted_obs)

        # Capture pre-o_proj at last token
        pre_oproj_clean = capture_attn_pre_oproj(
            model, tokenizer, prompt_clean, phase_f_layer)
        pre_oproj_corrupt = capture_attn_pre_oproj(
            model, tokenizer, prompt_corrupt, phase_f_layer)

        head_results, full_proj, sum_proj = decompose_attn_by_head(
            pre_oproj_clean, pre_oproj_corrupt, o_proj_weight,
            action_dir, n_heads=n_heads, head_dim=head_dim)

        sample_row = {
            "sample_id": sample["sample_id"],
            "full_proj": full_proj,
            "sum_head_proj": sum_proj,
            "verification_err": abs(full_proj - sum_proj),
            "heads": head_results,
        }
        f1_per_sample.append(sample_row)

        for hr in head_results:
            head_attributions[hr["head"]].append(hr["action_proj"])

        if (i + 1) % 10 == 0:
            print(f"  F1 [{i+1}/{len(samples)}]", flush=True)

    # Summarize F1
    print(f"\nPhase F1 Results (per-head contribution to attn_L{phase_f_layer} action effect):")
    print(f"{'Head':>6} {'KV_grp':>7} {'mean_proj':>11} {'|mean|_proj':>12} {'std':>8}")
    f1_summary = []
    for h in range(n_heads):
        projs = np.array(head_attributions[h])
        row = {
            "head": h,
            "kv_group": h // (n_heads // 4),  # 7 Q heads per KV head
            "mean_proj": float(projs.mean()),
            "abs_mean_proj": float(np.abs(projs).mean()),
            "std_proj": float(projs.std()),
        }
        f1_summary.append(row)

    # Sort by |mean| for display
    f1_sorted = sorted(f1_summary, key=lambda r: r["abs_mean_proj"], reverse=True)
    for row in f1_sorted[:10]:
        print(f"  H{row['head']:<4} KV{row['kv_group']:<4} {row['mean_proj']:>+11.4f} "
              f"{row['abs_mean_proj']:>12.4f} {row['std_proj']:>8.4f}")

    # F2: Causal patching for top-5 heads
    top5_heads = [r["head"] for r in f1_sorted[:5]]
    print(f"\nPhase F2: Causal patching top-5 heads: {top5_heads}")
    f2_results = {h: [] for h in top5_heads}

    for i, sample in enumerate(samples):
        rng_copy = random.Random(args.seed)
        for j in range(i):
            make_corrupted_obs(samples[j], "A", rng_copy)
        clean_obs, corrupted_obs = make_corrupted_obs(sample, "A", rng_copy)

        prompt_clean = build_prompt(tokenizer, sample["question"],
                                    sample["step0_query"], clean_obs)
        prompt_corrupt = build_prompt(tokenizer, sample["question"],
                                      sample["step0_query"], corrupted_obs)

        # Get baseline action projections
        clean_comp = extract_component_outputs(model, tokenizer, prompt_clean,
                                                n_layers=n_layers)
        corrupt_comp = extract_component_outputs(model, tokenizer, prompt_corrupt,
                                                  n_layers=n_layers)
        action_clean = float(np.dot(clean_comp['residual_L20'], action_dir))
        action_corrupt = float(np.dot(corrupt_comp['residual_L20'], action_dir))
        delta = action_clean - action_corrupt
        if abs(delta) < 1e-6:
            continue

        # Capture clean pre-o_proj
        pre_oproj_clean = capture_attn_pre_oproj(
            model, tokenizer, prompt_clean, phase_f_layer)

        for h in top5_heads:
            action_patched = per_head_output_patch_and_measure(
                model, tokenizer, prompt_corrupt,
                pre_oproj_clean, h,
                phase_f_layer, action_dir,
                measure_layer=args.measure_layer,
                n_heads=n_heads, head_dim=head_dim)
            recovery = (action_patched - action_corrupt) / delta
            f2_results[h].append({
                "sample_id": sample["sample_id"],
                "recovery": recovery,
            })

        if (i + 1) % 10 == 0:
            print(f"  F2 [{i+1}/{len(samples)}]", flush=True)

    # Summarize F2
    print(f"\nPhase F2 Results (per-head causal recovery):")
    print(f"{'Head':>6} {'mean_recov':>12} {'median_recov':>14} {'n':>5}")
    f2_summary = []
    for h in top5_heads:
        rows = f2_results[h]
        if not rows:
            continue
        recoveries = np.array([r["recovery"] for r in rows])
        row = {
            "head": h,
            "mean_recovery": float(recoveries.mean()),
            "median_recovery": float(np.median(recoveries)),
            "std_recovery": float(recoveries.std()),
            "n": len(recoveries),
        }
        f2_summary.append(row)
        print(f"  H{h:<4} {row['mean_recovery']:>+12.4f} "
              f"{row['median_recovery']:>14.4f} {row['n']:>5}")

    # ── Phase F3: KV Group level patching ───────────────────────────────────
    n_kv_heads = 4
    heads_per_kv = n_heads // n_kv_heads  # 7
    print(f"\n{'='*70}")
    print(f"PHASE F3: KV GROUP LEVEL PATCHING (L{phase_f_layer})")
    print(f"  4 KV groups, each with {heads_per_kv} Q-heads")
    print(f"  KV0=H0-H6, KV1=H7-H13, KV2=H14-H20, KV3=H21-H27")
    print("=" * 70)

    f3_results = {g: [] for g in range(n_kv_heads)}

    for i, sample in enumerate(samples):
        rng_copy = random.Random(args.seed)
        for j in range(i):
            make_corrupted_obs(samples[j], "A", rng_copy)
        clean_obs, corrupted_obs = make_corrupted_obs(sample, "A", rng_copy)

        prompt_clean = build_prompt(tokenizer, sample["question"],
                                    sample["step0_query"], clean_obs)
        prompt_corrupt = build_prompt(tokenizer, sample["question"],
                                      sample["step0_query"], corrupted_obs)

        clean_comp = extract_component_outputs(model, tokenizer, prompt_clean,
                                                n_layers=n_layers)
        corrupt_comp = extract_component_outputs(model, tokenizer, prompt_corrupt,
                                                  n_layers=n_layers)
        action_clean = float(np.dot(clean_comp['residual_L20'], action_dir))
        action_corrupt = float(np.dot(corrupt_comp['residual_L20'], action_dir))
        delta = action_clean - action_corrupt
        if abs(delta) < 1e-6:
            continue

        pre_oproj_clean = capture_attn_pre_oproj(
            model, tokenizer, prompt_clean, phase_f_layer)

        for g in range(n_kv_heads):
            action_patched = per_kv_group_output_patch_and_measure(
                model, tokenizer, prompt_corrupt,
                pre_oproj_clean, g,
                phase_f_layer, action_dir,
                measure_layer=args.measure_layer,
                n_heads=n_heads, head_dim=head_dim, n_kv_heads=n_kv_heads)
            recovery = (action_patched - action_corrupt) / delta
            f3_results[g].append({
                "sample_id": sample["sample_id"],
                "recovery": recovery,
            })

        if (i + 1) % 10 == 0:
            print(f"  F3 [{i+1}/{len(samples)}]", flush=True)

    print(f"\nPhase F3 Results (KV group causal recovery):")
    print(f"{'KV_Group':>10} {'Heads':>12} {'mean_recov':>12} {'median_recov':>14} {'n':>5}")
    f3_summary = []
    for g in range(n_kv_heads):
        rows = f3_results[g]
        if not rows:
            continue
        recoveries = np.array([r["recovery"] for r in rows])
        head_range = f"H{g*heads_per_kv}-H{(g+1)*heads_per_kv-1}"
        row = {
            "kv_group": g,
            "heads": head_range,
            "mean_recovery": float(recoveries.mean()),
            "median_recovery": float(np.median(recoveries)),
            "std_recovery": float(recoveries.std()),
            "n": len(recoveries),
        }
        f3_summary.append(row)
        print(f"  KV{g:<7} {head_range:>12} {row['mean_recovery']:>+12.4f} "
              f"{row['median_recovery']:>14.4f} {row['n']:>5}")

    # ── Phase F4: Attention pattern analysis ──────────────────────────────────
    print(f"\n{'='*70}")
    print(f"PHASE F4: ATTENTION PATTERN ANALYSIS (L{phase_f_layer})")
    print(f"  Attention mass on observation vs non-observation tokens")
    print(f"  at last (decision) token, per head")
    print("=" * 70)

    # Collect per-head attention mass on observation tokens
    f4_obs_mass_clean = defaultdict(list)   # head -> [mass on obs tokens]
    f4_obs_mass_corrupt = defaultdict(list)
    f4_per_sample = []

    for i, sample in enumerate(samples):
        rng_copy = random.Random(args.seed)
        for j in range(i):
            make_corrupted_obs(samples[j], "A", rng_copy)
        clean_obs, corrupted_obs = make_corrupted_obs(sample, "A", rng_copy)

        prompt_clean = build_prompt(tokenizer, sample["question"],
                                    sample["step0_query"], clean_obs)
        prompt_corrupt = build_prompt(tokenizer, sample["question"],
                                      sample["step0_query"], corrupted_obs)

        ids_clean = tokenizer.encode(prompt_clean)
        ids_corrupt = tokenizer.encode(prompt_corrupt)
        obs_start, obs_end_c, obs_end_x = find_divergent_range(ids_clean, ids_corrupt)

        if obs_end_c <= obs_start:
            continue

        # Capture attention weights (pre-RoPE approximation)
        attn_clean, _ = capture_attention_weights(
            model, tokenizer, prompt_clean, phase_f_layer)
        attn_corrupt, _ = capture_attention_weights(
            model, tokenizer, prompt_corrupt, phase_f_layer)

        # Compute obs attention mass per head
        sample_row = {"sample_id": sample["sample_id"],
                      "obs_range_clean": [obs_start, obs_end_c],
                      "obs_range_corrupt": [obs_start, obs_end_x],
                      "seq_len_clean": len(ids_clean),
                      "seq_len_corrupt": len(ids_corrupt),
                      "heads_clean": [], "heads_corrupt": []}
        for h in range(n_heads):
            mass_c = float(attn_clean[h, obs_start:obs_end_c].sum())
            mass_x = float(attn_corrupt[h, obs_start:obs_end_x].sum())
            f4_obs_mass_clean[h].append(mass_c)
            f4_obs_mass_corrupt[h].append(mass_x)
            sample_row["heads_clean"].append(mass_c)
            sample_row["heads_corrupt"].append(mass_x)
        f4_per_sample.append(sample_row)

        if (i + 1) % 10 == 0:
            print(f"  F4 [{i+1}/{len(samples)}]", flush=True)

    # Summarize F4
    print(f"\nPhase F4 Results (attention mass on observation tokens at last position):")
    print(f"{'Head':>6} {'KV':>4} {'clean_obs%':>11} {'corrupt_obs%':>13} {'delta':>8}")
    f4_summary = []
    for h in range(n_heads):
        mc = np.array(f4_obs_mass_clean[h])
        mx = np.array(f4_obs_mass_corrupt[h])
        row = {
            "head": h,
            "kv_group": h // heads_per_kv,
            "mean_obs_mass_clean": float(mc.mean()),
            "mean_obs_mass_corrupt": float(mx.mean()),
            "delta_obs_mass": float(mc.mean() - mx.mean()),
        }
        f4_summary.append(row)

    # Sort by clean obs mass to find which heads attend most to observations
    f4_sorted = sorted(f4_summary, key=lambda r: r["mean_obs_mass_clean"], reverse=True)
    for row in f4_sorted[:10]:
        print(f"  H{row['head']:<4} KV{row['kv_group']:<2} "
              f"{row['mean_obs_mass_clean']:>10.4f} "
              f"{row['mean_obs_mass_corrupt']:>12.4f} "
              f"{row['delta_obs_mass']:>+8.4f}")

    # Also show per-KV-group aggregate
    print(f"\n  Per KV-group aggregate:")
    for g in range(n_kv_heads):
        g_heads = range(g * heads_per_kv, (g + 1) * heads_per_kv)
        mc = np.mean([np.mean(f4_obs_mass_clean[h]) for h in g_heads])
        mx = np.mean([np.mean(f4_obs_mass_corrupt[h]) for h in g_heads])
        print(f"  KV{g}: clean_obs={mc:.4f}  corrupt_obs={mx:.4f}  delta={mc-mx:+.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": args.model, "n_samples": len(samples),
            "measure_layer": args.measure_layer,
            "top_k_patched": top_k, "seed": args.seed,
            "cos_evidence_action": cos_ev_act,
        },
        "phase_a": {
            "component_summary": comp_summary,
            "mean_verification_error": float(np.mean(verification_errors)),
        },
        "phase_b": {
            "patch_summary": patch_summary,
            "per_sample": {name: patch_results[name] for name in patch_results},
        },
        "phase_c_corrected": {
            "summary": routing_corrected_summary,
            "per_sample": routing_residual_rows,
        },
        "phase_d_jacobian": {
            "summary": jacobian_summary,
            "per_sample": jacobian_rows,
        },
        "phase_e_cross_position": {
            "summary": phase_e_summary,
            "per_sample": {str(l): phase_e_results[l] for l in phase_e_layers},
        },
        "phase_f_head_analysis": {
            "layer": phase_f_layer,
            "f1_summary": f1_sorted,
            "f1_per_sample": f1_per_sample,
            "f2_summary": f2_summary,
            "f2_per_sample": {str(h): f2_results[h] for h in top5_heads},
            "f3_kv_group_summary": f3_summary,
            "f3_per_sample": {str(g): f3_results[g] for g in range(n_kv_heads)},
            "f4_attention_summary": f4_summary,
            "f4_per_sample": f4_per_sample,
        },
        "per_sample_decomposition": per_sample_data,
    }
    out_path = os.path.join(args.output_dir, "activation_patching_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--baseline-trace",
                    default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--hotpotqa-data",
                    default="data/hotpotqa/hotpot_dev_distractor_v1.json")
    ap.add_argument("--action-dir",
                    default="steering/directions/direction_search_v3_layer20.npz")
    ap.add_argument("--evidence-dir",
                    default="results/phase1_probe/probe_direction_l20.npz")
    ap.add_argument("--output-dir", default="results/paired_corruption")
    ap.add_argument("--measure-layer", type=int, default=20)
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=10,
                    help="Number of top components to patch in Phase B")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

