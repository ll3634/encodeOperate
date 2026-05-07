#!/usr/bin/env python3
"""
SAE Feature Steering Pipeline for Qwen2.5-7B-Instruct.

Steps:
  1. Load model + SAE (resid_post_layer_11, BatchTopKSAE)
  2. Reconstruct step-1 decision prompts from baseline traces
  3. Capture Layer 11 hidden states + compute action margins
  4. Encode hidden states with SAE → find differential features
  5. Save top feature's decoder vector as .npz steering direction

Usage:
    cd tmc/scripts/e2e_agent
    python scripts/sae_feature_steering.py \
        --baseline-trace results/direction_comparison_n200/baseline_results.jsonl \
        --sae-path ../../../sae_weights/resid_post_layer_11/trainer_2/ae.pt \
        --output steering/directions/direction_sae_search_feature.npz \
        --n-samples 200
"""

import os, sys, json, argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import get_model_layers


def load_model(model_name="Qwen/Qwen2.5-7B-Instruct"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # suppress sdpa warning
    )
    mdl.eval()
    # Print device map so we can see where layers ended up
    if hasattr(mdl, "hf_device_map"):
        devices = set(mdl.hf_device_map.values())
        print(f"  Device map: {devices}")
    return mdl, tok


def load_sae(sae_path, device="cpu"):
    from dictionary_learning.trainers.batch_top_k import BatchTopKSAE
    print(f"Loading SAE from: {sae_path}")
    ae = BatchTopKSAE.from_pretrained(sae_path, device=device)
    ae.eval()
    print(f"  dict_size={ae.dict_size}, activation_dim={ae.activation_dim}, k={ae.k}")
    return ae


def load_baseline_traces(path, n_samples=None):
    """Load baseline episode traces and extract step-1 decision contexts."""
    episodes = []
    with open(path) as f:
        for line in f:
            ep = json.loads(line)
            episodes.append(ep)
    if n_samples:
        episodes = episodes[:n_samples]

    valid = []
    for ep in episodes:
        if not ep.get("steps") or len(ep["steps"]) < 1:
            continue
        s0 = ep["steps"][0]
        if s0.get("action") != "search" or not s0.get("observation"):
            continue
        valid.append({
            "sample_id": ep["sample_id"],
            "question": ep["question"],
            "step0_query": s0["action_input"],
            "step0_obs": s0["observation"],
        })
    print(f"Loaded {len(valid)}/{len(episodes)} valid episodes with step-0 search")
    return valid


def collect_hidden_states_and_margins(model, tokenizer, episodes, layer=11):
    """Capture Layer 11 hidden states at step-1 decision point."""
    pb = PromptBuilder(tools=["search", "calculator"])
    layers = get_model_layers(model)
    device = next(model.parameters()).device

    # Get action token IDs
    tool_tokens, finish_tokens = [], []
    for t in ACTION_TOKENS["tool_call"]:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if ids: tool_tokens.append(ids[0])
    for t in ACTION_TOKENS["finish"]:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if ids: finish_tokens.append(ids[0])

    hidden_states = []
    margins = []
    sample_ids = []

    print(f"Starting forward passes for {len(episodes)} episodes at layer {layer}...")
    print("  (first checkpoint at episode 20)")

    for i, ep in enumerate(episodes):
        # Build step-1 prompt: system + question + step0 scratchpad
        steps = [{
            "action": "search",
            "action_input": ep["step0_query"],
            "observation": ep["step0_obs"],
        }]
        messages = pb.build_full_prompt(ep["question"], steps)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Forward pass with hook to capture layer 11 hidden state
        print(f"  [{i+1}/{len(episodes)}] {ep['sample_id'][:20]}...", end="", flush=True)
        captured = {}

        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured["hidden"] = h.detach()

        handle = layers[layer].register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                outputs = model(input_ids)
        except Exception as e:
            handle.remove()
            print(f" ERROR: {e}")
            continue
        handle.remove()

        if "hidden" not in captured:
            print(f" SKIP: hook did not fire")
            continue

        # Extract hidden state at last position
        h_last = captured["hidden"][0, -1, :].float().cpu().numpy()  # [3584]
        hidden_states.append(h_last)

        # Compute margin
        logits = outputs.logits[0, -1, :]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        tool_lp = torch.logsumexp(log_probs[tool_tokens], dim=0).item() if tool_tokens else -100.0
        fin_lp = torch.logsumexp(log_probs[finish_tokens], dim=0).item() if finish_tokens else -100.0
        margin = tool_lp - fin_lp
        margins.append(margin)
        sample_ids.append(ep["sample_id"])

        action = "search" if margin > 0 else "finish"
        print(f" margin={margin:+.2f} → {action}", flush=True)

    hidden_states = np.array(hidden_states, dtype=np.float32)  # [N, 3584]
    margins = np.array(margins, dtype=np.float32)
    print(f"\nCollected {len(hidden_states)} hidden states")
    n_search = (margins > 0).sum()
    print(f"  Margin > 0 (prefer search): {n_search}")
    print(f"  Margin <= 0 (prefer finish): {len(margins) - n_search}")
    return hidden_states, margins, sample_ids


def identify_search_features(sae, hidden_states, margins, top_k=20):
    """Encode hidden states with SAE and find features most correlated with search."""
    device = next(sae.parameters()).device

    # Encode all hidden states
    x = torch.tensor(hidden_states, dtype=torch.float32).to(device)
    with torch.no_grad():
        features = sae.encode(x)  # [N, dict_size]
    features = features.cpu().numpy()  # [N, 131072]

    # Split by margin sign
    search_mask = margins > 0  # model prefers search
    finish_mask = ~search_mask

    n_search = search_mask.sum()
    n_finish = finish_mask.sum()
    print(f"\nSAE encoding complete: {features.shape}")
    print(f"  Search group: {n_search}, Finish group: {n_finish}")

    if n_search < 5 or n_finish < 5:
        print("WARNING: One group has <5 samples, results may be unreliable")

    # Compute mean activation per feature for each group
    mean_search = features[search_mask].mean(axis=0)  # [131072]
    mean_finish = features[finish_mask].mean(axis=0)  # [131072]

    # Differential activation: search - finish
    diff = mean_search - mean_finish  # positive = more active during search

    # Also compute effect size (Cohen's d)
    std_search = features[search_mask].std(axis=0) + 1e-8
    std_finish = features[finish_mask].std(axis=0) + 1e-8
    pooled_std = np.sqrt((std_search**2 * (n_search-1) + std_finish**2 * (n_finish-1))
                         / (n_search + n_finish - 2))
    cohens_d = diff / (pooled_std + 1e-8)

    # Find top features by absolute differential activation
    top_by_diff = np.argsort(np.abs(diff))[::-1][:top_k]
    # Find top features by Cohen's d
    top_by_d = np.argsort(np.abs(cohens_d))[::-1][:top_k]

    print(f"\nTop {top_k} features by |mean diff| (search - finish):")
    print(f"  {'Rank':>4} {'Feature':>8} {'MeanDiff':>10} {'CohenD':>8} "
          f"{'MeanSearch':>11} {'MeanFinish':>11} {'Sparsity':>8}")
    for rank, idx in enumerate(top_by_diff):
        sparsity = (features[:, idx] != 0).mean()
        print(f"  {rank+1:>4} {idx:>8} {diff[idx]:>+10.4f} {cohens_d[idx]:>+8.3f} "
              f"{mean_search[idx]:>11.4f} {mean_finish[idx]:>11.4f} {sparsity:>8.3f}")

    print(f"\nTop {top_k} features by |Cohen's d|:")
    for rank, idx in enumerate(top_by_d):
        sparsity = (features[:, idx] != 0).mean()
        print(f"  {rank+1:>4} {idx:>8} {diff[idx]:>+10.4f} {cohens_d[idx]:>+8.3f} "
              f"{mean_search[idx]:>11.4f} {mean_finish[idx]:>11.4f} {sparsity:>8.3f}")

    return {
        "diff": diff,
        "cohens_d": cohens_d,
        "top_by_diff": top_by_diff,
        "top_by_d": top_by_d,
        "features": features,
        "mean_search": mean_search,
        "mean_finish": mean_finish,
    }


def save_sae_direction(sae, feature_idx, output_path, metadata):
    """Extract decoder vector for feature and save as .npz."""
    device = next(sae.parameters()).device
    # decoder.weight shape: [activation_dim, dict_size]
    dec_vec = sae.decoder.weight[:, feature_idx].detach().cpu().numpy().astype(np.float32)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        str(out),
        decision_direction=dec_vec,
        layer=11,
        method="sae_feature_steering",
        sae_feature_idx=int(feature_idx),
        **metadata,
    )
    norm = float(np.linalg.norm(dec_vec))
    rms = float(np.sqrt(np.mean(dec_vec**2)))
    print(f"\nSaved SAE direction to {out}")
    print(f"  Feature index: {feature_idx}")
    print(f"  Decoder vector norm: {norm:.4f}, RMS: {rms:.6f}")
    return dec_vec


def main():
    parser = argparse.ArgumentParser(description="SAE Feature Steering Pipeline")
    parser.add_argument("--baseline-trace", required=True, help="Baseline JSONL trace")
    parser.add_argument("--sae-path", required=True, help="Path to ae.pt")
    parser.add_argument("--output", default="steering/directions/direction_sae_search_feature.npz")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=int, default=11, help="Layer for hidden state capture")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20, help="Number of top features to report")
    parser.add_argument("--feature-select", default="cohens_d",
                        choices=["cohens_d", "mean_diff"],
                        help="Criterion for selecting the steering feature")
    args = parser.parse_args()

    print(f"=== SAE Feature Steering Pipeline ===")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"  Layer: {args.layer}")
    print()

    # 1. Load baseline traces
    episodes = load_baseline_traces(args.baseline_trace, args.n_samples)

    # 2. Load model
    model, tokenizer = load_model(args.model)

    # 3. Collect hidden states and margins
    hidden_states, margins, sample_ids = collect_hidden_states_and_margins(
        model, tokenizer, episodes, layer=args.layer
    )

    # Free model memory before loading SAE
    del model
    torch.cuda.empty_cache()
    print("Freed model memory")

    # 4. Load SAE and identify features
    sae = load_sae(args.sae_path, device="cpu")
    result = identify_search_features(sae, hidden_states, margins, top_k=args.top_k)

    # 5. Select best feature and save direction
    if args.feature_select == "cohens_d":
        best_idx = result["top_by_d"][0]
        criterion = f"cohens_d={result['cohens_d'][best_idx]:.4f}"
    else:
        best_idx = result["top_by_diff"][0]
        criterion = f"mean_diff={result['diff'][best_idx]:.4f}"

    metadata = {
        "feature_select_criterion": args.feature_select,
        "feature_score": criterion,
        "n_samples": len(hidden_states),
        "n_search": int((margins > 0).sum()),
        "n_finish": int((margins <= 0).sum()),
        "mean_diff": float(result["diff"][best_idx]),
        "cohens_d_value": float(result["cohens_d"][best_idx]),
    }

    dec_vec = save_sae_direction(sae, best_idx, args.output, metadata)

    # Also save top-5 features as separate directions for comparison
    top5_key = "top_by_d" if args.feature_select == "cohens_d" else "top_by_diff"
    out_dir = Path(args.output).parent
    for rank in range(min(5, len(result[top5_key]))):
        fidx = result[top5_key][rank]
        save_sae_direction(
            sae, fidx,
            str(out_dir / f"direction_sae_feature_rank{rank+1}_f{fidx}.npz"),
            {**metadata, "rank": rank + 1, "feature_idx": int(fidx)},
        )

    # Save full analysis
    analysis_path = Path(args.output).with_suffix(".json")
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "n_episodes": len(episodes),
        "n_search": int((margins > 0).sum()),
        "n_finish": int((margins <= 0).sum()),
        "top_features_by_diff": [
            {"rank": r+1, "feature_idx": int(result["top_by_diff"][r]),
             "mean_diff": float(result["diff"][result["top_by_diff"][r]]),
             "cohens_d": float(result["cohens_d"][result["top_by_diff"][r]])}
            for r in range(min(20, len(result["top_by_diff"])))
        ],
        "top_features_by_cohens_d": [
            {"rank": r+1, "feature_idx": int(result["top_by_d"][r]),
             "mean_diff": float(result["diff"][result["top_by_d"][r]]),
             "cohens_d": float(result["cohens_d"][result["top_by_d"][r]])}
            for r in range(min(20, len(result["top_by_d"])))
        ],
        "selected_feature": int(best_idx),
        "selected_criterion": criterion,
    }
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved analysis to {analysis_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()

