#!/usr/bin/env python3
"""
Build composite steering directions from SAE features.

Two approaches:
  A. Composite Feature Direction: Weighted sum of top SAE decoder vectors
  B. SAE-Projected CAA: Encode CAA direction → filter bad features → reconstruct

Usage:
    cd tmc/scripts/e2e_agent
    python scripts/build_sae_composite_direction.py \
        --sae-path ../../../sae_weights/resid_post_layer_11/trainer_2/ae.pt \
        --feature-analysis steering/directions/direction_sae_search_feature.json \
        --caa-direction steering/directions/direction_search_post_runtime_trace_clean_eval200_seed42_bridge_v2_hook_fixed.npz \
        --output-dir steering/directions \
        --top-n 10 --min-cohens-d 0.4
"""

import os, sys, json, argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_sae(sae_path, device="cpu"):
    from dictionary_learning.trainers.batch_top_k import BatchTopKSAE
    print(f"Loading SAE from: {sae_path}")
    ae = BatchTopKSAE.from_pretrained(sae_path, device=device)
    ae.eval()
    print(f"  dict_size={ae.dict_size}, activation_dim={ae.activation_dim}, k={ae.k}")
    return ae


def build_composite_direction(sae, feature_analysis, top_n=10, min_d=0.3,
                               weighting="cohens_d"):
    """Build composite direction from top SAE features weighted by effect size.

    Combines positive-d features (pro-search) additively and negative-d features
    (pro-finish) subtractively, so the composite pushes toward search.
    """
    # Get features sorted by |Cohen's d|
    features_by_d = feature_analysis["top_features_by_cohens_d"]

    selected = []
    for f in features_by_d:
        d = f["cohens_d"]
        if abs(d) < min_d:
            continue
        selected.append(f)
        if len(selected) >= top_n:
            break

    if not selected:
        raise ValueError(f"No features with |Cohen's d| >= {min_d}")

    print(f"\nBuilding composite from {len(selected)} features (|d| >= {min_d}):")
    print(f"  {'Idx':>8} {'CohenD':>8} {'MeanDiff':>10} {'Weight':>10}")

    decoder = sae.decoder.weight.detach().cpu().numpy()  # [activation_dim, dict_size]
    composite = np.zeros(decoder.shape[0], dtype=np.float64)

    feature_info = []
    for f in selected:
        idx = f["feature_idx"]
        d = f["cohens_d"]
        diff = f["mean_diff"]

        # Weight: sign of Cohen's d determines if we add or subtract
        # Use d * |diff| to weight by both effect size and magnitude
        if weighting == "cohens_d":
            w = d  # simple: weight by Cohen's d (sign included)
        elif weighting == "d_times_diff":
            w = d * abs(diff)
        else:
            w = 1.0 if d > 0 else -1.0  # uniform weight, sign from d

        dec_vec = decoder[:, idx]
        composite += w * dec_vec
        print(f"  {idx:>8} {d:>+8.3f} {diff:>+10.4f} {w:>+10.4f}")
        feature_info.append({"idx": idx, "cohens_d": d, "mean_diff": diff, "weight": w})

    composite = composite.astype(np.float32)
    norm = float(np.linalg.norm(composite))
    rms = float(np.sqrt(np.mean(composite ** 2)))
    print(f"\n  Composite norm: {norm:.4f}, RMS: {rms:.6f}")
    print(f"  N features: {len(selected)}")
    return composite, feature_info


def build_sae_projected_caa(sae, caa_direction, feature_analysis,
                             min_d=0.0, keep_mode="positive"):
    """Project CAA direction through SAE, filter, reconstruct.

    1. Encode: h = SAE.encode(v_CAA) → sparse activations
    2. Filter: zero out features with negative Cohen's d (anti-search)
    3. Decode: v_clean = SAE.decode(h_filtered)
    """
    device = next(sae.parameters()).device

    # Build lookup of Cohen's d for each feature
    d_lookup = {}
    for flist in [feature_analysis["top_features_by_cohens_d"],
                  feature_analysis["top_features_by_diff"]]:
        for f in flist:
            d_lookup[f["feature_idx"]] = f["cohens_d"]

    # Encode CAA direction through SAE
    x = torch.tensor(caa_direction, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        features = sae.encode(x)  # [1, dict_size]

    features_np = features[0].cpu().numpy()
    nonzero = np.nonzero(features_np)[0]
    print(f"\nSAE-Projected CAA: {len(nonzero)} non-zero features out of {len(features_np)}")

    # Show top activations
    top_acts = sorted(zip(nonzero, features_np[nonzero]),
                      key=lambda x: abs(x[1]), reverse=True)[:20]
    print(f"  Top activations:")
    print(f"  {'Idx':>8} {'Act':>10} {'CohenD':>8} {'Keep':>6}")

    kept, removed = 0, 0
    mask = torch.ones_like(features[0])
    for idx, act in top_acts:
        d = d_lookup.get(int(idx), 0.0)
        if keep_mode == "positive":
            keep = d >= min_d
        elif keep_mode == "all_known_good":
            keep = d > -0.3  # remove only strongly anti-search
        else:
            keep = True
        if not keep:
            mask[idx] = 0
            removed += 1
        else:
            kept += 1
        print(f"  {idx:>8} {act:>+10.4f} {d:>+8.3f} {'✓' if keep else '✗':>6}")

    # Apply mask to ALL features (not just top-20)
    for idx in nonzero:
        d = d_lookup.get(int(idx), 0.0)
        if keep_mode == "positive" and d < min_d:
            mask[idx] = 0
        elif keep_mode == "all_known_good" and d < -0.3:
            mask[idx] = 0

    features_filtered = features * mask.unsqueeze(0)
    n_kept = (features_filtered[0] != 0).sum().item()
    print(f"\n  Kept {n_kept}/{len(nonzero)} features after filtering")

    # Decode back to direction space
    with torch.no_grad():
        # Manual decode: x_hat = features_filtered @ decoder.weight.T + bias
        reconstructed = features_filtered @ sae.decoder.weight.T
        if hasattr(sae, 'b_dec'):
            reconstructed += sae.b_dec

    result = reconstructed[0].cpu().numpy().astype(np.float32)
    norm = float(np.linalg.norm(result))
    rms = float(np.sqrt(np.mean(result ** 2)))
    print(f"  Projected direction norm: {norm:.4f}, RMS: {rms:.6f}")

    # Compare with original CAA
    caa_norm = float(np.linalg.norm(caa_direction))
    cosine = float(np.dot(result.flatten(), caa_direction.flatten()) / (norm * caa_norm + 1e-10))
    print(f"  Cosine with original CAA: {cosine:.4f}")

    return result, {"n_nonzero_original": len(nonzero), "n_kept": n_kept,
                    "cosine_with_caa": cosine}


def save_direction(vec, output_path, method, metadata):
    """Save direction as .npz compatible with load_direction()."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out), decision_direction=vec, layer=11, method=method, **metadata)
    norm = float(np.linalg.norm(vec))
    rms = float(np.sqrt(np.mean(vec ** 2)))
    print(f"  Saved to {out}  (norm={norm:.4f}, rms={rms:.6f})")


def main():
    parser = argparse.ArgumentParser(description="Build SAE composite direction")
    parser.add_argument("--sae-path", required=True, help="Path to ae.pt")
    parser.add_argument("--feature-analysis", required=True,
                        help="JSON from sae_feature_steering.py")
    parser.add_argument("--caa-direction", default=None,
                        help="CAA direction .npz for SAE-projected approach")
    parser.add_argument("--output-dir",
                        default="steering/directions")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top features for composite")
    parser.add_argument("--min-cohens-d", type=float, default=0.4,
                        help="Min |Cohen's d| for feature inclusion")
    parser.add_argument("--weighting", default="cohens_d",
                        choices=["cohens_d", "d_times_diff", "uniform"])
    args = parser.parse_args()

    print(f"=== SAE Composite Direction Builder ===")
    print(f"  Time: {datetime.now().isoformat()}")

    # Load feature analysis
    with open(args.feature_analysis) as f:
        analysis = json.load(f)
    print(f"  Features from: {args.feature_analysis}")
    print(f"  N_search={analysis['n_search']}, N_finish={analysis['n_finish']}")

    # Load SAE
    sae = load_sae(args.sae_path)
    out_dir = Path(args.output_dir)

    # === Approach A: Composite Feature Direction ===
    print("\n" + "=" * 60)
    print("  APPROACH A: Composite Feature Direction")
    print("=" * 60)

    composite, feature_info = build_composite_direction(
        sae, analysis, top_n=args.top_n, min_d=args.min_cohens_d,
        weighting=args.weighting,
    )
    meta_a = {
        "n_features": len(feature_info),
        "weighting": args.weighting,
        "min_cohens_d": args.min_cohens_d,
        "top_n": args.top_n,
        "timestamp": datetime.now().isoformat(),
    }
    path_a = out_dir / "direction_sae_composite.npz"
    save_direction(composite, path_a, "sae_composite_feature", meta_a)

    # === Approach B: SAE-Projected CAA (if CAA direction provided) ===
    if args.caa_direction:
        print("\n" + "=" * 60)
        print("  APPROACH B: SAE-Projected CAA")
        print("=" * 60)

        from steering.directions import load_direction
        caa_vec, caa_meta = load_direction(args.caa_direction)
        print(f"  CAA direction: {args.caa_direction}")
        print(f"  CAA norm={caa_meta['norm']:.4f}, rms={caa_meta['rms']:.6f}")

        projected, proj_info = build_sae_projected_caa(
            sae, caa_vec, analysis, min_d=0.0, keep_mode="positive"
        )
        meta_b = {
            "caa_source": args.caa_direction,
            "filter_mode": "positive_cohens_d",
            "timestamp": datetime.now().isoformat(),
            **proj_info,
        }
        path_b = out_dir / "direction_sae_projected_caa.npz"
        save_direction(projected, path_b, "sae_projected_caa", meta_b)

        # Also try less aggressive filtering
        projected2, proj_info2 = build_sae_projected_caa(
            sae, caa_vec, analysis, min_d=-0.3, keep_mode="all_known_good"
        )
        meta_b2 = {
            "caa_source": args.caa_direction,
            "filter_mode": "all_known_good_d>-0.3",
            "timestamp": datetime.now().isoformat(),
            **proj_info2,
        }
        path_b2 = out_dir / "direction_sae_projected_caa_relaxed.npz"
        save_direction(projected2, path_b2, "sae_projected_caa_relaxed", meta_b2)

    print("\n=== Done! ===")
    print(f"Generated directions:")
    print(f"  A. Composite:  {path_a}")
    if args.caa_direction:
        print(f"  B1. Projected (strict):  {path_b}")
        print(f"  B2. Projected (relaxed): {path_b2}")


if __name__ == "__main__":
    main()

