#!/usr/bin/env python3
"""
Reverse-engineer the JES (Judgment-Edit-Steer) subspace by analyzing why
random directions outperform carefully extracted probe/CAA/SAE directions.

Key analyses:
  1. Angle analysis: cosine similarity of random2 vs all learned directions
  2. Multi-seed random sweep: how common are "good" random directions?
  3. SAE projection: which features does random2 activate vs learned directions?
  4. PF-risk decomposition: what makes a direction "high pollution"?

Usage:
    cd tmc/scripts/e2e_agent
    python scripts/analyze_jes_subspace.py \
        --sae-path ../../../sae_weights/resid_post_layer_11/trainer_2/ae.pt \
        --baseline-trace results/probe_comparison_n200/baseline_results.jsonl \
        --oracle-trace results/probe_comparison_n200/oracle_results.jsonl
"""

import sys, json, argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_dir(path, key="decision_direction"):
    """Load a direction from .npz and return (unit_vector, metadata_dict)."""
    d = np.load(path, allow_pickle=True)
    vec = d[key].astype(np.float64).flatten()
    meta = {}
    for k in d.keys():
        if k != key:
            try:
                v = d[k]
                if v.ndim == 0:
                    meta[k] = v.item()
            except:
                pass
    unit = vec / (np.linalg.norm(vec) + 1e-30)
    return unit, vec, meta


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def analyze_angles(directions_dict, reference_name="random2"):
    """Compute pairwise cosine similarities, focusing on reference."""
    print("\n" + "=" * 70)
    print(f"  ANGLE ANALYSIS (reference = {reference_name})")
    print("=" * 70)

    ref_unit = directions_dict[reference_name][0]
    names = list(directions_dict.keys())

    # Reference vs all others
    print(f"\n{'Direction':<45} | {'cos(ref)':<10} | {'|cos|':<8} | {'angle°':<8}")
    print("-" * 80)
    for name in names:
        if name == reference_name:
            continue
        other_unit = directions_dict[name][0]
        c = cosine(ref_unit, other_unit)
        angle = np.degrees(np.arccos(np.clip(abs(c), 0, 1)))
        print(f"{name:<45} | {c:>+8.4f}   | {abs(c):>6.4f}   | {angle:>6.1f}°")

    # Pairwise matrix for learned directions only
    learned = [n for n in names if "random" not in n]
    if len(learned) > 1:
        print(f"\n--- Pairwise cosines among learned directions ---")
        print(f"{'':>25}", end="")
        for n in learned:
            short = n[:12]
            print(f" {short:>12}", end="")
        print()
        for n1 in learned:
            print(f"{n1[:25]:<25}", end="")
            u1 = directions_dict[n1][0]
            for n2 in learned:
                u2 = directions_dict[n2][0]
                c = cosine(u1, u2)
                print(f" {c:>+11.3f}", end="")
            print()


def analyze_random_sweep(dim, n_random=200, learned_dirs=None, seed_start=0):
    """Generate many random directions, analyze their distribution."""
    print("\n" + "=" * 70)
    print(f"  MULTI-SEED RANDOM SWEEP (n={n_random}, dim={dim})")
    print("=" * 70)

    cosines_per_learned = {name: [] for name in learned_dirs} if learned_dirs else {}
    norms_random = []

    for i in range(n_random):
        np.random.seed(seed_start + i)
        r = np.random.randn(dim).astype(np.float64)
        r_unit = r / np.linalg.norm(r)

        for name, (u, _, _) in (learned_dirs or {}).items():
            cosines_per_learned[name].append(cosine(r_unit, u))

    # Expected |cos| in high dim ≈ sqrt(2/(π*d))
    expected_abs_cos = np.sqrt(2.0 / (np.pi * dim))
    print(f"\nExpected |cos| for dim={dim}: {expected_abs_cos:.6f}")
    print(f"Expected angle from any fixed direction: ~{np.degrees(np.arccos(expected_abs_cos)):.1f}°")

    if learned_dirs:
        print(f"\n{'Direction':<45} | {'mean|cos|':>10} | {'max|cos|':>10} | {'P(|cos|>2x)':>12}")
        print("-" * 85)
        for name in learned_dirs:
            arr = np.abs(cosines_per_learned[name])
            threshold = 2 * expected_abs_cos
            p_high = np.mean(arr > threshold)
            print(f"{name:<45} | {arr.mean():>10.6f} | {arr.max():>10.6f} | {p_high:>11.1%}")

    return cosines_per_learned


def analyze_sae_projection(sae_path, directions_dict, device="cpu"):
    """Project directions onto SAE feature space."""
    print("\n" + "=" * 70)
    print("  SAE PROJECTION ANALYSIS")
    print("=" * 70)

    import torch
    sae_data = torch.load(sae_path, map_location=device, weights_only=False)

    # Extract decoder weights
    W_dec = None
    if isinstance(sae_data, dict):
        if "W_dec" in sae_data:
            W_dec = sae_data["W_dec"]
        elif "state_dict" in sae_data:
            sd = sae_data["state_dict"]
            for k in sd:
                if "dec" in k.lower() and "weight" in k.lower():
                    W_dec = sd[k]
                    break
        else:
            # Try to find decoder weight by name
            for k, v in sae_data.items():
                if hasattr(v, 'shape') and len(v.shape) == 2:
                    print(f"  Found matrix: {k} shape={v.shape}")
                    if "dec" in k.lower() and W_dec is None:
                        W_dec = v
    else:
        # It's a module
        W_dec = sae_data.W_dec if hasattr(sae_data, 'W_dec') else None

    if W_dec is None:
        print("  Could not find decoder weights, skipping SAE analysis")
        return

    W_dec = W_dec.detach().float().cpu().numpy()
    # decoder.weight may be (d_model, n_features) or (n_features, d_model)
    # We need (n_features, d_model) so each row is a feature direction
    if W_dec.shape[0] < W_dec.shape[1]:
        # (d_model, n_features) → transpose
        W_dec = W_dec.T
    n_features, d_model = W_dec.shape
    print(f"  SAE: {n_features} features × {d_model} dims")

    # Normalize decoder columns
    dec_norms = np.linalg.norm(W_dec, axis=1, keepdims=True)
    W_dec_unit = W_dec / (dec_norms + 1e-10)

    print(f"\n{'Direction':<35} | {'L1 proj':>9} | {'Top feat':>9} | {'Top cos':>9} | "
          f"{'#|cos|>.05':>10} | {'Top 5 features'}")
    print("-" * 120)

    feature_profiles = {}
    for name, (unit, raw, meta) in directions_dict.items():
        if len(unit) != d_model:
            print(f"{name:<35} | dim mismatch ({len(unit)} vs {d_model}), skipping")
            continue

        # Project direction onto each SAE feature
        projections = W_dec_unit @ unit  # (n_features,)
        abs_proj = np.abs(projections)

        l1 = abs_proj.sum()
        top_idx = np.argmax(abs_proj)
        top_cos = abs_proj[top_idx]
        n_significant = np.sum(abs_proj > 0.05)

        top5_idx = np.argsort(abs_proj)[-5:][::-1]
        top5_str = ", ".join(f"f{i}({projections[i]:+.3f})" for i in top5_idx)

        feature_profiles[name] = {
            "projections": projections,
            "top5_idx": top5_idx,
            "top_cos": top_cos,
            "n_significant": n_significant,
        }

        print(f"{name:<35} | {l1:>9.2f} | {top_idx:>9d} | {top_cos:>9.4f} | "
              f"{n_significant:>10d} | {top5_str}")

    # Compare feature overlap between random2 and learned directions
    if "random2" in feature_profiles:
        r2_top = set(feature_profiles["random2"]["top5_idx"])
        print(f"\n--- Feature overlap with random2's top-5 ---")
        for name, prof in feature_profiles.items():
            if name == "random2":
                continue
            other_top = set(prof["top5_idx"])
            overlap = r2_top & other_top
            print(f"  {name}: {len(overlap)}/5 overlap ({overlap if overlap else 'none'})")

    return feature_profiles


def analyze_sign_symmetry(directions_dict, reference_name="random2"):
    """Test if flipping random2's sign changes behavior (asymmetry = directional signal)."""
    print("\n" + "=" * 70)
    print("  SIGN SYMMETRY ANALYSIS")
    print("=" * 70)
    print("  (If random2 is truly 'random', +random2 and -random2 should perform")
    print("   similarly. Asymmetry would indicate directional signal.)")

    ref_unit = directions_dict[reference_name][0]
    print(f"\n  To test: evaluate with -random2 (flip sign) at same rho values.")
    print(f"  Save flipped direction for E2E evaluation.")

    # Save flipped direction
    _, raw, meta = directions_dict[reference_name]
    flipped = -raw.astype(np.float32)
    out_path = Path("steering/directions/direction_random2_flipped.npz")
    np.savez(str(out_path), decision_direction=flipped)
    print(f"  Saved flipped direction: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Analyze JES subspace")
    parser.add_argument("--sae-path", default=None,
                        help="Path to SAE ae.pt for projection analysis")
    parser.add_argument("--baseline-trace", default=None)
    parser.add_argument("--oracle-trace", default=None)
    parser.add_argument("--n-random", type=int, default=200,
                        help="Number of random directions for sweep")
    args = parser.parse_args()

    print("=" * 70)
    print("  JES SUBSPACE ANALYSIS")
    print("  'Why does random2 outperform learned directions?'")
    print("=" * 70)

    # Load all directions
    dir_base = Path("steering/directions")
    direction_files = OrderedDict([
        ("random2", dir_base / "direction_random_control.npz"),
        ("random1_seed101", dir_base / "direction_random_control_1.npz"),
        ("random1_seed202", dir_base / "direction_random_control_2.npz"),
        ("probe_L12_pca", dir_base / "direction_probe_layer12_pca.npz"),
        ("probe_L20", dir_base / "direction_probe_layer20.npz"),
        ("probe_L18_sweep", dir_base / "layer_sweep/direction_probe_layer18.npz"),
        ("mean_diff_L18", dir_base / "layer_sweep/direction_mean_diff_layer18.npz"),
        ("probe_pca_L11", dir_base / "direction_probe_pca.npz"),
        ("probe_mean_diff_L11", dir_base / "direction_probe_mean_diff.npz"),
        ("caa_search_v3", dir_base / "direction_search_v3.npz"),
        ("caa_post_hook_fixed", dir_base / "direction_search_post_runtime_trace_clean_eval200_seed42_bridge_v2_hook_fixed.npz"),
        ("sae_composite", dir_base / "direction_sae_composite.npz"),
        ("sae_projected_caa", dir_base / "direction_sae_projected_caa.npz"),
        ("sae_feature_rank1", dir_base / "direction_sae_feature_rank1_f112115.npz"),
        ("v12_post_scaled", dir_base / "direction_v12_post_scaled.npz"),
    ])

    directions = OrderedDict()
    for name, path in direction_files.items():
        if path.exists():
            try:
                unit, raw, meta = load_dir(path)
                directions[name] = (unit, raw, meta)
                layer = meta.get("layer", "?")
                print(f"  ✓ {name:<35} dim={len(unit)}, norm={np.linalg.norm(raw):.4f}, layer={layer}")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
        else:
            print(f"  - {name}: not found")

    dim = len(directions["random2"][0])

    # === Analysis 1: Angles ===
    analyze_angles(directions, reference_name="random2")

    # === Analysis 2: Multi-seed random sweep ===
    learned_only = OrderedDict(
        (k, v) for k, v in directions.items() if "random" not in k
    )
    analyze_random_sweep(dim, n_random=args.n_random, learned_dirs=learned_only)

    # === Analysis 3: SAE projection ===
    if args.sae_path and Path(args.sae_path).exists():
        analyze_sae_projection(args.sae_path, directions)

    # === Analysis 4: Sign symmetry ===
    flipped_path = analyze_sign_symmetry(directions, reference_name="random2")

    # === Analysis 5: Generate diverse random directions for E2E sweep ===
    print("\n" + "=" * 70)
    print("  GENERATING RANDOM DIRECTIONS FOR E2E SWEEP")
    print("=" * 70)

    ref_norm = np.linalg.norm(directions["random2"][1])
    n_new = 20
    out_dir = Path("steering/directions/random_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_new):
        np.random.seed(1000 + i)
        r = np.random.randn(dim).astype(np.float32)
        r = r / np.linalg.norm(r) * ref_norm
        out_path = out_dir / f"direction_random_seed{1000+i}.npz"
        np.savez(str(out_path), decision_direction=r, seed=1000+i)

    print(f"  Generated {n_new} random directions in {out_dir}/")
    print(f"  Each with norm={ref_norm:.4f} (matching random2)")

    # === Summary ===
    print("\n" + "=" * 70)
    print("  NEXT STEPS")
    print("=" * 70)
    print("  1. Run E2E eval on 5-10 of the new random directions at rho=0.5,1.0")
    print("     to determine if random2 is lucky or representative")
    print("  2. Run E2E eval on -random2 (flipped) to test sign symmetry")
    print("  3. If many randoms work: the effective subspace is wide")
    print("     → Focus on removing PF-causing components from learned directions")
    print("  4. If only random2 works: it's a lucky draw")
    print("     → Focus on understanding what makes it special")


if __name__ == "__main__":
    main()

