#!/usr/bin/env python3
"""tmc.scripts.e2e_agent.steering.directions

Direction vector loading and random control generation.

Best-practice safety note (research rigor): direction files with suffix "_v2"
are deprecated due to historical layer-indexing semantic drift. Loading them is
blocked by default to prevent accidental misuse.
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


_ALLOW_DEPRECATED_ENV = "JES_ALLOW_DEPRECATED_DIRECTIONS"


def _is_deprecated_direction_path(path: Path) -> bool:
    # Block legacy v2 direction files where the stem *ends* with "_v2"
    # (e.g. direction_search_v2.npz, direction_search_post_v2.npz).
    # Files where "_v2" appears mid-name (e.g. ..._v2_hook_fixed.npz) are NOT legacy.
    stem = path.stem.lower()  # filename without .npz extension
    return stem.endswith("_v2")


def load_direction(
    path: str,
    key: str = "decision_direction",
    normalize_rms: Optional[float] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Load direction vector from NPZ file, optionally normalizing to a target RMS.

    **Why normalize?**  The JES alpha formula is ``α = ρ × (hidden_rms / direction_rms)``.
    If two directions have different RMS values, the *same* ρ produces different α values,
    which means ``alpha_max`` clips them at different effective ρ budgets — making
    comparisons unfair.  Normalizing all directions to the same RMS (e.g. 1.0) ensures
    that ρ maps to the same α for every direction, so ``alpha_max`` and ``max_rho``
    constrain them identically.

    Args:
        path: Path to NPZ file
        key: Key name for direction array
        normalize_rms: If not None, rescale the loaded direction so that its
            element-wise RMS equals this value.  Recommended: ``1.0`` (unit RMS).
            The direction's *orientation* is preserved; only its scale changes.

    Returns:
        (direction_array, metadata_dict)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
			f"Direction file not found: {path}\n"
			f"Best-practice defaults in this repo:\n"
			f"  - PopQA/search    -> steering/directions/direction_search_v3.npz\n"
			f"  - GSM8K/MATH/calc -> steering/directions/direction_calculator_v1.npz\n"
			f"(If you are running older scripts, update their --direction-path accordingly.)"
        )

    if _is_deprecated_direction_path(path) and os.environ.get(_ALLOW_DEPRECATED_ENV, "0") not in ("1", "true", "TRUE"):
        raise ValueError(
            f"Deprecated direction file detected: {path}\n"
            f"This repo now defaults to tool-specific, corrected directions:\n"
            f"  - PopQA/search    -> steering/directions/direction_search_v3.npz\n"
            f"  - GSM8K/MATH/calc -> steering/directions/direction_calculator_v1.npz\n"
            f"If you *must* reproduce legacy v2 runs, set {_ALLOW_DEPRECATED_ENV}=1 explicitly."
        )

    data = np.load(path, allow_pickle=True)

    if key not in data:
        available = list(data.keys())
        raise KeyError(
            f"Key '{key}' not found in {path}. Available keys: {available}"
        )

    direction = data[key].astype(np.float32)

    # Record original stats before any normalization
    original_norm = float(np.linalg.norm(direction))
    original_rms = float(np.sqrt(np.mean(direction ** 2)))

    # --- RMS normalization ---
    if normalize_rms is not None:
        if original_rms < 1e-12:
            raise ValueError(
                f"Direction from {path} has near-zero RMS ({original_rms:.2e}); "
                f"cannot normalize.  The file may be corrupted."
            )
        scale = normalize_rms / original_rms
        direction = direction * scale

    # Final stats (after normalization, if any)
    final_norm = float(np.linalg.norm(direction))
    final_rms = float(np.sqrt(np.mean(direction ** 2)))

    metadata = {
        "path": str(path),
        "key": key,
        "dim": direction.shape[-1],
        # Original (on-disk) statistics
        "original_norm": original_norm,
        "original_rms": original_rms,
        # Post-normalization statistics (== original if normalize_rms is None)
        "norm": final_norm,
        "rms": final_rms,
        "normalize_rms": normalize_rms,
    }

    return direction, metadata


def generate_random_orthogonal_direction(
    reference_direction: np.ndarray,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a random direction orthogonal to the reference direction.
    Used for control experiments to verify direction specificity.
    
    Args:
        reference_direction: The decision direction to be orthogonal to
        seed: Random seed for reproducibility
        
    Returns:
        Random direction with same norm as reference, orthogonal to it
    """
    if seed is not None:
        np.random.seed(seed)
    
    dim = reference_direction.shape[-1]
    ref_norm = np.linalg.norm(reference_direction)
    
    if ref_norm < 1e-10:
        raise ValueError("Reference direction has near-zero norm")
    
    # Generate random vector
    random_vec = np.random.randn(dim).astype(np.float32)
    
    # Orthogonalize: remove component along reference direction
    ref_unit = reference_direction / ref_norm
    projection = np.dot(random_vec, ref_unit) * ref_unit
    orthogonal = random_vec - projection
    
    # Normalize to same norm as reference
    orth_norm = np.linalg.norm(orthogonal)
    if orth_norm < 1e-10:
        # Extremely unlikely, but handle it
        raise ValueError("Random vector was parallel to reference (try different seed)")
    
    orthogonal = orthogonal * (ref_norm / orth_norm)
    
    return orthogonal.astype(np.float32)


def save_random_direction(
    reference_path: str,
    output_path: str,
    seed: int = 42,
    reference_key: str = "decision_direction",
    output_key: str = "random_orthogonal_direction"
) -> dict:
    """
    Generate and save a random orthogonal direction for control experiments.
    
    Args:
        reference_path: Path to reference direction NPZ
        output_path: Path to save random direction NPZ
        seed: Random seed
        reference_key: Key for reference direction
        output_key: Key for output direction
        
    Returns:
        Metadata dict
    """
    ref_direction, ref_meta = load_direction(reference_path, reference_key)
    random_direction = generate_random_orthogonal_direction(ref_direction, seed)
    
    # Verify orthogonality
    dot_product = np.dot(ref_direction.flatten(), random_direction.flatten())
    ref_norm = np.linalg.norm(ref_direction)
    rand_norm = np.linalg.norm(random_direction)
    cosine = dot_product / (ref_norm * rand_norm + 1e-10)
    
    metadata = {
        "reference_path": reference_path,
        "reference_key": reference_key,
        "seed": seed,
        "dim": random_direction.shape[-1],
        "norm": float(rand_norm),
        "reference_norm": float(ref_norm),
        "cosine_similarity": float(cosine),  # Should be ~0
        "is_orthogonal": abs(cosine) < 0.01,
    }
    
    np.savez(
        output_path,
        **{output_key: random_direction},
        metadata=metadata
    )
    
    print(f"Saved random orthogonal direction to {output_path}")
    print(f"  Cosine similarity with reference: {cosine:.6f} (should be ~0)")
    print(f"  Norm: {rand_norm:.4f} (reference: {ref_norm:.4f})")
    
    return metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate random orthogonal direction")
    parser.add_argument("--reference", required=True, help="Reference direction NPZ")
    parser.add_argument("--output", required=True, help="Output NPZ path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    save_random_direction(args.reference, args.output, args.seed)

