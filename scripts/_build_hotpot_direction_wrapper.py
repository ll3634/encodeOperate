"""Build c2-compatible directions.npz from audit2_direction.npz.

Repackages HotpotQA-derived `direction` (already unit-norm) under the
`action_dir` key expected by c2_step1_action_margins_qwen3_32b.py. Sets
`evidence_dir = action_dir` as a placeholder; evidence projections are
ignored downstream.

Convention: L_act = peak_layer (matches the indexing convention used in
cross_model_full.py extraction and in c2_step1's `out.hidden_states[L_act]`
read; see scaling_law_summary.md note on shared off-by-one).
"""
import argparse
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit2-npz", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = np.load(args.audit2_npz, allow_pickle=True)
    direction = src["direction"].astype(np.float32)
    norm = float(np.linalg.norm(direction))
    if abs(norm - 1.0) > 1e-3:
        direction = direction / norm
    L = int(src["peak_layer"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             action_dir=direction,
             evidence_dir=direction,
             L_act=L, L_evi=L,
             cos_action_evidence=1.0,
             evidence_auroc=float("nan"),
             action_quality=float("nan"))
    print(f"[ok] saved {out_path}  L={L}  ||dir||={float(np.linalg.norm(direction)):.6f}")


if __name__ == "__main__":
    main()
