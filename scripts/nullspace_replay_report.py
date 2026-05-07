#!/usr/bin/env python3
"""Re-render rotation_report.md from cached per_direction_results.json.

No GPU needed.  Used after refining the verdict logic.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nullspace_report import write_report

OUT = Path("results/evidence_erasure_test/nullspace_rotation")
src = OUT / "per_direction_results.json"
fig = json.loads(src.read_text())
# write_report only needs (out_dir, figure, meta, max_dev, t0_ok, base, measured).
# The latter two are unused by the new verdict path; pass empties.
meta = {
    "c": fig["constant_cos"],
    "sqrt_one_minus_c2": float(np.sqrt(1.0 - fig["constant_cos"] ** 2)),
    "E_perp_norm": float(np.sqrt(1.0 - fig["constant_cos"] ** 2)),
    "cos_raw_target_with_E_perp": {
        "E_to_D3": -0.060079,
        "E_to_D1": +0.008482,
        "E_to_random": +0.000000,
    },
    "cos_orth_target_with_raw_target": {},
}
write_report(OUT, fig, meta,
             max_dev=fig["verification"]["max_cos_deviation"],
             t0_ok=fig["verification"]["theta_0_matches_E"],
             base=np.zeros(0), measured={})
print("[done] re-rendered from cache")
