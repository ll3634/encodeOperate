"""Quick inspector for direction npz files."""
import numpy as np

paths = [
    "results/phase1_probe/probe_direction_l20.npz",
    "steering/directions/direction_search_v3_layer20.npz",
    "steering/directions/direction_decomp_perp_layer20.npz",
    "steering/directions/direction_decomp_parallel_layer20.npz",
    "steering/directions/direction_decomp_full_layer20.npz",
    "steering/directions/direction_probe_layer20.npz",
]
for p in paths:
    try:
        d = np.load(p)
        print(p)
        for k in d.keys():
            v = d[k]
            if hasattr(v, "shape") and v.ndim > 0:
                norm = float(np.linalg.norm(v.flatten()))
                print(f"  {k}: shape={v.shape} dtype={v.dtype} norm={norm:.3f}")
            else:
                print(f"  {k}: {v}")
    except Exception as e:
        print(p, "ERR:", e)

# Also: cosine between probe direction and search direction
print("\n=== cosine sanity ===")
ev = np.load("results/phase1_probe/probe_direction_l20.npz")
sr = np.load("steering/directions/direction_search_v3_layer20.npz")
ev_v = next(ev[k].flatten() for k in ev.keys() if hasattr(ev[k], "shape") and ev[k].ndim >= 1 and ev[k].size > 100)
sr_v = next(sr[k].flatten() for k in sr.keys() if hasattr(sr[k], "shape") and sr[k].ndim >= 1 and sr[k].size > 100)
ev_v = ev_v / np.linalg.norm(ev_v)
sr_v = sr_v / np.linalg.norm(sr_v)
print(f"cos(evidence_l20, search_v3_l20) = {float(np.dot(ev_v, sr_v)):.4f}")
print(f"  expected per CLAUDE.md: -0.0135")
