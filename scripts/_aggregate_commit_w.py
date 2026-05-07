"""Aggregate commit_W / first_search_rate / mean_ml across the 4 models."""
import json

REF7 = {  # from c1_behavioral_baseline_qwen3_32b.QWEN25_7B_REF
    "hotpotqa": {"N0": (0.00, 0.96,  7.854),
                 "T0": (0.44, 0.56,  2.936),
                 "S0": (1.00, 0.00, -7.651)},
    "musique":  {"N0": (0.02, 0.08, 12.961),
                 "T0": (0.46, 0.06, 10.322),
                 "S0": (0.58, 0.04,  9.349)},
}

FILES = [
    ("Qwen2.5-14B-Instruct",
     "tmc/scripts/e2e_agent/results/qwen_14b_scaling_audit/c1/behavioral_baseline.json"),
    ("Qwen2.5-32B-Instruct",
     "tmc/scripts/e2e_agent/results/qwen2_5_32b_scale_check/c1/behavioral_baseline.json"),
    ("Qwen3-32B",
     "tmc/scripts/e2e_agent/results/qwen3_32b_scale_check/c1/behavioral_baseline.json"),
]

hdr = ["model", "dataset", "N0_cw", "T0_cw", "S0_cw", "dT-N", "T0_fs", "T0_ml"]
print("|" + "|".join(hdr) + "|")
print("|" + "|".join("---" for _ in hdr) + "|")

for ds, cells in REF7.items():
    n, t, s = cells["N0"][0], cells["T0"][0], cells["S0"][0]
    fs, ml  = cells["T0"][1], cells["T0"][2]
    row = ["Qwen2.5-7B-Instruct", ds,
           f"{n:.2f}", f"{t:.2f}", f"{s:.2f}", f"{t-n:+.2f}",
           f"{fs:.2f}", f"{ml:+.2f}"]
    print("|" + "|".join(row) + "|")

for label, p in FILES:
    d = json.load(open(p))
    cells_key = next(k for k in d["datasets"]["hotpotqa"] if k.startswith("cells_"))
    for ds in ("hotpotqa", "musique"):
        c = d["datasets"][ds][cells_key]
        n  = c["N0"]["commit_W"]; t = c["T0"]["commit_W"]; s = c["S0"]["commit_W"]
        fs = c["T0"]["search_rate"]; ml = c["T0"]["mean_ml"]
        row = [label, ds,
               f"{n:.2f}", f"{t:.2f}", f"{s:.2f}", f"{t-n:+.2f}",
               f"{fs:.2f}", f"{ml:+.2f}"]
        print("|" + "|".join(row) + "|")
