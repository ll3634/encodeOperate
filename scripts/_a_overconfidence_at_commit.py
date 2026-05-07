"""A. Per-example over-confidence at T0 commit=True.

Tests the "Reasoning Trap" prediction that reasoning-RL models commit with
higher first-token confidence than instruct-tuned base models at the same
capacity.

For each (model, dataset) we restrict to T0 samples with contains_W==1 and
report:
  - margin_first_token = lp(Action_first) - lp(Final_first)
      more negative == more confident commit (more mass on Final)
  - lp_Final          (closer to 0  == more confident commit)
  - lp_Action         (more negative  == less mass on alternative)

Then we run a Mann-Whitney U test: Qwen3-32B vs each Qwen2.5 model on
margin_first_token and on lp_Final, restricted to commit-W rows.
"""
import json, statistics as st
from pathlib import Path

EVAL = {
    "Qwen2.5-7B-Instruct":  {
        "hotpotqa": "tmc/scripts/e2e_agent/results/cross_model_extractability/eval_results_qwen2_5_7b.jsonl",
        "musique":  "tmc/scripts/e2e_agent/results/second_benchmark_extractability/qwen/eval.jsonl",
    },
    "Qwen2.5-14B-Instruct": {
        "hotpotqa": "tmc/scripts/e2e_agent/results/qwen_14b_scaling_audit/c1/eval_hotpotqa.jsonl",
        "musique":  "tmc/scripts/e2e_agent/results/qwen_14b_scaling_audit/c1/eval_musique.jsonl",
    },
    "Qwen2.5-32B-Instruct": {
        "hotpotqa": "tmc/scripts/e2e_agent/results/qwen2_5_32b_scale_check/c1/eval_hotpotqa.jsonl",
        "musique":  "tmc/scripts/e2e_agent/results/qwen2_5_32b_scale_check/c1/eval_musique.jsonl",
    },
    "Qwen3-32B": {
        "hotpotqa": "tmc/scripts/e2e_agent/results/qwen3_32b_scale_check/c1/eval_hotpotqa.jsonl",
        "musique":  "tmc/scripts/e2e_agent/results/qwen3_32b_scale_check/c1/eval_musique.jsonl",
    },
}

def commit_W_rows(p):
    return [r for r in (json.loads(l) for l in open(p))
            if r.get("condition") == "T0" and r.get("contains_W")]

# Mann-Whitney U (two-sided, no scipy dependency).
def mann_whitney(x, y):
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0: return None
    combined = [(v, "x") for v in x] + [(v, "y") for v in y]
    combined.sort()
    rs = {"x": 0.0, "y": 0.0}
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j+1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2  # ranks are 1-indexed
        for k in range(i, j+1):
            rs[combined[k][1]] += avg_rank
        i = j + 1
    Ux = rs["x"] - nx * (nx + 1) / 2
    Uy = rs["y"] - ny * (ny + 1) / 2
    U = min(Ux, Uy)
    mu = nx * ny / 2
    sd = (nx * ny * (nx + ny + 1) / 12) ** 0.5
    z = (U - mu) / sd if sd > 0 else 0.0
    # two-sided p via normal approx
    import math
    p = math.erfc(abs(z) / 2 ** 0.5)
    return dict(U=U, z=z, p=p, n_x=nx, n_y=ny, median_x=st.median(x), median_y=st.median(y))

print("="*100)
print("A. Over-confidence at commit (T0, contains_W=True only)")
print("="*100)
print(f"{'model':22s} {'ds':10s} {'n':>3s} | {'mft mean':>10s} {'mft med':>10s} {'mft min':>10s} | {'lpF mean':>10s} {'lpF med':>10s} {'lpA mean':>10s}")
print("-"*120)
data = {}
for m, paths in EVAL.items():
    for ds, p in paths.items():
        rows = commit_W_rows(p)
        if not rows: continue
        mft = [r["margin_first_token"] for r in rows]
        lpF = [r["lp_Final"] for r in rows]
        lpA = [r["lp_Action"] for r in rows]
        data[(m, ds)] = {"mft": mft, "lpF": lpF, "lpA": lpA}
        print(f"{m:22s} {ds:10s} {len(rows):>3d} | "
              f"{st.fmean(mft):>+10.3f} {st.median(mft):>+10.3f} {min(mft):>+10.3f} | "
              f"{st.fmean(lpF):>+10.3f} {st.median(lpF):>+10.3f} {st.fmean(lpA):>+10.3f}")

print()
print("="*100)
print("Mann-Whitney U  (Qwen3-32B vs each Qwen2.5 model)  on commit=W samples only")
print("  H1: Qwen3-32B has MORE NEGATIVE margin_first_token  (more confident commit)")
print("  H1: Qwen3-32B has lp_Final CLOSER TO 0           (higher Final-token prob)")
print("="*100)
for ds in ("hotpotqa", "musique"):
    if ("Qwen3-32B", ds) not in data: continue
    yr = data[("Qwen3-32B", ds)]
    for m in ("Qwen2.5-7B-Instruct","Qwen2.5-14B-Instruct","Qwen2.5-32B-Instruct"):
        if (m, ds) not in data: continue
        xr = data[(m, ds)]
        print(f"\n[{ds}]  {m}  vs  Qwen3-32B")
        for k, label, direction in (("mft","margin_first_token","lower"), ("lpF","lp_Final","higher")):
            r = mann_whitney(xr[k], yr[k])
            if r is None: continue
            sign = "Qwen3 < Qwen2.5" if r["median_y"] < r["median_x"] else "Qwen3 > Qwen2.5"
            print(f"   {label:20s}  med(Qwen2.5)={r['median_x']:+.3f}  med(Qwen3)={r['median_y']:+.3f}  ({sign})  "
                  f"U={r['U']:.1f}  z={r['z']:+.3f}  p={r['p']:.4g}  (n_2.5={r['n_x']}, n_3={r['n_y']})")

print()
print("="*100)
print("Compact histogram of margin_first_token at commit=W per model x dataset")
print("  [bins capped at -25 / +5; commit-end is the negative tail]")
print("="*100)
edges = [-25,-20,-15,-12,-10,-8,-6,-4,-2,0,2,5]
labels = [f"<= {edges[0]}"] + [f"({edges[i-1]:+g},{edges[i]:+g}]" for i in range(1,len(edges))] + [f"> {edges[-1]}"]
for m in ["Qwen2.5-7B-Instruct","Qwen2.5-14B-Instruct","Qwen2.5-32B-Instruct","Qwen3-32B"]:
    for ds in ("hotpotqa","musique"):
        if (m, ds) not in data: continue
        vals = data[(m, ds)]["mft"]
        bins = [0]*(len(edges)+1)
        for v in vals:
            placed = False
            for i,e in enumerate(edges):
                if v <= e:
                    bins[i] += 1; placed = True; break
            if not placed: bins[-1] += 1
        print(f"\n  {m:22s} {ds:8s}  n={len(vals)}  (lower=more confident commit)")
        for lab, c in zip(labels, bins):
            print(f"    {lab:>14s}  {c:>3d}  {'#'*c}")
