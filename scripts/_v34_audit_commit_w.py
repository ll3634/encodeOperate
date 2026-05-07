"""V3 (N0/S0 boundary consistency) + V4 (margin distribution) audit.

For each (model, dataset) cell we recompute commit_W / search_rate from the
saved per-example jsonl, then for the suspect cell (Qwen2.5-32B MuSiQue T0)
we dump the per-sample margin / action_type / parse_failure breakdown and a
histogram over `margin_label`.
"""
import json, statistics as st
from collections import Counter

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

def cell(rows):
    n = len(rows)
    if n == 0:
        return None
    return dict(
        n=n,
        commit_W=sum(int(r.get("contains_W", 0)) for r in rows)/n,
        search_rate=sum(1 for r in rows if r.get("action_type")=="search")/n,
        stop_rate=sum(1 for r in rows if r.get("action_type")=="stop")/n,
        none_rate=sum(1 for r in rows if r.get("action_type") is None)/n,
        parse_fail=sum(int(r.get("parse_failure", 0)) for r in rows)/n,
    )

print("="*100)
print("V3 -- N0 / T0 / S0 boundary consistency")
print("  N0 expectation: commit_W ~ 0   (no doc, model SHOULD search)")
print("  S0 expectation: commit_W ~ 1   (true evidence given, model SHOULD commit to true ans which == W in S0)")
print("="*100)
hdr = f"{'model':22s} {'ds':8s} {'cond':4s} {'n':>3s} | {'commit_W':>9s} {'search':>7s} {'stop':>6s} {'none':>6s} {'pfail':>6s}"
print(hdr); print("-"*len(hdr))
for model, paths in EVAL.items():
    for ds, p in paths.items():
        recs = [json.loads(l) for l in open(p)]
        by = {c: [r for r in recs if r.get("condition")==c] for c in ("N0","T0","S0")}
        for c in ("N0","T0","S0"):
            s = cell(by[c])
            if s is None:
                continue
            mark = ""
            if c == "N0" and s["commit_W"] > 0.20: mark = "  <-- N0 HIGH"
            if c == "S0" and s["commit_W"] < 0.50: mark = "  <-- S0 LOW"
            print(f"{model:22s} {ds:8s} {c:4s} {s['n']:>3d} | {s['commit_W']:>9.3f} {s['search_rate']:>7.3f} {s['stop_rate']:>6.3f} {s['none_rate']:>6.3f} {s['parse_fail']:>6.3f}{mark}")
        print()

print("="*100)
print("V4 -- per-sample margin_label distribution at T0 (Qwen2.5-32B-Instruct, both datasets)")
print("  margin_label > 0 == prefers 'Action: search' (continue searching)")
print("  margin_label < 0 == prefers 'Final Answer:' (commit)")
print("="*100)

def hist(vals, edges=(-50,-20,-10,-5,-2,0,2,5,10,20,50)):
    bins = [0]*(len(edges)+1)
    for v in vals:
        placed = False
        for i,e in enumerate(edges):
            if v <= e:
                bins[i]+=1; placed=True; break
        if not placed:
            bins[-1]+=1
    labels = [f"<= {edges[0]}"] + [f"({edges[i-1]:+g},{edges[i]:+g}]" for i in range(1,len(edges))] + [f"> {edges[-1]}"]
    return list(zip(labels, bins))

for model in ("Qwen2.5-7B-Instruct","Qwen2.5-14B-Instruct","Qwen2.5-32B-Instruct","Qwen3-32B"):
    for ds in ("hotpotqa","musique"):
        recs = [json.loads(l) for l in open(EVAL[model][ds])]
        t0 = [r for r in recs if r.get("condition")=="T0"]
        if not t0: continue
        ml = [r["margin_label"] for r in t0 if r.get("margin_label") is not None]
        cw = sum(int(r.get("contains_W",0)) for r in t0)/len(t0)
        print(f"\n--- {model:22s} {ds} T0  n={len(t0)} commit_W={cw:.2f} ---")
        print(f"  margin_label   mean={st.fmean(ml):+.2f}  median={st.median(ml):+.2f}  min={min(ml):+.2f}  max={max(ml):+.2f}")
        # split by commit
        ml_W   = [r["margin_label"] for r in t0 if r.get("contains_W")]
        ml_nW  = [r["margin_label"] for r in t0 if not r.get("contains_W")]
        if ml_W:  print(f"    commit=W  (n={len(ml_W)}):  mean={st.fmean(ml_W):+.2f}  median={st.median(ml_W):+.2f}")
        if ml_nW: print(f"    commit=~W (n={len(ml_nW)}): mean={st.fmean(ml_nW):+.2f}  median={st.median(ml_nW):+.2f}")
        for lab,c in hist(ml):
            bar = "#"*c
            print(f"    {lab:>14s} {c:>3d}  {bar}")

print("\n" + "="*100)
print("V4b -- action_type x contains_W cross-tab at T0 (all 4 models, both datasets)")
print("="*100)
for model in EVAL:
    for ds in ("hotpotqa","musique"):
        recs = [json.loads(l) for l in open(EVAL[model][ds])]
        t0 = [r for r in recs if r.get("condition")=="T0"]
        ct = Counter()
        for r in t0:
            at = r.get("action_type") or "none"
            cw = "W" if r.get("contains_W") else "~W"
            ct[(at,cw)] += 1
        cells_str = ", ".join(f"{at}/{cw}={n}" for (at,cw),n in sorted(ct.items()))
        print(f"  {model:22s} {ds:8s}  {cells_str}")
