#!/usr/bin/env python3
"""
Robustness checks on narrative-bypass mode in R1 v3.

Questions answered:
  Q1. Does narrative classification flip the evidence-sensitivity direction?
      → Try 3 classification schemes, check Δ2ndSR under each.
  Q2. Does margin-action alignment survive if narrative is counted as stop?
      → Confusion matrix: sign(margin_B) vs {Action:search, Final+The:stop}.
  Q3. Are narrative samples different on EM accuracy?
      → If narrative has much worse EM, it's "give up"; if comparable, it's style.
  Q4. Is narrative triggered by specific samples (cross-condition consistency)?
      → Is the same sample_id narrative in both 1SF and 2SF?
"""
import json
import numpy as np
from collections import Counter

V3 = "results/sf_counterfactual_r1_v3"


def load(cond):
    return [json.loads(l) for l in open(f"{V3}/r1_{cond}_trajectories_v3.jsonl")]


def classify_behavior(r):
    """Map first_action_token to {search, stop, unknown}."""
    tok = r.get("first_action_token")
    if tok is None:
        return "unknown"
    t = tok.strip().lower()
    if t.startswith("action"):
        return "search"
    if t.startswith("final") or t == "final":
        return "stop"
    # Anything else (The, K, Dan, ...) = narrative/prose start = behaviorally stop
    return "stop_narrative"


def main():
    rows_1 = load("1sf")
    rows_2 = load("2sf")

    print("=" * 78)
    print("Q1. CLASSIFICATION ROBUSTNESS — does narrative flip Δ2ndSR?")
    print("=" * 78)
    schemes = {
        "A: Action=search, Final+narrative=stop (paper's plan)":
            lambda r: {"search", "stop_narrative", "stop"}.intersection({classify_behavior(r)}),
        "B: Action=search, Final only=stop, narrative=excluded":
            lambda r: classify_behavior(r) != "stop_narrative" and classify_behavior(r) != "unknown",
        "C: v2-parse-style (full-gen Final-priority, BUGGY reference)":
            lambda r: r.get("action_type_v2"),
    }

    for name, _ in schemes.items():
        print(f"\n  Scheme {name}")
        for cond, rows in [("1SF", rows_1), ("2SF", rows_2)]:
            if "v2-parse" in name:
                n = len(rows)
                s = sum(1 for r in rows if r["action_type_v2"] == "search")
                st = sum(1 for r in rows if r["action_type_v2"] == "stop")
                print(f"    {cond}: search={s}/{n}={s/n*100:.1f}%  stop={st}/{n}={st/n*100:.1f}%")
            elif "Scheme B" in name or "narrative=excluded" in name:
                # denominator excludes narrative & unknown
                valid = [r for r in rows if classify_behavior(r) in ("search","stop")]
                s = sum(1 for r in valid if classify_behavior(r) == "search")
                print(f"    {cond}: search={s}/{len(valid)}={s/len(valid)*100:.1f}%  (narrative excluded, N={len(valid)})")
            else:
                n = len(rows)
                s = sum(1 for r in rows if classify_behavior(r) == "search")
                st = sum(1 for r in rows if classify_behavior(r) in ("stop","stop_narrative"))
                print(f"    {cond}: search={s}/{n}={s/n*100:.1f}%  stop(incl.narr)={st}/{n}={st/n*100:.1f}%")

    # Δ under each
    print(f"\n  Δ2ndSR under each scheme:")
    for name, _ in schemes.items():
        if "v2-parse" in name:
            s1 = sum(1 for r in rows_1 if r["action_type_v2"] == "search") / len(rows_1)
            s2 = sum(1 for r in rows_2 if r["action_type_v2"] == "search") / len(rows_2)
        elif "narrative=excluded" in name:
            v1 = [r for r in rows_1 if classify_behavior(r) in ("search","stop")]
            v2 = [r for r in rows_2 if classify_behavior(r) in ("search","stop")]
            s1 = sum(1 for r in v1 if classify_behavior(r)=="search")/len(v1)
            s2 = sum(1 for r in v2 if classify_behavior(r)=="search")/len(v2)
        else:
            s1 = sum(1 for r in rows_1 if classify_behavior(r) == "search") / len(rows_1)
            s2 = sum(1 for r in rows_2 if classify_behavior(r) == "search") / len(rows_2)
        print(f"    {name[:55]:<55s}  Δ = {(s1-s2)*100:+.1f}pp")

    print("\n" + "=" * 78)
    print("Q2. MARGIN-ACTION ALIGNMENT incl. narrative (narrative→stop)")
    print("=" * 78)
    for cond, rows in [("1SF", rows_1), ("2SF", rows_2)]:
        valid = [r for r in rows if r.get("margin_B_true") is not None]
        # Treat narrative as stop for this check
        for name, behav_map in [("Action vs Final only (template responders)",
                                 {"search":"search","stop":"stop"}),
                                ("Action vs Final+narrative (all responders)",
                                 {"search":"search","stop":"stop","stop_narrative":"stop"})]:
            subset = [r for r in valid if classify_behavior(r) in behav_map]
            tp = sum(1 for r in subset if r["margin_B_true"] > 0 and behav_map[classify_behavior(r)]=="search")
            tn = sum(1 for r in subset if r["margin_B_true"] <= 0 and behav_map[classify_behavior(r)]=="stop")
            fp = sum(1 for r in subset if r["margin_B_true"] > 0 and behav_map[classify_behavior(r)]=="stop")
            fn = sum(1 for r in subset if r["margin_B_true"] <= 0 and behav_map[classify_behavior(r)]=="search")
            acc = (tp+tn)/max(len(subset),1)
            print(f"  {cond} — {name}")
            print(f"    n={len(subset)}  TP={tp} TN={tn} FP={fp} FN={fn}  accuracy={acc*100:.1f}%")

    print("\n" + "=" * 78)
    print("Q3. EM accuracy by decision mode (among samples that produced a final_answer)")
    print("=" * 78)
    for cond, rows in [("1SF", rows_1), ("2SF", rows_2)]:
        modes = {"Action→eventual answer": [], "Final (template)": [], "narrative (The...)": []}
        for r in rows:
            if r.get("em") is None: continue
            b = classify_behavior(r)
            if b == "search": modes["Action→eventual answer"].append(r["em"])
            elif b == "stop": modes["Final (template)"].append(r["em"])
            elif b == "stop_narrative": modes["narrative (The...)"].append(r["em"])
        print(f"  {cond}:")
        for m, vals in modes.items():
            if vals:
                print(f"    {m:<32s}  EM={np.mean(vals)*100:.1f}%  (N={len(vals)})")
            else:
                print(f"    {m:<32s}  N=0")

    print("\n" + "=" * 78)
    print("Q4. NARRATIVE CONSISTENCY — same sample narrative in both conditions?")
    print("=" * 78)
    narr_1 = {r["sample_id"] for r in rows_1 if classify_behavior(r) == "stop_narrative"}
    narr_2 = {r["sample_id"] for r in rows_2 if classify_behavior(r) == "stop_narrative"}
    both = narr_1 & narr_2
    only_1 = narr_1 - narr_2
    only_2 = narr_2 - narr_1
    print(f"  narrative in 1SF only: {len(only_1)}")
    print(f"  narrative in 2SF only: {len(only_2)}")
    print(f"  narrative in BOTH:     {len(both)}")
    print(f"  overlap rate: {len(both)}/{min(len(narr_1),len(narr_2))} = "
          f"{len(both)/max(min(len(narr_1),len(narr_2)),1)*100:.1f}%")
    # Expected-by-chance overlap if narrative is evidence-independent:
    # p(narr|1sf) * p(narr|2sf) * N
    N = len(rows_1)
    exp = (len(narr_1)/N) * (len(narr_2)/N) * N
    print(f"  expected overlap if independent: {exp:.1f}")
    print(f"  observed/expected ratio: {len(both)/max(exp,0.1):.2f}x "
          f"({'sample-specific' if len(both) > 1.5*exp else 'roughly independent'})")


if __name__ == "__main__":
    main()
