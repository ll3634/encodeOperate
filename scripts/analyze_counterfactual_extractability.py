#!/usr/bin/env python3
"""Aggregate Part B counterfactual results into a single summary JSON."""
import json, math
from pathlib import Path
from scipy.stats import binomtest, wilcoxon

D = Path("results/natural_extractability_audit")


def load(name):
    return {r["sample_id"]: r for r in
            (json.loads(l) for l in open(D / f"counterfactual_{name}_results.jsonl"))}


def pair_2x2(T1, T2, ids, key, target):
    a = b = c = d = 0
    for i in ids:
        f1 = int(T1[i][key] == target)
        f2 = int(T2[i][key] == target)
        if f1 == 1 and f2 == 1: a += 1
        elif f1 == 1 and f2 == 0: b += 1
        elif f1 == 0 and f2 == 1: c += 1
        else: d += 1
    return a, b, c, d


def main():
    base, repl, remv, ctrl = load("base"), load("replace"), load("remove"), load("control")
    ids = sorted(set(base) & set(repl) & set(remv) & set(ctrl))
    base_stop_ids = [i for i in ids if base[i]["first_action_token"] == "stop"]

    out = {"paired_n": len(ids), "n_base_stop": len(base_stop_ids), "conditions": {}}

    for tag, T in [("base", base), ("replace_W", repl),
                   ("remove_W", remv), ("irrelevant_control", ctrl)]:
        first = [T[i]["first_action_token"] for i in ids]
        n_stop = sum(1 for x in first if x == "stop")
        n_search = sum(1 for x in first if x == "search")
        n_pf = sum(1 for x in first if x == "parse_fail")

        stops_with_fa = [i for i in ids if T[i]["first_action_token"] == "stop"
                         and T[i]["final_answer"]]
        n_w = sum(1 for i in stops_with_fa if T[i]["contains_W"])
        n_em = sum(1 for i in stops_with_fa if T[i].get("em") == 1)

        out["conditions"][tag] = {
            "n": len(ids),
            "first_action": {"stop": n_stop, "search": n_search, "parse_fail": n_pf},
            "stop_rate": n_stop / len(ids),
            "search_rate": n_search / len(ids),
            "n_stops_with_final_answer": len(stops_with_fa),
            "contains_W_among_stops": n_w,
            "contains_W_rate_among_stops": n_w / len(stops_with_fa) if stops_with_fa else None,
            "em_among_stops": n_em,
            "em_rate_among_stops": n_em / len(stops_with_fa) if stops_with_fa else None,
        }

    # Paired flips on BASE=stop
    out["paired_flips_on_base_stop"] = {}
    for tag, T in [("replace_W", repl), ("remove_W", remv), ("irrelevant_control", ctrl)]:
        flip = sum(1 for i in base_stop_ids if T[i]["first_action_token"] == "search")
        stay = sum(1 for i in base_stop_ids if T[i]["first_action_token"] == "stop")
        other = len(base_stop_ids) - flip - stay
        out["paired_flips_on_base_stop"][tag] = {
            "n": len(base_stop_ids),
            "stop_to_search": flip,
            "stay_stop": stay,
            "other": other,
            "flip_rate": flip / len(base_stop_ids) if base_stop_ids else None,
        }

    # McNemar (exact) treatment vs control on flip-to-search
    out["mcnemar_flip_vs_control"] = {}
    for tag, T in [("replace_W", repl), ("remove_W", remv)]:
        a, b, c, d = pair_2x2(T, ctrl, base_stop_ids, "first_action_token", "search")
        n = b + c
        p = (binomtest(b, n=n, p=0.5, alternative="greater").pvalue
             if n > 0 else float("nan"))
        out["mcnemar_flip_vs_control"][tag] = {
            "treatment_only_search": b, "control_only_search": c,
            "both_search": a, "neither": d, "p_exact_T_greater_C": p,
        }

    # Paired contains_W comparisons (treatment vs base, treatment vs control)
    out["paired_contains_W"] = {}
    base_stop_with_fa = [i for i in base_stop_ids if base[i]["final_answer"]]
    for tag, T in [("replace_W", repl), ("remove_W", remv), ("irrelevant_control", ctrl)]:
        # comparison vs base on BASE=stop (paired):
        b_w = [int(base[i]["contains_W"]) for i in base_stop_with_fa]
        t_w = [int(T[i]["contains_W"]) for i in base_stop_with_fa]
        # discordant pairs: base=1, T=0 (W disappeared) vs base=0, T=1 (W appeared)
        b_to_no = sum(1 for x, y in zip(b_w, t_w) if x == 1 and y == 0)
        no_to_b = sum(1 for x, y in zip(b_w, t_w) if x == 0 and y == 1)
        n_disc = b_to_no + no_to_b
        p_disc = (binomtest(b_to_no, n=n_disc, p=0.5, alternative="greater").pvalue
                  if n_disc > 0 else float("nan"))
        out["paired_contains_W"][tag + "_vs_base"] = {
            "n_paired": len(base_stop_with_fa),
            "base_W_dropped_after_T": b_to_no,
            "T_introduced_W": no_to_b,
            "p_exact_T_drops_more": p_disc,
        }

    # Margin shift (paired) on BASE=stop subset
    out["paired_margin_shift_on_base_stop"] = {}
    for tag, T in [("replace_W", repl), ("remove_W", remv), ("irrelevant_control", ctrl)]:
        diffs = [T[i]["margin"] - base[i]["margin"] for i in base_stop_ids]
        mean = sum(diffs) / len(diffs)
        var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
        sd = math.sqrt(var)
        pos = sum(1 for d in diffs if d > 0)
        try:
            wp = float(wilcoxon(diffs, alternative="greater").pvalue)
        except Exception:
            wp = float("nan")
        out["paired_margin_shift_on_base_stop"][tag] = {
            "n": len(diffs),
            "mean_delta_margin": mean,
            "sd": sd,
            "n_positive": pos,
            "wilcoxon_p_greater_than_0": wp,
        }

    # Treatment vs control margin (paired) — does T shift more than ctrl?
    out["paired_margin_T_minus_C"] = {}
    for tag, T in [("replace_W", repl), ("remove_W", remv)]:
        diffs = [(T[i]["margin"] - base[i]["margin"]) -
                 (ctrl[i]["margin"] - base[i]["margin"]) for i in base_stop_ids]
        # equivalently T - ctrl
        mean = sum(diffs) / len(diffs)
        try:
            wp = float(wilcoxon(diffs, alternative="greater").pvalue)
        except Exception:
            wp = float("nan")
        out["paired_margin_T_minus_C"][tag] = {
            "mean_T_minus_C_margin_shift": mean,
            "wilcoxon_p": wp,
        }

    out_path = D / "counterfactual_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
