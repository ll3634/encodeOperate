#!/usr/bin/env python3
"""Analyze steering-trap probe.

Robust classification (handles format-drift cases like
`Action: Final Answer\nAction Input: <X>` which parse_action misses):
  - if raw contains 'Action: search' -> action='search'
  - else classify by whether raw contains V or W

Outputs:
  - per_intervention table per (condition, rho)
  - paired margin shifts (intervention - baseline)
  - McNemar on search rate A3 vs each control at same rho
  - rho sweep table on Trap-B0
"""
import json, re, glob, math
from collections import defaultdict
from pathlib import Path

ROOT = Path("results/steering_trap")
PAIRS = {(j["sample_id"], j["condition_id"]): j
         for j in (json.loads(l) for l in open("results/unsupported_trap/pairs.jsonl"))}


def classify(r, pair):
    V = pair["V"]; W = pair.get("W") or ""
    raw = r["raw_output"]
    if re.search(r"Action:\s*search\b", raw, re.IGNORECASE):
        return "search", False, False
    cV = V.lower() in raw.lower()
    cW = bool(W) and W.lower() in raw.lower()
    cls = "stop" if (cV or cW) else "other"
    return cls, cV and not cW, cW and not cV


def per_bucket(rs):
    out = defaultdict(list)
    for r in rs:
        out[(r["condition_id"], r["intervention"])].append(r)
    return out


def summarize_bucket(bucket):
    S = CW = CV = O = 0; margins = []; per_sample = {}
    for r in bucket:
        p = PAIRS[(r["sample_id"], r["condition_id"])]
        cls, cv, cw = classify(r, p)
        S += int(cls == "search"); CW += int(cw); CV += int(cv)
        O += int(cls == "other" and not cv and not cw)
        margins.append(r["margin"])
        per_sample[r["sample_id"]] = {"cls": cls, "cv": cv, "cw": cw,
                                       "margin": r["margin"]}
    n = len(bucket)
    mm = sum(margins) / n
    return {"n": n, "search": S, "commits_W": CW, "commits_V": CV, "other": O,
            "mean_margin": mm, "per_sample": per_sample}


def mcnemar(a, b):
    """a[sid] -> 0/1, b[sid] -> 0/1. Returns b01, b10, p (exact two-sided)."""
    b01 = b10 = 0
    for k in a:
        if k in b:
            if a[k] == 0 and b[k] == 1: b01 += 1
            elif a[k] == 1 and b[k] == 0: b10 += 1
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    p = sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** (n - 1)
    return b01, b10, min(1.0, p)


def main():
    # rho=-0.20 main file
    out = {"rho020": {}, "rho150": {}, "trap_sweep": {}, "do_no_harm": {}}

    for tag, path in [("rho020", "rho020.jsonl"), ("rho150", "rho150.jsonl")]:
        rs = [json.loads(l) for l in open(ROOT / path)]
        for k, bkt in per_bucket(rs).items():
            out[tag][f"{k[0]}|{k[1]}"] = summarize_bucket(bkt)

    # Trap-B0 sweep
    out["trap_sweep"]["0.00"] = out["rho020"]["Trap-B0|baseline"]
    out["trap_sweep"]["-0.20"] = out["rho020"]["Trap-B0|a3"]
    for path in sorted(glob.glob(str(ROOT / "sweep_rho*.jsonl"))):
        rho = re.search(r"rho(-?\d+\.\d+)", path).group(1)
        rs = [json.loads(l) for l in open(path)]
        out["trap_sweep"][rho] = summarize_bucket(rs)

    # True-D0 do-no-harm
    out["do_no_harm"]["0.00"] = out["rho020"]["True-D0|baseline"]
    out["do_no_harm"]["-0.20"] = out["rho020"]["True-D0|a3"]
    for rho in ["-0.5", "-1.0"]:
        rs = [json.loads(l) for l in open(ROOT / f"trueD0_rho{rho}.jsonl")]
        out["do_no_harm"][rho] = summarize_bucket(rs)
    out["do_no_harm"]["-1.50"] = out["rho150"]["True-D0|a3"]

    # McNemar on search at rho=-1.5: A3 vs controls on Trap-B0
    a3 = {k: int(v["cls"] == "search")
          for k, v in out["rho150"]["Trap-B0|a3"]["per_sample"].items()}
    base = {k: int(v["cls"] == "search")
            for k, v in out["rho020"]["Trap-B0|baseline"]["per_sample"].items()}
    rnd = {k: int(v["cls"] == "search")
           for k, v in out["rho150"]["Trap-B0|random"]["per_sample"].items()}
    ep = {k: int(v["cls"] == "search")
          for k, v in out["rho150"]["Trap-B0|evidence_parallel"]["per_sample"].items()}
    out["mcnemar_rho150"] = {}
    for name, ctrl in [("vs_baseline_rho0", base), ("vs_random_rho150", rnd),
                       ("vs_evidence_parallel_rho150", ep)]:
        b01, b10, p = mcnemar(ctrl, a3)
        out["mcnemar_rho150"][name] = {"ctrl_to_a3": b01, "a3_to_ctrl": b10, "p": p}

    # Paired margin shifts at rho=-0.20
    out["paired_margin_delta_rho020"] = {}
    for cond in ["Trap-B0", "True-D0"]:
        base_m = {k: v["margin"]
                  for k, v in out["rho020"][f"{cond}|baseline"]["per_sample"].items()}
        for itv in ["a3", "evidence_parallel", "random"]:
            m = {k: v["margin"]
                 for k, v in out["rho020"][f"{cond}|{itv}"]["per_sample"].items()}
            deltas = [m[k] - base_m[k] for k in base_m if k in m]
            n_pos = sum(d > 0 for d in deltas)
            mean = sum(deltas) / len(deltas)
            out["paired_margin_delta_rho020"][f"{cond}|{itv}"] = {
                "n": len(deltas), "mean": mean, "n_positive": n_pos}

    # strip per_sample before writing
    def strip(d):
        if isinstance(d, dict):
            return {k: strip(v) for k, v in d.items() if k != "per_sample"}
        return d
    out_clean = strip(out)

    Path("results/steering_trap/summary.json").write_text(
        json.dumps(out_clean, indent=2, default=float))
    print(json.dumps(out_clean, indent=2, default=float))


if __name__ == "__main__":
    main()
