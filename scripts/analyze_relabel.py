#!/usr/bin/env python3
"""Aggregate the relabel sweep + original generation results.

The relabel files (results/steering_trap/relabel/*.jsonl) were produced with
--skip-gen, so raw_output is empty but margin_label / margin_first_token are
the corrected logprob-based metrics.

Behavioral classification is pulled from the ORIGINAL generation files
(results/steering_trap/{rho020,rho150,sweep_rho-*.jsonl}) whose raw_output
is the actual decoded text.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

RELABEL = Path("results/steering_trap/relabel")
GEN = Path("results/steering_trap")
PAIRS = {
    (j["sample_id"], j["condition_id"]): j
    for j in (json.loads(l) for l in open("results/unsupported_trap/pairs.jsonl"))
}

# ---------- classifier on generated text ----------
SEARCH_RE = re.compile(r"Action:\s*search\b", re.I)
FINAL_RE = re.compile(r"Action:\s*Final\s*Answer\b|Final\s*Answer\s*:", re.I)


def classify(rec: dict) -> str:
    raw = rec["raw_output"] or ""
    pair = PAIRS.get((rec["sample_id"], rec["condition_id"]), {})
    w = (pair.get("W") or "").strip()
    v = (pair.get("V") or "").strip()
    if SEARCH_RE.search(raw):
        return "search"
    # Anything that is not a search action is a commit; decide W vs V by content
    if w and w.lower() in raw.lower():
        return "commit-W"
    if v and v.lower() in raw.lower():
        return "commit-V"
    return "other"


def schema(sid: str) -> str:
    return "_".join(sid.split("_")[:-1])


def load(path: Path):
    return [json.loads(l) for l in open(path)]


# ---------- discover files ----------
RELABEL_FILES = {
    "rho020_all": RELABEL / "rho020_all.jsonl",
    "rho150_controls": RELABEL / "rho150_controls.jsonl",
    "sweep_-0.5": RELABEL / "sweep_rho-0.5.jsonl",
    "sweep_-1.0": RELABEL / "sweep_rho-1.0.jsonl",
    "sweep_-1.5": RELABEL / "sweep_rho-1.5.jsonl",
    "sweep_-2.0": RELABEL / "sweep_rho-2.0.jsonl",
    "sweep_-3.0": RELABEL / "sweep_rho-3.0.jsonl",
}
GEN_FILES = {
    "rho0": GEN / "rho020.jsonl",        # baseline (rho=0 baseline rows) + a3 @ rho=-0.2
    "rho150": GEN / "rho150.jsonl",       # a3, evidence_parallel, random @ rho=-1.5
    "sweep_-0.5": GEN / "sweep_rho-0.5.jsonl",
    "sweep_-1.0": GEN / "sweep_rho-1.0.jsonl",
    "sweep_-1.5": GEN / "sweep_rho-1.5.jsonl",
    "sweep_-2.0": GEN / "sweep_rho-2.0.jsonl",
    "sweep_-3.0": GEN / "sweep_rho-3.0.jsonl",
}
relabel_rows = {k: load(v) for k, v in RELABEL_FILES.items()}
gen_rows = {k: load(v) for k, v in GEN_FILES.items()}


def get_behavior(gen_key, cond, itv):
    rs = [r for r in gen_rows[gen_key]
          if r["condition_id"] == cond and r["intervention"] == itv]
    counts = defaultdict(int)
    for r in rs:
        counts[classify(r)] += 1
    return len(rs), counts


def get_margins(relabel_key, cond, itv):
    rs = [r for r in relabel_rows[relabel_key]
          if r["condition_id"] == cond and r["intervention"] == itv]
    if not rs:
        return 0, float("nan"), float("nan")
    mml = sum(r["margin_label"] for r in rs) / len(rs)
    mmft = sum(r["margin_first_token"] for r in rs) / len(rs)
    return len(rs), mml, mmft


# ---------- 1. rho sweep on Trap-B0 a3 ----------
print("=" * 80)
print("1. Trap-B0 | a3 — rho sweep (behavior from gen, margin from relabel)")
print("=" * 80)
print(f"{'rho':>7s} {'n':>4s} {'search':>7s} {'cW':>4s} {'cV':>4s} {'oth':>4s} "
      f"{'mean ml':>9s} {'mean mft':>9s}")
rows = [("0.00", "rho0", "rho020_all", "baseline"),
        ("-0.20", "rho0", "rho020_all", "a3"),
        ("-0.50", "sweep_-0.5", "sweep_-0.5", "a3"),
        ("-1.00", "sweep_-1.0", "sweep_-1.0", "a3"),
        ("-1.50", "sweep_-1.5", "sweep_-1.5", "a3"),
        ("-2.00", "sweep_-2.0", "sweep_-2.0", "a3"),
        ("-3.00", "sweep_-3.0", "sweep_-3.0", "a3")]
for rho, gkey, rkey, itv in rows:
    n, counts = get_behavior(gkey, "Trap-B0", itv)
    _, mml, mmft = get_margins(rkey, "Trap-B0", itv)
    print(f"{rho:>7s} {n:>4d} {counts['search']:>7d} "
          f"{counts['commit-W']:>4d} {counts['commit-V']:>4d} {counts['other']:>4d} "
          f"{mml:>+9.3f} {mmft:>+9.3f}")


# ---------- 2. True-D0 do-no-harm (a3) ----------
print()
print("=" * 80)
print("2. True-D0 | a3 — do-no-harm")
print("=" * 80)
print(f"{'rho':>7s} {'n':>4s} {'cV':>4s} {'cW':>4s} {'search':>7s} {'oth':>4s} "
      f"{'mean ml':>9s} {'mean mft':>9s}")
for rho, gkey, rkey, itv in rows:
    n, counts = get_behavior(gkey, "True-D0", itv)
    _, mml, mmft = get_margins(rkey, "True-D0", itv)
    print(f"{rho:>7s} {n:>4d} {counts['commit-V']:>4d} {counts['commit-W']:>4d} "
          f"{counts['search']:>7d} {counts['other']:>4d} "
          f"{mml:>+9.3f} {mmft:>+9.3f}")


# ---------- 3. multi-seed random @ rho=-1.5 ----------
print()
print("=" * 80)
print("3. rho=-1.5 on Trap-B0: A3 vs controls (multi-seed random + ev_par)")
print("    (behavior: a3/evidence_parallel/random have gen data;")
print("     random_s17/42/99 have margin_label only [no gen])")
print("=" * 80)
print(f"{'intervention':<22s} {'n':>4s} {'search':>7s} {'cW':>4s} {'mean ml':>9s} {'mean mft':>9s}")

def beh_from_gen(gkey, cond, itv):
    rs = [r for r in gen_rows[gkey]
          if r["condition_id"] == cond and r["intervention"] == itv]
    counts = defaultdict(int)
    for r in rs:
        counts[classify(r)] += 1
    return len(rs), counts

# baseline @ rho=0 (gen + relabel)
n, cnt = beh_from_gen("rho0", "Trap-B0", "baseline")
_, mml, mmft = get_margins("rho020_all", "Trap-B0", "baseline")
print(f"{'baseline(ρ=0)':<22s} {n:>4d} {cnt['search']:>7d} {cnt['commit-W']:>4d} "
      f"{mml:>+9.3f} {mmft:>+9.3f}")

# a3 @ rho=-1.5 (gen + relabel)
n, cnt = beh_from_gen("sweep_-1.5", "Trap-B0", "a3")
_, mml, mmft = get_margins("sweep_-1.5", "Trap-B0", "a3")
print(f"{'a3':<22s} {n:>4d} {cnt['search']:>7d} {cnt['commit-W']:>4d} "
      f"{mml:>+9.3f} {mmft:>+9.3f}")

# controls @ rho=-1.5: evidence_parallel, random have gen (rho150.jsonl); s17/42/99 relabel only
for name in ["evidence_parallel", "random"]:
    n, cnt = beh_from_gen("rho150", "Trap-B0", name)
    _, mml, mmft = get_margins("rho150_controls", "Trap-B0", name)
    print(f"{name:<22s} {n:>4d} {cnt['search']:>7d} {cnt['commit-W']:>4d} "
          f"{mml:>+9.3f} {mmft:>+9.3f}")
for name in ["random_s17", "random_s42", "random_s99"]:
    n, mml, mmft = get_margins("rho150_controls", "Trap-B0", name)
    print(f"{name:<22s} {n:>4d} {'—':>7s} {'—':>4s} "
          f"{mml:>+9.3f} {mmft:>+9.3f}")


# ---------- 4. per-schema rescue rate (Trap-B0, a3) ----------
print()
print("=" * 80)
print("4. Per-schema rescue rate (Trap-B0 | a3, from generation files)")
print("=" * 80)
header = "schema".ljust(30) + "  ρ=0  -0.2  -0.5  -1.0  -1.5  -2.0  -3.0"
print(header)
schemas = sorted({schema(r["sample_id"])
                  for r in gen_rows["rho0"]
                  if r["condition_id"] == "Trap-B0"})
for sch in schemas:
    row = sch.ljust(30)
    for rho, gkey, rkey, itv in rows:
        rs = [r for r in gen_rows[gkey]
              if r["condition_id"] == "Trap-B0"
              and r["intervention"] == itv
              and schema(r["sample_id"]) == sch]
        n_search = sum(1 for r in rs if classify(r) == "search")
        row += f"  {n_search:>2d}/{len(rs):>2d}"
    print(row)


# ---------- 5. paired Δmargin_label at rho=-0.2 ----------
print()
print("=" * 80)
print("5. Paired Δmargin_label at rho=-0.2 (intervention − baseline, relabel file)")
print("=" * 80)
rs_020 = relabel_rows["rho020_all"]
base = {(r["sample_id"], r["condition_id"]): r
        for r in rs_020 if r["intervention"] == "baseline"}
print(f"{'condition':<10s} {'intervention':<20s} {'n':>4s} {'mean Δml':>9s} "
      f"{'n>0':>5s}  {'mean Δmft':>10s}")
for cond in ["Trap-B0", "True-D0"]:
    for itv in ["a3", "evidence_parallel", "random", "random_s17", "random_s42", "random_s99"]:
        deltas_ml = []
        deltas_mft = []
        for r in rs_020:
            if r["intervention"] != itv or r["condition_id"] != cond:
                continue
            b = base.get((r["sample_id"], cond))
            if not b:
                continue
            deltas_ml.append(r["margin_label"] - b["margin_label"])
            deltas_mft.append(r["margin_first_token"] - b["margin_first_token"])
        n = len(deltas_ml)
        if n == 0:
            continue
        mean_ml = sum(deltas_ml) / n
        mean_mft = sum(deltas_mft) / n
        npos = sum(1 for d in deltas_ml if d > 0)
        print(f"{cond:<10s} {itv:<20s} {n:>4d} {mean_ml:>+9.4f} {npos:>5d}  {mean_mft:>+10.4f}")


# ---------- 6. McNemar at rho=-1.5 using gen-file behavior ----------
print()
print("=" * 80)
print("6. McNemar at rho=-1.5 (Trap-B0, gen-file behavior)")
print("=" * 80)
from math import comb

def mcnemar_p(b, c):
    # exact binomial two-sided, minimum of b,c
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 0.0
    for i in range(k + 1):
        p += comb(n, i)
    p *= 2.0 / (2 ** n)
    return min(p, 1.0)

def sampleset_gen(gkey, itv):
    rs = [r for r in gen_rows[gkey]
          if r["condition_id"] == "Trap-B0" and r["intervention"] == itv]
    return {r["sample_id"]: (classify(r) == "search") for r in rs}

a3_map = sampleset_gen("sweep_-1.5", "a3")
base_map = sampleset_gen("rho0", "baseline")

def mcn(treatment_map, control_map, name):
    b = c = 0
    for sid, t in treatment_map.items():
        co = control_map.get(sid)
        if co is None:
            continue
        if t and not co:
            b += 1
        elif co and not t:
            c += 1
    p = mcnemar_p(b, c)
    print(f"  a3 vs {name:<30s}  discord b={b}, c={c}  p={p:.3e}")

mcn(a3_map, base_map, "baseline (ρ=0)")
for name in ["evidence_parallel", "random"]:
    mcn(a3_map, sampleset_gen("rho150", name), name + " (ρ=-1.5)")

print()
print("DONE.")
