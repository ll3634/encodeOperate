#!/usr/bin/env python3
"""Diagnose why org_code_city and company_product_country schemas resist steering.

Compares per-schema at rho=-1.5:
  - margin_label (new metric): did the decision-level margin cross 0?
  - W-token length (is W longer, harder to suppress?)
  - W position in prompt (closer to decision point?)
  - P(Final)/P(Action) at the zero-steering baseline (structural bias?)
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

pairs = [json.loads(l) for l in open("results/unsupported_trap/pairs.jsonl")]
pairs = {(p["sample_id"], p["condition_id"]): p for p in pairs}


def schema(sid: str) -> str:
    return "_".join(sid.split("_")[:-1])


# Load relabel @ rho=-1.5 (a3) and rho=0 (baseline)
r150 = [json.loads(l) for l in open("results/steering_trap/relabel/sweep_rho-1.5.jsonl")
        if json.loads(l)["condition_id"] == "Trap-B0"]
r020 = [json.loads(l) for l in open("results/steering_trap/relabel/rho020_all.jsonl")
        if json.loads(l)["condition_id"] == "Trap-B0"]

rbase = {r["sample_id"]: r for r in r020 if r["intervention"] == "baseline"}
r150_a3 = {r["sample_id"]: r for r in r150 if r["intervention"] == "a3"}


print("=" * 92)
print("Per-schema diagnostics at rho=-1.5 (Trap-B0, a3)")
print("=" * 92)
print(f"{'schema':<26s} "
      f"{'n':>3s} {'search':>7s}  "
      f"{'ml_base':>8s} {'ml_a3':>8s} Δml     "
      f"{'lpS_a3':>7s} {'lpF_a3':>7s}  "
      f"{'|W|ch':>5s} {'Wtok':>5s} {'W_pos':>6s}")

by_sch: dict[str, list[dict]] = defaultdict(list)
for sid, rec in r150_a3.items():
    by_sch[schema(sid)].append(rec)

# Need tokenizer for W-token length
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)


def w_info(pair):
    """Return (char_len, token_len, char_position_in_obs)."""
    w = pair.get("W") or ""
    obs = pair.get("obs") or ""
    tl = len(tok.encode(" " + w, add_special_tokens=False)) if w else 0
    pos = obs.lower().rfind(w.lower()) if (w and obs) else -1
    return len(w), tl, pos


# Number of search acts via generation file
gen150 = [json.loads(l) for l in open("results/steering_trap/sweep_rho-1.5.jsonl")
          if json.loads(l)["condition_id"] == "Trap-B0"
          and json.loads(l)["intervention"] == "a3"]
search_re = re.compile(r"Action:\s*search\b", re.I)
gen_search = {r["sample_id"]: bool(search_re.search(r["raw_output"] or ""))
              for r in gen150}

for sch in sorted(by_sch.keys()):
    recs = by_sch[sch]
    n = len(recs)
    searches = sum(gen_search.get(r["sample_id"], False) for r in recs)
    ml_base = sum(rbase[r["sample_id"]]["margin_label"] for r in recs) / n
    ml_a3 = sum(r["margin_label"] for r in recs) / n
    lpS_a3 = sum(r["lp_search_after"] for r in recs) / n
    lpF_a3 = sum(r["lp_Final_after"] for r in recs) / n
    # W stats averaged per schema
    winfos = [w_info(pairs[(r["sample_id"], "Trap-B0")]) for r in recs]
    wchar = sum(i[0] for i in winfos) / n
    wtok = sum(i[1] for i in winfos) / n
    wpos = sum(i[2] for i in winfos) / n
    print(f"{sch:<26s} "
          f"{n:>3d} {searches:>7d}  "
          f"{ml_base:>+8.3f} {ml_a3:>+8.3f} {ml_a3-ml_base:>+7.3f}   "
          f"{lpS_a3:>+7.3f} {lpF_a3:>+7.3f}  "
          f"{wchar:>5.1f} {wtok:>5.1f} {wpos:>6.0f}")


print()
print("=" * 92)
print("Refractory sample detail: org_code_city + company_product_country (a3, rho=-1.5)")
print("=" * 92)
print(f"{'sample_id':<32s} {'W':<18s} {'ml_base':>8s} {'ml_a3':>8s} "
      f"{'lpS_a3':>7s} {'lpF_a3':>7s}")
for sid, rec in sorted(r150_a3.items()):
    sch = schema(sid)
    if sch not in {"org_code_city", "company_product_country"}:
        continue
    pair = pairs[(sid, "Trap-B0")]
    w = pair.get("W") or ""
    mb = rbase[sid]["margin_label"]
    ma = rec["margin_label"]
    print(f"{sid:<32s} {w[:18]:<18s} "
          f"{mb:>+8.3f} {ma:>+8.3f} "
          f"{rec['lp_search_after']:>+7.3f} {rec['lp_Final_after']:>+7.3f}")


print()
print("=" * 92)
print("Prompt structure comparison (first sample of each schema)")
print("=" * 92)
seen = set()
for p in pairs.values():
    if p["condition_id"] != "Trap-B0":
        continue
    sch = schema(p["sample_id"])
    if sch in seen:
        continue
    seen.add(sch)
    w = p.get("W") or ""
    v = p.get("V") or ""
    q = p.get("question") or ""
    obs = p.get("obs") or ""
    print(f"\n--- {sch} | sample={p['sample_id']} ---")
    print(f"  Q: {q}")
    print(f"  W (commit-extractable): {w!r}   V (target): {v!r}")
    print(f"  K (cue/key): {p.get('K')!r}   A (anchor): {p.get('A')!r}")
    print(f"  obs (full):")
    print("    " + obs.replace("\n", "\n    ")[:900])
