#!/usr/bin/env python3
"""Build a derangement (sid -> donor_sid != sid) for the mismatched-donor
locality control over the same N samples used in exp1."""
import argparse, json, random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default="results/anti_cue_tm_n100/pairs.jsonl")
ap.add_argument("--limit", type=int, default=50)
ap.add_argument("--seed", type=int, default=20260426)
ap.add_argument("--out",
                default="results/gemma_circuit_sanity/exp1_residual_sweep_mismatched/donor_map.json")
args = ap.parse_args()

records = [json.loads(l) for l in open(args.pairs)]
sids = sorted(set(r["sample_id"] for r in records))[:args.limit]
need = {("sf", "task_missingness"), ("distractor", "task_missingness")}
have = {s: set() for s in sids}
for r in records:
    if r["sample_id"] in have:
        have[r["sample_id"]].add((r["target"], r["cue"]))
sids = [s for s in sids if need.issubset(have[s])]

rng = random.Random(args.seed)
shuf = sids[:]
for _ in range(50):
    rng.shuffle(shuf)
    if all(s != d for s, d in zip(sids, shuf)):
        break
else:
    raise SystemExit("derangement search failed")
mapping = dict(zip(sids, shuf))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump({"map": mapping, "seed": args.seed, "n": len(mapping)},
          open(args.out, "w"), indent=2)
print(f"[wrote] {args.out}  ({len(mapping)} entries, seed={args.seed})")
