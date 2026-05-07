"""V1 -- raw output sanity sampling for c1 baselines.

Buckets (all from T0 condition):
  V1a  Qwen2.5-32B-Instruct  MuSiQue   commit_W=False  -> 10 samples (the 0.22 cell)
  V1b  Qwen2.5-32B-Instruct  MuSiQue   commit_W=True   ->  5 samples
  V1c  Qwen2.5-32B-Instruct  HotpotQA  commit_W=True   ->  5 samples (the 0.82 cell)
  V1d  Qwen3-32B             HotpotQA  commit_W=True   ->  5 samples (the 1.00 cell)

For each sample we print:
  - condition, action_type, contains_W, parse_failure
  - margin_first_token, margin_label, lp_search_seq - lp_final_seq sanity
  - len(raw_output) and last 40 chars (to check for truncation; raw is capped at 400 chars by run_one)
  - first 350 chars of raw_output
"""
import json, random
random.seed(0)

EVAL = {
    "32B-Instruct": {
        "hotpotqa": "tmc/scripts/e2e_agent/results/qwen2_5_32b_scale_check/c1/eval_hotpotqa.jsonl",
        "musique":  "tmc/scripts/e2e_agent/results/qwen2_5_32b_scale_check/c1/eval_musique.jsonl",
    },
    "Qwen3-32B": {
        "hotpotqa": "tmc/scripts/e2e_agent/results/qwen3_32b_scale_check/c1/eval_hotpotqa.jsonl",
    },
}

def show(label, recs, k):
    recs = list(recs)
    print(f"\n{'#'*100}\n# {label}  (showing min({k}, {len(recs)})={min(k, len(recs))} of {len(recs)})\n{'#'*100}")
    for i, r in enumerate(recs[:k]):
        raw = r.get("raw_output", "") or ""
        last40 = raw[-40:].replace("\n", "\\n")
        body = raw[:350].replace("\n", "\n      ")
        print(f"\n--- [{i+1}/{k}]  sample_id={r['sample_id']}  cond={r['condition']}  schema={r.get('schema_type')} ---")
        print(f"  W={r.get('candidate_W')!r}")
        print(f"  action_type={r.get('action_type')}  contains_W={r.get('contains_W')}  parse_failure={r.get('parse_failure')}")
        print(f"  margin_first_token={r.get('margin_first_token'):+.3f}  margin_label(seq)={r.get('margin_label'):+.3f}")
        print(f"  lp_Action={r.get('lp_Action'):+.3f}  lp_Final={r.get('lp_Final'):+.3f}")
        print(f"  raw_output: len={len(raw)}  last40={last40!r}")
        print(f"  raw_output (first 350 chars):\n      {body}")

# ---------- V1a: 32B Mu T0 commit=False ----------
mu = [json.loads(l) for l in open(EVAL["32B-Instruct"]["musique"])]
mu_t0_nW = [r for r in mu if r["condition"] == "T0" and not r.get("contains_W")]
mu_t0_W  = [r for r in mu if r["condition"] == "T0" and     r.get("contains_W")]
random.shuffle(mu_t0_nW); random.shuffle(mu_t0_W)
show("V1a  Qwen2.5-32B-Instruct  MuSiQue T0  contains_W=False  (the 39/50 'search' samples in the 0.22 cell)", mu_t0_nW, 10)
show("V1b  Qwen2.5-32B-Instruct  MuSiQue T0  contains_W=True   (the 11/50 commit samples)", mu_t0_W, 5)

# ---------- V1c: 32B HQ T0 commit=True ----------
hq25 = [json.loads(l) for l in open(EVAL["32B-Instruct"]["hotpotqa"])]
hq25_t0_W = [r for r in hq25 if r["condition"] == "T0" and r.get("contains_W")]
random.shuffle(hq25_t0_W)
show("V1c  Qwen2.5-32B-Instruct  HotpotQA T0  contains_W=True   (the 41/50 commit samples in the 0.82 cell)", hq25_t0_W, 5)

# ---------- V1d: Qwen3-32B HQ T0 commit=True ----------
hq3 = [json.loads(l) for l in open(EVAL["Qwen3-32B"]["hotpotqa"])]
hq3_t0_W = [r for r in hq3 if r["condition"] == "T0" and r.get("contains_W")]
random.shuffle(hq3_t0_W)
show("V1d  Qwen3-32B  HotpotQA T0  contains_W=True   (the 50/50 commit samples in the 1.00 cell)", hq3_t0_W, 5)

# ---------- V1d-aux: Qwen3-32B HQ T0 ANY 5 (not pre-filtered to W) for stylistic comparison vs V1c ----------
hq3_t0 = [r for r in hq3 if r["condition"] == "T0"]
print(f"\n{'#'*100}")
print(f"# V1-styling-cross-check: pair-up first 3 same-sample_id rows across 32B-Instruct and Qwen3-32B (HotpotQA T0)")
print(f"{'#'*100}")
hq25_by_id = {r["sample_id"]: r for r in hq25 if r["condition"] == "T0"}
hq3_by_id  = {r["sample_id"]: r for r in hq3  if r["condition"] == "T0"}
shared = sorted(set(hq25_by_id) & set(hq3_by_id))[:3]
for sid in shared:
    a = hq25_by_id[sid]; b = hq3_by_id[sid]
    print(f"\n=== sample_id={sid} ===")
    print(f"  [32B-Instruct]  cw={a['contains_W']}  at={a['action_type']}  raw[:200]={(a['raw_output'] or '')[:200]!r}")
    print(f"  [Qwen3-32B]    cw={b['contains_W']}  at={b['action_type']}  raw[:200]={(b['raw_output'] or '')[:200]!r}")
