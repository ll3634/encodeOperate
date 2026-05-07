#!/usr/bin/env python3
"""Robustness C: build natural-trace counterfactual pairs from real Qwen
HotpotQA failures.

Source: results/hotpotqa_search_post_amax8/baseline_results.jsonl  (200 samples).

Selection criteria for a "natural trap" sample:
  1. Model issued a Final Answer (action_type = stop).
  2. The predicted answer is wrong (gold and pred differ on a normalized
     substring match).
  3. The wrong predicted answer (W) is present in the concatenated
     observations seen by the model.
  4. The gold answer is NOT present in the observations (so commitment to W
     cannot be excused by partial gold support).

For each selected sample produce two records:
  - condition = "natT" : original observation (W intact)
  - condition = "natN" : observation with all case-insensitive occurrences
                         of W replaced by "[REDACTED]" (minimal edit;
                         everything else preserved verbatim).

Output schema matches results/extractability_support_toggle/pairs.jsonl so
the existing eval_extractability_cross_model.py runs unchanged.
"""
import argparse, json, re
from pathlib import Path


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def select_natural_traps(rows):
    """Selection rules:
      - pred wrong vs gold (no-substring-overlap)
      - W (pred) appears in observation
      - gold does NOT appear in observation
      - W does NOT appear in the question (otherwise redacting obs leaves W
        leaking via the question, contaminating the counterfactual)."""
    out = []
    for r in rows:
        pred = (r.get("final_answer") or "").strip()
        gold = (r.get("gold_answer") or "").strip()
        if not pred or not gold:
            continue
        np_, ng = norm(pred), norm(gold)
        if np_ == ng or ng in np_ or np_ in ng:
            continue
        obs = " ".join((s.get("observation") or "") for s in r.get("steps") or [])
        if pred.lower() not in obs.lower() or gold.lower() in obs.lower():
            continue
        if pred.lower() in r["question"].lower():
            continue
        out.append({"row": r, "obs": obs, "W": pred, "gold": gold})
    return out


def redact(obs: str, W: str, placeholder: str = "[REDACTED]") -> str:
    pattern = re.compile(re.escape(W), flags=re.IGNORECASE)
    return pattern.sub(placeholder, obs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline",
                    default="results/hotpotqa_search_post_amax8/baseline_results.jsonl")
    ap.add_argument("--out", default="results/cross_model_extractability/robustness/natural_trap_pairs.jsonl")
    ap.add_argument("--placeholder", default="[REDACTED]",
                    help="Token used to replace W in the natN counterfactual.")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.baseline)]
    cands = select_natural_traps(rows)
    print(f"[info] selected {len(cands)} natural traps from {len(rows)} samples")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_redact_count = []
    with open(args.out, "w") as f:
        for i, c in enumerate(cands):
            r = c["row"]
            sid = f"nat_{i:03d}"
            edited = redact(c["obs"], c["W"], args.placeholder)
            n_repl = c["obs"].lower().count(c["W"].lower())
            n_redact_count.append(n_repl)
            for cond, obs in (("natT", c["obs"]), ("natN", edited)):
                rec = {
                    "sample_id": sid,
                    "schema_type": "natural_hotpotqa",
                    "question":   r["question"],
                    "candidate_W": c["W"],
                    "gold_answer": c["gold"],
                    "gold_answers": r.get("gold_answers") or [c["gold"]],
                    "condition":  cond,
                    "obs":        obs,
                    "observation": obs,
                    "n_W_replacements": n_repl if cond == "natN" else 0,
                    "source_sample_id": r.get("sample_id"),
                    "source_n_steps": r.get("n_steps"),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[info] mean W replacements per natN obs: {sum(n_redact_count)/max(1,len(n_redact_count)):.2f}")
    print(f"[info] median W replacements: {sorted(n_redact_count)[len(n_redact_count)//2]}")
    print(f"[done] wrote {len(cands)*2} records -> {args.out}")


if __name__ == "__main__":
    main()
