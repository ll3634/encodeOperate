#!/usr/bin/env python3
"""Audit v3: structural presence of an extractable_unsupported candidate.

The core asymmetry: step1_search trajectories have no emitted W, so
'extractable_unsupported' was 0 by construction. To make the wrong-stop vs
search Fisher contrast meaningful at the structural level, we ask instead:

    For each (Q, observation), does there exist AT LEAST ONE type-compatible,
    salient, paragraph-level-unsupported candidate string in the observation?

Then we compare the rate of 'structural presence' across categories. If the
trap structure is genuinely associated with wrong-stop failures, the rate
should be higher in wrong-stops than in search trajectories.
"""
import json, re, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.audit_natural_failures import (
    norm, fold, _contains_alias, question_entities, split_sentences,
    question_type, w_type_compatible,
)
from scripts.audit_v2_stronger_support import support_complete_v2

try:
    from scipy.stats import fisher_exact
except ImportError:
    fisher_exact = None


_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
_NUM_RE = re.compile(r"\b\d{1,4}(?:,\d{3})*\b")
_NPHRASE_RE = re.compile(r"\b[A-Z][\w'\-\.]*(?:\s+[A-Z][\w'\-\.]*){0,4}\b")


def candidate_pool(observation, qtype):
    """Enumerate plausible answer candidates from the observation, by qtype."""
    obs = observation or ""
    cands = set()
    if qtype == "year":
        cands |= set(_YEAR_RE.findall(obs))
    elif qtype == "number":
        for m in _NUM_RE.finditer(obs):
            cands.add(m.group(0))
    elif qtype == "yesno":
        # yes/no questions don't have observable extractable candidates in the
        # usual sense; structural presence is undefined.
        return set()
    else:
        # noun-phrase types
        for m in _NPHRASE_RE.finditer(obs):
            s = m.group(0).strip()
            if 2 <= len(s) <= 80 and s.lower() not in {"the", "a", "an"}:
                cands.add(s)
    return cands


def is_salient(W, observation, question):
    """Cheap re-implementation: W appears in obs, AND ((appears in first 2
    sentences of any doc) OR (appears >=2 times) OR (in same sentence with
    a question entity))."""
    if not W or not observation:
        return False
    obs_f = fold(observation)
    wf = fold(W)
    if wf and wf in obs_f:
        # appears at least once
        count = obs_f.count(wf)
        if count >= 2:
            return True
        sents = split_sentences(observation)
        qents = question_entities(question)
        for s in sents:
            if _contains_alias(s, W) and any(_contains_alias(s, q) for q in qents):
                return True
        # first 2 sentences of any doc - approximate by absolute position
        rel = obs_f.index(wf) / max(1, len(obs_f))
        if rel < 0.2:
            return True
    return False


def has_extractable_unsupported_candidate(question, observation, gold_answer):
    """Does the observation contain at least one candidate that is
    type-compatible, salient, and paragraph-level-unsupported?
    Excludes the gold answer itself (we want a *spurious* candidate).
    """
    qtype = question_type(question)
    pool = candidate_pool(observation, qtype)
    if not pool:
        return False, None, qtype, 0
    qents_norm = {norm(q) for q in question_entities(question)}
    gold_norm = norm(gold_answer or "")
    n_examined = 0
    for c in pool:
        cn = norm(c)
        if not cn or cn == gold_norm or cn in qents_norm:
            continue
        if not w_type_compatible(c, qtype):
            continue
        if not is_salient(c, observation, question):
            continue
        sup, _ = support_complete_v2(c, observation, question, gold_answer)
        n_examined += 1
        if not sup:
            return True, c, qtype, n_examined
    return False, None, qtype, n_examined


def main():
    raw_path = Path("results/natural_extractability_audit/natural_audit_raw.jsonl")
    records = [json.loads(l) for l in open(raw_path)]
    by_cat = {"step1_stop_wrong": [], "step1_stop_correct": [], "step1_search": []}
    for r in records:
        if r["category"] in by_cat and r.get("observation_full"):
            by_cat[r["category"]].append(r)
    print("Category counts:", {k: len(v) for k, v in by_cat.items()})

    summary = {}
    for cat, items in by_cat.items():
        n = len(items); k_present = 0; n_skipped_yesno = 0
        examples = []
        for r in items:
            present, c, qtype, n_examined = has_extractable_unsupported_candidate(
                r["question"], r["observation_full"], r["gold_answer"]
            )
            if qtype == "yesno":
                n_skipped_yesno += 1
                continue
            if present:
                k_present += 1
                if len(examples) < 5:
                    examples.append((r["sample_id"], qtype, c, r["question"][:80]))
        n_eff = n - n_skipped_yesno
        rate = k_present / n_eff if n_eff else None
        summary[cat] = {"n": n, "n_eff_excl_yesno": n_eff, "k_present": k_present,
                        "rate": rate, "n_skipped_yesno": n_skipped_yesno}
        print(f"\n{cat}: n_eff={n_eff} (excluding {n_skipped_yesno} yes/no), "
              f"structural_present={k_present} ({rate:.3f})")
        for ex in examples:
            print(f"  e.g. {ex[0]} [{ex[1]}] '{ex[2]}' from Q='{ex[3]}'")

    print()
    print("=== Fisher exact (one-sided) on structural presence ===")
    if fisher_exact is not None:
        for c1, c2 in [("step1_stop_wrong", "step1_search"),
                       ("step1_stop_wrong", "step1_stop_correct")]:
            a = summary[c1]["k_present"]; b = summary[c1]["n_eff_excl_yesno"] - a
            c = summary[c2]["k_present"]; d = summary[c2]["n_eff_excl_yesno"] - c
            odds, p = fisher_exact([[a, b], [c, d]])
            print(f"  {c1} ({a}/{a+b}) vs {c2} ({c}/{c+d}): odds={odds:.3f}, p={p:.4g}")

    out = Path("results/natural_extractability_audit/audit_summary_v3_structural.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
