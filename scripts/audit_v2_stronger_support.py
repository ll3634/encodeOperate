#!/usr/bin/env python3
"""Audit v2: re-score the existing natural_audit_raw.jsonl with a stronger
support_complete heuristic and report whether the wrong-stop vs correct-stop
contrast becomes informative.

support_complete_v2:
  W is considered 'supported' if any of:
    (a) original strict same-sentence relation holds (v1)
    (b) W and at least one question entity co-occur within a window of
        K=3 consecutive sentences (paragraph-level proxy)
    (c) W and at least one question entity co-occur within the same doc
        block (delimited by [n] Title:), AND that doc block also contains
        a relation verb in any sentence containing W

This produces a much LESS aggressive 'extractable_unsupported' flag.
"""
import json, re, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.audit_natural_failures import (
    _contains_alias, question_entities, split_sentences,
    support_complete_for_W,
)

try:
    from scipy.stats import fisher_exact
except ImportError:
    fisher_exact = None


_DOC_RE = re.compile(r"\[(\d+)\]\s*([^:\n]{1,100}?):\s*", re.MULTILINE)
_REL_RE = re.compile(
    r"\b(is|was|are|were|named|called|located in|based in|known as|"
    r"founded|composed|wrote|directed|capital|headquartered|home to|"
    r"member of|part of|won|received|achieved|completed|published|"
    r"released|born|died|created|appointed|elected|served|appears|"
    r"plays|starring|features)\b"
)


def split_docs(observation):
    """Split observation into doc blocks; returns list of doc-text strings."""
    if not observation:
        return []
    spans = list(_DOC_RE.finditer(observation))
    if not spans:
        return [observation]
    docs = []
    for i, m in enumerate(spans):
        start = m.start()
        end = spans[i + 1].start() if i + 1 < len(spans) else len(observation)
        docs.append(observation[start:end])
    return docs


def support_complete_v2(W, observation, question, gold):
    """Relaxed support: paragraph + same-doc proxies."""
    if support_complete_for_W(W, observation, question, gold):
        return True, "v1_strict_match"
    if not W or not observation or not question:
        return False, None
    qents = question_entities(question)
    if not qents:
        return False, None

    # (b) sentence-window proxy: W and qent within K=3 consecutive sentences
    sents = split_sentences(observation)
    K = 3
    for i, s in enumerate(sents):
        if not _contains_alias(s, W):
            continue
        window = sents[max(0, i - K): i + K + 1]
        # also require a relation verb somewhere in the window
        if any(_contains_alias(t, q) for t in window for q in qents):
            joined = " ".join(window).lower()
            if _REL_RE.search(joined):
                return True, "v2_window_K3"

    # (c) same doc-block: W + qent + relation verb anywhere in same block
    for doc in split_docs(observation):
        if not _contains_alias(doc, W):
            continue
        if not any(_contains_alias(doc, q) for q in qents):
            continue
        if _REL_RE.search(doc.lower()):
            return True, "v2_same_doc"
    return False, None


def fisher(a, b, c, d):
    if fisher_exact is None:
        return None, None
    odds, p = fisher_exact([[a, b], [c, d]])
    return float(odds), float(p)


def main():
    raw_path = Path("results/natural_extractability_audit/natural_audit_raw.jsonl")
    records = [json.loads(l) for l in open(raw_path)]
    print(f"Loaded {len(records)} records from {raw_path}")

    # re-score each record's support_complete and extractable_unsupported.
    # Search trajectories never stop with a Final Answer, so they have W=null
    # and extractable_unsupported is 0 by construction; keep them in the count
    # so the wrong-stop vs search Fisher contrast remains computable.
    by_cat = {"step1_stop_wrong": [], "step1_stop_correct": [], "step1_search": []}
    method_counter = Counter()
    for r in records:
        if r["category"] not in by_cat:
            continue
        if r["category"] == "step1_search":
            r["support_complete_v2"] = None
            r["extractable_unsupported_v2"] = False
            by_cat[r["category"]].append(r)
            continue
        if not r.get("emitted_answer_W"):
            continue
        sup, m = support_complete_v2(
            r["emitted_answer_W"], r["observation_full"],
            r["question"], r["gold_answer"],
        )
        r["support_complete_v2"] = sup
        if m: method_counter[m] += 1
        sal_v2 = r.get("W_salient", False)
        r["extractable_unsupported_v2"] = bool(
            r["W_in_observation"] and r["W_type_compatible"]
            and sal_v2 and not sup
        )
        by_cat[r["category"]].append(r)

    print(f"\nv2 support-method hits: {dict(method_counter)}")
    print()
    print("=== Prevalence under v1 (strict) vs v2 (relaxed) support heuristic ===")
    print(f"{'category':<22s} {'n':>5s} {'v1_extr':>8s} {'v1_rate':>8s} {'v2_extr':>8s} {'v2_rate':>8s}")
    summary = {}
    for cat, items in by_cat.items():
        n = len(items)
        v1 = sum(1 for r in items if r.get("extractable_unsupported"))
        v2 = sum(1 for r in items if r.get("extractable_unsupported_v2"))
        v1_rate = v1 / n if n else None
        v2_rate = v2 / n if n else None
        v1s = f"{v1_rate:.3f}" if v1_rate is not None else "n/a"
        v2s = f"{v2_rate:.3f}" if v2_rate is not None else "n/a"
        print(f"{cat:<22s} {n:>5d} {v1:>8d} {v1s:>8s} {v2:>8d} {v2s:>8s}")
        summary[cat] = {"n": n, "v1": v1, "v2": v2, "v1_rate": v1_rate, "v2_rate": v2_rate}

    print()
    print("=== Fisher exact (one-sided) under v2 ===")
    for (c1, c2) in [("step1_stop_wrong", "step1_search"),
                     ("step1_stop_wrong", "step1_stop_correct")]:
        a = summary[c1]["v2"]; b = summary[c1]["n"] - a
        c = summary[c2]["v2"]; d = summary[c2]["n"] - c
        odds, p = fisher(a, b, c, d)
        print(f"  {c1} ({a}/{summary[c1]['n']}) vs {c2} ({c}/{summary[c2]['n']}): odds={odds}, p={p}")

    # write v2 raw file and summary
    out_jsonl = raw_path.parent / "natural_audit_raw_v2.jsonl"
    with open(out_jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    out_json = raw_path.parent / "audit_summary_v2.json"
    with open(out_json, "w") as f:
        json.dump({
            "support_method_counter": dict(method_counter),
            "by_category": summary,
        }, f, indent=2)
    print(f"\nWrote {out_jsonl} and {out_json}")


if __name__ == "__main__":
    main()
