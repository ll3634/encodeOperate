#!/usr/bin/env python3
"""Natural Failure Audit (Part A).

Loads the existing Qwen2.5-7B HotpotQA-bridge baseline trajectories
(results/l20_rho020_n500/baseline_results.jsonl), classifies each
sample into wrong-stop / correct-stop / search, regenerates the FULL
observation text the model actually saw (the saved obs is truncated
to 200 chars), extracts the natural candidate W from the model's
final_answer, runs auto-heuristics for extractable_unsupported, and
emits:
  results/natural_extractability_audit/
    natural_audit_raw.jsonl   - one record per sample with all features
    audit_sheet.csv           - blind-review sheet (50 wrong-stop + 20 search)
    audit_summary.json        - prevalence stats + Fisher exact tests
"""
import argparse, csv, json, random, re, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.search_tool import SearchTool  # noqa: E402

try:
    from scipy.stats import fisher_exact
except ImportError:
    fisher_exact = None


# ---------- normalisation / W extraction --------------------------------
_PUNCT_RE = re.compile(r"[\W_]+")
_ARTICLES = {"the", "a", "an"}


def norm(s):
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[\.,;:!\?\"'`\(\)\[\]]+$", "", s)
    s = re.sub(r"^[\.,;:!\?\"'`\(\)\[\]]+", "", s)
    return s.strip()


def strip_articles(s):
    toks = [t for t in re.split(r"\s+", norm(s)) if t and t not in _ARTICLES]
    return " ".join(toks)


def fold(s):
    return _PUNCT_RE.sub(" ", norm(s)).strip()


def aliases_of(s):
    """A small set of cheap aliases for matching."""
    base = norm(s)
    out = {base}
    out.add(strip_articles(base))
    out.add(fold(base))
    # drop trailing parenthetical
    out.add(re.sub(r"\s*\([^)]+\)\s*$", "", base).strip())
    # drop comma tail (e.g. 'New Rochelle, New York' -> 'New Rochelle')
    if "," in base:
        out.add(base.split(",", 1)[0].strip())
    return {a for a in out if a}


def extract_candidate_W(final_answer):
    """Return (W_string, extraction_method).
    For short answers, W is the whole normalised string.
    For long evasive answers, try to extract a salient capitalised
    noun phrase or a quoted phrase. Returns ('', 'none') if no clean W."""
    if not final_answer:
        return "", "none"
    fa = final_answer.strip()
    if len(fa) <= 80:
        return norm(fa), "whole"
    # quoted phrase
    m = re.search(r'"([^"]{2,60})"', fa)
    if m:
        return norm(m.group(1)), "quoted"
    # 'is/was/are <Capitalised...>' pattern
    m = re.search(r"\b(?:is|was|are|were|named|called|known as)\s+([A-Z][\w'\.\-]*(?:\s+[A-Z][\w'\.\-]*){0,4})", fa)
    if m:
        return norm(m.group(1)), "is_pattern"
    # first multi-word capitalised phrase
    m = re.search(r"\b([A-Z][\w'\.\-]*(?:\s+[A-Z][\w'\.\-]*){1,4})\b", fa)
    if m:
        return norm(m.group(1)), "first_caps"
    # fallback: first 6 words
    return norm(" ".join(fa.split()[:6])), "first6"


# ---------- W_in_observation / type / salience / support ---------------
def w_in_obs(W, obs):
    if not W or not obs:
        return False, None
    obs_n = norm(obs)
    obs_f = fold(obs)
    for a in aliases_of(W):
        if not a:
            continue
        if a in obs_n:
            return True, a
        if fold(a) and fold(a) in obs_f:
            return True, a
    return False, None


_QTYPE_PATTERNS = [
    ("year",     r"\b(in what year|what year|when was|when did|when were|when is)\b"),
    ("number",   r"\b(how many|how much|how often|how long|what is the population|population of)\b"),
    ("person",   r"\b(who is|who was|who are|who were|who composed|who wrote|who directed|who founded|who created|whose|who plays|who sang)\b"),
    ("place",    r"\b(where is|where was|where are|where were|in which (city|country|state|town)|what (city|country|state|town))\b"),
    ("org",      r"\b(what (company|organization|band|team|university|college|institution))\b"),
    ("title",    r"\b(what (book|film|movie|album|song|series|show|novel|game))\b"),
    ("yesno",    r"\b(is the|did the|are both|did both|do both)\b"),
]


def question_type(question):
    if not question:
        return "other"
    ql = question.lower()
    for label, pat in _QTYPE_PATTERNS:
        if re.search(pat, ql):
            return label
    if ql.lstrip().startswith("what"):
        return "what_other"
    if ql.lstrip().startswith("which"):
        return "which_other"
    return "other"


_PERSON_HINT = re.compile(r"^[A-Z][a-zA-Z'\-\.]+(?:\s+[A-Z][a-zA-Z'\-\.]+){1,3}$")
_NUMBER_RE = re.compile(r"^[\d,\.]+(?:\s*(million|billion|thousand|m|bn|k))?$", re.IGNORECASE)
_YEAR_RE = re.compile(r"^(1[0-9]{3}|20[0-2][0-9])$")


def w_type_compatible(W, qtype):
    if not W:
        return False
    w = W.strip()
    if qtype == "year":
        return bool(_YEAR_RE.match(w))
    if qtype == "number":
        return bool(_NUMBER_RE.match(w.replace(" ", "")))
    if qtype == "person":
        # at least 2 capitalised tokens, no leading number
        return bool(_PERSON_HINT.match(w)) and not _NUMBER_RE.match(w)
    if qtype in ("place", "org", "title", "what_other", "which_other"):
        return bool(re.match(r"^[A-Z]", w)) or len(w.split()) >= 1
    return True


# ---------- salience -------------------------------------------------------
_STOPS = set("the a an of in on at to for and or but is was are were be been being "
             "by from with as which what who when where how that this these those "
             "have has had do does did".split())


def question_entities(question):
    """Extract capitalised multi-word phrases from the question (cheap NER)."""
    if not question:
        return []
    ents = re.findall(r"[A-Z][\w'\-\.]*(?:\s+[A-Z][\w'\-\.]*){0,5}", question)
    out = []
    for e in ents:
        en = norm(e)
        if en and en not in _STOPS and not _YEAR_RE.match(en) and len(en) >= 2:
            out.append(en)
    return out


def split_sentences(text):
    if not text:
        return []
    text = re.sub(r"\[\d+\]\s*[^:\n]{0,60}:\s*", " ", text)
    parts = re.split(r"(?<=[\.\!\?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _contains_alias(text, target):
    if not text or not target:
        return False
    tn, tf = norm(text), fold(text)
    for a in aliases_of(target):
        if not a:
            continue
        if a in tn or (fold(a) and fold(a) in tf):
            return True
    return False


def w_salience(W, observation, question):
    """Return dict of salience flags."""
    sents = split_sentences(observation)
    qents = question_entities(question)
    same_sent_qent = any(
        _contains_alias(s, W) and any(_contains_alias(s, q) for q in qents)
        for s in sents
    )
    in_candidate_list = bool(re.search(
        r"(?:[A-Z][\w'\-\.]+(?:\s+[A-Z][\w'\-\.]+)*\s*,\s*){2,}\s*and\s+[A-Z][\w'\-\.]+",
        observation or ""
    )) and any(_contains_alias(observation or "", W) for _ in [0])
    in_relation_ctx = False
    for s in sents:
        if _contains_alias(s, W) and re.search(
            r"\b(is|was|are|were|named|called|located in|based in|known as|"
            r"founded|composed|wrote|directed|capital|headquartered|home to)\b",
            s.lower()
        ):
            in_relation_ctx = True
            break
    pos = -1
    obs_f = fold(observation or "")
    for a in aliases_of(W):
        if a:
            af = fold(a)
            if af and af in obs_f:
                pos = obs_f.index(af)
                break
    obs_len = len(obs_f) or 1
    rel_pos = pos / obs_len if pos >= 0 else None
    salient = same_sent_qent or in_candidate_list or in_relation_ctx
    return {
        "W_position_relpos": rel_pos,
        "W_same_sentence_with_question_entity": same_sent_qent,
        "W_in_candidate_list_or_relation_context": in_candidate_list or in_relation_ctx,
        "W_salient": salient,
    }


def support_complete_for_W(W, observation, question, gold):
    """Heuristic: observation 'completely supports' W as the answer iff
    a single sentence simultaneously mentions (a) W, (b) at least one
    question entity, AND (c) a relation token whose object is W (not a
    different entity).

    This is intentionally conservative — when in doubt we mark
    'support_complete=False' so 'extractable_unsupported' is the strong
    case (W is fishable but evidence is incomplete). The blind audit
    column lets the human auditor override.
    """
    if not W or not observation:
        return False
    qents = question_entities(question)
    if not qents:
        return False
    sents = split_sentences(observation)
    rel = re.compile(
        r"\b(is|was|are|were|named|called|located in|based in|known as|"
        r"founded|composed|wrote|directed|capital|headquartered|home to|"
        r"member of|part of)\b"
    )
    for s in sents:
        sl = s.lower()
        if not (_contains_alias(s, W) and rel.search(sl)):
            continue
        if not any(_contains_alias(s, q) for q in qents):
            continue
        # heuristic guard: if the sentence also contains the gold answer
        # nearby AND gold != W, it likely supports gold rather than W
        if gold and not _contains_alias(s, gold):
            return True
        if gold and _contains_alias(s, gold):
            # both present: ambiguous, treat as supported only if W is closer
            # to a relation verb than gold
            return False
    return False


# ---------- categorisation ------------------------------------------------
def categorise(r):
    if r["n_steps"] < 2:
        return "no_step1"
    s0 = r["steps"][0]
    s1 = r["steps"][1]
    if s0.get("action") != "search":
        return "step0_not_search"
    if s1.get("parse_failure_reason"):
        return "step1_parsefail"
    if s1.get("action") == "search":
        return "step1_search"
    if s1.get("final_answer") is not None:
        return "step1_stop_correct" if r.get("is_correct") else "step1_stop_wrong"
    return "step1_other"



# ---------- main ----------------------------------------------------------
def build_record(r, search_tool, cache):
    cat = categorise(r)
    s0 = r["steps"][0] if r["n_steps"] >= 1 else {}
    s1 = r["steps"][1] if r["n_steps"] >= 2 else {}
    query = s0.get("action_input") or ""
    # regenerate full observation deterministically
    if query in cache:
        observation = cache[query]
    else:
        observation = search_tool(query) if query else ""
        cache[query] = observation
    fa = s1.get("final_answer") if cat in ("step1_stop_correct", "step1_stop_wrong") else None
    W, w_method = extract_candidate_W(fa or "")
    in_obs, alias_match = w_in_obs(W, observation)
    qtype = question_type(r["question"])
    type_ok = w_type_compatible(W, qtype)
    sal = w_salience(W, observation, r["question"])
    support_ok = support_complete_for_W(W, observation, r["question"], r["gold_answer"])
    extractable_unsupported = bool(in_obs and type_ok and sal["W_salient"] and not support_ok)
    return {
        "sample_id": r["sample_id"],
        "category": cat,
        "question": r["question"],
        "question_type": qtype,
        "gold_answer": r["gold_answer"],
        "gold_answers": r.get("gold_answers", [r["gold_answer"]]),
        "is_correct": r.get("is_correct"),
        "first_search_query": query,
        "observation_full": observation,
        "observation_token_len": len(observation.split()),
        "step1_action": s1.get("action"),
        "step1_final_answer": fa,
        "step1_raw_model_text": s1.get("raw_model_text"),
        "step1_margin_before": s1.get("margin_before"),
        "emitted_answer_W": W,
        "W_extraction_method": w_method,
        "W_in_observation": in_obs,
        "W_alias_match": alias_match,
        "W_type_compatible": type_ok,
        **sal,
        "support_complete_for_W": support_ok,
        "extractable_unsupported": extractable_unsupported,
    }


def fisher(a, b, c, d):
    """odds ratio + p-value for 2x2 [[a,b],[c,d]]; falls back if scipy missing."""
    if fisher_exact is None:
        return None, None
    odds, p = fisher_exact([[a, b], [c, d]])
    return float(odds), float(p)


def write_audit_sheet(records, out_path, n_wrong=50, n_search=20, seed=0):
    rng = random.Random(seed)
    wrong = [r for r in records if r["category"] == "step1_stop_wrong" and r["emitted_answer_W"]]
    search = [r for r in records if r["category"] == "step1_search"]
    rng.shuffle(wrong)
    rng.shuffle(search)
    rows = wrong[:n_wrong] + search[:n_search]
    rng.shuffle(rows)  # shuffle so blind reviewer doesn't know group
    cols = ["sample_id", "question", "first_search_query", "observation_full",
            "candidate_W_proposal", "blind_W_in_obs", "blind_W_type_compatible",
            "blind_W_salient", "blind_support_complete", "blind_extractable_unsupported",
            "blind_notes"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({
                "sample_id": r["sample_id"],
                "question": r["question"],
                "first_search_query": r["first_search_query"],
                "observation_full": r["observation_full"],
                "candidate_W_proposal": r["emitted_answer_W"],
                "blind_W_in_obs": "", "blind_W_type_compatible": "",
                "blind_W_salient": "", "blind_support_complete": "",
                "blind_extractable_unsupported": "", "blind_notes": "",
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--corpus",   default="data/hotpotqa/corpus.jsonl")
    ap.add_argument("--out-dir",  default="results/natural_extractability_audit")
    ap.add_argument("--limit",    type=int, default=None,
                    help="Truncate to first N records (smoke test).")
    ap.add_argument("--top-k",    type=int, default=5)
    ap.add_argument("--max-chars", type=int, default=500)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/4] loading baseline: {args.baseline}")
    rows = [json.loads(l) for l in open(args.baseline)]
    if args.limit:
        rows = rows[:args.limit]
    print(f"  {len(rows)} records")

    print(f"[2/4] loading search tool: {args.corpus}")
    search_tool = SearchTool(args.corpus, top_k=args.top_k, max_chars=args.max_chars)

    print("[3/4] building per-sample audit records ...")
    cache = {}
    records = []
    for i, r in enumerate(rows, 1):
        try:
            records.append(build_record(r, search_tool, cache))
        except Exception as e:
            print(f"  [warn] sample {r.get('sample_id')}: {e}")
        if i % 50 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}")

    raw_path = out_dir / "natural_audit_raw.jsonl"
    with open(raw_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  -> {raw_path}")

    sheet_path = out_dir / "audit_sheet.csv"
    write_audit_sheet(records, sheet_path)
    print(f"  -> {sheet_path}")

    print("[4/4] computing summary stats ...")
    cat_counts = Counter(r["category"] for r in records)
    def grp(cat):
        return [r for r in records if r["category"] == cat]
    ws, cs, sr = grp("step1_stop_wrong"), grp("step1_stop_correct"), grp("step1_search")
    def pct_extractable(group):
        n = len(group)
        if n == 0: return None
        k = sum(1 for r in group if r["extractable_unsupported"])
        return {"n": n, "k_extractable_unsupported": k, "rate": k / n}

    eu_ws = pct_extractable(ws)
    eu_cs = pct_extractable(cs)
    eu_sr = pct_extractable(sr)

    # Fisher: wrong-stop vs search (extractable_unsupported)
    a = eu_ws["k_extractable_unsupported"] if eu_ws else 0
    b = (eu_ws["n"] - a) if eu_ws else 0
    c = eu_sr["k_extractable_unsupported"] if eu_sr else 0
    d = (eu_sr["n"] - c) if eu_sr else 0
    odds_ws_sr, p_ws_sr = fisher(a, b, c, d)
    a2 = eu_ws["k_extractable_unsupported"] if eu_ws else 0
    b2 = (eu_ws["n"] - a2) if eu_ws else 0
    c2 = eu_cs["k_extractable_unsupported"] if eu_cs else 0
    d2 = (eu_cs["n"] - c2) if eu_cs else 0
    odds_ws_cs, p_ws_cs = fisher(a2, b2, c2, d2)

    summary = {
        "baseline_file": args.baseline,
        "n_total": len(records),
        "category_counts": dict(cat_counts),
        "extractable_unsupported": {
            "wrong_stop": eu_ws,
            "correct_stop": eu_cs,
            "search": eu_sr,
        },
        "fisher": {
            "wrong_stop_vs_search":  {"odds_ratio": odds_ws_sr, "p_value": p_ws_sr},
            "wrong_stop_vs_correct": {"odds_ratio": odds_ws_cs, "p_value": p_ws_cs},
        },
    }
    with open(out_dir / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
