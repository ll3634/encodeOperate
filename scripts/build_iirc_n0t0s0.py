"""
IIRC N0/T0/S0 builder audit.
N0: main passage only (no W, no gold evidence)
T0: main passage + W candidate (extractable but wrong)
S0: main passage + gold linked passage (sufficient evidence)
W candidates come from the initial paragraph via regex for same-type spans.
"""

import json, re, sys, pathlib, random, collections, argparse

DATE_PAT = re.compile(
    r'\b(?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)(?:\s+\d{1,2},?)?\s+\d{4}\b', re.I)
YEAR_PAT  = re.compile(r'\b(1[2-9][0-9]{2}|20[0-2][0-9])\b')
NUMBER_PAT = re.compile(r'\b[1-9]\d{2,}\b')   # 3+ digit numbers only (avoids "68", "30")
MONEY_PAT  = re.compile(r'\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|thousand))?', re.I)
PHRASE_PAT = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b')


def _gold_type(gold: str) -> str:
    """Rough entity type of gold answer."""
    if DATE_PAT.match(gold.strip()): return "date"
    if YEAR_PAT.fullmatch(gold.strip()): return "year"
    if MONEY_PAT.match(gold.strip()): return "money"
    if re.match(r'^\d', gold.strip()): return "number"
    if re.match(r'^[A-Z]', gold.strip()): return "phrase"
    return "other"


def extract_candidates(text, gold):
    """Extract W candidates from text that are same type as gold and != gold."""
    gold_lc = gold.strip().lower()
    gtype = _gold_type(gold)
    cands = set()

    if gtype == "date":
        for m in DATE_PAT.finditer(text):
            v = m.group().strip()
            if v.lower() != gold_lc:
                cands.add(v)
    if gtype in ("year", "date"):
        for m in YEAR_PAT.finditer(text):
            v = m.group()
            if v != gold.strip() and v.lower() != gold_lc:
                cands.add(v)
    if gtype == "money":
        for m in MONEY_PAT.finditer(text):
            v = m.group().strip()
            if v.lower() != gold_lc:
                cands.add(v)
    if gtype == "number":
        for m in NUMBER_PAT.finditer(text):
            v = m.group()
            if v.lower() != gold_lc:
                cands.add(v)
    if gtype in ("phrase", "other"):
        for m in PHRASE_PAT.finditer(text):
            v = m.group().strip()
            if v.lower() != gold_lc and gold_lc not in v.lower() and len(v) < 80:
                cands.add(v)

    return [c for c in cands
            if c.lower() not in gold_lc and gold_lc not in c.lower()
            and len(c.strip()) >= 2]


def gold_in_text(gold, text):
    return gold.strip().lower() in text.lower()


def rough_wc(text):
    return len(text.split())


def try_build(art, q, max_obs_words=150):
    r = {"status": None, "reject_reason": None}
    ans_type = q["answer"].get("type", "")
    if ans_type not in ("span", "value"):
        r["status"] = "rejected"; r["reject_reason"] = "answer_type"; return r
    spans = q["answer"].get("answer_spans", [])
    if not spans or not spans[0].get("text", "").strip():
        r["status"] = "rejected"; r["reject_reason"] = "answer_type"; return r
    gold = spans[0]["text"].strip()

    ctx = q.get("context", [])
    main_snips = [c["text"] for c in ctx if c["passage"].lower() == "main"]
    link_snips = [(c["passage"], c["text"]) for c in ctx if c["passage"].lower() != "main"]
    if not main_snips:
        r["status"] = "rejected"; r["reject_reason"] = "no_main_ctx"; return r
    if not link_snips:
        r["status"] = "rejected"; r["reject_reason"] = "no_linked_ctx"; return r

    main_obs = " ".join(main_snips)
    linked_passage, linked_text = link_snips[0]
    s0_obs = main_obs + " " + linked_text

    if gold_in_text(gold, art["text"]) or gold_in_text(gold, main_obs):
        r["status"] = "rejected"; r["reject_reason"] = "gold_in_initial"; return r
    if rough_wc(s0_obs) > max_obs_words:
        r["status"] = "rejected"; r["reject_reason"] = "obs_too_long"; return r

    cands = extract_candidates(art["text"] + " " + main_obs, gold)
    if not cands:
        r["status"] = "rejected"; r["reject_reason"] = "no_W_found"; return r

    gold_len = len(gold.split())
    cands.sort(key=lambda c: abs(len(c.split()) - gold_len))
    W = cands[0]

    # Build T0 obs
    if W.lower() in main_obs.lower():
        t0_obs, w_src = main_obs, "main_snippet"
    else:
        sents = re.split(r'(?<=[.!?])\s+', art["text"])
        w_sents = [s for s in sents if W.lower() in s.lower()]
        if not w_sents:
            r["status"] = "rejected"; r["reject_reason"] = "no_W_found"; return r
        t0_obs, w_src = main_obs + " " + w_sents[0], "initial_sentence"

    if rough_wc(t0_obs) > max_obs_words:
        r["status"] = "rejected"; r["reject_reason"] = "obs_too_long"; return r

    r.update({
        "status": "clean",
        "sample_id": q["qid"],
        "article_title": art["title"],
        "question": q["question"],
        "gold_answer": gold,
        "candidate_W": W,
        "w_source": w_src,
        "obs_N0": main_obs,
        "obs_T0": t0_obs,
        "obs_S0": s0_obs,
        "linked_passage": linked_passage,
        "answer_type": ans_type,
        "n0_words": rough_wc(main_obs),
        "t0_words": rough_wc(t0_obs),
        "s0_words": rough_wc(s0_obs),
    })
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/featurize/iirc_data/iirc_train_dev/train.json")
    ap.add_argument("--scan-n", type=int, default=300)
    ap.add_argument("--out", default="results/iirc_builder_audit/builder_audit.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    train = json.load(open(args.data))
    random.shuffle(train)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    articles_scanned = questions_scanned = 0
    clean = []
    reject_counts = collections.Counter()

    for art in train:
        if articles_scanned >= args.scan_n:
            break
        articles_scanned += 1
        for q in art["questions"]:
            questions_scanned += 1
            r = try_build(art, q)
            r["article_title"] = art.get("title", "")
            if r["status"] == "clean":
                clean.append(r)
            else:
                reject_counts[r["reject_reason"]] += 1

    with open(out_path, "w") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "articles_scanned": articles_scanned,
        "questions_scanned": questions_scanned,
        "clean_examples": len(clean),
        "yield_pct": round(100 * len(clean) / max(questions_scanned, 1), 1),
        "rejection_counts": dict(reject_counts),
        "success_criteria": {
            "yield_ge_25pct": len(clean) / max(questions_scanned, 1) >= 0.25,
            "clean_ge_80": len(clean) >= 80,
        }
    }
    (out_path.parent / "builder_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if clean:
        ex = clean[0]
        print(f"\nSample:")
        print(f"  Q:    {ex['question']}")
        print(f"  gold: {ex['gold_answer']!r}")
        print(f"  W:    {ex['candidate_W']!r}  (from {ex['w_source']})")
        print(f"  N0:   {ex['obs_N0'][:180]}")
        print(f"  T0:   {ex['obs_T0'][:180]}")
        print(f"  S0:   {ex['obs_S0'][:180]}")


if __name__ == "__main__":
    main()
