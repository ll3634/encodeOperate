#!/usr/bin/env python3
"""
Build paired Low-L / High-L observations for the local-answerability causal test.

Design (deterministic, gold-context based; no LLM rewriting):
  For each HotpotQA bridge sample whose gold answer string appears in one of
  its supporting-fact (SF) sentences:
    - P1  = paragraph containing the OTHER SF (the "bridge clue" / first-hop)
    - P2  = paragraph containing the gold answer (the second-hop SF)
    - The observation ALWAYS excludes P2 and any sentence containing the gold
      answer. This guarantees global insufficiency by construction.
    - Low-L : P1's bridge SF sentence + 1-2 *low-overlap* filler sentences
              from P1 (no declarative answer-like cues).
    - High-L: P1's bridge SF sentence + 1-2 *high-overlap, declarative*
              sentences from P1 that make the paragraph *look* more focused on
              the question topic (but do not state the answer).
    - A single fixed distractor paragraph is appended to both sides as [2].
    - Length ratio target: < 1.20 (flagged otherwise).

Output: JSONL of per-condition rows (two per sample) + pair-level metadata JSONL.
"""
import argparse, json, re, sys, random
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.hotpotqa import HotpotQADataset
from eval.scorers import normalize_answer
from tools.search_tool import STOPWORDS

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-']*")
_COPULA = {"is", "was", "are", "were", "be", "been", "being", "has", "have", "had"}


def tokens(text):
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def content_tokens(text):
    return [t for t in tokens(text) if t not in STOPWORDS and len(t) > 1]


def has_copula(sent):
    return any(t in _COPULA for t in tokens(sent))


def entity_count(text):
    """Rough capitalised-span count (proper-noun proxy)."""
    return len(re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}\b", text or ""))


def answer_components(answer):
    """Split compound answers like 'X, Y', 'X and Y', 'X & Y' into components."""
    if not answer:
        return []
    parts = re.split(r",|&|\band\b", answer, flags=re.IGNORECASE)
    comps = [p.strip() for p in parts if p.strip() and len(p.strip()) >= 3]
    return comps if comps else [answer.strip()]


def contains_answer_text(haystack, answer, strict=True):
    """If strict, also reject when ANY component of a compound answer appears."""
    hn = normalize_answer(haystack)
    an = normalize_answer(answer)
    if not an:
        return False
    if an in hn:
        return True
    if strict:
        for comp in answer_components(answer):
            cn = normalize_answer(comp)
            if cn and cn != an and cn in hn:
                return True
    return False


def paragraph_leaks_answer(sents, answer):
    """True if the concatenated paragraph contains the full answer OR jointly
    contains ALL components of a compound answer."""
    joined = " ".join(sents)
    if contains_answer_text(joined, answer, strict=False):
        return True
    comps = answer_components(answer)
    if len(comps) >= 2:
        jn = normalize_answer(joined)
        if all(normalize_answer(c) in jn for c in comps):
            return True
    return False


def split_sf(sample, answer):
    """Return (p1_title, p1_sents, p1_sf_sent, p2_title) or None if degenerate.

    p1 is the SF paragraph whose SF sentence does NOT contain the answer.
    p2 is the paragraph whose SF sentence contains the answer.
    """
    ctx_by_title = {t: s for t, s in sample.context}
    sf_bucket = {}
    for title, idx in sample.supporting_facts:
        sf_bucket.setdefault(title, []).append(idx)
    if len(sf_bucket) < 2:
        return None
    p1_title = p2_title = None
    p1_sf_sent = None
    for title, idxs in sf_bucket.items():
        if title not in ctx_by_title:
            return None
        sents = ctx_by_title[title]
        sf_sents = [sents[i] for i in idxs if 0 <= i < len(sents)]
        if not sf_sents:
            return None
        if any(contains_answer_text(s, answer) for s in sf_sents):
            p2_title = title
        else:
            p1_title = title
            p1_sf_sent = " ".join(sf_sents).strip()
    if not (p1_title and p2_title and p1_sf_sent):
        return None
    if paragraph_leaks_answer(ctx_by_title[p1_title], answer):
        return None
    return p1_title, ctx_by_title[p1_title], p1_sf_sent, p2_title


def pick_distractor(sample, p1_title, p2_title, answer, rng):
    """A neutral non-SF, non-answer paragraph from the sample's context."""
    cands = []
    for title, sents in sample.context:
        if title in (p1_title, p2_title):
            continue
        joined = " ".join(sents).strip()
        if not joined or contains_answer_text(joined, answer):
            continue
        cands.append((title, joined))
    if not cands:
        return None
    return rng.choice(cands)


def rank_p1_sentences(sents, sf_sent, q_terms, answer):
    """Return per-sentence dicts with overlap score; excludes SF sent itself
    and any answer-leaking sentence."""
    sf_norm = sf_sent.strip()
    qset = set(q_terms)
    rows = []
    for s in sents:
        s = s.strip()
        if not s or s == sf_norm:
            continue
        if contains_answer_text(s, answer):
            continue
        ct = content_tokens(s)
        overlap = sum(1 for t in ct if t in qset)
        rows.append({
            "sent": s,
            "overlap": overlap,
            "copula": has_copula(s),
            "len": len(tokens(s)),
        })
    return rows


def build_variant(p1_title, sf_sent, extras, distractor, max_chars=500):
    body = " ".join([sf_sent] + [e["sent"] for e in extras]).strip()
    result_1 = f"[1] {p1_title}: {body[:max_chars]}"
    if distractor is None:
        return result_1
    d_title, d_text = distractor
    d_text = d_text[:max_chars]
    return f"{result_1}\n\n[2] {d_title}: {d_text}"


def build_pair(sample, rng, k_extras=2, max_chars=500):
    """Return dict with low/high observations + metadata, or None if the sample
    cannot be built deterministically under our constraints."""
    ans = sample.answer.strip()
    if not ans:
        return None
    split = split_sf(sample, ans)
    if split is None:
        return None
    p1_title, p1_sents, p1_sf_sent, p2_title = split

    q_terms = set(content_tokens(sample.question))
    ranked = rank_p1_sentences(p1_sents, p1_sf_sent, q_terms, ans)
    if len(ranked) < 1:
        return None

    high_sorted = sorted(ranked, key=lambda r: (-r["overlap"], -int(r["copula"]), -r["len"]))
    low_sorted = sorted(ranked, key=lambda r: (r["overlap"], int(r["copula"]), r["len"]))
    high_extras = high_sorted[:k_extras]
    low_extras = [r for r in low_sorted if r not in high_extras][:k_extras]

    high_score = sum(e["overlap"] for e in high_extras)
    low_score = sum(e["overlap"] for e in low_extras)
    if high_score <= low_score:
        return None

    distractor = pick_distractor(sample, p1_title, p2_title, ans, rng)

    obs_low = build_variant(p1_title, p1_sf_sent, low_extras, distractor, max_chars=max_chars)
    obs_high = build_variant(p1_title, p1_sf_sent, high_extras, distractor, max_chars=max_chars)

    for tag, obs in (("low", obs_low), ("high", obs_high)):
        if contains_answer_text(obs, ans):
            return None

    def feats(text):
        return {
            "char_len": len(text),
            "tok_len": len(tokens(text)),
            "q_overlap": sum(1 for t in content_tokens(text) if t in q_terms),
            "entity_count": entity_count(text),
            "copula_count": sum(1 for t in tokens(text) if t in _COPULA),
        }

    f_low, f_high = feats(obs_low), feats(obs_high)
    len_ratio = max(f_low["tok_len"], f_high["tok_len"]) / max(1, min(f_low["tok_len"], f_high["tok_len"]))
    return {
        "sample_id": sample.id,
        "question": sample.question,
        "gold_answer": ans,
        "gold_answers": sample.answers,
        "p1_title": p1_title,
        "p2_title": p2_title,
        "p1_sf_sent": p1_sf_sent,
        "distractor_title": distractor[0] if distractor else None,
        "obs_low": obs_low,
        "obs_high": obs_high,
        "low_extras": [e["sent"] for e in low_extras],
        "high_extras": [e["sent"] for e in high_extras],
        "low_overlap_score": low_score,
        "high_overlap_score": high_score,
        "feat_low": f_low,
        "feat_high": f_high,
        "length_ratio": len_ratio,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpot", default="data/hotpotqa/hotpot_dev_distractor_v1.json")
    ap.add_argument("--baseline", default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--out-dir", default="results/local_answerability")
    ap.add_argument("--n", type=int, default=60, help="target number of usable pairs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-extras", type=int, default=2)
    ap.add_argument("--max-length-ratio", type=float, default=1.25)
    ap.add_argument("--type-filter", default="bridge")
    ap.add_argument("--require-premature-stop", action="store_true", default=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    premature_ids = set()
    if args.require_premature_stop:
        with open(args.baseline) as f:
            for line in f:
                r = json.loads(line)
                if (r.get("n_steps") == 2
                        and not r.get("em_correct", False)
                        and (r.get("steps") or [{}])[0].get("action") == "search"):
                    premature_ids.add(r["sample_id"])
        print(f"[info] premature-stop incorrect candidates: {len(premature_ids)}")

    ds = HotpotQADataset(args.hotpot)
    by_id = {s.id: s for s in ds.samples}
    if args.require_premature_stop:
        pool = [by_id[i] for i in premature_ids if i in by_id]
    else:
        pool = list(ds.samples)
    if args.type_filter:
        pool = [s for s in pool if s.type == args.type_filter]
    print(f"[info] {args.type_filter} pool size: {len(pool)}")
    rng.shuffle(pool)

    pairs = []
    reasons = Counter()
    for s in pool:
        pair = build_pair(s, rng, k_extras=args.k_extras)
        if pair is None:
            reasons["build_failed"] += 1
            continue
        if pair["length_ratio"] > args.max_length_ratio:
            reasons["length_ratio_too_high"] += 1
            continue
        pairs.append(pair)
        if len(pairs) >= args.n:
            break

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "pairs.jsonl"
    with open(pairs_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    summary = {
        "n_pairs": len(pairs),
        "pool_size": len(pool),
        "reject_reasons": dict(reasons),
        "seed": args.seed,
        "k_extras": args.k_extras,
        "max_length_ratio": args.max_length_ratio,
        "mean_length_ratio": sum(p["length_ratio"] for p in pairs) / max(1, len(pairs)),
        "mean_low_overlap": sum(p["low_overlap_score"] for p in pairs) / max(1, len(pairs)),
        "mean_high_overlap": sum(p["high_overlap_score"] for p in pairs) / max(1, len(pairs)),
    }
    with open(out_dir / "build_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote {len(pairs)} pairs -> {pairs_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
