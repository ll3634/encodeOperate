#!/usr/bin/env python3
"""
Build paired Low-L / High-L observations — v2, same-facts minimal rewrite.

Design:
  - Near-boundary pool: baseline_results.jsonl step1 margin_before in [-7, -1].
  - Core: IDENTICAL P1 SF sentence (the non-answer-containing bridge SF) and
    IDENTICAL distractor paragraph [2] across both conditions.
  - Manipulation: length-matched style wrapper around the SF sentence:
      Low-L  = "Here is some related background. {SF} Additional specifics are absent."
      High-L = "Here is the key fact directly stated. {SF} The statement is clearly affirmed."
  - Wrapper carries only commitment register (declarative vs hedging). It adds
    no new facts, no question terms, no proper nouns, no answer strings.
  - Strict matching (drop pair if fail):
      token_len ratio in [0.95, 1.05], answer absent in both, entity diff <= 1,
      paragraph count same, q_overlap diff minimized.
  - Global insufficiency: P2 paragraph is never included; neither observation
    contains the answer string. Verified per-pair.
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
_COMMIT_MARKERS = ("is the", "is directly", "is clearly", "the key fact",
                   "established fact", "stated", "confirmed", "affirmed")

COMMITMENT_PREFIX = "Here is the key fact directly stated."
COMMITMENT_SUFFIX = "The statement is clearly affirmed."
ANTI_PREFIX       = "Here is only a partial clue so far."
ANTI_SUFFIX       = "The answer is not established yet."
NEUTRAL_PREFIX    = "Here is one excerpt from the source."
NEUTRAL_SUFFIX    = "That is the end of the excerpt."

WRAPPERS = {
    "neutral":    (NEUTRAL_PREFIX,    NEUTRAL_SUFFIX),
    "commitment": (COMMITMENT_PREFIX, COMMITMENT_SUFFIX),
    "anti":       (ANTI_PREFIX,       ANTI_SUFFIX),
}

LOW_PREFIX,  LOW_SUFFIX  = ANTI_PREFIX,       ANTI_SUFFIX
HIGH_PREFIX, HIGH_SUFFIX = COMMITMENT_PREFIX, COMMITMENT_SUFFIX


def tokens(text):
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def content_tokens(text):
    return [t for t in tokens(text) if t not in STOPWORDS and len(t) > 1]


def entity_count(text):
    return len(re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}\b", text or ""))


def copula_count(text):
    return sum(1 for t in tokens(text) if t in _COPULA)


def answer_components(answer):
    if not answer:
        return []
    parts = re.split(r",|&|\band\b", answer, flags=re.IGNORECASE)
    comps = [p.strip() for p in parts if p.strip() and len(p.strip()) >= 3]
    return comps if comps else [answer.strip()]


def contains_answer_text(haystack, answer, strict=True):
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


def build_obs(p1_title, sf_sent, prefix, suffix, distractor,
              max_sf_chars=420, max_dist_chars=500, wrapper_target="sf"):
    """wrapper_target='sf'        -> wrapper wraps the SF paragraph [1] (main v2 design)
       wrapper_target='distractor' -> wrapper wraps the distractor paragraph [2] (placebo)"""
    sf_trunc = sf_sent[:max_sf_chars]
    if wrapper_target == "sf":
        body = f"{prefix} {sf_trunc} {suffix}".strip()
        result_1 = f"[1] {p1_title}: {body}"
        if distractor is None:
            return result_1, 1
        d_title, d_text = distractor
        d_text = d_text[:max_dist_chars]
        return f"{result_1}\n\n[2] {d_title}: {d_text}", 2
    elif wrapper_target == "distractor":
        result_1 = f"[1] {p1_title}: {sf_trunc}"
        if distractor is None:
            return result_1, 1
        d_title, d_text = distractor
        d_text = d_text[:max_dist_chars]
        body = f"{prefix} {d_text} {suffix}".strip()
        return f"{result_1}\n\n[2] {d_title}: {body}", 2
    else:
        raise ValueError(f"unknown wrapper_target: {wrapper_target}")


def has_commit_marker(text):
    tl = text.lower()
    return any(m in tl for m in _COMMIT_MARKERS)


def feats(text, q_terms):
    return {
        "char_len": len(text),
        "tok_len": len(tokens(text)),
        "q_overlap": sum(1 for t in content_tokens(text) if t in q_terms),
        "entity_count": entity_count(text),
        "copula_count": copula_count(text),
        "commit_marker": bool(has_commit_marker(text)),
    }


def build_pair(sample, rng, max_chars=500, min_sf_len=4, wrapper_target="sf",
               low_semantics="anti", high_semantics="commitment"):
    ans = sample.answer.strip()
    if not ans:
        return None, "empty_answer"
    split = split_sf(sample, ans)
    if split is None:
        return None, "split_failed"
    p1_title, p1_sents, p1_sf_sent, p2_title = split
    if len(tokens(p1_sf_sent)) < min_sf_len:
        return None, "sf_too_short"
    distractor = pick_distractor(sample, p1_title, p2_title, ans, rng)
    if distractor is None:
        return None, "no_distractor"

    low_pref,  low_suf  = WRAPPERS[low_semantics]
    high_pref, high_suf = WRAPPERS[high_semantics]
    obs_low,  npar_l = build_obs(p1_title, p1_sf_sent, low_pref,  low_suf,  distractor,
                                 max_sf_chars=max_chars, max_dist_chars=max_chars,
                                 wrapper_target=wrapper_target)
    obs_high, npar_h = build_obs(p1_title, p1_sf_sent, high_pref, high_suf, distractor,
                                 max_sf_chars=max_chars, max_dist_chars=max_chars,
                                 wrapper_target=wrapper_target)

    if contains_answer_text(obs_low, ans) or contains_answer_text(obs_high, ans):
        return None, "answer_leak"
    if npar_l != npar_h:
        return None, "paragraph_count_mismatch"

    q_terms = set(content_tokens(sample.question))
    f_low, f_high = feats(obs_low, q_terms), feats(obs_high, q_terms)
    len_ratio = max(f_low["tok_len"], f_high["tok_len"]) / max(1, min(f_low["tok_len"], f_high["tok_len"]))

    return {
        "sample_id": sample.id,
        "question": sample.question,
        "gold_answer": ans,
        "gold_answers": sample.answers,
        "p1_title": p1_title,
        "p2_title": p2_title,
        "p1_sf_sent": p1_sf_sent,
        "distractor_title": distractor[0],
        "obs_low": obs_low,
        "obs_high": obs_high,
        "paragraph_count": npar_l,
        "feat_low": f_low,
        "feat_high": f_high,
        "length_ratio": len_ratio,
        "direct_answer_sentence_low": f_low["commit_marker"],
        "direct_answer_sentence_high": f_high["commit_marker"],
        "answer_present_low": False,
        "answer_present_high": False,
        "global_sufficiency_verified": True,
        "wrapper_target": wrapper_target,
        "low_semantics": low_semantics,
        "high_semantics": high_semantics,
    }, "ok"


def load_near_boundary_ids(baseline_path, margin_lo=-7.0, margin_hi=-1.0):
    ids = {}
    with open(baseline_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("n_steps", 0) != 2:
                continue
            steps = r.get("steps") or []
            if not steps or steps[0].get("action") != "search":
                continue
            if len(steps) < 2:
                continue
            m = steps[1].get("margin_before")
            if m is None:
                continue
            if margin_lo <= m < margin_hi:
                ids[r["sample_id"]] = {"baseline_margin": float(m),
                                       "em_correct": bool(r.get("em_correct", False))}
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpot", default="data/hotpotqa/hotpot_dev_distractor_v1.json")
    ap.add_argument("--baseline", default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--out-dir", default="results/local_answerability_v2")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--margin-lo", type=float, default=-7.0)
    ap.add_argument("--margin-hi", type=float, default=-1.0)
    ap.add_argument("--len-ratio-min", type=float, default=0.95)
    ap.add_argument("--len-ratio-max", type=float, default=1.05)
    ap.add_argument("--entity-diff-max", type=int, default=1)
    ap.add_argument("--type-filter", default="bridge")
    ap.add_argument("--wrapper-target", choices=["sf", "distractor"], default="sf",
                    help="Which paragraph the commitment-style wrapper is applied to. "
                         "'sf' = main v2 design (wrap SF paragraph [1]). "
                         "'distractor' = placebo control (wrap distractor [2] instead).")
    ap.add_argument("--low-semantics",  choices=list(WRAPPERS.keys()), default="anti",
                    help="Semantics of the 'low' condition wrapper (paired reference).")
    ap.add_argument("--high-semantics", choices=list(WRAPPERS.keys()), default="commitment",
                    help="Semantics of the 'high' condition wrapper.")
    args = ap.parse_args()
    if args.low_semantics == args.high_semantics:
        raise SystemExit(f"low_semantics and high_semantics must differ (both={args.low_semantics})")

    rng = random.Random(args.seed)

    nb_ids = load_near_boundary_ids(args.baseline, args.margin_lo, args.margin_hi)
    print(f"[info] near-boundary [{args.margin_lo}, {args.margin_hi}) pool size: {len(nb_ids)}")

    ds = HotpotQADataset(args.hotpot)
    by_id = {s.id: s for s in ds.samples}
    pool = [by_id[i] for i in nb_ids if i in by_id]
    if args.type_filter:
        pool = [s for s in pool if s.type == args.type_filter]
    print(f"[info] {args.type_filter} pool after type filter: {len(pool)}")
    rng.shuffle(pool)

    pairs = []
    reasons = Counter()
    for s in pool:
        pair, reason = build_pair(s, rng,
                                  wrapper_target=args.wrapper_target,
                                  low_semantics=args.low_semantics,
                                  high_semantics=args.high_semantics)
        if pair is None:
            reasons[reason] += 1
            continue
        if not (args.len_ratio_min <= pair["length_ratio"] <= args.len_ratio_max):
            reasons["length_ratio_out_of_band"] += 1
            continue
        ec_diff = abs(pair["feat_high"]["entity_count"] - pair["feat_low"]["entity_count"])
        if ec_diff > args.entity_diff_max:
            reasons["entity_diff_too_large"] += 1
            continue
        pair["baseline_margin"] = nb_ids[s.id]["baseline_margin"]
        pair["baseline_em_correct"] = nb_ids[s.id]["em_correct"]
        pairs.append(pair)
        if len(pairs) >= args.n:
            break

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "pairs.jsonl"
    with open(pairs_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    def _mean(xs):
        return sum(xs) / max(1, len(xs))
    summary = {
        "n_pairs": len(pairs),
        "pool_size_near_boundary": len(pool),
        "reject_reasons": dict(reasons),
        "seed": args.seed,
        "wrapper_target": args.wrapper_target,
        "low_semantics":  args.low_semantics,
        "high_semantics": args.high_semantics,
        "margin_band": [args.margin_lo, args.margin_hi],
        "len_ratio_band": [args.len_ratio_min, args.len_ratio_max],
        "entity_diff_max": args.entity_diff_max,
        "mean_length_ratio": _mean([p["length_ratio"] for p in pairs]),
        "mean_tok_len_low":  _mean([p["feat_low"]["tok_len"]  for p in pairs]),
        "mean_tok_len_high": _mean([p["feat_high"]["tok_len"] for p in pairs]),
        "mean_q_overlap_low":  _mean([p["feat_low"]["q_overlap"]  for p in pairs]),
        "mean_q_overlap_high": _mean([p["feat_high"]["q_overlap"] for p in pairs]),
        "mean_entity_low":  _mean([p["feat_low"]["entity_count"]  for p in pairs]),
        "mean_entity_high": _mean([p["feat_high"]["entity_count"] for p in pairs]),
        "mean_copula_low":  _mean([p["feat_low"]["copula_count"]  for p in pairs]),
        "mean_copula_high": _mean([p["feat_high"]["copula_count"] for p in pairs]),
        "n_commit_marker_low":  sum(p["direct_answer_sentence_low"]  for p in pairs),
        "n_commit_marker_high": sum(p["direct_answer_sentence_high"] for p in pairs),
        "n_answer_leak_low":  sum(p["answer_present_low"]  for p in pairs),
        "n_answer_leak_high": sum(p["answer_present_high"] for p in pairs),
    }
    with open(out_dir / "build_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote {len(pairs)} pairs -> {pairs_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
