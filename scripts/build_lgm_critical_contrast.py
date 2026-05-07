#!/usr/bin/env python3
"""Build L/G/M critical contrast: B0 / B1 / C0 per sample.

B0 = High-L / Low-G / No-M   (neutral-wrapped p1 SF + distractor)       [reused from anti_cue]
B1 = High-L / Low-G / With-M (task_missingness-wrapped p1 SF + distractor) [reused from anti_cue]
C0 = Low-L  / High-G / No-M  (neutral-wrapped p1 SF + neutral-wrapped p2 SF)

Pool: the 100 near-boundary bridge samples in results/anti_cue_tm_n100/pairs.jsonl.
After C0 salience filter we keep the first N=50 (or --n) feasible samples.

Writes:
  results/lgm_critical_contrast/pairs.jsonl
  results/lgm_critical_contrast/build_summary.json
  results/lgm_critical_contrast/audit_sheet.csv
  results/lgm_critical_contrast/audit_report.json
"""
import argparse, csv, json, random, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.hotpotqa import HotpotQADataset
from eval.scorers import normalize_answer
from tools.search_tool import STOPWORDS
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_local_answerability_pairs_v2 import (
    tokens, content_tokens, entity_count, copula_count,
    contains_answer_text, split_sf,
)
from build_anti_cue_specificity import CUES, has_react_leak

NEUTRAL_PRE, NEUTRAL_SUF = CUES["neutral"]
TM_PRE,      TM_SUF      = CUES["task_missingness"]

_COPULA = {"is", "was", "are", "were", "be", "been", "being"}
_CONCLUSION_MARKERS = (
    "therefore", "thus", "hence", "the answer is", "in conclusion",
    "this confirms", "this establishes", "final answer",
    "clearly the answer", "in summary",
)


def extract_p2_sf(sample, answer):
    """Return the SF sentence(s) from the answer-containing paragraph (p2), joined."""
    split = split_sf(sample, answer)
    if split is None:
        return None, None, None
    p1_title, _p1_sents, p1_sf_sent, p2_title = split
    ctx = {t: s for t, s in sample.context}
    p2_sents = ctx.get(p2_title)
    if not p2_sents:
        return None, None, None
    idxs = sorted(i for tt, i in sample.supporting_facts if tt == p2_title)
    p2_sf = [p2_sents[i] for i in idxs if 0 <= i < len(p2_sents)]
    if not p2_sf:
        return None, None, None
    p2_sf_joined = " ".join(p2_sf).strip()
    if not contains_answer_text(p2_sf_joined, answer):
        return None, None, None
    return p1_title, p1_sf_sent, (p2_title, p2_sf_joined)


def answer_candidate_salience(sentence, answer):
    """Coarse salience of answer inside sentence."""
    s = sentence or ""; a = answer or ""
    s_low, a_low = s.lower(), a.lower()
    if a_low not in s_low:
        return "none"
    toks = tokens(s); n_tok = len(toks)
    ans_start = s_low.find(a_low)
    rel_end = (ans_start + len(a_low)) / max(1, len(s_low))
    pre_copula = sum(1 for w in re.findall(r"[a-z]+", s_low[:ans_start]) if w in _COPULA)
    if n_tok <= 15 and pre_copula >= 1:
        return "high"
    if n_tok <= 18 and rel_end > 0.6 and pre_copula >= 1:
        return "high"
    if n_tok <= 35:
        return "medium"
    return "low"


def conclusion_marker_present(text):
    t = (text or "").lower()
    return any(m in t for m in _CONCLUSION_MARKERS)


def direct_answer_sentence_present(p2_sf_sent, answer):
    """True if p2_sf contains a short copula sentence of the form '... is|was {answer}'."""
    sents = re.split(r"(?<=[\.\?\!])\s+", p2_sf_sent or "")
    a_low = (answer or "").lower()
    for s in sents:
        s_low = s.lower()
        if a_low not in s_low:
            continue
        if answer_candidate_salience(s, answer) == "high":
            return True
        # direct "X is A" pattern at end (covers "was changed to X in December 2013")
        m = re.search(r"\b(is|was|are|were|be|being|been)\b[^.]{0,60}" + re.escape(a_low),
                      s_low)
        if m and len(tokens(s)) <= 20:
            return True
    return False


def answer_components(answer):
    parts = re.split(r",|&|\band\b", answer or "", flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) >= 3] or [answer.strip()]


def build_c0_obs(p1_title, p1_sf, p2_title, p2_sf,
                 max_sf_chars=420, max_p2_chars=500):
    sf1 = p1_sf[:max_sf_chars]
    sf2 = p2_sf[:max_p2_chars]
    body1 = f"{NEUTRAL_PRE} {sf1} {NEUTRAL_SUF}".strip()
    body2 = f"{NEUTRAL_PRE} {sf2} {NEUTRAL_SUF}".strip()
    return f"[1] {p1_title}: {body1}\n\n[2] {p2_title}: {body2}"


def c0_feats(obs, q_terms):
    return {
        "char_len": len(obs),
        "tok_len": len(tokens(obs)),
        "q_overlap": sum(1 for t in content_tokens(obs) if t in q_terms),
        "entity_count": entity_count(obs),
        "copula_count": copula_count(obs),
        "react_leak": has_react_leak(obs),
        "paragraph_count": obs.count("\n\n") + 1,
    }


def normalize_anti_cue_rec(r, condition, L, G, M, feat_override=None):
    """Repackage an existing anti_cue record into the new schema."""
    feat = feat_override or r["feat"]
    return {
        "sample_id": r["sample_id"],
        "question": r["question"],
        "gold_answer": r["gold_answer"],
        "gold_answers": r.get("gold_answers") or [r["gold_answer"]],
        "condition": condition,
        "L_label_intended": L,
        "G_label_intended": G,
        "M_label_intended": M,
        "observation": r["obs"],
        "p1_title": r["p1_title"], "p2_title": r["p2_title"],
        "distractor_title": r.get("distractor_title"),
        "baseline_margin": r.get("baseline_margin"),
        "baseline_em_correct": r.get("baseline_em_correct"),
        "feat": feat,
        "missingness_cue_present": (M == 1),
        "answer_string_present": False,
        "answer_candidate_present": False,
        "answer_candidate_salience": "medium" if L == 1 else "low",
        "direct_answer_sentence_present": False,
        "conclusion_marker_present": False,
        "global_sufficiency_verified": bool(G),
        "local_answerability_intended": "high" if L == 1 else "low",
        "included_supporting_facts": [r["p1_title"]],
        "missing_supporting_facts": [r["p2_title"]],
    }


def build_c0_record(sample, b0_rec, rng):
    """Construct the C0 record for a sample, or return (None, reason)."""
    ans = sample.answer.strip()
    ext = extract_p2_sf(sample, ans)
    if ext is None or ext[0] is None:
        return None, "p2_extract_failed"
    p1_title_chk, p1_sf, (p2_title, p2_sf) = ext
    if p1_title_chk != b0_rec["p1_title"]:
        return None, "p1_title_mismatch"
    # Hard rejects for C0 quality
    if answer_candidate_salience(p2_sf, ans) == "high":
        return None, "p2_answer_salience_high"
    if direct_answer_sentence_present(p2_sf, ans):
        return None, "direct_answer_sentence"
    if conclusion_marker_present(p2_sf):
        return None, "conclusion_marker"
    # Construct obs
    obs = build_c0_obs(p1_title_chk, p1_sf, p2_title, p2_sf)
    if has_react_leak(obs):
        return None, "react_leak"
    # Verify global sufficiency: all answer components should appear somewhere in obs
    comps = answer_components(ans)
    obs_n = normalize_answer(obs)
    if not all(normalize_answer(c) in obs_n for c in comps):
        return None, "answer_not_in_obs"
    q_terms = set(content_tokens(sample.question))
    feat = c0_feats(obs, q_terms)
    if feat["react_leak"]:
        return None, "react_leak_feat"
    rec = {
        "sample_id": sample.id,
        "question": sample.question,
        "gold_answer": ans,
        "gold_answers": sample.answers,
        "condition": "C0_lowL_highG_noM",
        "L_label_intended": 0,
        "G_label_intended": 1,
        "M_label_intended": 0,
        "observation": obs,
        "p1_title": p1_title_chk, "p2_title": p2_title,
        "distractor_title": None,
        "p1_sf_sent": p1_sf,
        "p2_sf_sent": p2_sf,
        "baseline_margin": b0_rec["baseline_margin"],
        "baseline_em_correct": b0_rec["baseline_em_correct"],
        "feat": feat,
        "missingness_cue_present": False,
        "answer_string_present": True,
        "answer_candidate_present": True,
        "answer_candidate_salience": answer_candidate_salience(p2_sf, ans),
        "direct_answer_sentence_present": False,
        "conclusion_marker_present": False,
        "global_sufficiency_verified": True,
        "local_answerability_intended": "low",
        "included_supporting_facts": [p1_title_chk, p2_title],
        "missing_supporting_facts": [],
    }
    return rec, "ok"


def automated_audit(pairs, out_path):
    """Strict per-record checks; returns (all_pass: bool, report: dict)."""
    by_sid_cond = {}
    for r in pairs:
        by_sid_cond.setdefault(r["sample_id"], {})[r["condition"]] = r
    issues = Counter()
    per_cond = Counter()
    for sid, cells in by_sid_cond.items():
        for cond in ("B0_highL_lowG_noM", "B1_highL_lowG_withM", "C0_lowL_highG_noM"):
            if cond not in cells:
                issues[f"missing_{cond}"] += 1
                continue
            r = cells[cond]; per_cond[cond] += 1
            obs = r["observation"]
            if has_react_leak(obs):
                issues[f"react_leak_{cond}"] += 1
            for bad in ("Action:", "Final Answer:", "Observation:", "Thought:"):
                if bad.lower() in obs.lower():
                    issues[f"react_token_{cond}"] += 1
            # B conditions: G=0, answer must be absent
            if cond.startswith("B"):
                if contains_answer_text(obs, r["gold_answer"]):
                    issues[f"answer_leak_in_{cond}"] += 1
                if r["G_label_intended"] != 0:
                    issues[f"G_label_wrong_{cond}"] += 1
            # B1 must have missingness cue, B0 must not
            if cond == "B1_highL_lowG_withM":
                if TM_PRE not in obs:
                    issues["B1_cue_missing"] += 1
                # cue must NOT be a search directive
                bad_directives = ("search again", "retrieve", "use Action", "continue searching")
                if any(bd.lower() in obs.lower() for bd in bad_directives):
                    issues["B1_contains_search_directive"] += 1
            if cond == "B0_highL_lowG_noM":
                if TM_PRE in obs or TM_SUF in obs:
                    issues["B0_has_tm_cue"] += 1
            # C0: G=1, direct answer must be absent
            if cond == "C0_lowL_highG_noM":
                if r["direct_answer_sentence_present"]:
                    issues["C0_direct_answer_sentence"] += 1
                if r["conclusion_marker_present"]:
                    issues["C0_conclusion_marker"] += 1
                if r["answer_candidate_salience"] == "high":
                    issues["C0_salience_high"] += 1
                if not r["global_sufficiency_verified"]:
                    issues["C0_G_not_verified"] += 1
                if TM_PRE in obs or TM_SUF in obs:
                    issues["C0_has_tm_cue"] += 1
    report = {
        "n_samples": len(by_sid_cond),
        "n_per_condition": dict(per_cond),
        "issues": dict(issues),
        "all_pass": (len(issues) == 0),
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return report["all_pass"], report


def write_audit_sheet(pairs, csv_path, n_blind=24, seed=0):
    """CSV of blinded observations for human/LLM audit."""
    rng = random.Random(seed)
    # group triplets
    sids = sorted(set(r["sample_id"] for r in pairs))
    rng.shuffle(sids)
    selected = sids[:max(1, n_blind // 3)]
    rows = []; rid = 0
    for sid in selected:
        triplet = [r for r in pairs if r["sample_id"] == sid]
        rng.shuffle(triplet)
        for r in triplet:
            rid += 1
            rows.append({
                "row_id": rid,
                "sample_id": r["sample_id"],
                "question": r["question"],
                "gold_answer": r["gold_answer"],
                "observation": r["observation"],
                "sufficient_to_answer?": "", "looks_like_direct_answer?": "",
                "says_relation_unresolved?": "",
                "_true_condition": r["condition"],
            })
    rng.shuffle(rows)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anti-cue-pairs", default="results/anti_cue_tm_n100/pairs.jsonl",
                    help="Existing N=100 anti-cue pairs file; provides B0 and B1 per sample.")
    ap.add_argument("--hotpot", default="data/hotpotqa/hotpot_dev_distractor_v1.json")
    ap.add_argument("--out-dir", default="results/lgm_critical_contrast")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-audit-samples", type=int, default=24,
                    help="Number of observations in the blind audit sheet (triplets: ~3 rows/sample).")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load existing B0 / B1 per sample
    recs = [json.loads(l) for l in open(args.anti_cue_pairs)]
    sf_neutral = {r["sample_id"]: r for r in recs if r["condition_id"] == "sf_neutral"}
    sf_tm      = {r["sample_id"]: r for r in recs if r["condition_id"] == "sf_task_missingness"}
    common_ids = sorted(set(sf_neutral) & set(sf_tm))
    print(f"[info] anti-cue pool: {len(common_ids)} samples with both B0 and B1")

    # Load HotpotQA
    ds = HotpotQADataset(args.hotpot)
    by_id = {s.id: s for s in ds.samples}

    # Deterministic traversal order: by sample_id lexicographic (stable across runs)
    rng.shuffle(common_ids)

    pairs = []; reasons = Counter(); n_kept = 0
    for sid in common_ids:
        if sid not in by_id:
            reasons["sample_not_in_hotpot"] += 1; continue
        b0 = sf_neutral[sid]; b1 = sf_tm[sid]
        c0, reason = build_c0_record(by_id[sid], b0, rng)
        if c0 is None:
            reasons[reason] += 1; continue
        # Build three records for this sample
        b0_rec = normalize_anti_cue_rec(b0, "B0_highL_lowG_noM", L=1, G=0, M=0)
        b1_rec = normalize_anti_cue_rec(b1, "B1_highL_lowG_withM", L=1, G=0, M=1)
        pairs.extend([b0_rec, b1_rec, c0])
        n_kept += 1
        if n_kept >= args.n:
            break

    pairs_path = out_dir / "pairs.jsonl"
    with open(pairs_path, "w") as f:
        for r in pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[info] kept {n_kept} samples ({len(pairs)} records) -> {pairs_path}")
    print(f"[info] reject reasons: {dict(reasons)}")

    # Automated audit
    all_pass, audit_rep = automated_audit(pairs, out_dir / "audit_report.json")

    # Blind audit sheet
    audit_sheet_path = out_dir / "audit_sheet.csv"
    write_audit_sheet(pairs, audit_sheet_path, n_blind=args.n_audit_samples, seed=args.seed)

    # Build summary
    def cell_stats(cond):
        rs = [r for r in pairs if r["condition"] == cond]
        if not rs: return {}
        n = len(rs)
        return {
            "n": n,
            "mean_tok_len": sum(r["feat"]["tok_len"] for r in rs) / n,
            "mean_q_overlap": sum(r["feat"]["q_overlap"] for r in rs) / n,
            "mean_entity_count": sum(r["feat"]["entity_count"] for r in rs) / n,
            "mean_copula_count": sum(r["feat"]["copula_count"] for r in rs) / n,
            "mean_paragraph_count": sum(r["feat"].get("paragraph_count", 2) for r in rs) / n,
            "n_answer_string_present": sum(int(r["answer_string_present"]) for r in rs),
            "n_answer_candidate_present": sum(int(r["answer_candidate_present"]) for r in rs),
            "salience_dist": dict(Counter(r["answer_candidate_salience"] for r in rs)),
            "n_direct_answer_sentence": sum(int(r["direct_answer_sentence_present"]) for r in rs),
            "n_conclusion_marker": sum(int(r["conclusion_marker_present"]) for r in rs),
            "n_missingness_cue": sum(int(r["missingness_cue_present"]) for r in rs),
        }
    summary = {
        "n_samples": n_kept,
        "n_records": len(pairs),
        "reject_reasons": dict(reasons),
        "cells": {c: cell_stats(c) for c in
                  ("B0_highL_lowG_noM", "B1_highL_lowG_withM", "C0_lowL_highG_noM")},
        "audit_all_pass": all_pass,
        "audit_issues": audit_rep["issues"],
        "audit_sheet_path": str(audit_sheet_path),
    }
    with open(out_dir / "build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== build summary ===")
    print(json.dumps(summary, indent=2))
    if not all_pass:
        print("\n[FAIL] automated audit found issues; DO NOT evaluate until resolved.")
        sys.exit(2)
    print("\n[OK] audit passed.")


if __name__ == "__main__":
    main()
