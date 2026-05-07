#!/usr/bin/env python3
"""Anti-cue specificity test: 2 target locations x 4 cue types.

Cells (8 total, per sample): target in {sf, distractor} x cue in
{neutral, task_missingness, generic_incompleteness, action_directive}.
Only the wrapper text around the targeted paragraph changes; SF + distractor
content and paragraph order are constant. Global insufficiency preserved
(P2 never appears; answer never leaks). Produces single pairs.jsonl with one
record per (sample_id, target, cue).
"""
import argparse, json, re, sys, random
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.hotpotqa import HotpotQADataset
from eval.scorers import normalize_answer
from tools.search_tool import STOPWORDS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_local_answerability_pairs_v2 import (
    tokens, content_tokens, entity_count, copula_count,
    contains_answer_text, split_sf, pick_distractor, load_near_boundary_ids,
)

CUES = {
    "neutral": (
        "The retrieved text includes the following passage.",
        "End of the retrieved passage.",
    ),
    "task_missingness": (
        "The passage leaves the relation needed by the question unresolved.",
        "That relation remains unresolved here.",
    ),
    "generic_incompleteness": (
        "The passage is incomplete on one contextual detail.",
        "One background detail remains unspecified here.",
    ),
    "action_directive": (
        "Additional retrieval is needed before this can be resolved.",
        "Another retrieval step is required here.",
    ),
}

_REACT_LEAK_RE = re.compile(
    r"(?:\bAction\s*:|\bSearch\s*:|\bFinal\s+Answer\s*:|\bObservation\s*:|\bThought\s*:)",
    re.IGNORECASE,
)


def has_react_leak(text):
    return bool(_REACT_LEAK_RE.search(text or ""))


def build_obs(p1_title, p1_sf_sent, distractor, target, cue_prefix, cue_suffix,
              max_sf_chars=420, max_dist_chars=500):
    sf_trunc = p1_sf_sent[:max_sf_chars]
    d_title, d_text = distractor
    d_trunc = d_text[:max_dist_chars]
    if target == "sf":
        body1 = f"{cue_prefix} {sf_trunc} {cue_suffix}".strip()
        return f"[1] {p1_title}: {body1}\n\n[2] {d_title}: {d_trunc}"
    if target == "distractor":
        body2 = f"{cue_prefix} {d_trunc} {cue_suffix}".strip()
        return f"[1] {p1_title}: {sf_trunc}\n\n[2] {d_title}: {body2}"
    raise ValueError(target)


def feats(text, q_terms):
    return {
        "char_len": len(text),
        "tok_len": len(tokens(text)),
        "q_overlap": sum(1 for t in content_tokens(text) if t in q_terms),
        "entity_count": entity_count(text),
        "copula_count": copula_count(text),
        "react_leak": has_react_leak(text),
    }


def build_sample_all_cells(sample, rng, active_cues, min_sf_len=4):
    ans = sample.answer.strip()
    if not ans:
        return None, "empty_answer"
    split = split_sf(sample, ans)
    if split is None:
        return None, "split_failed"
    p1_title, _p1_sents, p1_sf_sent, p2_title = split
    if len(tokens(p1_sf_sent)) < min_sf_len:
        return None, "sf_too_short"
    distractor = pick_distractor(sample, p1_title, p2_title, ans, rng)
    if distractor is None:
        return None, "no_distractor"

    q_terms = set(content_tokens(sample.question))
    recs = []
    for target in ("sf", "distractor"):
        for cue_name in active_cues:
            pref, suf = CUES[cue_name]
            obs = build_obs(p1_title, p1_sf_sent, distractor, target, pref, suf)
            if contains_answer_text(obs, ans):
                return None, f"answer_leak_{target}_{cue_name}"
            f = feats(obs, q_terms)
            if f["react_leak"]:
                return None, f"react_leak_{target}_{cue_name}"
            recs.append({
                "sample_id": sample.id,
                "question": sample.question,
                "gold_answer": ans,
                "gold_answers": sample.answers,
                "p1_title": p1_title, "p2_title": p2_title,
                "distractor_title": distractor[0],
                "target": target, "cue": cue_name,
                "condition_id": f"{target}_{cue_name}",
                "obs": obs, "feat": f,
                "answer_present": False, "global_sufficiency_verified": True,
                "cue_prefix": pref, "cue_suffix": suf,
                "wrapper_tok_len": len(tokens(pref + " " + suf)),
            })

    for tgt in ("sf", "distractor"):
        cells = [r for r in recs if r["target"] == tgt]
        toks = [c["feat"]["tok_len"] for c in cells]
        ent  = [c["feat"]["entity_count"] for c in cells]
        ov   = [c["feat"]["q_overlap"] for c in cells]
        if max(toks) / max(1, min(toks)) > 1.05:
            return None, f"length_ratio_out_of_band_{tgt}"
        if max(ent) - min(ent) > 1:
            return None, f"entity_diff_too_large_{tgt}"
        if max(ov) - min(ov) > 1:
            return None, f"q_overlap_diff_too_large_{tgt}"
    return recs, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpot", default="data/hotpotqa/hotpot_dev_distractor_v1.json")
    ap.add_argument("--baseline", default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--out-dir", default="results/anti_cue_specificity")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--margin-lo", type=float, default=-7.0)
    ap.add_argument("--margin-hi", type=float, default=-1.0)
    ap.add_argument("--type-filter", default="bridge")
    ap.add_argument("--cues", default=",".join(CUES.keys()),
                    help="Comma-separated cue keys to build (default: all 4).")
    args = ap.parse_args()

    active_cues = [c.strip() for c in args.cues.split(",") if c.strip()]
    for c in active_cues:
        if c not in CUES:
            raise ValueError(f"unknown cue: {c}; choices={list(CUES.keys())}")
    if "neutral" not in active_cues:
        raise ValueError("'neutral' must be in --cues (used as reference)")

    rng = random.Random(args.seed)
    nb_ids = load_near_boundary_ids(args.baseline, args.margin_lo, args.margin_hi)
    ds = HotpotQADataset(args.hotpot)
    by_id = {s.id: s for s in ds.samples}
    pool = [by_id[i] for i in nb_ids if i in by_id]
    if args.type_filter:
        pool = [s for s in pool if s.type == args.type_filter]
    rng.shuffle(pool)
    print(f"[info] near-boundary {args.type_filter} pool: {len(pool)}")

    all_records, reasons, n_s = [], Counter(), 0
    for s in pool:
        recs, reason = build_sample_all_cells(s, rng, active_cues)
        if recs is None:
            reasons[reason] += 1
            continue
        for r in recs:
            r["baseline_margin"] = nb_ids[s.id]["baseline_margin"]
            r["baseline_em_correct"] = nb_ids[s.id]["em_correct"]
        all_records.extend(recs); n_s += 1
        if n_s >= args.n:
            break

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pairs.jsonl", "w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    per_cell = {}
    for r in all_records:
        per_cell.setdefault(r["condition_id"], []).append(r)
    cells = {
        k: {
            "n": len(v),
            "mean_tok_len": sum(x["feat"]["tok_len"] for x in v) / len(v),
            "mean_q_overlap": sum(x["feat"]["q_overlap"] for x in v) / len(v),
            "mean_entity_count": sum(x["feat"]["entity_count"] for x in v) / len(v),
            "mean_copula_count": sum(x["feat"]["copula_count"] for x in v) / len(v),
            "n_react_leak": sum(int(x["feat"]["react_leak"]) for x in v),
            "n_answer_leak": sum(int(x["answer_present"]) for x in v),
            "wrapper_tok_len": v[0]["wrapper_tok_len"] if v else 0,
        } for k, v in per_cell.items()
    }
    summary = {
        "n_samples": n_s, "n_records": len(all_records),
        "reject_reasons": dict(reasons),
        "cues": active_cues, "targets": ["sf", "distractor"],
        "cells": cells,
    }
    with open(out_dir / "build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] wrote {len(all_records)} records across {n_s} samples -> {out_dir}/pairs.jsonl")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
