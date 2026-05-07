#!/usr/bin/env python3
"""Build N0/T0/S0 extractability replication on MuSiQue (2-hop bridge).

Per 2-hop bridge example with hop_1=bridge, hop_2=answer:
  W = final answer (extractable in support_para_2).
  S0 = support_para_1 + support_para_2 + 2 distractors (full chain).
  T0 = support_para_2 + 3 distractors (W extractable, bridge withheld).
  N0 = 4 distractors only (no W, no gold leakage).

Outputs (in --out-dir):
  dataset_name.txt          dataset id
  build_candidates.jsonl    every candidate considered (one per source example)
  rejected_candidates.jsonl rejected candidates with reason
  selected_examples.jsonl   selected source examples (one per kept sample)
  pairs.jsonl               final {sample_id, condition, ...} records (3 per sample)
  audit_sheet.csv           20 random + 10 edge + barely-pass
  build_summary.json        QC counters
"""
import argparse, csv, json, random, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_anti_cue_specificity import has_react_leak  # noqa: E402

DATASET = "MuSiQue (musique_ans_v1.0_dev, 2hop bridge)"

_WORD = re.compile(r"[A-Za-z0-9]+")


def _tok(s):
    return _WORD.findall(s or "")


def _norm(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _contains(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    h = " " + _norm(haystack) + " "
    n = _norm(needle)
    return f" {n} " in h or h.strip().endswith(" " + n) or h.strip().startswith(n + " ")


def _build_obs(paragraphs):
    parts = []
    for i, p in enumerate(paragraphs, 1):
        title = (p.get("title") or "Untitled").strip()
        body = (p.get("paragraph_text") or "").strip()
        parts.append(f"[{i}] {title}: {body}")
    return "\n\n".join(parts)


REACT_TOKENS = ("Action:", "Final Answer:", "Observation:", "Action Input:")


def _has_protocol_leak(obs: str) -> bool:
    if has_react_leak(obs):
        return True
    return any(tok in obs for tok in REACT_TOKENS)


def _aliases(rec):
    out = {rec.get("answer", "")}
    for a in rec.get("answer_aliases") or []:
        if isinstance(a, str):
            out.add(a)
    # add bridge answer too — it is part of the gold reasoning chain.
    for d in rec.get("question_decomposition", []) or []:
        if isinstance(d.get("answer"), str):
            out.add(d["answer"])
    return [a for a in out if a]


def _eligible_distractors(rec, support_idxs, forbid_strs):
    """Distractor pool: not support, not containing any forbidden string."""
    pool = []
    for p in rec.get("paragraphs", []):
        if p.get("idx") in support_idxs:
            continue
        body = p.get("paragraph_text", "") + " " + p.get("title", "")
        if any(_contains(body, s) for s in forbid_strs):
            continue
        pool.append(p)
    return pool


def _build_conditions(rec, rng):
    decomp = rec["question_decomposition"]
    idx1 = decomp[0]["paragraph_support_idx"]
    idx2 = decomp[1]["paragraph_support_idx"]
    paras = {p["idx"]: p for p in rec["paragraphs"]}
    sup1, sup2 = paras[idx1], paras[idx2]
    W = rec["answer"].strip()
    bridge = decomp[0]["answer"].strip()
    forbid_n0 = list({a for a in _aliases(rec)} | {W, bridge})
    distractors = _eligible_distractors(rec, {idx1, idx2}, forbid_n0)
    if len(distractors) < 4:
        return None, "too_few_clean_distractors"
    rng.shuffle(distractors)
    # S0: sup1 + sup2 + 2 distractors (4 paragraphs, shuffled).
    s0_paras = [sup1, sup2] + distractors[:2]
    rng.shuffle(s0_paras)
    # T0: sup2 + 3 distractors (W extractable, bridge withheld).
    t0_paras = [sup2] + distractors[2:5] if len(distractors) >= 5 else None
    if t0_paras is None:
        # fall back: reuse some other distractors
        t0_paras = [sup2] + distractors[:3]
        if any(p["idx"] in (idx1, idx2) and p["idx"] != idx2 for p in t0_paras):
            return None, "t0_pool_collision"
    rng.shuffle(t0_paras)
    # N0: 4 distractors with no W/bridge leak (already filtered).
    n0_pool = distractors[:8]
    if len(n0_pool) < 4:
        return None, "n0_pool_too_small"
    n0_paras = list(n0_pool[:4])
    rng.shuffle(n0_paras)
    return {"N0": n0_paras, "T0": t0_paras, "S0": s0_paras}, None


def _qc(obs, W, gold_aliases, condition):
    issues = []
    obs_low = " " + _norm(obs) + " "
    w_present = _contains(obs, W)
    gold_present = any(_contains(obs, g) for g in gold_aliases)
    if condition == "N0":
        if w_present:
            issues.append("N0_W_present")
        if gold_present:
            issues.append("N0_gold_present")
    else:
        if not w_present:
            issues.append(f"{condition}_W_absent")
    if _has_protocol_leak(obs):
        issues.append(f"{condition}_react_leak")
    return issues, w_present, gold_present



def _expected_type_label(question: str) -> str:
    """Cheap relation-tail heuristic for `expected_answer_type`."""
    q = question.lower()
    for k in ("country", "city", "state", "place", "where", "located"):
        if k in q: return "location"
    for k in ("who", "spouse", "father", "mother", "wife", "husband",
             "child", "sibling", "performer", "author", "director", "founder",
             "owner", "ceo", "president", "actor"):
        if k in q: return "person_or_org"
    for k in ("when", "year", "date", "century"):
        if k in q: return "date"
    for k in ("language", "religion", "currency", "genre"):
        if k in q: return "concept"
    return "entity"


def build_record(rec, rng):
    decomp = rec.get("question_decomposition") or []
    if len(decomp) != 2:
        return None, None, "not_2hop"
    idx1 = decomp[0].get("paragraph_support_idx")
    idx2 = decomp[1].get("paragraph_support_idx")
    if idx1 is None or idx2 is None or idx1 == idx2:
        return None, None, "missing_or_dup_support_idx"
    paras_by_idx = {p["idx"]: p for p in rec.get("paragraphs", [])}
    if idx1 not in paras_by_idx or idx2 not in paras_by_idx:
        return None, None, "support_idx_oob"
    W = (rec.get("answer") or "").strip()
    if not W or len(W) > 60 or len(W) < 1:
        return None, None, "answer_length_oob"
    if str(decomp[1].get("answer", "")).strip() != W:
        return None, None, "hop2_answer_mismatch"
    sup2_text = paras_by_idx[idx2].get("paragraph_text", "")
    if not _contains(sup2_text, W):
        return None, None, "answer_not_extractable_in_sup2"
    cond_paras, why = _build_conditions(rec, rng)
    if cond_paras is None:
        return None, None, why

    aliases = _aliases(rec)
    sample_id = "musique_" + rec["id"]
    out = {
        "sample_id": sample_id,
        "dataset": DATASET,
        "musique_id": rec["id"],
        "question": rec["question"],
        "gold_answer": W,
        "gold_answers": list({W, *(rec.get("answer_aliases") or [])}),
        "candidate_W": W,
        "expected_answer_type": _expected_type_label(rec["question"]),
        "support_paragraph_ids": [idx1, idx2],
        "bridge_entity": decomp[0].get("answer"),
    }
    qc_issues = []
    feat = {}
    for cond in ("N0", "T0", "S0"):
        obs = _build_obs(cond_paras[cond])
        issues, w_present, g_present = _qc(obs, W, aliases, cond)
        qc_issues.extend(issues)
        out[f"{cond}_observation"] = obs
        out[f"{cond}_paragraph_idxs"] = [p["idx"] for p in cond_paras[cond]]
        feat[cond] = {
            "tok_len": len(_tok(obs)),
            "char_len": len(obs),
            "n_paragraphs": len(cond_paras[cond]),
            "W_present": w_present,
            "gold_present": g_present,
            "react_leak": _has_protocol_leak(obs),
        }
    feat["T0"]["support_complete"] = False
    feat["S0"]["support_complete"] = True
    feat["N0"]["support_complete"] = False
    out["features"] = feat
    out["construction_notes"] = (
        f"2hop bridge: hop1={decomp[0].get('question')!r} -> "
        f"{decomp[0].get('answer')!r}; hop2={decomp[1].get('question')!r} -> {W!r}. "
        "S0 keeps both supports + 2 distractors; T0 drops hop1 (bridge withheld); "
        "N0 uses 4 distractors filtered against W/aliases/bridge."
    )

    toks = [feat[c]["tok_len"] for c in ("N0", "T0", "S0")]
    if min(toks) > 0 and max(toks) / min(toks) > 1.35:
        qc_issues.append("length_ratio_out_of_band")

    out["qc_issues"] = qc_issues
    return out, feat, None



def _to_pair_records(rec):
    """Expand a built record into 3 condition rows compatible with eval pipeline."""
    out = []
    for cond in ("N0", "T0", "S0"):
        out.append({
            "sample_id": rec["sample_id"],
            "schema_type": "musique_2hop_bridge",
            "condition": cond,
            "condition_id": cond,
            "question": rec["question"],
            "gold_answer": rec["gold_answer"] if cond == "S0" else None,
            "gold_answers": rec["gold_answers"] if cond == "S0" else [],
            "candidate_W": rec["candidate_W"],
            "W": rec["candidate_W"],
            "expected_answer_type": rec["expected_answer_type"],
            "obs": rec[f"{cond}_observation"],
            "observation": rec[f"{cond}_observation"],
            "paragraph_idxs": rec[f"{cond}_paragraph_idxs"],
            "support_paragraph_ids": rec["support_paragraph_ids"],
            "bridge_entity": rec.get("bridge_entity"),
            "musique_id": rec.get("musique_id"),
            "feat": rec["features"][cond],
            "E_intended": cond in ("T0", "S0"),
            "S_intended": cond == "S0",
        })
    return out


def _audit_rows(built):
    """One row per (sample, condition) suitable for blind audit."""
    rows = []
    for r in built:
        for cond in ("N0", "T0", "S0"):
            f = r["features"][cond]
            rows.append({
                "sample_id":          r["sample_id"],
                "condition":          cond,
                "question":           r["question"],
                "gold":               r["gold_answer"],
                "W":                  r["candidate_W"],
                "obs":                r[f"{cond}_observation"],
                "N0_W_present":       r["features"]["N0"]["W_present"],
                "T0_W_present":       r["features"]["T0"]["W_present"],
                "S0_W_present":       r["features"]["S0"]["W_present"],
                "T0_support_complete": r["features"]["T0"]["support_complete"],
                "S0_support_complete": r["features"]["S0"]["support_complete"],
                "tok_len":            f["tok_len"],
                "n_paragraphs":       f["n_paragraphs"],
                "react_leak":         f["react_leak"],
                "qc_issues":          ";".join(r["qc_issues"]),
                "audit_question":     "Does T0 contain W but lack support? Does S0 support W? Does N0 avoid W?",
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",
        default="data/musique/musique_ans_v1.0_dev.jsonl",
        help="Path to musique_ans_v1.0_dev.jsonl (relative to e2e_agent/).")
    ap.add_argument("--out-dir", default="results/second_benchmark_extractability")
    ap.add_argument("--n", type=int, default=50,
                    help="Number of selected samples to keep.")
    ap.add_argument("--n-pool", type=int, default=80,
                    help="Number of candidate samples to BUILD before filtering "
                         "down to --n. (Spec: at least 80.)")
    ap.add_argument("--seed", type=int, default=20260426)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    src_path = Path(args.data)
    if not src_path.is_absolute():
        src_path = Path(__file__).resolve().parent.parent / src_path
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    # Stream candidates, keep 2-hop only.
    src = []
    with open(src_path) as f:
        for line in f:
            r = json.loads(line)
            if (r.get("id") or "").startswith("2hop__"):
                src.append(r)
    rng.shuffle(src)
    print(f"[info] {len(src)} 2-hop candidates in source")

    built, rejected = [], []
    reject_counter = Counter()
    for r in src:
        if len(built) >= args.n_pool:
            break
        rec, feat, why = build_record(r, rng)
        if why:
            rejected.append({"musique_id": r.get("id"), "reason": why,
                             "question": r.get("question"),
                             "answer": r.get("answer")})
            reject_counter[why] += 1
            continue
        if rec.get("qc_issues"):
            for q in rec["qc_issues"]:
                reject_counter["qc:" + q] += 1
            rejected.append({"musique_id": r.get("id"),
                             "reason": "qc:" + ";".join(rec["qc_issues"]),
                             "question": r.get("question"),
                             "answer": r.get("answer")})
            continue
        built.append(rec)

    print(f"[info] built {len(built)} clean candidates "
          f"(rejected {len(rejected)})")

    # Final selection: keep the first --n built (already shuffled).
    selected = built[:args.n]
    print(f"[info] selected {len(selected)} for the official run")

    # Outputs.
    (out_dir / "dataset_name.txt").write_text(DATASET + "\n")

    with open(out_dir / "build_candidates.jsonl", "w") as f:
        for r in built:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "rejected_candidates.jsonl", "w") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "selected_examples.jsonl", "w") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pair_rows = [pr for r in selected for pr in _to_pair_records(r)]
    with open(out_dir / "pairs.jsonl", "w") as f:
        for pr in pair_rows:
            f.write(json.dumps(pr, ensure_ascii=False) + "\n")

    # audit_sheet.csv: 20 random + 10 edge (smallest length parity margin) + barely-pass.
    rng_a = random.Random(args.seed + 1)
    audit_pool = _audit_rows(selected)
    by_sample = {}
    for row in audit_pool:
        by_sample.setdefault(row["sample_id"], []).append(row)
    sample_ids = list(by_sample.keys())
    rng_a.shuffle(sample_ids)
    chosen_ids = set(sample_ids[:20])

    # Edge: 10 with the largest length spread.
    def spread(sid):
        toks = [by_sample[sid][i]["tok_len"] for i in range(len(by_sample[sid]))]
        return max(toks) / max(1, min(toks))
    edge_ids = sorted(sample_ids, key=spread, reverse=True)[:10]
    chosen_ids.update(edge_ids)
    # Barely-pass: any sample whose qc_issues column is non-empty (=== should be empty
    # by construction since rec["qc_issues"] was the rejection gate; include any with
    # T0 W_present false or S0 length close to T0 length border).
    for sid in sample_ids:
        rows = by_sample[sid]
        toks = [r["tok_len"] for r in rows]
        if max(toks) / max(1, min(toks)) > 1.20:
            chosen_ids.add(sid)
    audit_rows = [r for sid in chosen_ids for r in by_sample[sid]]

    audit_path = out_dir / "audit_sheet.csv"
    if audit_rows:
        with open(audit_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
            w.writeheader()
            for row in audit_rows:
                w.writerow(row)

    # Per-cell summary stats.
    per_cond = {c: [] for c in ("N0", "T0", "S0")}
    for r in selected:
        for c in ("N0", "T0", "S0"):
            per_cond[c].append(r["features"][c])
    cells = {}
    for c, xs in per_cond.items():
        n = max(1, len(xs))
        cells[c] = {
            "n": len(xs),
            "mean_tok_len": sum(x["tok_len"] for x in xs) / n,
            "mean_char_len": sum(x["char_len"] for x in xs) / n,
            "n_W_present": sum(int(x["W_present"]) for x in xs),
            "n_gold_present": sum(int(x["gold_present"]) for x in xs),
            "n_react_leak": sum(int(x["react_leak"]) for x in xs),
            "n_support_complete": sum(int(x["support_complete"]) for x in xs),
        }

    summary = {
        "dataset": DATASET,
        "n_built": len(built),
        "n_selected": len(selected),
        "n_records": len(pair_rows),
        "n_pool_target": args.n_pool,
        "n_target": args.n,
        "seed": args.seed,
        "reject_reasons": dict(reject_counter),
        "cells": cells,
    }
    with open(out_dir / "build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] {len(selected)} samples; {len(pair_rows)} records -> {out_dir}/pairs.jsonl")
    print(f"[done] audit sheet ({len(audit_rows)} rows over {len(chosen_ids)} samples) -> {audit_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

