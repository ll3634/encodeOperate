#!/usr/bin/env python3
"""Build N0/T0/S0 extractability replication on 2WikiMultiHopQA (compositional, 2-hop bridge).

Per 2-hop compositional example with evidences=[(A,rel1,K),(K,rel2,W)]:
  W = final answer (extractable in sup2, the paragraph titled K).
  S0 = sup1 + sup2 + 2 distractors (full chain).
  T0 = sup2 + 3 distractors (W extractable, bridge K withheld).
  N0 = 4 distractors only (no W, no K, no aliases).

Outputs (in --out-dir): mirrors build_musique_extractability.py.
"""
import argparse, csv, json, random, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_anti_cue_specificity import has_react_leak  # noqa: E402
from build_musique_extractability import (              # noqa: E402
    _tok, _norm, _contains, _has_protocol_leak, _qc,
    _expected_type_label, _audit_rows,
)

DATASET = "2WikiMultiHopQA (dev, compositional 2-hop bridge)"


def _build_obs(paragraphs):
    parts = []
    for i, (title, sents) in enumerate(paragraphs, 1):
        body = " ".join(sents).strip()
        parts.append(f"[{i}] {title.strip()}: {body}")
    return "\n\n".join(parts)


def _aliases(rec):
    out = {(rec.get("answer") or "").strip()}
    for ev in rec.get("evidences", []) or []:
        if isinstance(ev, list) and len(ev) == 3 and isinstance(ev[2], str):
            out.add(ev[2].strip())
    return [a for a in out if a]


def _select_supports(rec):
    """Return (sup1_para, sup2_para, K, W) or (None, None, None, None) on failure."""
    evs = rec.get("evidences") or []
    if len(evs) != 2:
        return None, None, None, None
    A_subj, _, K = evs[0]
    K2, _, W = evs[1]
    if not (isinstance(A_subj, str) and isinstance(K, str)
            and isinstance(K2, str) and isinstance(W, str)):
        return None, None, None, None
    if _norm(K) != _norm(K2):
        return None, None, None, None
    A_subj, K, W = A_subj.strip(), K.strip(), W.strip()
    by_title = {(t or "").strip(): (t, s) for t, s in (rec.get("context") or [])}
    sup1 = by_title.get(A_subj)
    sup2 = by_title.get(K)
    if sup1 is None or sup2 is None:
        return None, None, None, None
    sup1_text = " ".join(sup1[1])
    sup2_text = " ".join(sup2[1])
    if not _contains(sup1_text, K):
        return None, None, None, None
    if not _contains(sup2_text, W):
        return None, None, None, None
    return sup1, sup2, K, W


def _build_conditions(rec, rng):
    sup1, sup2, K, W = _select_supports(rec)
    if sup1 is None:
        return None, "support_resolution_failed"
    forbid = list({W, K, *_aliases(rec)})
    sup_titles = {sup1[0], sup2[0]}
    distractors = []
    for t, s in rec.get("context", []):
        if t in sup_titles:
            continue
        body = " ".join(s) + " " + t
        if any(_contains(body, x) for x in forbid):
            continue
        distractors.append((t, s))
    if len(distractors) < 4:
        return None, "too_few_clean_distractors"
    rng.shuffle(distractors)
    s0 = [sup1, sup2] + distractors[:2]; rng.shuffle(s0)
    if len(distractors) >= 5:
        t0 = [sup2] + distractors[2:5]
    else:
        t0 = [sup2] + distractors[:3]
    rng.shuffle(t0)
    n0 = list(distractors[:4]); rng.shuffle(n0)
    return {"N0": n0, "T0": t0, "S0": s0, "K": K, "W": W,
            "sup1_title": sup1[0], "sup2_title": sup2[0]}, None


def build_record(rec, rng):
    if rec.get("type") != "compositional":
        return None, None, "not_compositional"
    cond_paras, why = _build_conditions(rec, rng)
    if cond_paras is None:
        return None, None, why
    W = cond_paras["W"]
    if not W or len(W) > 60:
        return None, None, "answer_length_oob"
    aliases = _aliases(rec)
    sample_id = "2wiki_" + str(rec.get("_id"))
    out = {
        "sample_id": sample_id,
        "dataset": DATASET,
        "twowiki_id": rec.get("_id"),
        "question": rec["question"],
        "gold_answer": W,
        "gold_answers": list({W, *aliases}),
        "candidate_W": W,
        "expected_answer_type": _expected_type_label(rec["question"]),
        "support_titles": [cond_paras["sup1_title"], cond_paras["sup2_title"]],
        "bridge_entity": cond_paras["K"],
    }
    qc_issues, feat = [], {}
    for cond in ("N0", "T0", "S0"):
        obs = _build_obs(cond_paras[cond])
        issues, w_present, g_present = _qc(obs, W, aliases, cond)
        qc_issues.extend(issues)
        out[f"{cond}_observation"] = obs
        out[f"{cond}_paragraph_titles"] = [p[0] for p in cond_paras[cond]]
        feat[cond] = {
            "tok_len": len(_tok(obs)), "char_len": len(obs),
            "n_paragraphs": len(cond_paras[cond]),
            "W_present": w_present, "gold_present": g_present,
            "react_leak": _has_protocol_leak(obs),
        }
    feat["T0"]["support_complete"] = False
    feat["S0"]["support_complete"] = True
    feat["N0"]["support_complete"] = False
    out["features"] = feat
    out["construction_notes"] = (
        f"compositional 2-hop: A->{cond_paras['K']!r}->{W!r}. "
        "S0 keeps sup1+sup2+2 distractors; T0 drops sup1 (bridge withheld); "
        "N0 uses 4 distractors filtered against W/K/aliases."
    )
    toks = [feat[c]["tok_len"] for c in ("N0", "T0", "S0")]
    if min(toks) > 0 and max(toks) / min(toks) > 1.35:
        qc_issues.append("length_ratio_out_of_band")
    out["qc_issues"] = qc_issues
    return out, feat, None



def _to_pair_records(rec):
    out = []
    for cond in ("N0", "T0", "S0"):
        out.append({
            "sample_id": rec["sample_id"],
            "schema_type": "twowiki_compositional_2hop",
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
            "paragraph_titles": rec[f"{cond}_paragraph_titles"],
            "support_titles": rec["support_titles"],
            "bridge_entity": rec.get("bridge_entity"),
            "twowiki_id": rec.get("twowiki_id"),
            "feat": rec["features"][cond],
            "E_intended": cond in ("T0", "S0"),
            "S_intended": cond == "S0",
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/2wikimultihopqa/dev.json")
    ap.add_argument("--out-dir", default="results/third_benchmark_extractability")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--n-pool", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260427)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    src_path = Path(args.data)
    if not src_path.is_absolute():
        src_path = Path(__file__).resolve().parent.parent / src_path
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    src = [r for r in json.load(open(src_path)) if r.get("type") == "compositional"]
    rng.shuffle(src)
    print(f"[info] {len(src)} compositional candidates in source")

    built, rejected = [], []
    reject_counter = Counter()
    for r in src:
        if len(built) >= args.n_pool:
            break
        rec, feat, why = build_record(r, rng)
        if why:
            rejected.append({"twowiki_id": r.get("_id"), "reason": why,
                             "question": r.get("question"), "answer": r.get("answer")})
            reject_counter[why] += 1
            continue
        if rec.get("qc_issues"):
            for q in rec["qc_issues"]:
                reject_counter["qc:" + q] += 1
            rejected.append({"twowiki_id": r.get("_id"),
                             "reason": "qc:" + ";".join(rec["qc_issues"]),
                             "question": r.get("question"), "answer": r.get("answer")})
            continue
        built.append(rec)

    print(f"[info] built {len(built)} clean candidates (rejected {len(rejected)})")
    selected = built[:args.n]
    print(f"[info] selected {len(selected)} for the official run")

    (out_dir / "dataset_name.txt").write_text(DATASET + "\n")
    with open(out_dir / "build_candidates.jsonl", "w") as f:
        for r in built: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "rejected_candidates.jsonl", "w") as f:
        for r in rejected: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "selected_examples.jsonl", "w") as f:
        for r in selected: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pair_rows = [pr for r in selected for pr in _to_pair_records(r)]
    with open(out_dir / "pairs.jsonl", "w") as f:
        for pr in pair_rows: f.write(json.dumps(pr, ensure_ascii=False) + "\n")

    rng_a = random.Random(args.seed + 1)
    audit_pool = _audit_rows(selected)
    by_sample = {}
    for row in audit_pool:
        by_sample.setdefault(row["sample_id"], []).append(row)
    sample_ids = list(by_sample.keys())
    rng_a.shuffle(sample_ids)
    chosen_ids = set(sample_ids[:20])
    def spread(sid):
        toks = [by_sample[sid][i]["tok_len"] for i in range(len(by_sample[sid]))]
        return max(toks) / max(1, min(toks))
    edge_ids = sorted(sample_ids, key=spread, reverse=True)[:10]
    chosen_ids.update(edge_ids)
    for sid in sample_ids:
        toks = [r["tok_len"] for r in by_sample[sid]]
        if max(toks) / max(1, min(toks)) > 1.20:
            chosen_ids.add(sid)
    audit_rows = [r for sid in chosen_ids for r in by_sample[sid]]
    if audit_rows:
        with open(out_dir / "audit_sheet.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
            w.writeheader()
            for row in audit_rows: w.writerow(row)

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
        "dataset": DATASET, "n_built": len(built), "n_selected": len(selected),
        "n_records": len(pair_rows), "n_pool_target": args.n_pool,
        "n_target": args.n, "seed": args.seed,
        "reject_reasons": dict(reject_counter), "cells": cells,
    }
    with open(out_dir / "build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] {len(selected)} samples; {len(pair_rows)} records -> {out_dir}/pairs.jsonl")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
