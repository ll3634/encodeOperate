#!/usr/bin/env python3
"""Extractability-Support-Missingness controlled toggle benchmark.

Four conditions per sample, all sharing the same A, same question, and the
same candidate W. Only the evidence toggle changes:

  N0 : Low-E, Low-S, No-M   W is absent; no A->K->W bridge. Model should search.
  T0 : High-E, Low-S, No-M  W is salient but unsupported (no bridge). Should search.
  T1 : High-E, Low-S, +M    T0 + explicit task_missingness cue on card1.
  S0 : High-E, High-S, No-M W is salient AND A->K, K->W present + direct answer.
                            Rational action: stop with W.

T0 and S0 use the SAME W by construction, controlling token length and
answer-candidate geometry. Only the support relation changes.

Pools / schemas reuse build_controlled_lgm_benchmark.py.
"""
import argparse, csv, json, random, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_controlled_lgm_benchmark import (                     # noqa: E402
    SCHEMAS as LGM_SCHEMAS, POOLS, NEUTRAL_PRE, NEUTRAL_SUF,
    TM_PRE, TM_SUF, _tok, _content_tokens, _entity_count,
    bridge_relation_present, direct_answer_sentence_present,
    answer_salience,
)
from build_anti_cue_specificity import has_react_leak             # noqa: E402
from build_unsupported_trap import TRAP, TITLES                   # noqa: E402


# --- N0 templates (Low-E, Low-S): A present, no K, no W, no answer candidate.
# Length and structural cadence matched to T0.  Card2 lists peer entities but
# deliberately contains no V-typed token.
N0 = {
    "org_code_city": {
        "c1": ("{A} is listed in a multinational operations database alongside "
               "peer research organizations."),
        "c2": ("The database tracks operating codes, regional affiliations, and "
               "schedule revisions. Other listed research organizations include "
               "{X1} and {X2}; all entries were updated in the recent catalog revision."),
    },
    "person_work_genre": {
        "c1": ("{A} is an author included in the regional literature board's "
               "current anthology of notable writers."),
        "c2": ("The anthology tracks author biographies, publication periods, and "
               "editorial notes. Other listed authors include {X1} and {X2}; "
               "classifications were revised in the latest edition."),
    },
    "company_product_country": {
        "c1": ("{A} is an industrial firm indexed in the international trade "
               "register of manufacturing enterprises."),
        "c2": ("The register tracks firm identifiers, production capacities, and "
               "compliance filings. Other listed firms include {X1} and {X2}; "
               "entries are updated annually."),
    },
    "event_venue_city": {
        "c1": ("{A} is listed in the regional events directory maintained by "
               "the programme authority."),
        "c2": ("The directory tracks programme identifiers, scheduling windows, "
               "and sponsor notes. Other listed programmes include {X1} and {X2}; "
               "the directory was revised during this cycle."),
    },
    "object_catalog_material": {
        "c1": ("{A} is an item cataloged in the regional museum database of "
               "conservation-tagged artefacts."),
        "c2": ("The database tracks object identifiers, acquisition records, and "
               "conservation notes. Other listed items include {X1} and {X2}; "
               "the register is revised annually."),
    },
    "species_family_habitat": {
        "c1": ("{A} is a species described in the regional zoological survey's "
               "observational records."),
        "c2": ("The survey tracks taxonomic identifiers, observation periods, and "
               "range notes. Other listed species include {X1} and {X2}; the list "
               "is revised each year."),
    },
    "book_publisher_hq": {
        "c1": ("{A} is a book indexed in the national bibliographic registry "
               "of published works."),
        "c2": ("The registry tracks imprint identifiers, release dates, and "
               "catalog revisions. Other listed titles include {X1} and {X2}; "
               "the registry is revised each cycle."),
    },
    "university_conference_hq": {
        "c1": ("{A} is a university listed in the national collegiate athletics "
               "directory of member institutions."),
        "c2": ("The directory tracks institution identifiers, sponsorship notes, "
               "and scheduling windows. Other listed universities include {X1} "
               "and {X2}; the directory is revised annually."),
    },
}


CONDITIONS = ("N0", "T0", "T1", "S0")

# Content indicators of the bridge-relation chain; used for included/missing fact labels.
BRIDGE_INDICATORS = ("assigned", "mapped", "designated", "catalogued", "registered",
                     "placed within", "competes in", "hosted at", "issued by",
                     "carries the internal reference", "is the model line")


def build_observation(schema, condition, A, K, W, X1, X2):
    """Return (obs, card1, card2, t1, t2)."""
    t_reg, t_trap, t_true = TITLES[schema]
    s = LGM_SCHEMAS[schema]

    if condition == "N0":
        c1 = N0[schema]["c1"].format(A=A)
        c2 = N0[schema]["c2"].format(X1=X1, X2=X2)
        t1, t2 = t_reg, t_trap
        cue = "neutral"
    elif condition in ("T0", "T1"):
        c1 = TRAP[schema]["c1"].format(A=A)
        c2 = TRAP[schema]["c2"].format(W=W, X1=X1, X2=X2)
        t1, t2 = t_reg, t_trap
        cue = "task_missingness" if condition == "T1" else "neutral"
    elif condition == "S0":
        # Bridge-based: A -> K and K -> W + direct answer sentence (V := W).
        c1 = s["c0_card1"].format(A=A, K=K, V=W, D1=X1, D2=X2)
        c2 = s["d0_card2"].format(A=A, K=K, V=W, D1=X1, D2=X2)
        t1, t2 = s["t1"], s["t2_c"]
        cue = "neutral"
    else:
        raise ValueError(condition)

    # Uniform cue wrapping of card1: T1 gets task_missingness; others neutral.
    pre, suf = (TM_PRE, TM_SUF) if cue == "task_missingness" else (NEUTRAL_PRE, NEUTRAL_SUF)
    m = re.match(r"(.+?\.)\s*(.*)", c1, flags=re.DOTALL)
    if m and m.group(2):
        c1 = f"{m.group(1)} {pre} {m.group(2)} {suf}".strip()
    else:
        c1 = f"{pre} {c1} {suf}".strip()

    obs = f"[1] {t1}: {c1}\n\n[2] {t2}: {c2}"
    return obs, c1, c2, t1, t2, cue



# --- features / audit ------------------------------------------------------
_CONCLUSION_MARKERS = ("therefore", "thus", "hence", "the answer is",
                       "in conclusion", "final answer", "clearly the answer")


def compute_features(obs, question, W, A, K):
    q_terms = set(_content_tokens(question))
    obs_low = obs.lower()
    return {
        "char_len": len(obs),
        "tok_len": len(_tok(obs)),
        "q_overlap": sum(1 for t in _content_tokens(obs) if t in q_terms),
        "entity_count": _entity_count(obs),
        "paragraph_count": obs.count("\n\n") + 1,
        "mentions_A": A.lower() in obs_low,
        "mentions_K": bool(K) and K.lower() in obs_low,
        "mentions_W": bool(W) and W.lower() in obs_low,
        "has_conclusion_marker": any(m in obs_low for m in _CONCLUSION_MARKERS),
        "react_leak": has_react_leak(obs),
        "has_missingness_cue": (TM_PRE in obs),
    }


def build_sample(schema, A, K, W, X1, X2, tok, sample_ix):
    question = LGM_SCHEMAS[schema]["question"].format(A=A)
    recs, feats_by_cond = [], {}
    w_tok_len = len(tok(" " + W)) if W else 0
    for cond in CONDITIONS:
        obs, _c1, _c2, _t1, _t2, cue = build_observation(schema, cond, A, K, W, X1, X2)
        f = compute_features(obs, question, W, A, K)
        f["answer_salience_W"] = answer_salience(obs, W, A) if W else "none"
        f["bridge_relation_present"] = bridge_relation_present(obs, A, K, W)
        f["direct_answer_sentence_present"] = direct_answer_sentence_present(obs, A, W, question)
        feats_by_cond[cond] = f
        # Per-condition expected flags.
        E_expected = cond in ("T0", "T1", "S0")
        S_expected = cond == "S0"
        M_expected = cond == "T1"
        included, missing = [], []
        if cond == "N0":
            included = ["A_membership"]
            missing  = ["candidate_W", "A_to_K", "K_to_W", "direct_answer_sentence"]
        elif cond == "T0":
            included = ["A_membership", "candidate_W"]
            missing  = ["A_to_K", "K_to_W", "direct_answer_sentence"]
        elif cond == "T1":
            included = ["A_membership", "candidate_W", "missingness_cue"]
            missing  = ["A_to_K", "K_to_W", "direct_answer_sentence"]
        elif cond == "S0":
            included = ["A_to_K", "K_to_W", "candidate_W", "direct_answer_sentence"]
            missing  = []

        recs.append({
            "sample_id": f"{schema}_{sample_ix:03d}",
            "schema_type": schema,
            "A": A, "K": K, "W": W, "X1": X1, "X2": X2,
            "question": question,
            "candidate_W": W,
            "gold_answer_if_supported": W,
            "gold_answer": W if cond == "S0" else None,
            "gold_answers": [W] if cond == "S0" else [],
            "condition": cond,
            "condition_id": cond,                     # alias for pipeline compatibility
            "E_intended": E_expected,
            "S_intended": S_expected,
            "M_intended": M_expected,
            "cue": "task_missingness" if cond == "T1" else "neutral",
            "obs": obs,
            "observation": obs,
            "included_facts": included,
            "missing_facts": missing,
            "candidate_present": f["mentions_W"],
            "candidate_salience": f["answer_salience_W"],
            "support_chain_present": f["bridge_relation_present"],
            "direct_answer_sentence_present": f["direct_answer_sentence_present"],
            "missingness_cue_present": f["has_missingness_cue"],
            "token_len": f["tok_len"],
            "entity_count": f["entity_count"],
            "q_overlap": f["q_overlap"],
            "answer_token_len": w_tok_len,
            "construction_notes": (
                f"cond={cond}; title=({_t1}|{_t2}); cue={cue}; "
                f"schema_family=bridge(A->K->W)"
            ),
            "target": "sf",
            "feat": f,
        })

    # Construction audit: per-cell expected flags.
    errs = []
    for cond in CONDITIONS:
        f = feats_by_cond[cond]
        # candidate_W presence
        if cond == "N0":
            if f["mentions_W"]:
                errs.append("N0_W_leaked")
            if f["answer_salience_W"] != "none":
                errs.append("N0_W_salience_non_none")
        else:
            if not f["mentions_W"]:
                errs.append(f"{cond}_W_missing")

        # bridge + direct answer sentence: only S0 has them
        if cond == "S0":
            if not f["bridge_relation_present"]:
                errs.append("S0_bridge_missing")
            if not f["direct_answer_sentence_present"]:
                errs.append("S0_direct_answer_sentence_missing")
        else:
            if f["bridge_relation_present"]:
                errs.append(f"{cond}_bridge_unexpected")
            if f["direct_answer_sentence_present"]:
                errs.append(f"{cond}_direct_answer_sentence_unexpected")

        # missingness cue: only T1 has it
        if cond == "T1" and not f["has_missingness_cue"]:
            errs.append("T1_cue_missing")
        if cond != "T1" and f["has_missingness_cue"]:
            errs.append(f"{cond}_unexpected_cue")

        if f["react_leak"]:
            errs.append(f"{cond}_react_leak")

    # Length parity: tighten T0-T1 (rescue pair), keep 4-way band at 1.35.
    t0, t1, n0, s0 = (feats_by_cond[c]["tok_len"] for c in ("T0", "T1", "N0", "S0"))
    if max(t0, t1) / max(1, min(t0, t1)) > 1.10:
        errs.append("T0_T1_length_mismatch")
    if max(t0, s0) / max(1, min(t0, s0)) > 1.25:
        errs.append("T0_S0_length_mismatch")
    toks = [t0, t1, n0, s0]
    if max(toks) / max(1, min(toks)) > 1.35:
        errs.append("length_ratio_out_of_band")

    return recs, errs



def _pick_unique(rng, pool, n):
    idxs = list(range(len(pool)))
    rng.shuffle(idxs)
    return [pool[i] for i in idxs[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/extractability_support_toggle")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260424)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Use the same word-level tokenizer approximation as answer_token_len.
    def _wtok(s):
        return _tok(s)

    schemas = list(N0.keys())
    per = args.n // len(schemas)
    extras = args.n - per * len(schemas)
    counts = [per + (1 if i < extras else 0) for i in range(len(schemas))]

    all_records, audit_rows, errs_counter = [], [], Counter()
    total_ok = 0
    for schema, want in zip(schemas, counts):
        pool = POOLS[schema]
        made, tries = 0, 0
        while made < want and tries < want * 16:
            tries += 1
            A  = rng.choice(pool["A"])
            K  = rng.choice(pool["K"])
            W  = rng.choice(pool["V"])         # single candidate shared by T0 and S0
            a_pool = [a for a in pool["A"] if a != A]
            rng.shuffle(a_pool)
            X1, X2 = a_pool[0], a_pool[1]
            recs, errs = build_sample(schema, A, K, W, X1, X2, _wtok, made)
            if errs:
                for e in errs:
                    errs_counter[e] += 1
                continue
            all_records.extend(recs)
            for r in recs:
                audit_rows.append({
                    "sample_id": r["sample_id"],
                    "schema_type": r["schema_type"],
                    "condition": r["condition"],
                    "question": r["question"],
                    "candidate_W": r["candidate_W"],
                    "E_intended": r["E_intended"],
                    "S_intended": r["S_intended"],
                    "M_intended": r["M_intended"],
                    "candidate_present": r["candidate_present"],
                    "candidate_salience": r["candidate_salience"],
                    "support_chain_present": r["support_chain_present"],
                    "direct_answer_sentence_present": r["direct_answer_sentence_present"],
                    "missingness_cue_present": r["missingness_cue_present"],
                    "token_len": r["token_len"],
                    "answer_token_len": r["answer_token_len"],
                    "obs": r["obs"],
                    "blind_extractable": "",
                    "blind_supported": "",
                    "blind_missingness": "",
                    "blind_notes": "",
                })
            made += 1
            total_ok += 1

    # Write pairs.jsonl
    with open(out_dir / "pairs.jsonl", "w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Blind audit sheet.
    audit_path = out_dir / "audit_sheet.csv"
    if audit_rows:
        with open(audit_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
            w.writeheader()
            for row in audit_rows:
                w.writerow(row)

    # Per-cell summary.
    per_cell = {}
    for r in all_records:
        per_cell.setdefault(r["condition"], []).append(r)
    cells = {
        k: {
            "n": len(v),
            "mean_tok_len": sum(x["feat"]["tok_len"] for x in v) / max(1, len(v)),
            "mean_char_len": sum(x["feat"]["char_len"] for x in v) / max(1, len(v)),
            "mean_q_overlap": sum(x["feat"]["q_overlap"] for x in v) / max(1, len(v)),
            "mean_entity_count": sum(x["feat"]["entity_count"] for x in v) / max(1, len(v)),
            "n_bridge_present": sum(int(x["feat"]["bridge_relation_present"]) for x in v),
            "n_direct_answer_sentence": sum(int(x["feat"]["direct_answer_sentence_present"]) for x in v),
            "n_mentions_W": sum(int(x["feat"]["mentions_W"]) for x in v),
            "n_mentions_K": sum(int(x["feat"]["mentions_K"]) for x in v),
            "n_answer_salience_W_high": sum(1 for x in v if x["feat"]["answer_salience_W"] == "high"),
            "n_missingness_cue": sum(int(x["feat"]["has_missingness_cue"]) for x in v),
            "n_react_leak": sum(int(x["feat"]["react_leak"]) for x in v),
        } for k, v in per_cell.items()
    }
    per_schema_counts = Counter(r["schema_type"] for r in all_records if r["condition"] == "T0")
    summary = {
        "n_samples": total_ok,
        "n_records": len(all_records),
        "schemas": list(N0.keys()),
        "per_schema_samples": dict(per_schema_counts),
        "reject_reasons": dict(errs_counter),
        "cells": cells,
    }
    with open(out_dir / "build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] {total_ok} samples, {len(all_records)} records -> {out_dir}/pairs.jsonl")
    print(f"[done] blind audit sheet -> {audit_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
