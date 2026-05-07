#!/usr/bin/env python3
"""Unsupported answer trap + missingness rescue benchmark.

Three conditions per sample:
  Trap-B0 : Low-G, High-L, no-M   extractable wrong candidate W is surfaced
                                   via an over-generalised collective
                                   statement; gold V is absent; no cue.
  Trap-B1 : Low-G, High-L, +M      same as Trap-B0 + task_missingness cue.
  True-D0 : High-G, High-L, no-M   gold V appears with a direct answer
                                   sentence; no W. Matches controlled_lgm D0.

All three use the same A (entity) and same question. Only observations
differ. Pools and schemas come from build_controlled_lgm_benchmark.
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

# --- trap templates (per schema) -------------------------------------------
# Each trap uses an over-generalised collective statement that presents a
# V-typed token W as the answer, while A is merely listed as a member.

TRAP = {
    "org_code_city": {
        "c1": ("{A} is listed in a multinational operations database alongside "
               "peer research organizations."),
        "c2": ("Operating codes in this database are mapped to {W} across "
               "entries. Other listed research organizations include {X1} "
               "and {X2}; all entries were updated in the recent catalog revision."),
    },
    "person_work_genre": {
        "c1": ("{A} is an author included in the regional literature board's "
               "current anthology of notable writers."),
        "c2": ("Principal works indexed in this anthology are classified as {W} "
               "across entries. Other listed authors include {X1} and {X2}; "
               "classifications were revised in the latest edition."),
    },
    "company_product_country": {
        "c1": ("{A} is an industrial firm indexed in the international trade "
               "register of manufacturing enterprises."),
        "c2": ("Flagship products indexed in this register are manufactured in "
               "{W} across entries. Other listed firms include {X1} and {X2}; "
               "entries are updated annually."),
    },
    "event_venue_city": {
        "c1": ("{A} is listed in the regional events directory maintained by "
               "the programme authority."),
        "c2": ("Venues hosting programmes in this directory are located in {W} "
               "across entries. Other listed programmes include {X1} and {X2}; "
               "the directory was revised during this cycle."),
    },
    "object_catalog_material": {
        "c1": ("{A} is an item cataloged in the regional museum database of "
               "conservation-tagged artefacts."),
        "c2": ("Objects cataloged in this register are composed of {W} across "
               "entries. Other listed items include {X1} and {X2}; the register "
               "is revised annually."),
    },
    "species_family_habitat": {
        "c1": ("{A} is a species described in the regional zoological survey's "
               "observational records."),
        "c2": ("Taxonomic families documented in this survey occupy {W} across "
               "entries. Other listed species include {X1} and {X2}; the list "
               "is revised each year."),
    },
    "book_publisher_hq": {
        "c1": ("{A} is a book indexed in the national bibliographic registry "
               "of published works."),
        "c2": ("Publishers indexed in this registry are headquartered in {W} "
               "across entries. Other listed titles include {X1} and {X2}; "
               "the registry is revised each cycle."),
    },
    "university_conference_hq": {
        "c1": ("{A} is a university listed in the national collegiate athletics "
               "directory of member institutions."),
        "c2": ("Athletic conferences affiliated with entries in this directory "
               "are headquartered in {W} across listings. Other listed "
               "universities include {X1} and {X2}; the directory is revised annually."),
    },
}

# Title strings (fixed per schema, parity with controlled_lgm layout).
TITLES = {
    "org_code_city":           ("Registry profile", "Database summary",    "Code lookup"),
    "person_work_genre":       ("Author note",      "Anthology summary",   "Work index"),
    "company_product_country": ("Company profile",  "Register summary",    "Product dossier"),
    "event_venue_city":        ("Event listing",    "Directory summary",   "Venue record"),
    "object_catalog_material": ("Catalog entry",    "Register summary",    "Catalog reference"),
    "species_family_habitat":  ("Species note",     "Survey summary",      "Family dossier"),
    "book_publisher_hq":       ("Book record",      "Registry summary",    "Publisher dossier"),
    "university_conference_hq":("University note",  "Directory summary",   "Conference dossier"),
}

CONDITIONS = ("Trap-B0", "Trap-B1", "True-D0")


def build_observation(schema, condition, A, K, V, W, X1, X2):
    """Return (obs, card1, card2, t1, t2, extras)."""
    t_reg, t_trap, t_true = TITLES[schema]
    if condition in ("Trap-B0", "Trap-B1"):
        c1 = TRAP[schema]["c1"].format(A=A)
        c2 = TRAP[schema]["c2"].format(W=W, X1=X1, X2=X2)
        t1, t2 = t_reg, t_trap
    elif condition == "True-D0":
        s = LGM_SCHEMAS[schema]
        c1 = s["c0_card1"].format(A=A, K=K, V=V, D1=X1, D2=X2)
        c2 = s["d0_card2"].format(A=A, K=K, V=V, D1=X1, D2=X2)
        t1, t2 = s["t1"], s["t2_c"]
    else:
        raise ValueError(condition)

    # Uniform cue wrapping of card1 (match controlled_lgm): neutral for
    # Trap-B0 and True-D0; task_missingness for Trap-B1. Keeps length matched.
    pre, suf = (TM_PRE, TM_SUF) if condition == "Trap-B1" else (NEUTRAL_PRE, NEUTRAL_SUF)
    m = re.match(r"(.+?\.)\s*(.*)", c1, flags=re.DOTALL)
    if m and m.group(2):
        c1 = f"{m.group(1)} {pre} {m.group(2)} {suf}".strip()
    else:
        c1 = f"{pre} {c1} {suf}"

    obs = f"[1] {t1}: {c1}\n\n[2] {t2}: {c2}"
    return obs, c1, c2, t1, t2



# --- features / audit ------------------------------------------------------

_CONCLUSION_MARKERS = ("therefore", "thus", "hence", "the answer is",
                       "in conclusion", "final answer", "clearly the answer")


def compute_features(obs, question, V, W, A, K):
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
        "mentions_V": V.lower() in obs_low,
        "mentions_W": bool(W) and W.lower() in obs_low,
        "has_conclusion_marker": any(m in obs_low for m in _CONCLUSION_MARKERS),
        "react_leak": has_react_leak(obs),
        "has_missingness_cue": (TM_PRE in obs),
    }


# --- sample construction ---------------------------------------------------

def build_sample(schema, A, K, V, W, X1, X2, sample_ix):
    question = LGM_SCHEMAS[schema]["question"].format(A=A)
    recs, feats_by_cond = [], {}
    for cond in CONDITIONS:
        obs, _c1, _c2, _t1, _t2 = build_observation(schema, cond, A, K, V, W, X1, X2)
        f = compute_features(obs, question, V, W, A, K)
        f["answer_salience_V"] = answer_salience(obs, V, A)
        f["answer_salience_W"] = answer_salience(obs, W, A) if W else "none"
        f["bridge_relation_present"] = bridge_relation_present(obs, A, K, V)
        f["direct_answer_sentence_present"] = direct_answer_sentence_present(obs, A, V, question)
        feats_by_cond[cond] = f
        recs.append({
            "sample_id": f"{schema}_{sample_ix:03d}",
            "schema": schema,
            "A": A, "K": K, "V": V, "W": W, "X1": X1, "X2": X2,
            "question": question,
            "gold_answer": V,
            "gold_answers": [V],
            "trap_answer": W,
            "target": "sf",
            "cue": "task_missingness" if cond == "Trap-B1" else "neutral",
            "condition_id": cond,
            "obs": obs,
            "feat": f,
            "answer_present": (cond == "True-D0"),
            "trap_present": cond in ("Trap-B0", "Trap-B1"),
            "global_sufficiency_verified": (cond == "True-D0"),
        })

    # Construction audit.
    errs = []
    for c in ("Trap-B0", "Trap-B1"):
        f = feats_by_cond[c]
        if f["mentions_V"]:
            errs.append(f"{c}_V_leaked")
        if not f["mentions_W"]:
            errs.append(f"{c}_W_missing")
        if f["bridge_relation_present"]:
            errs.append(f"{c}_bridge_unexpectedly_present")
        if f["direct_answer_sentence_present"]:
            errs.append(f"{c}_direct_answer_sentence_present")
    f = feats_by_cond["True-D0"]
    if not f["mentions_V"]:
        errs.append("True-D0_V_missing")
    if f["mentions_W"]:
        errs.append("True-D0_W_leaked")
    if not f["bridge_relation_present"]:
        errs.append("True-D0_bridge_missing")
    if not f["direct_answer_sentence_present"]:
        errs.append("True-D0_direct_answer_sentence_missing")
    # Primary length parity: Trap-B0 vs Trap-B1 (the rescue contrast) must be
    # tightly matched. True-D0 is inherently longer (explicit answer sentence)
    # so the three-way band is loosened to 1.25.
    b0_tok = feats_by_cond["Trap-B0"]["tok_len"]
    b1_tok = feats_by_cond["Trap-B1"]["tok_len"]
    if max(b0_tok, b1_tok) / max(1, min(b0_tok, b1_tok)) > 1.10:
        errs.append("trap_pair_length_mismatch")
    toks = [feats_by_cond[c]["tok_len"] for c in CONDITIONS]
    if max(toks) / max(1, min(toks)) > 1.25:
        errs.append("length_ratio_out_of_band")
    for c in CONDITIONS:
        has_cue = feats_by_cond[c]["has_missingness_cue"]
        if c == "Trap-B1" and not has_cue:
            errs.append("Trap-B1_cue_missing")
        if c != "Trap-B1" and has_cue:
            errs.append(f"{c}_unexpected_cue")
        if feats_by_cond[c]["react_leak"]:
            errs.append(f"{c}_react_leak")
    return recs, errs


def _pick_unique(rng, pool, n):
    idxs = list(range(len(pool)))
    rng.shuffle(idxs)
    return [pool[i] for i in idxs[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/unsupported_trap")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260424)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    schemas = list(TRAP.keys())
    per = args.n // len(schemas)
    extras = args.n - per * len(schemas)
    counts = [per + (1 if i < extras else 0) for i in range(len(schemas))]

    all_records, audit_rows, errs_counter = [], [], Counter()
    total_ok = 0
    for schema, want in zip(schemas, counts):
        pool = POOLS[schema]
        made, tries = 0, 0
        while made < want and tries < want * 12:
            tries += 1
            A  = rng.choice(pool["A"])
            K  = rng.choice(pool["K"])
            # V is the gold; W is a different V-typed token (the trap).
            vs = _pick_unique(rng, pool["V"], 2)
            V, W = vs[0], vs[1]
            # X1, X2 are peer entity fillers (same type as A), drawn from A pool ≠ A.
            a_pool = [a for a in pool["A"] if a != A]
            rng.shuffle(a_pool)
            X1, X2 = a_pool[0], a_pool[1]
            recs, errs = build_sample(schema, A, K, V, W, X1, X2, made)
            if errs:
                for e in errs:
                    errs_counter[e] += 1
                continue
            all_records.extend(recs)
            for r in recs:
                audit_rows.append({
                    "sample_id": r["sample_id"],
                    "schema": r["schema"],
                    "condition_id": r["condition_id"],
                    "question": r["question"],
                    "obs": r["obs"],
                    "A": r["A"], "K": r["K"], "V": r["V"], "W": r["W"],
                    "tok_len": r["feat"]["tok_len"],
                    "char_len": r["feat"]["char_len"],
                    "q_overlap": r["feat"]["q_overlap"],
                    "entity_count": r["feat"]["entity_count"],
                    "answer_salience_V": r["feat"]["answer_salience_V"],
                    "answer_salience_W": r["feat"]["answer_salience_W"],
                    "bridge_relation_present": r["feat"]["bridge_relation_present"],
                    "direct_answer_sentence_present": r["feat"]["direct_answer_sentence_present"],
                    "mentions_V": r["feat"]["mentions_V"],
                    "mentions_W": r["feat"]["mentions_W"],
                    "has_missingness_cue": r["feat"]["has_missingness_cue"],
                    "blind_answerable_now": "",
                    "blind_trap_extractable": "",
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
        per_cell.setdefault(r["condition_id"], []).append(r)
    cells = {
        k: {
            "n": len(v),
            "mean_tok_len": sum(x["feat"]["tok_len"] for x in v) / max(1, len(v)),
            "mean_char_len": sum(x["feat"]["char_len"] for x in v) / max(1, len(v)),
            "mean_q_overlap": sum(x["feat"]["q_overlap"] for x in v) / max(1, len(v)),
            "mean_entity_count": sum(x["feat"]["entity_count"] for x in v) / max(1, len(v)),
            "n_bridge_present": sum(int(x["feat"]["bridge_relation_present"]) for x in v),
            "n_direct_answer_sentence": sum(int(x["feat"]["direct_answer_sentence_present"]) for x in v),
            "n_mentions_V": sum(int(x["feat"]["mentions_V"]) for x in v),
            "n_mentions_W": sum(int(x["feat"]["mentions_W"]) for x in v),
            "n_answer_salience_V_high": sum(1 for x in v if x["feat"]["answer_salience_V"] == "high"),
            "n_answer_salience_W_high": sum(1 for x in v if x["feat"]["answer_salience_W"] == "high"),
            "n_missingness_cue": sum(int(x["feat"]["has_missingness_cue"]) for x in v),
            "n_react_leak": sum(int(x["feat"]["react_leak"]) for x in v),
        } for k, v in per_cell.items()
    }
    per_schema_counts = Counter(r["schema"] for r in all_records if r["condition_id"] == "Trap-B0")
    summary = {
        "n_samples": total_ok,
        "n_records": len(all_records),
        "schemas": list(TRAP.keys()),
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

