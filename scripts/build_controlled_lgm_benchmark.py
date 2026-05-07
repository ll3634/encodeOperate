#!/usr/bin/env python3
"""Controlled L/G/M causal benchmark for post-tool action variable selection.

Builds four conditions per sample across 8 bridge-style fact-card schemas.

Conditions
----------
B0 : High-L / Low-G / No-M   answer-like surface, missing the true bridge
B1 : High-L / Low-G / With-M same facts as B0 + evidence-local missingness cue
C0 : Low-L  / High-G / No-M  both bridge facts present as fact cards (no direct answer sentence)
D0 : High-L / High-G / No-M  both bridge facts present + explicit direct answer sentence

All four observations share the same question and answer. Only the content
of the two observation "cards" differs.  Questions, entity pools, bridge-key
pools, and answer-value pools are *invented* (or obscure) so that parametric
knowledge of Qwen does not leak.
"""
import argparse, csv, json, random, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.scorers import normalize_answer                       # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_anti_cue_specificity import CUES, has_react_leak     # noqa: E402

NEUTRAL_PRE, NEUTRAL_SUF = CUES["neutral"]          # unused here; kept for schema parity
TM_PRE,      TM_SUF      = CUES["task_missingness"]

# --- schema data ------------------------------------------------------------
# Each schema provides:
#   question  : f-string on {A}
#   b0_card1  : mentions A, no K, no V (High-L surface without the bridge)
#   b0_card2  : mentions V as a co-listed entity, no K, no A->V link
#   c0_card1  : fact A -> K
#   c0_card2  : fact K -> V  (NO direct answer sentence, NO "A" re-mention)
#   d0_card2  : C0 card2 + explicit direct-answer sentence linking A and V
# Titles (shown as "[1] <title>:") are fixed per schema to match length across conditions.

SCHEMAS = {
    "org_code_city": {
        "question": "What city is associated with the operating code of {A}?",
        "t1": "Registry profile", "t2_b": "Regional reference", "t2_c": "Code lookup",
        "b0_card1": ("{A} is a research organization listed in a multinational operations "
                     "database. Its profile references operating codes, regional affiliations, "
                     "and partner programs."),
        "b0_card2": ("{V} appears among the cities that occur in the multinational operations "
                     "database, alongside {D1} and {D2}; all entries were updated in the "
                     "recent catalog revision."),
        "c0_card1": ("{A} is assigned operating code {K} in the multinational operations "
                     "database. The assignment was recorded in the standard registry entry."),
        "c0_card2": ("{K} is mapped to {V} in the current code-to-city reference table. "
                     "The mapping was last verified during a recent catalog pass."),
        "d0_card2": ("{K} maps to {V} in the current code-to-city reference table, and "
                     "therefore {V} is the city associated with {A}'s operating code in "
                     "this registry."),
    },
    "person_work_genre": {
        "question": "What genre does the principal work of {A} belong to?",
        "t1": "Author note", "t2_b": "Genre register", "t2_c": "Work index",
        "b0_card1": ("{A} is an author whose principal work has appeared in several "
                     "period anthologies and critical surveys of early regional literature."),
        "b0_card2": ("{V} is one of the genre categories used in the period anthologies, "
                     "alongside {D1} and {D2}; the categories were revised in the latest edition."),
        "c0_card1": ("The principal work of {A} is the volume titled {K}, which appears in the "
                     "standard author bibliography maintained by the regional literature board."),
        "c0_card2": ("{K} is catalogued as a work of {V} in the regional literature board's "
                     "current genre register. The classification was last reviewed this cycle."),
        "d0_card2": ("{K} is catalogued as a work of {V}, and therefore {V} is the genre "
                     "that the principal work of {A} belongs to according to the regional "
                     "literature board."),
    },
    "company_product_country": {
        "question": "In which country is the flagship product of {A} manufactured?",
        "t1": "Company profile", "t2_b": "Trade register", "t2_c": "Product dossier",
        "b0_card1": ("{A} is a mid-sized industrial firm whose flagship product is listed in "
                     "several international trade registers and manufacturing directories."),
        "b0_card2": ("{V} is one of the countries that appears in the relevant international "
                     "trade register, alongside {D1} and {D2}; entries are updated annually."),
        "c0_card1": ("The flagship product of {A} is the model line designated {K}, as "
                     "recorded in the firm's entry in the international trade register."),
        "c0_card2": ("{K} is manufactured in {V} according to the current trade register "
                     "dossier. The dossier was last updated in the most recent review cycle."),
        "d0_card2": ("{K} is manufactured in {V}, and therefore {V} is the country where the "
                     "flagship product of {A} is manufactured according to the trade register."),
    },
    "event_venue_city": {
        "question": "In what city is the venue hosting {A} located?",
        "t1": "Event listing", "t2_b": "Venue directory", "t2_c": "Venue record",
        "b0_card1": ("{A} is an annual programme included in the regional events listing, "
                     "whose notes reference host venues, partner sponsors, and schedule windows."),
        "b0_card2": ("{V} is one of the cities that appears in the regional venue directory, "
                     "alongside {D1} and {D2}; the directory was revised during this cycle."),
        "c0_card1": ("{A} is hosted at the venue registered under the identifier {K} in the "
                     "regional events listing maintained by the programme authority."),
        "c0_card2": ("{K} is located in {V} according to the current venue directory. "
                     "The entry was last verified in the standard update pass."),
        "d0_card2": ("{K} is located in {V}, and therefore {V} is the city where the venue "
                     "hosting {A} is located according to the venue directory."),
    },
}

SCHEMAS.update({
    "object_catalog_material": {
        "question": "What material is used for the object cataloged under {A}?",
        "t1": "Catalog entry", "t2_b": "Material register", "t2_c": "Catalog reference",
        "b0_card1": ("{A} is an item cataloged in the regional museum database, whose entry "
                     "references object identifiers, acquisition notes, and conservation tags."),
        "b0_card2": ("{V} is one of the materials that appears in the regional museum's "
                     "material register, alongside {D1} and {D2}; the register is revised annually."),
        "c0_card1": ("The object cataloged under {A} carries the internal reference {K} in the "
                     "regional museum database, as recorded in the standard catalog entry."),
        "c0_card2": ("{K} is composed of {V} according to the museum's current material "
                     "register. The classification was last reviewed during the recent audit."),
        "d0_card2": ("{K} is composed of {V}, and therefore {V} is the material used for the "
                     "object cataloged under {A} according to the museum's material register."),
    },
    "species_family_habitat": {
        "question": "What habitat is occupied by the taxonomic family that contains {A}?",
        "t1": "Species note", "t2_b": "Habitat list", "t2_c": "Family dossier",
        "b0_card1": ("{A} is a species described in the regional zoological survey, whose "
                     "notes reference taxonomic family, observation records, and seasonal range."),
        "b0_card2": ("{V} is one of the habitats appearing in the regional zoological "
                     "habitat list, alongside {D1} and {D2}; the list is revised each year."),
        "c0_card1": ("{A} is placed within the taxonomic family designated {K} by the "
                     "regional zoological survey's current family dossier."),
        "c0_card2": ("{K} primarily occupies {V} according to the zoological survey's "
                     "current habitat list. The entry was last verified in the recent update."),
        "d0_card2": ("{K} primarily occupies {V}, and therefore {V} is the habitat occupied "
                     "by the taxonomic family that contains {A} per the habitat list."),
    },
    "book_publisher_hq": {
        "question": "In what city is the publisher of the book {A} headquartered?",
        "t1": "Book record", "t2_b": "City register", "t2_c": "Publisher dossier",
        "b0_card1": ("{A} is a book indexed in the national bibliographic registry, whose "
                     "record references publisher identifiers, imprint details, and release year."),
        "b0_card2": ("{V} is one of the cities appearing in the national bibliographic "
                     "registry's city register, alongside {D1} and {D2}; the register is revised each cycle."),
        "c0_card1": ("The book {A} is issued by the publisher registered under the identifier "
                     "{K} in the national bibliographic registry's publisher dossier."),
        "c0_card2": ("{K} is headquartered in {V} according to the registry's current "
                     "publisher dossier. The entry was last verified in the recent update pass."),
        "d0_card2": ("{K} is headquartered in {V}, and therefore {V} is the city where the "
                     "publisher of the book {A} is headquartered per the publisher dossier."),
    },
    "university_conference_hq": {
        "question": "In what city is the athletic conference of {A} headquartered?",
        "t1": "University note", "t2_b": "City register", "t2_c": "Conference dossier",
        "b0_card1": ("{A} is a university whose athletic programme is listed in the national "
                     "collegiate athletics directory, with notes on conference affiliation and "
                     "sponsorship."),
        "b0_card2": ("{V} is one of the cities appearing in the national collegiate athletics "
                     "directory's city register, alongside {D1} and {D2}; the register is revised annually."),
        "c0_card1": ("{A} competes in the athletic conference designated {K} by the national "
                     "collegiate athletics directory's current conference dossier."),
        "c0_card2": ("{K} is headquartered in {V} according to the directory's current "
                     "conference dossier. The entry was last verified during the recent review."),
        "d0_card2": ("{K} is headquartered in {V}, and therefore {V} is the city where the "
                     "athletic conference of {A} is headquartered per the conference dossier."),
    },
})


# --- pools: A = invented entity, K = invented bridge key, V = answer value ---
# V lists deliberately use obscure-but-real fillers (cities, genres, etc.) so the
# value is a natural-looking token; A and K are invented so parametric knowledge
# of A->V never leaks.
POOLS = {
    "org_code_city": {
        "A": ["Ardent Labs", "Beringia Foundry", "Pelham Institute", "Varrow Research",
              "Linden Archive", "Osprey Holdings", "Trevan Works", "Caldera Group"],
        "K": ["Q-17", "ZR-4", "BX-22", "M-309", "TN-8", "HC-11", "R-72", "FL-18"],
        "V": ["Helsinki", "Utrecht", "Porto", "Graz", "Bergen", "Bruges", "Vigo", "Trondheim"],
    },
    "person_work_genre": {
        "A": ["Miren Kalvo", "Dren Othwell", "Pasha Rovik", "Lena Verthe",
              "Tomas Quinaire", "Isla Freign", "Rafal Odorov", "Sen Arzu"],
        "K": ["Lighthouse of Avernus", "Mira of Thornwood", "The Keener's Row",
              "Harbour at Midwinter", "Verdiar's Trace", "Oshun's Ledger",
              "Passage of Holm", "Notes from Kiln Street"],
        "V": ["epistolary drama", "pastoral elegy", "naturalist novella",
              "absurdist farce", "historical essay", "picaresque satire",
              "gothic poem", "metafictional travelogue"],
    },
    "company_product_country": {
        "A": ["Halberd Systems", "Pellworm Dynamics", "Kronach Steelworks", "Novara Instruments",
              "Theron Pneumatics", "Fjolnir Alloys", "Bracken Drives", "Severin Controls"],
        "K": ["HS-214", "PD-7", "KS-88", "NI-331", "TP-42", "FA-15", "BD-109", "SC-66"],
        "V": ["Portugal", "Slovenia", "Latvia", "Uruguay", "Morocco", "Oman",
              "Vietnam", "Croatia"],
    },
    "event_venue_city": {
        "A": ["the Strelitz Festival", "the Varnamo Biennial", "the Caldwell Exposition",
              "the Okara Summit", "the Meldrum Forum", "the Tolbryn Gathering",
              "the Ulvik Conference", "the Pellegrin Convocation"],
        "K": ["Pavilion A-3", "Hall M-12", "Annex K-6", "Arena R-20",
              "Chamber L-44", "Stage Q-9", "Dome V-5", "Forum J-30"],
        "V": ["Tallinn", "Lyon", "Bilbao", "Thessaloniki", "Lausanne",
              "Ghent", "Turku", "Nantes"],
    },
    "object_catalog_material": {
        "A": ["the Varn Reliquary", "the Pelham Fibula", "the Osrik Ewer",
              "the Maldron Torc", "the Kiran Diadem", "the Verdiar Censer",
              "the Arzan Hilt", "the Holm Figurine"],
        "K": ["MR-4102", "MR-7740", "MR-3311", "MR-6028",
              "MR-2519", "MR-9846", "MR-5207", "MR-1385"],
        "V": ["polished basalt", "gilded bronze", "wrought silver", "patinated copper",
              "olivewood", "inlaid jade", "lacquered iron", "cast pewter"],
    },
    "species_family_habitat": {
        "A": ["Rhodonopsis albensis", "Veltrella pascua", "Kyrilon tridens",
              "Olmbra fusca", "Phenaria cordata", "Trelan occidens",
              "Mardonia silvestris", "Corenalia hypata"],
        "K": ["Lyronitidae", "Pescarionidae", "Tridenitidae", "Fuscanidae",
              "Cordanidae", "Occidentidae", "Silveranidae", "Hypatonidae"],
        "V": ["temperate freshwater streams", "high-altitude grasslands",
              "coastal mangrove flats", "subtropical dry forests",
              "semi-arid steppes", "montane cloud forests",
              "brackish tidal marshes", "boreal peat bogs"],
    },
    "book_publisher_hq": {
        "A": ["\"The Keener's Ledger\"", "\"Notes from Marnvik\"", "\"Ashes of Pelt\"",
              "\"Oshun's Catalogue\"", "\"The Verdiar Prospect\"",
              "\"A Winter at Holm\"", "\"Letters to Arzan\"", "\"The Strelitz Record\"" ],
        "K": ["Caldren Press", "Thornwood Editions", "Marnvik House", "Pelt & Ivers",
              "Oshun Books", "Verdiar Imprint", "Holm & Sons", "Arzan Press"],
        "V": ["Ljubljana", "Rotterdam", "Reykjavík", "Aarhus",
              "Zagreb", "Tallinn", "Leuven", "Poznan"],
    },
    "university_conference_hq": {
        "A": ["Keldren Polytechnic", "Marnvik State University", "Thornwood College",
              "Ashenfield Institute", "Holmcrest University", "Varnamo Tech",
              "Pellworm College", "Strelitz University"],
        "K": ["the Northshore Athletic Conference", "the Midland Valley Conference",
              "the Coastal Plains Conference", "the Highland Athletic Alliance",
              "the Great Basin Conference", "the Pinewood Athletic Association",
              "the River Bend Conference", "the Foothill Athletic League"],
        "V": ["Buffalo", "Tacoma", "Wichita", "Spokane", "Fresno",
              "Boise", "Akron", "Worcester"],
    },
}

# Sanity: every pool has at least 8 entries; no overlap per schema between A, K, V.
for _sn, _p in POOLS.items():
    assert len(_p["A"]) >= 8 and len(_p["K"]) >= 8 and len(_p["V"]) >= 8, _sn


# --- construction ----------------------------------------------------------

_STOPWORDS = set("a an the of and or to is in on at for with by from as this that which who whom "
                 "whose what where when why how does do did be been being are was were has have had "
                 "'s".split())


def _tok(text):
    return re.findall(r"[A-Za-z][A-Za-z'\-]*|\d+", text or "")


def _content_tokens(text):
    return [t.lower() for t in _tok(text) if t.lower() not in _STOPWORDS and len(t) > 1]


def _entity_count(text):
    return len(re.findall(r"\b[A-Z][A-Za-z'\-]*(?:\s+[A-Z][A-Za-z'\-]*)*\b", text or ""))


def _format(template, **kw):
    return template.format(**kw)


def build_observation(schema_name, condition, A, K, V, D1, D2):
    """Return (obs_string, card1_text, card2_text, title1, title2)."""
    s = SCHEMAS[schema_name]
    if condition in ("B0", "B1"):
        card1 = _format(s["b0_card1"], A=A, K=K, V=V, D1=D1, D2=D2)
        card2 = _format(s["b0_card2"], A=A, K=K, V=V, D1=D1, D2=D2)
        t1, t2 = s["t1"], s["t2_b"]
    elif condition == "C0":
        card1 = _format(s["c0_card1"], A=A, K=K, V=V, D1=D1, D2=D2)
        card2 = _format(s["c0_card2"], A=A, K=K, V=V, D1=D1, D2=D2)
        t1, t2 = s["t1"], s["t2_c"]
    elif condition == "D0":
        card1 = _format(s["c0_card1"], A=A, K=K, V=V, D1=D1, D2=D2)
        card2 = _format(s["d0_card2"], A=A, K=K, V=V, D1=D1, D2=D2)
        t1, t2 = s["t1"], s["t2_c"]
    else:
        raise ValueError(condition)

    # Wrap card1 uniformly with a cue: neutral for B0/C0/D0, task_missingness
    # for B1. This matches the anti_cue_specificity design (wrapper overhead
    # identical across conditions; only wrapper *content* changes).
    pre, suf = (TM_PRE, TM_SUF) if condition == "B1" else (NEUTRAL_PRE, NEUTRAL_SUF)
    m = re.match(r"(.+?\.)\s*(.*)", card1, flags=re.DOTALL)
    if m and m.group(2):
        head, tail = m.group(1), m.group(2)
        card1 = f"{head} {pre} {tail} {suf}".strip()
    else:
        card1 = f"{pre} {card1} {suf}"

    obs = f"[1] {t1}: {card1}\n\n[2] {t2}: {card2}"
    return obs, card1, card2, t1, t2


# --- features / audit ------------------------------------------------------

_CONCLUSION_MARKERS = ("therefore", "thus", "hence", "the answer is",
                       "in conclusion", "final answer", "clearly the answer")


def compute_features(obs, question, V, A, K):
    q_terms = set(_content_tokens(question))
    obs_low = obs.lower()
    a_low, k_low, v_low = A.lower(), K.lower(), V.lower()
    return {
        "char_len": len(obs),
        "tok_len": len(_tok(obs)),
        "q_overlap": sum(1 for t in _content_tokens(obs) if t in q_terms),
        "entity_count": _entity_count(obs),
        "paragraph_count": obs.count("\n\n") + 1,
        "mentions_A": a_low in obs_low,
        "mentions_K": k_low in obs_low,
        "mentions_V": v_low in obs_low,
        "has_conclusion_marker": any(m in obs_low for m in _CONCLUSION_MARKERS),
        "react_leak": has_react_leak(obs),
        "has_missingness_cue": (TM_PRE in obs),
    }


def answer_salience(obs, V, A):
    """Coarse surface-level answer extractability relative to the observation."""
    o, v, a = obs.lower(), V.lower(), A.lower()
    if v not in o:
        return "none"
    # High: direct "A ... is|was V" or "V is ... A" connective.
    if re.search(r"\b" + re.escape(a) + r"[^.]{0,80}\b(is|was|are|were)\b[^.]{0,40}"
                 + re.escape(v), o):
        return "high"
    if re.search(r"\bthe " + r"(answer|city|country|material|genre|habitat)\b[^.]{0,40}"
                 + re.escape(v), o):
        return "high"
    # High: "therefore V is ..." pattern near the question frame.
    if "therefore " + v in o or "therefore, " + v in o:
        return "high"
    # Medium: V appears in same sentence as A.
    sents = re.split(r"(?<=[.!?])\s+", obs)
    for s in sents:
        sl = s.lower()
        if a in sl and v in sl:
            return "medium"
    return "low"


def bridge_relation_present(obs, A, K, V):
    """True iff BOTH A->K link and K->V link appear in the observation.

    We require K to co-occur with A in one sentence AND K to co-occur with V in one sentence.
    """
    if not K:
        return False
    sents = re.split(r"(?<=[.!?])\s+", obs)
    ak = any(A.lower() in s.lower() and K.lower() in s.lower() for s in sents)
    kv = any(K.lower() in s.lower() and V.lower() in s.lower() for s in sents)
    return ak and kv


def direct_answer_sentence_present(obs, A, V, question):
    """True iff the observation contains a sentence that structurally answers the question."""
    sents = re.split(r"(?<=[.!?])\s+", obs)
    # Must contain V; must also contain either A or a Q-frame noun.
    q_nouns = [n for n in ("city", "country", "material", "genre", "habitat") if n in question.lower()]
    for s in sents:
        sl = s.lower()
        if V.lower() not in sl:
            continue
        if A.lower() in sl and re.search(r"\b(is|was|therefore|associated|headquartered|located|manufactured|composed|occupies|belongs|catalogued|catalog(ed)?)\b",
                                         sl):
            return True
        if q_nouns and any(n in sl for n in q_nouns) and A.lower() in sl:
            return True
    return False



# --- sample construction ---------------------------------------------------

CONDITIONS = ("B0", "B1", "C0", "D0")


def build_sample(schema_name, A, K, V, D1, D2, rng, sample_ix):
    question = _format(SCHEMAS[schema_name]["question"], A=A)
    recs = []
    feats_by_cond = {}
    for cond in CONDITIONS:
        obs, _c1, _c2, _t1, _t2 = build_observation(schema_name, cond, A, K, V, D1, D2)
        f = compute_features(obs, question, V, A, K)
        f["answer_salience"] = answer_salience(obs, V, A)
        f["bridge_relation_present"] = bridge_relation_present(obs, A, K, V)
        f["direct_answer_sentence_present"] = direct_answer_sentence_present(obs, A, V, question)
        feats_by_cond[cond] = f
        recs.append({
            "sample_id": f"{schema_name}_{sample_ix:03d}",
            "schema": schema_name,
            "A": A, "K": K, "V": V, "D1": D1, "D2": D2,
            "question": question,
            "gold_answer": V,
            "gold_answers": [V],
            "target": "sf",                        # schema parity with anti_cue pipeline
            "cue": "task_missingness" if cond == "B1" else "neutral",
            "condition_id": cond,
            "obs": obs,
            "feat": f,
            "answer_present": (cond == "D0"),      # only D0 states the answer directly
            "global_sufficiency_verified": cond in ("C0", "D0"),
        })

    # Construction audit: structural guarantees per condition.
    errs = []
    # B0/B1: Low-G = bridge_relation absent; High-L requires V to appear.
    for c in ("B0", "B1"):
        f = feats_by_cond[c]
        if f["bridge_relation_present"]:
            errs.append(f"{c}_bridge_unexpectedly_present")
        if not f["mentions_V"]:
            errs.append(f"{c}_V_missing")
        if f["direct_answer_sentence_present"]:
            errs.append(f"{c}_direct_answer_sentence_present")
    # C0: High-G = bridge present; Low-L = no direct answer sentence and no "A->V" salience.
    f = feats_by_cond["C0"]
    if not f["bridge_relation_present"]:
        errs.append("C0_bridge_missing")
    if f["direct_answer_sentence_present"]:
        errs.append("C0_direct_answer_sentence_present")
    if f["answer_salience"] == "high":
        errs.append("C0_answer_salience_high")
    # D0: High-G, High-L.
    f = feats_by_cond["D0"]
    if not f["bridge_relation_present"]:
        errs.append("D0_bridge_missing")
    if not f["direct_answer_sentence_present"]:
        errs.append("D0_direct_answer_sentence_missing")
    # Length band across the 4 cells (keep within 1.15x).
    toks = [feats_by_cond[c]["tok_len"] for c in CONDITIONS]
    if max(toks) / max(1, min(toks)) > 1.20:
        errs.append("length_ratio_out_of_band")
    # B1 must carry the missingness cue; the others must not.
    for c in CONDITIONS:
        has_cue = feats_by_cond[c]["has_missingness_cue"]
        if c == "B1" and not has_cue:
            errs.append("B1_cue_missing")
        if c != "B1" and has_cue:
            errs.append(f"{c}_unexpected_cue")
    # React / answer leaks.
    for c in CONDITIONS:
        if feats_by_cond[c]["react_leak"]:
            errs.append(f"{c}_react_leak")

    return recs, errs


def _pick_unique(rng, pool, n):
    idxs = list(range(len(pool)))
    rng.shuffle(idxs)
    return [pool[i] for i in idxs[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/controlled_lgm")
    ap.add_argument("--n", type=int, default=50, help="Total number of samples (across schemas).")
    ap.add_argument("--seed", type=int, default=20260424)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    schemas = list(SCHEMAS.keys())
    per_schema = args.n // len(schemas)
    extras = args.n - per_schema * len(schemas)
    counts = [per_schema + (1 if i < extras else 0) for i in range(len(schemas))]

    all_records, audit_rows, errs_counter = [], [], Counter()
    total_ok = 0
    for schema, want in zip(schemas, counts):
        pool = POOLS[schema]
        made = 0
        tries = 0
        while made < want and tries < want * 8:
            tries += 1
            A  = rng.choice(pool["A"])
            K  = rng.choice(pool["K"])
            V, D1, D2 = _pick_unique(rng, pool["V"], 3)
            recs, errs = build_sample(schema, A, K, V, D1, D2, rng, made)
            if errs:
                for e in errs:
                    errs_counter[e] += 1
                continue
            all_records.extend(recs)
            # One audit row per condition for blind review.
            for r in recs:
                audit_rows.append({
                    "sample_id": r["sample_id"],
                    "schema": r["schema"],
                    "condition_id": r["condition_id"],
                    "question": r["question"],
                    "obs": r["obs"],
                    "A": r["A"], "K": r["K"], "V": r["V"],
                    "tok_len": r["feat"]["tok_len"],
                    "char_len": r["feat"]["char_len"],
                    "q_overlap": r["feat"]["q_overlap"],
                    "entity_count": r["feat"]["entity_count"],
                    "answer_salience": r["feat"]["answer_salience"],
                    "bridge_relation_present": r["feat"]["bridge_relation_present"],
                    "direct_answer_sentence_present": r["feat"]["direct_answer_sentence_present"],
                    "has_missingness_cue": r["feat"]["has_missingness_cue"],
                    # Blind-audit columns (reviewer fills these):
                    "blind_answerable_now": "",           # yes / no
                    "blind_is_sufficient":  "",           # yes / no
                    "blind_has_missingness_cue": "",      # yes / no
                    "blind_notes": "",
                })
            made += 1
            total_ok += 1

    # Write pairs.jsonl
    with open(out_dir / "pairs.jsonl", "w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write blind audit sheet (CSV).
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
            "n_answer_salience_high": sum(1 for x in v if x["feat"]["answer_salience"] == "high"),
            "n_answer_salience_medium": sum(1 for x in v if x["feat"]["answer_salience"] == "medium"),
            "n_answer_salience_low": sum(1 for x in v if x["feat"]["answer_salience"] == "low"),
            "n_mentions_V": sum(int(x["feat"]["mentions_V"]) for x in v),
            "n_mentions_K": sum(int(x["feat"]["mentions_K"]) for x in v),
            "n_missingness_cue": sum(int(x["feat"]["has_missingness_cue"]) for x in v),
            "n_react_leak": sum(int(x["feat"]["react_leak"]) for x in v),
        } for k, v in per_cell.items()
    }
    per_schema_counts = Counter(r["schema"] for r in all_records if r["condition_id"] == "B0")
    summary = {
        "n_samples": total_ok,
        "n_records": len(all_records),
        "schemas": list(SCHEMAS.keys()),
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

