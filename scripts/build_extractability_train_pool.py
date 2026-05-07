#!/usr/bin/env python3
"""Build a non-overlapping training pool of T0/N0 records for the LoRA SFT pilot.

Re-uses build_musique_extractability.build_record and
build_2wiki_extractability.build_record, but (a) iterates with a fresh seed,
(b) excludes any source-ids already used in the held-out pairs.jsonl files for
MuSiQue (results/second_benchmark_extractability/pairs.jsonl) and 2Wiki
(results/third_benchmark_extractability/pairs.jsonl).

Outputs (default --out-dir data/extractability_train/):
  train_T0.jsonl       SFT records: T0 prompt -> "Action: search ..."
  train_N0.jsonl       SFT records: N0 prompt -> "Action: search ..." (control)
  train_pool_audit.json  per-source counts + overlap=0 verification
"""
import argparse, json, random, sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import build_musique_extractability as bme   # noqa: E402
import build_2wiki_extractability as bwe     # noqa: E402

# Match exactly the prompt format produced by eval_second_benchmark.build_messages_clean.
sys.path.insert(0, str(_HERE.parent))
from scripts.eval_second_benchmark import build_messages_clean   # noqa: E402


def load_held_out(path: Path, id_field: str) -> set:
    if not path.exists():
        return set()
    return {json.loads(l).get(id_field) for l in open(path)}


def make_target(rec: dict, condition: str) -> str:
    """Per-condition gold target.

    T0/N0 -> Action: search (with bridge as query).
    S0    -> Final Answer: <gold_answer>.
    """
    if condition == "S0":
        gold = (rec.get("gold_answer") or rec.get("candidate_W") or "").strip()
        return f"Final Answer: {gold}"
    bridge = rec.get("bridge_entity") or ""
    return f"Action: search\nAction Input: {bridge}".strip()


def emit_train_record(rec: dict, condition: str, source: str) -> dict:
    """Convert a per-condition pair record into an SFT training record."""
    sub = dict(rec); sub["condition"] = condition
    msgs = build_messages_clean(rec["question"], rec[f"{condition}_observation"],
                                prompt_variant="v1", obs_style="factcard")
    return {
        "sample_id": rec["sample_id"],
        "source": source,
        "condition": condition,
        "prompt_messages": msgs,
        "target_text": make_target(rec, condition),
        "bridge_entity": rec.get("bridge_entity"),
        "candidate_W": rec.get("candidate_W"),
        "gold_answer": rec.get("gold_answer"),
        "question": rec.get("question"),
    }


def collect_pool(source_name: str, src_path: Path, build_record, id_key: str,
                 held_out_ids: set, target_n: int, seed: int) -> tuple:
    """Stream source rows, build, QC-filter, drop held-out; return list of pair-dicts."""
    rng = random.Random(seed)
    src_rows = []
    if source_name == "musique":
        with open(src_path) as f:
            for line in f:
                r = json.loads(line)
                if (r.get("id") or "").startswith("2hop__"):
                    src_rows.append(r)
    elif source_name == "2wiki":
        data = json.load(open(src_path))
        for r in data:
            if (r.get("type") or "").lower() == "compositional":
                src_rows.append(r)
    rng.shuffle(src_rows)
    print(f"[{source_name}] source pool: {len(src_rows)} compositional/2hop rows")

    accepted, dropped_overlap, reject_reasons = [], 0, Counter()
    for r in src_rows:
        if len(accepted) >= target_n:
            break
        sid_field = r.get("id") if source_name == "musique" else r.get("_id")
        if sid_field in held_out_ids:
            dropped_overlap += 1
            continue
        rec, feat, why = build_record(r, rng)
        if why:
            reject_reasons[why] += 1
            continue
        if rec.get("qc_issues"):
            for q in rec["qc_issues"]:
                reject_reasons["qc:" + q] += 1
            continue
        # rec contains obs_N0/T0/S0; tag source ID for audit.
        rec["_source_id"] = sid_field
        accepted.append(rec)
    print(f"[{source_name}] accepted={len(accepted)}  overlap_dropped={dropped_overlap}")
    return accepted, dropped_overlap, reject_reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/extractability_train")
    ap.add_argument("--n-musique", type=int, default=200)
    ap.add_argument("--n-2wiki", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260501)
    ap.add_argument("--musique-data",
        default="data/musique/musique_ans_v1.0_dev.jsonl")
    ap.add_argument("--2wiki-data", dest="twowiki_data",
        default="data/2wikimultihopqa/dev.json")
    ap.add_argument("--held-musique",
        default="results/second_benchmark_extractability/pairs.jsonl")
    ap.add_argument("--held-2wiki",
        default="results/third_benchmark_extractability/pairs.jsonl")
    args = ap.parse_args()

    held_m = load_held_out(Path(args.held_musique), "musique_id")
    held_w = load_held_out(Path(args.held_2wiki), "twowiki_id")
    print(f"[held-out] musique ids={len(held_m)}  2wiki ids={len(held_w)}")

    base = Path(__file__).resolve().parent.parent
    pool_m, dm, rj_m = collect_pool("musique", base / args.musique_data,
                                    bme.build_record, "musique_id", held_m,
                                    args.n_musique, args.seed)
    pool_w, dw, rj_w = collect_pool("2wiki", base / args.twowiki_data,
                                    bwe.build_record, "twowiki_id", held_w,
                                    args.n_2wiki, args.seed + 1)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    t0_recs, n0_recs, s0_recs = [], [], []
    for rec in pool_m:
        t0_recs.append(emit_train_record(rec, "T0", "musique"))
        n0_recs.append(emit_train_record(rec, "N0", "musique"))
        s0_recs.append(emit_train_record(rec, "S0", "musique"))
    for rec in pool_w:
        t0_recs.append(emit_train_record(rec, "T0", "2wiki"))
        n0_recs.append(emit_train_record(rec, "N0", "2wiki"))
        s0_recs.append(emit_train_record(rec, "S0", "2wiki"))
    rng = random.Random(args.seed + 2)
    rng.shuffle(t0_recs); rng.shuffle(n0_recs); rng.shuffle(s0_recs)

    with open(out_dir / "train_T0.jsonl", "w") as f:
        for r in t0_recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "train_N0.jsonl", "w") as f:
        for r in n0_recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "train_S0.jsonl", "w") as f:
        for r in s0_recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Balanced pool = interleave T0/N0/S0, then shuffle.
    balanced = list(t0_recs) + list(n0_recs) + list(s0_recs)
    rng2 = random.Random(args.seed + 3); rng2.shuffle(balanced)
    with open(out_dir / "train_balanced.jsonl", "w") as f:
        for r in balanced: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Diagnostic: target distribution + 5 examples per condition.
    print("\n=== target distribution (train_balanced.jsonl) ===")
    cond_counts = Counter(r["condition"] for r in balanced)
    first_word_by_cond = {c: Counter() for c in ("T0", "N0", "S0")}
    for r in balanced:
        tgt = r["target_text"].strip()
        first = tgt.split()[0] if tgt else "<empty>"
        first_word_by_cond[r["condition"]][first] += 1
    for c in ("T0", "N0", "S0"):
        print(f"  {c}: n={cond_counts[c]} first-word={dict(first_word_by_cond[c])}")
    print("\n=== 5 decoded examples per condition ===")
    for c in ("T0", "N0", "S0"):
        examples = [r for r in balanced if r["condition"] == c][:5]
        print(f"\n--- {c} ---")
        for i, r in enumerate(examples):
            q = (r.get("question") or "")[:90]
            tgt = (r.get("target_text") or "").replace("\n", " | ")[:120]
            print(f"  [{c}#{i}] sid={r['sample_id']}  Q: {q!r}")
            print(f"           TARGET: {tgt!r}")

    audit = {
        "seed": args.seed,
        "n_T0": len(t0_recs), "n_N0": len(n0_recs), "n_S0": len(s0_recs),
        "n_balanced": len(balanced),
        "n_musique_accepted": len(pool_m), "n_musique_held_overlap": dm,
        "n_2wiki_accepted": len(pool_w), "n_2wiki_held_overlap": dw,
        "musique_reject_reasons": dict(rj_m),
        "2wiki_reject_reasons": dict(rj_w),
        "held_out_id_overlap": 0,
        "first_word_by_cond": {c: dict(first_word_by_cond[c])
                               for c in ("T0", "N0", "S0")},
    }
    json.dump(audit, open(out_dir / "train_pool_audit.json", "w"), indent=2)
    print(f"\n[done] T0={len(t0_recs)} N0={len(n0_recs)} S0={len(s0_recs)} "
          f"balanced={len(balanced)} -> {out_dir}")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
