#!/usr/bin/env python3
"""Step 1: Minimal paired evidence corruption at sentence level.

For each bridge-type HotpotQA sample:
  - Identify the supporting-fact (SF) sentence(s) inside the observation
  - clean: original observation (all sentences intact)
  - corrupted (A): SF sentence replaced with distractor sentence of ~same length
  - control (B): non-SF sentence replaced with distractor sentence of ~same length
  - identity (C): clean == corrupted

This ensures corruption is truly "minimal" — only the bridge evidence changes.
"""

import json, re, sys, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder

# ── Observation parsing ─────────────────────────────────────────────────────

OBS_ENTRY_RE = re.compile(r'\[(\d+)\]\s*([^:]+):\s*(.*?)(?=\n\n\[\d+\]|\Z)', re.DOTALL)


def parse_observation(obs: str):
    entries = []
    for m in OBS_ENTRY_RE.finditer(obs):
        entries.append({"idx": int(m.group(1)), "title": m.group(2).strip(),
                        "text": m.group(3).strip()})
    return entries


def title_match(a: str, b: str) -> bool:
    al, bl = a.lower(), b.lower()
    return al in bl or bl in al


# ── Sentence-level matching ────────────────────────────────────────────────


def find_sentence_in_text(sent: str, text: str) -> tuple:
    """Find sentence position in text, handling whitespace differences.
    Returns (start, end) or None."""
    sent_clean = sent.strip()
    # Try exact match first
    idx = text.find(sent_clean)
    if idx >= 0:
        return (idx, idx + len(sent_clean))
    # Normalize whitespace
    sent_norm = re.sub(r'\s+', ' ', sent_clean)
    text_norm = re.sub(r'\s+', ' ', text)
    idx = text_norm.find(sent_norm)
    if idx >= 0:
        return (idx, idx + len(sent_norm))
    # Try first 40 chars (for truncated observations)
    if len(sent_clean) > 40:
        idx = text.find(sent_clean[:40])
        if idx >= 0:
            # Find end: look for next sentence boundary or end
            end = idx + len(sent_clean)
            return (idx, min(end, len(text)))
    return None


def get_replacement_sentence(target_len: int, distractor_sents: list, rng) -> str:
    """Pick a distractor sentence closest in length to target."""
    if not distractor_sents:
        return "This information is currently unavailable."
    scored = [(abs(len(s) - target_len), s) for s in distractor_sents if len(s.strip()) > 10]
    if not scored:
        return distractor_sents[0]
    scored.sort(key=lambda x: x[0])
    # Pick from top-3 to add variation
    top = scored[:min(3, len(scored))]
    return rng.choice(top)[1].strip()


# ── Sample selection and corruption ────────────────────────────────────────


def select_minimal_samples(baseline_path, hotpotqa_path, n=100, seed=42):
    """Select samples where SF sentences can be precisely located in observation."""
    with open(hotpotqa_path) as f:
        hotpot_by_id = {s["_id"]: s for s in json.load(f)}

    candidates = []
    with open(baseline_path) as f:
        for line in f:
            ep = json.loads(line)
            sid = ep["sample_id"]
            hp = hotpot_by_id.get(sid)
            if not hp or hp.get("type") != "bridge":
                continue
            steps = ep.get("steps", [])
            s0 = steps[0] if steps else None
            if not s0 or s0.get("action") != "search" or not s0.get("observation"):
                continue

            obs = s0["observation"]
            sf_facts = hp.get("supporting_facts", [])
            sf_titles = list(set(t for t, _ in sf_facts))
            entries = parse_observation(obs)
            if not entries:
                continue

            # Find SF paragraph in observation
            sf_entry = None
            sf_ctx_title = None
            sf_ctx_sents = None
            for e in entries:
                for t, sents in hp["context"]:
                    if title_match(e["title"], t) and t in sf_titles:
                        sf_entry = e
                        sf_ctx_title = t
                        sf_ctx_sents = sents
                        break
                if sf_entry:
                    break
            if not sf_entry or not sf_ctx_sents:
                continue

            # Get SF sentence indices for this title
            sf_indices = [idx for title, idx in sf_facts
                          if title == sf_ctx_title and idx < len(sf_ctx_sents)]
            if not sf_indices:
                continue

            # Find SF sentences in observation text
            sf_locations = []
            all_found = True
            for si in sf_indices:
                sent = sf_ctx_sents[si]
                loc = find_sentence_in_text(sent, sf_entry["text"])
                if loc is None:
                    all_found = False
                    break
                sf_locations.append({"sent_idx": si, "start": loc[0], "end": loc[1],
                                     "text": sent.strip()})
            if not all_found:
                continue

            # Need at least 1 non-SF sentence for control
            non_sf_indices = [i for i in range(len(sf_ctx_sents))
                              if i not in sf_indices and len(sf_ctx_sents[i].strip()) > 10]
            if not non_sf_indices:
                continue

            # Find non-SF sentence locations
            non_sf_locations = []
            for ni in non_sf_indices:
                sent = sf_ctx_sents[ni]
                loc = find_sentence_in_text(sent, sf_entry["text"])
                if loc:
                    non_sf_locations.append({"sent_idx": ni, "start": loc[0],
                                             "end": loc[1], "text": sent.strip()})
            if not non_sf_locations:
                continue

            # Gather distractor sentences for replacement
            dist_sents = []
            for t, sents in hp["context"]:
                if not any(title_match(t, st) for st in sf_titles):
                    dist_sents.extend([s.strip() for s in sents if len(s.strip()) > 10])

            if len(dist_sents) < 3:
                continue

            candidates.append({
                "sample_id": sid, "question": ep["question"],
                "answer": hp["answer"],
                "step0_query": s0["action_input"], "step0_obs": obs,
                "sf_entry": sf_entry, "sf_ctx_title": sf_ctx_title,
                "sf_ctx_sents": sf_ctx_sents,
                "sf_locations": sf_locations, "non_sf_locations": non_sf_locations,
                "dist_sents": dist_sents, "entries": entries,
            })

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:n]
    print(f"Selected {len(selected)} / {len(candidates)} candidates (need {n})")
    return selected


def make_sentence_corruption(sample, group, rng):
    """Create (clean_obs, corrupted_obs) with sentence-level manipulation.

    Group A (evidence swap): replace SF sentence(s) with distractor sentence(s)
    Group B (control swap): replace a non-SF sentence with distractor sentence
    Group C (identity): clean == corrupted
    """
    entry = sample["sf_entry"]
    text = entry["text"]

    if group == "C":
        return text, text

    if group == "A":
        # Replace SF sentences
        modified = text
        offset = 0
        for loc in sorted(sample["sf_locations"], key=lambda x: x["start"]):
            repl = get_replacement_sentence(len(loc["text"]), sample["dist_sents"], rng)
            start = loc["start"] + offset
            end = loc["end"] + offset
            modified = modified[:start] + repl + modified[end:]
            offset += len(repl) - (end - start)
        return text, modified

    if group == "B":
        # Replace a non-SF sentence (control)
        loc = rng.choice(sample["non_sf_locations"])
        repl = get_replacement_sentence(len(loc["text"]), sample["dist_sents"], rng)
        modified = text[:loc["start"]] + repl + text[loc["end"]:]
        return text, modified

    raise ValueError(f"Unknown group: {group}")


def rebuild_obs_with_modified_entry(sample, modified_text):
    """Rebuild the full observation with one entry's text replaced."""
    parts = []
    for i, e in enumerate(sample["entries"], 1):
        if e["title"] == sample["sf_entry"]["title"]:
            parts.append(f"[{i}] {e['title']}: {modified_text}")
        else:
            parts.append(f"[{i}] {e['title']}: {e['text']}")
    return "\n\n".join(parts)


def build_prompt(tokenizer, question, query, observation):
    pb = PromptBuilder(tools=["search", "calculator"])
    steps = [{"action": "search", "action_input": query,
              "observation": observation[:1500]}]
    messages = pb.build_full_prompt(question, steps)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


# ── Main: construct and validate dataset ───────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--output-dir", default="results/minimal_corruption")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    baseline_path = "results/l20_rho020_n500/baseline_results.jsonl"
    hotpotqa_path = "data/hotpotqa/hotpot_dev_distractor_v1.json"

    samples = select_minimal_samples(baseline_path, hotpotqa_path, n=args.n, seed=args.seed)

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, s in enumerate(samples):
        clean_a, corrupt_a = make_sentence_corruption(s, "A", rng)
        clean_b, corrupt_b = make_sentence_corruption(s, "B", rng)
        clean_c, corrupt_c = make_sentence_corruption(s, "C", rng)

        # Verify clean is consistent
        assert clean_a == clean_b == clean_c, f"Clean mismatch for {s['sample_id']}"

        # Compute edit distances for validation
        def edit_frac(a, b):
            return sum(1 for x, y in zip(a, b) if x != y) / max(len(a), len(b)) if max(len(a), len(b)) > 0 else 0

        rec = {
            "sample_id": s["sample_id"],
            "question": s["question"],
            "answer": s["answer"],
            "step0_query": s["step0_query"],
            "sf_title": s["sf_ctx_title"],
            "n_sf_sents": len(s["sf_locations"]),
            "sf_sent_texts": [loc["text"][:80] for loc in s["sf_locations"]],
            "clean_entry_text": clean_a[:200],
            "corrupt_A_text": corrupt_a[:200],
            "corrupt_B_text": corrupt_b[:200],
            "edit_frac_A": round(edit_frac(clean_a, corrupt_a), 3),
            "edit_frac_B": round(edit_frac(clean_b, corrupt_b), 3),
            "len_clean": len(clean_a),
            "len_corrupt_A": len(corrupt_a),
            "len_corrupt_B": len(corrupt_b),
        }
        records.append(rec)

        if i < 3:
            print(f"\n=== Sample {s['sample_id']} ===")
            print(f"Q: {s['question']}")
            print(f"SF sents: {[loc['text'][:60] for loc in s['sf_locations']]}")
            print(f"Clean (first 120): {clean_a[:120]}")
            print(f"Corrupt A (first 120): {corrupt_a[:120]}")
            print(f"Corrupt B (first 120): {corrupt_b[:120]}")
            print(f"Edit fraction: A={rec['edit_frac_A']:.3f}, B={rec['edit_frac_B']:.3f}")
            print(f"Lengths: clean={rec['len_clean']}, A={rec['len_corrupt_A']}, B={rec['len_corrupt_B']}")

    # Summary stats
    ef_a = [r["edit_frac_A"] for r in records]
    ef_b = [r["edit_frac_B"] for r in records]
    import numpy as np
    print(f"\n{'='*60}")
    print(f"Dataset: {len(records)} samples")
    print(f"Edit fraction A (evidence): mean={np.mean(ef_a):.3f}, median={np.median(ef_a):.3f}")
    print(f"Edit fraction B (control):  mean={np.mean(ef_b):.3f}, median={np.median(ef_b):.3f}")
    print(f"Length ratio A/clean: mean={np.mean([r['len_corrupt_A']/r['len_clean'] for r in records]):.3f}")
    print(f"Length ratio B/clean: mean={np.mean([r['len_corrupt_B']/r['len_clean'] for r in records]):.3f}")

    # Save
    with open(out_dir / "corruption_dataset.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Also save full data for later steps
    full_records = []
    rng2 = random.Random(args.seed)
    for s in samples:
        clean_a, corrupt_a = make_sentence_corruption(s, "A", rng2)
        clean_b, corrupt_b = make_sentence_corruption(s, "B", rng2)
        obs_clean = rebuild_obs_with_modified_entry(s, clean_a)
        obs_corrupt_A = rebuild_obs_with_modified_entry(s, corrupt_a)
        obs_corrupt_B = rebuild_obs_with_modified_entry(s, corrupt_b)
        full_records.append({
            "sample_id": s["sample_id"],
            "question": s["question"],
            "answer": s["answer"],
            "step0_query": s["step0_query"],
            "obs_clean": obs_clean,
            "obs_corrupt_A": obs_corrupt_A,
            "obs_corrupt_B": obs_corrupt_B,
        })

    with open(out_dir / "full_corruption_data.jsonl", "w") as f:
        for r in full_records:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()

