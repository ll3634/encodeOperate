#!/usr/bin/env python3
"""
Paired Evidence Corruption + Distractor Control
================================================
Three groups × 50 paired samples.  For each pair we extract L20 (p0)
activations and measure Δh projected onto evidence_dir and action_dir.

Group A – Evidence Corruption:   swap supporting paragraph → distractor
Group B – Distractor Swap Ctrl:  swap one distractor → another distractor
Group C – Identity Control:      no change (sanity: Δh ≈ 0)

Output → results/paired_corruption/
"""

import os, sys, re, json, argparse, random
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from scipy.stats import mannwhitneyu, wilcoxon, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder
from steering.hook_utils import get_model_layers

# ── Observation parsing / reconstruction ─────────────────────────────────────

OBS_ENTRY_RE = re.compile(r'\[(\d+)\]\s*([^:]+):\s*(.*?)(?=\n\n\[\d+\]|\Z)', re.DOTALL)


def parse_observation(obs: str):
    """Parse '[N] Title: text...' entries from observation string."""
    entries = []
    for m in OBS_ENTRY_RE.finditer(obs):
        entries.append({"idx": int(m.group(1)), "title": m.group(2).strip(),
                        "text": m.group(3).strip()})
    return entries


def rebuild_observation(entries):
    """Reconstruct observation string from parsed entries."""
    parts = []
    for i, e in enumerate(entries, 1):
        parts.append(f"[{i}] {e['title']}: {e['text']}")
    return "\n\n".join(parts)


def title_match(a: str, b: str) -> bool:
    al, bl = a.lower(), b.lower()
    return al in bl or bl in al


def get_hotpot_paragraph_text(context, title: str) -> str:
    """Get concatenated sentence text for a title from HotpotQA context."""
    for t, sents in context:
        if title_match(t, title):
            return " ".join(sents)
    return ""


# ── Sample selection ─────────────────────────────────────────────────────────

def select_samples(baseline_path, hotpotqa_path, n=50, seed=42):
    """Select n 1-doc samples with exactly 1 supporting para in observation.

    For samples where the observation has no distractors, we build a
    standardized 2-entry observation by appending a distractor from the
    HotpotQA context.  This ensures both Group A and Group B have a
    valid swap target while keeping prompt structure identical.
    """
    with open(hotpotqa_path) as f:
        hotpot_by_id = {s["_id"]: s for s in json.load(f)}

    candidates = []
    with open(baseline_path) as f:
        for line in f:
            ep = json.loads(line)
            sid = ep["sample_id"]
            hp = hotpot_by_id.get(sid)
            if not hp:
                continue
            steps = ep.get("steps", [])
            s0 = steps[0] if steps else None
            if not s0 or s0.get("action") != "search" or not s0.get("observation"):
                continue

            obs = s0["observation"]
            sf_titles = list(set(t for t, _ in hp.get("supporting_facts", [])))
            entries = parse_observation(obs)
            if not entries:
                continue

            # Count supporting entries in observation
            sup_entries = [e for e in entries if any(title_match(e["title"], st) for st in sf_titles)]

            # Need exactly 1 supporting entry in obs
            if len(sup_entries) != 1:
                continue

            # Gather ALL distractor titles from HotpotQA context (not sup)
            all_ctx_dist = [t for t, _ in hp["context"]
                            if not any(title_match(t, st) for st in sf_titles)]
            # Need at least 3 distractors for A-replacement, B-base, B-replacement
            if len(all_ctx_dist) < 3:
                continue

            candidates.append({
                "sample_id": sid,
                "question": ep["question"],
                "step0_query": s0["action_input"],
                "step0_obs": obs,
                "entries": entries,
                "sup_entry_idx": entries.index(sup_entries[0]),
                "sf_titles": sf_titles,
                "context": hp["context"],
                "all_ctx_dist": all_ctx_dist,
            })

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:n]
    print(f"Selected {len(selected)} / {len(candidates)} candidates (need {n})")
    return selected


# ── Prompt building ──────────────────────────────────────────────────────────

def _apply_chat_template_safe(tokenizer, messages):
    """Apply chat template; handles:
    - Qwen3: enable_thinking=False to prevent <think> tokens breaking margin.
    - Gemma: merge system→user if model rejects system role.
    """
    chat_template = getattr(tokenizer, "chat_template", "") or ""
    extra_kwargs = {"enable_thinking": False} if "enable_thinking" in chat_template else {}
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **extra_kwargs)
    except Exception as e:
        if "system" not in str(e).lower() and "System" not in str(e):
            raise
        merged, sys_text = [], ""
        for msg in messages:
            if msg["role"] == "system":
                sys_text = msg["content"]
            elif msg["role"] == "user" and sys_text:
                merged.append({"role": "user",
                                "content": sys_text + "\n\n" + msg["content"]})
                sys_text = ""
            else:
                merged.append(msg)
        return tokenizer.apply_chat_template(
            merged, tokenize=False, add_generation_prompt=True, **extra_kwargs)


def build_prompt(tokenizer, question, query, observation):
    pb = PromptBuilder(tools=["search", "calculator"])
    steps = [{"action": "search", "action_input": query,
              "observation": observation[:1500]}]
    messages = pb.build_full_prompt(question, steps)
    return _apply_chat_template_safe(tokenizer, messages)


# ── Activation extraction ───────────────────────────────────────────────────

def extract_l20_hidden(model, tokenizer, prompt, layer_idx=20):
    """Extract L20 last-token hidden state."""
    layers = get_model_layers(model)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    captured = {}

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h[0, -1, :].detach().float().cpu().numpy()

    handle = layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(input_ids)
    handle.remove()
    return captured["h"]


# ── Group construction ───────────────────────────────────────────────────────

def make_corrupted_obs(sample, group, rng):
    """
    Return (clean_obs, corrupted_obs) for a given group.

    We build standardized 2-entry observations:
      entry[0] = supporting paragraph  (position 0 = sup_entry_idx)
      entry[1] = distractor paragraph  (from context, chosen deterministically
                                        per-sample so clean obs is consistent)

    Group A: swap entry[0] (supporting) → different distractor  (evidence removed)
    Group B: swap entry[1] (distractor) → different distractor  (evidence unchanged)
    Group C: identity (clean == corrupted)

    All three groups use the *same* clean observation for a given sample.
    """
    ctx = sample["context"]
    distractors = list(sample["all_ctx_dist"])  # copy

    # Deterministic per-sample: pick dist_base as the first distractor
    # alphabetically (so clean obs is identical across groups)
    distractors_sorted = sorted(distractors)
    dist_base = distractors_sorted[0]
    dist_base_text = get_hotpot_paragraph_text(ctx, dist_base) or f"Info about {dist_base}."

    # Build clean 2-entry observation: [supporting, dist_base]
    sup_entry = sample["entries"][sample["sup_entry_idx"]]
    clean_entries = [
        {"idx": 1, "title": sup_entry["title"], "text": sup_entry["text"]},
        {"idx": 2, "title": dist_base, "text": dist_base_text},
    ]
    clean_obs = rebuild_observation(clean_entries)

    if group == "C":
        return clean_obs, clean_obs

    # Pick replacement distractor (not dist_base, not supporting)
    other_dist = [d for d in distractors_sorted if d != dist_base]

    if group == "A":
        # Swap supporting (entry 0) with a random other distractor
        repl_title = rng.choice(other_dist)
        repl_text = get_hotpot_paragraph_text(ctx, repl_title) or f"Info about {repl_title}."
        corrupt_entries = [
            {"idx": 1, "title": repl_title, "text": repl_text},
            clean_entries[1],
        ]

    elif group == "B":
        # Swap distractor (entry 1) with a random other distractor
        repl_title = rng.choice(other_dist)
        repl_text = get_hotpot_paragraph_text(ctx, repl_title) or f"Info about {repl_title}."
        corrupt_entries = [
            clean_entries[0],
            {"idx": 2, "title": repl_title, "text": repl_text},
        ]

    corrupted_obs = rebuild_observation(corrupt_entries)
    return clean_obs, corrupted_obs


# ── Main experiment ──────────────────────────────────────────────────────────

def run_experiment(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    # Load directions
    ev_data = np.load(args.evidence_dir)
    evidence_dir = ev_data["decision_direction"].astype(np.float32)
    evidence_dir /= np.linalg.norm(evidence_dir) + 1e-12

    act_data = np.load(args.action_dir)
    action_dir = act_data["decision_direction"].astype(np.float32)
    action_dir /= np.linalg.norm(action_dir) + 1e-12

    print(f"cos(evidence, action) = {np.dot(evidence_dir, action_dir):.4f}")

    # Select samples
    samples = select_samples(args.baseline_trace, args.hotpotqa_data,
                             n=args.n_samples, seed=args.seed)
    if len(samples) < args.n_samples:
        print(f"WARNING: only {len(samples)} samples available")

    rng = random.Random(args.seed)
    results = {"A": [], "B": [], "C": []}

    for gi, group in enumerate(["A", "B", "C"]):
        print(f"\n=== Group {group} ({len(samples)} pairs) ===", flush=True)
        for i, sample in enumerate(samples):
            clean_obs, corrupted_obs = make_corrupted_obs(sample, group, rng)

            prompt_clean = build_prompt(tokenizer, sample["question"],
                                        sample["step0_query"], clean_obs)
            prompt_corrupt = build_prompt(tokenizer, sample["question"],
                                          sample["step0_query"], corrupted_obs)

            h_clean = extract_l20_hidden(model, tokenizer, prompt_clean,
                                          layer_idx=args.layer)
            h_corrupt = extract_l20_hidden(model, tokenizer, prompt_corrupt,
                                            layer_idx=args.layer)

            delta_h = h_clean - h_corrupt
            delta_ev = abs(float(np.dot(delta_h, evidence_dir)))
            delta_act = abs(float(np.dot(delta_h, action_dir)))
            delta_norm = float(np.linalg.norm(delta_h))

            results[group].append({
                "sample_id": sample["sample_id"],
                "delta_evidence": delta_ev,
                "delta_action": delta_act,
                "delta_norm": delta_norm,
            })

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(samples)}]", flush=True)

    # ── Statistics ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    stats = {}
    for group in ["A", "B", "C"]:
        dev = np.array([r["delta_evidence"] for r in results[group]])
        dac = np.array([r["delta_action"] for r in results[group]])
        dn = np.array([r["delta_norm"] for r in results[group]])
        stats[group] = {
            "mean_delta_evidence": float(dev.mean()),
            "std_delta_evidence": float(dev.std()),
            "mean_delta_action": float(dac.mean()),
            "std_delta_action": float(dac.std()),
            "mean_delta_norm": float(dn.mean()),
        }
        print(f"\nGroup {group}:")
        print(f"  mean |Δ evidence| = {dev.mean():.4f} ± {dev.std():.4f}")
        print(f"  mean |Δ action|   = {dac.mean():.4f} ± {dac.std():.4f}")
        print(f"  mean ‖Δh‖         = {dn.mean():.4f}")

    # Mann-Whitney tests
    tests = {}

    # A vs B: delta_action
    da_A = np.array([r["delta_action"] for r in results["A"]])
    da_B = np.array([r["delta_action"] for r in results["B"]])
    de_A = np.array([r["delta_evidence"] for r in results["A"]])
    de_B = np.array([r["delta_evidence"] for r in results["B"]])

    u_act, p_act = mannwhitneyu(da_A, da_B, alternative="greater")
    u_ev, p_ev = mannwhitneyu(de_A, de_B, alternative="greater")

    tests["A_vs_B_delta_action"] = {"U": float(u_act), "p": float(p_act)}
    tests["A_vs_B_delta_evidence"] = {"U": float(u_ev), "p": float(p_ev)}

    print(f"\n--- Statistical Tests ---")
    print(f"A vs B delta_action:   U={u_act:.0f}, p={p_act:.4f}")
    print(f"A vs B delta_evidence: U={u_ev:.0f}, p={p_ev:.4f}")

    # A delta_action vs 0 (Wilcoxon)
    try:
        w_a, p_a0 = wilcoxon(da_A, alternative="greater")
        tests["A_delta_action_vs_0"] = {"W": float(w_a), "p": float(p_a0)}
        print(f"A delta_action vs 0:   W={w_a:.0f}, p={p_a0:.4f}")
    except Exception as e:
        print(f"A delta_action vs 0:   skipped ({e})")

    # B delta_action vs 0
    try:
        w_b, p_b0 = wilcoxon(da_B, alternative="greater")
        tests["B_delta_action_vs_0"] = {"W": float(w_b), "p": float(p_b0)}
        print(f"B delta_action vs 0:   W={w_b:.0f}, p={p_b0:.4f}")
    except Exception as e:
        print(f"B delta_action vs 0:   skipped ({e})")

    # Spearman correlation within Group A
    rho_sp, p_sp = spearmanr(de_A, da_A)
    tests["A_spearman_evidence_action"] = {"rho": float(rho_sp), "p": float(p_sp)}
    print(f"\nGroup A Spearman(Δevidence, Δaction): rho={rho_sp:.4f}, p={p_sp:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_samples": len(samples),
            "layer": args.layer,
            "seed": args.seed,
            "evidence_dir_path": args.evidence_dir,
            "action_dir_path": args.action_dir,
            "cos_evidence_action": float(np.dot(evidence_dir, action_dir)),
        },
        "per_group_stats": stats,
        "tests": tests,
        "per_sample": {g: results[g] for g in ["A", "B", "C"]},
    }

    out_path = os.path.join(args.output_dir, "paired_corruption_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--baseline-trace",
                    default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--hotpotqa-data",
                    default="data/hotpotqa/hotpot_dev_distractor_v1.json")
    ap.add_argument("--evidence-dir",
                    default="results/phase1_probe/probe_direction_l20.npz")
    ap.add_argument("--action-dir",
                    default="steering/directions/direction_search_v3_layer20.npz")
    ap.add_argument("--output-dir", default="results/paired_corruption")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

