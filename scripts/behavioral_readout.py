#!/usr/bin/env python3
"""
Behavioral Readout (追加2) — First-Token Logit Margin
=====================================================
For Group A (evidence corruption) and Group B (distractor swap) pairs,
compute the first-token logit margin:

    margin = logit("Action") - logit("Final")

at the decision point (first generated token after prompt).

A positive margin ⇒ model leans toward searching again.
A negative margin ⇒ model leans toward stopping (Final Answer).

We measure Δmargin = margin_clean - margin_corrupt for each sample.
If evidence corruption (A) shifts the margin more than distractor swap (B),
the internal action shift translates to a behavioral-level effect.

Output → results/paired_corruption/behavioral_readout_results.json
"""

import os, sys, json, argparse, random
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from scipy.stats import mannwhitneyu, wilcoxon

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, get_action_token_ids
from scripts.paired_corruption_analysis import (
    select_samples, make_corrupted_obs,
)


def build_prompt(tokenizer, question, query, observation):
    """Build prompt using DEFAULT_SYSTEM_PROMPT (first token = Action or Final)."""
    pb = PromptBuilder(tools=["search", "calculator"])
    steps = [{"action": "search", "action_input": query,
              "observation": observation[:1500]}]
    messages = pb.build_full_prompt(question, steps)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def get_first_token_margin(model, tokenizer, prompt, action_tid, final_tid):
    """Compute logit(Action) - logit(Final) at the first generated token."""
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(input_ids)
    logits = out.logits[0, -1, :]  # last position → first generated token
    margin = float(logits[action_tid].item() - logits[final_tid].item())
    top_action = "search" if margin > 0 else "stop"
    return margin, top_action


def run_experiment(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    # Get token IDs for "Action" and "Final"
    action_tids = get_action_token_ids(tokenizer, "tool_call")
    final_tids = get_action_token_ids(tokenizer, "finish")
    assert len(action_tids) >= 1 and len(final_tids) >= 1, \
        f"Need at least 1 token each: action={action_tids}, final={final_tids}"
    action_tid = action_tids[0]
    final_tid = final_tids[0]
    print(f"Token IDs: Action={action_tid} ({tokenizer.decode([action_tid])}), "
          f"Final={final_tid} ({tokenizer.decode([final_tid])})")

    samples = select_samples(args.baseline_trace, args.hotpotqa_data,
                             n=args.n_samples, seed=args.seed)

    rng = random.Random(args.seed)
    results = {"A": [], "B": []}

    # Storage for statistics
    margins_clean = {"A": [], "B": []}
    margins_corrupt = {"A": [], "B": []}
    delta_margins = {"A": [], "B": []}    # clean - corrupt (signed)
    abs_delta_margins = {"A": [], "B": []}  # |clean - corrupt|

    for group in ["A", "B"]:
        print(f"\n=== Group {group} ({len(samples)} pairs) ===", flush=True)
        for i, sample in enumerate(samples):
            clean_obs, corrupted_obs = make_corrupted_obs(sample, group, rng)

            prompt_clean = build_prompt(tokenizer, sample["question"],
                                        sample["step0_query"], clean_obs)
            prompt_corrupt = build_prompt(tokenizer, sample["question"],
                                          sample["step0_query"], corrupted_obs)

            m_clean, act_clean = get_first_token_margin(
                model, tokenizer, prompt_clean, action_tid, final_tid)
            m_corrupt, act_corrupt = get_first_token_margin(
                model, tokenizer, prompt_corrupt, action_tid, final_tid)

            dm = m_clean - m_corrupt
            margins_clean[group].append(m_clean)
            margins_corrupt[group].append(m_corrupt)
            delta_margins[group].append(dm)
            abs_delta_margins[group].append(abs(dm))

            results[group].append({
                "sample_id": sample["sample_id"],
                "margin_clean": m_clean,
                "margin_corrupt": m_corrupt,
                "delta_margin": dm,
                "action_clean": act_clean,
                "action_corrupt": act_corrupt,
                "flipped": act_clean != act_corrupt,
            })

            if (i + 1) % 10 == 0:
                flips = sum(1 for r in results[group] if r["flipped"])
                print(f"  [{i+1}/{len(samples)}] flipped={flips}  "
                      f"mean |Δm|={np.mean(abs_delta_margins[group]):.3f}",
                      flush=True)

    # ── Statistics ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BEHAVIORAL READOUT — FIRST-TOKEN LOGIT MARGIN")
    print("=" * 70)

    stats = {}
    for g in ["A", "B"]:
        mc = np.array(margins_clean[g])
        mr = np.array(margins_corrupt[g])
        dm = np.array(delta_margins[g])
        adm = np.array(abs_delta_margins[g])
        flips = sum(1 for r in results[g] if r["flipped"])
        stats[g] = {
            "mean_margin_clean": float(mc.mean()),
            "mean_margin_corrupt": float(mr.mean()),
            "mean_delta_margin": float(dm.mean()),
            "mean_abs_delta_margin": float(adm.mean()),
            "std_abs_delta_margin": float(adm.std()),
            "n_flipped": flips,
            "n_total": len(results[g]),
        }
        print(f"\nGroup {g}:")
        print(f"  mean margin clean   = {mc.mean():.4f} ± {mc.std():.4f}")
        print(f"  mean margin corrupt = {mr.mean():.4f} ± {mr.std():.4f}")
        print(f"  mean Δmargin        = {dm.mean():.4f} ± {dm.std():.4f}")
        print(f"  mean |Δmargin|      = {adm.mean():.4f} ± {adm.std():.4f}")
        print(f"  flipped (argmax)    = {flips}/{len(results[g])}")

    # Tests
    tests = {}

    # 1. A vs B: |Δmargin| (MW)
    adm_A = np.array(abs_delta_margins["A"])
    adm_B = np.array(abs_delta_margins["B"])
    u, p = mannwhitneyu(adm_A, adm_B, alternative="greater")
    tests["abs_delta_margin_A_vs_B"] = {"U": float(u), "p": float(p)}
    print(f"\n--- Tests ---")
    print(f"|Δmargin| A vs B (MW):  U={u:.0f}, p={p:.4f}")

    # 2. A |Δmargin| vs 0 (Wilcoxon)
    try:
        w, pw = wilcoxon(adm_A, alternative="greater")
        tests["abs_delta_margin_A_vs_0"] = {"W": float(w), "p": float(pw)}
        print(f"|Δmargin| A vs 0 (Wilcoxon): W={w:.0f}, p={pw:.6f}")
    except Exception as e:
        print(f"|Δmargin| A vs 0: skipped ({e})")

    # 3. Signed Δmargin: does A shift margin MORE negative than B?
    #    (removing evidence should make the model LESS inclined to search)
    dm_A = np.array(delta_margins["A"])
    dm_B = np.array(delta_margins["B"])
    u2, p2 = mannwhitneyu(dm_A, dm_B, alternative="greater")
    tests["signed_delta_margin_A_vs_B"] = {"U": float(u2), "p": float(p2)}
    print(f"Signed Δmargin A vs B (MW): U={u2:.0f}, p={p2:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {"n_samples": len(samples), "seed": args.seed,
                    "model": args.model,
                    "action_token_id": action_tid, "final_token_id": final_tid},
        "per_group_stats": stats,
        "tests": tests,
        "per_sample": {g: results[g] for g in ["A", "B"]},
    }
    out_path = os.path.join(args.output_dir, "behavioral_readout_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--baseline-trace",
                    default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--hotpotqa-data",
                    default="data/hotpotqa/hotpot_dev_distractor_v1.json")
    ap.add_argument("--output-dir", default="results/paired_corruption")
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

