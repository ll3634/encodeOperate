#!/usr/bin/env python3
"""
Positive Control Validity Check for Condition A and A_v2.

Tests whether the judge conditions (A=YES/NO, A_v2=SUFFICIENT/INSUFFICIENT) actually
perform evidence evaluation, or whether they exhibit a "default to insufficient" bias.

Three groups:
  P1  : 103 baseline-correct samples (is_correct=True in baseline traces)
        If A is unbiased: insufficient_rate << 88.9% (overall rate)
        If A is biased:   insufficient_rate ≈ 88.9%

  P2  : 5 samples with n_sf_retrieved=2 (both SFs found in step-0 observation)
        These have GENUINELY sufficient evidence in the retrieved text.
        Expected: insufficient_rate should be low if A works.

  SYNTH: 10 synthetic samples — both SF paragraphs concatenated as observation.
        These are FULLY sufficient by construction (the model has both answers).
        If A still says INSUFFICIENT on most of these, the condition is broken.

Usage:
    cd tmc/scripts/e2e_agent
    python scripts/positive_control_judge.py \\
        --labels results/phase1_probe/labels.jsonl \\
        --baseline results/l20_rho020_n500/baseline_results.jsonl \\
        --hotpotqa data/hotpotqa/hotpot_dev_distractor_v1.json \\
        --output-dir results/positive_control \\
        --model Qwen/Qwen2.5-7B-Instruct
"""

import os, sys, json, argparse, re
from pathlib import Path
from typing import Optional, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Reuse prompt definitions from agent_specific_dissociation ──────────────────
SYSTEM_PROMPT_A = """You are an expert at judging whether retrieved evidence is sufficient.

Given a question and a retrieved search result, respond with exactly one word:
YES if the evidence is sufficient to answer the question completely.
NO if the evidence is insufficient and more information is needed.

Your response must be exactly YES or NO, nothing else."""


def build_user_prompt_a(question: str, observation: str) -> str:
    return f"""Question: {question}

Retrieved evidence:
{observation}

Is this evidence sufficient to answer the question? (YES/NO)"""


SYSTEM_PROMPT_A_V2 = """You are an expert at judging evidence quality.

Given a question and a retrieved search result, assess whether the evidence is sufficient to answer the question completely and correctly.

First explain your reasoning in 2-3 sentences. Then conclude your response with exactly one of these two words on its own line:
SUFFICIENT
INSUFFICIENT"""


def build_user_prompt_a_v2(question: str, observation: str) -> str:
    return f"""Question: {question}

Retrieved evidence:
{observation}

Assess the evidence. Explain your reasoning (2-3 sentences), then conclude with SUFFICIENT or INSUFFICIENT."""


def parse_a(text: str) -> Optional[bool]:
    """True=sufficient, False=insufficient, None=fail."""
    t = text.strip().upper()
    if t.startswith("YES"):
        return True
    if t.startswith("NO"):
        return False
    if "YES" in t:
        return True
    if "NO" in t:
        return False
    return None


def parse_a_v2(text: str) -> Optional[bool]:
    """True=sufficient, False=insufficient, None=fail."""
    t = text.strip().upper()
    last_ins = t.rfind("INSUFFICIENT")
    last_suf = t.rfind("SUFFICIENT")
    if last_ins == -1 and last_suf == -1:
        return None
    if last_ins == -1:
        return True
    if last_suf == -1:
        return False
    suf_inside_ins = last_ins + 2
    if last_suf == suf_inside_ins:
        return False
    elif last_suf > suf_inside_ins:
        return True
    else:
        return False


def load_model(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, messages: List[dict], max_new_tokens: int = 20) -> str:
    import torch
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=1.0, pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_conditions_on_group(model, tokenizer, samples: List[dict], group_name: str) -> dict:
    """Run Condition A and A_v2 on a list of samples. Returns stats."""
    results = []
    print(f"\n[{group_name}] Running N={len(samples)} samples...")

    for i, s in enumerate(samples):
        q = s["question"]
        obs = s["observation"]

        # Condition A
        msg_a = [
            {"role": "system", "content": SYSTEM_PROMPT_A},
            {"role": "user", "content": build_user_prompt_a(q, obs)},
        ]
        raw_a = generate_response(model, tokenizer, msg_a, max_new_tokens=5)
        parsed_a = parse_a(raw_a)

        # Condition A_v2
        msg_av2 = [
            {"role": "system", "content": SYSTEM_PROMPT_A_V2},
            {"role": "user", "content": build_user_prompt_a_v2(q, obs)},
        ]
        raw_av2 = generate_response(model, tokenizer, msg_av2, max_new_tokens=100)
        parsed_av2 = parse_a_v2(raw_av2)

        results.append({
            "sample_id": s.get("sample_id", f"synth_{i}"),
            "question": q[:80],
            "is_correct": s.get("is_correct", None),
            "label": s.get("label", None),
            "n_sf_retrieved": s.get("n_sf_retrieved", None),
            "raw_a": raw_a,
            "a_sufficient": parsed_a,          # True=sufficient
            "a_parse_ok": parsed_a is not None,
            "raw_av2": raw_av2[:200],
            "av2_sufficient": parsed_av2,
            "av2_parse_ok": parsed_av2 is not None,
        })

        if (i + 1) % 10 == 0 or i < 3:
            a_insuff = sum(1 for r in results if r["a_parse_ok"] and not r["a_sufficient"])
            a_ok = sum(1 for r in results if r["a_parse_ok"])
            print(f"  [{i+1}/{len(samples)}] A_insuff_so_far={a_insuff}/{a_ok} raw_a={repr(raw_a)}")

    # Aggregate stats
    a_valid = [r for r in results if r["a_parse_ok"]]
    av2_valid = [r for r in results if r["av2_parse_ok"]]
    a_insuff_rate = sum(1 for r in a_valid if not r["a_sufficient"]) / len(a_valid) if a_valid else None
    av2_insuff_rate = sum(1 for r in av2_valid if not r["av2_sufficient"]) / len(av2_valid) if av2_valid else None

    print(f"\n  [{group_name}] Results:")
    print(f"    N = {len(samples)}")
    print(f"    A  parse_ok={len(a_valid)}/{len(results)}  insufficient_rate={a_insuff_rate:.1%}" if a_insuff_rate is not None else "    A: N/A")
    print(f"    A_v2 parse_ok={len(av2_valid)}/{len(results)}  insufficient_rate={av2_insuff_rate:.1%}" if av2_insuff_rate is not None else "    A_v2: N/A")

    # Print 3 examples
    for r in results[:3]:
        print(f"\n    Ex: {r['question']}")
        print(f"       A:   {repr(r['raw_a'])} → {'insuff' if not r['a_sufficient'] else 'suff' if r['a_sufficient'] else 'fail'}")
        print(f"       Av2: {repr(r['raw_av2'][:100])} → {'insuff' if not r['av2_sufficient'] else 'suff' if r['av2_sufficient'] else 'fail'}")

    return {
        "group": group_name,
        "n": len(results),
        "a_parse_ok": len(a_valid),
        "a_insufficient_rate": a_insuff_rate,
        "av2_parse_ok": len(av2_valid),
        "av2_insufficient_rate": av2_insuff_rate,
        "per_sample": results,
    }


def build_synthetic_samples(labels: List[dict], hotpotqa_data: List[dict], n: int = 10) -> List[dict]:
    """
    Build synthetic positive controls: concatenate BOTH SF paragraphs as the observation.
    Selects samples from the 486 where behavioral_stop=True, label=0 (the most interesting
    dissociation cases), matching to HotpotQA for their SF paragraphs.
    """
    hpqa = {d["_id"]: d for d in hotpotqa_data}

    # Prefer: behavioral_stop=True (model stopped), label=0 (0-doc insufficient)
    candidates = [r for r in labels if r["behavioral_stop"] and r["label"] == 0]
    if len(candidates) < n:
        candidates += [r for r in labels if r["behavioral_stop"] and r["label"] == 1]

    synth = []
    for lb in candidates:
        sid = lb["sample_id"]
        if sid not in hpqa:
            continue
        ex = hpqa[sid]

        # Build synthetic observation with both SF paragraphs
        context_map = {c[0]: " ".join(c[1]) for c in ex["context"]}
        sf_titles = list(dict.fromkeys(sf[0] for sf in ex["supporting_facts"]))  # preserve order, unique
        sf_texts = []
        for t in sf_titles:
            if t in context_map:
                sf_texts.append(f"[{len(sf_texts)+1}] {t}: {context_map[t][:500]}")
        if len(sf_texts) < 2:
            continue  # skip if can't find both SFs

        obs = "\n\n".join(sf_texts)
        synth.append({
            "sample_id": sid,
            "question": lb["question"],
            "gold_answer": lb.get("gold_answer", ex.get("answer", "")),
            "observation": obs,
            "label": lb["label"],
            "n_sf_retrieved": lb["n_sf_retrieved"],
            "behavioral_stop": lb["behavioral_stop"],
            "is_correct": lb.get("is_correct", False),
            "note": "synthetic_both_SFs",
        })
        if len(synth) >= n:
            break

    print(f"Built {len(synth)} synthetic samples (both SFs concatenated)")
    return synth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="results/phase1_probe/labels.jsonl")
    parser.add_argument("--baseline", default="results/l20_rho020_n500/baseline_results.jsonl")
    parser.add_argument("--hotpotqa", default="data/hotpotqa/hotpot_dev_distractor_v1.json")
    parser.add_argument("--output-dir", default="results/positive_control")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n-synth", type=int, default=10,
                        help="Number of synthetic (both-SF) samples to run")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading labels...")
    labels = [json.loads(l) for l in open(args.labels)]
    label_map = {r["sample_id"]: r for r in labels}

    print("Loading baseline traces...")
    baseline = {r["sample_id"]: r for r in (json.loads(l) for l in open(args.baseline))}

    print("Loading HotpotQA data...")
    with open(args.hotpotqa) as f:
        hotpotqa_data = json.load(f)

    # ── Build sample groups ──────────────────────────────────────────────────

    # Group P1: baseline-correct samples
    p1_samples = []
    for lb in labels:
        sid = lb["sample_id"]
        ep = baseline.get(sid, {})
        if not ep.get("is_correct", False):
            continue
        steps = ep.get("steps", [])
        if not steps or steps[0].get("action") != "search" or not steps[0].get("observation"):
            continue
        p1_samples.append({
            "sample_id": sid,
            "question": lb["question"],
            "observation": steps[0]["observation"],
            "label": lb["label"],
            "n_sf_retrieved": lb["n_sf_retrieved"],
            "is_correct": True,
        })
    print(f"\nGroup P1 (correct): {len(p1_samples)} samples")
    print(f"  label=0: {sum(1 for s in p1_samples if s['label']==0)}, "
          f"label=1: {sum(1 for s in p1_samples if s['label']==1)}, "
          f"label=2: {sum(1 for s in p1_samples if s['label']==2)}")

    # Group P2: 2-SF retrieved samples
    p2_samples = []
    for lb in labels:
        if lb["n_sf_retrieved"] < 2:
            continue
        sid = lb["sample_id"]
        ep = baseline.get(sid, {})
        steps = ep.get("steps", [])
        if not steps or steps[0].get("action") != "search" or not steps[0].get("observation"):
            continue
        p2_samples.append({
            "sample_id": sid,
            "question": lb["question"],
            "observation": steps[0]["observation"],
            "label": lb["label"],
            "n_sf_retrieved": lb["n_sf_retrieved"],
            "is_correct": ep.get("is_correct", False),
        })
    print(f"Group P2 (2-SF in obs): {len(p2_samples)} samples")

    # Group SYNTH: synthetic both-SF samples
    synth_samples = build_synthetic_samples(labels, hotpotqa_data, n=args.n_synth)

    # ── Reference: overall rates from full dissociation test ────────────────
    # (from agent_specific_dissociation results if available)
    overall_path = Path("results/agent_specific_dissociation/metrics.json")
    overall_a_insuff = 0.889  # fallback from previous run
    if overall_path.exists():
        m = json.load(open(overall_path))
        overall_a_insuff = m.get("continue_rates", {}).get("A_insufficient", overall_a_insuff)
    print(f"\nReference: overall A_insufficient_rate = {overall_a_insuff:.1%}")

    # ── Load model ───────────────────────────────────────────────────────────
    model, tokenizer = load_model(args.model)

    # ── Run conditions ───────────────────────────────────────────────────────
    all_results = {}

    if p1_samples:
        all_results["P1_correct"] = run_conditions_on_group(model, tokenizer, p1_samples, "P1_correct")

    if p2_samples:
        all_results["P2_2sf"] = run_conditions_on_group(model, tokenizer, p2_samples, "P2_2sf")

    if synth_samples:
        all_results["SYNTH_both_SFs"] = run_conditions_on_group(model, tokenizer, synth_samples, "SYNTH_both_SFs")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("POSITIVE CONTROL SUMMARY")
    print("=" * 70)
    print(f"\nOverall (full 486): A_insufficient={overall_a_insuff:.1%}")
    print(f"\n{'Group':<25} {'N':>5} {'A_insuff':>10} {'Av2_insuff':>12} {'Delta_A':>10}")
    print("-" * 65)
    for grp_name, gr in all_results.items():
        a_r = gr["a_insufficient_rate"]
        av2_r = gr["av2_insufficient_rate"]
        delta = (a_r - overall_a_insuff) if a_r is not None else None
        print(f"{grp_name:<25} {gr['n']:>5} "
              f"{f'{a_r:.1%}':>10} {f'{av2_r:.1%}' if av2_r is not None else 'N/A':>12} "
              f"{f'{delta:+.1%}' if delta is not None else 'N/A':>10}")

    print(f"\nInterpretation:")
    p1 = all_results.get("P1_correct", {})
    synth = all_results.get("SYNTH_both_SFs", {})
    a_p1 = p1.get("a_insufficient_rate")
    a_synth = synth.get("a_insufficient_rate")
    if a_p1 is not None:
        if a_p1 < 0.60:
            print(f"  P1: A shows meaningful sensitivity (insuff rate {a_p1:.1%} << {overall_a_insuff:.1%} overall)")
        elif a_p1 > 0.80:
            print(f"  P1: A has severe response bias (insuff rate {a_p1:.1%} ≈ {overall_a_insuff:.1%} overall)")
        else:
            print(f"  P1: A shows moderate sensitivity ({a_p1:.1%} vs {overall_a_insuff:.1%} overall)")
    if a_synth is not None:
        if a_synth > 0.70:
            print(f"  SYNTH: A still says INSUFFICIENT on {a_synth:.1%} of FULLY SUFFICIENT samples → condition is BROKEN")
        elif a_synth < 0.30:
            print(f"  SYNTH: A correctly says SUFFICIENT on most ({1-a_synth:.1%}) of fully sufficient samples → condition WORKS")
        else:
            print(f"  SYNTH: A shows partial sensitivity ({1-a_synth:.1%} sufficient rate on fully sufficient samples)")

    # ── Save ─────────────────────────────────────────────────────────────────
    summary = {
        "overall_a_insufficient_rate": overall_a_insuff,
        "groups": {k: {kk: vv for kk, vv in v.items() if kk != "per_sample"}
                   for k, v in all_results.items()},
        "per_sample": {k: v.get("per_sample", []) for k, v in all_results.items()},
    }
    out_path = output_dir / "positive_control_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
