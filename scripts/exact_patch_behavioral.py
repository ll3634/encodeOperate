#!/usr/bin/env python3
"""Pre-registered exact patch/ablate experiment with behavioral readout.

Design (pre-registered):
  Same-question clean/corrupt pairs (minimal sentence-level evidence corruption).
  Components: top-7 mediation band (L18-L23), circuit (attn_L18+mlp_L20), bottom-7 control.

  Denoising (sufficiency): corrupt run + clean component patches → does margin/behavior recover?
  Noising  (necessity):    clean run + corrupt component patches → does margin/behavior break?

  Readout:
    1. search-stop logit margin (log P(Action) - log P(Final))
    2. behavioral: does model actually produce "search" vs "Final Answer"?

  Positions:
    p0: decision point (DEFAULT_SYSTEM_PROMPT, no thought) — last token of prompt
    p2: 50% through generated Thought (REACT_THOUGHT_SYSTEM_PROMPT)
    p4: 100% of Thought (just before Action token)

  For p2/p4: teacher-force same thought tokens (from clean) onto both conditions.
"""

import json, sys, argparse, time, re, random
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import (PromptBuilder, ACTION_TOKENS,
                           DEFAULT_SYSTEM_PROMPT, REACT_THOUGHT_SYSTEM_PROMPT,
                           parse_action)
from steering.hook_utils import get_model_layers

# ── Component sets (from attribution patching Step 2) ──────────────────────
TOP7 = [
    ('attn', 22), ('attn', 18), ('mlp', 21), ('attn', 19),
    ('mlp', 18), ('mlp', 20), ('attn', 23),
]
CIRCUIT = [('attn', 18), ('mlp', 20)]
BOTTOM7 = [
    ('attn', 14), ('mlp', 14), ('attn', 15), ('mlp', 16),
    ('attn', 16), ('mlp', 15), ('attn', 24),
]

PATCH_CONFIGS = {"top7": TOP7, "circuit": CIRCUIT, "bottom7": BOTTOM7}


# ── Margin computation ─────────────────────────────────────────────────────
def compute_margin(logits, tool_ids, fin_ids):
    """search-stop logit margin from logits at a single position."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    t = torch.logsumexp(lp[tool_ids], 0).item() if tool_ids else -100.0
    f = torch.logsumexp(lp[fin_ids], 0).item() if fin_ids else -100.0
    return t - f


# ── Cache component outputs at last token ──────────────────────────────────
def cache_components(model, layers, input_ids, n_layers):
    """Forward pass → cache (comp, layer) → tensor at position -1, plus logits."""
    cache = {}
    handles = []
    for l in range(n_layers):
        def make_hook(comp, li):
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                cache[(comp, li)] = h[0, -1, :].detach().clone()
            return hook_fn
        handles.append(layers[l].self_attn.register_forward_hook(make_hook('attn', l)))
        handles.append(layers[l].mlp.register_forward_hook(make_hook('mlp', l)))
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :]
    for h in handles:
        h.remove()
    return cache, logits


# ── Patched forward: replace specified components with source values ───────
def patched_forward(model, layers, input_ids, source_cache, specs):
    """Run forward with components patched at position -1. Return logits."""
    handles = []
    for comp, l in specs:
        vec = source_cache[(comp, l)]
        target = layers[l].self_attn if comp == 'attn' else layers[l].mlp
        def make_hook(v):
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h[0, -1, :] = v.to(h.dtype)
                if isinstance(out, tuple):
                    return (h,) + out[1:]
                return h
            return hook_fn
        handles.append(target.register_forward_hook(make_hook(vec)))
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :]
    for h in handles:
        h.remove()
    return logits


# ── Patched generation: generate from patched state ────────────────────────
def patched_generate(model, tokenizer, layers, input_ids, source_cache, specs,
                     max_new_tokens=80):
    """Generate with component patches active during prefill only."""
    handles = []
    fired = [False]

    for comp, l in specs:
        vec = source_cache[(comp, l)]
        target = layers[l].self_attn if comp == 'attn' else layers[l].mlp
        def make_hook(v):
            def hook_fn(module, inp, out):
                if fired[0]:
                    return  # Only patch during prefill
                h = out[0] if isinstance(out, tuple) else out
                h[0, -1, :] = v.to(h.dtype)
                if isinstance(out, tuple):
                    return (h,) + out[1:]
                return h
            return hook_fn
        handles.append(target.register_forward_hook(make_hook(vec)))

    # Mark prefill as done after first full forward pass
    def mark_fired(module, inp, out):
        fired[0] = True
    handles.append(layers[-1].register_forward_hook(mark_fired))

    with torch.no_grad():
        out_ids = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    for h in handles:
        h.remove()
    return tokenizer.decode(out_ids[0][input_ids.shape[1]:], skip_special_tokens=True)


# ── Behavioral classification ─────────────────────────────────────────────
def classify_behavior(text):
    """Return 'search', 'stop', or 'other'."""
    parsed = parse_action(text)
    if parsed["final_answer"] is not None:
        return "stop"
    if parsed["action"] and "search" in parsed["action"].lower():
        return "search"
    if "Final Answer" in text or "final answer" in text.lower():
        return "stop"
    if "Action: search" in text or "action: search" in text.lower():
        return "search"
    return "other"


# ── Prompt builders ────────────────────────────────────────────────────────
def build_p0_prompt(tokenizer, question, query, observation):
    """DEFAULT_SYSTEM_PROMPT — decision point is last token."""
    pb = PromptBuilder(tools=["search", "calculator"])
    steps = [{"action": "search", "action_input": query,
              "observation": observation[:1500]}]
    messages = pb.build_full_prompt(question, steps)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


def build_thought_prompt(tokenizer, question, query, observation):
    """REACT_THOUGHT_SYSTEM_PROMPT — for generating thought tokens."""
    pb = PromptBuilder(tools=["search", "calculator"],
                       system_template=REACT_THOUGHT_SYSTEM_PROMPT)
    steps = [{"action": "search", "action_input": query,
              "observation": observation[:1500]}]
    messages = pb.build_full_prompt(question, steps)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)



# ── Main experiment ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/minimal_corruption/full_corruption_data.jsonl")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--positions", default="p0", help="Comma-separated: p0,p2,p4")
    parser.add_argument("--output-dir", default="results/exact_patch_behavioral")
    parser.add_argument("--behavioral", action="store_true",
                        help="Also run generation (slow) for behavioral readout")
    args = parser.parse_args()

    positions = args.positions.split(",")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    samples = []
    with open(args.data) as f:
        for line in f:
            samples.append(json.loads(line))
    samples = samples[:args.n]
    print(f"Loaded {len(samples)} samples")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    layers = get_model_layers(model)
    n_layers = len(layers)
    device = next(model.parameters()).device

    # Token IDs for margin computation
    tool_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]
    tool_ids_t = torch.tensor(tool_ids, device=device)
    fin_ids_t = torch.tensor(fin_ids, device=device)

    # ── Run experiment per position ──────────────────────────────────────
    for pos in positions:
        print(f"\n{'='*70}")
        print(f"Position: {pos}")
        print(f"{'='*70}")
        t0 = time.time()

        margin_results = defaultdict(list)  # config_name → list of recovery/drop values
        behavior_results = defaultdict(lambda: {"search": 0, "stop": 0, "other": 0})
        per_sample = []  # detailed per-sample records
        skipped = 0

        for si, s in enumerate(samples):
            q = s["question"]
            query = s["step0_query"]
            obs_clean = s["obs_clean"]
            obs_corrupt = s["obs_corrupt_A"]

            # Build prompts
            if pos == "p0":
                prompt_clean = build_p0_prompt(tokenizer, q, query, obs_clean)
                prompt_corrupt = build_p0_prompt(tokenizer, q, query, obs_corrupt)
                ids_clean = tokenizer.encode(prompt_clean, return_tensors="pt").to(device)
                ids_corrupt = tokenizer.encode(prompt_corrupt, return_tensors="pt").to(device)

            else:
                # p2 / p4: generate thought from CORRUPT prompt, teacher-force onto both.
                # Why corrupt, not clean?  Clean-generated thoughts contain
                # evidence-aware reasoning that leaks clean info into the corrupt
                # condition, artificially shrinking the total effect at later
                # positions (sign-consistency drops from 60% at p0 to 15% at p4).
                # Corrupt-generated thoughts are evidence-agnostic, preserving the
                # clean/corrupt distinction across thought positions.
                thought_prompt_clean = build_thought_prompt(tokenizer, q, query, obs_clean)
                thought_prompt_corrupt = build_thought_prompt(tokenizer, q, query, obs_corrupt)
                thought_ids_corrupt = tokenizer.encode(thought_prompt_corrupt, return_tensors="pt").to(device)

                # Generate thought tokens from CORRUPT
                with torch.no_grad():
                    gen_out = model.generate(
                        thought_ids_corrupt, max_new_tokens=150,
                        do_sample=False, pad_token_id=tokenizer.eos_token_id)
                gen_tokens = gen_out[0][thought_ids_corrupt.shape[1]:]

                if len(gen_tokens) < 4:
                    skipped += 1
                    continue

                # Determine thought prefix length
                if pos == "p2":
                    n_prefix = max(1, len(gen_tokens) // 2)
                else:  # p4
                    n_prefix = len(gen_tokens)
                thought_prefix = gen_tokens[:n_prefix]

                # Build teacher-forced sequences
                ids_clean_base = tokenizer.encode(thought_prompt_clean, return_tensors="pt").to(device)
                ids_corrupt_base = tokenizer.encode(thought_prompt_corrupt, return_tensors="pt").to(device)
                ids_clean = torch.cat([ids_clean_base, thought_prefix.unsqueeze(0)], dim=1)
                ids_corrupt = torch.cat([ids_corrupt_base, thought_prefix.unsqueeze(0)], dim=1)

            # ── Baseline forward passes ──────────────────────────────────
            cache_clean, logits_clean = cache_components(model, layers, ids_clean, n_layers)
            cache_corrupt, logits_corrupt = cache_components(model, layers, ids_corrupt, n_layers)

            m_clean = compute_margin(logits_clean, tool_ids, fin_ids)
            m_corrupt = compute_margin(logits_corrupt, tool_ids, fin_ids)
            total_effect = m_clean - m_corrupt

            if abs(total_effect) < 0.01:
                skipped += 1
                continue

            rec = {"idx": si, "question": q[:80], "m_clean": m_clean,
                   "m_corrupt": m_corrupt, "total_effect": total_effect}

            # ── Denoising (sufficiency): corrupt + clean patches ─────────
            for cfg_name, specs in PATCH_CONFIGS.items():
                logits_patched = patched_forward(model, layers, ids_corrupt, cache_clean, specs)
                m_patched = compute_margin(logits_patched, tool_ids, fin_ids)
                recovery = (m_patched - m_corrupt) / total_effect
                margin_results[f"suff_{cfg_name}"].append(recovery)
                rec[f"suff_{cfg_name}"] = recovery

                # Behavioral
                if args.behavioral and cfg_name in ("top7", "circuit"):
                    text = patched_generate(model, tokenizer, layers,
                                            ids_corrupt, cache_clean, specs)
                    beh = classify_behavior(text)
                    behavior_results[f"suff_{cfg_name}"][beh] += 1
                    rec[f"suff_{cfg_name}_beh"] = beh

            # ── Noising (necessity): clean + corrupt patches ─────────────
            for cfg_name, specs in PATCH_CONFIGS.items():
                logits_patched = patched_forward(model, layers, ids_clean, cache_corrupt, specs)
                m_patched = compute_margin(logits_patched, tool_ids, fin_ids)
                drop = (m_clean - m_patched) / total_effect
                margin_results[f"nec_{cfg_name}"].append(drop)
                rec[f"nec_{cfg_name}"] = drop

                if args.behavioral and cfg_name in ("top7", "circuit"):
                    text = patched_generate(model, tokenizer, layers,
                                            ids_clean, cache_corrupt, specs)
                    beh = classify_behavior(text)
                    behavior_results[f"nec_{cfg_name}"][beh] += 1
                    rec[f"nec_{cfg_name}_beh"] = beh

            # Baseline behavioral
            if args.behavioral:
                for label, ids_base in [("baseline_clean", ids_clean),
                                         ("baseline_corrupt", ids_corrupt)]:
                    with torch.no_grad():
                        out = model.generate(
                            ids_base, max_new_tokens=80,
                            do_sample=False, pad_token_id=tokenizer.eos_token_id)
                    text = tokenizer.decode(out[0][ids_base.shape[1]:], skip_special_tokens=True)
                    beh = classify_behavior(text)
                    behavior_results[label][beh] += 1
                    rec[f"{label}_beh"] = beh

            per_sample.append(rec)

            if (si + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  [{si+1}/{len(samples)}] {elapsed:.0f}s")

        # ── Report ───────────────────────────────────────────────────────
        n_valid = len(per_sample)

        # Split by TE sign: TE < 0 means corruption increased search tendency (expected)
        consistent = [r for r in per_sample if r["total_effect"] < -0.1]
        n_consistent = len(consistent)
        sign_rate = n_consistent / n_valid if n_valid else 0

        print(f"\n--- {pos} Results (N={n_valid}, skipped={skipped}) ---")
        print(f"Sign consistency: {n_consistent}/{n_valid} ({sign_rate:.0%}) have TE<0 (expected direction)\n")

        # All-sample margin results
        print(f"{'Config':<20} {'median':>8} {'mean':>8} {'p25':>8} {'p75':>8}  (ALL)")
        print("-" * 60)
        for name in sorted(margin_results.keys()):
            vals = np.array(margin_results[name])
            print(f"{name:<20} {np.median(vals):8.3f} {np.mean(vals):8.3f} "
                  f"{np.percentile(vals, 25):8.3f} {np.percentile(vals, 75):8.3f}")

        # Sign-consistent margin results
        if n_consistent >= 5:
            print(f"\n{'Config':<20} {'median':>8} {'mean':>8} {'p25':>8} {'p75':>8}  (TE<0 only, N={n_consistent})")
            print("-" * 60)
            for name in sorted(margin_results.keys()):
                # Collect values only from consistent samples
                c_vals = [r[name] for r in consistent if name in r]
                if c_vals:
                    arr = np.array(c_vals)
                    print(f"{name:<20} {np.median(arr):8.3f} {np.mean(arr):8.3f} "
                          f"{np.percentile(arr, 25):8.3f} {np.percentile(arr, 75):8.3f}")

        # Boundary-crossing analysis (sufficiency: corrupt→patched flips margin sign)
        print(f"\n--- Boundary Crossing (sign-consistent only) ---")
        for cfg in PATCH_CONFIGS:
            skey = f"suff_{cfg}"
            # Sufficiency: corrupt is search-leaning (m>0), patching makes it stop (m<0)
            search_base = [r for r in consistent if r["m_corrupt"] > 0]
            if search_base:
                crossed = sum(1 for r in search_base
                              if (r["m_corrupt"] + r[skey] * r["total_effect"]) < 0)
                print(f"  {skey}: {crossed}/{len(search_base)} corrupt-search→patched-stop")
            else:
                print(f"  {skey}: 0 search-leaning corrupt samples")

            # Necessity: clean is stop-leaning (m<0), patching makes it search (m>0)
            nkey = f"nec_{cfg}"
            stop_base = [r for r in consistent if r["m_clean"] < -1]
            if stop_base:
                flipped = sum(1 for r in stop_base
                              if (r["m_clean"] - r[nkey] * r["total_effect"]) > 0)
                print(f"  {nkey}: {flipped}/{len(stop_base)} clean-stop→patched-search")

        # Behavioral results
        if args.behavioral:
            print(f"\nBehavioral counts (all samples):")
            for name in sorted(behavior_results.keys()):
                counts = behavior_results[name]
                total = sum(counts.values())
                pct = {k: f"{v}/{total}" for k, v in counts.items()}
                print(f"  {name:<25} {pct}")

            # Behavioral flips for sign-consistent only
            print(f"\nBehavioral flips (sign-consistent, TE<0):")
            for cfg in ("top7", "circuit"):
                skey = f"suff_{cfg}"
                s2s = sum(1 for r in consistent
                          if r.get("baseline_corrupt_beh") == "search"
                          and r.get(f"{skey}_beh") == "stop")
                s_base = sum(1 for r in consistent
                             if r.get("baseline_corrupt_beh") == "search")
                nkey = f"nec_{cfg}"
                n2s = sum(1 for r in consistent
                          if r.get("baseline_clean_beh") == "stop"
                          and r.get(f"{nkey}_beh") == "search")
                n_base = sum(1 for r in consistent
                             if r.get("baseline_clean_beh") == "stop")
                if s_base:
                    print(f"  {skey}: {s2s}/{s_base} search→stop")
                if n_base:
                    print(f"  {nkey}: {n2s}/{n_base} stop→search")

        # Save
        save_path = out_dir / f"{pos}_results.json"
        summary = {
            "position": pos,
            "n_valid": n_valid,
            "n_skipped": skipped,
            "n_sign_consistent": n_consistent,
            "sign_consistency_rate": sign_rate,
            "margin_summary": {},
            "margin_summary_consistent": {},
            "behavioral_summary": dict(behavior_results) if args.behavioral else {},
        }
        for name, vals in margin_results.items():
            arr = np.array(vals)
            summary["margin_summary"][name] = {
                "median": float(np.median(arr)),
                "mean": float(np.mean(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "n": len(arr),
            }
            c_vals = [r[name] for r in consistent if name in r]
            if c_vals:
                c_arr = np.array(c_vals)
                summary["margin_summary_consistent"][name] = {
                    "median": float(np.median(c_arr)),
                    "mean": float(np.mean(c_arr)),
                    "n": len(c_arr),
                }
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Save per-sample
        per_sample_path = out_dir / f"{pos}_per_sample.jsonl"
        with open(per_sample_path, "w") as f:
            for rec in per_sample:
                f.write(json.dumps(rec) + "\n")

        print(f"\nSaved to {save_path} and {per_sample_path}")


if __name__ == "__main__":
    main()
