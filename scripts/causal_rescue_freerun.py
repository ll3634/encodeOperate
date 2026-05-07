#!/usr/bin/env python3
"""Free-run causal rescue test at p0.

For each minimal-corruption pair, run a full mini-episode:
  1. Build p0 prompt (question + 1st search obs)
  2. Generate freely (with/without component patches during prefill)
  3. Parse: if model outputs "search" → execute via SearchTool → generate step 2 → extract answer
           if model outputs "Final Answer" → extract answer directly
  4. Score with exact match

Conditions:
  baseline_clean, baseline_corrupt,
  suff_top7, suff_circuit  (corrupt + clean patches),
  nec_top7, nec_circuit    (clean + corrupt patches)

Metrics: 2ndSR, EM accuracy, rescue, regression, causal purity.
"""

import json, sys, argparse, time
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import (PromptBuilder, ACTION_TOKENS,
                           DEFAULT_SYSTEM_PROMPT, parse_action)
from steering.hook_utils import get_model_layers
from tools.search_tool import SearchTool
from eval.scorers import exact_match

# ── Component sets ────────────────────────────────────────────────────────
TOP7 = [
    ('attn', 22), ('attn', 18), ('mlp', 21), ('attn', 19),
    ('mlp', 18), ('mlp', 20), ('attn', 23),
]
CIRCUIT = [('attn', 18), ('mlp', 20)]
PATCH_CONFIGS = {"top7": TOP7, "circuit": CIRCUIT}


# ── Cache component outputs at last token ─────────────────────────────────
def cache_components(model, layers, input_ids, n_layers):
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
        model(input_ids)
    for h in handles:
        h.remove()
    return cache


# ── Patched generation (prefill-only patches) ────────────────────────────
def patched_generate(model, tokenizer, layers, input_ids, source_cache, specs,
                     max_new_tokens=150):
    handles = []
    fired = [False]
    for comp, l in specs:
        vec = source_cache[(comp, l)]
        target = layers[l].self_attn if comp == 'attn' else layers[l].mlp
        def make_hook(v):
            def hook_fn(module, inp, out):
                if fired[0]:
                    return
                h = out[0] if isinstance(out, tuple) else out
                h[0, -1, :] = v.to(h.dtype)
                if isinstance(out, tuple):
                    return (h,) + out[1:]
                return h
            return hook_fn
        handles.append(target.register_forward_hook(make_hook(vec)))
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


def plain_generate(model, tokenizer, input_ids, max_new_tokens=150):
    with torch.no_grad():
        out_ids = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out_ids[0][input_ids.shape[1]:], skip_special_tokens=True)


def build_p0_prompt(tokenizer, question, query, observation):
    pb = PromptBuilder(tools=["search", "calculator"])
    steps = [{"action": "search", "action_input": query,
              "observation": observation[:1500]}]
    messages = pb.build_full_prompt(question, steps)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


def build_step2_prompt(tokenizer, question, query1, obs1, query2, obs2):
    """Build prompt after two searches."""
    pb = PromptBuilder(tools=["search", "calculator"])
    steps = [
        {"action": "search", "action_input": query1, "observation": obs1[:1500]},
        {"action": "search", "action_input": query2, "observation": obs2[:1500]},
    ]
    messages = pb.build_full_prompt(question, steps)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


def run_episode(text, model, tokenizer, search_tool, question, query1, obs1, device):
    """Parse step-1 output. If search, execute and do step 2. Return (answer, did_search, query2)."""
    parsed = parse_action(text)

    if parsed["final_answer"] is not None:
        return parsed["final_answer"], False, None

    if parsed["action"] and "search" in parsed["action"].lower() and parsed["action_input"]:
        query2 = parsed["action_input"]
        obs2 = search_tool(query2)
        prompt2 = build_step2_prompt(tokenizer, question, query1, obs1, query2, obs2)
        ids2 = tokenizer.encode(prompt2, return_tensors="pt").to(device)
        text2 = plain_generate(model, tokenizer, ids2, max_new_tokens=150)
        parsed2 = parse_action(text2)
        answer = parsed2["final_answer"] or ""
        return answer, True, query2

    # Fallback: try to extract any answer-like content
    return text.strip()[:200], False, None


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/minimal_corruption/full_corruption_data.jsonl")
    parser.add_argument("--corpus", default="data/hotpotqa/corpus.jsonl")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--output-dir", default="results/causal_rescue_freerun")
    args = parser.parse_args()

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

    # Load search tool
    print(f"Loading corpus from {args.corpus}...")
    search_tool = SearchTool(args.corpus, top_k=3)

    # ── Conditions to run ─────────────────────────────────────────────────
    # Each condition: (label, input_obs_key, patch_source_key, specs_or_none)
    #   suff = corrupt input + clean patches; nec = clean input + corrupt patches
    CONDITIONS = [
        ("baseline_clean",  "clean",   None,      None),
        ("baseline_corrupt","corrupt", None,      None),
        ("suff_top7",       "corrupt", "clean",   TOP7),
        ("suff_circuit",    "corrupt", "clean",   CIRCUIT),
        ("nec_top7",        "clean",   "corrupt", TOP7),
        ("nec_circuit",     "clean",   "corrupt", CIRCUIT),
    ]

    per_sample = []
    t0 = time.time()

    for si, s in enumerate(samples):
        q = s["question"]
        query1 = s["step0_query"]
        obs = {"clean": s["obs_clean"], "corrupt": s["obs_corrupt_A"]}
        gold = s["answer"]

        # Build prompts and cache for both clean and corrupt
        prompts, ids, caches = {}, {}, {}
        for key in ("clean", "corrupt"):
            prompts[key] = build_p0_prompt(tokenizer, q, query1, obs[key])
            ids[key] = tokenizer.encode(prompts[key], return_tensors="pt").to(device)
            caches[key] = cache_components(model, layers, ids[key], n_layers)

        rec = {"idx": si, "sample_id": s["sample_id"], "question": q[:100],
               "gold": gold}

        for label, inp_key, patch_key, specs in CONDITIONS:
            input_ids = ids[inp_key]
            obs_used = obs[inp_key]

            if specs is not None and patch_key is not None:
                text = patched_generate(model, tokenizer, layers,
                                        input_ids, caches[patch_key], specs)
            else:
                text = plain_generate(model, tokenizer, input_ids)

            answer, did_search, query2 = run_episode(
                text, model, tokenizer, search_tool, q, query1, obs_used, device)

            em = exact_match(answer, gold) if answer else False
            rec[f"{label}_text"] = text[:300]
            rec[f"{label}_answer"] = answer[:200] if answer else ""
            rec[f"{label}_search"] = did_search
            rec[f"{label}_em"] = em
            if did_search and query2:
                rec[f"{label}_query2"] = query2

        per_sample.append(rec)

        if (si + 1) % 5 == 0:
            elapsed = time.time() - t0
            # Quick progress stats
            bl_sr = sum(1 for r in per_sample if r.get("baseline_clean_search")) / len(per_sample)
            st_sr = sum(1 for r in per_sample if r.get("suff_top7_search")) / len(per_sample)
            print(f"  [{si+1}/{len(samples)}] {elapsed:.0f}s  "
                  f"bl_clean_2ndSR={bl_sr:.0%}  suff_top7_2ndSR={st_sr:.0%}")

    # ── Save per-sample ───────────────────────────────────────────────────
    with open(out_dir / "per_sample.jsonl", "w") as f:
        for r in per_sample:
            f.write(json.dumps(r) + "\n")

    # ── Report ────────────────────────────────────────────────────────────
    N = len(per_sample)
    print(f"\n{'='*70}")
    print(f"FREE-RUN CAUSAL RESCUE RESULTS (N={N})")
    print(f"{'='*70}")

    cond_labels = [c[0] for c in CONDITIONS]

    # 2nd Search Rate
    print(f"\n--- 2nd Search Rate ---")
    for label in cond_labels:
        sr = sum(1 for r in per_sample if r.get(f"{label}_search")) / N
        print(f"  {label:20s}: {sr:.1%} ({sum(1 for r in per_sample if r.get(f'{label}_search'))}/{N})")

    # EM Accuracy
    print(f"\n--- Exact Match Accuracy ---")
    for label in cond_labels:
        acc = sum(1 for r in per_sample if r.get(f"{label}_em")) / N
        print(f"  {label:20s}: {acc:.1%} ({sum(1 for r in per_sample if r.get(f'{label}_em'))}/{N})")

    # Rescue / Regression (vs baseline_corrupt for suff, vs baseline_clean for nec)
    print(f"\n--- Rescue / Regression ---")
    for cfg_name in ("top7", "circuit"):
        skey = f"suff_{cfg_name}"
        rescued = sum(1 for r in per_sample
                      if not r.get("baseline_corrupt_em") and r.get(f"{skey}_em"))
        regressed = sum(1 for r in per_sample
                        if r.get("baseline_corrupt_em") and not r.get(f"{skey}_em"))
        # Purity: of rescued, how many via search?
        rescued_via_search = sum(1 for r in per_sample
                                 if not r.get("baseline_corrupt_em") and r.get(f"{skey}_em")
                                 and r.get(f"{skey}_search"))
        purity = rescued_via_search / rescued if rescued else 0
        print(f"  {skey}: rescued={rescued}, regressed={regressed}, "
              f"net={rescued-regressed}, purity={purity:.0%} ({rescued_via_search}/{rescued})")

        nkey = f"nec_{cfg_name}"
        broken = sum(1 for r in per_sample
                     if r.get("baseline_clean_em") and not r.get(f"{nkey}_em"))
        fixed = sum(1 for r in per_sample
                    if not r.get("baseline_clean_em") and r.get(f"{nkey}_em"))
        print(f"  {nkey}: broken={broken}, spurious_fix={fixed}, net_damage={broken-fixed}")

    # Search-mediated rescue detail
    print(f"\n--- Rescue Detail (suff_top7) ---")
    for r in per_sample:
        if not r.get("baseline_corrupt_em") and r.get("suff_top7_em"):
            via = "search→answer" if r.get("suff_top7_search") else "direct_answer"
            print(f"  sample {r['idx']}: {via}  q={r['question'][:60]}  "
                  f"gold={r['gold']}  pred={r.get('suff_top7_answer','')[:60]}")

    # Summary JSON
    summary = {
        "n_samples": N,
        "conditions": {},
    }
    for label in cond_labels:
        sr = sum(1 for r in per_sample if r.get(f"{label}_search"))
        em = sum(1 for r in per_sample if r.get(f"{label}_em"))
        summary["conditions"][label] = {"search_rate": sr/N, "em_accuracy": em/N,
                                         "n_search": sr, "n_correct": em}
    for cfg_name in ("top7", "circuit"):
        skey = f"suff_{cfg_name}"
        rescued = sum(1 for r in per_sample
                      if not r.get("baseline_corrupt_em") and r.get(f"{skey}_em"))
        regressed = sum(1 for r in per_sample
                        if r.get("baseline_corrupt_em") and not r.get(f"{skey}_em"))
        rescued_via_search = sum(1 for r in per_sample
                                 if not r.get("baseline_corrupt_em") and r.get(f"{skey}_em")
                                 and r.get(f"{skey}_search"))
        summary[f"{skey}_rescue"] = rescued
        summary[f"{skey}_regression"] = regressed
        summary[f"{skey}_purity"] = rescued_via_search / rescued if rescued else 0

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_dir}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

