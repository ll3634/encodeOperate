#!/usr/bin/env python3
"""
OCFT — Stage 2: Injection sweeps for candidate decompositions.

For each OCFT candidate D_k that passed R1 + R2:
  - Loads the cached baseline + full + random per-example margin shifts
    from results/decomposition_ci_null/per_example_shifts.npz (the §3
    pipeline) so baseline / full / random are *bit-identical* to §3.
  - Runs two NEW conditions per candidate at the same N=100 prompts:
        parallel_k = (A · D̂_k) D̂_k  (RMS-normalised at injection)
        perp_k     = A - parallel_k  (RMS-normalised at injection)
  - Persists per_example_shifts_<DK>.npz with full / parallel_k / perp_k
    aligned to the same sample_ids order.

This script is the GPU-heavy step. The downstream analysis (bootstrap,
permutation, R3 evaluation) is in scripts/ocft_analyze.py and runs CPU-only
on the saved per-example shifts.

Operating point — IDENTICAL to §3:
  rho = -0.20, layer = 20, position = -1 (p0 last token), max_interventions = 1
  hidden_rms = 0.65, normalize_rms = 1.0  on every direction
"""

import os, sys, json, argparse, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import SteeringHook
from steering.directions import load_direction

LAYER = 20
RHO = -0.20
HIDDEN_RMS = 0.65


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached-shifts",
                    default="results/decomposition_ci_null/per_example_shifts.npz")
    ap.add_argument("--baseline-trace",
                    default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--labels-path",
                    default="results/phase1_probe/labels.jsonl")
    ap.add_argument("--probes-summary",
                    default="results/ocft/probes_summary.json")
    ap.add_argument("--out-dir", default="results/ocft")
    ap.add_argument("--steering-dir", default="steering/directions")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--candidates", nargs="+", default=None,
                    help="Subset of candidate names. Default = all in summary.")
    ap.add_argument("--auroc-min", type=float, default=0.75)
    ap.add_argument("--cos-max",   type=float, default=0.10)
    return ap.parse_args()


def build_p0_prompt(tokenizer, question, query, observation):
    pb = PromptBuilder(tools=["search", "calculator"])
    steps = [{"action": "search", "action_input": query,
              "observation": observation[:1500]}]
    msgs = pb.build_full_prompt(question, steps)
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)


def margin_from_logits(logits, tool_ids, fin_ids):
    import torch
    lp = torch.log_softmax(logits, dim=-1)
    return (torch.logsumexp(lp[tool_ids], 0) -
            torch.logsumexp(lp[fin_ids], 0)).item()


def compute_margin(model, tok, prompt, direction, rho, layer, tool_ids, fin_ids):
    import torch
    device = next(model.parameters()).device
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    if direction is not None and abs(rho) > 1e-8:
        d_rms = float(np.sqrt(np.mean(direction ** 2)))
        alpha = rho * (HIDDEN_RMS / d_rms)
        with SteeringHook(model, direction, alpha, layer=layer,
                          position=-1, max_interventions=1):
            with torch.no_grad():
                logits = model(ids).logits[0, -1, :]
    else:
        with torch.no_grad():
            logits = model(ids).logits[0, -1, :]
    return margin_from_logits(logits, tool_ids, fin_ids)


def reconstruct_prompts(sample_ids, baseline_trace, labels_path, tok):
    label_data = {json.loads(l)["sample_id"]: json.loads(l)
                  for l in open(labels_path)}
    bl_map = {}
    with open(baseline_trace) as f:
        for line in f:
            ep = json.loads(line)
            bl_map[ep["sample_id"]] = ep
    prompts = []
    for sid in sample_ids:
        ld = label_data.get(sid); ep = bl_map.get(sid)
        if ld is None or ep is None:
            raise RuntimeError(f"sample_id {sid} missing in labels/trace")
        s0 = ep["steps"][0]
        prompts.append(build_p0_prompt(
            tok, ld["question"], s0["action_input"], s0["observation"]))
    return prompts


def select_candidates(summary: dict, names, auroc_min: float, cos_max: float):
    """Apply R1 (AUROC>=auroc_min) and R2 (|cos|<=cos_max). Return list of dicts."""
    cands = summary["candidates"]
    chosen, dropped = [], []
    target = list(cands.keys()) if names is None else names
    for k in target:
        c = cands[k]
        ok_r1 = c["auroc"] >= auroc_min
        ok_r2 = abs(c["cos_with_action"]) <= cos_max
        c_out = dict(c, name=k, passed_R1=ok_r1, passed_R2=ok_r2)
        (chosen if ok_r1 and ok_r2 else dropped).append(c_out)
    return chosen, dropped


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary = json.load(open(args.probes_summary))
    chosen, dropped = select_candidates(
        summary, args.candidates, args.auroc_min, args.cos_max)

    print("=" * 72)
    print("  OCFT — STAGE 2: injection sweeps for candidates passing R1+R2")
    print("=" * 72)
    print(f"  R1 AUROC>={args.auroc_min}  R2 |cos|<={args.cos_max}")
    print(f"  passed:  {[c['name'] for c in chosen]}")
    print(f"  dropped: {[(c['name'], round(c['auroc'],3), round(c['cos_with_action'],3)) for c in dropped]}")

    cached = np.load(args.cached_shifts, allow_pickle=True)
    sample_ids = list(cached["sample_ids"])
    baseline = cached["baseline"]; full = cached["full"]
    n = len(sample_ids)
    print(f"  cached N={n}  baseline.mean={baseline.mean():+.3f}  full.mean={full.mean():+.3f}")

    if not chosen:
        print("No candidates pass R1+R2. Writing skip marker.")
        with open(out_dir / "injection_skipped.json", "w") as f:
            json.dump({"chosen": [], "dropped": dropped}, f, indent=2)
        return

    # ── Load model + tokenizer ──────────────────────────────────────────────
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\n[load] {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    tool_ids = [tok.encode(t, add_special_tokens=False)[0]
                for t in ACTION_TOKENS["tool_call"]]
    fin_ids = [tok.encode(t, add_special_tokens=False)[0]
               for t in ACTION_TOKENS["finish"]]

    print(f"\n[prompts] reconstructing N={n} prompts …")
    prompts = reconstruct_prompts(sample_ids, args.baseline_trace,
                                  args.labels_path, tok)

    for c in chosen:
        name = c["name"]
        par_path = c["files"]["parallel"]; perp_path = c["files"]["perp"]
        d_par,  _ = load_direction(par_path,  normalize_rms=1.0)
        d_perp, _ = load_direction(perp_path, normalize_rms=1.0)
        print(f"\n=== Injecting candidate {name} ===")
        print(f"   parallel_RMS={np.sqrt(np.mean(d_par**2)):.3f}"
              f"  perp_RMS={np.sqrt(np.mean(d_perp**2)):.3f}")
        par_m = np.zeros(n, dtype=np.float32)
        perp_m = np.zeros(n, dtype=np.float32)
        t0 = time.time()
        for i, p in enumerate(prompts):
            par_m[i]  = compute_margin(model, tok, p, d_par,  RHO, LAYER, tool_ids, fin_ids)
            perp_m[i] = compute_margin(model, tok, p, d_perp, RHO, LAYER, tool_ids, fin_ids)
            if (i+1) % 10 == 0 or i == 0:
                eta = (time.time()-t0)/(i+1) * (n-i-1)
                print(f"   [{i+1}/{n}] elapsed={time.time()-t0:.0f}s  ETA={eta:.0f}s")
        par_sh  = par_m  - baseline
        perp_sh = perp_m - baseline
        out = out_dir / f"per_example_shifts_{name}.npz"
        np.savez(out, baseline=baseline, full=full,
                 parallel=par_sh, perp=perp_sh,
                 sample_ids=np.array(sample_ids))
        print(f"   saved {out}")
        print(f"   par.mean={par_sh.mean():+.3f}  perp.mean={perp_sh.mean():+.3f}"
              f"  full.mean={full.mean():+.3f}")

    with open(out_dir / "injection_chosen.json", "w") as f:
        json.dump({"chosen": chosen, "dropped": dropped,
                   "auroc_min": args.auroc_min, "cos_max": args.cos_max},
                  f, indent=2)


if __name__ == "__main__":
    main()
