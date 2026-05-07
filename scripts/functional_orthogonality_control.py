#!/usr/bin/env python3
"""
Functional Orthogonality Specificity Control
=============================================
Tests whether the zero effect of evidence-parallel steering is SPECIFIC to
the evidence direction, or trivially true for ANY direction orthogonal to
action_dir.

Design (margin-only, no agent loop):
  For N samples at the p0 decision point, measure margin shift under:
    1. No steering (baseline)
    2. Full action_dir (rho=-0.20, normalize_rms=1.0)
    3. Evidence-parallel (rho=-0.20, normalize_rms=1.0)
    4. K random directions (rho=-0.20, normalize_rms=1.0 each)

  All directions are RMS-normalized to 1.0, so same effective magnitude.

If evidence-parallel Δmargin ≈ random Δmargin ≈ 0:
  → Functional orthogonality is trivial (any ⊥ direction has no effect)
If some random directions have |Δmargin| >> evidence:
  → Evidence is specifically "sheltered" from action pathway

Usage:
  cd tmc/scripts/e2e_agent
  python scripts/functional_orthogonality_control.py
"""

import os, sys, json, argparse, random, time
import numpy as np
from pathlib import Path

import torch
from scipy.stats import mannwhitneyu, percentileofscore

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import SteeringHook, get_model_layers
from steering.directions import load_direction

LAYER = 20
RHO = -0.20


def get_margin(logits, tokenizer):
    log_probs = torch.log_softmax(logits, dim=-1)
    tool_ids = [tokenizer.encode(t, add_special_tokens=False)[0]
                for t in ACTION_TOKENS["tool_call"]]
    fin_ids = [tokenizer.encode(t, add_special_tokens=False)[0]
               for t in ACTION_TOKENS["finish"]]
    return (torch.logsumexp(log_probs[tool_ids], 0) -
            torch.logsumexp(log_probs[fin_ids], 0)).item()


def compute_margin(model, tokenizer, prompt, direction=None, rho=0.0,
                   layer=20):
    """Forward pass with optional steering, return margin."""
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    if direction is not None and abs(rho) > 1e-8:
        direction_rms = float(np.sqrt(np.mean(direction ** 2)))
        # Estimate hidden_rms from a quick forward
        hidden_rms = 0.65  # approximate; consistent with existing code
        alpha = rho * (hidden_rms / direction_rms)
        with SteeringHook(model, direction, alpha, layer=layer,
                          position=-1, max_interventions=1):
            with torch.no_grad():
                logits = model(input_ids).logits[0, -1, :]
    else:
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1, :]

    return get_margin(logits, tokenizer)


def normalize_rms(direction, target_rms=1.0):
    """Normalize direction to target RMS (same as load_direction normalize_rms)."""
    rms = float(np.sqrt(np.mean(direction ** 2)))
    if rms < 1e-12:
        return direction
    return direction * (target_rms / rms)


def generate_random_directions(dim, n, seed=42):
    """Generate n random unit-RMS directions."""
    rng = np.random.RandomState(seed)
    dirs = []
    for _ in range(n):
        d = rng.randn(dim).astype(np.float32)
        d = normalize_rms(d, target_rms=1.0)
        dirs.append(d)
    return dirs


def build_p0_prompt(tokenizer, question, query, observation):
    pb = PromptBuilder(tools=["search", "calculator"])
    steps = [{"action": "search", "action_input": query,
              "observation": observation[:1500]}]
    msgs = pb.build_full_prompt(question, steps)
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--n-random", type=int, default=50,
                    help="Number of random directions to test")
    ap.add_argument("--rho", type=float, default=-0.20)
    ap.add_argument("--baseline-trace",
                    default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--labels-path", default="results/phase1_probe/labels.jsonl")
    ap.add_argument("--output-dir",
                    default="results/functional_orthogonality_control")
    args = ap.parse_args()

    global RHO
    RHO = args.rho
    os.makedirs(args.output_dir, exist_ok=True)

    # Load directions (all normalized to RMS=1.0)
    full_dir, _ = load_direction(
        "steering/directions/direction_decomp_full_layer20.npz",
        normalize_rms=1.0)
    par_dir, _ = load_direction(
        "steering/directions/direction_decomp_parallel_layer20.npz",
        normalize_rms=1.0)
    perp_dir, _ = load_direction(
        "steering/directions/direction_decomp_perp_layer20.npz",
        normalize_rms=1.0)

    dim = full_dir.shape[0]
    random_dirs = generate_random_directions(dim, args.n_random)

    # Cosines with action_dir for reference
    act_dir, _ = load_direction(
        "steering/directions/direction_search_v3_layer20.npz",
        key="decision_direction_normalized")
    evi_dir_raw = np.load(
        "results/phase1_probe/probe_direction_l20.npz")["decision_direction"]
    evi_dir = evi_dir_raw / np.linalg.norm(evi_dir_raw)

    print("=== Direction cosines with action_dir ===")
    print(f"  full:     cos={np.dot(full_dir/np.linalg.norm(full_dir), act_dir):.4f}")
    print(f"  parallel: cos={np.dot(par_dir/np.linalg.norm(par_dir), act_dir):.4f}")
    print(f"  perp:     cos={np.dot(perp_dir/np.linalg.norm(perp_dir), act_dir):.4f}")
    rand_cos = [abs(np.dot(r/np.linalg.norm(r), act_dir)) for r in random_dirs]
    print(f"  random:   mean|cos|={np.mean(rand_cos):.4f} "
          f"max={np.max(rand_cos):.4f}")
    print(f"  evidence: cos={np.dot(evi_dir, act_dir):.4f}")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    # Load samples
    label_data = []
    with open(args.labels_path) as f:
        for line in f:
            label_data.append(json.loads(line))
    bl_map = {}
    with open(args.baseline_trace) as f:
        for line in f:
            ep = json.loads(line)
            bl_map[ep["sample_id"]] = ep

    # Build prompts
    prompts = []
    for ld in label_data:
        ep = bl_map.get(ld["sample_id"])
        if not ep or not ep.get("steps") or len(ep["steps"]) < 1:
            continue
        s0 = ep["steps"][0]
        if s0.get("action") != "search" or not s0.get("observation"):
            continue
        prompt = build_p0_prompt(tokenizer, ld["question"],
                                s0["action_input"], s0["observation"])
        prompts.append(prompt)
        if len(prompts) >= args.n_samples:
            break
    print(f"  {len(prompts)} prompts ready")

    # ── Measure margins ──
    conditions = {
        "baseline": None,
        "full": full_dir,
        "evidence_parallel": par_dir,
        "evidence_perp": perp_dir,
    }
    for i, rd in enumerate(random_dirs):
        conditions[f"random_{i:03d}"] = rd

    results = {name: [] for name in conditions}
    n_total = len(prompts)

    print(f"\nMeasuring margins for {len(conditions)} conditions × {n_total} samples...")
    t0 = time.time()

    for si, prompt in enumerate(prompts):
        for name, direction in conditions.items():
            m = compute_margin(model, tokenizer, prompt, direction=direction,
                               rho=RHO if direction is not None else 0.0,
                               layer=LAYER)
            results[name].append(m)

        if (si + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (si + 1) * (n_total - si - 1)
            print(f"  [{si+1}/{n_total}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    # ── Analysis ──
    baseline_margins = np.array(results["baseline"])
    full_shifts = np.array(results["full"]) - baseline_margins
    evi_par_shifts = np.array(results["evidence_parallel"]) - baseline_margins
    evi_perp_shifts = np.array(results["evidence_perp"]) - baseline_margins

    random_mean_shifts = []
    random_abs_mean_shifts = []
    for i in range(args.n_random):
        rs = np.array(results[f"random_{i:03d}"]) - baseline_margins
        random_mean_shifts.append(np.mean(rs))
        random_abs_mean_shifts.append(np.mean(np.abs(rs)))

    random_mean_shifts = np.array(random_mean_shifts)
    random_abs_mean_shifts = np.array(random_abs_mean_shifts)

    print(f"\n{'='*70}")
    print(f"  ★ FUNCTIONAL ORTHOGONALITY SPECIFICITY CONTROL")
    print(f"{'='*70}")
    print(f"  N={n_total}, rho={RHO}, L{LAYER}, normalize_rms=1.0")
    print(f"\n  A. Mean margin shift (Δmargin) by condition:")
    print(f"     Baseline mean margin:   {np.mean(baseline_margins):.3f}")
    print(f"     Full action_dir:        Δ={np.mean(full_shifts):+.3f} "
          f"(|Δ|={np.mean(np.abs(full_shifts)):.3f})")
    print(f"     Evidence-parallel:      Δ={np.mean(evi_par_shifts):+.3f} "
          f"(|Δ|={np.mean(np.abs(evi_par_shifts)):.3f})")
    print(f"     Evidence-perp:          Δ={np.mean(evi_perp_shifts):+.3f} "
          f"(|Δ|={np.mean(np.abs(evi_perp_shifts)):.3f})")
    print(f"     Random (N={args.n_random}):  "
          f"mean Δ={np.mean(random_mean_shifts):+.3f}±{np.std(random_mean_shifts):.3f}, "
          f"mean|Δ|={np.mean(random_abs_mean_shifts):.3f}±{np.std(random_abs_mean_shifts):.3f}")
    print(f"     Random range of mean Δ: [{np.min(random_mean_shifts):+.3f}, "
          f"{np.max(random_mean_shifts):+.3f}]")

    # Is evidence-parallel within random distribution?
    evi_par_mean = np.mean(evi_par_shifts)
    evi_par_abs_mean = np.mean(np.abs(evi_par_shifts))
    pctile_signed = percentileofscore(random_mean_shifts, evi_par_mean)
    pctile_abs = percentileofscore(random_abs_mean_shifts, evi_par_abs_mean)

    print(f"\n  B. Evidence-parallel vs random null:")
    print(f"     Evidence mean Δ = {evi_par_mean:+.3f}, "
          f"percentile in random = {pctile_signed:.1f}%")
    print(f"     Evidence mean|Δ| = {evi_par_abs_mean:.3f}, "
          f"percentile in random = {pctile_abs:.1f}%")

    # Full direction should be the outlier
    full_mean = np.mean(full_shifts)
    pctile_full = percentileofscore(random_mean_shifts, full_mean)
    print(f"     Full action mean Δ = {full_mean:+.3f}, "
          f"percentile in random = {pctile_full:.1f}%")

    print(f"\n  C. Interpretation:")
    if pctile_abs > 10 and pctile_abs < 90:
        print(f"     → Evidence-parallel |Δ| is WITHIN random distribution")
        print(f"     → Functional orthogonality is TRIVIAL:")
        print(f"       any direction ⊥ action_dir has near-zero effect")
        print(f"     → The decomposition result is explained by high-d geometry,")
        print(f"       not by evidence being specifically decoupled from action")
    else:
        print(f"     → Evidence-parallel |Δ| is OUTSIDE random distribution")
        if pctile_abs < 10:
            print(f"     → Evidence direction has LESS effect than random")
            print(f"     → Evidence is specifically sheltered from action pathway")
        else:
            print(f"     → Evidence direction has MORE effect than random")
    print(f"{'='*70}")

    # Save
    out = {
        "n_samples": n_total, "n_random": args.n_random,
        "rho": RHO, "layer": LAYER,
        "baseline_mean_margin": float(np.mean(baseline_margins)),
        "full_mean_shift": float(full_mean),
        "full_abs_mean_shift": float(np.mean(np.abs(full_shifts))),
        "evidence_parallel_mean_shift": float(evi_par_mean),
        "evidence_parallel_abs_mean_shift": float(evi_par_abs_mean),
        "evidence_perp_mean_shift": float(np.mean(evi_perp_shifts)),
        "random_mean_shifts_mean": float(np.mean(random_mean_shifts)),
        "random_mean_shifts_std": float(np.std(random_mean_shifts)),
        "random_abs_mean_shifts_mean": float(np.mean(random_abs_mean_shifts)),
        "random_abs_mean_shifts_std": float(np.std(random_abs_mean_shifts)),
        "evidence_percentile_signed": float(pctile_signed),
        "evidence_percentile_abs": float(pctile_abs),
        "full_percentile_signed": float(pctile_full),
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
