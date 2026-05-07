#!/usr/bin/env python3
"""
Decomposition Bootstrap CIs + Expanded Random Null + Pairwise Permutation Tests
================================================================================
Adds statistical rigor to the §8.3 functional decomposition (results.json in
results/functional_orthogonality_control/) by re-running the same measurement
with per-example shifts persisted, plus K=200 random directions and the four
permutation tests requested in the task.

Pipeline (matches functional_orthogonality_control.py exactly):
  - Same prompt construction (build_p0_prompt over labels.jsonl × baseline_results.jsonl)
  - Same N=100 samples
  - Same direction files (direction_decomp_{full,parallel,perp}_layer20.npz)
  - Same SteeringHook setup, same hidden_rms=0.65, same rho=-0.20, same layer L20

Persists per-example margin shifts so bootstrap and permutation tests are
reproducible from saved data without re-running the model.

Usage:
  cd tmc/scripts/e2e_agent
  python scripts/decomposition_ci_null.py
"""

import os, sys, json, argparse, time
import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import SteeringHook
from steering.directions import load_direction

LAYER = 20
RHO = -0.20
SEED = 20260429
N_BOOT = 10_000
N_PERM = 10_000


# ─── Forward / measurement ───────────────────────────────────────────────────

def margin_from_logits(logits, tool_ids, fin_ids):
    lp = torch.log_softmax(logits, dim=-1)
    return (torch.logsumexp(lp[tool_ids], 0) -
            torch.logsumexp(lp[fin_ids], 0)).item()


def compute_margin(model, tokenizer, prompt, direction, rho, layer,
                   tool_ids, fin_ids):
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if direction is not None and abs(rho) > 1e-8:
        d_rms = float(np.sqrt(np.mean(direction ** 2)))
        hidden_rms = 0.65
        alpha = rho * (hidden_rms / d_rms)
        with SteeringHook(model, direction, alpha, layer=layer,
                          position=-1, max_interventions=1):
            with torch.no_grad():
                logits = model(input_ids).logits[0, -1, :]
    else:
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1, :]
    return margin_from_logits(logits, tool_ids, fin_ids)


def build_p0_prompt(tokenizer, question, query, observation):
    pb = PromptBuilder(tools=["search", "calculator"])
    steps = [{"action": "search", "action_input": query,
              "observation": observation[:1500]}]
    msgs = pb.build_full_prompt(question, steps)
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)


def normalize_rms(d, target=1.0):
    rms = float(np.sqrt(np.mean(d ** 2)))
    return d * (target / rms) if rms > 1e-12 else d


def gen_random_dirs(dim, n, seed):
    rng = np.random.RandomState(seed)
    return [normalize_rms(rng.randn(dim).astype(np.float32), 1.0)
            for _ in range(n)]


# ─── Statistics ──────────────────────────────────────────────────────────────

def bootstrap_ci(values, n_boot=N_BOOT, ci=95.0, seed=SEED):
    """Percentile bootstrap CI for the mean."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, (100 - ci) / 2.0))
    hi = float(np.percentile(boot_means, 100 - (100 - ci) / 2.0))
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if n > 1 else 0.0,
        "n": int(n),
        "ci_low": lo, "ci_high": hi, "ci_level": ci,
        "boot_mean": float(boot_means.mean()),
        "boot_std": float(boot_means.std(ddof=1)),
    }


def paired_permutation_test(a, b, n_perm=N_PERM, seed=SEED, two_sided=True):
    """Paired sign-flip permutation test on |mean(a) - mean(b)|.
    Per pair, randomly swap (a_i, b_i) with probability 0.5."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    diff = a - b
    n = len(diff)
    obs = float(np.abs(diff.mean())) if two_sided else float(diff.mean())
    rng = np.random.RandomState(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_means = (signs * diff).mean(axis=1)
    if two_sided:
        ge = int(np.sum(np.abs(perm_means) >= obs))
    else:
        ge = int(np.sum(perm_means >= obs))
    p = (ge + 1) / (n_perm + 1)
    return {
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_diff": float((a - b).mean()),
        "abs_mean_diff": float(np.abs((a - b).mean())),
        "n_pairs": int(n),
        "n_perm": int(n_perm),
        "n_perm_ge_observed": int(ge),
        "p_value": float(p),
        "two_sided": bool(two_sided),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--baseline-trace",
                    default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--labels-path",
                    default="results/phase1_probe/labels.jsonl")
    ap.add_argument("--output-dir", default="results/decomposition_ci_null")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[init] L{LAYER} rho={RHO} seed={SEED} N={args.n_samples}"
          f" K_random={args.n_random} N_boot={N_BOOT} N_perm={N_PERM}")

    # ── Load directions (same as functional_orthogonality_control.py) ──────
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
    print(f"[init] dim={dim}  full_rms={np.sqrt(np.mean(full_dir**2)):.3f}"
          f"  par_rms={np.sqrt(np.mean(par_dir**2)):.3f}"
          f"  perp_rms={np.sqrt(np.mean(perp_dir**2)):.3f}")

    random_dirs = gen_random_dirs(dim, args.n_random, SEED)
    print(f"[init] generated K={len(random_dirs)} random unit-RMS directions")

    # ── Load model ──────────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"[load] {name}")
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    tool_ids = [tok.encode(t, add_special_tokens=False)[0]
                for t in ACTION_TOKENS["tool_call"]]
    fin_ids = [tok.encode(t, add_special_tokens=False)[0]
               for t in ACTION_TOKENS["finish"]]
    print(f"[init] tool_ids={tool_ids}  fin_ids={fin_ids}")
    run(model, tok, full_dir, par_dir, perp_dir, random_dirs,
        tool_ids, fin_ids, args)


def run(model, tok, full_dir, par_dir, perp_dir, random_dirs,
        tool_ids, fin_ids, args):
    # ── Build prompts (same selection logic as the original script) ────────
    label_data = [json.loads(l) for l in open(args.labels_path)]
    bl_map = {}
    with open(args.baseline_trace) as f:
        for line in f:
            ep = json.loads(line)
            bl_map[ep["sample_id"]] = ep
    prompts, sample_ids = [], []
    for ld in label_data:
        ep = bl_map.get(ld["sample_id"])
        if not ep or not ep.get("steps") or len(ep["steps"]) < 1:
            continue
        s0 = ep["steps"][0]
        if s0.get("action") != "search" or not s0.get("observation"):
            continue
        prompts.append(build_p0_prompt(
            tok, ld["question"], s0["action_input"], s0["observation"]))
        sample_ids.append(ld["sample_id"])
        if len(prompts) >= args.n_samples:
            break
    n = len(prompts)
    print(f"[prompts] N={n}")

    # ── Measure: baseline + 3 named directions + K random ──────────────────
    base = np.zeros(n, dtype=np.float32)
    full_m = np.zeros(n, dtype=np.float32)
    par_m = np.zeros(n, dtype=np.float32)
    perp_m = np.zeros(n, dtype=np.float32)
    K = len(random_dirs)
    rand_m = np.zeros((K, n), dtype=np.float32)

    t0 = time.time()
    n_total_fw = n * (4 + K)
    fw_done = 0
    for i, p in enumerate(prompts):
        base[i] = compute_margin(model, tok, p, None, 0.0, LAYER, tool_ids, fin_ids); fw_done += 1
        full_m[i] = compute_margin(model, tok, p, full_dir, RHO, LAYER, tool_ids, fin_ids); fw_done += 1
        par_m[i] = compute_margin(model, tok, p, par_dir, RHO, LAYER, tool_ids, fin_ids); fw_done += 1
        perp_m[i] = compute_margin(model, tok, p, perp_dir, RHO, LAYER, tool_ids, fin_ids); fw_done += 1
        for k in range(K):
            rand_m[k, i] = compute_margin(model, tok, p, random_dirs[k], RHO, LAYER, tool_ids, fin_ids)
            fw_done += 1
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (n - i - 1)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{n}] {elapsed:.0f}s elapsed, ~{eta:.0f}s ETA"
                  f"  ({fw_done}/{n_total_fw} forwards)")

    # ── Per-example margin shifts ───────────────────────────────────────────
    full_sh = full_m - base
    par_sh = par_m - base
    perp_sh = perp_m - base
    rand_sh = rand_m - base[None, :]

    # Per-prompt random aggregate (avg over K random dirs at each prompt) →
    # the natural "random shift" comparison value at prompt level.
    rand_per_prompt = rand_sh.mean(axis=0)            # shape (n,)
    rand_per_dir_means = rand_sh.mean(axis=1)         # shape (K,) for null
    rand_per_dir_abs_means = np.abs(rand_sh).mean(axis=1)  # shape (K,)

    # ── Save per-example data first (so re-running stats is cheap) ─────────
    np.savez(os.path.join(args.output_dir, "per_example_shifts.npz"),
             baseline=base, full=full_sh, parallel=par_sh, perp=perp_sh,
             random=rand_sh, sample_ids=np.array(sample_ids))
    with open(os.path.join(args.output_dir, "per_example_shifts.jsonl"), "w") as f:
        for i in range(n):
            f.write(json.dumps({
                "sample_id": sample_ids[i],
                "baseline_margin": float(base[i]),
                "full_shift": float(full_sh[i]),
                "parallel_shift": float(par_sh[i]),
                "perp_shift": float(perp_sh[i]),
                "random_mean_shift_at_prompt": float(rand_per_prompt[i]),
            }) + "\n")
    print(f"[save] per_example_shifts.{{npz,jsonl}}")

    # ── Bootstrap CIs (10k) for each named condition ────────────────────────
    bootstrap = {
        "full":         bootstrap_ci(full_sh, N_BOOT, 95.0, SEED),
        "parallel":     bootstrap_ci(par_sh,  N_BOOT, 95.0, SEED + 1),
        "perpendicular":bootstrap_ci(perp_sh, N_BOOT, 95.0, SEED + 2),
    }
    with open(os.path.join(args.output_dir, "bootstrap_cis.json"), "w") as f:
        json.dump({"n_boot": N_BOOT, "ci_level": 95.0, "seed": SEED,
                   "n_samples": int(n),
                   "bootstrap": bootstrap}, f, indent=2)

    # ── Expanded random-direction null (K=200, per-direction mean shifts) ──
    null = {
        "K": int(K),
        "n_samples_per_dir": int(n),
        "signed": {
            "mean": float(rand_per_dir_means.mean()),
            "std":  float(rand_per_dir_means.std(ddof=1)),
            "p2_5": float(np.percentile(rand_per_dir_means, 2.5)),
            "p97_5":float(np.percentile(rand_per_dir_means, 97.5)),
            "p50":  float(np.percentile(rand_per_dir_means, 50.0)),
            "min":  float(rand_per_dir_means.min()),
            "max":  float(rand_per_dir_means.max()),
        },
        "abs": {
            "mean": float(rand_per_dir_abs_means.mean()),
            "std":  float(rand_per_dir_abs_means.std(ddof=1)),
            "p2_5": float(np.percentile(rand_per_dir_abs_means, 2.5)),
            "p97_5":float(np.percentile(rand_per_dir_abs_means, 97.5)),
            "p50":  float(np.percentile(rand_per_dir_abs_means, 50.0)),
        },
        "per_direction_mean_shifts": rand_per_dir_means.tolist(),
        "per_direction_abs_mean_shifts": rand_per_dir_abs_means.tolist(),
    }
    with open(os.path.join(args.output_dir, "null_distribution.json"), "w") as f:
        json.dump(null, f, indent=2)
    print(f"[save] null_distribution.json  K={K}")

    # ── Pairwise paired permutation tests (N=n pairs each) ──────────────────
    test_a = paired_permutation_test(par_sh,  rand_per_prompt, N_PERM, SEED + 10)
    test_b = paired_permutation_test(perp_sh, full_sh,         N_PERM, SEED + 11)
    test_c = paired_permutation_test(perp_sh, par_sh,          N_PERM, SEED + 12)

    # ── (d) Dissociation gap: paired sign-flip exchangeability test ────────
    obs_gap = float(perp_sh.mean() - par_sh.mean())
    rng_d = np.random.RandomState(SEED + 13)
    diff_pp = perp_sh.astype(np.float64) - par_sh.astype(np.float64)
    signs = rng_d.choice([-1.0, 1.0], size=(N_PERM, n))
    perm_gaps = (signs * diff_pp).mean(axis=1)
    ge = int(np.sum(np.abs(perm_gaps) >= abs(obs_gap)))
    p_d = (ge + 1) / (N_PERM + 1)
    test_d = {
        "observed_gap_perp_minus_par": obs_gap,
        "abs_observed_gap": abs(obs_gap),
        "n_perm": int(N_PERM),
        "n_perm_ge_observed_abs": int(ge),
        "p_value": float(p_d),
        "perm_gap_mean": float(perm_gaps.mean()),
        "perm_gap_std": float(perm_gaps.std(ddof=1)),
        "perm_gap_p2_5": float(np.percentile(perm_gaps, 2.5)),
        "perm_gap_p97_5":float(np.percentile(perm_gaps, 97.5)),
    }
    pairwise = {
        "n_perm": int(N_PERM), "n_pairs": int(n), "seed": SEED,
        "a_parallel_vs_random_per_prompt": test_a,
        "b_perp_vs_full":                  test_b,
        "c_perp_vs_parallel":              test_c,
        "d_dissociation_gap_exchangeability": test_d,
    }
    with open(os.path.join(args.output_dir, "pairwise_tests.json"), "w") as f:
        json.dump(pairwise, f, indent=2)
    print(f"[save] pairwise_tests.json")

    # ── Summary ─────────────────────────────────────────────────────────────
    summary = {
        "config": {"layer": LAYER, "rho": RHO, "seed": SEED,
                   "n_samples": int(n), "K_random": int(K),
                   "n_boot": N_BOOT, "n_perm": N_PERM,
                   "model": "Qwen/Qwen2.5-7B-Instruct",
                   "directions": {
                       "full": "steering/directions/direction_decomp_full_layer20.npz",
                       "parallel": "steering/directions/direction_decomp_parallel_layer20.npz",
                       "perp": "steering/directions/direction_decomp_perp_layer20.npz",
                   }},
        "point_estimates": {
            "baseline_margin_mean": float(base.mean()),
            "full_mean_shift": float(full_sh.mean()),
            "parallel_mean_shift": float(par_sh.mean()),
            "perp_mean_shift": float(perp_sh.mean()),
            "parallel_abs_mean_shift": float(np.abs(par_sh).mean()),
            "random_per_prompt_mean": float(rand_per_prompt.mean()),
        },
        "bootstrap_95ci": {k: {"mean": v["mean"],
                                "ci_low": v["ci_low"],
                                "ci_high": v["ci_high"]}
                           for k, v in bootstrap.items()},
        "random_null_K": {"K": int(K),
                          "signed_mean": null["signed"]["mean"],
                          "signed_std":  null["signed"]["std"],
                          "signed_p2_5": null["signed"]["p2_5"],
                          "signed_p97_5":null["signed"]["p97_5"]},
        "pairwise_p_values": {
            "a_parallel_vs_random_per_prompt": test_a["p_value"],
            "b_perp_vs_full":                  test_b["p_value"],
            "c_perp_vs_parallel":              test_c["p_value"],
            "d_dissociation_gap":              test_d["p_value"],
        },
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] summary.json")
    print(f"\n[done] full mean Δ = {full_sh.mean():+.3f}"
          f" 95%CI [{bootstrap['full']['ci_low']:+.3f}, {bootstrap['full']['ci_high']:+.3f}]")
    print(f"[done] par  mean Δ = {par_sh.mean():+.3f}"
          f" 95%CI [{bootstrap['parallel']['ci_low']:+.3f}, {bootstrap['parallel']['ci_high']:+.3f}]")
    print(f"[done] perp mean Δ = {perp_sh.mean():+.3f}"
          f" 95%CI [{bootstrap['perpendicular']['ci_low']:+.3f}, {bootstrap['perpendicular']['ci_high']:+.3f}]")
    print(f"[done] random K={K}: signed mean {null['signed']['mean']:+.3f}"
          f" 95% [{null['signed']['p2_5']:+.3f}, {null['signed']['p97_5']:+.3f}]")
    print(f"[done] perm p: par~rand={test_a['p_value']:.4f}"
          f"  perp~full={test_b['p_value']:.4f}"
          f"  perp~par={test_c['p_value']:.4f}"
          f"  gap={test_d['p_value']:.4f}")


if __name__ == "__main__":
    main()
