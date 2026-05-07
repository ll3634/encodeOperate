#!/usr/bin/env python3
"""
Dose-Response Gain-Ratio Sweep — p0 decision-only.

For each direction in {action, evidence_parallel, random_seed_{i}} run the
TimedRhoStep2OnlyPolicy at every rho in --rhos and record per-episode results.
All directions are RMS-normalized to 1.0 so a fixed rho maps to identical alpha.

Outputs (under --out):
    baseline_results.jsonl                   freegen baseline (reused if present)
    <dir_label>/rho_<MAG>_results.jsonl      per-condition episode JSONL
    summary.json                             per-condition aggregate stats
    report.md                                table + slope + gain-ratio
    figure.png / figure.pdf                  dose-response curve

Resume-friendly: existing per-condition JSONLs with the expected sample_ids
are loaded as-is and skipped.
"""

import os, sys, json, argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from agent.react_loop import ReActAgent, AgentConfig
from agent.policies_verify import FreeGenBaselinePolicy, TimedRhoStep2OnlyPolicy
from datasets.hotpotqa import HotpotQADataset
from tools.search_tool import SearchTool
from steering.directions import load_direction
from scripts.run_verify_critical_pipeline import (
    run_episode, compute_stats, compute_activation_stats,
)


def load_model_and_tokenizer(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    mdl.eval()
    return mdl, tok


def load_samples_from_ids(dataset, id_file: Path, max_n=None, seed=42):
    """Load samples whose IDs appear in id_file, preserving file order."""
    ids_in_order = []
    seen = set()
    for line in open(id_file):
        sid = json.loads(line)["sample_id"]
        if sid not in seen:
            ids_in_order.append(sid)
            seen.add(sid)
    target_n = len(ids_in_order)
    pool = dataset.get_subset(target_n * 3, seed=seed, type_filter="bridge")
    by_id = {s.id: s for s in pool}
    samples = [by_id[sid] for sid in ids_in_order if sid in by_id]
    if max_n is not None:
        samples = samples[:max_n]
    print(f"  Loaded {len(samples)} samples (target {target_n}, capped at {max_n})")
    return samples


def make_random_unit_rms_direction(dim: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    rms = float(np.sqrt(np.mean(v ** 2)))
    return v / max(rms, 1e-12)


def jsonl_load(path: Path):
    return [json.loads(line) for line in open(path)]


def jsonl_dump(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_condition(out_path: Path, agent, samples, policy, score_mode, label):
    """Run one (direction, rho) condition with resume support."""
    expected = {s.id for s in samples}
    if out_path.exists():
        existing = jsonl_load(out_path)
        if {r["sample_id"] for r in existing} == expected:
            print(f"  [{label}] reusing cached {out_path.name} ({len(existing)} rows)")
            return existing
        else:
            print(f"  [{label}] cache id-set mismatch, re-running")
    results = []
    for s in tqdm(samples, desc=label):
        results.append(run_episode(agent, s, policy, score_mode))
    jsonl_dump(out_path, results)
    return results


def step1_pre_margin_stats(results):
    """Mean/median pre-steering margin at step 1 (sample-population sanity check).

    NOTE: margin_before in AgentStep is the natural margin before the steering
    vector is added, so this is a property of the *sample distribution* and is
    expected to be approximately constant across rho/direction conditions. It is
    NOT a measure of the steering effect.
    """
    vals = []
    for r in results:
        for st in r.get("steps", []):
            if st.get("step_idx") == 1 and st.get("margin_before") is not None:
                vals.append(float(st["margin_before"]))
                break
    if not vals:
        return {"n": 0, "mean": float("nan"), "median": float("nan")}
    return {
        "n": len(vals),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
    }


def summarise(label, bl_results, run_results):
    fs = compute_stats(bl_results, run_results)
    act = compute_activation_stats(run_results)
    pre_margin = step1_pre_margin_stats(run_results)
    n = fs.get("n", 0)
    pf = fs.get("parse_failures", 0)
    rsr = act["second_search_activation_rate"] * 100
    print(
        f"  [{label:24s}] n={n}  steered={fs.get('policy_rate', 0)*100:5.1f}%  "
        f"2ndSR={rsr:5.1f}%  PF={pf}/{n}  m_pre_mean={pre_margin['mean']:+.3f}"
    )
    return {"stats": fs, "activation": act, "margin_step1_pre": pre_margin}


def main():
    p = argparse.ArgumentParser(description="Dose-response gain-ratio sweep")
    p.add_argument("--data-path", required=True)
    p.add_argument("--corpus-path", required=True)
    p.add_argument("--baseline-ids", required=True,
                   help="JSONL with sample_id field; the first --n-samples are used.")
    p.add_argument("--baseline-cache", default=None,
                   help="Optional path to existing baseline_results.jsonl to copy in.")
    p.add_argument("--dir-action",
                   default="steering/directions/direction_decomp_full_layer20.npz")
    p.add_argument("--dir-evidence",
                   default="steering/directions/direction_decomp_parallel_layer20.npz")
    p.add_argument("--n-random", type=int, default=20,
                   help="Number of matched-RMS random directions.")
    p.add_argument("--random-seed-base", type=int, default=1000)
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--rhos", default="0.05,0.10,0.20,0.50,1.00,1.50",
                   help="Comma-separated rho magnitudes (positive). Sign applied per --rho-sign.")
    p.add_argument("--rho-sign", type=int, default=-1, choices=[-1, 1],
                   help="Sign convention. -1 = continue/search direction (canonical).")
    p.add_argument("--alpha-max", type=float, default=8.0)
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--score-mode", default="exact")
    p.add_argument("--out", default="results/dose_response_gain_ratio")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke mode: --n-samples=5, 2 rhos {0.20,1.50}, K=1 random.")
    args = p.parse_args()

    if args.smoke:
        args.n_samples = 5
        args.rhos = "0.20,1.50"
        args.n_random = 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rho_mags = [float(x) for x in args.rhos.split(",")]
    rhos_signed = [args.rho_sign * m for m in rho_mags]

    print("=" * 76)
    print("  DOSE-RESPONSE GAIN-RATIO SWEEP")
    print("=" * 76)
    print(f"  layer={args.layer}  timing=p0  N={args.n_samples}")
    print(f"  rho magnitudes: {rho_mags}")
    print(f"  rho sign      : {args.rho_sign:+d}  (signed: {rhos_signed})")
    print(f"  K_random      : {args.n_random}")

    print("\n[1/5] Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model)

    print("\n[2/5] Loading dataset and samples...")
    dataset = HotpotQADataset(args.data_path)
    samples = load_samples_from_ids(dataset, Path(args.baseline_ids), args.n_samples)
    expected_ids = {s.id for s in samples}

    print("\n[3/5] Loading directions...")
    d_action, _ = load_direction(args.dir_action, normalize_rms=1.0)
    d_evidence, _ = load_direction(args.dir_evidence, normalize_rms=1.0)
    dim = d_action.shape[-1]
    cos_ae = float(
        np.dot(d_action, d_evidence)
        / (np.linalg.norm(d_action) * np.linalg.norm(d_evidence) + 1e-12)
    )
    print(f"  dim={dim}  cos(action, evidence) = {cos_ae:+.4f}")
    direction_specs = [("action", d_action), ("evidence_parallel", d_evidence)]
    for k in range(args.n_random):
        seed = args.random_seed_base + k
        v = make_random_unit_rms_direction(dim, seed)
        cos_ar = float(np.dot(d_action, v) / (np.linalg.norm(d_action) * np.linalg.norm(v)))
        direction_specs.append((f"random_s{seed}", v))
        if k < 3:
            print(f"  random_s{seed}: cos(action) = {cos_ar:+.4f}")

    search_tool = SearchTool(corpus_path=args.corpus_path)
    tools = {"search": search_tool}
    config = AgentConfig(
        max_steps=5, max_tokens_per_step=256, temperature=0.0,
        layer=args.layer, tools=list(tools.keys()), score_mode=args.score_mode,
    )

    bl_path = out_dir / "baseline_results.jsonl"
    if not bl_path.exists() and args.baseline_cache and Path(args.baseline_cache).exists():
        cached = jsonl_load(Path(args.baseline_cache))
        kept = [r for r in cached if r["sample_id"] in expected_ids]
        if {r["sample_id"] for r in kept} == expected_ids:
            jsonl_dump(bl_path, kept)
            print(f"  Imported baseline rows from cache: {len(kept)}")

    print("\n[4/5] Baseline (freegen, rho=0)...")
    agent_bl = ReActAgent(
        model=model, tokenizer=tokenizer, tools=tools, config=config,
        direction=d_action, direction_rms=1.0,
    )
    bl_policy = FreeGenBaselinePolicy()
    bl_results = run_condition(bl_path, agent_bl, samples, bl_policy, args.score_mode, "baseline")
    bl_acc = sum(r["is_correct"] for r in bl_results) / len(bl_results)
    bl_2sr = sum(1 for r in bl_results if r["tool_calls"] >= 2) / len(bl_results)
    bl_margin = step1_pre_margin_stats(bl_results)
    print(f"  Baseline: acc={bl_acc*100:.1f}%  2ndSR={bl_2sr*100:.1f}%  "
          f"m_pre_mean={bl_margin['mean']:+.3f}")

    print("\n[5/5] Running dose-response sweep...")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model, "layer": args.layer, "timing": "p0",
        "n_samples": len(samples), "rho_magnitudes": rho_mags,
        "rho_sign": args.rho_sign, "alpha_max": args.alpha_max,
        "cos_action_evidence": cos_ae,
        "baseline": {
            "acc": bl_acc, "2nd_search_rate": bl_2sr,
            "margin_step1_pre": bl_margin,
        },
        "conditions": {},
    }
    for label, vec in direction_specs:
        print(f"\n  === Direction: {label} ===")
        agent_d = ReActAgent(
            model=model, tokenizer=tokenizer, tools=tools, config=config,
            direction=vec, direction_rms=1.0,
        )
        per_rho = {}
        for mag, rho in zip(rho_mags, rhos_signed):
            cond_path = out_dir / label / f"rho_{mag:.2f}_results.jsonl"
            policy = TimedRhoStep2OnlyPolicy(rho=rho, timing="p0", alpha_max=args.alpha_max)
            tag = f"{label}|rho={rho:+.2f}"
            run_results = run_condition(cond_path, agent_d, samples, policy, args.score_mode, tag)
            per_rho[f"{mag:.2f}"] = summarise(tag, bl_results, run_results)
        summary["conditions"][label] = per_rho
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSummary saved to {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
