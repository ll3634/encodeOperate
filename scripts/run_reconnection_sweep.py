#!/usr/bin/env python3
"""
Reconnection Interventions Sweep.

Tests whether any naive intervention makes the (otherwise inert) evidence axis
operative for the stop/continue decision.  All conditions are decision-only
at step 1 (p0).  Three arms:

  Arm A — Combined steering (action + evidence at L20)
    A_low / A_med / A_high : co-inject action_dir at canonical magnitude plus
    evidence_dir at 0.5x / 1.0x / 2.0x matched-RMS magnitude (both pushing
    toward search).  Tests whether evidence becomes effective once the action
    pathway is concurrently activated.

  Arm B — Evidence-to-action rank-1 rotation
    B_rotate : at decision token, h' = h + alpha * (h . u_ev) * u_act with
    alpha=+10.25 (calibrated so mean injection RMS ≈ canonical 0.13).  Tests
    whether mapping the evidence signal into the action subspace makes it
    operative.

  Arm C — Evidence-direction injection at earlier layers
    C_L16 / C_L17 / C_L18 : inject evidence_dir (from L20 probe) at L16/L17/L18
    decision token at canonical RMS magnitude.  Tests whether earlier-layer
    injection bypasses any formation-gain bottleneck downstream.

Reuses results/decomposition_test/baseline_results.jsonl (N=483) as the
shared baseline for all paired statistics.
"""

import os, sys, json, argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from agent.react_loop import ReActAgent, AgentConfig
from agent.policies_verify import (
    FreeGenBaselinePolicy,
    FixedAlphaStep2OnlyPolicy,
    CrossAxisStep2OnlyPolicy,
)
from datasets.hotpotqa import HotpotQADataset
from tools.search_tool import SearchTool
from steering.directions import load_direction
from scripts.run_decomposition_test import (
    load_model_and_tokenizer, load_samples_from_ids, summarise_run,
)
from scripts.run_verify_critical_pipeline import (
    run_episode, compute_stats, compute_activation_stats,
)


# Calibration constants.  hidden_rms is empirically ~1.83 at L20 for this
# model (printed by ReActAgent._calibrate_hidden_rms during smoke run).
# Canonical action injection at rho=-0.20 with direction normalised to RMS=1
# therefore has alpha = -0.20 * 1.83 = -0.366 and inject RMS = 0.366.
HIDDEN_RMS_L20 = 1.83
ACTION_INJECT_RMS = 0.366
# Arm B alpha s.t. mean-abs-proj injection RMS matches ACTION_INJECT_RMS:
#   |alpha| * mean_abs(h.u_ev_unit) * RMS(u_act_unit) = 0.366
#   |alpha| * 0.7595          * 0.01670              = 0.366  →  alpha ≈ 28.8
ARM_B_ALPHA = 28.8
ARM_A_EV_SCALES = {"low": 0.5, "med": 1.0, "high": 2.0}


def build_combined_direction(u_act_rms1: np.ndarray, u_ev_rms1: np.ndarray,
                              c_act: float, c_ev: float) -> np.ndarray:
    """direction = c_act * u_act_rms1 + c_ev * u_ev_rms1 (both inputs RMS=1)."""
    return c_act * u_act_rms1 + c_ev * u_ev_rms1


def build_conditions(arms, u_act_rms1, u_ev_rms1, u_act_unit, u_ev_unit):
    """
    Each condition is a dict with:
      name, arm, layer, kind ('linear' | 'cross'), direction (for linear),
      alpha (for linear and cross), u_in/u_out (for cross), description.
    """
    conditions = []

    # Arm A: combined steering at L20.  Action component injects at canonical
    # rho=-0.20 magnitude (c_act = -ACTION_INJECT_RMS = -0.13 on RMS=1 vec);
    # evidence component co-injects at low/med/high * matched magnitude.
    if "A" in arms:
        c_act = -ACTION_INJECT_RMS
        for tag, scale in ARM_A_EV_SCALES.items():
            c_ev = -ACTION_INJECT_RMS * scale
            d_combined = build_combined_direction(u_act_rms1, u_ev_rms1, c_act, c_ev)
            conditions.append({
                "name": f"A_{tag}_act+ev_x{scale:g}",
                "arm": "A",
                "layer": 20,
                "kind": "linear",
                "direction": d_combined.astype(np.float32),
                "alpha": 1.0,            # all scaling baked into direction
                "description": (
                    f"L20 co-injection: c_act={c_act:+.3f}, "
                    f"c_ev={c_ev:+.3f} (×{scale:g} of action magnitude)"
                ),
            })

    # Arm B: rank-1 cross-axis at L20.  u_in = +u_ev_unit (sufficient direction),
    # u_out = +u_act_unit (stop direction); alpha=+10.25 makes injection on
    # insufficient samples push toward search via natural sign flip.
    if "B" in arms:
        conditions.append({
            "name": "B_rotate_ev->act",
            "arm": "B",
            "layer": 20,
            "kind": "cross",
            "u_in": u_ev_unit.copy(),
            "u_out": u_act_unit.copy(),
            "alpha": ARM_B_ALPHA,
            "description": (
                f"L20 rank-1: h += {ARM_B_ALPHA:+.2f} * (h.u_ev) * u_act"
            ),
        })

    # Arm C: evidence-direction injection at earlier layers.  We use the L20
    # probe direction at L16/L17/L18 (shared because residual stream is
    # gradual).  Magnitude matches canonical action injection.
    if "C" in arms:
        c_ev_layer = -ACTION_INJECT_RMS  # push toward search
        d_ev_only = (c_ev_layer * u_ev_rms1).astype(np.float32)
        for layer in (16, 17, 18):
            conditions.append({
                "name": f"C_L{layer}_ev_only",
                "arm": "C",
                "layer": layer,
                "kind": "linear",
                "direction": d_ev_only,
                "alpha": 1.0,
                "description": (
                    f"L{layer} ev-only injection: c_ev={c_ev_layer:+.3f}"
                ),
            })

    return conditions


def run_all_conditions(model, tokenizer, tools, base_config,
                        conditions, samples, bl_results, score_mode, out_dir):
    """Run each condition end-to-end; save per-condition jsonl + return stats."""
    out = {}
    for cond in conditions:
        name = cond["name"]
        print(f"\n  === {name} ({cond['arm']} | L{cond['layer']}) ===")
        print(f"      {cond['description']}")

        if cond["kind"] == "linear":
            d = cond["direction"]
            d_rms = float(np.sqrt((d ** 2).mean()))
            agent = ReActAgent(
                model=model, tokenizer=tokenizer, tools=tools,
                config=base_config, direction=d, direction_rms=d_rms,
            )
            policy = FixedAlphaStep2OnlyPolicy(
                alpha=cond["alpha"],
                steer_layer=(cond["layer"] if cond["layer"] != base_config.layer else None),
            )
        elif cond["kind"] == "cross":
            agent = ReActAgent(
                model=model, tokenizer=tokenizer, tools=tools,
                config=base_config, direction=None, direction_rms=1.0,
            )
            policy = CrossAxisStep2OnlyPolicy(
                u_in=cond["u_in"], u_out=cond["u_out"],
                alpha=cond["alpha"], layer=cond["layer"],
            )
        else:
            raise ValueError(f"unknown kind: {cond['kind']}")

        run_results = []
        for s in tqdm(samples, desc=name):
            run_results.append(run_episode(agent, s, policy, score_mode))

        with open(out_dir / f"{name}_results.jsonl", "w") as f:
            for r in run_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Restrict baseline to the same subset of IDs (paired)
        ids = set(r["sample_id"] for r in run_results)
        bl_subset = [r for r in bl_results if r["sample_id"] in ids]
        fs, act = summarise_run(name, bl_subset, run_results)
        out[name] = {
            "condition": cond_meta(cond),
            "stats": fs,
            "activation": act,
        }
    return out


def cond_meta(cond):
    """Strip large numpy arrays from the condition dict for JSON serialisation."""
    return {k: v for k, v in cond.items()
            if k not in ("direction", "u_in", "u_out")}


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_report(out_dir, args, conditions, results, bl_acc, bl_2sr,
                  cos_av, rms_act, rms_ev):
    """Print summary table + write reconnection_sweep_report.{json,md}."""
    print("\n" + "=" * 78)
    print("  RECONNECTION SWEEP SUMMARY")
    print("=" * 78)
    print(f"  Baseline: acc={bl_acc*100:.1f}%  2ndSR={bl_2sr*100:.1f}%  "
          f"cos(act,ev)={cos_av:+.4f}")
    print()
    header = (f"  {'Condition':<26} {'Arm':>3} {'Lyr':>3} "
              f"{'Acc':>6} {'2ndSR':>7} {'Resc':>5} {'Regr':>5} "
              f"{'Net':>5} {'NetC':>5} {'PF':>4} {'Purity':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for cond in conditions:
        name = cond["name"]
        if name not in results:
            continue
        fs = results[name]["stats"]
        act = results[name]["activation"]
        resc = fs.get("rescued", 0)
        rws = fs.get("rescued_with_more_search", 0)
        purity = f"{rws/resc*100:.0f}%" if resc > 0 else "n/a"
        net_c = fs.get("net_gain_corrected", fs.get("net_gain", 0))
        print(
            f"  {name:<26} {cond['arm']:>3} {cond['layer']:>3} "
            f"{fs['policy_rate']*100:>5.1f}%  "
            f"{act['second_search_activation_rate']*100:>5.1f}%  "
            f"{resc:>5}  {fs.get('regressed', 0):>5}  "
            f"{fs.get('net_gain', 0):>+5}  {net_c:>+5}  "
            f"{fs.get('parse_failures', 0):>4}  {purity:>7}"
        )

    report = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "adapter_path": args.adapter_path,
        "n_samples": args.n_samples,
        "baseline_acc": bl_acc,
        "baseline_2nd_search_rate": bl_2sr,
        "cos_action_evidence": cos_av,
        "rms_action": rms_act,
        "rms_evidence": rms_ev,
        "calibration": {
            "hidden_rms_l20": HIDDEN_RMS_L20,
            "action_inject_rms": ACTION_INJECT_RMS,
            "arm_b_alpha": ARM_B_ALPHA,
            "arm_a_evidence_scales": ARM_A_EV_SCALES,
        },
        "conditions": [_to_jsonable(cond_meta(c)) for c in conditions],
        "results": _to_jsonable(results),
    }
    json_path = out_dir / "reconnection_sweep_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  → JSON report: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Reconnection interventions sweep")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--baseline-results",
                        default="results/decomposition_test/baseline_results.jsonl",
                        help="Reused baseline jsonl (sample IDs and EM).")
    parser.add_argument("--action-dir-file",
                        default="steering/directions/direction_search_v3_layer20.npz")
    parser.add_argument("--evidence-dir-file",
                        default="results/phase1_probe/probe_direction_l20.npz")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--score-mode", default="exact")
    parser.add_argument("--out", default="results/reconnection_sweep")
    parser.add_argument("--arms", default="A,B,C",
                        help="Comma-separated subset of arms to run (A,B,C).")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke run: forces n-samples=8 if larger.")
    args = parser.parse_args()

    if args.smoke and args.n_samples > 8:
        args.n_samples = 8
    arms = set(s.strip().upper() for s in args.arms.split(","))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  RECONNECTION INTERVENTIONS SWEEP")
    print("=" * 70)
    print(f"  arms={sorted(arms)}  n_samples={args.n_samples}  smoke={args.smoke}")

    # ── Load model + tokenizer ────────────────────────────────────────────────
    print("\n[1/6] Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model, adapter_path=args.adapter_path)

    # ── Load directions ──────────────────────────────────────────────────────
    print("[2/6] Loading directions...")
    d_act_raw, _ = load_direction(args.action_dir_file, normalize_rms=None)
    d_ev_raw = np.load(args.evidence_dir_file, allow_pickle=True)["decision_direction"]
    rms_act = float(np.sqrt((d_act_raw ** 2).mean()))
    rms_ev = float(np.sqrt((d_ev_raw ** 2).mean()))
    u_act_rms1 = (d_act_raw / rms_act).astype(np.float32)   # RMS = 1
    u_ev_rms1 = (d_ev_raw / rms_ev).astype(np.float32)      # RMS = 1
    u_act_unit = (d_act_raw / np.linalg.norm(d_act_raw)).astype(np.float32)
    u_ev_unit = (d_ev_raw / np.linalg.norm(d_ev_raw)).astype(np.float32)
    cos_av = float(u_act_unit @ u_ev_unit)
    print(f"  action: ||={np.linalg.norm(d_act_raw):.3f}  RMS={rms_act:.4f}")
    print(f"  evidence: ||={np.linalg.norm(d_ev_raw):.3f}  RMS={rms_ev:.4f}")
    print(f"  cos(action, evidence) = {cos_av:+.4f}")

    # ── Load samples ──────────────────────────────────────────────────────────
    print("[3/6] Loading samples + baseline...")
    dataset = HotpotQADataset(args.data_path)
    samples = load_samples_from_ids(dataset, Path(args.baseline_results), args.n_samples)

    # Load baseline results (reuse) and filter to selected sample IDs
    bl_all = [json.loads(l) for l in open(args.baseline_results)]
    sample_id_set = set(s.id for s in samples)
    bl_results = [r for r in bl_all if r["sample_id"] in sample_id_set]
    bl_acc = sum(r["is_correct"] for r in bl_results) / len(bl_results)
    bl_2sr = sum(1 for r in bl_results if r["tool_calls"] >= 2) / len(bl_results)
    print(f"  baseline: n={len(bl_results)}  acc={bl_acc*100:.1f}%  2ndSR={bl_2sr*100:.1f}%")

    # ── Build search tool + base agent config ────────────────────────────────
    search_tool = SearchTool(corpus_path=args.corpus_path)
    tools = {"search": search_tool}
    base_config = AgentConfig(
        max_steps=5, max_tokens_per_step=256, temperature=0.0,
        layer=20, tools=list(tools.keys()), score_mode=args.score_mode,
    )

    # Hand off remaining work to the conditions runner (kept in the same
    # script via a helper so this function stays under the line limit).
    print("\n[4/6] Building condition list...")
    conditions = build_conditions(arms, u_act_rms1, u_ev_rms1, u_act_unit, u_ev_unit)
    print(f"  {len(conditions)} conditions to run")

    print("\n[5/6] Running conditions...")
    results = run_all_conditions(model, tokenizer, tools, base_config,
                                  conditions, samples, bl_results,
                                  args.score_mode, out_dir)

    print("\n[6/6] Writing report...")
    write_report(out_dir, args, conditions, results, bl_acc, bl_2sr,
                 cos_av, rms_act, rms_ev)


if __name__ == "__main__":
    main()
