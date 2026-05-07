#!/usr/bin/env python3
"""
Probe Robustness Sweep
======================
Trains 10 logistic-regression evidence probes at L20 with different random seeds,
checks that the §8.3 functional decomposition pattern (parallel inert,
perpendicular ≈ full) is robust to the choice of probe.

Pipeline (mirrors decomposition_ci_null.py exactly):
  - Same prompt construction (build_p0_prompt over labels.jsonl × baseline_results.jsonl)
  - Same N=100 samples, layer L20, rho=-0.20, hidden_rms=0.65
  - Per-direction RMS normalization to 1.0 before injection (matches load_direction)

For each seed (0..9):
  1. Train LR(class_weight="balanced", C=1.0) on standardized L20 activations,
     80/20 stratified split (test for AUROC), then refit on full data for direction.
  2. cos(evidence_dir_seed, action_dir)            (target: |cos| << 1, ~ -0.013)
  3. cos(evidence_dir_seed, evidence_dir_original) (target: > 0.8)
  4. Decompose action_dir = parallel + perp w.r.t. seed's evidence_dir.
  5. Measure margin shifts for {full, parallel, perp} at the decision point.

PASS criteria per seed:
  - parallel mean-shift inside cached random null 95% CI [-0.373, +0.543]
  - perpendicular mean-shift / full mean-shift in [0.7, 1.3]
Overall: PASS if >=9/10 seeds satisfy both; BORDERLINE 7-8/10; FAIL <7/10.
"""

import os, sys, json, argparse, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import SteeringHook

LAYER = 20
RHO = -0.20
HIDDEN_RMS = 0.65
N_SEEDS = 10
SEEDS = list(range(N_SEEDS))

# Cached random-null 95% CI of mean shift (K=200, N=100) at the same protocol;
# source: results/decomposition_ci_null/null_distribution.json
NULL_CI_LOW = -0.37256250232458116
NULL_CI_HIGH = 0.5427812680602074
NULL_MEAN = 0.048090629279613495
NULL_STD = 0.22289563715457916

PERP_RATIO_LOW = 0.7
PERP_RATIO_HIGH = 1.3


# ─── Probe training (same configuration as scripts/phase1_multilayer_probe.py) ─

def train_probe_seed(X, y, seed):
    """Train a probe on an 80% stratified split.

    Note: the per-seed probe direction is taken from the *train-set* fit (not
    a refit-on-all-data) so the direction varies with the split — that is the
    point of the robustness sweep. The original phase1 probe refits on full
    data for the canonical direction; here we deliberately diverge to stress
    the dissociation pattern across alternative probes.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(sss.split(X, y))

    # Standardize using train-set statistics only (no leakage)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[train_idx])
    X_test_s = scaler.transform(X[test_idx])

    clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                             solver="lbfgs", random_state=seed)
    clf.fit(X_train_s, y[train_idx])
    y_prob = clf.predict_proba(X_test_s)[:, 1]
    y_pred = clf.predict(X_test_s)
    bal_acc = float(balanced_accuracy_score(y[test_idx], y_pred))
    auroc = float(roc_auc_score(y[test_idx], y_prob))

    # Direction comes from the train-set probe (varies per seed)
    w_orig = clf.coef_[0] / scaler.scale_
    direction = (w_orig / np.linalg.norm(w_orig)).astype(np.float32)  # unit-norm
    return direction, {"seed": int(seed), "balanced_accuracy": bal_acc,
                       "auroc": auroc, "n_train": int(len(train_idx)),
                       "n_test": int(len(test_idx))}


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def normalize_rms(d, target=1.0):
    rms = float(np.sqrt(np.mean(d ** 2)))
    return d * (target / rms) if rms > 1e-12 else d


def decompose(action, evidence_unit):
    parallel_coef = float(np.dot(action, evidence_unit))
    parallel = parallel_coef * evidence_unit
    perp = action - parallel
    return parallel, perp, parallel_coef


# ─── Margin measurement (mirrors decomposition_ci_null.compute_margin) ────────

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
        alpha = rho * (HIDDEN_RMS / d_rms)
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


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--baseline-trace",
                    default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--labels-path",
                    default="results/phase1_probe/labels.jsonl")
    ap.add_argument("--activations-path",
                    default="results/phase1_probe/activations_multilayer.npz")
    ap.add_argument("--evidence-orig-path",
                    default="results/phase1_probe/probe_direction_l20.npz")
    ap.add_argument("--action-path",
                    default="steering/directions/direction_search_v3_layer20.npz")
    ap.add_argument("--output-dir", default="results/probe_robustness_sweep")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[init] L{LAYER} rho={RHO} hidden_rms={HIDDEN_RMS} N={args.n_samples} seeds={SEEDS}")

    # ── Load reference directions ─────────────────────────────────────────────
    ev_orig = np.load(args.evidence_orig_path)["decision_direction"].astype(np.float32)
    action = np.load(args.action_path)["decision_direction"].astype(np.float32)
    ev_orig_u = unit(ev_orig)
    action_u = unit(action)
    print(f"[init] ||ev_orig||={np.linalg.norm(ev_orig):.4f}  ||action||={np.linalg.norm(action):.4f}"
          f"  cos(ev_orig, action)={float(np.dot(ev_orig_u, action_u)):+.4f}")

    # ── Train 10 probes from cached L20 activations (CPU, fast) ──────────────
    act = np.load(args.activations_path)
    X = act[f"layer_{LAYER}"]
    y = act["y"]
    print(f"[probe-data] X={X.shape}  label0={int((y==0).sum())} label1={int((y==1).sum())}")

    seed_dirs = {}
    seed_metrics = {}
    print("[train] training 10 probes ...")
    t0 = time.time()
    for s in SEEDS:
        d, m = train_probe_seed(X, y, s)
        # Sign-align to original probe so the cosine sign is meaningful
        if float(np.dot(d, ev_orig_u)) < 0:
            d = -d
        seed_dirs[s] = d
        seed_metrics[s] = m
        cos_act = float(np.dot(d, action_u))
        cos_orig = float(np.dot(d, ev_orig_u))
        print(f"  seed={s}: AUROC={m['auroc']:.3f} BalAcc={m['balanced_accuracy']:.3f}"
              f"  cos(ev,act)={cos_act:+.4f}  cos(ev,ev_orig)={cos_orig:+.4f}")
    print(f"[train] done in {time.time()-t0:.1f}s")

    # ── Load model ────────────────────────────────────────────────────────────
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

    # ── Build prompts (same selection logic as decomposition_ci_null.py) ─────
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

    # ── Pre-compute the seed-specific parallel/perp directions (RMS=1.0) ─────
    full_dir = normalize_rms(action.astype(np.float32), 1.0)
    par_dirs = {}
    perp_dirs = {}
    norms_info = {}
    for s in SEEDS:
        ev_u = unit(seed_dirs[s])
        par, perp, p_coef = decompose(action.astype(np.float32), ev_u)
        par_n = float(np.linalg.norm(par))
        perp_n = float(np.linalg.norm(perp))
        # If parallel norm is essentially zero (orthogonal), still build the
        # direction via the unit evidence vector (RMS-normalize ev_u directly).
        if par_n < 1e-6:
            par_inj = normalize_rms(ev_u.astype(np.float32), 1.0)
        else:
            par_inj = normalize_rms(par.astype(np.float32), 1.0)
        perp_inj = normalize_rms(perp.astype(np.float32), 1.0)
        par_dirs[s] = par_inj
        perp_dirs[s] = perp_inj
        norms_info[s] = {
            "parallel_coef": p_coef,
            "parallel_norm": par_n,
            "perp_norm": perp_n,
            "action_norm": float(np.linalg.norm(action)),
            "fraction_parallel_energy_pct": (par_n / float(np.linalg.norm(action))) * 100.0,
        }



    # ── Forward passes: baseline + full + (par/perp × 10 seeds) per prompt ──
    base_m = np.zeros(n, dtype=np.float32)
    full_m = np.zeros(n, dtype=np.float32)
    par_m = {s: np.zeros(n, dtype=np.float32) for s in SEEDS}
    perp_m = {s: np.zeros(n, dtype=np.float32) for s in SEEDS}
    n_total_fw = n * (2 + 2 * N_SEEDS)
    fw_done = 0
    t0 = time.time()
    for i, p in enumerate(prompts):
        base_m[i] = compute_margin(model, tok, p, None, 0.0, LAYER, tool_ids, fin_ids); fw_done += 1
        full_m[i] = compute_margin(model, tok, p, full_dir, RHO, LAYER, tool_ids, fin_ids); fw_done += 1
        for s in SEEDS:
            par_m[s][i] = compute_margin(model, tok, p, par_dirs[s], RHO, LAYER, tool_ids, fin_ids); fw_done += 1
            perp_m[s][i] = compute_margin(model, tok, p, perp_dirs[s], RHO, LAYER, tool_ids, fin_ids); fw_done += 1
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (n - i - 1)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{n}] {elapsed:.0f}s elapsed, ~{eta:.0f}s ETA"
                  f"  ({fw_done}/{n_total_fw} forwards)")

    # ── Per-seed aggregation ────────────────────────────────────────────────
    full_sh = full_m - base_m
    full_mean = float(full_sh.mean())
    full_std = float(full_sh.std(ddof=1))
    print(f"[full] mean shift = {full_mean:+.4f}  std = {full_std:.4f}")

    per_seed = {}
    pass_count = 0
    for s in SEEDS:
        par_sh = par_m[s] - base_m
        perp_sh = perp_m[s] - base_m
        par_mean = float(par_sh.mean()); par_std = float(par_sh.std(ddof=1))
        perp_mean = float(perp_sh.mean()); perp_std = float(perp_sh.std(ddof=1))
        ev_u = unit(seed_dirs[s])
        cos_act = float(np.dot(ev_u, action_u))
        cos_orig = float(np.dot(ev_u, ev_orig_u))

        par_ok = (NULL_CI_LOW <= par_mean <= NULL_CI_HIGH)
        perp_ratio = perp_mean / full_mean if abs(full_mean) > 1e-6 else 0.0
        perp_ok = (PERP_RATIO_LOW <= perp_ratio <= PERP_RATIO_HIGH)
        pattern_holds = bool(par_ok and perp_ok)
        if pattern_holds:
            pass_count += 1

        per_seed[str(s)] = {
            "seed": int(s),
            "probe": seed_metrics[s],
            "cos_with_action": cos_act,
            "cos_with_original": cos_orig,
            "decomposition_norms": norms_info[s],
            "full_shift_mean": full_mean,
            "parallel_shift_mean": par_mean,
            "parallel_shift_std": par_std,
            "perp_shift_mean": perp_mean,
            "perp_shift_std": perp_std,
            "perp_over_full_ratio": perp_ratio,
            "parallel_in_null_ci": par_ok,
            "perp_ratio_in_band": perp_ok,
            "pattern_holds": pattern_holds,
            "per_example_baseline": base_m.tolist(),
            "per_example_parallel_shift": par_sh.tolist(),
            "per_example_perp_shift": perp_sh.tolist(),
        }
        flag = "PASS" if pattern_holds else "FAIL"
        print(f"  seed={s} [{flag}] cos(act)={cos_act:+.4f} cos(orig)={cos_orig:+.4f}"
              f"  par={par_mean:+.4f}{'(null✓)' if par_ok else '(null✗)'}"
              f"  perp={perp_mean:+.4f} perp/full={perp_ratio:.3f}"
              f"{'(band✓)' if perp_ok else '(band✗)'}")

    # ── Verdict ─────────────────────────────────────────────────────────────
    if pass_count >= 9:
        verdict = "PASS"
    elif pass_count >= 7:
        verdict = "BORDERLINE"
    else:
        verdict = "FAIL"
    print(f"\n=== Verdict: {pass_count}/{N_SEEDS} → {verdict} ===")

    # ── Save outputs ────────────────────────────────────────────────────────
    summary = {
        "n_seeds": N_SEEDS,
        "n_samples": n,
        "layer": LAYER,
        "rho": RHO,
        "hidden_rms": HIDDEN_RMS,
        "verdict": verdict,
        "n_pass": int(pass_count),
        "pass_threshold": {"PASS": 9, "BORDERLINE": 7, "FAIL": "<7"},
        "criteria": {
            "parallel_in_null_ci_95": [NULL_CI_LOW, NULL_CI_HIGH],
            "perp_over_full_ratio_band": [PERP_RATIO_LOW, PERP_RATIO_HIGH],
        },
        "random_null_reference": {
            "source": "results/decomposition_ci_null/null_distribution.json (K=200, N=100)",
            "mean": NULL_MEAN, "std": NULL_STD,
            "ci95_low": NULL_CI_LOW, "ci95_high": NULL_CI_HIGH,
        },
        "full_shift_mean": full_mean,
        "full_shift_std": full_std,
        "table": [
            {
                "seed": int(s),
                "auroc": per_seed[str(s)]["probe"]["auroc"],
                "cos_with_action": per_seed[str(s)]["cos_with_action"],
                "cos_with_original": per_seed[str(s)]["cos_with_original"],
                "full_shift": full_mean,
                "parallel_shift": per_seed[str(s)]["parallel_shift_mean"],
                "perp_shift": per_seed[str(s)]["perp_shift_mean"],
                "perp_over_full_ratio": per_seed[str(s)]["perp_over_full_ratio"],
                "pattern_holds": per_seed[str(s)]["pattern_holds"],
            } for s in SEEDS
        ],
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "per_seed_results.json"), "w") as f:
        json.dump({"per_seed": per_seed,
                   "sample_ids": sample_ids,
                   "baseline_margins": base_m.tolist(),
                   "full_shifts": full_sh.tolist()}, f, indent=2)

    # Save the 10 seed direction vectors for reproducibility
    np.savez(os.path.join(args.output_dir, "seed_directions.npz"),
             ev_orig=ev_orig, action=action,
             **{f"ev_seed_{s}": seed_dirs[s] for s in SEEDS})

    write_report(args.output_dir, summary, per_seed)
    print(f"[save] summary.json, per_seed_results.json, seed_directions.npz, report.md")


def write_report(out_dir, summary, per_seed):
    lines = []
    lines.append("# Probe Robustness Sweep — Evidence Direction Across 10 Seeds\n")
    lines.append(f"**Verdict:** {summary['verdict']} ({summary['n_pass']}/{summary['n_seeds']} seeds reproduce the dissociation pattern)\n")
    lines.append(f"- Pass threshold: PASS ≥ 9, BORDERLINE 7–8, FAIL < 7\n")
    lines.append(f"- Layer: L{summary['layer']}, ρ={summary['rho']}, hidden_rms={summary['hidden_rms']}, N={summary['n_samples']} prompts\n")
    lines.append(f"- Reference random null (K=200, N=100): mean={summary['random_null_reference']['mean']:+.4f},"
                 f" 95% CI [{summary['random_null_reference']['ci95_low']:+.4f}, {summary['random_null_reference']['ci95_high']:+.4f}]\n")
    lines.append(f"- Reference full-shift (this run): {summary['full_shift_mean']:+.4f} ± {summary['full_shift_std']:.4f}\n")
    lines.append("\n## Pass criteria (per seed)\n")
    lines.append("- parallel mean shift ∈ random null 95% CI "
                 f"[{summary['criteria']['parallel_in_null_ci_95'][0]:+.4f}, "
                 f"{summary['criteria']['parallel_in_null_ci_95'][1]:+.4f}]\n")
    lines.append(f"- perpendicular / full ratio ∈ "
                 f"[{summary['criteria']['perp_over_full_ratio_band'][0]:.2f}, "
                 f"{summary['criteria']['perp_over_full_ratio_band'][1]:.2f}]\n")
    lines.append("\n## Per-seed table\n")
    lines.append("| seed | AUROC | cos(ev,act) | cos(ev,ev_orig) | full | parallel | perp | perp/full | holds |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|\n")
    for row in summary["table"]:
        lines.append(f"| {row['seed']} | {row['auroc']:.3f} | {row['cos_with_action']:+.4f} |"
                     f" {row['cos_with_original']:+.4f} | {row['full_shift']:+.4f} |"
                     f" {row['parallel_shift']:+.4f} | {row['perp_shift']:+.4f} |"
                     f" {row['perp_over_full_ratio']:.3f} |"
                     f" {'✅' if row['pattern_holds'] else '❌'} |\n")
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
