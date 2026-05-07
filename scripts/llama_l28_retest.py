#!/usr/bin/env python3
"""Llama-3.1-8B paired corruption re-run at L28 (peak_act_layer).

Closes the L24-vs-L28 caveat from `results/action_ab_revisit/`.  The original
paired corruption hidden states were captured at L24 (peak_evi_layer).
`action_dir` is from L28 (peak_act_layer); cross-layer projection might
understate routing.  This script re-runs the paired corruption at L28 and
recomputes E5 (both directions), action AB, and S/N at L28, side-by-side
with the L24 results loaded from existing artefacts.

Pipeline (single model load, ~5 min):
  1. Collect step-1 hidden states at L28 (last token, sufficiency labels)
  2. Collect paired corruption at L28 (last token, A/B/C, N=200)
  3. Train cross-sectional evi_dir at L28 from step-1
  4. Train paired-induced direction at L28 from paired-A (5-fold CV)
  5. Compute E5 (cross-sectional + paired-induced), action AB, S/N at L28
  6. Side-by-side L24 vs L28 report

Decision rule:
  - L28 action AB > 1.2 AND p < 0.05 → MAJOR FINDING (Llama routing is
    layer-dependent; promotes Llama back to routing-positive at the
    appropriate layer)
  - L28 action AB null → informative null confirmed across both layers

Output: results/llama_l28_retest/{summary.json, README.md}
"""
import json, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from agent.prompts import PromptBuilder  # noqa: E402
from steering.hook_utils import get_model_layers  # noqa: E402
from scripts.cross_model_full import apply_chat_template_safe  # noqa: E402
from scripts.paired_corruption_analysis import select_samples, make_corrupted_obs  # noqa: E402

SEED = 20260503
N_BOOT = 1000
N_FOLDS = 5
N_PAIRS = 200
LAYER = 28                                # action peak layer for Llama
MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct"
LLAMA_DIR = ROOT / "results" / "cross_model_llama31_v2"
OUT_DIR = ROOT / "results" / "llama_l28_retest"
LABELS_PATH = ROOT / "results" / "phase1_probe" / "labels.jsonl"
BASELINE_PATH = ROOT / "results" / "l20_rho020_n500" / "baseline_results.jsonl"
HOTPOTQA_PATH = ROOT / "data" / "hotpotqa" / "hotpot_dev_distractor_v1.json"


# ── helpers (copies of established protocol) ─────────────────────────────────

def geom_median(x, n_iter=200, eps=1e-9):
    y = float(np.median(x))
    for _ in range(n_iter):
        d = np.maximum(np.abs(x - y), eps)
        w = 1.0 / d
        y_new = float(np.sum(w * x) / np.sum(w))
        if abs(y_new - y) < eps:
            break
        y = y_new
    return y


def lognormal_boot_ratio_ci(a, b, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    na, nb = len(a), len(b)
    lr = np.empty(n_boot)
    for i in range(n_boot):
        lr[i] = (np.log(geom_median(a[rng.integers(0, na, na)]))
                 - np.log(geom_median(b[rng.integers(0, nb, nb)])))
    return float(np.exp(np.quantile(lr, 0.025))), float(np.exp(np.quantile(lr, 0.975)))


def ab_stats(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    gmA, gmB = geom_median(a), geom_median(b)
    ratio = gmA / gmB if gmB > 0 else float("nan")
    lo, hi = lognormal_boot_ratio_ci(a, b)
    mw = mannwhitneyu(a, b, alternative="two-sided")
    return {"gm_A": float(gmA), "gm_B": float(gmB),
            "AB_ratio": float(ratio), "CI95": [lo, hi],
            "MW_p_two": float(mw.pvalue), "n_A": int(len(a)), "n_B": int(len(b))}


def cv_auroc(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, baccs = [], []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr]); X_te = sc.transform(X[te])
        p = LogisticRegression(class_weight="balanced", C=1.0,
                               max_iter=2000, solver="lbfgs", random_state=42).fit(X_tr, y[tr])
        aucs.append(roc_auc_score(y[te], p.predict_proba(X_te)[:, 1]))
        baccs.append(balanced_accuracy_score(y[te], p.predict(X_te)))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(baccs))


def fit_dir(X, y, seed=SEED):
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    lr = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                            solver="lbfgs", random_state=seed).fit(Xs, y)
    w = lr.coef_[0] / (sc.scale_ + 1e-12)
    return (w / (np.linalg.norm(w) + 1e-12)).astype(np.float64)


def extract_last_token(model, tokenizer, prompt, layer_idx):
    layers_obj = get_model_layers(model)
    device = next(model.parameters()).device
    captured = {}
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h[0, -1, :].detach().float().cpu().numpy()
    handle = layers_obj[layer_idx].register_forward_hook(hook)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model(input_ids)
    handle.remove()
    return captured["h"]


def build_prompt_for_step(tokenizer, question, query, observation):
    pb = PromptBuilder(tools=["search", "calculator"])
    steps = [{"action": "search", "action_input": query,
              "observation": observation[:1500]}]
    return apply_chat_template_safe(tokenizer, pb.build_full_prompt(question, steps))


# ── Collection ────────────────────────────────────────────────────────────────

def collect_step1(model, tokenizer):
    """Return X (N,D), y (N,), sids list at LAYER (last token)."""
    label_recs = [json.loads(l) for l in open(LABELS_PATH)]
    bl = {}
    with open(BASELINE_PATH) as f:
        for line in f:
            ep = json.loads(line)
            bl[ep["sample_id"]] = ep
    Xs, ys, sids = [], [], []
    for i, ld in enumerate(label_recs):
        ep = bl.get(ld["sample_id"])
        if not ep or not ep.get("steps"):
            continue
        s0 = ep["steps"][0]
        if s0.get("action") != "search" or not s0.get("observation"):
            continue
        prompt = build_prompt_for_step(tokenizer, ld["question"],
                                       s0["action_input"], s0["observation"])
        h = extract_last_token(model, tokenizer, prompt, LAYER)
        Xs.append(h); ys.append(int(ld["label"])); sids.append(ld["sample_id"])
        if (i + 1) % 100 == 0:
            print(f"  step1 [{i+1}/{len(label_recs)}]", flush=True)
    print(f"  step1 done: N={len(Xs)}", flush=True)
    return np.asarray(Xs, np.float32), np.asarray(ys, np.int32), sids


def collect_paired(model, tokenizer):
    """Return dict: group→{'clean':(N,D),'corrupted':(N,D)} at LAYER."""
    import random
    samples = select_samples(str(BASELINE_PATH), str(HOTPOTQA_PATH),
                             n=N_PAIRS, seed=42)
    print(f"  selected {len(samples)} paired samples", flush=True)
    out = {g: {"clean": [], "corrupted": []} for g in ("A", "B", "C")}
    for gi, group in enumerate(("A", "B", "C")):
        for i, sample in enumerate(samples):
            rng = random.Random(42 + gi * 10000)
            for j in range(i):
                make_corrupted_obs(samples[j], group, rng)
            clean_obs, corr_obs = make_corrupted_obs(sample, group, rng)
            for kind, obs in (("clean", clean_obs), ("corrupted", corr_obs)):
                prompt = build_prompt_for_step(tokenizer, sample["question"],
                                               sample["step0_query"], obs)
                h = extract_last_token(model, tokenizer, prompt, LAYER)
                out[group][kind].append(h)
            if (i + 1) % 50 == 0:
                print(f"  pair [{group} {i+1}/{N_PAIRS}]", flush=True)
        print(f"  group {group} done", flush=True)
    return {g: {k: np.asarray(v, np.float32) for k, v in d.items()}
            for g, d in out.items()}


# ── Analysis ─────────────────────────────────────────────────────────────────

def paired_induced_oof(paired, evi_cross, seed=SEED):
    ph_c = np.stack([paired[g]["clean"] for g in ("A", "B", "C")])
    ph_x = np.stack([paired[g]["corrupted"] for g in ("A", "B", "C")])
    N = ph_c.shape[1]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    out_A = np.empty(N); out_B = np.empty(N); out_C = np.empty(N)
    cos_pc = []
    evi_cross_u = evi_cross / (np.linalg.norm(evi_cross) + 1e-12)
    for tr, te in skf.split(np.arange(N), np.zeros(N)):
        X = np.concatenate([ph_c[0, tr], ph_x[0, tr]], axis=0)
        y = np.concatenate([np.ones(len(tr)), np.zeros(len(tr))])
        w = fit_dir(X, y, seed=seed)
        cos_pc.append(float(w @ evi_cross_u))
        for arr, gi in ((out_A, 0), (out_B, 1), (out_C, 2)):
            d = (ph_c[gi, te] - ph_x[gi, te]) @ w
            arr[te] = np.abs(d)
    return out_A, out_B, out_C, cos_pc


def signal_noise(paired, action_dir, evi_paired_dir):
    ph_c = paired["A"]["clean"].astype(np.float64)
    ph_x = paired["A"]["corrupted"].astype(np.float64)
    dh = ph_c - ph_x
    norms = np.linalg.norm(dh, axis=1)
    D = ph_c.shape[-1]
    proj_act = np.abs(dh @ action_dir)
    proj_paired = np.abs(dh @ evi_paired_dir)
    floor = norms / np.sqrt(D)
    return {"median_dh_norm": float(np.median(norms)),
            "median_proj_action": float(np.median(proj_act)),
            "median_proj_paired_evi": float(np.median(proj_paired)),
            "median_noise_floor": float(np.median(floor)),
            "SN_action_over_floor": float(np.median(proj_act) / max(np.median(floor), 1e-12)),
            "SN_paired_evi_over_floor": float(np.median(proj_paired) / max(np.median(floor), 1e-12))}


def load_l24_metrics():
    """Pull L24 reference numbers from existing artefacts."""
    ref = {}
    pi = json.load(open(ROOT / "results" / "paired_induced_e5" / "summary.json"))
    aar = json.load(open(ROOT / "results" / "action_ab_revisit" / "summary.json"))
    for r in pi["rows"]:
        if r["model"] == "llama31_8b":
            ref["E5_paired_induced"] = r["E5_paired_induced_dir"]
            ref["E5_cross_sectional"] = r["E5_cross_sectional_dir_reference"]
            ref["cos_paired_to_cross_median"] = r["cos_paired_to_cross_median"]
    for r in aar["rows"]:
        if r["model"] == "llama31_8b":
            ref["action_AB"] = r["action_AB_geom"]
            ref["signal_noise"] = r["signal_noise"]
            ref["cos_triple"] = r["cos_triple"]
    return ref


def write_readme(out):
    L28 = out["L28"]
    l24 = out["L24_reference"]
    e5p_28 = L28["E5_paired_induced"]; e5c_28 = L28["E5_cross_sectional"]
    e5p_24 = l24.get("E5_paired_induced", {}); e5c_24 = l24.get("E5_cross_sectional", {})
    aa_28 = L28["action_AB"]; aa_24 = l24.get("action_AB", {})
    sn_28 = L28["signal_noise"]; sn_24 = l24.get("signal_noise", {})
    ct_28 = L28["cos_triple"]; ct_24 = l24.get("cos_triple", {})

    def fmt(v, p="{:.3f}"):
        try: return p.format(float(v))
        except Exception: return "n/a"

    md = []
    md.append("# Llama-3.1-8B paired corruption — L24 vs L28 side-by-side")
    md.append("")
    md.append(f"spec_version: {out['spec_version']}")
    md.append(f"generated_at: {out['generated_at']}")
    md.append(f"model: {out['model_id']}")
    md.append(f"n_pairs={out['n_pairs']}, n_step1={out['n_step1']}")
    md.append("")
    md.append("Closes the L24-vs-L28 caveat from `results/action_ab_revisit/`. The")
    md.append("paired corruption was previously captured at L24 (peak_evi_layer);")
    md.append("`action_dir` lives at L28 (peak_act_layer). This script re-collects")
    md.append("paired corruption at L28 with the same protocol and recomputes all")
    md.append("metrics for direct comparison.")
    md.append("")
    md.append(f"## Step-1 evidence probe @ L28 (5-fold CV)")
    md.append("")
    sp = out["step1_probe_at_L28"]
    md.append(f"AUROC = {sp['AUROC_5fold']:.4f} ± {sp['AUROC_sd']:.4f}, BalAcc = {sp['BalAcc']:.4f}")
    md.append("")
    md.append("## Side-by-side metrics")
    md.append("")
    md.append("| metric | L24 | L28 |")
    md.append("|---|---|---|")
    md.append(f"| **E5 cross-sectional** AB | {fmt(e5c_24.get('AB_ratio'))} | {fmt(e5c_28['AB_ratio'])} |")
    md.append(f"| E5 cross-sectional p (MW two) | {fmt(e5c_24.get('MW_p_two'),'{:.2e}')} | {fmt(e5c_28['MW_p_two'],'{:.2e}')} |")
    md.append(f"| **E5 paired-induced** AB | {fmt(e5p_24.get('AB_ratio'))} | {fmt(e5p_28['AB_ratio'])} |")
    md.append(f"| E5 paired-induced p (MW two) | {fmt(e5p_24.get('MW_p_two'),'{:.2e}')} | {fmt(e5p_28['MW_p_two'],'{:.2e}')} |")
    md.append(f"| **action AB** | {fmt(aa_24.get('AB_ratio'))} | {fmt(aa_28['AB_ratio'])} |")
    md.append(f"| action AB p (MW two) | {fmt(aa_24.get('MW_p_two'),'{:.2e}')} | {fmt(aa_28['MW_p_two'],'{:.2e}')} |")
    md.append(f"| action AB CI95 | [{fmt(aa_24.get('CI95',[None,None])[0])},{fmt(aa_24.get('CI95',[None,None])[1])}] | [{fmt(aa_28['CI95'][0])},{fmt(aa_28['CI95'][1])}] |")
    md.append(f"| ‖Δh‖ median | {fmt(sn_24.get('median_dh_norm'))} | {fmt(sn_28['median_dh_norm'])} |")
    md.append(f"| |Δh·action| median | {fmt(sn_24.get('median_proj_action'))} | {fmt(sn_28['median_proj_action'])} |")
    md.append(f"| noise floor (‖Δh‖/√D) | {fmt(sn_24.get('median_noise_floor'))} | {fmt(sn_28['median_noise_floor'])} |")
    md.append(f"| **S/N(action)** | {fmt(sn_24.get('SN_action_over_floor'))} | {fmt(sn_28['SN_action_over_floor'])} |")
    md.append(f"| S/N(evi_paired) | {fmt(sn_24.get('SN_paired_evi_over_floor'))} | {fmt(sn_28['SN_paired_evi_over_floor'])} |")
    md.append(f"| cos(evi_cross, action) | {fmt(ct_24.get('evi_cross__action'),'{:+.4f}')} | {fmt(ct_28['evi_cross__action'],'{:+.4f}')} |")
    md.append(f"| cos(evi_paired, action) | {fmt(ct_24.get('evi_paired__action'),'{:+.4f}')} | {fmt(ct_28['evi_paired__action'],'{:+.4f}')} |")
    md.append(f"| cos(evi_paired, evi_cross) | {fmt(ct_24.get('evi_paired__evi_cross'),'{:+.4f}')} | {fmt(ct_28['evi_paired__evi_cross'],'{:+.4f}')} |")
    md.append("")
    md.append(f"## Verdict: **{out['verdict']}**")
    md.append("")
    md.append("Decision rule: action AB > 1.2 AND MW p < 0.05 at L28 → MAJOR FINDING")
    md.append("(layer-dependent routing); else informative null confirmed across both layers.")
    md.append("")
    md.append(f"Observed at L28: action AB = {aa_28['AB_ratio']:.3f}, p = {aa_28['MW_p_two']:.3g}.")
    md.append("")
    md.append("## Outputs")
    md.append("")
    md.append("- `results/llama_l28_retest/summary.json`")
    md.append("- `results/llama_l28_retest/README.md`")
    md.append("- `results/llama_l28_retest/per_sample_l28.npz`")
    (OUT_DIR / "README.md").write_text("\n".join(md) + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now(timezone.utc).isoformat()}] llama_l28_retest start", flush=True)

    # Load action_dir from existing artefacts (L28, PopQA-derived).
    dirs = np.load(LLAMA_DIR / "directions.npz")
    action_dir = dirs["action_dir"].astype(np.float64)
    action_dir = action_dir / (np.linalg.norm(action_dir) + 1e-12)
    L_evi_existing = int(dirs["L_evi"])
    L_act_existing = int(dirs["L_act"])
    assert L_act_existing == LAYER, f"action_dir layer mismatch: {L_act_existing} vs {LAYER}"
    print(f"  loaded action_dir from {LLAMA_DIR.name} (L_evi={L_evi_existing}, L_act={L_act_existing})", flush=True)

    # Load Llama-3.1-8B-Instruct.
    print(f"  loading {MODEL_ID} ...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print(f"  model loaded, dtype={next(model.parameters()).dtype}", flush=True)

    # 1. Step-1 collection at L28.
    print(f"[{datetime.now(timezone.utc).isoformat()}] collect step1 @ L{LAYER}", flush=True)
    X1, y1, sids1 = collect_step1(model, tokenizer)

    # 2. Paired corruption at L28.
    print(f"[{datetime.now(timezone.utc).isoformat()}] collect paired @ L{LAYER}", flush=True)
    paired = collect_paired(model, tokenizer)

    # Free GPU before analysis.
    del model
    torch.cuda.empty_cache()

    # 3. Cross-sectional evidence direction + AUROC at L28.
    print("  fitting cross-sectional evi_dir @ L28 ...", flush=True)
    auroc, auroc_sd, bacc = cv_auroc(X1, y1)
    evi_cross = fit_dir(X1, y1).astype(np.float64)
    evi_cross /= np.linalg.norm(evi_cross) + 1e-12

    # 4. Paired-induced direction (5-fold OOF) + cos to cross-sectional.
    print("  fitting paired-induced dir @ L28 (5-fold OOF) ...", flush=True)
    pi_A, pi_B, pi_C, cos_pc = paired_induced_oof(paired, evi_cross)
    e5_paired_induced = ab_stats(pi_A, pi_B)
    e5_paired_induced["gm_C"] = float(geom_median(np.asarray(pi_C, np.float64)))

    # 5. E5 cross-sectional at L28.
    e5_cross_A = np.abs((paired["A"]["clean"].astype(np.float64) - paired["A"]["corrupted"].astype(np.float64)) @ evi_cross)
    e5_cross_B = np.abs((paired["B"]["clean"].astype(np.float64) - paired["B"]["corrupted"].astype(np.float64)) @ evi_cross)
    e5_cross_C = np.abs((paired["C"]["clean"].astype(np.float64) - paired["C"]["corrupted"].astype(np.float64)) @ evi_cross)
    e5_cross = ab_stats(e5_cross_A, e5_cross_B)
    e5_cross["gm_C"] = float(geom_median(e5_cross_C))

    # 6. Action AB at L28.
    act_A = np.abs((paired["A"]["clean"].astype(np.float64) - paired["A"]["corrupted"].astype(np.float64)) @ action_dir)
    act_B = np.abs((paired["B"]["clean"].astype(np.float64) - paired["B"]["corrupted"].astype(np.float64)) @ action_dir)
    act_C = np.abs((paired["C"]["clean"].astype(np.float64) - paired["C"]["corrupted"].astype(np.float64)) @ action_dir)
    action_ab = ab_stats(act_A, act_B)
    action_ab["gm_C"] = float(geom_median(act_C))

    # Train a fitted paired-induced full-data direction for S/N (uses all 200; not for E5).
    full_X = np.concatenate([paired["A"]["clean"], paired["A"]["corrupted"]], axis=0)
    full_y = np.concatenate([np.ones(N_PAIRS), np.zeros(N_PAIRS)])
    evi_paired_full = fit_dir(full_X, full_y).astype(np.float64)
    evi_paired_full /= np.linalg.norm(evi_paired_full) + 1e-12
    sn = signal_noise(paired, action_dir, evi_paired_full)

    # Cosine triple for L28.
    cos_triple_l28 = {
        "evi_cross__action": float(evi_cross @ action_dir),
        "evi_paired__action": float(evi_paired_full @ action_dir),
        "evi_paired__evi_cross": float(evi_paired_full @ evi_cross),
    }

    # 7. Decision rule.
    ab_ratio = action_ab["AB_ratio"]
    p_val = action_ab["MW_p_two"]
    if ab_ratio > 1.2 and p_val < 0.05:
        verdict = "MAJOR_FINDING_layer_dependent_routing"
    else:
        verdict = "informative_null_confirmed_at_both_layers"

    # 8. Side-by-side payload.
    l24 = load_l24_metrics()
    out = {
        "spec_version": "llama-l28-retest-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "layer_under_test": LAYER,
        "n_pairs": N_PAIRS,
        "n_step1": int(len(y1)),
        "step1_probe_at_L28": {"AUROC_5fold": auroc, "AUROC_sd": auroc_sd, "BalAcc": bacc},
        "L28": {
            "E5_cross_sectional": e5_cross,
            "E5_paired_induced": e5_paired_induced,
            "cos_paired_to_cross_per_fold": cos_pc,
            "cos_paired_to_cross_median": float(np.median(cos_pc)),
            "action_AB": action_ab,
            "signal_noise": sn,
            "cos_triple": cos_triple_l28,
        },
        "L24_reference": l24,
        "verdict": verdict,
    }

    # Save raw arrays for reproducibility.
    np.savez_compressed(OUT_DIR / "per_sample_l28.npz",
                        step1_h=X1, step1_y=y1,
                        pair_h_clean=np.stack([paired[g]["clean"] for g in ("A", "B", "C")]),
                        pair_h_corrupted=np.stack([paired[g]["corrupted"] for g in ("A", "B", "C")]),
                        evi_cross_l28=evi_cross, evi_paired_full_l28=evi_paired_full,
                        action_dir_l28=action_dir)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {OUT_DIR / 'summary.json'}", flush=True)
    print(f"  VERDICT: {verdict}  (action AB={ab_ratio:.3f}, p={p_val:.3g})", flush=True)
    write_readme(out)


if __name__ == "__main__":
    main()
