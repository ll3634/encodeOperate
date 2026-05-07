#!/usr/bin/env python3
"""Hypothesis (2) — Llama-3.1-8B multi-position evidence representation.

Tests whether Llama's E5 instrument failure (last-token AB=0.97, p=0.91) is
fixed by mean-pooling residuals over (a) Observation tokens or (b) the last
K=10 input tokens.

Pipeline (Llama only, single model load):
  1. Re-collect step-1 hidden states at L24 (evidence) + L28 (action) with
     full-sequence capture; compute three pooled vectors per prompt:
       last     : last-token (sanity reproduction of existing protocol)
       obs      : mean over Observation: body tokens
       lastK10  : mean over the last 10 input tokens
  2. Re-collect paired-corruption hidden states (Group A/B/C, N=200) at L24+L28
     with the same three pooling strategies, both clean and corrupted.
  3. For each pool ∈ {last, obs, lastK10}:
       - Train evidence probe on step1 pool@L24 (5-fold CV, L_loose labels) →
         AUROC + new evidence_dir
       - E5: project paired Δh@L24 (same pool) onto new evidence_dir →
         AB ratio + MW p (vs B group)
       - If E5 passes (AB > 1.2 AND p < 0.05): also project paired Δh@L28
         (same pool) onto existing PopQA-derived L28 last-token action_dir
         → action AB ratio + MW p
  4. Output → results/llama_multiposition_probe/{summary.json, README.md}

Llama-only. Other models untouched. ~10–30 min wall.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from agent.prompts import PromptBuilder  # noqa: E402
from steering.hook_utils import get_model_layers  # noqa: E402
from scripts.cross_model_full import apply_chat_template_safe  # noqa: E402
from scripts.paired_corruption_analysis import (  # noqa: E402
    select_samples, make_corrupted_obs,
)

SEED = 20260503
N_BOOT = 10000
MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct"
LLAMA_DIR = ROOT / "results" / "cross_model_llama31_v2"
OUT_DIR = ROOT / "results" / "llama_multiposition_probe"
LABELS_PATH = ROOT / "results" / "phase1_probe" / "labels.jsonl"
BASELINE_PATH = ROOT / "results" / "l20_rho020_n500" / "baseline_results.jsonl"
HOTPOTQA_PATH = ROOT / "data" / "hotpotqa" / "hotpot_dev_distractor_v1.json"
N_PAIRS = 200
LAYERS = (24, 28)
K_LAST = 10
POOLS = ("last", "obs", "lastK10")


def find_obs_token_range(prompt, obs_text, tokenizer, n_input_tokens):
    """Return (start, end) token indices spanning the Observation: body."""
    body_start_char = prompt.rfind("Observation: ")
    if body_start_char < 0:
        return None
    body_start_char += len("Observation: ")
    body_end_char = body_start_char + len(obs_text)
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    if len(offsets) != n_input_tokens:
        return None  # tokenization mismatch; skip
    obs_idx = [i for i, (s, e) in enumerate(offsets)
               if s >= body_start_char and e <= body_end_char and e > s]
    if not obs_idx:
        return None
    return (obs_idx[0], obs_idx[-1] + 1)


def pool_vectors(H_full, obs_range):
    """H_full: (T, D) numpy. Return dict[pool]→(D,) float32."""
    T = H_full.shape[0]
    out = {"last": H_full[-1].astype(np.float32),
           "lastK10": H_full[max(0, T - K_LAST):].mean(0).astype(np.float32)}
    if obs_range is None:
        out["obs"] = out["last"]   # fallback (flagged by caller)
    else:
        s, e = obs_range
        out["obs"] = H_full[s:e].mean(0).astype(np.float32)
    return out


def forward_capture(model, tokenizer, prompt, layers_obj, device, layer_ids):
    """One forward pass; return dict[layer_id]→(T, D) numpy and input_ids tensor."""
    captured = {}

    def make_hook(li):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[li] = h[0].detach().float().cpu().numpy()
        return fn

    handles = [layers_obj[li].register_forward_hook(make_hook(li))
               for li in layer_ids]
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model(input_ids)
    for h in handles:
        h.remove()
    return captured, input_ids


def cv_auroc(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, baccs = [], []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        p = LogisticRegression(class_weight="balanced", C=1.0,
                               max_iter=2000, solver="lbfgs", random_state=42)
        p.fit(X_tr, y[tr])
        aucs.append(roc_auc_score(y[te], p.predict_proba(X_te)[:, 1]))
        baccs.append(balanced_accuracy_score(y[te], p.predict(X_te)))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(baccs))


def fit_probe_direction(X, y):
    sc = StandardScaler()
    X_s = sc.fit_transform(X)
    p = LogisticRegression(class_weight="balanced", C=1.0,
                           max_iter=2000, solver="lbfgs", random_state=42)
    p.fit(X_s, y)
    w = p.coef_[0] / sc.scale_
    return (w / (np.linalg.norm(w) + 1e-12)).astype(np.float32)


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


def ab_stats(d_proj_A, d_proj_B):
    a = np.asarray(d_proj_A, np.float64)
    b = np.asarray(d_proj_B, np.float64)
    gmA, gmB = geom_median(a), geom_median(b)
    ratio = gmA / gmB if gmB > 0 else float("nan")
    lo, hi = lognormal_boot_ratio_ci(a, b)
    mw2 = mannwhitneyu(a, b, alternative="two-sided")
    return {"gm_A": float(gmA), "gm_B": float(gmB),
            "AB_ratio": float(ratio), "CI95": [lo, hi],
            "MW_p_two": float(mw2.pvalue),
            "n_A": int(len(a)), "n_B": int(len(b))}


# ── Step-1 collection ────────────────────────────────────────────────────────

def collect_step1(model, tokenizer):
    layers_obj = get_model_layers(model)
    device = next(model.parameters()).device

    label_recs = [json.loads(l) for l in open(LABELS_PATH)]
    bl = {}
    with open(BASELINE_PATH) as f:
        for line in f:
            ep = json.loads(line)
            bl[ep["sample_id"]] = ep

    pb = PromptBuilder(tools=["search", "calculator"])
    pooled_X = {p: [] for p in POOLS}
    labels, sids, n_obs_fallback = [], [], 0

    for i, ld in enumerate(label_recs):
        ep = bl.get(ld["sample_id"])
        if not ep or not ep.get("steps"):
            continue
        s0 = ep["steps"][0]
        if s0.get("action") != "search" or not s0.get("observation"):
            continue
        obs_text = s0["observation"]
        steps = [{"action": "search", "action_input": s0["action_input"],
                  "observation": obs_text}]
        messages = pb.build_full_prompt(ld["question"], steps)
        prompt = apply_chat_template_safe(tokenizer, messages)

        cap, ids = forward_capture(model, tokenizer, prompt, layers_obj, device, [LAYERS[0]])
        T = ids.shape[1]
        rng_obs = find_obs_token_range(prompt, obs_text, tokenizer, T)
        if rng_obs is None:
            n_obs_fallback += 1
        pooled = pool_vectors(cap[LAYERS[0]], rng_obs)
        for p in POOLS:
            pooled_X[p].append(pooled[p])
        labels.append(int(ld["label"]))
        sids.append(ld["sample_id"])

        if (i + 1) % 100 == 0:
            print(f"  step1 [{i+1}/{len(label_recs)}]  obs_fallback={n_obs_fallback}",
                  flush=True)

    print(f"  step1 done: N={len(labels)}, obs_fallback={n_obs_fallback}", flush=True)
    return ({p: np.asarray(pooled_X[p], np.float32) for p in POOLS},
            np.asarray(labels, np.int32), sids, n_obs_fallback)


# ── Paired corruption collection ─────────────────────────────────────────────

def build_pair_prompt(tokenizer, sample, observation):
    pb = PromptBuilder(tools=["search", "calculator"])
    steps = [{"action": "search", "action_input": sample["step0_query"],
              "observation": observation[:1500]}]
    messages = pb.build_full_prompt(sample["question"], steps)
    return apply_chat_template_safe(tokenizer, messages)


def collect_paired(model, tokenizer):
    import random
    layers_obj = get_model_layers(model)
    device = next(model.parameters()).device

    samples = select_samples(str(BASELINE_PATH), str(HOTPOTQA_PATH),
                             n=N_PAIRS, seed=42)
    print(f"  selected {len(samples)} paired samples", flush=True)

    out = {g: {"clean": {p: {li: [] for li in LAYERS} for p in POOLS},
               "corrupted": {p: {li: [] for li in LAYERS} for p in POOLS}}
           for g in ("A", "B", "C")}
    n_obs_fallback = 0

    for gi, group in enumerate(("A", "B", "C")):
        for i, sample in enumerate(samples):
            rng = random.Random(42 + gi * 10000)
            for j in range(i):
                make_corrupted_obs(samples[j], group, rng)
            clean_obs, corr_obs = make_corrupted_obs(sample, group, rng)

            for kind, obs in (("clean", clean_obs), ("corrupted", corr_obs)):
                prompt = build_pair_prompt(tokenizer, sample, obs)
                cap, ids = forward_capture(model, tokenizer, prompt, layers_obj,
                                           device, list(LAYERS))
                T = ids.shape[1]
                rng_obs = find_obs_token_range(
                    prompt, obs[:1500], tokenizer, T)
                if rng_obs is None:
                    n_obs_fallback += 1
                for li in LAYERS:
                    pooled = pool_vectors(cap[li], rng_obs)
                    for p in POOLS:
                        out[group][kind][p][li].append(pooled[p])

            if (i + 1) % 50 == 0:
                print(f"  pair [{group} {i+1}/{N_PAIRS}]  obs_fallback={n_obs_fallback}",
                      flush=True)
        print(f"  group {group} done", flush=True)

    # Stack to arrays: out[g][kind][p][li] -> (N_PAIRS, D)
    for g in ("A", "B", "C"):
        for kind in ("clean", "corrupted"):
            for p in POOLS:
                for li in LAYERS:
                    out[g][kind][p][li] = np.asarray(out[g][kind][p][li], np.float32)
    return out, n_obs_fallback


# ── Analysis & main ─────────────────────────────────────────────────────────

def analyse(step1_X, step1_y, paired, action_dir_L28):
    rows = []
    for p in POOLS:
        # Probe
        X = step1_X[p]
        y = step1_y
        auroc, auroc_std, bacc = cv_auroc(X, y)
        evi_dir = fit_probe_direction(X, y)

        # E5 on L24 with this evidence_dir
        def proj_delta(group, layer, pool):
            hc = paired[group]["clean"][pool][layer]
            hx = paired[group]["corrupted"][pool][layer]
            return np.abs((hc - hx) @ evi_dir)
        d_A = proj_delta("A", LAYERS[0], p)
        d_B = proj_delta("B", LAYERS[0], p)
        d_C = proj_delta("C", LAYERS[0], p)
        e5 = ab_stats(d_A, d_B)
        e5["gm_C"] = float(geom_median(d_C.astype(np.float64)))

        # Conditional action A/B at L28
        action_ab = None
        if e5["AB_ratio"] > 1.2 and e5["MW_p_two"] < 0.05:
            def proj_delta_act(group):
                hc = paired[group]["clean"][p][LAYERS[1]]
                hx = paired[group]["corrupted"][p][LAYERS[1]]
                return np.abs((hc - hx) @ action_dir_L28)
            a_A = proj_delta_act("A")
            a_B = proj_delta_act("B")
            a_C = proj_delta_act("C")
            action_ab = ab_stats(a_A, a_B)
            action_ab["gm_C"] = float(geom_median(a_C.astype(np.float64)))

        rows.append({
            "pool": p,
            "evidence_probe": {
                "layer": LAYERS[0],
                "auroc_mean": auroc, "auroc_std": auroc_std, "bacc_mean": bacc,
                "n": int(len(y)),
                "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
            },
            "E5_L24_new_evidence_dir": e5,
            "action_AB_L28_existing_action_dir": action_ab,
        })
    return rows


def write_readme(rows, out_dir, meta):
    lines = ["# Hypothesis (2) — Llama-3.1-8B multi-position evidence probe\n"]
    lines.append(f"spec_version: {meta['spec_version']}")
    lines.append(f"model: {meta['model']}  layers: L{LAYERS[0]} (evidence) / L{LAYERS[1]} (action)")
    lines.append(f"n_step1: {meta['n_step1']}  n_pairs_per_group: {meta['n_pairs']}")
    lines.append(f"step1 obs_fallback: {meta['n_obs_fallback_step1']}  "
                 f"paired obs_fallback: {meta['n_obs_fallback_paired']}\n")
    lines.append("## Pool definitions")
    lines.append("- **last**: residual at last input token (reproduces existing protocol)")
    lines.append("- **obs**: mean over `Observation: <body>` token positions")
    lines.append(f"- **lastK10**: mean over the last K={K_LAST} input tokens\n")
    lines.append("## Evidence probe (5-fold CV, L_loose labels) and E5 positive control")
    lines.append("| pool | AUROC | E5 AB | E5 CI95 | E5 p (two-sided) | E5 verdict |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        ev = r["evidence_probe"]
        e = r["E5_L24_new_evidence_dir"]
        ok = (e["AB_ratio"] > 1.2 and e["MW_p_two"] < 0.05)
        verdict = "PASS" if ok else "FAIL"
        lines.append(f"| {r['pool']} | {ev['auroc_mean']:.3f}±{ev['auroc_std']:.3f} | "
                     f"{e['AB_ratio']:.3f} | [{e['CI95'][0]:.3f}, {e['CI95'][1]:.3f}] | "
                     f"{e['MW_p_two']:.4g} | {verdict} |")
    lines.append("\n## Action A/B at L28 (existing PopQA action_dir; only computed if E5 passed)")
    lines.append("| pool | action AB | action CI95 | action p (two-sided) |")
    lines.append("|---|---|---|---|")
    for r in rows:
        a = r["action_AB_L28_existing_action_dir"]
        if a is None:
            lines.append(f"| {r['pool']} | n/a (E5 failed) | n/a | n/a |")
        else:
            lines.append(f"| {r['pool']} | {a['AB_ratio']:.3f} | "
                         f"[{a['CI95'][0]:.3f}, {a['CI95'][1]:.3f}] | "
                         f"{a['MW_p_two']:.4g} |")
    lines.append(f"\n## Reference: last-token E5 (original Phase-α): AB=0.972, p=0.91")
    any_pass = any((r["E5_L24_new_evidence_dir"]["AB_ratio"] > 1.2
                    and r["E5_L24_new_evidence_dir"]["MW_p_two"] < 0.05)
                   for r in rows)
    lines.append(f"\n## Verdict\n")
    if any_pass:
        lines.append("**Hypothesis (2) supported under at least one pooling strategy.** "
                     "Llama's E5 instrument failure is fixed by mean-pooling. "
                     "Routing results should be re-evaluated with the corrected instrument; "
                     "this report does NOT re-run routing — see follow-up decision.")
    else:
        lines.append("**Hypothesis (2) NOT supported.** "
                     "E5 fails under both pooling strategies (`obs` and `lastK10`) at L24. "
                     "Llama remains instrument-suspended.")
    with open(out_dir / "README.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"loading {MODEL_ID}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    print("\n=== Step 1: collecting step-1 hidden states ===", flush=True)
    step1_X, step1_y, step1_sids, n_fb1 = collect_step1(model, tokenizer)

    print(f"\n=== Step 2: collecting paired-corruption (N={N_PAIRS}/group) ===", flush=True)
    paired, n_fb2 = collect_paired(model, tokenizer)

    # Existing PopQA action_dir at L28 (last-token derived)
    z = np.load(LLAMA_DIR / "per_sample.npz", allow_pickle=False)
    action_dir_L28 = z["action_dir"].astype(np.float32)
    action_dir_L28 = action_dir_L28 / (np.linalg.norm(action_dir_L28) + 1e-12)
    assert int(z["peak_act_layer"]) == LAYERS[1], (
        f"per_sample peak_act_layer={int(z['peak_act_layer'])} != expected {LAYERS[1]}")

    print("\n=== Step 3: probe + E5 + action A/B per pool ===", flush=True)
    rows = analyse(step1_X, step1_y, paired, action_dir_L28)

    meta = {
        "spec_version": "llama-multiposition-probe-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_ID,
        "seed": SEED, "n_bootstrap": N_BOOT,
        "n_step1": int(len(step1_y)),
        "n_pairs": N_PAIRS,
        "K_last": K_LAST,
        "evidence_layer": LAYERS[0], "action_layer": LAYERS[1],
        "n_obs_fallback_step1": int(n_fb1),
        "n_obs_fallback_paired": int(n_fb2),
        "label": "L_loose (n_sf_retrieved >= 1)",
        "action_dir_source": str(LLAMA_DIR / "per_sample.npz") + " key=action_dir (L28 last-token PopQA)",
    }
    summary = {**meta, "rows": rows}
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    write_readme(rows, OUT_DIR, meta)

    # Console table
    print("\n" + "=" * 78)
    print(f"{'pool':<10} {'AUROC':<8} {'E5_AB':>8} {'E5_p':>10} {'verdict':<6} "
          f"{'act_AB':>8} {'act_p':>10}")
    for r in rows:
        ev, e = r["evidence_probe"], r["E5_L24_new_evidence_dir"]
        a = r["action_AB_L28_existing_action_dir"]
        ok = (e["AB_ratio"] > 1.2 and e["MW_p_two"] < 0.05)
        if a is None:
            print(f"{r['pool']:<10} {ev['auroc_mean']:.3f}    {e['AB_ratio']:>8.3f} "
                  f"{e['MW_p_two']:>10.4g} {'PASS' if ok else 'FAIL':<6} "
                  f"{'n/a':>8} {'n/a':>10}")
        else:
            print(f"{r['pool']:<10} {ev['auroc_mean']:.3f}    {e['AB_ratio']:>8.3f} "
                  f"{e['MW_p_two']:>10.4g} {'PASS' if ok else 'FAIL':<6} "
                  f"{a['AB_ratio']:>8.3f} {a['MW_p_two']:>10.4g}")
    print("=" * 78)
    print(f"\nReference (existing per_sample.npz, last-token L24 E5): AB=0.972, p=0.91")
    print(f"Outputs: {OUT_DIR}/summary.json, {OUT_DIR}/README.md")


if __name__ == "__main__":
    main()
