#!/usr/bin/env python3
"""
Probe-S with clean sufficiency labels via synthetic augmentation.

For each 1-SF sample, construct:
  - 1-SF observation (insufficient): original observation with 1 supporting fact
  - 2-SF observation (sufficient):   inject missing SF into observation

Extract L20 activations for both, train probe, compare with Probe-R and action_dir.
"""
import os, sys, re, json, argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder
from steering.hook_utils import get_model_layers

# Reuse infrastructure from paired_corruption
from scripts.paired_corruption_analysis import (
    parse_observation, rebuild_observation, title_match,
    get_hotpot_paragraph_text, build_prompt, extract_l20_hidden,
)


def select_1sf_samples(baseline_path, hotpotqa_path, labels_path, max_n=None, seed=42):
    """Select samples with exactly 1 SF retrieved. Return list of dicts with
    both 1-SF and 2-SF observation text ready to go."""
    import random

    with open(hotpotqa_path) as f:
        hotpot_by_id = {s["_id"]: s for s in json.load(f)}

    lab_by_id = {json.loads(l)["sample_id"]: json.loads(l)
                 for l in open(labels_path)}

    candidates = []
    with open(baseline_path) as f:
        for line in f:
            ep = json.loads(line)
            sid = ep["sample_id"]
            hp = hotpot_by_id.get(sid)
            lab = lab_by_id.get(sid)
            if not hp or not lab:
                continue
            if lab["label"] != 1 or lab["n_sf_retrieved"] != 1:
                continue

            steps = ep.get("steps", [])
            s0 = steps[0] if steps else None
            if not s0 or s0.get("action") != "search" or not s0.get("observation"):
                continue

            obs = s0["observation"]
            sf_titles = list(set(t for t, _ in hp.get("supporting_facts", [])))
            entries = parse_observation(obs)
            if not entries:
                continue

            # Find which SF is in observation
            retrieved_sf = [st for st in sf_titles
                            if any(title_match(e["title"], st) for e in entries)]
            missing_sf = [st for st in sf_titles if st not in retrieved_sf]
            if len(retrieved_sf) != 1 or len(missing_sf) != 1:
                continue

            # Get missing SF text from HotpotQA context
            missing_text = get_hotpot_paragraph_text(hp["context"], missing_sf[0])
            if not missing_text or len(missing_text) < 20:
                continue

            # Get distractor titles (not supporting facts)
            all_ctx_dist = [t for t, _ in hp["context"]
                            if not any(title_match(t, st) for st in sf_titles)]
            if not all_ctx_dist:
                continue

            # Pick a distractor for the 1-SF condition
            dist_title = sorted(all_ctx_dist)[0]  # deterministic
            dist_text = get_hotpot_paragraph_text(hp["context"], dist_title)
            if not dist_text or len(dist_text) < 20:
                continue

            # Build BOTH observations as 2-entry standardized format
            # This controls for number of entries and roughly controls length
            sup_entry = entries[0]  # first entry is the supporting one
            # Find the actual supporting entry
            for e in entries:
                if any(title_match(e["title"], st) for st in sf_titles):
                    sup_entry = e
                    break

            # 1-SF: [SF1, Distractor]
            obs_1sf = rebuild_observation([
                {"idx": 1, "title": sup_entry["title"], "text": sup_entry["text"]},
                {"idx": 2, "title": dist_title, "text": dist_text},
            ])

            # 2-SF: [SF1, SF2]
            obs_2sf = rebuild_observation([
                {"idx": 1, "title": sup_entry["title"], "text": sup_entry["text"]},
                {"idx": 2, "title": missing_sf[0], "text": missing_text},
            ])

            candidates.append({
                "sample_id": sid,
                "question": ep["question"],
                "query": s0["action_input"],
                "obs_1sf": obs_1sf,
                "obs_2sf": obs_2sf,
                "retrieved_sf": retrieved_sf[0],
                "missing_sf": missing_sf[0],
                "dist_title": dist_title,
                "is_correct": lab["is_correct"],
                "margin": lab.get("margin_before", 0),
            })

    rng = random.Random(seed)
    rng.shuffle(candidates)
    if max_n:
        candidates = candidates[:max_n]
    print(f"Selected {len(candidates)} valid 1-SF samples")
    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--baseline-trace",
                    default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--hotpotqa-data",
                    default="data/hotpotqa/hotpot_dev_distractor_v1.json")
    ap.add_argument("--labels", default="results/phase1_probe/labels.jsonl")
    ap.add_argument("--output-dir", default="results/probe_sufficiency_synthetic")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--max-n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Select samples
    samples = select_1sf_samples(
        args.baseline_trace, args.hotpotqa_data, args.labels,
        max_n=args.max_n, seed=args.seed)

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")
    model.eval()
    layers = get_model_layers(model)
    print(f"Model loaded. {len(layers)} layers. Extracting L{args.layer}.")

    # Extract activations
    acts_1sf, acts_2sf, meta = [], [], []
    for s in tqdm(samples, desc="Extracting activations"):
        try:
            p1 = build_prompt(tokenizer, s["question"], s["query"], s["obs_1sf"])
            p2 = build_prompt(tokenizer, s["question"], s["query"], s["obs_2sf"])
            h1 = extract_l20_hidden(model, tokenizer, p1, layer_idx=args.layer)
            h2 = extract_l20_hidden(model, tokenizer, p2, layer_idx=args.layer)
            acts_1sf.append(h1)
            acts_2sf.append(h2)
            meta.append(s)
        except Exception as e:
            print(f"  SKIP {s['sample_id']}: {e}")

    X_1sf = np.array(acts_1sf, dtype=np.float32)
    X_2sf = np.array(acts_2sf, dtype=np.float32)
    N = len(meta)
    print(f"\nExtracted {N} pairs of activations, shape={X_1sf.shape}")

    # Save raw data
    np.savez(str(out_dir / "activations.npz"),
             X_1sf=X_1sf, X_2sf=X_2sf,
             sample_ids=np.array([m["sample_id"] for m in meta]))
    with open(out_dir / "meta.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    print(f"Saved to {out_dir}")

    # Free GPU
    del model
    torch.cuda.empty_cache()

    # ── Analysis ──
    run_analysis(X_1sf, X_2sf, meta, out_dir, args)


def run_analysis(X_1sf, X_2sf, meta, out_dir, args):
    """Train probe and compute all metrics."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import roc_auc_score
    from scipy.stats import pearsonr

    N = len(meta)
    SEP = "=" * 60

    # Load reference directions
    dir_r = np.load("results/phase1_probe/probe_direction_l20.npz")["decision_direction"].astype(np.float64)
    dir_r /= np.linalg.norm(dir_r)
    dir_a = np.load("steering/directions/direction_search_v3_layer20.npz")["decision_direction"].astype(np.float64)
    dir_a /= np.linalg.norm(dir_a)

    # Stack: X = [1-SF samples; 2-SF samples], y = [0...0, 1...1]
    X_all = np.vstack([X_1sf, X_2sf]).astype(np.float64)
    y_all = np.array([0]*N + [1]*N, dtype=np.int32)

    # Train Probe-S (sufficiency)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    auroc_scores = cross_val_score(
        LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                           solver="lbfgs", random_state=42),
        X_scaled, y_all, cv=cv, scoring="roc_auc")
    ba_scores = cross_val_score(
        LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                           solver="lbfgs", random_state=42),
        X_scaled, y_all, cv=cv, scoring="balanced_accuracy")

    clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                             solver="lbfgs", random_state=42)
    clf.fit(X_scaled, y_all)
    w = clf.coef_[0] / scaler.scale_
    dir_s = (w / np.linalg.norm(w)).astype(np.float64)

    print(f"\n{SEP}\nProbe-S (1-SF vs 2-SF, N={2*N})\n{SEP}")
    print(f"5-fold CV AUROC: {auroc_scores.mean():.3f} +/- {auroc_scores.std():.3f}")
    print(f"  Per-fold: {[f'{s:.3f}' for s in auroc_scores]}")
    print(f"5-fold CV BalAcc: {ba_scores.mean():.3f} +/- {ba_scores.std():.3f}")

    # Cosines
    cos_rs = float(np.dot(dir_r, dir_s))
    cos_sa = float(np.dot(dir_s, dir_a))
    cos_ra = float(np.dot(dir_r, dir_a))
    print(f"\n{SEP}\nCOSINE SIMILARITIES\n{SEP}")
    print(f"cos(Probe-R, Probe-S_synth) = {cos_rs:.4f}")
    print(f"cos(Probe-S_synth, action)  = {cos_sa:.4f}")
    print(f"cos(Probe-R, action)        = {cos_ra:.4f}")
    angle_rs = np.degrees(np.arccos(np.clip(abs(cos_rs), 0, 1)))
    angle_sa = np.degrees(np.arccos(np.clip(abs(cos_sa), 0, 1)))
    print(f"Angle(Probe-R, Probe-S_synth) = {angle_rs:.1f} deg")
    print(f"Angle(Probe-S_synth, action)  = {angle_sa:.1f} deg")

    # Paired delta analysis
    delta_h = X_2sf.astype(np.float64) - X_1sf.astype(np.float64)
    delta_evi = delta_h @ dir_r     # projection shift on evidence_dir
    delta_act = delta_h @ dir_a     # projection shift on action_dir
    delta_suf = delta_h @ dir_s     # projection shift on sufficiency_dir
    delta_norm = np.linalg.norm(delta_h, axis=1)

    print(f"\n{SEP}\nPAIRED DELTA (2-SF minus 1-SF)\n{SEP}")
    print(f"mean delta on evidence_dir: {delta_evi.mean():.4f} +/- {delta_evi.std():.4f}")
    print(f"mean delta on action_dir:   {delta_act.mean():.4f} +/- {delta_act.std():.4f}")
    print(f"mean delta on suffic_dir:   {delta_suf.mean():.4f} +/- {delta_suf.std():.4f}")
    print(f"mean ||delta_h||:           {delta_norm.mean():.4f} +/- {delta_norm.std():.4f}")

    from scipy.stats import wilcoxon, ttest_1samp
    for name, vals in [("evidence", delta_evi), ("action", delta_act), ("sufficiency", delta_suf)]:
        t, p_t = ttest_1samp(vals, 0)
        try:
            w, p_w = wilcoxon(vals)
        except Exception:
            w, p_w = 0, 1.0
        print(f"  {name}: t={t:.3f} p_t={p_t:.2e}, Wilcoxon p={p_w:.2e}")

    # Variance decomposition on margins (within 1-SF only)
    margins = np.array([m["margin"] for m in meta], dtype=np.float64)
    proj_evi_1sf = X_1sf.astype(np.float64) @ dir_r
    proj_act_1sf = X_1sf.astype(np.float64) @ dir_a
    proj_suf_1sf = X_1sf.astype(np.float64) @ dir_s

    print(f"\n{SEP}\nVARIANCE DECOMPOSITION (margin, 1-SF only, N={N})\n{SEP}")
    for name, proj in [("evidence", proj_evi_1sf), ("action", proj_act_1sf),
                        ("sufficiency", proj_suf_1sf)]:
        r, p = pearsonr(proj, margins)
        print(f"  {name}_proj → margin: r={r:.4f}, R²={r**2:.4f}, p={p:.2e}")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "N_pairs": N,
        "probe_s_auroc_mean": float(auroc_scores.mean()),
        "probe_s_auroc_std": float(auroc_scores.std()),
        "probe_s_auroc_folds": auroc_scores.tolist(),
        "probe_s_balac_mean": float(ba_scores.mean()),
        "cos_probeR_probeS": cos_rs,
        "cos_probeS_action": cos_sa,
        "cos_probeR_action": cos_ra,
        "delta_evidence_mean": float(delta_evi.mean()),
        "delta_action_mean": float(delta_act.mean()),
        "delta_sufficiency_mean": float(delta_suf.mean()),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save probe direction
    np.savez(str(out_dir / "probe_sufficiency_dir.npz"),
             decision_direction=dir_s.astype(np.float32),
             layer=args.layer, method="synthetic_1sf_vs_2sf", n_pairs=N,
             auroc=float(auroc_scores.mean()))
    print(f"\nAll results saved to {out_dir}")


if __name__ == "__main__":
    main()
