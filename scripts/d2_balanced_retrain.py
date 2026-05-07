#!/usr/bin/env python3
"""D2 balanced retrain: train D2'_subsample (11+11) and D2'_balanced
(class_weight='balanced'), then re-run erase/flip under the locked
Figure 1 protocol on the same 100 prompts. No change to D2_original.
"""
import json, time, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import get_model_layers
from evidence_erasure_test import (
    ProjectionFlipHook, margin_from_logits, forward_margin,
    build_p0_prompt, boot_mean_ci, LAYER, SEED, N,
)
from ocft_lib import train_probe_on

OUT = Path("results/d2_balanced_retrain"); OUT.mkdir(parents=True, exist_ok=True)
SUB_SEED = 20260424
TRAIN_SEED = SEED  # 20260502, same as original


def build_subsample_probe(X_all, mb_all, seed):
    """11 pos + 11 neg balanced subsample, no class_weight needed."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                                 precision_score, recall_score, f1_score)
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    pos_idx = np.where(mb_all > 0)[0]
    neg_idx = np.where(mb_all <= 0)[0]
    rng = np.random.default_rng(seed)
    neg_pick = rng.choice(neg_idx, size=len(pos_idx), replace=False)
    sel = np.concatenate([pos_idx, neg_pick])
    X = X_all[sel].astype(np.float32)
    y = np.concatenate([np.ones(len(pos_idx), int), np.zeros(len(neg_pick), int)])

    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    # 5-fold CV (k=4 if n too small for k=5)
    k = 5 if len(y) >= 10 else 3
    cv_aurocs, cv_baccs = [], []
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=TRAIN_SEED)
    for tr, te in skf.split(Xs, y):
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs",
                                 random_state=TRAIN_SEED)
        clf.fit(Xs[tr], y[tr])
        yh = clf.predict_proba(Xs[te])[:, 1]; yp = clf.predict(Xs[te])
        cv_aurocs.append(roc_auc_score(y[te], yh) if len(set(y[te])) == 2 else float("nan"))
        cv_baccs.append(balanced_accuracy_score(y[te], yp))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=TRAIN_SEED)
    tr, te = next(sss.split(Xs, y))
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs",
                             random_state=TRAIN_SEED).fit(Xs[tr], y[tr])
    yp = clf.predict(Xs[te]); yh = clf.predict_proba(Xs[te])[:, 1]
    metrics = {
        "n_total": int(len(y)), "n_label0": int((y == 0).sum()),
        "n_label1": int((y == 1).sum()),
        "balanced_accuracy": float(balanced_accuracy_score(y[te], yp)),
        "auroc": float(roc_auc_score(y[te], yh)),
        "f1": float(f1_score(y[te], yp, zero_division=0)),
        "precision": float(precision_score(y[te], yp, zero_division=0)),
        "recall": float(recall_score(y[te], yp, zero_division=0)),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "cv_k": k,
        "cv_auroc_mean": float(np.nanmean(cv_aurocs)),
        "cv_auroc_std": float(np.nanstd(cv_aurocs)),
        "cv_balacc_mean": float(np.mean(cv_baccs)),
        "cv_balacc_std": float(np.std(cv_baccs)),
        "subsample_seed": int(seed), "train_seed": int(TRAIN_SEED),
        "subsample_indices": sel.tolist(),
    }
    clf_all = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs",
                                 random_state=TRAIN_SEED).fit(Xs, y)
    w = clf_all.coef_[0] / scaler.scale_
    direction = (w / np.linalg.norm(w)).astype(np.float32)
    return direction, metrics


def main():
    print(f"[init] L{LAYER} N={N} TRAIN_SEED={TRAIN_SEED}")

    # -------- Load cached L20 activations + margin_before from labels --------
    cache = np.load("results/phase1_probe/activations_multilayer.npz")
    X_l20 = cache["layer_20"]                       # (486, 3584)
    cache_sids = cache["sample_ids"]
    labels = [json.loads(l) for l in open("results/phase1_probe/labels.jsonl")]
    sid_to_mb = {l["sample_id"]: l["margin_before"] for l in labels}
    mb = np.array([sid_to_mb[s] for s in cache_sids], dtype=np.float32)
    print(f"  X_l20={X_l20.shape}  mb>0: {(mb>0).sum()}  mb<=0: {(mb<=0).sum()}")

    # -------- D2'_subsample (11+11) --------
    d_sub, m_sub = build_subsample_probe(X_l20, mb, seed=SUB_SEED)
    np.save(OUT / "direction_D2prime_subsample.npy", d_sub)
    json.dump({k: v for k, v in m_sub.items() if k != "subsample_indices"},
              open(OUT / "probe_D2prime_subsample.json", "w"), indent=2)
    print(f"  D2'_subsample n={m_sub['n_total']} AUROC={m_sub['auroc']:.3f} "
          f"BalAcc={m_sub['balanced_accuracy']:.3f} CV-AUROC="
          f"{m_sub['cv_auroc_mean']:.3f}±{m_sub['cv_auroc_std']:.3f}")

    # -------- D2'_balanced (n=486, class_weight='balanced') --------
    y_bal = (mb > 0).astype(int)
    probe_bal = train_probe_on(X_l20, y_bal, seed=TRAIN_SEED)
    d_bal = probe_bal["direction"].astype(np.float32)
    np.save(OUT / "direction_D2prime_balanced.npy", d_bal)
    m_bal = {k: v for k, v in probe_bal.items() if k != "direction"}
    m_bal.update({"n_total": 486, "n_label0": int((y_bal == 0).sum()),
                  "n_label1": int((y_bal == 1).sum())})
    json.dump(m_bal, open(OUT / "probe_D2prime_balanced.json", "w"), indent=2)
    print(f"  D2'_balanced n=486 AUROC={m_bal['auroc']:.3f} BalAcc={m_bal['balanced_accuracy']:.3f}")

    # -------- Geometry --------
    A = np.load("steering/directions/direction_decomp_full_layer20.npz",
                allow_pickle=True)["decision_direction"].astype(np.float32)
    E = np.load("results/phase1_probe/probe_direction_l20.npz",
                allow_pickle=True)["decision_direction"].astype(np.float32)
    D2_orig = np.load("results/ocft/per_candidate/D2_action_prior/direction.npy"
                      ).astype(np.float32)

    def cos(u, v):
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    geom = {
        "cos_D2sub_A": cos(d_sub, A), "cos_D2bal_A": cos(d_bal, A),
        "cos_D2orig_A": cos(D2_orig, A),
        "cos_D2sub_D2orig": cos(d_sub, D2_orig),
        "cos_D2bal_D2orig": cos(d_bal, D2_orig),
        "cos_D2sub_D2bal": cos(d_sub, d_bal),
        "cos_D2sub_E": cos(d_sub, E), "cos_D2bal_E": cos(d_bal, E),
    }
    json.dump(geom, open(OUT / "geometry_summary.json", "w"), indent=2)
    print("\n[geometry]"); [print(f"  {k}: {v:+.5f}") for k, v in geom.items()]

    if "--probe-only" in sys.argv:
        print("[probe-only] skipping erasure forwards"); return

    # -------- Erasure under locked Figure 1 protocol --------
    # D2'_balanced is bit-equivalent to D2_original (cos=+1.0): reuse
    # cached margins from results/evidence_erasure_test/random_control/.
    # Only D2'_subsample needs new forwards.
    run_erasure(d_sub, d_bal, geom)


def run_erasure(d_sub, d_bal, geom):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\n[load] Qwen/Qwen2.5-7B-Instruct")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    labels = [json.loads(l) for l in open("results/phase1_probe/labels.jsonl")]
    bl_map = {}
    with open("results/l20_rho020_n500/baseline_results.jsonl") as f:
        for line in f:
            ep = json.loads(line); bl_map[ep["sample_id"]] = ep
    prompts, sample_ids = [], []
    for ld in labels:
        ep = bl_map.get(ld["sample_id"])
        if not ep or not ep.get("steps"): continue
        s0 = ep["steps"][0]
        if not s0.get("observation"): continue
        prompts.append(build_p0_prompt(tok, ld["question"], s0["action_input"], s0["observation"]))
        sample_ids.append(ld["sample_id"])
        if len(prompts) >= N: break
    print(f"[prompts] N={len(prompts)}")

    cached = np.load("results/evidence_erasure_test/per_prompt_margins.npz")
    cached_sids = list(cached["sample_ids"])
    assert sample_ids == cached_sids, "prompt cohort mismatch with cached Figure 1"

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True).eval()
    tool_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]

    bal_equiv_orig = abs(geom["cos_D2bal_D2orig"]) > 0.999
    cond_specs = [
        ("erase_D2sub", lambda: ProjectionFlipHook(model, d_sub, factor=1.0)),
        ("flip_D2sub",  lambda: ProjectionFlipHook(model, d_sub, factor=2.0)),
    ]
    if not bal_equiv_orig:
        cond_specs += [
            ("erase_D2bal", lambda: ProjectionFlipHook(model, d_bal, factor=1.0)),
            ("flip_D2bal",  lambda: ProjectionFlipHook(model, d_bal, factor=2.0)),
        ]
    else:
        print(f"[skip] D2'_balanced equiv D2_original (cos={geom['cos_D2bal_D2orig']:+.4f}); "
              f"reusing cached D2 margins from random_control/new_margins.npz")
    n = len(prompts)
    margins = {c: np.zeros(n, dtype=np.float32) for c, _ in cond_specs}
    base_cached = cached["baseline"].astype(np.float32)
    t0 = time.time()
    for i, p in enumerate(prompts):
        for c, hf in cond_specs:
            margins[c][i] = forward_margin(model, tok, p, hf, tool_ids, fin_ids)
        if (i + 1) % 10 == 0 or i == 0:
            eta = (time.time() - t0) / (i + 1) * (n - i - 1)
            print(f"  [{i+1:>3d}/{n}]  ETA={eta:.0f}s")
    print(f"[done] forwards: {time.time() - t0:.0f}s")

    np.savez(OUT / "per_prompt_margins.npz",
             sample_ids=np.array(sample_ids), baseline=base_cached,
             **{c: margins[c] for c, _ in cond_specs})

    # Pull cached D2_original margins for D2'_balanced reuse and as reference
    cc = np.load("results/evidence_erasure_test/random_control/new_margins.npz")
    cc_base = cc["base"].astype(np.float32)

    def stats(m_cond, base):
        dm = (m_cond - base).astype(np.float32)
        signed_mean = float(dm.mean())
        signed_mean_abs = abs(signed_mean)              # convention used by figure_spectrum.json
        m_abs, lo_abs, hi_abs = boot_mean_ci(np.abs(dm))
        # bootstrap CI on |signed mean|
        rng = np.random.default_rng(SEED); B = 2000
        idx = rng.integers(0, len(dm), size=(B, len(dm)))
        sm = np.abs(dm[idx].mean(axis=1))
        lo_sm, hi_sm = np.percentile(sm, [2.5, 97.5])
        flip_rate = float(((np.sign(m_cond) != np.sign(base)) &
                           (np.abs(base) > 1e-6)).mean())
        return {
            "signed_mean_dm": signed_mean,
            "abs_signed_mean_dm": signed_mean_abs,
            "abs_signed_mean_dm_ci": [float(lo_sm), float(hi_sm)],
            "mean_abs_dm": float(m_abs),
            "mean_abs_dm_ci": [float(lo_abs), float(hi_abs)],
            "flip_rate": flip_rate,
        }

    res = {"N": n,
           "baseline_source": "cached from results/evidence_erasure_test/per_prompt_margins.npz",
           "convention": "abs_signed_mean_dm matches figure_spectrum.json dm_flip / dm_erase",
           "D2_original_cached": {
               "source": "results/evidence_erasure_test/random_control/new_margins.npz",
               "erase": stats(cc["D2_erase"].astype(np.float32), cc_base),
               "flip":  stats(cc["D2_flip"].astype(np.float32),  cc_base),
           },
           "D2prime_subsample": {
               "erase": stats(margins["erase_D2sub"], base_cached),
               "flip":  stats(margins["flip_D2sub"],  base_cached),
           }}
    if bal_equiv_orig:
        res["D2prime_balanced"] = {
            "note": "bit-equivalent to D2_original (cos=+1.0); reusing cached D2 margins",
            "erase": res["D2_original_cached"]["erase"],
            "flip":  res["D2_original_cached"]["flip"],
        }
    else:
        res["D2prime_balanced"] = {
            "erase": stats(margins["erase_D2bal"], base_cached),
            "flip":  stats(margins["flip_D2bal"],  base_cached),
        }
    json.dump(res, open(OUT / "erasure_results.json", "w"), indent=2)
    print("\n[erasure summary]")
    for k in ("D2_original_cached", "D2prime_subsample", "D2prime_balanced"):
        v = res[k]
        e = v["erase"]; f = v["flip"]
        print(f"  {k}: erase |dm|={e['abs_signed_mean_dm']:.4f} CI{e['abs_signed_mean_dm_ci']}  "
              f"flip |dm|={f['abs_signed_mean_dm']:.4f} CI{f['abs_signed_mean_dm_ci']}  "
              f"flip_rate={f['flip_rate']:.2%}")


if __name__ == "__main__":
    main()
