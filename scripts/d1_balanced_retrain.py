#!/usr/bin/env python3
"""D1 balanced control: train D1' on a balanced subsample (200 hotpot vs
200 MuSiQue) and re-run erase/flip under the locked Figure 1 protocol."""
import json, time, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from agent.prompts import ACTION_TOKENS
from evidence_erasure_test import (
    ProjectionFlipHook, forward_margin, build_p0_prompt,
    boot_mean_ci, LAYER, SEED, N,
)

OUT = Path("results/d1_balanced_retrain"); OUT.mkdir(parents=True, exist_ok=True)
SUB_SEED = 20260424
TRAIN_SEED = SEED  # 20260502

X_CACHE = "results/ocft/per_candidate/D1_source/X.npz"
D1_ORIG = "results/ocft/per_candidate/D1_source/direction.npy"


def train_probe(X, y, *, use_class_weight, seed_run):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                                 precision_score, recall_score, f1_score)
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    X = X.astype(np.float32); y = y.astype(int)
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    cw = "balanced" if use_class_weight else None

    cv_aurocs, cv_baccs = [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_run)
    for tr, te in skf.split(Xs, y):
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs",
                                 class_weight=cw, random_state=seed_run).fit(Xs[tr], y[tr])
        yh = clf.predict_proba(Xs[te])[:, 1]; yp = clf.predict(Xs[te])
        cv_aurocs.append(roc_auc_score(y[te], yh))
        cv_baccs.append(balanced_accuracy_score(y[te], yp))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed_run)
    tr, te = next(sss.split(Xs, y))
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs",
                             class_weight=cw, random_state=seed_run).fit(Xs[tr], y[tr])
    yp = clf.predict(Xs[te]); yh = clf.predict_proba(Xs[te])[:, 1]
    metrics = {
        "n_total": int(len(y)), "n_label0": int((y == 0).sum()),
        "n_label1": int((y == 1).sum()), "class_weight": cw,
        "auroc": float(roc_auc_score(y[te], yh)),
        "balanced_accuracy": float(balanced_accuracy_score(y[te], yp)),
        "f1": float(f1_score(y[te], yp, zero_division=0)),
        "precision": float(precision_score(y[te], yp, zero_division=0)),
        "recall": float(recall_score(y[te], yp, zero_division=0)),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "cv_auroc_mean": float(np.mean(cv_aurocs)),
        "cv_auroc_std": float(np.std(cv_aurocs)),
        "cv_balacc_mean": float(np.mean(cv_baccs)),
        "cv_balacc_std": float(np.std(cv_baccs)),
        "train_seed": int(seed_run),
        "sub_seed": int(SUB_SEED),
    }
    clf_all = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs",
                                 class_weight=cw, random_state=seed_run).fit(Xs, y)
    w = clf_all.coef_[0] / scaler.scale_
    direction = (w / np.linalg.norm(w)).astype(np.float32)
    return direction, metrics


def main():
    print(f"[init] L{LAYER} N={N}  TRAIN_SEED={TRAIN_SEED}  SUB_SEED={SUB_SEED}")

    cache = np.load(X_CACHE)
    X = cache["X"]; y_orig = cache["y"]
    idx_hp = np.where(y_orig == 0)[0]   # hotpot, n=486
    idx_mq = np.where(y_orig == 1)[0]   # musique, n=200
    print(f"  hotpot: {len(idx_hp)}  musique: {len(idx_mq)}")

    rng = np.random.default_rng(SUB_SEED)
    hp_pick = rng.choice(idx_hp, size=len(idx_mq), replace=False)
    sel = np.concatenate([hp_pick, idx_mq])
    yv = np.concatenate([np.zeros(len(hp_pick), int), np.ones(len(idx_mq), int)])

    d, m = train_probe(X[sel], yv, use_class_weight=False, seed_run=TRAIN_SEED)
    np.save(OUT / "direction_D1prime.npy", d)
    json.dump(m, open(OUT / "probe_D1prime.json", "w"), indent=2)
    print(f"  D1prime  n={m['n_total']} (1:{m['n_label1']}, 0:{m['n_label0']})  "
          f"AUROC={m['auroc']:.3f}  BalAcc={m['balanced_accuracy']:.3f}  "
          f"CV-AUROC={m['cv_auroc_mean']:.3f}\u00b1{m['cv_auroc_std']:.3f}")

    # Geometry
    A = np.load("steering/directions/direction_decomp_full_layer20.npz",
                allow_pickle=True)["decision_direction"].astype(np.float32)
    A = A / np.linalg.norm(A)
    E = np.load("results/phase1_probe/probe_direction_l20.npz",
                allow_pickle=True)["decision_direction"].astype(np.float32)
    E = E / np.linalg.norm(E)
    D1o = np.load(D1_ORIG).astype(np.float32); D1o = D1o / np.linalg.norm(D1o)

    def cos(u, v): return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    geom = {
        "cos_D1prime_D1orig": cos(d, D1o),
        "cos_D1prime_A":      cos(d, A),
        "cos_D1prime_E":      cos(d, E),
        "cos_D1orig_A":       cos(D1o, A),
        "cos_D1orig_E":       cos(D1o, E),
    }
    json.dump(geom, open(OUT / "geometry_summary.json", "w"), indent=2)
    print("\n[geometry]"); [print(f"  {k}: {v:+.5f}") for k, v in geom.items()]

    if "--probe-only" in sys.argv:
        print("[probe-only] skipping erasure forwards"); return

    run_erasure(d)


def run_erasure(direction):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\n[load] Qwen/Qwen2.5-7B-Instruct")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    labels = [json.loads(l) for l in open("results/phase1_probe/labels.jsonl")]
    bl_map = {}
    with open("results/l20_rho020_n500/baseline_results.jsonl") as f:
        for line in f: ep = json.loads(line); bl_map[ep["sample_id"]] = ep
    prompts, sample_ids = [], []
    for ld in labels:
        ep = bl_map.get(ld["sample_id"])
        if not ep or not ep.get("steps"): continue
        s0 = ep["steps"][0]
        if not s0.get("observation"): continue
        prompts.append(build_p0_prompt(tok, ld["question"], s0["action_input"], s0["observation"]))
        sample_ids.append(ld["sample_id"])
        if len(prompts) >= N: break
    cached = np.load("results/evidence_erasure_test/per_prompt_margins.npz")
    cached_sids = list(cached["sample_ids"])
    assert sample_ids == cached_sids, "prompt cohort mismatch with cached Figure 1"
    print(f"[prompts] N={len(prompts)}  (matched cached cohort)")

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True).eval()
    tool_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]

    cond_specs = [
        ("erase_D1prime", lambda: ProjectionFlipHook(model, direction, factor=1.0)),
        ("flip_D1prime",  lambda: ProjectionFlipHook(model, direction, factor=2.0)),
    ]
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

    def stats(m_cond, base):
        dm = (m_cond - base).astype(np.float32)
        signed_mean = float(dm.mean())
        m_abs, lo_abs, hi_abs = boot_mean_ci(np.abs(dm))
        rng = np.random.default_rng(SEED); B = 2000
        idx = rng.integers(0, len(dm), size=(B, len(dm)))
        sm = np.abs(dm[idx].mean(axis=1))
        lo_sm, hi_sm = np.percentile(sm, [2.5, 97.5])
        flip_rate = float(((np.sign(m_cond) != np.sign(base)) &
                           (np.abs(base) > 1e-6)).mean())
        return {"signed_mean_dm": signed_mean,
                "abs_signed_mean_dm": abs(signed_mean),
                "abs_signed_mean_dm_ci": [float(lo_sm), float(hi_sm)],
                "mean_abs_dm": float(m_abs),
                "mean_abs_dm_ci": [float(lo_abs), float(hi_abs)],
                "flip_rate": flip_rate}

    res = {"N": n,
           "convention": "abs_signed_mean_dm matches figure_spectrum.json dm_flip / dm_erase",
           "baseline_source": "cached from results/evidence_erasure_test/per_prompt_margins.npz",
           "D1prime": {
               "erase": stats(margins["erase_D1prime"], base_cached),
               "flip":  stats(margins["flip_D1prime"],  base_cached),
           }}
    json.dump(res, open(OUT / "erasure_results.json", "w"), indent=2)
    print("\n[erasure summary]")
    v = res["D1prime"]; e = v["erase"]; f = v["flip"]
    print(f"  D1prime  erase |dm|={e['abs_signed_mean_dm']:.4f}  "
          f"flip |dm|={f['abs_signed_mean_dm']:.4f} CI{f['abs_signed_mean_dm_ci']}  "
          f"flip_rate={f['flip_rate']:.2%}")


if __name__ == "__main__":
    main()
