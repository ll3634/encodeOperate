#!/usr/bin/env python3
"""Step 2-4 of per-prompt heterogeneity: predictor search + subgroup analysis.
Reads intermediate.npz produced by per_prompt_heterogeneity.py."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT = Path("results/per_prompt_heterogeneity")
LABEL_PATH = "results/phase1_probe/labels.jsonl"

RNG = np.random.default_rng(20240102)


def auroc_binary(y_score, y_true):
    """AUROC via Mann-Whitney U/(n0*n1)."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    u, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    auroc = u / (len(pos) * len(neg))
    return float(auroc), float(p)


def bootstrap_mean_ci(x, n_boot=5000, alpha=0.05):
    n = len(x)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        boot[b] = x[RNG.integers(0, n, n)].mean()
    return float(x.mean()), float(np.percentile(boot, 100 * alpha / 2)), \
           float(np.percentile(boot, 100 * (1 - alpha / 2)))


def perm_test_means(a, b, n_perm=10000):
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    na = len(a)
    cnt = 0
    for _ in range(n_perm):
        RNG.shuffle(pool)
        s = pool[:na].mean() - pool[na:].mean()
        if abs(s) >= abs(obs):
            cnt += 1
    return float(obs), (cnt + 1) / (n_perm + 1)


def main():
    inter = np.load(OUT / "intermediate.npz", allow_pickle=True)
    sample_ids = [str(s) for s in inter["sample_ids"]]
    r = inter["r"]
    parallel = inter["parallel"]
    full = inter["full"]
    H = inter["H"]
    E = inter["E"]
    A = inter["A"]
    random_shifts = inter["random_shifts"]  # (200, 100)
    N = len(sample_ids)

    # Reload labels for the cohort
    by_id = {json.loads(l)["sample_id"]: json.loads(l) for l in open(LABEL_PATH)}
    lab = [by_id[s] for s in sample_ids]

    # ---- Build predictors ----
    h_dot_E = (H @ E)
    h_dot_A = (H @ A)
    h_norm = np.linalg.norm(H, axis=1)
    cos_h_E = h_dot_E / (h_norm + 1e-12)
    cos_h_A = h_dot_A / (h_norm + 1e-12)
    margin_before = np.array([float(x["margin_before"]) for x in lab])
    n_sf_retrieved = np.array([int(x["n_sf_retrieved"]) for x in lab])
    n_sf_total = np.array([int(x["n_sf_total"]) for x in lab])
    q_len = np.array([len(x["question"].split()) for x in lab])
    abs_full = np.abs(full)
    is_correct = np.array([int(bool(x["is_correct"])) for x in lab])
    behav_continue = np.array([int(bool(x["behavioral_continue"])) for x in lab])
    label_bin = np.array([int(x["label"]) for x in lab])

    cont_predictors = {
        "margin_before": margin_before,
        "n_sf_retrieved": n_sf_retrieved.astype(float),
        "n_sf_total": n_sf_total.astype(float),
        "question_word_count": q_len.astype(float),
        "abs_full_shift": abs_full,
        "h_dot_E": h_dot_E,
        "h_dot_A": h_dot_A,
        "cos_h_E": cos_h_E,
        "cos_h_A": cos_h_A,
    }
    cat_predictors = {
        "is_correct": is_correct,
        "behavioral_continue": behav_continue,
        "label_0_vs_1": label_bin,
    }
    K = len(cont_predictors) + len(cat_predictors)
    rows = []
    for name, x in cont_predictors.items():
        rho, p = stats.spearmanr(x, r, nan_policy="omit")
        rows.append({"predictor": name, "type": "continuous",
                     "stat_name": "spearman_rho", "stat": float(rho),
                     "p_raw": float(p), "p_bonf": float(min(1.0, p * K)),
                     "n": int(N)})
    for name, y in cat_predictors.items():
        if len(np.unique(y)) < 2:
            rows.append({"predictor": name, "type": "categorical",
                         "stat_name": "auroc",
                         "stat": float("nan"), "p_raw": 1.0, "p_bonf": 1.0,
                         "note": "single class only", "n": int(N)})
            continue
        auc, p = auroc_binary(r, y)
        rows.append({"predictor": name, "type": "categorical",
                     "stat_name": "auroc",
                     "stat": float(auc), "p_raw": float(p),
                     "p_bonf": float(min(1.0, p * K)),
                     "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum())})
    json.dump({"n_tests": K, "rows": rows},
              open(OUT / "predictor_table.json", "w"), indent=2, default=float)

    # ---- Discovery ----
    discovery = []
    for r_ in rows:
        if r_["p_bonf"] >= 0.01:
            continue
        if r_["type"] == "continuous" and abs(r_["stat"]) > 0.4:
            discovery.append(r_)
        elif r_["type"] == "categorical" and (r_["stat"] > 0.65 or r_["stat"] < 0.35):
            discovery.append(r_)
    print(f"[predictor] tested K={K}, discovery-grade: {len(discovery)}")
    for r_ in sorted(rows, key=lambda x: x["p_bonf"])[:6]:
        print(f"  {r_['predictor']:>22s}  {r_['stat_name']:>13s}={r_['stat']:+.3f}  "
              f"p_raw={r_['p_raw']:.2e}  p_bonf={r_['p_bonf']:.2e}")

    # Strongest by |stat|, excluding NaN (constant arrays)
    cont_rows = [r_ for r_ in rows if r_["type"] == "continuous"
                 and not (isinstance(r_["stat"], float) and np.isnan(r_["stat"]))]
    strongest_cont = sorted(cont_rows, key=lambda r_: -abs(r_["stat"]))[0]
    cat_rows = [r_ for r_ in rows if r_["type"] == "categorical"
                and not (isinstance(r_["stat"], float) and np.isnan(r_["stat"]))]
    strongest_cat = sorted(cat_rows, key=lambda r_: -abs(r_["stat"] - 0.5))[0]
    print(f"\n[strongest_cont] {strongest_cont['predictor']} rho={strongest_cont['stat']:+.3f} p_bonf={strongest_cont['p_bonf']:.2e}")
    print(f"[strongest_cat]  {strongest_cat['predictor']} auroc={strongest_cat['stat']:+.3f} p_bonf={strongest_cat['p_bonf']:.2e}")

    # Artifact check: is the strongest predictor |Δm_full| or h·Â?
    artifact_predictors = {"abs_full_shift", "h_dot_A", "cos_h_A"}
    artifact_flag = (strongest_cont["predictor"] in artifact_predictors and
                     strongest_cont["p_bonf"] < 0.01 and abs(strongest_cont["stat"]) > 0.4)

    # ---- Subgroup analysis ----
    K_sub = max(20, N // 5)
    subgroups = {}
    for r_ in rows:
        if r_["p_bonf"] >= 0.05:
            continue
        name = r_["predictor"]
        if r_["type"] == "continuous":
            x = cont_predictors[name]
            order = np.argsort(x)
            bot_idx = order[:K_sub]; top_idx = order[-K_sub:]
        else:
            y = cat_predictors[name]
            top_idx = np.where(y == 1)[0]
            bot_idx = np.where(y == 0)[0]
            if len(top_idx) < 5 or len(bot_idx) < 5: continue
        r_top, r_top_lo, r_top_hi = bootstrap_mean_ci(r[top_idx])
        r_bot, r_bot_lo, r_bot_hi = bootstrap_mean_ci(r[bot_idx])
        par_top, par_top_lo, par_top_hi = bootstrap_mean_ci(np.abs(parallel[top_idx]))
        par_bot, par_bot_lo, par_bot_hi = bootstrap_mean_ci(np.abs(parallel[bot_idx]))
        diff_obs, diff_p = perm_test_means(r[top_idx], r[bot_idx], n_perm=5000)
        # parallel-vs-random null check in top subgroup: |parallel| vs random null
        rand_abs_per_prompt = np.abs(random_shifts).mean(axis=0)  # (100,)
        rand_top, rand_top_lo, rand_top_hi = bootstrap_mean_ci(rand_abs_per_prompt[top_idx])
        rand_bot, rand_bot_lo, rand_bot_hi = bootstrap_mean_ci(rand_abs_per_prompt[bot_idx])
        # parallel ratio against random in top group
        par_vs_rand_top = par_top / max(rand_top, 1e-6)
        rejects_null_top = par_top_lo > rand_top_hi
        subgroups[name] = {
            "k_per_side": int(K_sub) if r_["type"] == "continuous" else
                          {"top_n": int(len(top_idx)), "bot_n": int(len(bot_idx))},
            "r_top_mean_ci": [r_top, r_top_lo, r_top_hi],
            "r_bot_mean_ci": [r_bot, r_bot_lo, r_bot_hi],
            "abs_par_top_mean_ci": [par_top, par_top_lo, par_top_hi],
            "abs_par_bot_mean_ci": [par_bot, par_bot_lo, par_bot_hi],
            "perm_test_diff_obs": diff_obs, "perm_test_p": diff_p,
            "abs_par_top_vs_rand_top_ratio": par_vs_rand_top,
            "rand_abs_top_mean_ci": [rand_top, rand_top_lo, rand_top_hi],
            "rand_abs_bot_mean_ci": [rand_bot, rand_bot_lo, rand_bot_hi],
            "rejects_random_null_in_top_subgroup": bool(rejects_null_top),
        }
    json.dump(subgroups, open(OUT / "subgroup_results.json", "w"), indent=2, default=float)
    print(f"\n[subgroup] examined {len(subgroups)} predictors with p_bonf<0.05")
    for name, d in subgroups.items():
        print(f"  {name}: r_top={d['r_top_mean_ci'][0]:.3f}[{d['r_top_mean_ci'][1]:.3f},{d['r_top_mean_ci'][2]:.3f}] "
              f"r_bot={d['r_bot_mean_ci'][0]:.3f}  perm_p={d['perm_test_p']:.4f}  "
              f"|par|_top vs rand_top: {d['abs_par_top_vs_rand_top_ratio']:.2f}x  "
              f"rejects_null={d['rejects_random_null_in_top_subgroup']}")

    # ---- Verdict ----
    dist_out = json.load(open(OUT / "distribution_analysis.json"))
    bimodal = (dist_out["verdict_distribution"] == "BIMODAL")
    has_disc = len(discovery) > 0
    discovery_subgroup_ok = False
    if has_disc:
        # Use top-ranked discovery predictor's subgroup result
        top_disc = max(discovery, key=lambda x: abs(x["stat"]))
        if top_disc["predictor"] in subgroups:
            sg = subgroups[top_disc["predictor"]]
            r_top_lo = sg["r_top_mean_ci"][1]
            r_top_mean = sg["r_top_mean_ci"][0]
            discovery_subgroup_ok = (r_top_mean > 0.4 and r_top_lo > 0.25)
    if artifact_flag:
        verdict = "ARTIFACT"
    elif bimodal and has_disc and discovery_subgroup_ok:
        verdict = "DISCOVERY"
    elif bimodal and not has_disc:
        verdict = "HETEROGENEOUS_NO_PREDICTOR"
    elif not bimodal and not has_disc:
        verdict = "UNIFORM"
    else:
        verdict = "AMBIGUOUS"
    print(f"\n[VERDICT] {verdict}")
    print(f"  bimodal={bimodal}  has_discovery={has_disc}  "
          f"discovery_subgroup_ok={discovery_subgroup_ok}  artifact={artifact_flag}")

    json.dump({
        "verdict": verdict,
        "bimodal": bimodal,
        "has_discovery_predictor": has_disc,
        "discovery_subgroup_passes_threshold": discovery_subgroup_ok,
        "artifact_flag": artifact_flag,
        "strongest_continuous": strongest_cont,
        "strongest_categorical": strongest_cat,
        "discovery_grade_predictors": discovery,
    }, open(OUT / "verdict.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
