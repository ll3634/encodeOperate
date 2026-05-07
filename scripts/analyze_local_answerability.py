#!/usr/bin/env python3
"""
Audit analysis for local-answerability paired experiment.

Sections:
  A. Robust statistics (paired permutation, bootstrap CI, sign test, trimmed mean)
  B. Saturation analysis (non-saturated vs saturated Low-L baseline margin)
  C. Manipulation audit (verify per-pair High-L vs Low-L confound properties)
  D. Regression control (Δmargin ~ Δlength + Δoverlap + Δentity + answer_present)
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.scorers import normalize_answer


def paired_permutation(diff, n_perm=20000, seed=0):
    """Two-sided paired permutation via random sign flip on diff."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff)
    obs = diff.mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    null = (signs * diff).mean(axis=1)
    p_less = float((null <= obs).mean())
    p_two = float((np.abs(null) >= abs(obs)).mean())
    return obs, p_less, p_two


def bootstrap_mean_ci(x, n_boot=20000, alpha=0.05, seed=1):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    n = len(x)
    means = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def sign_test(diff, alternative="less"):
    diff = np.asarray(diff)
    pos = int((diff > 0).sum())
    neg = int((diff < 0).sum())
    n = pos + neg
    if n == 0:
        return 0, 0, 1.0
    # number favoring alt (high<low => diff<0 => neg)
    k = neg if alternative == "less" else pos
    p = stats.binomtest(k, n, 0.5, alternative="greater").pvalue
    return pos, neg, float(p)


def trimmed_mean(x, prop=0.05):
    return float(stats.trim_mean(np.asarray(x), prop))


def contains_any(hay, needles):
    h = normalize_answer(hay)
    return any(normalize_answer(n) in h for n in needles if n and len(n) >= 3)


def answer_components(ans):
    parts = re.split(r",|&|\band\b", ans or "", flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) >= 3] or [ans or ""]


def audit_manipulation(pairs):
    """Check that High-L really is 'more locally answerable' in the intended way
    and that global insufficiency holds in both conditions."""
    n = len(pairs)
    cnt = {
        "high_longer": 0,
        "high_more_overlap": 0,
        "high_more_entities": 0,
        "high_more_copula": 0,
        "low_has_answer_sub": 0,
        "high_has_answer_sub": 0,
        "length_ratio_over_1_25": 0,
    }
    ratios = []
    for p in pairs:
        fl, fh = p["feat_low"], p["feat_high"]
        if fh["tok_len"] > fl["tok_len"]:
            cnt["high_longer"] += 1
        if fh["q_overlap"] > fl["q_overlap"]:
            cnt["high_more_overlap"] += 1
        if fh["entity_count"] > fl["entity_count"]:
            cnt["high_more_entities"] += 1
        if fh["copula_count"] > fl["copula_count"]:
            cnt["high_more_copula"] += 1
        comps = answer_components(p["gold_answer"])
        if contains_any(p["obs_low"], comps):
            cnt["low_has_answer_sub"] += 1
        if contains_any(p["obs_high"], comps):
            cnt["high_has_answer_sub"] += 1
        if p["length_ratio"] > 1.25:
            cnt["length_ratio_over_1_25"] += 1
        ratios.append(p["length_ratio"])
    return cnt, ratios


def regress(y, X, names):
    """Plain OLS, return beta, se, t, p. Drops constant regressors (which would
    make the design matrix singular) and flags them separately."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = [i for i in range(X.shape[1]) if np.std(X[:, i]) > 1e-12]
    dropped = [names[i] for i in range(X.shape[1]) if i not in keep]
    Xk = X[:, keep]
    kept_names = [names[i] for i in keep]
    Xd = np.column_stack([np.ones(len(y)), Xk])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    dof = max(1, len(y) - Xd.shape[1])
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(Xd.T @ Xd)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, 0.0)
    p = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    coefs = list(zip(["const"] + kept_names, beta, se, t, p))
    return coefs, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/local_answerability/pairs.jsonl")
    ap.add_argument("--eval", default="results/local_answerability/eval_results.jsonl")
    ap.add_argument("--out", default="results/local_answerability/audit_report.json")
    ap.add_argument("--sat-cut", type=float, default=-6.0)
    args = ap.parse_args()

    pairs = {json.loads(l)["sample_id"]: json.loads(l) for l in open(args.pairs)}
    evals = [json.loads(l) for l in open(args.eval)]
    low = {r["sample_id"]: r for r in evals if r["condition"] == "low"}
    high = {r["sample_id"]: r for r in evals if r["condition"] == "high"}
    ids = [i for i in low if i in high and i in pairs]
    m_low = np.array([low[i]["margin"] for i in ids])
    m_high = np.array([high[i]["margin"] for i in ids])
    diff = m_high - m_low
    sr_low = np.array([int(low[i]["action_type"] == "search") for i in ids])
    sr_high = np.array([int(high[i]["action_type"] == "search") for i in ids])

    # === A. Robust stats ===
    obs, p_perm_less, p_perm_two = paired_permutation(diff)
    mean, lo, hi = bootstrap_mean_ci(diff)
    pos, neg, p_sign = sign_test(diff, alternative="less")
    tr_mean = trimmed_mean(diff, 0.05)
    wilc = stats.wilcoxon(m_high, m_low, alternative="less")
    tt = stats.ttest_rel(m_high, m_low, alternative="less")

    A = {
        "n": len(ids),
        "mean_diff": float(mean),
        "median_diff": float(np.median(diff)),
        "trimmed_mean_5pct": tr_mean,
        "bootstrap_ci95": [lo, hi],
        "permutation_p_less": p_perm_less,
        "permutation_p_two_sided": p_perm_two,
        "sign_test_pos": pos, "sign_test_neg": neg, "sign_test_p_less": p_sign,
        "wilcoxon_stat": float(wilc.statistic), "wilcoxon_p_less": float(wilc.pvalue),
        "ttest_t": float(tt.statistic), "ttest_p_less": float(tt.pvalue),
    }

    # === B. Saturation split (by Low-L margin) ===
    nonsat_mask = m_low > args.sat_cut
    sat_mask = ~nonsat_mask
    B = {}
    for tag, mask in (("non_saturated", nonsat_mask), ("saturated", sat_mask)):
        d = diff[mask]
        if len(d) < 3:
            B[tag] = {"n": int(mask.sum()), "note": "too few"}
            continue
        m_b, lo_b, hi_b = bootstrap_mean_ci(d)
        _, p_less_b, _ = paired_permutation(d, n_perm=10000, seed=2)
        B[tag] = {
            "n": int(mask.sum()),
            "mean_diff": float(m_b),
            "median_diff": float(np.median(d)),
            "bootstrap_ci95": [lo_b, hi_b],
            "perm_p_less": p_less_b,
            "pct_high_less_than_low": float((d < 0).mean()),
        }

    # === C. Manipulation audit ===
    pair_list = [pairs[i] for i in ids]
    C, ratios = audit_manipulation(pair_list)
    C["mean_length_ratio"] = float(np.mean(ratios))
    C["max_length_ratio"] = float(np.max(ratios))

    # === D. Regression on Δmargin ===
    dlen = np.array([pairs[i]["feat_high"]["tok_len"] - pairs[i]["feat_low"]["tok_len"] for i in ids])
    dover = np.array([pairs[i]["feat_high"]["q_overlap"] - pairs[i]["feat_low"]["q_overlap"] for i in ids])
    dent = np.array([pairs[i]["feat_high"]["entity_count"] - pairs[i]["feat_low"]["entity_count"] for i in ids])
    dcop = np.array([pairs[i]["feat_high"]["copula_count"] - pairs[i]["feat_low"]["copula_count"] for i in ids])
    X = np.column_stack([dlen, dover, dent, dcop])
    names = ["d_tok_len", "d_q_overlap", "d_entity_count", "d_copula_count"]
    reg, dropped = regress(diff, X, names)
    D = {
        "design": "Δmargin = β0 + β1·Δlen + β2·Δoverlap + β3·Δentity + β4·Δcopula",
        "dropped_constant_regressors": dropped,
        "coefs": [{"name": n, "beta": float(b), "se": float(s), "t": float(t), "p": float(p)}
                  for (n, b, s, t, p) in reg],
    }

    # === 2ndSR ===
    rate = {
        "low_2ndSR": float(sr_low.mean()),
        "high_2ndSR": float(sr_high.mean()),
        "L1H0": int(((sr_low == 1) & (sr_high == 0)).sum()),
        "L0H1": int(((sr_low == 0) & (sr_high == 1)).sum()),
    }

    report = {"A_robust": A, "B_saturation_cut": args.sat_cut, "B_saturation": B,
              "C_manipulation": C, "D_regression": D, "action_rate": rate}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
