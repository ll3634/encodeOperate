#!/usr/bin/env python3
"""
Paired statistical tests and bootstrap confidence intervals for E2E evaluation.

Provides:
 - mcnemar_test()       : McNemar exact test (paired binary outcomes)
 - bootstrap_ci()       : Bootstrap CI for any paired metric (success_diff, etc.)
 - do_no_harm_metrics() : regression_rate, rescue_rate, unnecessary_tool_use, step_inflation
 - full_paired_report() : convenience wrapper that returns a publication-ready dict
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any


def _mcnemar_exact_p_value(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value via binomial test under p=0.5.

    Matches the classic exact McNemar formulation:
      X ~ Binomial(n=b+c, p=0.5)
      p = 2 * min(P(X <= min(b,c)), P(X >= max(b,c)))
    """
    import math
    n = int(b + c)
    if n == 0:
        return 1.0
    x = int(min(b, c))

    # Compute tail probability P(X <= x) exactly as sum_{i=0..x} C(n,i) / 2^n.
    num = 0
    for i in range(x + 1):
        num += math.comb(n, i)
    p = 2.0 * (num / (2 ** n))
    return float(min(1.0, p))


# ---------------------------------------------------------------------------
# McNemar's test (paired binary outcomes)
# ---------------------------------------------------------------------------

def mcnemar_test(
    baseline_correct: List[bool],
    policy_correct: List[bool],
) -> Dict[str, float]:
    """
    Exact McNemar test (two-sided) on paired binary vectors.

    Returns dict with: b (BL-only), c (policy-only), statistic, p_value.
    Uses scipy if available, otherwise manual exact binomial.
    """
    assert len(baseline_correct) == len(policy_correct), "Mismatched lengths"
    n = len(baseline_correct)

    # Discordant cells
    b = sum(bl and (not po) for bl, po in zip(baseline_correct, policy_correct))  # regressed
    c = sum((not bl) and po for bl, po in zip(baseline_correct, policy_correct))  # rescued

    # Prefer SciPy if present, but always provide an exact no-deps fallback.
    total_disc = b + c
    if total_disc == 0:
        p_value = 1.0
    else:
        try:
            # SciPy >=1.7: binomtest; older: binom_test
            from scipy.stats import binomtest as _binomtest
            p_value = float(_binomtest(b, total_disc, 0.5, alternative="two-sided").pvalue)
        except Exception:
            try:
                from scipy.stats import binom_test as _binom_test  # type: ignore
                p_value = float(_binom_test(b, total_disc, 0.5))
            except Exception:
                p_value = _mcnemar_exact_p_value(b, c)

    return {
        "n": n,
        "b_regressed": int(b),
        "c_rescued": int(c),
        "discordant": int(b + c),
        "mcnemar_p": round(p_value, 6),
    }


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    baseline_correct: List[bool],
    policy_correct: List[bool],
    metric: str = "success_diff",
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Bootstrap CI for paired comparison metrics.

    Supported metrics:
      - 'success_diff'  : P(policy correct) - P(baseline correct)
      - 'rescue_rate'   : P(rescued) = P(~BL & PO) / n
      - 'regression_rate': P(regressed) = P(BL & ~PO) / n

    Returns dict with: observed, ci_lower, ci_upper, ci_level, n_bootstrap, seed
    """
    rng = np.random.RandomState(seed)
    bl = np.array(baseline_correct, dtype=float)
    po = np.array(policy_correct, dtype=float)
    n = len(bl)

    def _compute(bl_arr, po_arr):
        if metric == "success_diff":
            return po_arr.mean() - bl_arr.mean()
        elif metric == "rescue_rate":
            return ((1 - bl_arr) * po_arr).mean()
        elif metric == "regression_rate":
            return (bl_arr * (1 - po_arr)).mean()
        else:
            raise ValueError(f"Unknown metric: {metric}")

    observed = float(_compute(bl, po))

    boot_vals = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_vals[i] = _compute(bl[idx], po[idx])

    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_vals, 100 * alpha))
    hi = float(np.percentile(boot_vals, 100 * (1 - alpha)))

    return {
        "metric": metric,
        "observed": round(observed, 6),
        "ci_lower": round(lo, 6),
        "ci_upper": round(hi, 6),
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Do-no-harm metrics
# ---------------------------------------------------------------------------

def do_no_harm_metrics(
    baseline_records: List[Dict],
    policy_records: List[Dict],
    indifferent_ids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute do-no-harm metrics between baseline and policy.

    Metrics:
      - regression_rate  : baseline correct → policy wrong (over all n)
      - rescue_rate      : baseline wrong → policy correct
      - net_gain         : rescues − regressions
      - unnecessary_tool_use_rate : among indifferent, policy uses MORE tools
      - step_inflation   : mean(policy_steps − baseline_steps) over indifferent
    """
    bl_by_id = {r["sample_id"]: r for r in baseline_records}
    po_by_id = {r["sample_id"]: r for r in policy_records}
    common = set(bl_by_id) & set(po_by_id)
    n = len(common)

    regressed = rescued = 0
    for sid in common:
        bl_ok = bl_by_id[sid]["is_correct"]
        po_ok = po_by_id[sid]["is_correct"]
        if bl_ok and not po_ok:
            regressed += 1
        if not bl_ok and po_ok:
            rescued += 1

    result = {
        "n_paired": n,
        "regression_count": regressed,
        "rescue_count": rescued,
        "regression_rate": regressed / n if n else 0,
        "rescue_rate": rescued / n if n else 0,
        "net_gain": rescued - regressed,
    }

    # Indifferent-only cost metrics
    ind_ids = set(indifferent_ids) if indifferent_ids else common
    ind_ids = ind_ids & common
    if ind_ids:
        extra_tools = []
        step_diffs = []
        for sid in ind_ids:
            bl_tc = bl_by_id[sid].get("tool_calls", 0)
            po_tc = po_by_id[sid].get("tool_calls", 0)
            extra_tools.append(int(po_tc > bl_tc))
            step_diffs.append(po_by_id[sid].get("steps", 0) - bl_by_id[sid].get("steps", 0))
        result["unnecessary_tool_use_rate"] = float(np.mean(extra_tools))
        result["step_inflation"] = float(np.mean(step_diffs))
    else:
        result["unnecessary_tool_use_rate"] = 0.0
        result["step_inflation"] = 0.0

    return result


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def full_paired_report(
    baseline_records: List[Dict],
    policy_records: List[Dict],
    policy_name: str = "",
    indifferent_ids: Optional[List[str]] = None,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate complete paired-comparison report (McNemar + bootstrap CI + do-no-harm).
    Records must be in unified format (from unified_output.py).
    """
    bl_by_id = {r["sample_id"]: r for r in baseline_records}
    po_by_id = {r["sample_id"]: r for r in policy_records}
    common = sorted(set(bl_by_id) & set(po_by_id))

    bl_correct = [bl_by_id[sid]["is_correct"] for sid in common]
    po_correct = [po_by_id[sid]["is_correct"] for sid in common]

    report = {
        "policy_name": policy_name or policy_records[0].get("policy_name", ""),
        "n_paired": len(common),
        "baseline_success": sum(bl_correct) / len(common) if common else 0,
        "policy_success": sum(po_correct) / len(common) if common else 0,
    }

    report["mcnemar"] = mcnemar_test(bl_correct, po_correct)

    for m in ["success_diff", "rescue_rate", "regression_rate"]:
        report[f"bootstrap_{m}"] = bootstrap_ci(
            bl_correct, po_correct, metric=m,
            n_bootstrap=n_bootstrap, seed=seed,
        )

    report["do_no_harm"] = do_no_harm_metrics(
        baseline_records, policy_records, indifferent_ids
    )

    return report

