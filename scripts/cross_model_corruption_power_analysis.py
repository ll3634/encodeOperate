"""Post-hoc power analysis for paired-corruption boundary cases.

Inputs:  existing paired_corruption summaries (cross_model_*_v2/full_results.json,
         results/paired_corruption/paired_corruption_results.json) and per-record
         margin files (cross_model_extractability/*, cross_model_behavior_alignment/*).
Outputs: results/cross_model_corruption_power_analysis/{summary.json,report.md,margin_distributions.png}.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("results")
OUT = ROOT / "cross_model_corruption_power_analysis"
OUT.mkdir(parents=True, exist_ok=True)

# Ordered to match paper convention: case-study → successes → boundaries
MODELS = [
    {"key": "qwen2_5_7b",        "display": "Qwen2.5-7B-Instruct (case study)",
     "pc_path": "paired_corruption/paired_corruption_results.json",  # native format
     "margin_path": "cross_model_extractability/eval_results_qwen2_5_7b.jsonl",
     "is_native_qwen": True},
    {"key": "mistral_7b_v03",    "display": "Mistral-7B-Instruct-v0.3",
     "pc_path": "cross_model_mistral_v2/full_results.json",
     "margin_path": "cross_model_extractability/eval_results_mistral_7b_v03.jsonl"},
    {"key": "gemma_2_9b_it",     "display": "Gemma-2-9B-it",
     "pc_path": "cross_model_gemma2_v2/full_results.json",
     "margin_path": "cross_model_behavior_alignment/eval_results_gemma.jsonl"},
    {"key": "llama_3_1_8b",      "display": "Llama-3.1-8B-Instruct (boundary)",
     "pc_path": "cross_model_llama31_v2/full_results.json",
     "margin_path": None},
    {"key": "r1_distill_qwen_7b","display": "DeepSeek-R1-Distill-Qwen-7B (boundary)",
     "pc_path": "cross_model_r1distill_v2/full_results.json",
     "margin_path": "cross_model_extractability/eval_results_r1_distill_qwen_7b.jsonl"},
]


def load_paired_corruption(spec):
    p = ROOT / spec["pc_path"]
    if spec.get("is_native_qwen"):
        r = json.load(open(p))
        A = [d["delta_action"] for d in r["per_sample"]["A"]]
        B = [d["delta_action"] for d in r["per_sample"]["B"]]
        return {"n_A": len(A), "n_B": len(B), "muA": float(np.mean(A)),
                "muB": float(np.mean(B)), "sigA": float(np.std(A, ddof=1)),
                "sigB": float(np.std(B, ddof=1)),
                "p_two": float(r["tests"]["A_vs_B_delta_action"]["p"]),
                "ratio": float(np.mean(A) / np.mean(B)),
                "per_sample_A": A, "per_sample_B": B}
    r = json.load(open(p))["paired_corruption"]
    return {"n_A": r["n_samples"], "n_B": r["n_samples"], "muA": r["A_mean_delta_action"],
            "muB": r["B_mean_delta_action"], "p_two": r["MW_action_p"],
            "ratio": r["AB_ratio_action"], "sigA": None, "sigB": None,
            "per_sample_A": None, "per_sample_B": None}


def infer_sigma_pool(muA, muB, p_two, n_A, n_B):
    """Invert normal approximation to two-sample test: z = (muA-muB)/(sig*sqrt(1/n_A+1/n_B)).
    Returns the pooled SD implied by the observed mean difference and p-value."""
    z = stats.norm.isf(p_two / 2.0)
    if z <= 1e-6:
        return float("nan")
    return abs(muA - muB) / (z * np.sqrt(1.0 / n_A + 1.0 / n_B))


def analytic_mde_ratio(muB, sigma_pool, n_per_group, alpha=0.05, power=0.80):
    """Smallest A/B ratio detectable at given power for two-sample test."""
    # Use t critical with df = 2(n-1); for n=200 the difference vs. z is negligible
    df = 2 * (n_per_group - 1)
    t_a = stats.t.isf(alpha / 2.0, df)
    z_b = stats.norm.isf(1.0 - power)  # power = P(Z > z_a - delta) ⇒ delta = z_a + z_b
    delta = (t_a + z_b) * sigma_pool * np.sqrt(2.0 / n_per_group)
    return float((muB + delta) / muB)


def analytic_required_n(muA, muB, sigma_pool, alpha=0.05, power=0.80):
    """n_per_group required to detect observed effect at given power."""
    d = abs(muA - muB) / sigma_pool
    if d < 1e-6:
        return float("inf")
    z_a = stats.norm.isf(alpha / 2.0)
    z_b = stats.norm.isf(1.0 - power)
    return float(2.0 * ((z_a + z_b) / d) ** 2)


def lognormal_params(mu, sigma):
    """Method-of-moments fit of lognormal so its mean=mu, std=sigma."""
    if mu <= 0 or sigma <= 0:
        return None
    var = sigma ** 2
    sig2 = np.log(1.0 + var / (mu ** 2))
    mu_log = np.log(mu) - 0.5 * sig2
    return mu_log, np.sqrt(sig2)


def bootstrap_ratio_ci(muA, sigA, muB, sigB, nA, nB, B=10000, rng=None,
                        per_A=None, per_B=None):
    rng = rng or np.random.default_rng(42)
    if per_A is not None and per_B is not None:
        a = np.asarray(per_A); b = np.asarray(per_B)
        idxA = rng.integers(0, len(a), size=(B, len(a)))
        idxB = rng.integers(0, len(b), size=(B, len(b)))
        rs = a[idxA].mean(1) / b[idxB].mean(1)
        method = "nonparametric"
    else:
        pA = lognormal_params(muA, sigA)
        pB = lognormal_params(muB, sigB)
        if pA is None or pB is None:
            return {"method": "failed", "ci_low": None, "ci_high": None,
                    "mean": None, "p_contains_1": None, "p_contains_1.6": None}
        sA = rng.lognormal(pA[0], pA[1], size=(B, nA))
        sB = rng.lognormal(pB[0], pB[1], size=(B, nB))
        rs = sA.mean(1) / sB.mean(1)
        method = "parametric_lognormal"
    lo, hi = np.quantile(rs, [0.025, 0.975])
    return {"method": method, "ci_low": float(lo), "ci_high": float(hi),
            "mean": float(rs.mean()), "median": float(np.median(rs)),
            "p_contains_1": bool(lo <= 1.0 <= hi),
            "p_contains_1p6": bool(lo <= 1.6 <= hi)}


def margin_distribution(jsonl_path):
    """Per-condition descriptive stats of margin_label."""
    if jsonl_path is None or not Path(jsonl_path).exists():
        return None
    by_cond = {}
    for line in open(jsonl_path):
        r = json.loads(line)
        c = r.get("condition", "ALL")
        by_cond.setdefault(c, []).append(float(r["margin_label"]))
    out = {}
    for c, vals in by_cond.items():
        a = np.asarray(vals)
        out[c] = {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std(ddof=1)),
                  "min": float(a.min()), "max": float(a.max()),
                  "p5": float(np.quantile(a, .05)), "p25": float(np.quantile(a, .25)),
                  "p50": float(np.quantile(a, .50)), "p75": float(np.quantile(a, .75)),
                  "p95": float(np.quantile(a, .95)), "values": a.tolist()}
    return out


def main():
    rng = np.random.default_rng(42)
    summary = {"models": {}, "thresholds": {"alpha": 0.05, "power": 0.80,
                                            "weakest_significant_ratio": 1.6,
                                            "null_ratio": 1.0}}

    for spec in MODELS:
        pc = load_paired_corruption(spec)
        # Sigma estimate
        if pc["sigA"] is not None:
            sigma_pool = float(np.sqrt((pc["sigA"] ** 2 + pc["sigB"] ** 2) / 2.0))
            sigma_source = "observed_per_sample"
        else:
            sigma_pool = infer_sigma_pool(pc["muA"], pc["muB"], pc["p_two"],
                                          pc["n_A"], pc["n_B"])
            sigma_source = "inferred_from_p_value"

        # Power calcs (analytical)
        mde = analytic_mde_ratio(pc["muB"], sigma_pool, pc["n_A"]) if np.isfinite(sigma_pool) else None
        n_req = analytic_required_n(pc["muA"], pc["muB"], sigma_pool) if np.isfinite(sigma_pool) else None
        if n_req is not None and (n_req == float("inf") or n_req > 10000):
            n_req_label = "infeasibly_large"
        else:
            n_req_label = int(np.ceil(n_req)) if n_req is not None else None

        # Bootstrap CI on the ratio
        sigA_used = pc["sigA"] if pc["sigA"] is not None else sigma_pool
        sigB_used = pc["sigB"] if pc["sigB"] is not None else sigma_pool
        boot = bootstrap_ratio_ci(pc["muA"], sigA_used, pc["muB"], sigB_used,
                                   pc["n_A"], pc["n_B"], B=10000, rng=rng,
                                   per_A=pc["per_sample_A"], per_B=pc["per_sample_B"])

        # Margins
        margins = margin_distribution(ROOT / spec["margin_path"]) if spec["margin_path"] else None
        margin_summary = None
        if margins is not None:
            margin_summary = {c: {k: v for k, v in d.items() if k != "values"}
                              for c, d in margins.items()}

        summary["models"][spec["key"]] = {
            "display": spec["display"],
            "paired_corruption": {
                "n_A": pc["n_A"], "n_B": pc["n_B"], "muA": pc["muA"], "muB": pc["muB"],
                "ratio_observed": pc["ratio"], "p_two": pc["p_two"],
                "sigma_pool": sigma_pool, "sigma_source": sigma_source,
            },
            "power": {
                "mde_ratio_at_80pct": mde,
                "n_per_group_required_for_observed": n_req,
                "n_per_group_required_label": n_req_label,
                "alpha": 0.05, "power": 0.80,
            },
            "bootstrap_ratio": boot,
            "margins_step1": margin_summary,
        }

    # Save JSON (compact arrays trimmed)
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"Wrote {OUT/'summary.json'}")

    # Try margin figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        models_with_margins = [(s["key"], s["display"], margin_distribution(ROOT / s["margin_path"]))
                                for s in MODELS if s["margin_path"]]
        n_panels = len(models_with_margins)
        fig, axes = plt.subplots(1, n_panels, figsize=(3.4 * n_panels, 3.5), sharey=True)
        if n_panels == 1:
            axes = [axes]
        cond_colors = {"N0": "#4682B4", "T0": "#D2691E", "S0": "#2E8B57"}
        for ax, (key, display, dist) in zip(axes, models_with_margins):
            for cond in ["N0", "T0", "S0"]:
                if cond in dist:
                    ax.hist(dist[cond]["values"], bins=20, alpha=0.55,
                            color=cond_colors[cond], label=cond, density=False)
            ax.axvline(0, color="black", lw=0.7, ls="--")
            ax.set_title(display, fontsize=9)
            ax.set_xlabel("margin_label (lp_search−lp_final)")
            ax.legend(fontsize=7)
        axes[0].set_ylabel("count")
        fig.suptitle("Step-1 action margin distributions across models",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / "margin_distributions.png", dpi=130)
        plt.close(fig)
        print(f"Wrote {OUT/'margin_distributions.png'}")
    except Exception as e:
        print(f"[warn] figure not produced: {e}")


if __name__ == "__main__":
    main()

