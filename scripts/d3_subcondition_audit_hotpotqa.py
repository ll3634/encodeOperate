#!/usr/bin/env python3
"""D3 sub-condition audit on HotpotQA-native cells (n_sf_retrieved).

Replaces the original LGM cell scheme {T0,T1,N0,S0} (undefined on HotpotQA)
with cell_0 / cell_1 / cell_2 (= n_sf_retrieved). Reuses cached per_example
shifts from §3 (decomposition_ci_null) and OCFT (D3 injection). No GPU.

Pipeline = STEPs 2-5 of the original audit prompt.
"""
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.stats import spearmanr

# ── Constants ──────────────────────────────────────────────────────────────
B_BOOT = 2_000
N_PERM = 10_000
SEED = 20260502
OUT_DIR = Path("results/ocft/d3_subcondition_audit_hotpotqa_native")

PRE_REGISTRATION = """\
## Pre-registration (audit trail — written BEFORE inspecting any results)

**Question (D3 sub-condition audit on HotpotQA-native cells):** Is the OCFT
D3-parallel operative effect concentrated in §3 prompts where the search
tool retrieved supporting facts (high_sf), as the availability hypothesis
predicts, or is it sf-retrieval-independent?

**Operating point (identical to §3 / §16.3 / OCFT):**
  Model = Qwen/Qwen2.5-7B-Instruct,  layer = 20,
  injection point = p0 last token (decision-only, max_interventions=1),
  rho = -0.20,  hidden_rms = 0.65,  normalize_rms(direction) = 1.0,
  N = 100 paired prompts (same sample_ids as §3 / OCFT),
  K = 200 random RMS-matched directions (cached, identical to §16.3),
  bootstrap = 2 000 resamples,  permutation = 10 000 sign-flips / shuffles,
  random seed = 20260502.

**Cells (HotpotQA-native, derived from labels.jsonl field n_sf_retrieved):**
  cell_0  : n_sf_retrieved == 0   (tool found no supporting fact)
  cell_1  : n_sf_retrieved == 1
  cell_2  : n_sf_retrieved == 2
  low_sf  : cell_0
  high_sf : cell_1 ∪ cell_2
  all     : full N=100

**Pre-registered Outcome rules — locked BEFORE looking at numbers:**
  Outcome A  high_sf |Δm_par_D3|/|Δm_full| ≥ 0.40  AND
             low_sf  |Δm_par_D3|/|Δm_full| ≤ 0.20  AND
             heterogeneity permutation p < 0.05
             ⇒ availability-consistent
  Outcome B  no cell ratio > 0.45 or < 0.20  AND
             heterogeneity permutation p ≥ 0.05
             ⇒ uniform / availability not verifiable
  Outcome C  low_sf ratio ≥ 0.40  AND  high_sf ratio ≤ 0.20
             ⇒ anti-prediction / corpus-transfer artefact
  Outcome D  Spearman ρ(D_evidence cell ratios, D3 cell ratios) > 0.7
             ⇒ flag D_evidence cell-conditionality (independent of A/B/C)

This audit does NOT modify the OCFT R1∧R2∧R3 verdict. D3 remains operative.
"""


def boot_mean_ci(x, B=B_BOOT, level=95.0, rng=None):
    rng = rng if rng is not None else np.random.default_rng(SEED)
    if len(x) == 0:
        return {"n": 0, "mean": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan")}
    if len(x) == 1:
        v = float(x[0])
        return {"n": 1, "mean": v, "ci_low": v, "ci_high": v}
    idx = rng.integers(0, len(x), size=(B, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(100 - level) / 2, 100 - (100 - level) / 2])
    return {"n": int(len(x)), "mean": float(x.mean()),
            "ci_low": float(lo), "ci_high": float(hi)}


def boot_ratio_ci(num, denom, B=B_BOOT, level=95.0, rng=None):
    """|mean(num)| / |mean(denom)| via paired bootstrap on prompts."""
    rng = rng if rng is not None else np.random.default_rng(SEED + 1)
    n = len(num)
    if n == 0:
        return {"point": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan")}
    eps = 1e-12
    if n == 1:
        v = float(abs(num[0]) / (abs(denom[0]) + eps))
        return {"point": v, "ci_low": v, "ci_high": v}
    idx = rng.integers(0, n, size=(B, n))
    n_b = num[idx].mean(axis=1)
    d_b = denom[idx].mean(axis=1)
    r_b = np.abs(n_b) / (np.abs(d_b) + eps)
    lo, hi = np.percentile(r_b, [(100 - level) / 2, 100 - (100 - level) / 2])
    point = float(np.abs(num.mean()) / (np.abs(denom.mean()) + eps))
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi)}


def perm_par_vs_random(par_cell, rand_cell, B=N_PERM, rng=None):
    """Paired sign-flip permutation on (par_i - rand_per_prompt_i).
    Two-sided p on |mean(diff)|. Floor at 1/B."""
    rng = rng if rng is not None else np.random.default_rng(SEED + 2)
    if len(par_cell) == 0:
        return {"p_value": float("nan"), "obs_mean_diff": float("nan")}
    diff = par_cell - rand_cell
    obs = abs(diff.mean())
    n = len(diff)
    if n == 1:
        return {"p_value": 1.0, "obs_mean_diff": float(diff.mean())}
    flips = rng.choice([-1.0, 1.0], size=(B, n))
    perm_means = np.abs((flips * diff).mean(axis=1))
    p = float((perm_means >= obs).mean())
    return {"p_value": float(max(p, 1.0 / B)),
            "obs_mean_diff": float(diff.mean())}



def cell_null_band(random_shifts, mask, level=95.0):
    """Per-cell null band: mean across (cell prompts) for each of K=200
    random directions, then 95% range across the 200."""
    if mask.sum() == 0:
        return {"low": float("nan"), "high": float("nan"), "mean": float("nan")}
    rs = random_shifts[:, mask].mean(axis=1)  # (K,)
    lo, hi = np.percentile(rs, [(100 - level) / 2, 100 - (100 - level) / 2])
    return {"low": float(lo), "high": float(hi), "mean": float(rs.mean())}


def per_cell_row(mask, par, perp, full, random_shifts, rand_per_prompt,
                 rng_root):
    if mask.sum() == 0:
        return None
    full_c = full[mask]
    par_c = par[mask]
    perp_c = perp[mask]
    rand_c = rand_per_prompt[mask]
    null_band = cell_null_band(random_shifts, mask)
    in_band = bool(null_band["low"] <= float(par_c.mean()) <= null_band["high"])
    return {
        "n": int(mask.sum()),
        "full": boot_mean_ci(full_c, rng=np.random.default_rng(rng_root + 10)),
        "parallel": boot_mean_ci(par_c, rng=np.random.default_rng(rng_root + 11)),
        "perp": boot_mean_ci(perp_c, rng=np.random.default_rng(rng_root + 12)),
        "ratio_par_over_full_abs":
            boot_ratio_ci(par_c, full_c, rng=np.random.default_rng(rng_root + 13)),
        "ratio_perp_over_full_abs":
            boot_ratio_ci(perp_c, full_c, rng=np.random.default_rng(rng_root + 14)),
        "null_band_95": null_band,
        "parallel_in_null_band": in_band,
        "perm_par_vs_random": perm_par_vs_random(
            par_c, rand_c, rng=np.random.default_rng(rng_root + 15)),
    }


def heterogeneity_test(par, n_sf, B=N_PERM, rng=None):
    """One-way permutation F-equivalent across {0,1,2}.
    Shuffles cell labels B times; reports observed F and permutation p."""
    rng = rng if rng is not None else np.random.default_rng(SEED + 3)
    cell_vals = {c: par[n_sf == c] for c in (0, 1, 2)}
    nonempty = [c for c, v in cell_vals.items() if len(v) > 0]
    k = len(nonempty)
    n = int(len(par))
    grand = float(par.mean())

    def F_stat(par_arr, labels):
        cs = [par_arr[labels == c] for c in nonempty]
        bt = sum(len(c) * (c.mean() - grand) ** 2 for c in cs if len(c) > 0)
        wt = sum(((c - c.mean()) ** 2).sum() for c in cs if len(c) > 0)
        denom = wt / max(n - k, 1) + 1e-12
        return (bt / max(k - 1, 1)) / denom

    F_obs = float(F_stat(par, n_sf))
    F_perm = np.empty(B)
    for b in range(B):
        F_perm[b] = F_stat(par, rng.permutation(n_sf))
    p = float((F_perm >= F_obs).mean())
    return {"F": F_obs, "p_value": float(max(p, 1.0 / B)),
            "k_cells_nonempty": k, "n": n,
            "cells_used": [int(c) for c in nonempty]}



def cell_ratio(par, full, mask):
    eps = 1e-12
    if mask.sum() == 0:
        return float("nan")
    return float(abs(par[mask].mean()) / (abs(full[mask].mean()) + eps))


def evaluate_outcomes(par_d3, par_evidence, full, n_sf, het_d3):
    base_cells = (0, 1, 2)
    ratios_d3 = {c: cell_ratio(par_d3, full, n_sf == c) for c in base_cells}
    ratios_ev = {c: cell_ratio(par_evidence, full, n_sf == c) for c in base_cells}
    ratio_low = cell_ratio(par_d3, full, n_sf == 0)
    ratio_high = cell_ratio(par_d3, full, n_sf >= 1)
    het_p = het_d3["p_value"]

    # Outcome A / B / C — exclusive ladder, evaluated in order
    if (not np.isnan(ratio_high) and not np.isnan(ratio_low)
            and ratio_high >= 0.40 and ratio_low <= 0.20 and het_p < 0.05):
        outcome_abc = "A"
    elif (not np.isnan(ratio_high) and not np.isnan(ratio_low)
          and ratio_low >= 0.40 and ratio_high <= 0.20):
        outcome_abc = "C"
    else:
        valid_ratios = [v for v in ratios_d3.values() if not np.isnan(v)]
        if (valid_ratios and max(valid_ratios) <= 0.45
                and min(valid_ratios) >= 0.20 and het_p >= 0.05):
            outcome_abc = "B"
        else:
            outcome_abc = "indeterminate"

    # Outcome D — Spearman of cell ratios D3 vs D_evidence over base_cells
    valid = [c for c in base_cells
             if not np.isnan(ratios_d3[c]) and not np.isnan(ratios_ev[c])
             and (n_sf == c).sum() >= 1]
    if len(valid) >= 3:
        sp = spearmanr([ratios_d3[c] for c in valid],
                       [ratios_ev[c] for c in valid])
        sp_rho = float(sp.statistic)
        sp_p = float(sp.pvalue)
    else:
        sp_rho, sp_p = float("nan"), float("nan")
    outcome_d = "yes" if (not np.isnan(sp_rho) and sp_rho > 0.7) else "no"

    return {
        "ratios_d3_per_cell": ratios_d3,
        "ratios_evidence_per_cell": ratios_ev,
        "ratio_low_sf_d3": ratio_low,
        "ratio_high_sf_d3": ratio_high,
        "heterogeneity_p_d3": het_p,
        "outcome_abc": outcome_abc,
        "spearman_d3_vs_evidence_cell_ratios": {"rho": sp_rho, "p": sp_p,
                                                 "n_cells_used": len(valid)},
        "outcome_d_independent_flag": outcome_d,
    }


def fmt_ci(d, key="mean", lo="ci_low", hi="ci_high", fmt="{:+.3f}"):
    if d is None or np.isnan(d.get(key, float("nan"))):
        return "n/a"
    return f"{fmt.format(d[key])} [{fmt.format(d[lo])}, {fmt.format(d[hi])}]"


def fmt_ratio(d):
    if d is None or np.isnan(d.get("point", float("nan"))):
        return "n/a"
    return (f"{d['point']:.2f} [{d['ci_low']:.2f}, {d['ci_high']:.2f}]")


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    return f"<{1.0/N_PERM:.0e}" if p <= 1.0 / N_PERM else f"{p:.4f}"


def render_table(rows, full_arr, label):
    out = [f"### {label} per-cell decomposition\n\n",
           "| cell | n | Δm_full | Δm_par | |par|/|full| | in null band? | "
           "perm p (par vs random) |\n",
           "|------|---:|---|---|---|---|---:|\n"]
    order = ["0", "1", "2", "low_sf", "high_sf", "all"]
    for c in order:
        r = rows.get(c)
        if r is None:
            out.append(f"| {c} | 0 | n/a | n/a | n/a | n/a | n/a |\n")
            continue
        out.append(
            f"| {c} | {r['n']} | "
            f"{fmt_ci(r['full'])} | "
            f"{fmt_ci(r['parallel'])} | "
            f"{fmt_ratio(r['ratio_par_over_full_abs'])} | "
            f"{'YES' if r['parallel_in_null_band'] else 'no'} "
            f"(band [{r['null_band_95']['low']:+.3f}, "
            f"{r['null_band_95']['high']:+.3f}]) | "
            f"{fmt_p(r['perm_par_vs_random']['p_value'])} |\n")
    return "".join(out)



def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load §3 cached shifts (D_evidence parallel/perp + full + random) ──
    ev = np.load("results/decomposition_ci_null/per_example_shifts.npz",
                 allow_pickle=True)
    sids = list(ev["sample_ids"])
    full = ev["full"].astype(np.float64)
    par_evidence = ev["parallel"].astype(np.float64)
    perp_evidence = ev["perp"].astype(np.float64)
    random_shifts = ev["random"].astype(np.float64)  # (K, N)
    K = random_shifts.shape[0]
    N = len(sids)

    # ── Load OCFT D3 shifts ─────────────────────────────────────────────
    d3 = np.load("results/ocft/per_example_shifts_D3_candidate_present.npz",
                 allow_pickle=True)
    if list(d3["sample_ids"]) != sids:
        sys.exit("[STOP] D3 sample_ids do not match §3 sample_ids order.")
    par_d3 = d3["parallel"].astype(np.float64)
    perp_d3 = d3["perp"].astype(np.float64)
    if not np.allclose(d3["full"].astype(np.float64), full):
        sys.exit("[STOP] D3 full-injection shifts do not match §3.")

    # ── Load HotpotQA n_sf_retrieved per sample ─────────────────────────
    labels = {}
    with open("results/phase1_probe/labels.jsonl") as f:
        for line in f:
            d = json.loads(line)
            labels[d["sample_id"]] = d
    missing = [s for s in sids
               if s not in labels or "n_sf_retrieved" not in labels[s]]
    if missing:
        sys.exit(f"[STOP] {len(missing)} samples missing n_sf_retrieved: "
                 f"{missing[:5]}")
    n_sf = np.array([int(labels[s]["n_sf_retrieved"]) for s in sids])

    cell_counts = {
        "0": int((n_sf == 0).sum()),
        "1": int((n_sf == 1).sum()),
        "2": int((n_sf == 2).sum()),
        "low_sf": int((n_sf == 0).sum()),
        "high_sf": int((n_sf >= 1).sum()),
        "all": int(N),
    }
    print(f"[init] N={N}  K={K}  cells={cell_counts}")

    masks = {
        "0": n_sf == 0,
        "1": n_sf == 1,
        "2": n_sf == 2,
        "low_sf": n_sf == 0,
        "high_sf": n_sf >= 1,
        "all": np.ones(N, dtype=bool),
    }
    rand_per_prompt = random_shifts.mean(axis=0)

    print("[D3] per-cell stats …")
    d3_cells = {c: per_cell_row(m, par_d3, perp_d3, full,
                                random_shifts, rand_per_prompt,
                                rng_root=20260601 + i * 100)
                for i, (c, m) in enumerate(masks.items())}
    print("[D_evidence] per-cell stats …")
    ev_cells = {c: per_cell_row(m, par_evidence, perp_evidence, full,
                                random_shifts, rand_per_prompt,
                                rng_root=20260701 + i * 100)
                for i, (c, m) in enumerate(masks.items())}

    print("[heterogeneity D3] …")
    het_d3 = heterogeneity_test(par_d3, n_sf)
    print(f"   F={het_d3['F']:.3f}  p={het_d3['p_value']:.4f}")
    print("[heterogeneity D_evidence] …")
    het_ev = heterogeneity_test(par_evidence, n_sf)
    print(f"   F={het_ev['F']:.3f}  p={het_ev['p_value']:.4f}")

    # Spearman cell-mean Δm_par_D3 vs cell-mean Δm_full (across base cells)
    base_cells_used = [c for c in (0, 1, 2) if (n_sf == c).sum() >= 1]
    cm_par_d3 = [par_d3[n_sf == c].mean() for c in base_cells_used]
    cm_par_ev = [par_evidence[n_sf == c].mean() for c in base_cells_used]
    cm_full = [full[n_sf == c].mean() for c in base_cells_used]
    if len(base_cells_used) >= 3:
        sp_d3 = spearmanr(cm_par_d3, cm_full)
        sp_ev = spearmanr(cm_par_ev, cm_full)
        sp_d3_pack = {"rho": float(sp_d3.statistic), "p": float(sp_d3.pvalue)}
        sp_ev_pack = {"rho": float(sp_ev.statistic), "p": float(sp_ev.pvalue)}
    else:
        sp_d3_pack = {"rho": float("nan"), "p": float("nan"),
                      "note": f"<3 non-empty cells (only {len(base_cells_used)})"}
        sp_ev_pack = {"rho": float("nan"), "p": float("nan"),
                      "note": f"<3 non-empty cells (only {len(base_cells_used)})"}

    outcomes = evaluate_outcomes(par_d3, par_evidence, full, n_sf, het_d3)
    print(f"[outcome] A/B/C = {outcomes['outcome_abc']}, "
          f"D-flag = {outcomes['outcome_d_independent_flag']}")
    print(f"   ratio_low_sf={outcomes['ratio_low_sf_d3']:.3f}, "
          f"ratio_high_sf={outcomes['ratio_high_sf_d3']:.3f}, "
          f"het p={outcomes['heterogeneity_p_d3']:.4f}")

    # ── JSON deliverables ───────────────────────────────────────────────
    per_cell_pack = {
        "config": {
            "model": "Qwen/Qwen2.5-7B-Instruct", "layer": 20,
            "rho": -0.20, "hidden_rms": 0.65, "normalize_rms": 1.0,
            "n_samples": int(N), "K_random": int(K),
            "n_boot": B_BOOT, "n_perm": N_PERM, "seed": SEED,
            "cells_source": "results/phase1_probe/labels.jsonl :: n_sf_retrieved",
            "cell_counts": cell_counts,
        },
        "D3_candidate_present": d3_cells,
        "outcomes": outcomes,
        "spearman_cellmean_par_vs_full_D3": sp_d3_pack,
    }
    with open(OUT_DIR / "per_cell_results.json", "w") as f:
        json.dump(per_cell_pack, f, indent=2, default=float)

    with open(OUT_DIR / "heterogeneity_test.json", "w") as f:
        json.dump({"D3_candidate_present": het_d3,
                   "D_evidence": het_ev,
                   "n_perm": N_PERM, "seed": SEED}, f, indent=2)

    with open(OUT_DIR / "d_evidence_per_cell.json", "w") as f:
        json.dump({"cells": ev_cells,
                   "heterogeneity": het_ev,
                   "spearman_cellmean_par_vs_full": sp_ev_pack},
                  f, indent=2, default=float)

    # ── Markdown report ─────────────────────────────────────────────────
    md = ["# D3 Sub-condition Audit (HotpotQA-native cells)\n\n",
          PRE_REGISTRATION,
          "\n---\n\n## Results\n\n",
          f"**N = {N}**, K_random = {K}.  "
          f"Cell counts (n_sf_retrieved): "
          f"cell_0={cell_counts['0']}, cell_1={cell_counts['1']}, "
          f"cell_2={cell_counts['2']}  "
          f"(low_sf={cell_counts['low_sf']}, "
          f"high_sf={cell_counts['high_sf']}).\n\n",
          render_table(d3_cells, full, "D3_candidate_present"),
          "\n",
          f"- Heterogeneity (F-permutation across non-empty cells "
          f"{het_d3['cells_used']}): "
          f"F = {het_d3['F']:.3f}, permutation p = {fmt_p(het_d3['p_value'])} "
          f"(B={N_PERM}).\n",
          f"- Spearman ρ(cell-mean Δm_par_D3, cell-mean Δm_full) "
          f"over {len(base_cells_used)} non-empty cells "
          f"{base_cells_used}: ρ = {sp_d3_pack['rho']:.3f}, "
          f"p = {sp_d3_pack['p']:.3f}"
          + (f"  (note: {sp_d3_pack.get('note', '')})\n"
             if "note" in sp_d3_pack else "\n"),
          "\n",
          render_table(ev_cells, full, "D_evidence (§3 evidence direction)"),
          "\n",
          f"- D_evidence heterogeneity F = {het_ev['F']:.3f}, "
          f"permutation p = {fmt_p(het_ev['p_value'])}.\n",
          f"- Spearman ρ(cell-mean Δm_par_evidence, cell-mean Δm_full) = "
          f"{sp_ev_pack['rho']:.3f}, p = {sp_ev_pack['p']:.3f}"
          + (f"  (note: {sp_ev_pack.get('note', '')})\n\n"
             if "note" in sp_ev_pack else "\n\n"),
          "### Outcome verdict\n\n",
          f"- **A/B/C verdict (D3): `{outcomes['outcome_abc']}`**\n",
          f"  - low_sf  |Δm_par_D3| / |Δm_full| = "
          f"{outcomes['ratio_low_sf_d3']:.3f}\n",
          f"  - high_sf |Δm_par_D3| / |Δm_full| = "
          f"{outcomes['ratio_high_sf_d3']:.3f}\n",
          f"  - heterogeneity permutation p = "
          f"{fmt_p(outcomes['heterogeneity_p_d3'])}\n",
          f"- **D-flag (D_evidence ↔ D3 cell-ratio similarity): "
          f"`{outcomes['outcome_d_independent_flag']}`**\n",
          f"  - Spearman ρ over base cells = "
          f"{outcomes['spearman_d3_vs_evidence_cell_ratios']['rho']:.3f}, "
          f"p = {outcomes['spearman_d3_vs_evidence_cell_ratios']['p']:.3f} "
          f"(n_cells_used = "
          f"{outcomes['spearman_d3_vs_evidence_cell_ratios']['n_cells_used']})\n",
          "\n### Notes & caveats\n\n",
          "- HotpotQA distribution is highly imbalanced for n_sf_retrieved on the "
          "§3 sample: cell_2 has n=1 (degenerate; bootstrap CI is a point estimate, "
          "and the per-cell permutation test on n=1 returns p=1 by construction). "
          "The high_sf aggregate (n=74) is the inferentially meaningful pooled cell.\n",
          "- The per-cell null band uses the cached K=200 random RMS-matched "
          "directions from §16.3 restricted to the cell mask (band = 95% range of "
          "the 200 cell-mean shifts).\n",
          "- `cell_2` having n=1 means the heterogeneity F-equivalent has effective "
          f"k_cells_nonempty = {het_d3['k_cells_nonempty']}; the permutation null "
          "is computed by shuffling the actual 3-level n_sf labels, so the test is "
          "well-defined despite the imbalance.\n",
          "- D3 was probe-trained on the LGM extractability-toggle corpus (where "
          "T0/T1/N0/S0 exist) and operativeness-measured on §3 HotpotQA prompts. "
          "n_sf_retrieved is HotpotQA's native availability proxy, not a "
          "construction-matched cell. This audit uses it as the closest "
          "HotpotQA-native availability surrogate, not as a reconstruction of "
          "the LGM cells.\n",
          f"\n_Generated: {datetime.now().isoformat(timespec='seconds')}_\n",
          ]
    with open(OUT_DIR / "audit_report.md", "w") as f:
        f.write("".join(md))
    print(f"[done] wrote {OUT_DIR}/")


if __name__ == "__main__":
    main()
