#!/usr/bin/env python3
"""
Statistical analysis across multiple policy runs.

Input:  directory with {policy}.jsonl + manifest.jsonl
Output: report.md + metrics.json

Includes:
  - Macro metrics: success, cost, regression, rescue per policy
  - Micro metrics (stratified by subset + bootstrap CI):
      stealth_choice_recovery, tool_harmful_protection, indifferent_regression
  - Paired statistics: McNemar, bootstrap 95% CI
  - Sign audit: rho sign vs margin delta correlation

Usage:
  python scripts/analyze_runs.py \
      --run-dir results/popqa_500 \
      --manifest results/popqa_500/manifest.jsonl \
      --out results/popqa_500/analysis
"""

import json, argparse, math
import numpy as np
from typing import Any
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.unified_output import load_records, write_summary
from eval.paired_stats import mcnemar_test, bootstrap_ci, do_no_harm_metrics


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_all_policies(run_dir: Path) -> Dict[str, List[Dict]]:
    """Load all {policy}.jsonl files from a directory."""
    records = {}
    for p in sorted(run_dir.glob("*.jsonl")):
        if p.stem == "manifest":
            continue
        recs = load_records(str(p))
        if recs:
            pname = recs[0].get("policy_name", p.stem)
            records[pname] = recs
    return records


def load_manifest(path: str) -> List[Dict]:
    """Load manifest.jsonl from label_tool_sensitivity."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Macro metrics
# ---------------------------------------------------------------------------

def compute_macro(records: List[Dict]) -> Dict:
    """Compute macro metrics for one policy's records."""
    n = len(records)
    if n == 0:
        return {}
    succ = [r["is_correct"] for r in records]
    tokens = [r.get("tokens_total", 0) for r in records]
    tc = [r.get("tool_calls", 0) for r in records]
    steps = [r.get("steps", 0) for r in records]
    return {
        "n": n,
        "success_rate": sum(succ) / n,
        "avg_tokens": float(np.mean(tokens)),
        "p50_tokens": float(np.percentile(tokens, 50)),
        "p90_tokens": float(np.percentile(tokens, 90)),
        "p95_tokens": float(np.percentile(tokens, 95)),
        "avg_tool_calls": float(np.mean(tc)),
        "avg_steps": float(np.mean(steps)),
    }


# ---------------------------------------------------------------------------
# Micro metrics (stratified by subset)
# ---------------------------------------------------------------------------

def compute_micro(
    policy_recs: List[Dict],
    manifest: List[Dict],
    baseline_recs: Optional[List[Dict]] = None,
) -> Dict:
    """Compute per-subset success + bootstrap CI."""
    label_map = {r["sample_id"]: r for r in manifest}
    po_by_id = {r["sample_id"]: r for r in policy_recs}
    bl_by_id = {r["sample_id"]: r for r in (baseline_recs or [])}

    result = {}
    for label in ["tool_critical", "tool_harmful", "indifferent"]:
        sids = [m["sample_id"] for m in manifest if m["label"] == label]
        if not sids:
            continue
        po_correct = [po_by_id[s]["is_correct"] for s in sids if s in po_by_id]
        n = len(po_correct)
        sr = sum(po_correct) / n if n else 0
        # Bootstrap CI for success rate
        rng = np.random.RandomState(42)
        arr = np.array(po_correct, dtype=float)
        boots = [arr[rng.randint(0, n, n)].mean() for _ in range(5000)]
        result[label] = {
            "n": n,
            "success_rate": sr,
            "ci_lower": float(np.percentile(boots, 2.5)),
            "ci_upper": float(np.percentile(boots, 97.5)),
        }
        # Stealth subdivisions
        if label == "tool_critical":
            for sub in ["stealth_choice", "stealth_query", "stealth_format"]:
                sub_sids = [m["sample_id"] for m in manifest
                            if m["label"] == "tool_critical" and m.get("subdivision") == sub]
                sub_correct = [po_by_id[s]["is_correct"] for s in sub_sids if s in po_by_id]
                sn = len(sub_correct)
                if sn == 0:
                    continue
                s_sr = sum(sub_correct) / sn
                s_arr = np.array(sub_correct, dtype=float)
                s_boots = [s_arr[rng.randint(0, sn, sn)].mean() for _ in range(5000)]
                result[sub] = {
                    "n": sn,
                    "success_rate": s_sr,
                    "ci_lower": float(np.percentile(s_boots, 2.5)),
                    "ci_upper": float(np.percentile(s_boots, 97.5)),
                }
    return result


# ---------------------------------------------------------------------------
# Sign audit: rho sign vs margin delta + action switch
# ---------------------------------------------------------------------------

def sign_audit(jes_recs: List[Dict]) -> Dict:
    """Audit that applied steering moves margins toward the intended target.

    NOTE: The previous implementation compared sign(rho) vs sign(Δmargin). That is
    NOT a valid invariant when the direction has negative slope (i.e., increasing
    rho decreases the margin). In such cases, correct behavior can have rho<0 with
    Δmargin>0.

    This audit instead checks:
      - progress toward m_target = +tau(step) (target_side is assumed positive)
      - implied slope sign via slope_hat = Δm / rho
      - switch rate via crossing margin 0 (tool vs finish preference boundary)
    """

    def _parse_tau_schedule(spec: str) -> dict:
        # spec like "1:3.0,2+:0.5" -> {1: 3.0, "rest": 0.5}
        if not spec:
            return {}
        out = {}
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            k, v = part.split(":")
            k = k.strip()
            v = float(v.strip())
            if k.endswith("+"):
                out["rest"] = v
            else:
                out[int(k)] = v
        return out

    def _tau_for_step(step_1b: int, jes_params: Dict[str, Any]) -> float:
        base_tau = float(jes_params.get("tau", 0.0) or 0.0)
        sched = _parse_tau_schedule(jes_params.get("tau_schedule"))
        if step_1b in sched:
            return float(sched[step_1b])
        if "rest" in sched:
            return float(sched["rest"])
        return base_tau

    rho_vals: List[float] = []
    deltas: List[float] = []
    progressed: List[int] = []
    achieved: List[int] = []
    crossed0: List[int] = []
    slope_hat_sign: List[int] = []  # 1 if >0, 0 if <0 (ignore ~0)

    for r in jes_recs:
        jes_params = r.get("jes_params") or {}
        for d in r.get("decision_trace", []):
            rho = float(d.get("rho", 0.0) or 0.0)
            m_before = d.get("margin_before")
            m_after = d.get("margin_after")
            if abs(rho) < 1e-12 or m_before is None or m_after is None:
                continue

            m_before = float(m_before)
            m_after = float(m_after)

            # Determine step index for tau scheduling.
            step_1b = d.get("step_1b")
            if step_1b is None:
                # Back-compat: older runs only have 0-based `step`.
                step0 = d.get("step")
                step_1b = int(step0) + 1 if step0 is not None else 1
            step_1b = int(step_1b)

            tau = _tau_for_step(step_1b, jes_params)
            # target_side in our runner is always "positive" (should use tool)
            m_target = float(tau)

            dm = m_after - m_before

            # progress = closer to m_target
            tol = 1e-9
            progressed.append(int(abs(m_after - m_target) <= abs(m_before - m_target) + tol))
            achieved.append(int(m_after >= m_target - tol))
            crossed0.append(int((m_before < 0 and m_after >= 0) or (m_before >= 0 and m_after < 0)))

            # implied slope sign: dm / rho
            slope_hat = dm / rho
            if abs(slope_hat) > 1e-12:
                slope_hat_sign.append(int(slope_hat > 0))

            rho_vals.append(rho)
            deltas.append(dm)

    n = len(rho_vals)
    if n < 5:
        return {
            "n_decisions": n,
            "frac_progress_to_target": None,
            "frac_achieved_target": None,
            "frac_crossed_zero": None,
            "frac_delta_positive": None,
            "slope_hat_pos_frac": None,
            "n_positive_rho": int(sum(1 for x in rho_vals if x > 0)),
            "n_negative_rho": int(sum(1 for x in rho_vals if x < 0)),
        }

    rho_arr = np.array(rho_vals, dtype=float)
    dm_arr = np.array(deltas, dtype=float)
    # Correlation is still informative as a magnitude relationship (not a sign invariant).
    corr = float(np.corrcoef(rho_arr, dm_arr)[0, 1]) if len(rho_arr) > 1 else 0.0

    return {
        "n_decisions": n,
        "pearson_rho_m_delta": round(corr, 4),
        "frac_progress_to_target": float(np.mean(progressed)) if progressed else None,
        "frac_achieved_target": float(np.mean(achieved)) if achieved else None,
        "frac_crossed_zero": float(np.mean(crossed0)) if crossed0 else None,
        "frac_delta_positive": float(np.mean(dm_arr > 0)),
        "slope_hat_pos_frac": float(np.mean(slope_hat_sign)) if slope_hat_sign else None,
        "n_positive_rho": int(np.sum(rho_arr > 0)),
        "n_negative_rho": int(np.sum(rho_arr < 0)),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report_md(metrics: Dict, out_path: Path):
    """Generate report.md from metrics dict."""
    lines = ["# E2E Evaluation Report\n"]

    # Table 1: Macro
    lines.append("## Table 1: Macro Results\n")
    hdr = "| Policy | N | Success% | AvgTokens | AvgToolCalls | AvgSteps |"
    lines.append(hdr)
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for pname, m in metrics.get("macro", {}).items():
        lines.append(f"| {pname} | {m['n']} | {m['success_rate']*100:.1f} | "
                     f"{m['avg_tokens']:.0f} | {m['avg_tool_calls']:.2f} | {m['avg_steps']:.1f} |")

    # Micro
    lines.append("\n## Micro Metrics (Stratified by Subset)\n")
    for pname, pm in metrics.get("micro", {}).items():
        lines.append(f"\n### {pname}\n")
        lines.append("| Subset | N | Success% | CI_low | CI_high |")
        lines.append("| --- | --- | --- | --- | --- |")
        for sname, sm in pm.items():
            lines.append(f"| {sname} | {sm['n']} | {sm['success_rate']*100:.1f} | "
                         f"{sm['ci_lower']*100:.1f} | {sm['ci_upper']*100:.1f} |")

    # Paired stats
    lines.append("\n## Paired Statistics (vs Baseline)\n")
    for pname, ps in metrics.get("paired", {}).items():
        mcn = ps.get("mcnemar", {})
        bci = ps.get("bootstrap_success_diff", {})
        dn = ps.get("do_no_harm", {})
        lines.append(f"\n### {pname}")
        lines.append(f"- McNemar p={mcn.get('mcnemar_p', 'N/A')}  "
                     f"(b={mcn.get('b_regressed', 0)}, c={mcn.get('c_rescued', 0)})")
        lines.append(f"- ΔSuccess: {bci.get('observed', 0):+.4f}  "
                     f"95% CI [{bci.get('ci_lower', 0):+.4f}, {bci.get('ci_upper', 0):+.4f}]")
        lines.append(f"- Regression: {dn.get('regression_rate', 0):.1%}  "
                     f"Rescue: {dn.get('rescue_rate', 0):.1%}  "
                     f"Net: {dn.get('net_gain', 0)}")

    # Sign audit
    sa = metrics.get("sign_audit")
    if sa and sa.get("n_decisions"):
        lines.append("\n## Sign Audit (JES)\n")
        lines.append(f"- N steered decisions (rho≠0): {sa['n_decisions']}")
        if sa.get("frac_progress_to_target") is not None:
            lines.append(f"- Progress toward m_target=+tau(step): {sa['frac_progress_to_target']:.1%}")
        if sa.get("frac_achieved_target") is not None:
            lines.append(f"- Achieved m_after ≥ m_target: {sa['frac_achieved_target']:.1%}")
        if sa.get("frac_crossed_zero") is not None:
            lines.append(f"- Crossed margin 0 boundary: {sa['frac_crossed_zero']:.1%}")
        if sa.get("frac_delta_positive") is not None:
            lines.append(f"- Δmargin > 0 rate: {sa['frac_delta_positive']:.1%}")
        if sa.get("slope_hat_pos_frac") is not None:
            lines.append(f"- Implied slope_hat=Δm/rho is positive: {sa['slope_hat_pos_frac']:.1%}")
        if sa.get("pearson_rho_m_delta") is not None:
            lines.append(f"- Pearson(rho, Δm): {sa['pearson_rho_m_delta']}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze E2E evaluation runs")
    parser.add_argument("--run-dir", required=True, help="Dir with {policy}.jsonl files")
    parser.add_argument("--manifest", required=True, help="manifest.jsonl from label_tool_sensitivity")
    parser.add_argument("--out", required=True, help="Output directory for report + metrics")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_recs = load_all_policies(run_dir)
    manifest = load_manifest(args.manifest)
    print(f"Loaded policies: {list(all_recs.keys())}")
    print(f"Manifest: {len(manifest)} samples")

    metrics = {}

    # Macro
    metrics["macro"] = {p: compute_macro(r) for p, r in all_recs.items()}

    # Micro (per policy)
    bl_recs = all_recs.get("baseline")
    metrics["micro"] = {}
    for pname, recs in all_recs.items():
        metrics["micro"][pname] = compute_micro(recs, manifest, bl_recs)

    # Paired stats (each policy vs baseline)
    if bl_recs:
        metrics["paired"] = {}
        bl_by_id = {r["sample_id"]: r for r in bl_recs}
        ind_ids = [m["sample_id"] for m in manifest if m["label"] == "indifferent"]
        for pname, recs in all_recs.items():
            if pname == "baseline":
                continue
            po_by_id = {r["sample_id"]: r for r in recs}
            common = sorted(set(bl_by_id) & set(po_by_id))
            bl_c = [bl_by_id[s]["is_correct"] for s in common]
            po_c = [po_by_id[s]["is_correct"] for s in common]
            paired = {
                "mcnemar": mcnemar_test(bl_c, po_c),
                "bootstrap_success_diff": bootstrap_ci(bl_c, po_c, "success_diff"),
                "bootstrap_rescue": bootstrap_ci(bl_c, po_c, "rescue_rate"),
                "bootstrap_regression": bootstrap_ci(bl_c, po_c, "regression_rate"),
                "do_no_harm": do_no_harm_metrics(bl_recs, recs, ind_ids),
            }
            metrics["paired"][pname] = paired

    # Sign audit (JES only)
    if "jes" in all_recs:
        metrics["sign_audit"] = sign_audit(all_recs["jes"])

    # Output
    write_summary(metrics, str(out / "metrics.json"))
    generate_report_md(metrics, out / "report.md")
    print(f"\nAll analysis outputs in {out}/")


if __name__ == "__main__":
    main()

