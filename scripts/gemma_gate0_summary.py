#!/usr/bin/env python3
"""Gate 0 summary for Gemma circuit sanity. Composes from existing files only:
  - results/cross_model_gemma2_v2/full_results.json  (probe / orthogonality / corruption)
  - results/cross_model_behavior_alignment/summary_gemma_2_9b_it.json  (N0/T0/S0)

Decision rule: pass iff all four gates pass. Otherwise stop.
"""
import json
from pathlib import Path

OUT_DIR = Path("results/gemma_circuit_sanity")
OUT_DIR.mkdir(parents=True, exist_ok=True)

repr_path = Path("results/cross_model_gemma2_v2/full_results.json")
beh_path  = Path("results/cross_model_behavior_alignment/summary_gemma_2_9b_it.json")
R = json.load(open(repr_path)); B = json.load(open(beh_path))

t0_vs_n0 = B["contrasts"]["T0_vs_N0"]["commit_W"]
g1 = {
    "name": "behavior: T0 commit-W > N0",
    "T0_commit_W": B["cells"]["T0"]["commit_W"],
    "N0_commit_W": B["cells"]["N0"]["commit_W"],
    "delta": t0_vs_n0["delta_rate"],
    "mcnemar_p": t0_vs_n0["mcnemar_p"],
    "passes": t0_vs_n0["delta_rate"] > 0 and t0_vs_n0["mcnemar_p"] < 0.05,
}
g2 = {
    "name": "evidence probe AUROC > 0.75",
    "auroc": R["evidence_probe"]["auroc_mean"],
    "layer": R["evidence_probe"]["layer"],
    "passes": R["evidence_probe"]["auroc_mean"] > 0.75,
}
g3 = {
    "name": "|cos(action, evidence)| < 0.05 (cross-layer)",
    "cos_xlayer":   R["orthogonality"]["cos_action_evidence"],
    "cos_samelayer": R["orthogonality"]["cos_same_layer"],
    "act_layer": R["orthogonality"]["action_layer"],
    "evi_layer": R["orthogonality"]["evidence_layer"],
    "passes": abs(R["orthogonality"]["cos_action_evidence"]) < 0.05 and
              abs(R["orthogonality"]["cos_same_layer"]) < 0.05,
}
g4 = {
    "name": "paired corruption A/B > 1.3 (action) with p<0.05",
    "AB_ratio_action": R["paired_corruption"]["AB_ratio_action"],
    "MW_p_action":     R["paired_corruption"]["MW_action_p"],
    "n_pairs":         R["paired_corruption"]["n_samples"],
    "passes": R["paired_corruption"]["AB_ratio_action"] > 1.3 and
              R["paired_corruption"]["MW_action_p"] < 0.05,
}
all_pass = all(g["passes"] for g in [g1, g2, g3, g4])

summary = {
    "model": R["model"],
    "n_layers": R["n_layers"],
    "hidden_size": R["hidden_size"],
    "peak_evidence_layer": R["peak_evidence_layer"],
    "peak_action_layer":   R["peak_action_layer"],
    "sources": {
        "representation": str(repr_path),
        "behavior":       str(beh_path),
    },
    "gates": {"G1_behavior": g1, "G2_probe": g2, "G3_ortho": g3, "G4_corruption": g4},
    "all_pass": all_pass,
    "decision": "PROCEED to Exp1+Exp2" if all_pass else "STOP — abstraction not present",
}
out_path = OUT_DIR / "gate0_summary.json"
json.dump(summary, open(out_path, "w"), indent=2)
print(f"[wrote] {out_path}")
print(f"\nGate 0 (Gemma-2-9B-it):")
for key, g in summary["gates"].items():
    flag = "PASS" if g["passes"] else "FAIL"
    print(f"  [{flag}] {key}: {g['name']}")
print(f"\nALL_PASS = {all_pass}")
print(f"Decision: {summary['decision']}")
