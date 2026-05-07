#!/usr/bin/env python3
"""Gate 0 summary for Mistral-7B-Instruct-v0.3 circuit sanity.
Composes from existing files:
  - results/cross_model_mistral_v2/full_results.json
  - results/cross_model_behavior_alignment/aligned_model_table.json (mistral_7b_v03 entry)
"""
import json
from pathlib import Path

OUT_DIR = Path("results/mistral_circuit_sanity")
OUT_DIR.mkdir(parents=True, exist_ok=True)

repr_path = Path("results/cross_model_mistral_v2/full_results.json")
table_path = Path("results/cross_model_behavior_alignment/aligned_model_table.json")
R = json.load(open(repr_path))
table = json.load(open(table_path))
B = table["models"]["mistral_7b_v03"]["behavior"]

g1 = {
    "name": "behavior: T0 commit-W > N0",
    "T0_commit_W": B["cells"]["T0"]["commit_W"],
    "N0_commit_W": B["cells"]["N0"]["commit_W"],
    "delta": B["contrast_T0_vs_N0"]["delta_commit_W"],
    "mcnemar_p": B["contrast_T0_vs_N0"]["mcnemar_p_commit_W"],
    "passes": B["contrast_T0_vs_N0"]["delta_commit_W"] > 0
              and B["contrast_T0_vs_N0"]["mcnemar_p_commit_W"] < 0.05,
}
# layer_sweep keyed by str(layer)
evi_L = R["peak_evidence_layer"]; act_L = R["peak_action_layer"]
auroc_at_evi = R["layer_sweep"][str(evi_L)]["auroc"]
cos_evi_layer = R["layer_sweep"][str(evi_L)]["cos_action_evidence"]
cos_act_layer = R["layer_sweep"][str(act_L)]["cos_action_evidence"]
g2 = {"name": "evidence probe AUROC > 0.75",
      "auroc": auroc_at_evi, "layer": evi_L,
      "passes": auroc_at_evi > 0.75}
g3 = {"name": "|cos(action, evidence)| < 0.05",
      "cos_at_evi_layer": cos_evi_layer,
      "cos_at_act_layer": cos_act_layer,
      "act_layer": act_L, "evi_layer": evi_L,
      "passes": abs(cos_evi_layer) < 0.05 and abs(cos_act_layer) < 0.05}
pc = R["paired_corruption"]
g4 = {"name": "paired corruption A/B > 1.3 (action) p<0.05",
      "AB_ratio_action": pc["AB_ratio_action"],
      "MW_p_action": pc["MW_action_p"],
      "n_pairs": pc["n_samples"],
      "passes": pc["AB_ratio_action"] > 1.3 and pc["MW_action_p"] < 0.05}

all_pass = all(g["passes"] for g in [g1, g2, g3, g4])
summary = {
    "model": R["model"], "n_layers": R["n_layers"], "hidden_size": R["hidden_size"],
    "peak_evidence_layer": evi_L, "peak_action_layer": act_L,
    "sources": {"representation": str(repr_path),
                "behavior": str(table_path) + " :: mistral_7b_v03"},
    "gates": {"G1_behavior": g1, "G2_probe": g2, "G3_ortho": g3, "G4_corruption": g4},
    "all_pass": all_pass,
    "decision": "PROCEED to Exp1+Exp2" if all_pass else "STOP — abstraction not present",
}
out_path = OUT_DIR / "gate0_summary.json"
json.dump(summary, open(out_path, "w"), indent=2)
print(f"[wrote] {out_path}\n")
print("Gate 0 (Mistral-7B-Instruct-v0.3):")
for key, g in summary["gates"].items():
    flag = "PASS" if g["passes"] else "FAIL"
    print(f"  [{flag}] {key}: {g['name']}")
print(f"\nALL_PASS = {all_pass}")
print(f"Decision: {summary['decision']}")
