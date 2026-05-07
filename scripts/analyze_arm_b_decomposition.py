"""
Decompose Arm B (rank-1 rotation) and canonical action steering into:
  (1) P(search triggered | injection)
  (2) P(rescue | newly triggered 2nd search)
  (3) Selectivity = rescued / (rescued + regressed)

No new GPU run required — operates entirely on existing N=483 result files.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BL_PATH    = REPO_ROOT / "results/decomposition_test/baseline_results.jsonl"
FULL_PATH  = REPO_ROOT / "results/decomposition_test/full_results.jsonl"
ARMB_PATH  = REPO_ROOT / "results/reconnection_sweep_arm_b_n483/B_rotate_ev->act_results.jsonl"
OUT_DIR    = REPO_ROOT / "results/reconnection_arm_b_decomposition"


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()]


def _n_searches(record: dict) -> int:
    return sum(1 for s in record.get("steps", []) if s.get("action") == "search")


def _has_pf(record: dict) -> bool:
    if record.get("failure_reason"):
        return True
    for s in record.get("steps", []):
        if s.get("action") == "parse_failure" or s.get("parse_failure_reason"):
            return True
    return False


def decompose(name: str, baseline_by: dict, steered_by: dict, ids: list[str]) -> dict:
    n_total = len(ids)
    n2_baseline = n2_steered = 0
    triggered_new_2nd = triggered_rescued = triggered_regressed = 0
    rescued = regressed = rescued_via_search = 0
    n_pf = 0
    for sid in ids:
        b, p = baseline_by[sid], steered_by[sid]
        bs, ps = _n_searches(b), _n_searches(p)
        bc, pc = b["is_correct"], p["is_correct"]
        if _has_pf(p):
            n_pf += 1
        if bs >= 2:
            n2_baseline += 1
        if ps >= 2:
            n2_steered += 1
        if bs < 2 and ps >= 2:
            triggered_new_2nd += 1
            if not bc and pc:
                triggered_rescued += 1
            if bc and not pc:
                triggered_regressed += 1
        if not bc and pc:
            rescued += 1
            if ps > bs:
                rescued_via_search += 1
        if bc and not pc:
            regressed += 1
    sel = rescued / (rescued + regressed) if (rescued + regressed) else None
    p_rescue_per_2nd = rescued / n2_steered if n2_steered else None
    p_rescue_per_trig = triggered_rescued / triggered_new_2nd if triggered_new_2nd else None
    p_regr_per_trig = triggered_regressed / triggered_new_2nd if triggered_new_2nd else None
    return {
        "name": name,
        "n": n_total,
        "n2_baseline": n2_baseline,
        "n2_steered": n2_steered,
        "triggered_new_2nd": triggered_new_2nd,
        "triggered_rescued": triggered_rescued,
        "triggered_regressed": triggered_regressed,
        "rescued": rescued,
        "regressed": regressed,
        "rescued_via_search": rescued_via_search,
        "parse_failures": n_pf,
        "selectivity": sel,
        "p_rescue_per_steered_2nd_search_sample": p_rescue_per_2nd,
        "p_rescue_per_newly_triggered": p_rescue_per_trig,
        "p_regression_per_newly_triggered": p_regr_per_trig,
    }


def main() -> None:
    bl   = _load(BL_PATH)
    full = _load(FULL_PATH)
    armB = _load(ARMB_PATH)
    bl_by   = {r["sample_id"]: r for r in bl}
    full_by = {r["sample_id"]: r for r in full}
    armB_by = {r["sample_id"]: r for r in armB}
    ids = sorted(set(bl_by) & set(full_by) & set(armB_by))
    assert len(ids) == 483, f"expected 483 paired ids, got {len(ids)}"

    bls   = decompose("baseline (no intervention)", bl_by, bl_by, ids)
    fulls = decompose("canonical action steering (L20, ρ=−0.20)",      bl_by, full_by, ids)
    armBs = decompose("Arm B rank-1 rotation evidence→action (L20, α=+28.8)", bl_by, armB_by, ids)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_paired": len(ids),
        "baseline": bls,
        "canonical_action": fulls,
        "arm_b_rotation": armBs,
        "definitions": {
            "n2_steered": "# samples with >=2 tool searches in steered run",
            "triggered_new_2nd": "samples where baseline had <2 searches AND steered has >=2 (intervention triggered the 2nd search)",
            "selectivity": "rescued / (rescued + regressed)",
            "p_rescue_per_newly_triggered": "(triggered AND rescued) / triggered",
            "p_regression_per_newly_triggered": "(triggered AND regressed) / triggered",
        },
    }
    with (OUT_DIR / "decomposition.json").open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
