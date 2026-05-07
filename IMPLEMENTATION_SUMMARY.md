# Verify-Critical Mining v5: Implementation Summary

## Overview
Completed implementation of decision-only steering, control budget diagnosis, and pipeline overhaul for the Verify-Critical Mining (VCM) methodology.

## Changes Made

### 1. Decision-Only Steering ✅
**Goal**: Restrict steering to only the decision token (Action vs Final), preventing cumulative side effects during generation.

**Files Modified**:
- `steering/hook_utils.py`: Added `max_interventions` parameter to `SteeringHook`
  - Tracks `_intervention_count` in hook function
  - Auto-disables after N forward passes
  - Resets counter in `remove()` method

- `agent/policies.py`: Added `decision_only: bool = False` field to `SteeringDecision`

- `agent/react_loop.py`: Modified `_generate_step()` to check `decision_only` flag
  - Passes `max_interventions=1` to `SteeringHook` when `decision_only=True`

- `agent/policies_verify.py`: Updated both JES policies
  - `JESStep2OnlyPolicy.decide()` sets `decision_only=True`
  - `JESStep2OnlyWithMarginPolicy.decide()` sets `decision_only=True`

### 2. Control Budget Diagnosis ✅
**Goal**: Compute ρ* distribution and determine Go/No-Go feasibility for a dataset.

**New File**: `scripts/control_budget_diagnosis.py`
- `compute_diagnosis()`: Analyzes baseline/oracle/JES results
  - Identifies verify-critical samples (baseline fails, oracle succeeds)
  - Computes ρ* (required steering strength) for each VC sample
  - Generates flip-feasibility table at thresholds: 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0
  - Issues GO/NO-GO verdict based on:
    - VC density ≥ 5%
    - Feasible flips at ρ ≤ 1.5 ≥ 30%
    - Any rescue observed

- `print_diagnosis()`: Pretty-prints diagnostic results
- `main()`: CLI interface

**Usage**:
```bash
python scripts/control_budget_diagnosis.py --results-dir results/verify_critical_v4
```

### 3. Pipeline Overhaul ✅
**Goal**: Automate sweep over (tau, max_rho) grid and integrate diagnosis.

**File Modified**: `scripts/run_verify_critical_pipeline.py`

**New Arguments**:
- `--tau-sweep`: List of tau values (e.g., `0.0 0.1`)
- `--max-rho-sweep`: List of max_rho values (e.g., `0.25 0.75 1.5`)

**New Phases**:
1. Load model/direction/dataset
2. Mining (baseline + oracle)
3. JES sweep over (tau, max_rho) grid
4. Control budget diagnosis
5. Final report generation

**Outputs**:
- Per-config JSONL: `jes_results_tau0.00_rho0.25.jsonl`
- Canonical best: `jes_results.jsonl` (copied from best config)
- Diagnosis: `diagnosis.json`
- Report: `report.json` + `report.md`

**Usage**:
```bash
python scripts/run_verify_critical_pipeline.py \
  --data-path data/popqa/popqa_test.jsonl \
  --corpus-path data/popqa/corpus.jsonl \
  --direction-path steering/directions/direction_search_v3.npz \
  --n-samples 200 \
  --tau-sweep 0.0 0.1 \
  --max-rho-sweep 0.25 0.75 1.5 \
  --out results/verify_critical_v5
```

### 4. HotpotQA Dataset Adapter ✅
**File Modified**: `datasets/hotpotqa.py`
- Added `answers: List[str]` field to `HotpotQASample`
- Populated with `[answer]` during load for pipeline compatibility
- Maintains same interface as PopQA (id, question, answer, answers, should_use_tool)

## Verification Results

✅ All 7 modified files pass syntax check
✅ All imports resolve correctly
✅ SteeringDecision has `decision_only` field
✅ SteeringHook accepts `max_interventions` parameter
✅ HotpotQASample has `answers` field
✅ Control budget diagnosis runs successfully on v4 data

## v4 Diagnosis Results

```
Sample Distribution (n=200):
  verify_critical:  7  (3.5%)
  verify_harmful:   2
  indifferent:      191

rho* Distribution (VC, n=7):
  mean=2.734  median=1.673
  p25=1.210  p75=2.001  p90=5.474

Flip-Feasibility at max_rho=1.5:
  VC: 3/7 (42.9%)
  All: 31/193 (16.1%)

VERDICT: NO-GO
  Reason: VC density 3.5% < 5% threshold
```

## Next Steps

1. Run pipeline on PopQA with new sweep parameters
2. If PopQA NO-GO, run same pipeline on HotpotQA
3. Analyze sweep results to find best (tau, max_rho) configuration
4. Validate decision-only steering prevents cumulative side effects

