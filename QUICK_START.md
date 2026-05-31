# Quick Start: Verify-Critical Mining v5

## Run Full Pipeline (PopQA)

```bash
cd <repo-root>

python scripts/run_verify_critical_pipeline.py \
  --data-path data/popqa/popqa_test.jsonl \
  --corpus-path data/popqa/corpus.jsonl \
  --direction-path steering/directions/direction_search_v3.npz \
  --n-samples 200 \
  --tau-sweep 0.0 0.1 \
  --max-rho-sweep 0.25 0.75 1.5 \
  --out results/verify_critical_v5
```

## Run Control Budget Diagnosis Only

```bash
python scripts/control_budget_diagnosis.py \
  --results-dir results/verify_critical_v4
```

## Key Features

### Decision-Only Steering
- Steering now only affects the first forward pass (decision token)
- Prevents cumulative side effects during generation
- Controlled via `max_interventions=1` in `SteeringHook`

### Control Budget Diagnosis
- Computes ρ* (required steering strength) distribution
- Generates flip-feasibility table at multiple thresholds
- Issues GO/NO-GO verdict based on:
  - VC density ≥ 5%
  - Feasible flips at ρ ≤ 1.5 ≥ 30%
  - Any rescue observed

### Pipeline Sweep
- Grid search over (tau, max_rho) combinations
- Per-config JSONL output
- Automatic best-config selection
- Integrated diagnosis phase

## Output Files

```
results/verify_critical_v5/
├── baseline_results.jsonl          # Baseline (1-hop) results
├── oracle_results.jsonl            # Oracle (2-hop) results
├── jes_results.jsonl               # Best JES config results
├── jes_results_tau0.00_rho0.25.jsonl  # Per-config results
├── jes_results_tau0.00_rho0.75.jsonl
├── jes_results_tau0.00_rho1.50.jsonl
├── jes_results_tau0.10_rho0.25.jsonl
├── ... (more configs)
├── manifest.jsonl                  # Sample labels (VC/VH/IND)
├── diagnosis.json                  # Control budget diagnosis
├── report.json                     # Summary statistics
└── report.md                       # Human-readable report
```

## Interpretation

**GO Verdict**: Dataset is viable for steering-based improvement
- VC density ≥ 5% (enough samples to rescue)
- Feasible flips at ρ ≤ 1.5 ≥ 30% (steering budget sufficient)
- Rescue observed (JES actually helps)

**NO-GO Verdict**: Dataset not viable
- Low VC density (not enough boundary cases)
- High ρ* required (steering budget insufficient)
- No rescue observed (steering doesn't help)

## Troubleshooting

**Model Download Error**: Network/proxy issue
- Solution: Use locally cached model or disable proxy

**Missing Dependencies**: `transformers`, `torch`, etc.
- Solution: `pip install torch transformers accelerate peft numpy scipy scikit-learn pandas matplotlib pyyaml tqdm`

**HotpotQA Data**: Need to download from https://hotpotqa.github.io/
- Place the distractor dev file at `data/hotpotqa/hotpot_dev_distractor_v1.json` and the
  matching corpus at `data/hotpotqa/corpus.jsonl`; both paths are referenced by
  `run_hotpotqa_v3.sh` and `configs/hotpotqa.yaml`.

