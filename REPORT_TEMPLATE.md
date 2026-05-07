# JES E2E Evaluation Report (Template)

> Fill in paths + numbers from your run directory (e.g., `results/gsm8k_muscle_YYYYMMDD_n500_s42/`).

## 0. Setup / Reproducibility

- Model: `<MODEL>`
- Direction: `<DIRECTION_PATH>` (layer `<LAYER>`, position `<POS>`)
- JES config: tau schedule `<SCHED>`, rho_max `<RHO_MAX>`, eps `<EPS>` (+ adaptive eps probing enabled)
- Seeds: `<SEEDS>`
- Run dir: `<RUN_DIR>`

## 1. Table 1 — Macro (Overall)

| Policy | N | Success% | Avg tokens | Avg tool calls | Avg steps |
| --- | ---:| ---:| ---:| ---:| ---:|
| baseline |  |  |  |  |  |
| force_adopt |  |  |  |  |  |
| force_reject |  |  |  |  |  |
| jes |  |  |  |  |  |

**Notes**
- Report paired deltas vs baseline and include confidence intervals.

## 2. Micro-control (Counterfactual Subsets)

**Definition (counterfactual, not heuristic):**
- tool_critical: baseline fail & force_adopt succeed
- tool_harmful: baseline succeed & force_adopt fail
- indifferent: otherwise
- stealth_choice/query/format: subdivisions within tool_critical

### Headline table (paper-ready)

| Policy | tool_critical | stealth_choice | tool_harmful | indifferent |
| --- | ---:| ---:| ---:| ---:|
| baseline |  |  |  |  |
| force_adopt |  |  |  |  |
| force_reject |  |  |  |  |
| jes |  |  |  |  |

**Headline claim to support (capability):**
- `stealth_choice` recovery: baseline → JES (Δpp)

## 3. Macro-safety (Do-no-harm)

Report on `indifferent` (and optionally overall):
- Regression rate (baseline ok → JES fail)
- Rescue rate (baseline fail → JES ok)
- Net gain
- Unnecessary tool use rate / step inflation (if tracked)

## 4. Pareto / Cost efficiency

Plot + summarize:
- Success vs avg tokens
- Success vs avg tool calls
- Optional: p50/p90/p95 cost distributions

## 5. Sign audit / Steering sanity

- Correlation between `rho` sign and `margin_after - margin_before`
- Fraction of steps with sign mismatch
- (Optional) Distribution of `eps_effective` for JES steps

## 6. Statistics

Paired tests vs baseline:
- McNemar test p-value
- Bootstrap CI for ΔSuccess, Regression, Rescue

## 7. Artifacts

- Records: `{policy}.jsonl`
- Manifest: `manifest.jsonl`
- Analysis: `analysis/report.md`, `analysis/metrics.json`
- Figures: `figures/micro.png`, `figures/pareto.png`, ...

