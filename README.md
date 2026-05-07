# Code for "Decodability Is Not Control: Why Tool-Using LLM Agents Commit Before Evidence Is Sufficient" — NeurIPS 2026 Submission

> **Note for reviewers.** This repository accompanies an anonymous NeurIPS 2026
> submission and is hosted via Anonymous GitHub. It contains both the
> experiments cited in the paper and a substantial number of exploratory
> and superseded experiments. The mapping in §"Paper → Script Mapping" below
> points to the scripts that produced the paper's main results. Exploratory
> and superseded directories are documented in §"Superseded Experiments" and
> are included for transparency about the research process.

## Overview

This repository implements the analyses for our paper on the dissociation
between linearly decodable evidence-sufficiency information and operative
directions at the decision token of agentic language models. The codebase
covers:

- Linear probe extraction and evaluation
- Functional decomposition (parallel vs.\ perpendicular causal interventions)
- Cross-prompt activation patching
- Distributed Alignment Search (DAS)
- Dose-response sweeps and bootstrap statistical tests
- Cross-family replication on Qwen / Mistral / Gemma / Llama / R1-Distill
- Within-family scale verification on Qwen-2.5-{7B,14B,32B} and Qwen-3-32B

## Quick start

```bash
# 1. Environment
pip install -r requirements.txt

# 2. Single representative result (Qwen L20 functional decomposition)
python experiments/champion/run_decomposition.py \
       --model qwen2.5-7b --layer 20 --config configs/main.yaml

# 3. Full reproduction (≈40 GPU-hours, 96GB GPU)
bash scripts/reproduce_all.sh
```

Outputs are written to `results/<experiment_name>/` with
`summary.json` + `report.md` for each run.

## Paper → Script Mapping

> Reviewers can locate the script behind any reported number using this table.
> Section numbers refer to the submitted paper.

| Paper § | Result | Script |
|---|---|---|
| §2 — Behavioral matrix | A2 clean matrix (3 families × 2 datasets, 5/6 cells significant) | `experiments/behavioral/run_a2_matrix.py` |
| §3 / §5 — Functional decomposition (Qwen L20) | full +0.910, parallel −0.157, perp +0.909 | `experiments/champion/qwen_l20_decomposition.py` |
| §6 — Anti-cue locality (2×3 factorial) | task-missingness as evidence-local semantic cue | `experiments/locality/anti_cue_factorial.py` |
| §8.3 — CI-hardened decomposition | bootstrap CIs + permutation gap test | `experiments/champion/ci_hardened.py` |
| §9 — Cross-model probes | AUROC 0.77–0.86 across 5 families | `experiments/probes/cross_model_auroc.py` |
| §10 — Qwen circuit localization | L18 attention + KV2 specificity | `experiments/circuit/cross_prompt_patching.py` |
| §15.1 — Agent-format dissociation | B_debiased 79.5% vs D 3.1% | `experiments/motivation/agent_format.py` |
| §15.2 — Probe insufficiency ≠ search | Fisher p=0.32, 95.4% vs 97.2% stop | `experiments/motivation/probe_vs_search.py` |
| §15.7 — Fine-tuning stress test | M2 confirmed under adapter | `experiments/ft_stress/run_phase_c.py` |
| §17 — Rotation 4-way decomposition | rank-1 rotation, three arms | `experiments/rotation/run_three_arms.py` |
| §18 — Dose-response (Qwen L20) | action/evidence slope ratio = 43× | `experiments/dose_response/qwen_sweep.py` |
| §19 — Probe robustness audit (B_structure) | decomposition holds across probe targets | `experiments/audit/probe_robustness.py` |
| §20 — Cross-family CI natural-norm | Qwen L20 / Gemma L37 / Mistral L28 | `experiments/crossfamily/ci_decomposition.py` |
| §21 — DAS | probe-direction IIA = 0.000 | `experiments/das/run_das.py` |
| §27 — Partial-alignment falsification | operative-subspace test (Test B) | `experiments/falsification/partial_alignment.py` |
| §28 — Within-Qwen-family scale | Δ = 0.331 ± 0.011 across 4 checkpoints | `experiments/scale/within_family.py` |
| §29 — Boundary-case characterization | Llama informative null + R1 saturated | `experiments/boundary/llama_r1.py` |

> If a script path differs from the table, search by experiment name —
> filenames are kept descriptive throughout.

## Repository Layout

```
.
├── experiments/         # All experiment scripts, organized by paper section
├── src/                 # Shared library code (probes, interventions, stats)
├── configs/             # YAML configurations for main + ablation runs
├── scripts/             # Top-level reproduction and analysis scripts
├── results/             # Saved outputs (.json, .md per run)
├── data/                # Sample IDs and preprocessing pipelines
├── superseded/          # See §"Superseded Experiments" below
├── exploratory/         # Pilot runs not cited in the paper
├── requirements.txt
├── LICENSE
└── README.md            # This file
```

## Superseded Experiments

The `superseded/` directory contains experiments that were run during the
research process but **are not part of the paper's claims**. They are
included for transparency. Each subdirectory has its own `WHY_SUPERSEDED.md`
explaining the reason. Summary:

- `superseded/popqa_cross_scale/` — Early cross-scale magnitude ratios on
  PopQA (e.g., "24×", "88×", "5.96 d") were found to be ceiling-effect
  artifacts (PopQA EM at floor across 7B/14B/32B). Replaced by the
  within-Qwen-family scale verification reported in §28.
- `superseded/smoke_alpha10/` — Initial calibration runs used α ≈ 10 before
  forward-pass measurement showed actual hidden-state RMS = 1.83 at L20.
  All main runs use α = 28.8 (recalibrated). The α ≈ 10 results are not
  cited.
- `superseded/gemma_commit_w/` — Gemma `commit_W` data is held pending a
  V1 + V3 audit and is not used in any reported result.
- `superseded/mistral_par_rms_renorm/` — Mistral 26.2% par_RMS / full ratio
  was an RMS renormalization artifact resolved by §20's natural-norm
  measurement (par_natural / full = 0.1%).

## Pre-Registered vs.\ Post-Hoc Analytic Decisions

A complete pre-registration vs.\ post-hoc decision table is provided at
`docs/preregistration_table.md`, mirroring the version in the paper's
appendix. This documents which analytic choices (rotation rank, α
calibration, Arm B follow-up at N=483, 4-way decomposition) were fixed
before data collection vs.\ made afterward.

## Environment and Hardware

- Python ≥ 3.10
- PyTorch (see `requirements.txt` for exact version)
- Single GPU with 96GB memory (results were obtained on H100-class hardware)
- Approximate total compute: ~40 GPU-hours for the main paper
  (per-experiment breakdown in `docs/compute_breakdown.md`)

## Datasets

All datasets used are publicly available:

- HotpotQA (CC BY-SA 4.0)
- MuSiQue (CC BY 4.0)
- PopQA (MIT)
- 2WikiMultiHop (Apache 2.0)

Sample IDs and preprocessing pipelines are committed under `data/`. No new
dataset is released as part of this submission.

## License

Anonymized for review. Final license (Apache 2.0) will be applied at
camera-ready de-anonymization.

## A note on code quality

This repository reflects an active research codebase rather than a
production-grade release. Some scripts are exploratory; some directories
contain failed experiments documented as such. The `experiments/` and
`src/` paths corresponding to the paper's main results have been verified
end-to-end against the reported numbers. We will provide a cleaned-up
release alongside the camera-ready version.
