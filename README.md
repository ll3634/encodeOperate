# Code for "Decodability Is Not Control: Why Tool-Using LLM Agents Commit Before Evidence Is Sufficient" — NeurIPS 2026 Submission

> **Note for reviewers.** This repository accompanies an anonymous NeurIPS
> 2026 submission and is hosted via Anonymous GitHub. It is an active
> research codebase: the directories listed under "Repository Layout"
> contain the library code used by every experiment in the paper, and
> `scripts/` contains the per-experiment entry points (both the runs
> cited in the paper and exploratory scripts kept for transparency about
> the research process).

## Overview

This repository implements the analyses for our paper on the dissociation
between linearly decodable evidence-sufficiency information and operative
directions at the decision token of tool-using language-model agents. The
codebase covers:

- ReAct-style agent runners over search / calculator tools
- Linear probe extraction and evaluation at the decision token
- Functional decomposition (parallel vs. perpendicular causal interventions)
- Steering-direction extraction, hooking, and the JES policy
- Activation / KV / path patching for circuit localization
- Dose-response, rotation, and decomposition sweeps with bootstrap CIs
- Cross-family / within-family replication on Qwen, Mistral, Gemma, Llama
  and R1-Distill checkpoints

## Quick start

A worked end-to-end example (the Verify-Critical mining pipeline on PopQA
and HotpotQA) lives in [`QUICK_START.md`](QUICK_START.md). The minimum
needed to reproduce the main steering runs is:

```bash
# 1. Environment (PyTorch + 🤗 transformers + numpy/scipy/scikit-learn/matplotlib).
#    No requirements.txt is committed; install versions appropriate for
#    your CUDA/driver setup.
pip install torch transformers accelerate peft \
            numpy scipy scikit-learn pandas matplotlib pyyaml tqdm

# 2. Verify-Critical pipeline on PopQA (baseline + oracle + JES sweep + diagnosis)
python scripts/run_verify_critical_pipeline.py \
    --data-path data/popqa/popqa_test.jsonl \
    --corpus-path data/popqa/corpus.jsonl \
    --direction-path steering/directions/direction_search_v3.npz \
    --n-samples 200 \
    --tau-sweep 0.0 0.1 \
    --max-rho-sweep 0.25 0.75 1.5 \
    --out results/verify_critical_v5

# 3. HotpotQA full sweep (see run_hotpotqa_v3.sh for the exact command line)
bash run_hotpotqa_v3.sh
```

Each runner writes its outputs (JSONL traces, per-config sweep files,
`report.json`, `report.md`) under `results/<experiment_name>/`. See
`QUICK_START.md` for the output layout and the GO / NO-GO interpretation
of the diagnosis files.

## Paper → Script Mapping

Section numbers refer to the submitted paper. Each row lists the primary
entry-point script; closely related helpers (figure generation,
aggregation, follow-up audits) are listed in parentheses.

| Paper § | Result | Script(s) |
|---|---|---|
| §2 — Behavioral matrix | Margin-projection A/B matrix across model families × datasets | `scripts/b2_margin_projection_matrix.py` (with `scripts/behavioral_readout.py`) |
| §3 / §5 — Functional decomposition (Qwen L20) | full / parallel / perpendicular causal intervention at ρ = −0.20 | `scripts/run_decomposition_test.py` |
| §6 — Anti-cue locality (2×3 factorial) | target_location × wrapper_semantics paired contrasts and locality patch | `scripts/analyze_factorial_2x3.py` (with `scripts/patch_mlp20_task_missingness_locality.py`) |
| §8.3 — CI-hardened decomposition | bootstrap CIs + permutation null on cross-model decomposition | `scripts/decomposition_ci_hardened_cross_model.py` (with `scripts/cos_evidence_action_bootstrap.py`) |
| §9 — Cross-model probes (AUROC) | Per-family probe / action-direction extraction + paired corruption | `scripts/cross_model_full.py` |
| §10 — Qwen circuit localization (L18 attention, KV2) | KV-group split of `attn_L18` cross-prompt patching | `scripts/patch_L18_kv_groups.py` (with `scripts/kv2_ablation_probe_auroc.py`) |
| §15.1 — Agent-format dissociation | B\_debiased vs D in the agent-vs-base contrast | `scripts/agent_specific_dissociation.py` |
| §15.2 — Probe insufficiency ≠ search | Clean-sufficiency probe via synthetic 1-SF → 2-SF augmentation | `scripts/probe_sufficiency_synthetic.py` |
| §15.7 — Fine-tuning stress test (Phase C verdict) | M1 / M2 verdict over per-adapter decompositions | `scripts/ft_in_adapter_verdict.py` (with `scripts/ft_in_adapter_decomposition.py`, `scripts/ft_in_adapter_aggregate_balanced.py`) |
| §17 — Rotation 4-way decomposition | Matched-geometry erasure scan along E → {D3, D1, random}; figure + permutation tests | `scripts/nullspace_rotation_scan.py` (with `scripts/fig3_rotation_plot.py`, `scripts/rotation_significance.py`) |
| §18 — Dose-response (Qwen L20, gain ratio) | Per-direction ρ-sweep + slope / gain-ratio readout | `scripts/dose_response_gain_ratio.py` |
| §19 — Probe robustness audit | 10-seed probe re-fit + §8.3 decomposition replay | `scripts/probe_robustness_sweep.py` |
| §20 — Cross-family CI natural-norm | Natural-norm parallel/perp decomposition across Qwen / Gemma / Mistral | `scripts/fig3_natural_norm_decomp.py` (with `scripts/decomposition_ci_hardened_cross_model.py`) |
| §21 — DAS | Distributed Alignment Search for evidence vs action | `scripts/das_evidence_action.py` |
| §27 — Partial-alignment falsification | Gatekeeping Gates 1–3 (margin distribution, forced push, slope) | `scripts/run_gatekeeping_experiments.py` |
| §28 — Within-Qwen-family scale | Scaling-law difficulty audit on Qwen2.5-{7B,14B,32B} + Qwen3-32B; figure | `scripts/scaling_difficulty_audit.py` (with `scripts/_figure1_invariance.py`, `scripts/_robustness_st_contrast_layersweep.py`) |
| §29 — Boundary-case Llama / R1 | Llama-3.1-8B paired-corruption root-cause + R1 margin-action decoupling | `scripts/llama_root_cause.py` (with `scripts/r1_decoupling_mechanism.py`, `scripts/r1_narrative_robustness.py`) |

If a script path differs from the table, search by experiment name —
filenames in `scripts/` are kept descriptive throughout.

## Repository Layout

```
.
├── agent/                 # ReAct loop, prompts, baseline / verify policies
├── analysis/              # Post-hoc analysis & plotting (Pareto, red-flag, corruption)
├── checkpoints/           # Pre-computed steering directions (direction.npz)
├── configs/               # YAML configs (hotpotqa, gaia, truthfulqa, corruption sweep)
├── eval/                  # Scorers, metrics, paired stats, subset labeling
├── reporting/             # summarize_runs.py, make_figures.py
├── runners/               # Per-dataset runners (HotpotQA, PopQA, GSM8K, TruthfulQA, GAIA, controls)
├── scripts/               # Per-experiment entry points (~320 scripts, see below)
├── steering/              # Direction extraction, JES, hook utilities; cached directions/
├── tools/                 # search_tool, calculator_tool, corruption
├── run_agent_dissociation.sh
├── run_hotpotqa_v3.sh
├── run_positive_control.sh
├── run_qwen3_sanity.sh
├── run_thought_erosion.sh # Top-level launchers for the experiments cited in the paper
├── test_pipeline_logic.py # Standalone unit checks for the pipeline plumbing
├── QUICK_START.md
├── LICENSE
└── README.md              # This file
```

### `runners/` vs. `scripts/`

- `runners/` contains the **dataset-level** entry points
  (`run_hotpotqa.py`, `run_popqa.py`, `run_gsm8k.py`,
  `run_truthfulqa.py`, `run_gaia_subset.py`,
  `run_red_flag_experiment.py`, `run_controls.py`) that load a YAML in
  `configs/` and call into `agent/` + `steering/` + `eval/`.
- `scripts/` contains the **experiment-specific** code that produced the
  individual analyses in the paper (direction extraction, decomposition,
  patching, dose-response, cross-model replications, anti-cue
  specificity, etc.) as well as exploratory and debugging scripts.
  Filenames are kept descriptive; scripts that begin with a leading
  underscore (e.g. `_debug_*`, `_diag_*`, `_v1_*`) are intermediate /
  diagnostic and not required to reproduce any reported number.

### Top-level launchers

The shell scripts in the repository root are the exact commands used to
launch the experiments cited in the paper. They invoke entries under
`scripts/` and pin the relevant flags:

| Launcher | Underlying script | Purpose |
|---|---|---|
| `run_hotpotqa_v3.sh` | `scripts/run_verify_critical_pipeline.py` | HotpotQA Verify-Critical pipeline (full τ × ρ sweep, bridge subset) |
| `run_agent_dissociation.sh` | `scripts/agent_specific_dissociation.py` | Agent-specific dissociation test (full N=486) |
| `run_thought_erosion.sh` | `scripts/thought_erosion_probe.py` | L20 thought-erosion probe across 5 generation positions |
| `run_positive_control.sh` | `scripts/positive_control_judge.py` | Positive-control validity check for judge conditions A / A_v2 |
| `run_qwen3_sanity.sh` | `scripts/qwen3_circuit_sanity.py` | Cross-family Qwen3-32B circuit sanity (waits for download, then runs) |

Each launcher `cd`s into its own directory (`cd "$(dirname "$0")"`), so
invoking them from anywhere — e.g. `bash run_hotpotqa_v3.sh` from the
repository root — resolves the relative paths inside (`scripts/...`,
`data/...`, `steering/directions/...`, `results/...`) against the
repository root. `run_qwen3_sanity.sh` additionally honours the
`MODEL_DIR` and `LOG` environment variables for locating the
Qwen3-32B snapshot.

## Environment and Hardware

- Python ≥ 3.10
- PyTorch + 🤗 `transformers` (a recent version that supports the target
  model architectures: Qwen2 / Qwen3, Mistral, Gemma, Llama, R1-Distill)
- Single GPU; main Qwen-2.5-7B runs fit on a 24 GB card, Qwen-3-32B and
  the cross-family 32B replications need ≥ 80 GB (H100-class) or
  multi-GPU `device_map=auto`
- Approximate total compute for the paper: ~40 GPU-hours on H100-class
  hardware, dominated by the cross-family sweeps and the Verify-Critical
  HotpotQA full sweep

## Models, directions, and datasets

- **Models** are downloaded on first use from Hugging Face (e.g.
  `Qwen/Qwen2.5-7B-Instruct`); none are committed.
- **Steering directions** used by the paper live under
  `steering/directions/` (`direction_search_v3.npz`,
  `direction_calculator_*.npz`, `direction_decomp_*_layer20*.npz`, …).
  A single representative direction is also cached at
  `checkpoints/direction.npz`.
- **LoRA adapter weights** are not committed (they exceed GitHub's 100 MB
  per-file limit and are excluded by `.gitignore`); they can be
  regenerated from the corresponding training scripts under `scripts/`.
- **Datasets** used are all publicly available — HotpotQA (CC BY-SA 4.0),
  MuSiQue (CC BY 4.0), PopQA (MIT), 2WikiMultiHop (Apache 2.0). The
  expected on-disk layout is `data/<dataset>/<file>` as referenced by the
  YAML configs in `configs/` and by the top-level launchers. No new
  dataset is released as part of this submission.

## License

Anonymized for review. Final license (Apache 2.0) will be applied at
camera-ready de-anonymization.

## A note on code quality

This repository reflects an active research codebase rather than a
production-grade release. `scripts/` contains exploratory and superseded
runs alongside the scripts cited in the paper; these are retained for
transparency about the research process. A cleaned-up release will
accompany the camera-ready version.
