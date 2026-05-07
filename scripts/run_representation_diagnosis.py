#!/usr/bin/env python3
"""Minimal representation-level diagnosis on runtime step1 prompt states.

Compares the projection/separation strength of:
  - search_post_runtime_trace_clean
  - calculator_post_clean

on the exact runtime step1 prompts reconstructed from the already-used 40-sample
HotpotQA bridge sweep artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.prompts import PromptBuilder
from steering.directions import load_direction
from steering.hook_utils import get_model_layers


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_project_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def by_id(records: Iterable[Dict]) -> Dict[str, Dict]:
    return {str(r["sample_id"]): r for r in records}


def extract_runtime_step1_state(record: Dict) -> Dict[str, object]:
    question = str(record["question"])
    for step in record.get("steps", []):
        if step.get("step_idx") != 0:
            continue
        if step.get("action") == "search":
            query = step.get("action_input")
            observation = step.get("observation")
            if not query or observation is None:
                break
            query = str(query)
            observation = str(observation)
            return {
                "kind": "step0_search",
                "question": question,
                "query": query,
                "observation": observation,
                "steps": [{"action": "search", "action_input": query, "observation": observation}],
            }
        if step.get("parse_failure_reason") or step.get("observation") == "[PARSE_FAILURE] No valid action parsed":
            return {
                "kind": "step0_parse_failure",
                "question": question,
                "query": None,
                "observation": None,
                "steps": [],
            }
        break
    raise ValueError(f"Could not reconstruct runtime step1 state for {record.get('sample_id')}")


def build_runtime_step1_prompt(tokenizer, question: str, steps: List[Dict[str, str]]) -> str:
    pb = PromptBuilder()
    messages = pb.build_full_prompt(question, steps)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def get_model_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def load_model(model_name: str, use_4bit: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_name}")
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto", "trust_remote_code": True}
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            print("  Using 4-bit quantization")
        except ImportError:
            print("  bitsandbytes not available; using bf16")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


def extract_hidden_hook(model, tokenizer, prompt: str, layer: int) -> np.ndarray:
    inputs = tokenizer(prompt, return_tensors="pt").to(get_model_device(model))
    layers = get_model_layers(model)
    actual_layer = layer if layer >= 0 else len(layers) + layer
    captured = {}

    def capture_hook(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden

    handle = layers[actual_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()

    if "hidden" not in captured:
        raise RuntimeError(f"Failed to capture hidden state at layer {layer}")
    return captured["hidden"][0, -1, :].float().cpu().numpy().astype(np.float32)


def projection(hidden: np.ndarray, direction: np.ndarray) -> float:
    return float(np.dot(hidden, direction))


def zscore(values: np.ndarray) -> np.ndarray:
    std = float(values.std())
    if std < 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def auc_pairwise(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    total = 0
    for p in pos:
        for n in neg:
            total += 1
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / total if total else float("nan")


def cohens_d(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    pos_var = float(np.var(pos, ddof=1))
    neg_var = float(np.var(neg, ddof=1))
    pooled_num = (len(pos) - 1) * pos_var + (len(neg) - 1) * neg_var
    pooled_den = len(pos) + len(neg) - 2
    pooled = np.sqrt(max(pooled_num / pooled_den, 1e-12))
    return float((pos.mean() - neg.mean()) / pooled)


def summarize_label(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    raw_auc = auc_pairwise(scores, labels)
    raw_d = cohens_d(scores, labels)
    zscores = zscore(scores)
    pos = zscores[labels == 1]
    neg = zscores[labels == 0]
    raw_gap = float(pos.mean() - neg.mean()) if len(pos) and len(neg) else float("nan")
    return {
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
        "mean_gap_z": raw_gap,
        "mean_gap_z_abs": abs(raw_gap),
        "auc_raw": raw_auc,
        "auc_abs": max(raw_auc, 1.0 - raw_auc) if raw_auc == raw_auc else float("nan"),
        "cohens_d_raw": raw_d,
        "cohens_d_abs": abs(raw_d) if raw_d == raw_d else float("nan"),
        "pos_mean_raw": float(scores[labels == 1].mean()) if (labels == 1).any() else float("nan"),
        "neg_mean_raw": float(scores[labels == 0].mean()) if (labels == 0).any() else float("nan"),
    }


def make_plot(rows: List[Dict], out_path: Path):
    labels_meta = [
        ("final_correct", "Final correct"),
        ("oracle_sensitive", "Oracle-sensitive"),
        ("same_observation_any_correct", "Same-observation answerable"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    colors = {0: "#B0B7C3", 1: "#4C78A8"}

    for ax, (label_key, title) in zip(axes, labels_meta):
        search_neg = [r["search_proj_z"] for r in rows if r[label_key] == 0]
        search_pos = [r["search_proj_z"] for r in rows if r[label_key] == 1]
        calc_neg = [r["calc_proj_z"] for r in rows if r[label_key] == 0]
        calc_pos = [r["calc_proj_z"] for r in rows if r[label_key] == 1]
        box = ax.boxplot(
            [search_neg, search_pos, calc_neg, calc_pos],
            positions=[1, 2, 4, 5],
            widths=0.65,
            patch_artist=True,
            showfliers=False,
        )
        for patch, color in zip(box["boxes"], [colors[0], colors[1], colors[0], colors[1]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.axhline(0.0, color="#777", linewidth=0.8, linestyle="--")
        ax.set_xticks([1, 2, 4, 5])
        ax.set_xticklabels(["S-0", "S-1", "C-0", "C-1"])
        ax.set_title(title)
        ax.set_ylabel("z-scored projection")
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Runtime step1 projection distributions (S=search, C=calculator)")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_markdown_table(summary: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    lines = [
        "| Label | Direction | n+ | n- | |Δz| | AUC | |d| |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = [
        ("final_correct", "final correct vs incorrect"),
        ("oracle_sensitive", "oracle-sensitive vs non-oracle-sensitive"),
        ("same_observation_any_correct", "same-observation any-correct vs all-wrong"),
    ]
    for key, title in order:
        for direction_name in ["search", "calculator"]:
            stats = summary[key][direction_name]
            lines.append(
                f"| {title} | {direction_name} | {stats['n_pos']} | {stats['n_neg']} | "
                f"{stats['mean_gap_z_abs']:.3f} | {stats['auc_abs']:.3f} | {stats['cohens_d_abs']:.3f} |"
            )
    return "\n".join(lines)


def decide_conclusion(summary: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, object]:
    labels = list(summary.keys())
    search_auc = np.mean([summary[k]["search"]["auc_abs"] for k in labels])
    calc_auc = np.mean([summary[k]["calculator"]["auc_abs"] for k in labels])
    search_d = np.mean([summary[k]["search"]["cohens_d_abs"] for k in labels])
    calc_d = np.mean([summary[k]["calculator"]["cohens_d_abs"] for k in labels])
    search_auc_wins = sum(summary[k]["search"]["auc_abs"] > summary[k]["calculator"]["auc_abs"] for k in labels)

    clearly_search_stronger = (
        search_auc > calc_auc and search_d > calc_d and search_auc_wins >= 2
    )
    if clearly_search_stronger:
        verdict = "A"
        rationale = (
            "search is stronger at representation-level separation on the runtime step1 states; "
            "the current E2E weakness is more consistent with interface/scorer/generation issues."
        )
    else:
        verdict = "B"
        rationale = (
            "search is not stronger at representation-level separation on the runtime step1 states; "
            "this failed candidate is more consistent with extractor/site/contrast failure."
        )
    return {
        "verdict": verdict,
        "search_auc_abs_mean": float(search_auc),
        "calculator_auc_abs_mean": float(calc_auc),
        "search_cohens_d_abs_mean": float(search_d),
        "calculator_cohens_d_abs_mean": float(calc_d),
        "search_auc_label_wins": int(search_auc_wins),
        "rationale": rationale,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal representation-level diagnosis")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument(
        "--baseline-path",
        default="results/outcome_first_sweep/search_post_runtime_trace_clean_rho051015/baseline_results.jsonl",
    )
    parser.add_argument(
        "--search-path",
        default="results/outcome_first_sweep/search_post_runtime_trace_clean_rho051015/jes_results_tau0.00_rho0.50.jsonl",
    )
    parser.add_argument(
        "--calculator-path",
        default="results/outcome_first_sweep/calculator_post_clean_rho051015/jes_results_tau0.00_rho0.50.jsonl",
    )
    parser.add_argument(
        "--oracle-path",
        default="results/outcome_first_sweep/search_post_runtime_trace_clean_rho051015/oracle_results.jsonl",
    )
    parser.add_argument(
        "--search-direction",
        default="steering/directions/direction_search_post_runtime_trace_clean_eval200_seed42_bridge_v1.npz",
    )
    parser.add_argument(
        "--calculator-direction",
        default="steering/directions/direction_calculator_post_clean_train_v1.npz",
    )
    parser.add_argument(
        "--outdir",
        default="results/representation_diagnosis/hotpotqa_bridge_step1_failed_candidate",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = resolve_project_path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    baseline = by_id(load_jsonl(resolve_project_path(args.baseline_path)))
    search = by_id(load_jsonl(resolve_project_path(args.search_path)))
    calc = by_id(load_jsonl(resolve_project_path(args.calculator_path)))
    oracle = by_id(load_jsonl(resolve_project_path(args.oracle_path)))

    common_ids = sorted(set(baseline) & set(search) & set(calc) & set(oracle))
    if len(common_ids) != 40:
        raise ValueError(f"Expected 40 common samples, got {len(common_ids)}")

    print(f"Loaded {len(common_ids)} shared samples")

    model, tokenizer = load_model(args.model, use_4bit=not args.no_4bit)
    search_direction, search_meta = load_direction(str(resolve_project_path(args.search_direction)), normalize_rms=1.0)
    calc_direction, calc_meta = load_direction(str(resolve_project_path(args.calculator_direction)), normalize_rms=1.0)

    rows = []
    for sid in common_ids:
        bl = baseline[sid]
        sr = search[sid]
        cr = calc[sid]
        orc = oracle[sid]

        state_bl = extract_runtime_step1_state(bl)
        state_sr = extract_runtime_step1_state(sr)
        state_cr = extract_runtime_step1_state(cr)
        sig_bl = (state_bl["kind"], state_bl["question"], state_bl["query"], state_bl["observation"])
        sig_sr = (state_sr["kind"], state_sr["question"], state_sr["query"], state_sr["observation"])
        sig_cr = (state_cr["kind"], state_cr["question"], state_cr["query"], state_cr["observation"])
        if sig_bl != sig_sr or sig_bl != sig_cr:
            raise ValueError(f"Step0 prompt state mismatch across baseline/search/calc for sample {sid}")

        prompt = build_runtime_step1_prompt(tokenizer, str(state_bl["question"]), list(state_bl["steps"]))
        hidden = extract_hidden_hook(model, tokenizer, prompt, args.layer)

        row = {
            "sample_id": sid,
            "question": state_bl["question"],
            "runtime_state_kind": state_bl["kind"],
            "search_query": state_bl["query"],
            "observation": state_bl["observation"],
            "gold_answer": bl.get("gold_answer"),
            "baseline_correct": bool(bl["is_correct"]),
            "search_correct": bool(sr["is_correct"]),
            "calculator_correct": bool(cr["is_correct"]),
            "oracle_correct": bool(orc["is_correct"]),
            "final_correct": int(bool(bl["is_correct"])),
            "oracle_sensitive": int((not bl["is_correct"]) and (not sr["is_correct"]) and (not cr["is_correct"]) and bool(orc["is_correct"])),
            "same_observation_any_correct": int(any([bool(bl["is_correct"]), bool(sr["is_correct"]), bool(cr["is_correct"])])),
            "search_proj": projection(hidden, search_direction),
            "calc_proj": projection(hidden, calc_direction),
        }
        rows.append(row)

    search_scores = np.array([r["search_proj"] for r in rows], dtype=np.float32)
    calc_scores = np.array([r["calc_proj"] for r in rows], dtype=np.float32)
    search_z = zscore(search_scores)
    calc_z = zscore(calc_scores)
    for row, spz, cpz in zip(rows, search_z, calc_z):
        row["search_proj_z"] = float(spz)
        row["calc_proj_z"] = float(cpz)

    summary = {}
    for label_key in ["final_correct", "oracle_sensitive", "same_observation_any_correct"]:
        labels = np.array([r[label_key] for r in rows], dtype=np.int32)
        summary[label_key] = {
            "search": summarize_label(search_scores, labels),
            "calculator": summarize_label(calc_scores, labels),
        }

    conclusion = decide_conclusion(summary)
    markdown_table = render_markdown_table(summary)
    plot_path = outdir / "projection_distributions.png"
    make_plot(rows, plot_path)

    payload = {
        "config": {
            "model": args.model,
            "layer": args.layer,
            "n_samples": len(rows),
            "step1_prompt_template": "runtime PromptBuilder default tools=[search, calculator]",
        },
        "directions": {"search": search_meta, "calculator": calc_meta},
        "label_definitions": {
            "final_correct": "baseline step1 final answer correct vs incorrect on the unsteered runtime state",
            "oracle_sensitive": "baseline/search/calculator all wrong but oracle correct",
            "same_observation_any_correct": "on the shared step0 observation, at least one of baseline/search/calculator finalizes correctly vs all three wrong",
        },
        "summary": summary,
        "conclusion": conclusion,
    }

    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with (outdir / "sample_projections.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (outdir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("## Representation-level diagnosis\n\n")
        f.write(markdown_table)
        f.write("\n\n")
        f.write(f"Conclusion: **{conclusion['verdict']}** — {conclusion['rationale']}\n")

    print("\n=== Summary table ===")
    print(markdown_table)
    print("\n=== Binary conclusion ===")
    print(f"{conclusion['verdict']}: {conclusion['rationale']}")
    print(f"\nArtifacts written to: {outdir}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()