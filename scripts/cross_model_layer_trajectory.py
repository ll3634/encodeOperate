#!/usr/bin/env python3
"""Cross-model layer-wise cos(E_layer, A_layer) trajectory.

Sweeps every other layer from L4 to L_max-2 for one model. Trains a logistic
evidence probe and extracts the p10/p90 action direction at each layer, then
computes cos(E_layer, A_layer) with B=50 bootstrap 95% CI. Skips paired
corruption to keep the per-model run light.

Output:
  <output-dir>/trajectory.json  per-layer rows with auroc, cos, CI, quality
  <output-dir>/run.log          stdout

Usage:
  cd tmc/scripts/e2e_agent
  python scripts/cross_model_layer_trajectory.py \
      --model unsloth/Meta-Llama-3.1-8B-Instruct \
      --output-dir results/cross_layer_cos/llama31
"""

import os, sys, json, argparse, time
import numpy as np
from pathlib import Path
from datetime import datetime

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from steering.hook_utils import get_model_layers
from scripts.cross_model_full import (
    collect_step1_states, collect_popqa_multilayer,
    extract_action_dir_from_popqa, train_probe,
)


def _finite_mask(X):
    return np.isfinite(X).all(axis=tuple(range(1, X.ndim)))


def bootstrap_cos_ci(step1_data, popqa_layer_data, layer, B=50, seed=42):
    """Bootstrap CI for cos(E_layer, A_layer)."""
    rng = np.random.default_rng(seed)

    X_evi = np.array([d["hidden"][layer] for d in step1_data], dtype=np.float32)
    y_evi = np.array([d["label"] for d in step1_data], dtype=np.int32)
    margins = np.array(popqa_layer_data["margins"], dtype=np.float32)
    hiddens = np.array(popqa_layer_data["hiddens"], dtype=np.float32)

    # Filter NaN/Inf samples once up-front
    me = _finite_mask(X_evi)
    X_evi, y_evi = X_evi[me], y_evi[me]
    mp = np.isfinite(margins) & _finite_mask(hiddens)
    margins, hiddens = margins[mp], hiddens[mp]

    n_evi = len(X_evi)
    n_pop = len(margins)
    if n_evi < 20 or n_pop < 20 or len(np.unique(y_evi)) < 2:
        return float("nan"), float("nan"), float("nan"), 0
    cos_samples = []
    for _ in range(B):
        idx_e = rng.integers(0, n_evi, size=n_evi)
        idx_p = rng.integers(0, n_pop, size=n_pop)
        X_b, y_b = X_evi[idx_e], y_evi[idx_e]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            evi_b, _ = train_probe(X_b, y_b, return_cv=False)
        except Exception:
            continue
        m_b, h_b = margins[idx_p], hiddens[idx_p]
        if not np.isfinite(m_b).all():
            continue
        p_lo = float(np.percentile(m_b, 10))
        p_hi = float(np.percentile(m_b, 90))
        lo_mask = m_b <= p_lo
        hi_mask = m_b >= p_hi
        if lo_mask.sum() == 0 or hi_mask.sum() == 0:
            continue
        lo = h_b[lo_mask].mean(0)
        hi = h_b[hi_mask].mean(0)
        a = lo - hi
        nrm = np.linalg.norm(a)
        if not np.isfinite(nrm) or nrm < 1e-12:
            continue
        a = a / nrm
        c = float(np.dot(a, evi_b))
        if np.isfinite(c):
            cos_samples.append(c)
    if not cos_samples:
        return float("nan"), float("nan"), float("nan"), 0
    arr = np.array(cos_samples)
    return (float(np.mean(arr)),
            float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)),
            len(cos_samples))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--popqa-path", default="data/popqa/popqa_test.jsonl")
    ap.add_argument("--labels-path", default="results/phase1_probe/labels.jsonl")
    ap.add_argument("--baseline-trace",
                    default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--n-popqa", type=int, default=400)
    ap.add_argument("--layer-start", type=int, default=4)
    ap.add_argument("--layer-step", type=int, default=2)
    ap.add_argument("--layer-end-offset", type=int, default=2,
                    help="Stop sweep at n_layers - this offset.")
    ap.add_argument("--bootstrap-B", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_lines = []
    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"[{datetime.now().isoformat()}] cross_model_layer_trajectory")
    log(f"  model: {args.model}")
    log(f"  output: {args.output_dir}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_token = os.environ.get("HF_TOKEN", None)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, token=hf_token)

    def _load_and_verify(max_attempts=3):
        for attempt in range(1, max_attempts + 1):
            log(f"loading model (attempt {attempt}/{max_attempts})...")
            t0 = time.time()
            m = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, token=hf_token)
            m.eval()
            bad = []
            for n, p in m.named_parameters():
                if torch.isnan(p).any() or torch.isinf(p).any():
                    bad.append(n)
                    if len(bad) >= 3:
                        break
            if not bad:
                log(f"  loaded+verified in {time.time()-t0:.1f}s")
                return m
            log(f"  [warn] corrupted weights detected after load: {bad}; retrying...")
            del m; torch.cuda.empty_cache()
            time.sleep(5)
        raise RuntimeError(
            f"Model {args.model} loaded with NaN/Inf in weights after {max_attempts} attempts. "
            "Likely GPU memory contention with another process.")

    model = _load_and_verify()

    n_layers = len(get_model_layers(model))
    D = model.config.hidden_size
    log(f"  n_layers={n_layers}, hidden={D}")

    sweep_layers = list(range(args.layer_start, n_layers - args.layer_end_offset + 1,
                              args.layer_step))
    log(f"  sweep_layers ({len(sweep_layers)}): {sweep_layers}")

    log("\n=== Step 1: step-1 hidden states (HotpotQA, multilayer) ===")
    t0 = time.time()
    step1_data = collect_step1_states(
        model, tokenizer, args.labels_path, args.baseline_trace, sweep_layers)
    log(f"  collected {len(step1_data)} samples in {time.time()-t0:.1f}s")

    log(f"\n=== Step 2: PopQA states (N={args.n_popqa}, multilayer) ===")
    t0 = time.time()
    popqa_by_layer = collect_popqa_multilayer(
        model, tokenizer, args.popqa_path, sweep_layers, n=args.n_popqa)
    log(f"  collected in {time.time()-t0:.1f}s")

    del model
    torch.cuda.empty_cache()

    log("\n=== Step 3: per-layer trajectory ===")
    log(f"  {'L':>3}  {'AUROC':>7}  {'cos':>9}  {'cos_lo':>8}  {'cos_hi':>8}  {'|cos|':>7}  {'A_quality':>9}  {'n_evi':>6}")
    rows = []
    for li in sweep_layers:
        X_full = np.array([d["hidden"][li] for d in step1_data], dtype=np.float32)
        y_full = np.array([d["label"] for d in step1_data], dtype=np.int32)
        m = _finite_mask(X_full)
        n_kept = int(m.sum())
        n_drop = int(len(m) - n_kept)
        if n_drop > 0:
            log(f"  [warn] L{li}: dropped {n_drop}/{len(m)} non-finite step-1 samples")
        if n_kept < 20 or len(np.unique(y_full[m])) < 2:
            log(f"  [skip] L{li}: insufficient finite samples (n={n_kept})")
            rows.append({
                "layer": int(li), "auroc": float("nan"), "auroc_std": float("nan"),
                "cos_point": float("nan"), "cos_bootstrap_mean": float("nan"),
                "cos_ci_lo": float("nan"), "cos_ci_hi": float("nan"),
                "bootstrap_B_effective": 0,
                "action_dir_quality": float("nan"),
                "action_margin_min": 0.0, "action_margin_max": 0.0,
                "n_step1_finite": n_kept, "skipped": True,
            })
            continue
        X, y = X_full[m], y_full[m]
        try:
            evi_dir, cv = train_probe(X, y, return_cv=True)
        except Exception as e:
            log(f"  [skip] L{li}: probe failed ({e})")
            rows.append({"layer": int(li), "skipped": True, "error": str(e),
                         "auroc": float("nan"), "cos_point": float("nan"),
                         "cos_ci_lo": float("nan"), "cos_ci_hi": float("nan"),
                         "action_dir_quality": float("nan"),
                         "n_step1_finite": n_kept})
            continue

        act_dir, quality, mstats = extract_action_dir_from_popqa(popqa_by_layer[li])
        if act_dir is None or not np.all(np.isfinite(act_dir)):
            cos_point = float("nan")
        else:
            cos_point = float(np.dot(act_dir, evi_dir))

        cos_mean, cos_lo, cos_hi, B_eff = bootstrap_cos_ci(
            step1_data, popqa_by_layer[li], li, B=args.bootstrap_B)

        row = {
            "layer": int(li),
            "auroc": float(cv["auroc_mean"]),
            "auroc_std": float(cv["auroc_std"]),
            "cos_point": cos_point,
            "cos_bootstrap_mean": cos_mean,
            "cos_ci_lo": cos_lo,
            "cos_ci_hi": cos_hi,
            "bootstrap_B_effective": int(B_eff),
            "action_dir_quality": float(quality) if quality is not None else float("nan"),
            "action_margin_min": float(mstats.get("min", 0.0)),
            "action_margin_max": float(mstats.get("max", 0.0)),
            "n_step1_finite": n_kept,
            "skipped": False,
        }
        rows.append(row)
        log(f"  L{li:<3}  {cv['auroc_mean']:>7.4f}  {cos_point:>+9.5f}  "
            f"{cos_lo:>+8.4f}  {cos_hi:>+8.4f}  {abs(cos_point):>7.4f}  "
            f"{row['action_dir_quality']:>9.4f}  {n_kept:>6}")

    out = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "n_layers": int(n_layers),
        "hidden_size": int(D),
        "sweep_layers": sweep_layers,
        "n_step1_samples": int(len(step1_data)),
        "n_popqa_samples": int(args.n_popqa),
        "bootstrap_B": int(args.bootstrap_B),
        "rows": rows,
    }
    with open(Path(args.output_dir) / "trajectory.json", "w") as f:
        json.dump(out, f, indent=2)
    with open(Path(args.output_dir) / "run.log", "w") as f:
        f.write("\n".join(log_lines))
    log(f"\nsaved: {args.output_dir}/trajectory.json")


if __name__ == "__main__":
    main()
