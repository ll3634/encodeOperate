#!/usr/bin/env python3
"""
OCFT — Stage 1: Build candidate directions D1..D4 at L20.

For each candidate, this script:
  - extracts L20 hidden states at the p0 last-token position
  - trains a logistic-regression probe on a labelled binary contrast
  - reports balanced-acc / AUROC on a held-out 20% stratified split
  - decomposes A_L20 (action direction) into A_par_k (along D_k) + A_perp_k
  - writes:
      steering/directions/direction_decomp_parallel_DK_layer20.npz
      steering/directions/direction_decomp_perp_DK_layer20.npz
      results/ocft/probes_summary.json
      results/ocft/per_candidate/<DK>/{X.npz, labels.json, probe.json}

Candidates (binary contrasts on L20 p0 hidden state):
  D1: source dataset      (hotpotqa vs musique)
  D2: action prior        (margin_before > 0  vs  <= 0)  on hotpot 486
  D3: candidate present   (T0/T1 vs N0)        on extractability pairs
  D4: observation length  (token_len > median) on extractability pairs

NOTE: this script does ONLY the probe-training + direction-construction step.
The downstream injection is run by scripts/ocft_run_injection.py.

Usage:
    cd tmc/scripts/e2e_agent
    python scripts/ocft_build_directions.py
"""

import os, sys, json, argparse, re
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

LAYER = 20
SEED = 20260502


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--musique-source",
                    default="data/extractability_train/train_N0.jsonl",
                    help="JSONL with prompt_messages for non-hotpot p0 prompts")
    ap.add_argument("--musique-n", type=int, default=200)
    ap.add_argument("--pairs-source",
                    default="results/extractability_support_toggle/pairs.jsonl",
                    help="JSONL with question/observation/condition/token_len")
    ap.add_argument("--pairs-n", type=int, default=200)
    ap.add_argument("--hotpot-cache",
                    default="results/phase1_probe/activations_multilayer.npz")
    ap.add_argument("--hotpot-labels",
                    default="results/phase1_probe/labels.jsonl")
    ap.add_argument("--action-direction",
                    default="steering/directions/direction_decomp_full_layer20.npz")
    ap.add_argument("--out-dir", default="results/ocft")
    ap.add_argument("--steering-dir", default="steering/directions")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--seed", type=int, default=SEED)
    return ap.parse_args()


# ── Helpers for prompt building ─────────────────────────────────────────────

_RE_TRAIN_USER = re.compile(
    r"^(?P<question>.+?)\n\nI have already run a search for you\.\n"
    r"Tool: search\nTool input: (?P<query>.+?)\nTool result:\n(?P<obs>.+)$",
    re.DOTALL,
)


def parse_train_user(content: str):
    m = _RE_TRAIN_USER.match(content)
    if not m:
        return None
    return m.group("question"), m.group("query"), m.group("obs")


def build_p0_prompt(tokenizer, question: str, query: str, observation: str,
                    obs_max_chars: int = 1500) -> str:
    """§3-aligned p0 prompt: system + user(question) + assistant(scratchpad)."""
    from agent.prompts import PromptBuilder
    pb = PromptBuilder(tools=["search", "calculator"])
    steps = [{"action": "search", "action_input": query,
              "observation": observation[:obs_max_chars]}]
    msgs = pb.build_full_prompt(question, steps)
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "per_candidate").mkdir(parents=True, exist_ok=True)
    Path(args.steering_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  OCFT — STAGE 1: build candidate directions D1..D4")
    print("=" * 72)
    print(f"  layer={LAYER}  seed={args.seed}  out={out_dir}")

    # Lazy-import the heavier modules so --help is fast.
    from scripts.ocft_lib import (
        load_action_direction,
        extract_hotpot_cache,
        extract_musique_p0,
        extract_pairs_p0,
        train_probe_on,
        decompose_and_save,
    )

    # ── Load A_L20 ──────────────────────────────────────────────────────────
    A, A_meta = load_action_direction(args.action_direction)
    print(f"  A_L20 shape={A.shape} RMS={A_meta['rms']:.4f}")

    # ── Cache / extract activations for each candidate ──────────────────────
    print("\n[1/3] Loading hotpot cache …")
    hp = extract_hotpot_cache(args.hotpot_cache, args.hotpot_labels, layer=LAYER)
    print(f"  hotpot: N={len(hp['y_label'])} (label0={int((hp['y_label']==0).sum())},"
          f" label1={int((hp['y_label']==1).sum())})")

    print("\n[2/3] Extracting MuSiQue p0 hidden states (D1) …")
    mq = extract_musique_p0(args.musique_source, args.musique_n, args.model,
                            layer=LAYER, build_prompt=build_p0_prompt,
                            parse_train_user=parse_train_user)
    print(f"  musique: N={len(mq['X'])}")

    print("\n[3/3] Extracting extractability pairs p0 hidden states (D3,D4) …")
    pr = extract_pairs_p0(args.pairs_source, args.pairs_n, args.model,
                          layer=LAYER, build_prompt=build_p0_prompt)
    print(f"  pairs: N={len(pr['X'])}")

    # ── Train probes + decompose ────────────────────────────────────────────
    summary = {"layer": LAYER, "seed": args.seed,
               "model": args.model, "candidates": {}}

    candidates = [
        ("D1_source",      hp["X"], mq["X"], None,    "binary",
         "hotpot vs musique",
         lambda: (np.concatenate([hp["X"], mq["X"]], 0),
                  np.concatenate([np.zeros(len(hp["X"]), int),
                                  np.ones(len(mq["X"]),  int)]))),
        ("D2_action_prior", hp["X"], None,   None,    "binary",
         "margin_before > 0  on hotpot p0",
         lambda: (hp["X"], (hp["margin_before"] > 0).astype(int))),
        ("D3_candidate_present", pr["X"], None, None, "binary",
         "T0/T1 (cand_present) vs N0/S0 (cand_absent)",
         lambda: (pr["X"], pr["candidate_present"].astype(int))),
        ("D4_obs_length",   pr["X"], None,   None,    "binary",
         "token_len > median",
         lambda: (pr["X"],
                  (pr["token_len"] > np.median(pr["token_len"])).astype(int))),
    ]

    for name, *_, contrast_desc, build_xy in candidates:
        X, y = build_xy()
        print(f"\n=== {name} :: {contrast_desc} ===")
        print(f"  X={X.shape}  y0={int((y==0).sum())} y1={int((y==1).sum())}")
        probe = train_probe_on(X, y, seed=args.seed)
        print(f"  AUROC={probe['auroc']:.3f}  BalAcc={probe['balanced_accuracy']:.3f}")
        save_paths = decompose_and_save(
            A, probe["direction"], name, args.steering_dir, layer=LAYER)
        summary["candidates"][name] = {
            "contrast": contrast_desc,
            "n_total": int(len(y)), "n_label0": int((y==0).sum()),
            "n_label1": int((y==1).sum()),
            "auroc": probe["auroc"],
            "balanced_accuracy": probe["balanced_accuracy"],
            "cos_with_action": probe["cos_with_action"]
                if "cos_with_action" in probe
                else float(np.dot(probe["direction"]/np.linalg.norm(probe["direction"]),
                                   A/np.linalg.norm(A))),
            "files": save_paths,
        }
        # Persist per-candidate artefacts
        cd = out_dir / "per_candidate" / name
        cd.mkdir(parents=True, exist_ok=True)
        np.savez(cd / "X.npz", X=X.astype(np.float32), y=y.astype(np.int32))
        with open(cd / "probe.json", "w") as f:
            json.dump({k: v for k, v in probe.items() if k != "direction"},
                      f, indent=2)
        np.save(cd / "direction.npy", probe["direction"].astype(np.float32))

    with open(out_dir / "probes_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] probes_summary → {out_dir/'probes_summary.json'}")


if __name__ == "__main__":
    main()
