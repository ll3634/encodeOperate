#!/usr/bin/env python3
"""
Evidence Signal Erosion During Thought Generation (Phase 1 Extension)

Hypothesis: At the step-1 decision point, L20 encodes "evidence is insufficient."
As the model generates its Thought (reasoning), this evidence signal ERODES and is
replaced by a "stop/commit" signal — causing the model to behaviorally commit despite
its initial internal state.

Method:
  For each of 486 baseline samples, generate step-1 output using REACT_THOUGHT_SYSTEM_PROMPT.
  Extract L20 residual stream at 5 normalized positions:
    Position 0: input boundary (last token before generation, matches Phase 1 data)
    Position 1: 25% through generated Thought
    Position 2: 50% through generated Thought
    Position 3: 75% through generated Thought
    Position 4: 100% (last thought token, just before Action: / Final Answer:)

  At each position, train a logistic regression probe on evidence sufficiency labels.
  Report AUROC and balanced accuracy at each position.

  Also compute: projection of L20 activations onto the EXISTING Phase-1 probe direction.
  This shows if the Phase-1 evidence signal (trained at the input boundary, DEFAULT_SYSTEM_PROMPT)
  persists or decays during thought generation.

Key outputs:
  1. AUROC vs position curve (all samples, label=0 subset, A3-rescued subset)
  2. Mean probe-direction projection vs position (for stop vs continue behavioral groups)
  3. Per-sample erosion scores

Usage:
    cd tmc/scripts/e2e_agent
    python scripts/thought_erosion_probe.py \\
        --labels results/phase1_probe/labels.jsonl \\
        --baseline results/l20_rho020_n500/baseline_results.jsonl \\
        --probe-direction results/phase1_probe/probe_direction_l20.npz \\
        --output-dir results/thought_erosion \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        [--dry-run]
"""

import os, sys, json, argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from tqdm import tqdm as _tqdm
    def tqdm(iterable, **kwargs):
        return _tqdm(iterable, **kwargs, dynamic_ncols=True)
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, REACT_THOUGHT_SYSTEM_PROMPT, TOOL_DESCRIPTIONS
from steering.hook_utils import get_model_layers

# ── Constants ──────────────────────────────────────────────────────────────────

# 19 rescued_via_search sample IDs from A3
A3_RESCUED_VIA_SEARCH = {
    "5abaee845542994c784ddb49",
    "5abbcfaf5542993f40c73ba9",
    "5ae2eda355429928c4239570",
    "5a8782f25542996e4f308818",
    "5a8f51185542992414482a3d",
    "5a85b2895542994c784ddb49",
    "5ae256435542992decbdccc3",
    "5ab29956554299194fa9342d",
    "5ae55d1e55429960a22e02cb",
    "5ab9cfe655429970cfb8ebaf",
    "5a821c95554299676cceb219",
    "5abdba405542993f32c2a023",
    "5abf92c45542993fe9a41e07",
    "5ac2a35055429967731025ce",
    "5ae7535c5542997b22f6a6d8",
    "5ae47cab5542996836b02cb9",
    "5a79311755429970f5fffe67",
    "5a7e02b75542997cc2c474f3",
    "5a83c2e25542996488c2e4bc",
}

POSITION_NAMES = ["p0_input", "p1_25pct", "p2_50pct", "p3_75pct", "p4_100pct"]
POSITION_FRACS = [None, 0.25, 0.50, 0.75, 1.00]   # None = input boundary

MIN_THOUGHT_TOKENS = 10   # samples with thought shorter than this are flagged

# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


# ── Activation extraction ──────────────────────────────────────────────────────

def extract_activations_at_positions(
    model, tokenizer, model_layers,
    input_ids: torch.Tensor,
    thought_ids: List[int],
    layer_idx: int = 20,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Given input_ids and thought_ids (already generated), run one teacher-forced
    forward pass on [input_ids | thought_ids] and extract L{layer_idx} activation
    at 5 normalized positions.

    Returns dict {position_name: activation_vector} or None on failure.
    """
    n_thought = len(thought_ids)
    if n_thought == 0:
        return None

    input_len = input_ids.shape[1]

    # Compute token indices for each position
    pos_idx = {}
    pos_idx["p0_input"] = input_len - 1                                    # last input token
    pos_idx["p1_25pct"] = input_len + max(0, int(round(0.25 * n_thought)) - 1)
    pos_idx["p2_50pct"] = input_len + max(0, int(round(0.50 * n_thought)) - 1)
    pos_idx["p3_75pct"] = input_len + max(0, int(round(0.75 * n_thought)) - 1)
    pos_idx["p4_100pct"] = input_len + n_thought - 1                       # last thought token

    # Build full token sequence
    thought_tensor = torch.tensor([thought_ids], dtype=torch.long, device=input_ids.device)
    full_ids = torch.cat([input_ids, thought_tensor], dim=1)

    # Capture L20 for all positions with a hook
    captured = {}

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        # Extract needed positions
        seq = h[0].detach().float().cpu()   # (seq_len, d_model)
        for name, idx in pos_idx.items():
            if idx < seq.shape[0]:
                captured[name] = seq[idx].numpy()

    handle = model_layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(full_ids)
    except Exception as e:
        handle.remove()
        return None
    handle.remove()

    if len(captured) != len(POSITION_NAMES):
        return None
    return captured


def generate_thought(
    model, tokenizer, input_ids: torch.Tensor, max_new_tokens: int = 120
) -> Tuple[List[int], str, int]:
    """
    Generate step-1 output using REACT_THOUGHT_SYSTEM_PROMPT context.
    Return (thought_token_ids, thought_text, boundary_type) where boundary_type:
      0 = no boundary found (used all generated tokens)
      1 = stopped at \\nAction
      2 = stopped at \\nFinal Answer

    IMPORTANT: We find the boundary in TOKEN space (not text space) to avoid
    re-tokenization mismatch.  Decoding → re-encoding can produce different
    token IDs due to BPE merge differences, whitespace normalisation, etc.
    """
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = output_ids[0][input_ids.shape[1]:].tolist()   # new tokens only

    # -- Find the boundary in TOKEN space --
    # We decode incrementally and look for "\nAction" or "\nFinal" in the
    # accumulated text.  Once found, the token index gives us the exact
    # split point in gen_ids without any re-encoding.
    boundary_tok_idx = len(gen_ids)          # default: use all tokens
    boundary_type = 0

    accumulated = ""
    for tok_idx, tid in enumerate(gen_ids):
        tok_text = tokenizer.decode([tid], skip_special_tokens=True)
        accumulated += tok_text

        # Check for boundaries after each token
        action_pos = accumulated.find("\nAction")
        final_pos  = accumulated.find("\nFinal")

        if action_pos >= 0 or final_pos >= 0:
            if action_pos >= 0 and final_pos >= 0:
                boundary_type = 1 if action_pos <= final_pos else 2
            elif action_pos >= 0:
                boundary_type = 1
            else:
                boundary_type = 2

            # The boundary starts at the newline token. Walk backwards from
            # tok_idx to find the token that introduced the newline.
            # A conservative choice: the thought tokens are everything
            # BEFORE the current token (the one that completed the boundary
            # string).  We include tok_idx itself only if the boundary
            # keyword started in a previous token (multi-token keyword).
            # Simplest correct approach: cut right before the newline.
            # Find where in accumulated the boundary text starts.
            cut_char = action_pos if boundary_type == 1 else final_pos
            # Find which token index corresponds to cut_char
            prefix_len = 0
            for j, t in enumerate(gen_ids[:tok_idx + 1]):
                prev_len = prefix_len
                prefix_len += len(tokenizer.decode([t], skip_special_tokens=True))
                if prefix_len > cut_char:
                    boundary_tok_idx = j
                    break
            else:
                boundary_tok_idx = tok_idx
            break

    thought_ids = gen_ids[:boundary_tok_idx]
    thought_text = tokenizer.decode(thought_ids, skip_special_tokens=True).strip()

    return thought_ids, thought_text, boundary_type


# ── Probe training ─────────────────────────────────────────────────────────────

def train_probe_at_position(
    activations: np.ndarray,  # (N, d)
    labels: np.ndarray,       # (N,) binary
    seed: int = 42,
) -> dict:
    """Train logistic regression probe with 80/20 stratified split."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(activations)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(sss.split(X, labels))

    clf = LogisticRegression(
        class_weight="balanced", C=1.0, max_iter=2000,
        solver="lbfgs", random_state=seed
    )
    clf.fit(X[train_idx], labels[train_idx])

    y_pred = clf.predict(X[test_idx])
    y_prob = clf.predict_proba(X[test_idx])[:, 1]

    bal_acc = balanced_accuracy_score(labels[test_idx], y_pred)
    try:
        auroc = roc_auc_score(labels[test_idx], y_prob)
    except ValueError:
        auroc = float("nan")

    # Direction from all data
    clf_all = LogisticRegression(
        class_weight="balanced", C=1.0, max_iter=2000,
        solver="lbfgs", random_state=seed
    )
    clf_all.fit(X, labels)
    w = clf_all.coef_[0] / scaler.scale_
    direction = (w / np.linalg.norm(w)).astype(np.float32)

    return {
        "balanced_accuracy": float(bal_acc),
        "auroc": float(auroc),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_test_label0": int((labels[test_idx] == 0).sum()),
        "n_test_label1": int((labels[test_idx] == 1).sum()),
        "direction": direction,
    }


# ── Projection analysis ────────────────────────────────────────────────────────

def compute_projections(
    activations_per_pos: Dict[str, List[np.ndarray]],
    labels: np.ndarray,
    probe_direction: np.ndarray,
    behavioral_stop: np.ndarray,
    is_a3: np.ndarray,
) -> dict:
    """
    Project activations at each position onto the Phase-1 probe direction.
    Report mean projection for different groups, plus the label gap.
    """
    results = {}
    for pos_name in POSITION_NAMES:
        if pos_name not in activations_per_pos:
            continue
        acts = np.array(activations_per_pos[pos_name])  # (N, d)
        proj = acts @ probe_direction                     # (N,) dot product

        m_l0 = float(proj[labels == 0].mean()) if (labels == 0).any() else None
        m_l1 = float(proj[labels == 1].mean()) if (labels == 1).any() else None

        results[pos_name] = {
            "mean_all":      float(proj.mean()),
            "mean_label0":   m_l0,
            "mean_label1":   m_l1,
            "mean_stop":     float(proj[behavioral_stop].mean()) if behavioral_stop.any() else None,
            "mean_continue": float(proj[~behavioral_stop].mean()) if (~behavioral_stop).any() else None,
            "mean_a3":       float(proj[is_a3].mean()) if is_a3.any() else None,
            # Gap: how well does the Phase-1 direction separate the two labels at this position?
            # Monotonic decrease in this gap = evidence signal erosion.
            "gap_l1_minus_l0": float(m_l1 - m_l0) if (m_l0 is not None and m_l1 is not None) else None,
            # std
            "std_all":       float(proj.std()),
            "std_label0":    float(proj[labels == 0].std()) if (labels == 0).any() else None,
            "std_label1":    float(proj[labels == 1].std()) if (labels == 1).any() else None,
        }
    return results


def compute_fixed_direction_auroc(
    activations_per_pos: Dict[str, List[np.ndarray]],
    labels: np.ndarray,
    probe_direction: np.ndarray,
) -> Dict[str, float]:
    """
    Compute AUROC at each position using a FIXED direction (the Phase-1 probe
    direction), rather than re-training a probe at each position.

    This answers: "does the ORIGINAL evidence signal persist during thought
    generation?"  Unlike per-position re-trained AUROC, this cannot rebound
    if a new (different) discriminative direction emerges at a later position.
    """
    from sklearn.metrics import roc_auc_score

    results = {}
    for pos_name in POSITION_NAMES:
        if pos_name not in activations_per_pos:
            continue
        acts = np.array(activations_per_pos[pos_name], dtype=np.float32)
        proj = acts @ probe_direction  # (N,)
        if len(set(labels)) < 2:
            continue
        try:
            results[pos_name] = float(roc_auc_score(labels, proj))
        except Exception:
            results[pos_name] = float("nan")
    return results


# ── Main collection loop ───────────────────────────────────────────────────────

def collect_erosion_data(
    model, tokenizer, model_layers,
    samples: List[dict],
    output_dir: Path,
    dry_run: bool = False,
) -> Tuple[Dict[str, List[np.ndarray]], List[dict]]:
    """
    For each sample: generate thought, extract L20 at 5 positions.
    Returns (activations_per_pos, metadata_list).
    """
    cache_path = output_dir / "raw_erosion_data.npz"
    meta_path = output_dir / "raw_erosion_meta.jsonl"

    # Resume support
    if cache_path.exists() and meta_path.exists():
        print(f"[Resume] Loading cached activations from {cache_path}")
        d = np.load(cache_path)
        acts = {k: list(d[k]) for k in POSITION_NAMES if k in d}
        meta = [json.loads(l) for l in open(meta_path)]
        n_done = min(len(meta), len(next(iter(acts.values()), [])))
        print(f"[Resume] Found {n_done} completed samples")
        # Check if all samples done
        done_ids = {m["sample_id"] for m in meta[:n_done]}
        remaining = [s for s in samples if s["sample_id"] not in done_ids]
        if not remaining:
            print("[Resume] All samples complete.")
            return acts, meta[:n_done]
        print(f"[Resume] {len(remaining)} samples remaining")
    else:
        acts = {pos: [] for pos in POSITION_NAMES}
        meta = []
        remaining = samples
        n_done = 0

    pb = PromptBuilder(tools=["search"])
    system_prompt_thought = REACT_THOUGHT_SYSTEM_PROMPT.format(
        tool_descriptions=pb.get_tool_descriptions()
    )
    print(f"[NOTE] Using REACT_THOUGHT_SYSTEM_PROMPT (allows Thought:) — "
          f"differs from DEFAULT_SYSTEM_PROMPT used in Phase 1 baseline. "
          f"p0_input activations here may not exactly match Phase 1 data.")
    device = next(model.parameters()).device

    skipped = 0
    short_thought = 0
    checkpoint_every = 50

    pbar = tqdm(remaining, desc="Erosion", unit="sample",
                initial=n_done, total=n_done + len(remaining))

    for i, sample in enumerate(pbar):
        sid = sample["sample_id"]
        q = sample["question"]
        obs = sample["observation"]
        step0_action_input = sample.get("action_input", q[:50])

        # Build step-1 input with REACT_THOUGHT_SYSTEM_PROMPT
        history = [{"action": "search", "action_input": step0_action_input,
                    "observation": obs}]
        messages = [
            {"role": "system", "content": system_prompt_thought},
            {"role": "user", "content": q},
            {"role": "assistant", "content": pb.build_scratchpad(history)},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Step 1: generate Thought
        try:
            thought_ids, thought_text, boundary_type = generate_thought(
                model, tokenizer, input_ids, max_new_tokens=120
            )
        except Exception as e:
            pbar.write(f"  GENERATION ERROR {sid[:20]}: {e}")
            skipped += 1
            continue

        n_thought = len(thought_ids)
        is_short = n_thought < MIN_THOUGHT_TOKENS

        if is_short:
            short_thought += 1
            # Still include but flag it

        # Step 2: extract L20 at 5 positions
        pos_acts = extract_activations_at_positions(
            model, tokenizer, model_layers, input_ids, thought_ids, layer_idx=20
        )
        if pos_acts is None:
            skipped += 1
            continue

        # Store
        for pos_name in POSITION_NAMES:
            acts[pos_name].append(pos_acts[pos_name])

        meta.append({
            "sample_id": sid,
            "question": q[:100],
            "evidence_label": sample["evidence_label"],
            "behavioral_stop": sample["behavioral_stop"],
            "is_a3_rescued": sid in A3_RESCUED_VIA_SEARCH,
            "n_thought_tokens": n_thought,
            "is_short_thought": is_short,
            "boundary_type": boundary_type,
            "thought_text_preview": thought_text[:200],
        })

        n_collected = len(meta)
        btype = "search" if boundary_type == 1 else "final" if boundary_type == 2 else "none"
        pbar.set_postfix(collected=n_collected, short=short_thought,
                         skip=skipped, tlen=n_thought, bnd=btype, refresh=False)

        # Checkpoint
        if n_collected % checkpoint_every == 0:
            _save_checkpoint(acts, meta, cache_path, meta_path)
            pbar.write(f"[Checkpoint] {n_collected} samples saved")

        if dry_run and n_collected >= 20:
            pbar.write(f"[Dry run] Stopping after {n_collected} samples")
            break

    pbar.close()

    # Final save
    _save_checkpoint(acts, meta, cache_path, meta_path)

    print(f"\nCollection complete: {len(meta)} samples, {skipped} skipped, {short_thought} short thoughts")
    return acts, meta


def _save_checkpoint(acts, meta, cache_path, meta_path):
    if acts and any(acts.values()):
        np.savez_compressed(
            cache_path,
            **{pos: np.array(acts[pos]) for pos in POSITION_NAMES if acts[pos]}
        )
    with open(meta_path, "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_erosion_plots(
    probe_results_per_pos: Dict[str, dict],
    projection_results: Dict[str, dict],
    fixed_auroc: Dict[str, float],
    labels: np.ndarray,
    behavioral_stop: np.ndarray,
    is_a3: np.ndarray,
    activations_per_pos: Dict[str, List[np.ndarray]],
    probe_direction: np.ndarray,
    output_dir: Path,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    x = list(range(len(POSITION_NAMES)))
    x_labels = ["Input\nboundary", "25%", "50%", "75%", "100%\n(thought end)"]

    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    def _plot(ax, vals, label, color, marker="o", ls="-", lw=2):
        valid = [(i, v) for i, v in enumerate(vals) if v is not None and not np.isnan(v)]
        if not valid:
            return
        xi, yi = zip(*valid)
        ax.plot(xi, yi, marker=marker, linestyle=ls, color=color, label=label, linewidth=lw)

    def _set_x(ax):
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.grid(True, alpha=0.3)

    # ── Panel 1: AUROC comparison (re-trained vs fixed-direction) ─────────
    ax1 = fig.add_subplot(gs[0, 0])
    aurocs_retrained = [probe_results_per_pos.get(p, {}).get("all", {}).get("auroc", None)
                        for p in POSITION_NAMES]
    aurocs_fixed = [fixed_auroc.get(p) for p in POSITION_NAMES]

    _plot(ax1, aurocs_retrained, "Re-trained probe", "#1f77b4", marker="o")
    _plot(ax1, aurocs_fixed, "Fixed Phase-1 direction", "#d62728", marker="s", ls="--")
    ax1.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance")
    _set_x(ax1)
    ax1.set_ylabel("AUROC")
    ax1.set_ylim(0.4, 1.0)
    ax1.set_title("Evidence Signal Discriminability\n(Re-trained vs Fixed-Direction AUROC)")
    ax1.legend(fontsize=8)

    # Annotate the key insight if rebound exists
    if (aurocs_retrained[3] is not None and aurocs_retrained[4] is not None
            and aurocs_fixed[3] is not None and aurocs_fixed[4] is not None):
        if aurocs_retrained[4] > aurocs_retrained[3] + 0.05:
            ax1.annotate("rebound\n(new direction,\nnot original signal)",
                         xy=(4, aurocs_retrained[4]), fontsize=7, color="#1f77b4",
                         ha="right", va="bottom",
                         xytext=(3.3, aurocs_retrained[4] + 0.04),
                         arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.8))

    # ── Panel 2: Projection gap (label1 − label0) ────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    gaps = [projection_results.get(p, {}).get("gap_l1_minus_l0", None) for p in POSITION_NAMES]
    _plot(ax2, gaps, "Gap (label1 − label0)", "#9467bd", marker="D")
    ax2.axhline(0.0, color="gray", linestyle=":", linewidth=1)
    _set_x(ax2)
    ax2.set_ylabel("Projection gap (Phase-1 direction)")
    ax2.set_title("Evidence Signal Separation\n(Monotonic Decrease = Signal Erosion)")
    ax2.legend(fontsize=8)
    # Annotate gap values
    for i, g in enumerate(gaps):
        if g is not None:
            ax2.annotate(f"{g:.2f}", (i, g), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8, color="#9467bd")

    # ── Panel 3: Projection trajectories by group ────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for group_name, label, color, marker in [
        ("mean_label0", "Label=0 (0-doc, N=97)", "#d62728", "s"),
        ("mean_label1", "Label=1 (1+-doc, N=389)", "#1f77b4", "o"),
        ("mean_a3", "A3 rescued (N=18)", "#2ca02c", "^"),
    ]:
        vals = [projection_results.get(p, {}).get(group_name, None) for p in POSITION_NAMES]
        _plot(ax3, vals, label, color, marker=marker)
    ax3.axhline(0.0, color="gray", linestyle=":", linewidth=1,
                label="Decision boundary")
    _set_x(ax3)
    ax3.set_ylabel("Mean projection on Phase-1 direction\n(+ = 'sufficient',  − = 'insufficient')")
    ax3.set_title("Evidence Signal Trajectory per Subgroup")
    ax3.legend(fontsize=7, loc="upper left")

    # Shade the "insufficient" region
    ax3.axhspan(ax3.get_ylim()[0], 0, alpha=0.05, color="#d62728")

    # ── Panel 4: Stop vs Continue projection ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    for group_name, label, color, marker in [
        ("mean_stop", f"Behavioral stop (N={int(behavioral_stop.sum())})", "#ff7f0e", "o"),
        ("mean_continue", f"Behavioral continue (N={int((~behavioral_stop).sum())})", "#2ca02c", "^"),
    ]:
        vals = [projection_results.get(p, {}).get(group_name, None) for p in POSITION_NAMES]
        _plot(ax4, vals, label, color, marker=marker)
    ax4.axhline(0.0, color="gray", linestyle=":", linewidth=1)
    _set_x(ax4)
    ax4.set_ylabel("Mean projection on Phase-1 direction")
    ax4.set_title("Stop vs Continue: Projection Trajectories")
    ax4.legend(fontsize=8)

    # ── Panel 5: Erosion score distribution (label=0 samples) ────────────
    ax5 = fig.add_subplot(gs[2, 0])
    label0_mask = labels == 0
    if label0_mask.any() and "p0_input" in activations_per_pos and "p4_100pct" in activations_per_pos:
        p0_acts = np.array(activations_per_pos["p0_input"])[label0_mask]
        p4_acts = np.array(activations_per_pos["p4_100pct"])[label0_mask]
        proj_p0 = p0_acts @ probe_direction
        proj_p4 = p4_acts @ probe_direction
        erosion = proj_p0 - proj_p4  # positive = evidence signal decayed

        ax5.hist(erosion, bins=25, color="#d62728", alpha=0.7,
                 edgecolor="white", linewidth=0.5,
                 label=f"Label=0 (n={label0_mask.sum()})")
        ax5.axvline(erosion.mean(), color="darkred", linestyle="--", linewidth=2,
                    label=f"Mean = {erosion.mean():+.2f}")
        pct_eroded = (erosion > 0).mean() * 100
        ax5.axvline(0, color="gray", linestyle=":", linewidth=1)
        ax5.set_xlabel("Erosion score (proj@p0 − proj@p4)\n"
                       "Positive = evidence signal decayed toward 'sufficient'")
        ax5.set_ylabel("Count")
        ax5.set_title(f"Per-Sample Erosion Distribution (Label=0)\n"
                      f"{pct_eroded:.0f}% of samples show erosion")
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                 transform=ax5.transAxes)

    # ── Panel 6: Balanced accuracy ───────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    baccs_all = [probe_results_per_pos.get(p, {}).get("all", {}).get("balanced_accuracy", None)
                 for p in POSITION_NAMES]
    _plot(ax6, baccs_all, "Balanced Accuracy (re-trained)", "#1f77b4")
    ax6.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance")
    _set_x(ax6)
    ax6.set_ylabel("Balanced Accuracy")
    ax6.set_ylim(0.4, 1.0)
    ax6.set_title("Re-trained Probe Balanced Accuracy")
    ax6.legend(fontsize=8)

    plt.suptitle("Evidence Signal Erosion During Thought Generation\n"
                 f"(L20, REACT_THOUGHT format, N={len(labels)}, "
                 f"label0={int((labels==0).sum())}, label1={int((labels==1).sum())})",
                 fontsize=13, fontweight="bold")

    out_path = output_dir / "erosion_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="results/phase1_probe/labels.jsonl")
    parser.add_argument("--baseline", default="results/l20_rho020_n500/baseline_results.jsonl")
    parser.add_argument("--probe-direction", default="results/phase1_probe/probe_direction_l20.npz")
    parser.add_argument("--output-dir", default="results/thought_erosion")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading labels...")
    labels_raw = [json.loads(l) for l in open(args.labels)]
    label_map = {r["sample_id"]: r for r in labels_raw}

    print("Loading baseline traces...")
    baseline_map = {r["sample_id"]: r
                    for r in (json.loads(l) for l in open(args.baseline))}

    print("Loading probe direction...")
    probe_npz = np.load(args.probe_direction)
    probe_direction = probe_npz["decision_direction"].astype(np.float32)
    print(f"  Probe direction: shape={probe_direction.shape}, "
          f"balanced_acc={float(probe_npz.get('balanced_accuracy', 0)):.3f}, "
          f"auroc={float(probe_npz.get('auroc', 0)):.3f}")

    # ── Build sample list ─────────────────────────────────────────────────────
    pb = PromptBuilder(tools=["search"])
    samples = []
    for lb in labels_raw:
        sid = lb["sample_id"]
        ep = baseline_map.get(sid, {})
        steps = ep.get("steps", [])
        if not steps or steps[0].get("action") != "search" or not steps[0].get("observation"):
            continue
        s0 = steps[0]
        samples.append({
            "sample_id": sid,
            "question": lb["question"],
            "observation": s0["observation"],
            "action_input": s0.get("action_input", ""),
            "evidence_label": lb["label"],
            "behavioral_stop": lb["behavioral_stop"],
        })

    print(f"Built sample list: {len(samples)} samples")
    if args.dry_run:
        samples = samples[:20]
        print(f"[Dry run] Using first {len(samples)} samples")

    # ── Prompt format note ────────────────────────────────────────────────────
    print("\n[NOTE] This experiment uses REACT_THOUGHT_SYSTEM_PROMPT (allows 'Thought: ...').")
    print("       Phase-1 probe direction was extracted with DEFAULT_SYSTEM_PROMPT (no Thought).")
    print("       p0_input activations come from a DIFFERENT prompt format than Phase-1.")
    print("       Within-experiment comparisons (p0→p4 erosion) are internally consistent.")
    print("       Cross-experiment comparison to Phase-1 AUROC requires caution.\n")

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer = load_model(args.model)
    model_layers = get_model_layers(model)

    # ── Collect activations ───────────────────────────────────────────────────
    activations_per_pos, metadata = collect_erosion_data(
        model, tokenizer, model_layers, samples, output_dir, dry_run=args.dry_run
    )

    if not metadata:
        print("No data collected. Exiting.")
        return

    # ── Align metadata with activations ──────────────────────────────────────
    N = len(metadata)
    labels = np.array([m["evidence_label"] for m in metadata], dtype=np.int32)
    behavioral_stop = np.array([m["behavioral_stop"] for m in metadata], dtype=bool)
    is_a3 = np.array([m["is_a3_rescued"] for m in metadata], dtype=bool)
    is_short = np.array([m["is_short_thought"] for m in metadata], dtype=bool)

    print(f"\nDataset: N={N}, label0={( labels==0).sum()}, label1={(labels==1).sum()}")
    print(f"Short thoughts (<{MIN_THOUGHT_TOKENS} tokens): {is_short.sum()}")
    print(f"Behavioral stop: {behavioral_stop.sum()}, continue: {(~behavioral_stop).sum()}")
    print(f"A3 rescued: {is_a3.sum()}")

    thought_lens = [m["n_thought_tokens"] for m in metadata]
    print(f"Thought length: mean={np.mean(thought_lens):.1f}, "
          f"median={np.median(thought_lens):.1f}, "
          f"min={np.min(thought_lens)}, max={np.max(thought_lens)}")

    # Trim activations to N
    for pos_name in POSITION_NAMES:
        if pos_name in activations_per_pos:
            activations_per_pos[pos_name] = activations_per_pos[pos_name][:N]

    # ── Train probes at each position ─────────────────────────────────────────
    print("\n=== Training probes at each position ===")
    probe_results_per_pos = {}

    def train_subset(acts_list, lbl, name, pos):
        if len(acts_list) == 0 or len(set(lbl)) < 2:
            return None
        X = np.array(acts_list, dtype=np.float32)
        try:
            return train_probe_at_position(X, lbl, seed=args.seed)
        except Exception as e:
            print(f"    [{pos}/{name}] probe failed: {e}")
            return None

    for pos_name in POSITION_NAMES:
        if pos_name not in activations_per_pos or not activations_per_pos[pos_name]:
            print(f"  {pos_name}: NO DATA")
            continue

        acts_all = activations_per_pos[pos_name]
        if len(acts_all) != N:
            print(f"  {pos_name}: length mismatch {len(acts_all)} vs {N}")
            continue

        probe_results_per_pos[pos_name] = {}

        # All samples — train probe on full data (label 0 vs 1+)
        res_all = train_subset(acts_all, labels, "all", pos_name)
        probe_results_per_pos[pos_name]["all"] = res_all

        # Label=0 / A3 subsets: DON'T train separate probes (conceptually
        # wrong — you can't compute AUROC on a single-class subset, and A3
        # has only ~19 samples).  Instead, use the "all" probe's predicted
        # probability to track how the evidence signal evolves for each
        # subgroup.  This is reported via mean_prob (from the all-data probe)
        # and mean_proj (from the Phase-1 direction).
        if res_all is not None and "direction" in res_all:
            all_arr = np.array(acts_all, dtype=np.float32)
            # Use the probe direction from the "all" model to compute
            # per-sample predictions (projected score) for subgroups.
            d = res_all["direction"]
            proj_scores = all_arr @ d

            l0_mask = labels == 0
            l0_proj = proj_scores[l0_mask] if l0_mask.any() else np.array([])
            probe_results_per_pos[pos_name]["label0"] = {
                "n": int(l0_mask.sum()),
                "mean_proj": float(l0_proj.mean()) if len(l0_proj) > 0 else None,
                "std_proj": float(l0_proj.std()) if len(l0_proj) > 0 else None,
                "note": "projection_from_all_probe (no separate AUROC)",
            }

            a3_proj = proj_scores[is_a3] if is_a3.any() else np.array([])
            probe_results_per_pos[pos_name]["a3"] = {
                "n": int(is_a3.sum()),
                "mean_proj": float(a3_proj.mean()) if len(a3_proj) > 0 else None,
                "std_proj": float(a3_proj.std()) if len(a3_proj) > 0 else None,
                "note": "projection_from_all_probe (no separate AUROC)",
            }
        else:
            probe_results_per_pos[pos_name]["label0"] = None
            probe_results_per_pos[pos_name]["a3"] = None

        # Report
        auroc_str = f"{res_all['auroc']:.4f}" if res_all else "N/A"
        bacc_str = f"{res_all['balanced_accuracy']:.4f}" if res_all else "N/A"
        print(f"  {pos_name}: AUROC={auroc_str}, BalAcc={bacc_str}")

    # ── Projection analysis ───────────────────────────────────────────────────
    print("\n=== Probe direction projection analysis ===")
    print("  NOTE: projection direction = Phase-1 probe (DEFAULT_SYSTEM_PROMPT, decision-point).")
    print("        p0_input activations use REACT_THOUGHT_SYSTEM_PROMPT — not directly comparable")
    print("        to Phase-1 values, but within-experiment position comparisons are valid.\n")
    projection_results = compute_projections(
        activations_per_pos, labels, probe_direction, behavioral_stop, is_a3
    )
    _f = lambda v: f"{v:+.4f}" if v is not None else "  N/A "
    for pos_name, pr in projection_results.items():
        print(f"  {pos_name}: "
              f"label0={_f(pr['mean_label0'])}  "
              f"label1={_f(pr['mean_label1'])}  "
              f"gap={_f(pr['gap_l1_minus_l0'])}  "
              f"stop={_f(pr['mean_stop'])}  "
              f"cont={_f(pr['mean_continue'])}")

    # ── Fixed-direction AUROC ─────────────────────────────────────────────────
    print("\n=== Fixed-direction AUROC (Phase-1 probe direction at all positions) ===")
    print("  This uses the SAME direction everywhere — no re-training.")
    print("  Measures whether the ORIGINAL evidence signal persists.\n")
    fixed_auroc = compute_fixed_direction_auroc(
        activations_per_pos, labels, probe_direction
    )
    for pos_name in POSITION_NAMES:
        fa = fixed_auroc.get(pos_name)
        fa_str = f"{fa:.4f}" if fa is not None else "N/A"
        # Also show re-trained for comparison
        rt = probe_results_per_pos.get(pos_name, {}).get("all", {})
        rt_str = f"{rt['auroc']:.4f}" if rt else "N/A"
        print(f"  {pos_name}:  fixed={fa_str}  retrained={rt_str}")

    # ── Erosion summary ───────────────────────────────────────────────────────
    print("\n=== Erosion Summary ===")
    p0_res = probe_results_per_pos.get("p0_input", {}).get("all", {})
    p4_res = probe_results_per_pos.get("p4_100pct", {}).get("all", {})

    # 1) Re-trained AUROC drop
    if p0_res and p4_res:
        auroc_drop = p0_res["auroc"] - p4_res["auroc"]
        print(f"  Re-trained AUROC drop (p0→p4): {auroc_drop:+.4f}")

    # 2) Fixed-direction AUROC drop (the methodologically correct measure)
    fa_p0 = fixed_auroc.get("p0_input")
    fa_p4 = fixed_auroc.get("p4_100pct")
    if fa_p0 is not None and fa_p4 is not None:
        fa_drop = fa_p0 - fa_p4
        print(f"  Fixed-dir AUROC drop (p0→p4): {fa_drop:+.4f}")
        if fa_drop > 0.15:
            print(f"  ✓ EVIDENCE SIGNAL EROSION DETECTED (fixed-dir, >{0.15:.2f})")
        elif fa_drop > 0.05:
            print(f"  ~ Moderate erosion (fixed-dir, {fa_drop:.4f})")
        else:
            print(f"  ✗ No significant erosion (fixed-dir, {fa_drop:.4f})")

    # 3) Projection gap erosion (most robust: monotonic if signal erodes)
    pr_p0 = projection_results.get("p0_input", {})
    pr_p4 = projection_results.get("p4_100pct", {})
    gap_p0 = pr_p0.get("gap_l1_minus_l0") if pr_p0 else None
    gap_p4 = pr_p4.get("gap_l1_minus_l0") if pr_p4 else None
    if gap_p0 is not None and gap_p4 is not None:
        gap_drop = gap_p0 - gap_p4
        print(f"  Projection gap drop (p0→p4): {gap_drop:+.4f}  ({gap_p0:.3f} → {gap_p4:.3f})")

    # 4) Label=0 projection drift (signal inversion)
    ml0_p0 = pr_p0.get("mean_label0") if pr_p0 else None
    ml0_p4 = pr_p4.get("mean_label0") if pr_p4 else None
    if ml0_p0 is not None and ml0_p4 is not None:
        drift = ml0_p4 - ml0_p0
        print(f"  Label=0 projection drift (p0→p4): {drift:+.4f}  ({ml0_p0:+.3f} → {ml0_p4:+.3f})")
        if ml0_p0 < 0 and ml0_p4 > 0:
            print(f"  ✓ SIGNAL INVERSION: label=0 crosses from 'insufficient' to 'sufficient' side")

    # ── Serialize results ────────────────────────────────────────────────────
    # Remove non-serializable direction vectors for JSON output
    def _clean(d):
        if isinstance(d, dict):
            return {k: _clean(v) for k, v in d.items() if k != "direction"}
        if isinstance(d, np.ndarray):
            return d.tolist()
        if isinstance(d, (np.int32, np.int64)):
            return int(d)
        if isinstance(d, (np.float32, np.float64)):
            return float(d)
        return d

    results_json = {
        "model": args.model,
        "layer": args.layer,
        "n_samples": N,
        "n_label0": int((labels == 0).sum()),
        "n_label1": int((labels == 1).sum()),
        "n_short_thought": int(is_short.sum()),
        "thought_length_stats": {
            "mean": float(np.mean(thought_lens)),
            "median": float(np.median(thought_lens)),
            "min": int(np.min(thought_lens)),
            "max": int(np.max(thought_lens)),
            "p10": float(np.percentile(thought_lens, 10)),
            "p90": float(np.percentile(thought_lens, 90)),
        },
        "probe_per_position": _clean(probe_results_per_pos),
        "projection_per_position": _clean(projection_results),
        "fixed_direction_auroc": _clean(fixed_auroc),
        "per_sample": metadata,
    }

    out_json = output_dir / "erosion_results.json"
    with open(out_json, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved results to {out_json}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    make_erosion_plots(
        probe_results_per_pos, projection_results, fixed_auroc,
        labels, behavioral_stop, is_a3,
        activations_per_pos, probe_direction,
        output_dir,
    )

    # ── Print final table ─────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("EROSION CURVE — Evidence Signal During Thought Generation")
    print("=" * 85)
    print(f"{'Position':<14} {'Retrained':>10} {'Fixed-Dir':>10} "
          f"{'Proj_L0':>9} {'Proj_L1':>9} {'Gap':>8} {'BalAcc':>8}")
    print(f"{'':14} {'AUROC':>10} {'AUROC':>10} "
          f"{'(mean)':>9} {'(mean)':>9} {'L1-L0':>8} {'':>8}")
    print("-" * 85)
    for pos_name in POSITION_NAMES:
        pr = probe_results_per_pos.get(pos_name, {}).get("all", None)
        pj = projection_results.get(pos_name, {})
        fa = fixed_auroc.get(pos_name)
        rt_auroc = f"{pr['auroc']:.4f}" if pr else "  N/A "
        fd_auroc = f"{fa:.4f}" if fa is not None else "  N/A "
        bacc = f"{pr['balanced_accuracy']:.4f}" if pr else "  N/A "
        _pj = lambda k: f"{pj[k]:+.4f}" if pj and pj.get(k) is not None else "   N/A"
        proj_l0 = _pj("mean_label0")
        proj_l1 = _pj("mean_label1")
        gap = _pj("gap_l1_minus_l0")
        print(f"{pos_name:<14} {rt_auroc:>10} {fd_auroc:>10} "
              f"{proj_l0:>9} {proj_l1:>9} {gap:>8} {bacc:>8}")

    print(f"\nDone. All outputs in: {output_dir}/")


if __name__ == "__main__":
    main()
