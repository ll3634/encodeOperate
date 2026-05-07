#!/usr/bin/env python3
"""
Search-domain Post-tool Direction Extraction (Paired Observation Method)

Best practice: Extract a direction that captures "the tool gave me useful information"
by comparing hidden states when the model sees:
  - Condition A (Relevant): Question + its own search results  → model tends toward "Final Answer"
  - Condition B (Irrelevant): Question + search results from a DIFFERENT question → model tends toward "Action: search"

Same question in both conditions; only the observation content differs.
This eliminates all confounds except observation relevance — which IS the signal we want.

Unlike V12's force-decode approach (which collapses to noise at position=-1 due to causal masking),
this method produces genuinely different hidden states because the prompts themselves differ.

Output:
  - direction_search_post.npz: Post-tool trust direction for search domain
    key="decision_direction" (toward NON-ADOPT, for compatibility with existing pipeline)
"""

import argparse
import json
import os
import sys
import random as pyrandom
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Add parent dir to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.prompts import PromptBuilder
from datasets.hotpotqa import HotpotQADataset
from tools.search_tool import SearchTool


def load_model(model_id: str, use_4bit: bool = True):
    """Load model with optional 4-bit quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_id}")
    kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            print("  Using 4-bit quantization")
        except ImportError:
            print("  bitsandbytes not available, using bf16")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, **kwargs)
    model.eval()
    return model, tokenizer


def get_model_layers(model):
    """Get transformer layers."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    raise ValueError("Cannot find model layers")


def build_step1_prompt(tokenizer, question: str, search_query: str, observation: str,
                       max_obs_chars: int = 1500) -> str:
    """
    Build the exact Step 1 prompt the agent would see:
      System: [tools prompt]
      User: question
      Assistant: Action: search\nAction Input: ...\nObservation: ...
      <generation prompt>
    
    The model must decide: "Final Answer: ..." or "Action: search\n..."
    """
    pb = PromptBuilder(tools=["search", "calculator"])
    obs_truncated = observation[:max_obs_chars]
    steps = [{
        "action": "search",
        "action_input": search_query,
        "observation": obs_truncated,
    }]
    messages = pb.build_full_prompt(question, steps)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_hidden_at_last_prompt_token(model, tokenizer, prompt: str,
                                         device, layer_idx: int) -> np.ndarray:
    """Extract hidden state at last token of prompt from given layer.

    Uses a forward hook on `model.model.layers[layer_idx]` so that the
    captured hidden state matches the runtime ``SteeringHook`` which also
    registers on ``layers[layer_idx]``.

    **Historical bug (fixed 2026-03-11)**: the previous implementation used
    ``outputs.hidden_states[layer_idx]``, which returns the output of layer
    ``layer_idx - 1`` (because ``hidden_states[0]`` is the embedding output).
    This caused the search direction to be extracted from one layer *before*
    the layer where the steering hook is applied, creating a silent layer
    mismatch that degraded E2E performance.
    """
    from steering.hook_utils import get_model_layers

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    layers = get_model_layers(model)
    num_layers = len(layers)
    actual_layer = layer_idx if layer_idx >= 0 else num_layers + layer_idx
    if actual_layer < 0 or actual_layer >= num_layers:
        raise ValueError(f"Layer {layer_idx} out of range [0, {num_layers})")

    captured = {}

    def capture_hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden

    handle = layers[actual_layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        handle.remove()

    if "hidden" not in captured:
        raise RuntimeError("Failed to capture hidden state via forward hook")

    h = captured["hidden"][0, -1, :].float().cpu().numpy()
    return h


def compute_margin(model, tokenizer, prompt: str, device) -> float:
    """Compute margin = logP('Action') - logP('Final') at the first generated token."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.log_softmax(logits, dim=-1)

    action_ids = tokenizer.encode("Action", add_special_tokens=False)
    final_ids = tokenizer.encode("Final", add_special_tokens=False)

    action_lp = log_probs[action_ids[0]].item() if action_ids else -100.0
    final_lp = log_probs[final_ids[0]].item() if final_ids else -100.0
    return action_lp - final_lp


def compute_rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr ** 2)))


def _extract_record_id(record) -> Optional[str]:
    if isinstance(record, str):
        record = record.strip()
        return record or None
    if isinstance(record, dict):
        for key in ("id", "question_id", "sample_id"):
            value = record.get(key)
            if value:
                return str(value)
    return None


def load_excluded_ids(path: str) -> Set[str]:
    """Load question IDs from plain-text, JSON, or JSONL files."""
    exclude_path = Path(path)
    text = exclude_path.read_text(encoding="utf-8").strip()
    if not text:
        return set()

    ids: Set[str] = set()
    if exclude_path.suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            for item in payload:
                record_id = _extract_record_id(item)
                if record_id:
                    ids.add(record_id)
        elif isinstance(payload, dict):
            for key in ("ids", "question_ids", "sample_ids"):
                values = payload.get(key, [])
                if isinstance(values, list):
                    for item in values:
                        record_id = _extract_record_id(item)
                        if record_id:
                            ids.add(record_id)
        else:
            raise ValueError(f"Unsupported exclusion JSON payload in {path}")
        return ids

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if exclude_path.suffix == ".jsonl":
            record_id = _extract_record_id(json.loads(line))
        else:
            record_id = line
        if record_id:
            ids.add(record_id)
    return ids


def extract_step0_search_query(result: Dict) -> Optional[str]:
    """Extract the runtime step-0 search query from a saved episode result."""
    for step in result.get("steps", []):
        if step.get("step_idx") != 0:
            continue
        if step.get("action") != "search":
            continue
        query = step.get("action_input")
        if query:
            return str(query)
    return None


def load_runtime_queries(path: str) -> Dict[str, str]:
    """Load sample_id -> runtime step-0 search query from JSON/JSONL traces."""
    trace_path = Path(path)
    if not trace_path.exists():
        raise FileNotFoundError(f"Runtime trace path not found: {path}")

    if trace_path.suffix == ".json":
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a list payload in runtime trace JSON: {path}")
        records = payload
    else:
        records = []
        with trace_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    queries: Dict[str, str] = {}
    for record in records:
        sample_id = record.get("sample_id") or record.get("question_id") or record.get("id")
        if not sample_id:
            continue
        query = extract_step0_search_query(record)
        if not query:
            continue
        sample_id = str(sample_id)
        existing = queries.get(sample_id)
        if existing and existing != query:
            raise ValueError(
                f"Conflicting runtime queries for sample_id={sample_id}: {existing!r} vs {query!r}"
            )
        queries[sample_id] = query

    if not queries:
        raise ValueError(f"No runtime step-0 search queries found in trace file: {path}")
    return queries


def select_samples_from_pool(
    samples: Sequence,
    n: int,
    seed: int,
    type_filter: Optional[str] = None,
    excluded_ids: Optional[Iterable[str]] = None,
    allowed_ids: Optional[Iterable[str]] = None,
) -> Tuple[List, int]:
    excluded = {str(x) for x in (excluded_ids or set())}
    allowed = {str(x) for x in allowed_ids} if allowed_ids is not None else None
    pool = list(samples)
    if type_filter:
        pool = [s for s in pool if getattr(s, "type", None) == type_filter]
    if allowed is not None:
        pool = [s for s in pool if str(getattr(s, "id", "")) in allowed]
    if excluded:
        pool = [s for s in pool if str(getattr(s, "id", "")) not in excluded]
    eligible_pool_size = len(pool)
    if n is None or n < 0 or n >= eligible_pool_size:
        return list(pool), eligible_pool_size
    rng = pyrandom.Random(seed)
    return rng.sample(pool, n), eligible_pool_size


def get_subset_ids_from_hotpot(
    data_path: str,
    n: int,
    seed: int,
    type_filter: Optional[str] = None,
) -> Set[str]:
    dataset = HotpotQADataset(data_path)
    subset = dataset.get_subset(n, seed=seed, type_filter=type_filter)
    return {str(sample.id) for sample in subset}


def build_irrelevant_pair_indices(n: int, rng: pyrandom.Random) -> List[int]:
    if n < 2:
        raise ValueError("Need at least 2 samples to build non-self irrelevant pairs")
    indices = list(range(n))
    rng.shuffle(indices)
    for i in range(n):
        if indices[i] == i:
            j = (i + 1) % n
            indices[i], indices[j] = indices[j], indices[i]
    if any(indices[i] == i for i in range(n)):
        raise RuntimeError("Failed to build non-self irrelevant pairing")
    return indices


def main():
    parser = argparse.ArgumentParser(
        description="Extract search-domain post-tool direction via paired observations")
    parser.add_argument("--data-path", required=True,
                        help="Path to hotpot_dev_distractor_v1.json")
    parser.add_argument("--corpus-path", required=True,
                        help="Path to BM25 corpus JSONL")
    parser.add_argument("--output", required=True, help="Output NPZ path")
    parser.add_argument("--model", default=None, help="Model ID")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--n-pairs", type=int, default=300,
                        help="Number of question pairs to extract")
    parser.add_argument("--seed", type=int, default=123,
                        help="Seed for extraction sample selection (!=42 to avoid eval overlap)")
    parser.add_argument("--type-filter", default="bridge",
                        help="HotpotQA question type filter")
    parser.add_argument("--max-obs-chars", type=int, default=1500,
                        help="Max observation chars (matches agent truncation)")
    parser.add_argument("--exclude-ids-path", default=None,
                        help="Optional path to newline/JSON/JSONL file of question IDs to exclude")
    parser.add_argument("--exclude-eval-n", type=int, default=None,
                        help="Optional eval subset size to exclude from extraction")
    parser.add_argument("--exclude-eval-seed", type=int, default=42,
                        help="Seed for eval subset exclusion")
    parser.add_argument("--exclude-eval-data-path", default=None,
                        help="Optional data path for eval subset exclusion (default: --data-path)")
    parser.add_argument("--exclude-eval-type-filter", default=None,
                        help="Optional type filter for eval subset exclusion (default: --type-filter)")
    parser.add_argument("--runtime-trace-path", default=None,
                        help="Optional JSON/JSONL runtime trace (e.g. baseline_results.jsonl). "
                             "When provided, uses runtime step-0 search queries instead of raw question text.")
    args = parser.parse_args()

    model_id = args.model or os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

    # --- 1. Load data & search tool ---
    print("[1/5] Loading dataset and search tool...")
    dataset = HotpotQADataset(args.data_path)
    excluded_ids: Set[str] = set()
    if args.exclude_ids_path:
        excluded_ids.update(load_excluded_ids(args.exclude_ids_path))
    excluded_eval_ids: Set[str] = set()
    if args.exclude_eval_n is not None and args.exclude_eval_n >= 0:
        excluded_eval_ids = get_subset_ids_from_hotpot(
            data_path=args.exclude_eval_data_path or args.data_path,
            n=args.exclude_eval_n,
            seed=args.exclude_eval_seed,
            type_filter=args.exclude_eval_type_filter or args.type_filter,
        )
        excluded_ids.update(excluded_eval_ids)

    runtime_queries: Optional[Dict[str, str]] = None
    allowed_ids: Optional[Set[str]] = None
    query_alignment_mode = "question"
    if args.runtime_trace_path:
        runtime_queries = load_runtime_queries(args.runtime_trace_path)
        allowed_ids = set(runtime_queries.keys())
        query_alignment_mode = "runtime_trace"
        print(
            f"  Loaded {len(runtime_queries)} runtime queries from {args.runtime_trace_path} "
            f"for trace-aligned extraction"
        )

    samples, eligible_pool_size = select_samples_from_pool(
        dataset.samples,
        n=args.n_pairs,
        seed=args.seed,
        type_filter=args.type_filter,
        excluded_ids=excluded_ids,
        allowed_ids=allowed_ids,
    )
    if len(samples) < 2:
        raise ValueError(
            f"Need at least 2 eligible samples after exclusion, got {len(samples)} "
            f"(eligible_pool_size={eligible_pool_size})"
        )
    print(
        f"  Selected {len(samples)} samples (seed={args.seed}, type={args.type_filter}, "
        f"eligible_pool={eligible_pool_size}, excluded={len(excluded_ids)})"
    )

    search_queries: List[str] = []
    runtime_query_diff_from_question_count = 0
    if runtime_queries is None:
        search_queries = [s.question for s in samples]
    else:
        missing_runtime_ids = []
        for s in samples:
            sid = str(s.id)
            query = runtime_queries.get(sid)
            if not query:
                missing_runtime_ids.append(sid)
                continue
            search_queries.append(query)
            if query != s.question:
                runtime_query_diff_from_question_count += 1
        if missing_runtime_ids:
            preview = ", ".join(missing_runtime_ids[:5])
            raise ValueError(
                f"Missing runtime queries for {len(missing_runtime_ids)} selected samples "
                f"(first few: {preview})"
            )
        print(
            f"  Runtime-trace alignment active: {runtime_query_diff_from_question_count}/{len(samples)} "
            f"selected queries differ from raw question text"
        )

    search_tool = SearchTool(args.corpus_path, top_k=5, max_chars=500)

    # --- 2. Run search for each question & collect observations ---
    print("[2/5] Running searches to collect observations...")
    observations = []
    for s, search_query in tqdm(list(zip(samples, search_queries)), desc="Searching"):
        obs = search_tool(search_query)
        observations.append(obs)

    # --- 3. Build paired prompts ---
    # For each question i:
    #   Relevant prompt:   question_i + observation_i  (its own search result)
    #   Irrelevant prompt: question_i + observation_j  (observation from a DIFFERENT question)
    print("[3/5] Building paired prompts...")
    rng = pyrandom.Random(args.seed)
    n = len(samples)
    irrelevant_indices = build_irrelevant_pair_indices(n, rng)

    # --- 4. Load model & extract hidden states ---
    print("[4/5] Loading model...")
    model, tokenizer = load_model(model_id, use_4bit=not args.no_4bit)
    device = next(model.parameters()).device

    relevant_hiddens = []
    irrelevant_hiddens = []
    pair_info = []

    print(f"[5/5] Extracting hidden states from layer {args.layer}...")
    for i in tqdm(range(n), desc="Extracting pairs"):
        s = samples[i]
        search_query = search_queries[i]
        obs_relevant = observations[i]
        obs_irrelevant = observations[irrelevant_indices[i]]

        # Build Step 1 prompts
        prompt_rel = build_step1_prompt(
            tokenizer, s.question, search_query, obs_relevant,
            max_obs_chars=args.max_obs_chars)
        prompt_irr = build_step1_prompt(
            tokenizer, s.question, search_query, obs_irrelevant,
            max_obs_chars=args.max_obs_chars)

        # Extract hidden states
        h_rel = extract_hidden_at_last_prompt_token(
            model, tokenizer, prompt_rel, device, args.layer)
        h_irr = extract_hidden_at_last_prompt_token(
            model, tokenizer, prompt_irr, device, args.layer)

        relevant_hiddens.append(h_rel)
        irrelevant_hiddens.append(h_irr)

        # Compute margins for diagnostics
        m_rel = compute_margin(model, tokenizer, prompt_rel, device)
        m_irr = compute_margin(model, tokenizer, prompt_irr, device)

        pair_info.append({
            "question_id": s.id,
            "question": s.question,
            "search_query": search_query,
            "query_alignment_mode": query_alignment_mode,
            "search_query_differs_from_question": search_query != s.question,
            "answer": s.answer,
            "irrelevant_from": samples[irrelevant_indices[i]].id,
            "margin_relevant": round(m_rel, 4),
            "margin_irrelevant": round(m_irr, 4),
            "margin_diff": round(m_irr - m_rel, 4),
        })

    # === Compute direction ===
    rel_mean = np.mean(relevant_hiddens, axis=0)
    irr_mean = np.mean(irrelevant_hiddens, axis=0)

    # adopt_direction: points TOWARD trusting tool output (relevant > irrelevant)
    adopt_direction = rel_mean - irr_mean

    # decision_direction: points toward NON-ADOPT (for compatibility with existing pipeline
    # where positive rho pushes model toward "Action: search" = reject current result)
    decision_direction = irr_mean - rel_mean

    # Stats
    direction_norm = float(np.linalg.norm(adopt_direction))
    direction_rms = compute_rms(adopt_direction)

    # Margin diagnostics
    margins_rel = [p["margin_relevant"] for p in pair_info]
    margins_irr = [p["margin_irrelevant"] for p in pair_info]
    margin_diffs = [p["margin_diff"] for p in pair_info]
    n_correct_sign = sum(1 for d in margin_diffs if d > 0)  # irrelevant should have higher margin (more "Action")

    print(f"\n{'='*60}")
    print(f"=== Search Post-tool Direction Extracted ===")
    print(f"{'='*60}")
    print(f"N pairs:                {n}")
    print(f"Layer:                  {args.layer}")
    print(f"Direction norm:         {direction_norm:.4f}")
    print(f"Direction RMS:          {direction_rms:.6f}")
    print(f"")
    print(f"Margin diagnostics:")
    print(f"  Relevant mean:        {np.mean(margins_rel):.4f}")
    print(f"  Irrelevant mean:      {np.mean(margins_irr):.4f}")
    print(f"  Diff (irr-rel) mean:  {np.mean(margin_diffs):.4f}")
    print(f"  Correct sign:         {n_correct_sign}/{n} ({100*n_correct_sign/n:.1f}%)")

    # Compare with existing directions if available
    for ref_name, ref_path in [
        ("search_v3", "steering/directions/direction_search_v3.npz"),
        ("v12_post", "steering/directions/direction_v12_post_scaled.npz"),
    ]:
        ref_full = Path(__file__).resolve().parent.parent / ref_path
        if ref_full.exists():
            ref_data = np.load(ref_full)
            ref_d = ref_data["decision_direction"].astype(np.float64)
            cos = float(np.dot(decision_direction.flatten(), ref_d.flatten()) /
                        (np.linalg.norm(decision_direction) * np.linalg.norm(ref_d) + 1e-10))
            print(f"  cosine(this, {ref_name}): {cos:.6f}")

    # Generate random control direction
    dim = decision_direction.shape[0]
    np.random.seed(42)
    random_d = np.random.randn(dim).astype(np.float32)
    random_d = random_d / np.linalg.norm(random_d) * direction_norm

    # === Save ===
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        decision_direction=decision_direction.astype(np.float32),
        adopt_direction=adopt_direction.astype(np.float32),
        random_direction=random_d,
        layer=args.layer,
        n_pairs=n,
        seed=args.seed,
        method="paired_observation",
        domain="search",
        context="post_tool",
        query_alignment_mode=query_alignment_mode,
        runtime_trace_path=args.runtime_trace_path or "",
    )
    print(f"\nDirection saved to: {out_path}")

    # Save pair info
    info_path = str(out_path).replace(".npz", "_pair_info.jsonl")
    with open(info_path, "w") as f:
        for info in pair_info:
            f.write(json.dumps(info) + "\n")
    print(f"Pair info saved to: {info_path}")

    # Save summary
    summary = {
        "n_pairs": n,
        "layer": args.layer,
        "seed": args.seed,
        "type_filter": args.type_filter,
        "eligible_pool_size": eligible_pool_size,
        "excluded_ids_count": len(excluded_ids),
        "excluded_eval_ids_count": len(excluded_eval_ids),
        "exclude_ids_path": args.exclude_ids_path,
        "exclude_eval_n": args.exclude_eval_n,
        "exclude_eval_seed": args.exclude_eval_seed if args.exclude_eval_n is not None else None,
        "exclude_eval_data_path": args.exclude_eval_data_path or args.data_path if args.exclude_eval_n is not None else None,
        "exclude_eval_type_filter": (args.exclude_eval_type_filter or args.type_filter) if args.exclude_eval_n is not None else None,
        "direction_norm": direction_norm,
        "direction_rms": direction_rms,
        "method": "paired_observation",
        "domain": "search",
        "context": "post_tool",
        "query_alignment_mode": query_alignment_mode,
        "runtime_trace_path": args.runtime_trace_path,
        "runtime_queries_loaded": len(runtime_queries) if runtime_queries is not None else None,
        "runtime_query_diff_from_question_count": runtime_query_diff_from_question_count,
        "runtime_query_diff_from_question_fraction": (
            runtime_query_diff_from_question_count / n if query_alignment_mode == "runtime_trace" else 0.0
        ),
        "margin_relevant_mean": float(np.mean(margins_rel)),
        "margin_irrelevant_mean": float(np.mean(margins_irr)),
        "margin_diff_mean": float(np.mean(margin_diffs)),
        "correct_sign_fraction": n_correct_sign / n,
        "description": (
            "d = h_irrelevant - h_relevant (decision_direction, toward NON-ADOPT). "
            "Paired observation method: same question/query, relevant vs irrelevant search results. "
            "Eliminates domain mismatch (search, not calculator) and context mismatch "
            "(post-tool Step 1, not pre-tool Step 0). "
            f"Query alignment mode={query_alignment_mode}."
        ),
    }
    summary_path = str(out_path).replace(".npz", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

