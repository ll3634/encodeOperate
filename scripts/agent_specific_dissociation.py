#!/usr/bin/env python3
"""
Agent-Specific Dissociation Test.

Tests whether premature commitment is agent-scaffolding-specific by comparing
4 conditions on the same 486 HotpotQA samples:

  A) Evidence sufficiency judge: "Is the evidence sufficient?" → Yes/No
  B) Non-agentic action selector: "Next step: A(search) or B(answer)?" → A/B
  C) Agent-lite A/B: same agent context (tool list) but A/B response format → A/B
  D) Full ReAct agent (existing baseline traces, no re-run needed)

C uses the agent's system prompt (with tool descriptions) but asks the model
to reply with "A" (search) or "B" (answer), matching B's response format.
margin_c = logit("A") - logit("B") is recorded as a continuous signal.
Non-canonical outputs (not starting with A or B) are counted separately.

Key metrics:
  - Suppression gap = cond_B_continue_rate - cond_D_continue_rate
  - Strict agent-induced dissociation = P(A=insufficient AND B=continue AND D=stop)
  - Lite-to-full gap = cond_C_continue_rate - cond_D_continue_rate
  - McNemar tests for B vs D, C vs D
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.prompts import DEFAULT_SYSTEM_PROMPT, TOOL_DESCRIPTIONS, parse_action


# ===========================================================================
# Constants
# ===========================================================================

A3_RESCUED_VIA_SEARCH = [
    "5abaee845542996606241696",
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
]


# ===========================================================================
# Prompt templates
# ===========================================================================

SYSTEM_PROMPT_A = """You are an expert at judging evidence quality.

Given a question and a retrieved search result, judge whether the evidence is sufficient to answer the question completely and correctly.

Respond with exactly one word: YES or NO.
- YES: The evidence is sufficient to give a complete, accurate answer.
- NO: The evidence is incomplete, unclear, or does not contain the required information."""

SYSTEM_PROMPT_B = """You are a research assistant helping answer multi-hop questions.

Given a question and a retrieved search result, decide whether to:
  A) Search for more information
  B) Answer the question with the current evidence

Respond with exactly one letter: A or B."""

SYSTEM_PROMPT_C = """You are a helpful assistant that answers questions using available tools.

Available tools:
- search(query): Search for information about a topic

Given a question and a search result you have already retrieved, choose your next step:
  A. Search for more information
  B. Give the final answer now

Answer with exactly one letter first: A or B."""


def build_user_prompt_a(question: str, observation: str) -> str:
    return f"""Question: {question}

Retrieved evidence:
{observation}

Is the evidence sufficient to answer this question completely and correctly? Answer YES or NO."""


def build_user_prompt_b(question: str, observation: str) -> str:
    return f"""Question: {question}

Retrieved evidence:
{observation}

Based on this evidence, should you:
  A) Search for more information
  B) Answer the question with the current evidence

Choose A or B."""


def build_user_prompt_c(question: str, observation: str) -> str:
    return f"""Question: {question}

Retrieved evidence:
{observation}

Choose A (search more) or B (answer now)."""


# Condition C_flipped: agent context, A=answer, B=search (consistent system+user, mirrors B_flipped)
SYSTEM_PROMPT_C_FLIPPED = """You are a helpful assistant that answers questions using available tools.

Available tools:
- search(query): Search for information about a topic

Given a question and a search result you have already retrieved, choose your next step:
  A. Give the final answer now
  B. Search for more information

Answer with exactly one letter first: A or B."""


def build_user_prompt_c_flipped(question: str, observation: str) -> str:
    return f"""Question: {question}

Retrieved evidence:
{observation}

Choose A (answer now) or B (search more)."""


def parse_c_flipped(greedy_first: str) -> Tuple[Optional[bool], bool]:
    """
    Strict parser for C_flipped (A=answer=stop, B=search=continue).
    Returns (decision, is_canonical).
    """
    s = greedy_first.strip().upper()
    if s == "A":
        return False, True   # A=answer=stop
    if s == "B":
        return True, True    # B=search=continue
    return None, False


# Condition B_flipped: same as B but A=answer, B=search (order-sensitivity check)
SYSTEM_PROMPT_B_FLIPPED = """You are a research assistant helping answer multi-hop questions.

Given a question and a retrieved search result, decide whether to:
  A) Answer the question with the current evidence
  B) Search for more information

Respond with exactly one letter: A or B."""


def build_user_prompt_b_flipped(question: str, observation: str) -> str:
    return f"""Question: {question}

Retrieved evidence:
{observation}

Based on this evidence, should you:
  A) Answer the question with the current evidence
  B) Search for more information

Choose A or B."""


# ---- A_v2: reasoning-first sufficiency judge ----------------------------------
# Asks the model to reason 2-3 sentences BEFORE giving the verdict, ruling out
# "default-to-NO" bias observed in Condition A (88.9% NO rate).
SYSTEM_PROMPT_A_V2 = """You are an expert at judging evidence quality.

Given a question and a retrieved search result, assess whether the evidence is sufficient to answer the question completely and correctly.

First explain your reasoning in 2-3 sentences. Then conclude your response with exactly one of these two words on its own line:
SUFFICIENT
INSUFFICIENT"""


def build_user_prompt_a_v2(question: str, observation: str) -> str:
    return f"""Question: {question}

Retrieved evidence:
{observation}

Assess the evidence. Explain your reasoning (2-3 sentences), then conclude with SUFFICIENT or INSUFFICIENT."""


def parse_a_v2(text: str) -> Optional[bool]:
    """
    Parse A_v2 response → True=sufficient, False=insufficient, None=fail.

    The model is instructed to end with SUFFICIENT or INSUFFICIENT.
    Key complexity: "INSUFFICIENT" contains "SUFFICIENT" as a suffix (+2 offset),
    so rfind("SUFFICIENT") may land inside an INSUFFICIENT match.

    Strategy: find last INSUFFICIENT (last_ins) and last SUFFICIENT (last_suf).
    If last_suf == last_ins + 2, that SUFFICIENT is embedded in INSUFFICIENT → False.
    If last_suf > last_ins + 2, a standalone SUFFICIENT appears after → True.
    If last_suf < last_ins + 2, INSUFFICIENT comes last → False.
    """
    t = text.strip().upper()
    last_ins = t.rfind("INSUFFICIENT")
    last_suf = t.rfind("SUFFICIENT")
    if last_ins == -1 and last_suf == -1:
        return None
    if last_ins == -1:
        return True   # only standalone SUFFICIENT found
    if last_suf == -1:
        return False  # only INSUFFICIENT found (shouldn't happen since it contains SUFFICIENT)
    # Both present. Is the last SUFFICIENT the one embedded in INSUFFICIENT?
    suf_inside_ins = last_ins + 2  # "IN" prefix = 2 chars
    if last_suf == suf_inside_ins:
        return False  # last SUFFICIENT is embedded in INSUFFICIENT
    elif last_suf > suf_inside_ins:
        return True   # standalone SUFFICIENT appears after the last INSUFFICIENT
    else:
        return False  # SUFFICIENT only appears before INSUFFICIENT → INSUFFICIENT is last


# ---- B_v2 / C_v2: per-sample randomised option order -------------------------
# For each sample index i, FLIP_MASK[i] is 0 (standard) or 1 (flipped).
# Standard:  A=search, B=answer
# Flipped:   A=answer, B=search
# The parser uses FLIP_MASK[i] to map the letter back to the action.
import random as _random

def build_flip_mask(n: int, seed: int = 42) -> List[int]:
    rng = _random.Random(seed)
    return [rng.randint(0, 1) for _ in range(n)]


def build_user_prompt_b_v2(question: str, observation: str, flip: int) -> str:
    if flip == 0:  # standard: A=search, B=answer
        opts = "  A) Search for more information\n  B) Answer the question with the current evidence"
    else:          # flipped: A=answer, B=search
        opts = "  A) Answer the question with the current evidence\n  B) Search for more information"
    return f"""Question: {question}

Retrieved evidence:
{observation}

Based on this evidence, should you:
{opts}

Choose A or B."""


def build_user_prompt_c_v2(question: str, observation: str, flip: int) -> str:
    if flip == 0:
        opts = "  A. Search for more information\n  B. Give the final answer now"
    else:
        opts = "  A. Give the final answer now\n  B. Search for more information"
    return f"""Question: {question}

Retrieved evidence:
{observation}

{opts}

Answer with exactly one letter first: A or B."""


def parse_bc_v2(text: str, flip: int) -> Optional[bool]:
    """
    Parse B_v2 / C_v2 response → True=continue_search, False=stop, None=fail.
    flip=0: A=search (True), B=answer (False)
    flip=1: A=answer (False), B=search (True)
    """
    t = text.strip().upper()
    if t.startswith("A"):
        letter = "A"
    elif t.startswith("B"):
        letter = "B"
    else:
        m = re.search(r'\b([AB])\b', t)
        if m:
            letter = m.group(1)
        else:
            return None
    if flip == 0:
        return letter == "A"   # A=search=continue
    else:
        return letter == "B"   # B=search=continue


# Condition E: prompted evaluation inside the full ReAct context
# System prompt is the real DEFAULT_SYSTEM_PROMPT (same as baseline).
SYSTEM_PROMPT_E = DEFAULT_SYSTEM_PROMPT.format(
    tool_descriptions="- " + TOOL_DESCRIPTIONS["search"]
)

# Assessment text inserted as a user turn after step-0 scratchpad,
# before step-1 generation.
ASSESSMENT_TEXT = (
    "Before generating your next step, briefly assess: "
    "do you have sufficient evidence to fully answer the question? "
    "Then proceed with your Action or Final Answer."
)


def build_messages_e(question: str, action_input: str, observation: str) -> List[dict]:
    """
    Build 4-turn conversation for Condition E:
      system  : DEFAULT_SYSTEM_PROMPT (identical to baseline)
      user    : question (no prefix, identical to baseline)
      assistant: step-0 scratchpad
      user    : assessment text (injected before step-1)
    """
    scratchpad = f"Action: search\nAction Input: {action_input}\nObservation: {observation}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT_E},
        {"role": "user", "content": question},
        {"role": "assistant", "content": scratchpad},
        {"role": "user", "content": ASSESSMENT_TEXT},
    ]


def parse_e(text: str) -> Optional[bool]:
    """
    Parse Condition E free-form output.
    Uses parse_action() from prompts.py.
    Returns True=continue (action=search), False=stop (final_answer set), None=fail.
    """
    parsed = parse_action(text)
    if parsed["final_answer"] is not None:
        return False  # stop
    if parsed["action"] is not None:
        action = parsed["action"].lower().strip()
        if "search" in action or action == "action":
            return True  # continue
    # Last-ditch: look for "Action: search" literally (case-insensitive)
    if re.search(r'\baction\b.*\bsearch\b', text, re.IGNORECASE | re.DOTALL):
        return True
    return None


def parse_b_flipped(text: str) -> Optional[bool]:
    """
    Parse Condition B_flipped → True=continue_search (B), False=answer (A), None=fail.
    (Inverted: B=search, A=answer.)
    """
    t = text.strip().upper()
    if t.startswith("A"):
        return False  # A=answer → stop
    if t.startswith("B"):
        return True   # B=search → continue
    m = re.search(r'\b([AB])\b', t)
    if m:
        return m.group(1) == "B"
    return None


# ===========================================================================
# Data loading
# ===========================================================================

def load_labels(labels_path: str) -> Dict[str, dict]:
    """Load phase1 labels into a dict keyed by sample_id."""
    labels = {}
    with open(labels_path) as f:
        for line in f:
            rec = json.loads(line)
            labels[rec["sample_id"]] = rec
    return labels


def load_baseline_step0(baseline_path: str) -> Dict[str, dict]:
    """Load step-0 data from baseline traces, keyed by sample_id.

    Returns dict with keys: observation, action_input (for Condition E scratchpad).
    """
    step0 = {}
    with open(baseline_path) as f:
        for line in f:
            rec = json.loads(line)
            sid = rec["sample_id"]
            for step in rec.get("steps", []):
                if step.get("step_idx") == 0 and step.get("observation"):
                    step0[sid] = {
                        "observation": step["observation"],
                        "action_input": step.get("action_input") or "",
                    }
                    break
    return step0


def build_sample_table(
    labels: Dict[str, dict],
    step0_data: Dict[str, dict],
) -> List[dict]:
    """Build the per-sample table for the experiment."""
    rows = []
    for sid, lbl in labels.items():
        if sid not in step0_data:
            continue
        rows.append({
            "sample_id": sid,
            "question": lbl["question"],
            "gold_answer": lbl["gold_answer"],
            "observation": step0_data[sid]["observation"],
            "action_input": step0_data[sid]["action_input"],  # for Condition E
            "evidence_label": lbl["label"],           # 0=insufficient, 1=sufficient(ish)
            "n_sf_retrieved": lbl["n_sf_retrieved"],
            "behavioral_stop_D": lbl["behavioral_stop"],  # Condition D
        })
    return rows


# ===========================================================================
# Model inference
# ===========================================================================

def load_model(model_name: str):
    """Load instruct model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    messages: List[dict],
    max_new_tokens: int = 20,
) -> str:
    """Run one forward pass and return generated text."""
    import torch

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_margin_c(model, tokenizer, messages: List[dict]) -> Tuple[float, str, str]:
    """
    For Condition C (agent-lite A/B format), compute margin and greedy first token.

    margin_c = logit("A") - logit("B")  (positive → model prefers search)

    Returns (margin, greedy_first_char, raw_generated_text).
    greedy_first_char is the single greedy token decoded, used to classify:
      - "A" → canonical continue
      - "B" → canonical stop
      - anything else → non-canonical
    """
    import torch

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Token IDs for "A" and "B" as standalone first tokens
    a_ids = tokenizer.encode("A", add_special_tokens=False)
    b_ids = tokenizer.encode("B", add_special_tokens=False)
    a_tok = a_ids[0] if a_ids else None
    b_tok = b_ids[0] if b_ids else None

    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    a_lp = log_probs[a_tok].item() if a_tok is not None else -100.0
    b_lp = log_probs[b_tok].item() if b_tok is not None else -100.0
    margin = a_lp - b_lp  # positive = prefers A (search)

    # Greedy first token
    top_tok = logits.argmax().item()
    top_str = tokenizer.decode([top_tok]).strip()

    return margin, top_str, top_str


# ===========================================================================
# Parsing helpers
# ===========================================================================

def parse_a(text: str) -> Optional[bool]:
    """Parse condition A response → True=sufficient, False=insufficient, None=fail."""
    t = text.strip().upper()
    if t.startswith("YES"):
        return True
    if t.startswith("NO"):
        return False
    # Try to find YES/NO anywhere
    if "YES" in t:
        return True
    if "NO" in t:
        return False
    return None


def parse_b(text: str) -> Optional[bool]:
    """Parse condition B → True=continue_search (A), False=answer (B), None=fail."""
    t = text.strip().upper()
    if t.startswith("A"):
        return True
    if t.startswith("B"):
        return False
    # Look for lone A or B
    m = re.search(r'\b([AB])\b', t)
    if m:
        return m.group(1) == "A"
    return None


def parse_c(greedy_first: str) -> Tuple[Optional[bool], bool]:
    """
    Parse condition C (A/B format) strictly.

    Only accepts greedy first token that is exactly "A" or "B" (case-insensitive).
    Everything else is non-canonical (counted separately).

    Returns:
        (decision, is_canonical)
        decision: True=continue (A), False=stop (B), None=non-canonical
        is_canonical: True if first token was "A" or "B"
    """
    s = greedy_first.strip().upper()
    if s == "A":
        return True, True
    if s == "B":
        return False, True
    return None, False


# ===========================================================================
# Statistical test
# ===========================================================================

def mcnemar_test(table: np.ndarray) -> float:
    """
    McNemar test on 2x2 table.
    table[i][j]: i=condition1 (0=stop,1=continue), j=condition2 (0=stop,1=continue)
    Returns p-value (two-sided).
    """
    from scipy.stats import binom

    b = table[1][0]  # c1=continue, c2=stop
    c = table[0][1]  # c1=stop, c2=continue
    n = b + c
    if n == 0:
        return 1.0
    # Exact binomial (mid-p)
    p = 2 * min(binom.cdf(min(b, c), n, 0.5), 1 - binom.cdf(min(b, c) - 1, n, 0.5))
    return p


# ===========================================================================
# Main experiment logic
# ===========================================================================

def run_conditions(
    model,
    tokenizer,
    samples: List[dict],
    output_dir: Path,
    dry_run: bool = False,
) -> List[dict]:
    """Run all conditions for all samples. D comes from labels."""

    # Per-sample flip mask for B_v2 / C_v2 (fixed seed, reproducible)
    flip_mask = build_flip_mask(len(samples), seed=42)

    out_file = output_dir / ("dry_run_raw.jsonl" if dry_run else "raw_results.jsonl")

    # Required fields for a valid cached row (all conditions present).
    REQUIRED_FIELDS = {"bf_parse_ok", "e_parse_ok", "bv2_parse_ok", "cv2_parse_ok", "av2_parse_ok", "cf_parse_ok"}

    # --- Resume: load already-completed sample IDs ---
    done_ids: set = set()
    results: List[dict] = []
    if out_file.exists() and not dry_run:
        rows_loaded = []
        with open(out_file, "r") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows_loaded.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        # Schema guard: reject cache if required fields are missing
        if rows_loaded and not REQUIRED_FIELDS.issubset(rows_loaded[0].keys()):
            print(f"[Resume] Stale cache detected (missing fields). Ignoring {len(rows_loaded)} cached rows.")
        elif rows_loaded:
            for row in rows_loaded:
                done_ids.add(row["sample_id"])
                results.append(row)
            print(f"[Resume] Found {len(done_ids)} already-completed samples, skipping them.")

    with open(out_file, "a") as fout:
        for i, sample in enumerate(tqdm(samples, desc="Running conditions")):
            sid = sample["sample_id"]

            # Skip if already done
            if sid in done_ids:
                continue
            q = sample["question"]
            obs = sample["observation"]
            flip = flip_mask[i]  # 0=standard, 1=flipped for B_v2/C_v2

            # --- Condition A ---
            msg_a = [
                {"role": "system", "content": SYSTEM_PROMPT_A},
                {"role": "user", "content": build_user_prompt_a(q, obs)},
            ]
            raw_a = generate_response(model, tokenizer, msg_a, max_new_tokens=10)
            parsed_a = parse_a(raw_a)  # True=sufficient, False=insufficient, None=fail

            # --- Condition B ---
            msg_b = [
                {"role": "system", "content": SYSTEM_PROMPT_B},
                {"role": "user", "content": build_user_prompt_b(q, obs)},
            ]
            raw_b = generate_response(model, tokenizer, msg_b, max_new_tokens=10)
            parsed_b = parse_b(raw_b)  # True=continue, False=stop, None=fail

            # --- Condition C ---
            msg_c = [
                {"role": "system", "content": SYSTEM_PROMPT_C},
                {"role": "user", "content": build_user_prompt_c(q, obs)},
            ]
            margin_c, greedy_c, raw_c_str = generate_margin_c(model, tokenizer, msg_c)
            parsed_c, c_canonical = parse_c(greedy_c)  # True=A(continue), False=B(stop), None=non-canonical

            # --- Condition A_v2 (reasoning + SUFFICIENT/INSUFFICIENT) ---
            msg_av2 = [
                {"role": "system", "content": SYSTEM_PROMPT_A_V2},
                {"role": "user", "content": build_user_prompt_a_v2(q, obs)},
            ]
            raw_av2 = generate_response(model, tokenizer, msg_av2, max_new_tokens=100)
            parsed_av2 = parse_a_v2(raw_av2)  # True=sufficient, False=insufficient, None=fail

            # --- Condition B_v2 (randomised option order) ---
            msg_bv2 = [
                {"role": "system", "content": SYSTEM_PROMPT_B},
                {"role": "user", "content": build_user_prompt_b_v2(q, obs, flip)},
            ]
            raw_bv2 = generate_response(model, tokenizer, msg_bv2, max_new_tokens=10)
            parsed_bv2 = parse_bc_v2(raw_bv2, flip)  # True=continue, False=stop, None=fail

            # --- Condition C_v2 (randomised option order, agent context) ---
            msg_cv2 = [
                {"role": "system", "content": SYSTEM_PROMPT_C},
                {"role": "user", "content": build_user_prompt_c_v2(q, obs, flip)},
            ]
            # Still use generate_margin_c to get the logit margin between A and B tokens
            margin_cv2, greedy_cv2, _ = generate_margin_c(model, tokenizer, msg_cv2)
            parsed_cv2, cv2_canonical = parse_c(greedy_cv2)  # True=A(continue), False=B(stop), None=non-canonical
            # Remap: if flip=1, A means "answer" (stop), so A canonical → parsed_cv2 = False actually
            # generate_margin_c's parse_c returns True if greedy=="A". With flip=1 that means stop.
            # We need to apply the flip correction after canonical check:
            if cv2_canonical and parsed_cv2 is not None and flip == 1:
                parsed_cv2 = not parsed_cv2  # A→stop, B→continue when flipped

            # --- Condition C_flipped (consistent A=answer, B=search; mirrors B_flipped for C) ---
            msg_cf = [
                {"role": "system", "content": SYSTEM_PROMPT_C_FLIPPED},
                {"role": "user", "content": build_user_prompt_c_flipped(q, obs)},
            ]
            margin_cf_raw, greedy_cf, _ = generate_margin_c(model, tokenizer, msg_cf)
            # For C_flipped: B=search=continue, so the "continue" margin = logit(B) - logit(A) = -raw
            margin_cf = -margin_cf_raw
            parsed_cf, cf_canonical = parse_c_flipped(greedy_cf)

            # --- Condition B_flipped (order sensitivity check) ---
            msg_bf = [
                {"role": "system", "content": SYSTEM_PROMPT_B_FLIPPED},
                {"role": "user", "content": build_user_prompt_b_flipped(q, obs)},
            ]
            raw_bf = generate_response(model, tokenizer, msg_bf, max_new_tokens=10)
            parsed_bf = parse_b_flipped(raw_bf)  # True=continue(B), False=stop(A), None=fail

            # --- Condition E (prompted ReAct evaluation) ---
            action_input = sample.get("action_input", "")
            msg_e = build_messages_e(q, action_input, obs)
            raw_e = generate_response(model, tokenizer, msg_e, max_new_tokens=80)
            parsed_e = parse_e(raw_e)  # True=continue, False=stop, None=fail

            # --- Condition D: from labels ---
            d_stop = sample["behavioral_stop_D"]  # True=stop, False=continue
            d_continue = not d_stop

            row = {
                "sample_id": sid,
                "question": q,
                "evidence_label": sample["evidence_label"],
                "n_sf_retrieved": sample["n_sf_retrieved"],
                "is_a3_rescued": sid in A3_RESCUED_VIA_SEARCH,
                # Condition A
                "raw_a": raw_a,
                "parsed_a": parsed_a,          # True=sufficient, False=insufficient
                "a_insufficient": (parsed_a == False),
                "a_parse_ok": (parsed_a is not None),
                # Condition B
                "raw_b": raw_b,
                "parsed_b": parsed_b,          # True=continue, False=stop
                "b_continue": (parsed_b == True),
                "b_parse_ok": (parsed_b is not None),
                # Condition C
                "raw_c": greedy_c,             # greedy first token (e.g. "A", "B", "Search", ...)
                "margin_c": margin_c,          # logit(A) - logit(B)
                "parsed_c": parsed_c,          # True=continue(A), False=stop(B), None=non-canonical
                "c_canonical": c_canonical,    # True iff first token was exactly A or B
                "c_continue": (parsed_c == True),
                "c_parse_ok": c_canonical,     # strict: only canonical outputs count
                # Condition A_v2 (reasoning + SUFFICIENT/INSUFFICIENT)
                "raw_av2": raw_av2[:300],
                "parsed_av2": parsed_av2,       # True=sufficient, False=insufficient, None=fail
                "av2_insufficient": (parsed_av2 == False),
                "av2_parse_ok": (parsed_av2 is not None),
                # Condition B_v2 (randomised option order, debiased)
                "flip_bv2": flip,
                "raw_bv2": raw_bv2,
                "parsed_bv2": parsed_bv2,       # True=continue, False=stop (already flip-corrected)
                "bv2_continue": (parsed_bv2 == True),
                "bv2_parse_ok": (parsed_bv2 is not None),
                # Condition C_v2 (randomised option order, agent context — contaminated by prompt contradiction)
                "flip_cv2": flip,
                "raw_cv2": greedy_cv2,
                # margin_cv2: always logit(search_option) - logit(answer_option) for comparability
                # flip=0: search=A → margin = logit(A)-logit(B) (positive)
                # flip=1: search=B → margin = logit(B)-logit(A) = -raw
                "margin_cv2": margin_cv2 if flip == 0 else -margin_cv2,
                "parsed_cv2": parsed_cv2,       # True=continue, False=stop (flip-corrected)
                "cv2_canonical": cv2_canonical,
                "cv2_continue": (parsed_cv2 == True),
                "cv2_parse_ok": cv2_canonical,
                # Condition C_flipped (consistent A=answer, B=search; clean debiasing partner for C)
                "raw_cf": greedy_cf,
                "margin_cf": margin_cf,         # logit(B)-logit(A), positive = prefers search
                "parsed_cf": parsed_cf,
                "cf_canonical": cf_canonical,
                "cf_continue": (parsed_cf == True),
                "cf_parse_ok": cf_canonical,
                # Condition B_flipped (A=answer, B=search)
                "raw_bf": raw_bf,
                "parsed_bf": parsed_bf,        # True=continue(B), False=stop(A), None=fail
                "bf_continue": (parsed_bf == True),
                "bf_parse_ok": (parsed_bf is not None),
                # Condition D
                "d_continue": d_continue,
                "d_stop": d_stop,
                # Condition E (prompted ReAct)
                "raw_e": raw_e[:200],          # truncate for storage
                "parsed_e": parsed_e,          # True=continue, False=stop, None=fail
                "e_continue": (parsed_e == True),
                "e_parse_ok": (parsed_e is not None),
            }
            results.append(row)
            fout.write(json.dumps(row) + "\n")
            fout.flush()

            if (i + 1) % 50 == 0:
                n_done = len(results)
                print(f"  [{i+1}/{len(samples)}] "
                      f"Cf_nc={sum(1 for r in results if not r.get('cf_canonical',True))}/{n_done} "
                      f"Bf_pf={sum(1 for r in results if not r.get('bf_parse_ok',True))}/{n_done} "
                      f"E_pf={sum(1 for r in results if not r.get('e_parse_ok',True))}/{n_done} "
                      f"Av2_pf={sum(1 for r in results if not r.get('av2_parse_ok',True))}/{n_done}")

    return results


def compute_metrics(results: List[dict]) -> dict:
    """Compute all key metrics from results."""
    n = len(results)

    # Parse failure / non-canonical rates
    a_pf = sum(1 for r in results if not r["a_parse_ok"])
    av2_pf = sum(1 for r in results if not r.get("av2_parse_ok", True))
    b_pf = sum(1 for r in results if not r["b_parse_ok"])
    bv2_pf = sum(1 for r in results if not r.get("bv2_parse_ok", True))
    bf_pf = sum(1 for r in results if not r.get("bf_parse_ok", True))
    c_pf = sum(1 for r in results if not r["c_parse_ok"])
    c_noncanon = sum(1 for r in results if not r["c_canonical"])
    c_noncanon_examples = [r["raw_c"] for r in results if not r["c_canonical"]][:10]
    cv2_noncanon = sum(1 for r in results if not r.get("cv2_canonical", True))
    cf_noncanon = sum(1 for r in results if not r.get("cf_canonical", True))
    e_pf = sum(1 for r in results if not r.get("e_parse_ok", True))

    # Among parseable samples
    a_valid = [r for r in results if r["a_parse_ok"]]
    av2_valid = [r for r in results if r.get("av2_parse_ok")]
    b_valid = [r for r in results if r["b_parse_ok"]]
    bv2_valid = [r for r in results if r.get("bv2_parse_ok")]
    bf_valid = [r for r in results if r.get("bf_parse_ok")]
    c_valid = [r for r in results if r["c_parse_ok"]]
    cv2_valid = [r for r in results if r.get("cv2_parse_ok")]
    cf_valid = [r for r in results if r.get("cf_parse_ok")]
    e_valid = [r for r in results if r.get("e_parse_ok")]

    # Continue / insufficient rates (for parseable samples)
    a_insufficient_rate = sum(1 for r in a_valid if r["a_insufficient"]) / len(a_valid) if a_valid else 0.0
    av2_insufficient_rate = sum(1 for r in av2_valid if r["av2_insufficient"]) / len(av2_valid) if av2_valid else 0.0
    b_continue_rate = sum(1 for r in b_valid if r["b_continue"]) / len(b_valid) if b_valid else 0.0
    bv2_continue_rate = sum(1 for r in bv2_valid if r["bv2_continue"]) / len(bv2_valid) if bv2_valid else 0.0
    bf_continue_rate = sum(1 for r in bf_valid if r["bf_continue"]) / len(bf_valid) if bf_valid else 0.0
    c_continue_rate = sum(1 for r in c_valid if r["c_continue"]) / len(c_valid) if c_valid else 0.0
    cv2_continue_rate = sum(1 for r in cv2_valid if r["cv2_continue"]) / len(cv2_valid) if cv2_valid else 0.0
    cf_continue_rate = sum(1 for r in cf_valid if r["cf_continue"]) / len(cf_valid) if cf_valid else 0.0
    d_continue_rate = sum(1 for r in results if r["d_continue"]) / n
    e_continue_rate = sum(1 for r in e_valid if r["e_continue"]) / len(e_valid) if e_valid else 0.0

    # Debiased estimates (average of matched-pair conditions with consistent prompts)
    # debiased_B = (B + B_flipped) / 2  — both have consistent system+user prompts
    debiased_B = (b_continue_rate + bf_continue_rate) / 2 if (b_valid and bf_valid) else None
    # debiased_C = (C + C_flipped) / 2  — both have consistent system+user prompts
    debiased_C = (c_continue_rate + cf_continue_rate) / 2 if (c_valid and cf_valid) else None

    # Definition 5: Suppression gap = B_continue_rate - D_continue_rate (B parseable)
    b_valid_d = [r for r in b_valid]
    b_continue_n = sum(1 for r in b_valid_d if r["b_continue"])
    d_continue_among_bvalid = sum(1 for r in b_valid_d if r["d_continue"])
    suppression_gap = (b_continue_n / len(b_valid_d) if b_valid_d else 0.0) - (d_continue_among_bvalid / len(b_valid_d) if b_valid_d else 0.0)

    # Definition 6: Strict agent-induced dissociation
    # Primary (biased): A=insufficient AND B=continue AND D=stop
    abc_valid = [r for r in results if r["a_parse_ok"] and r["b_parse_ok"]]
    strict_dissociation_n = sum(1 for r in abc_valid if r["a_insufficient"] and r["b_continue"] and r["d_stop"])
    strict_dissociation_rate = strict_dissociation_n / len(abc_valid) if abc_valid else 0.0

    # Debiased: A=insufficient AND B_flipped=continue AND D=stop
    # B_flipped uses consistent prompts (A=answer, B=search) → unbiased toward first option
    abf_valid = [r for r in results if r["a_parse_ok"] and r.get("bf_parse_ok")]
    strict_dissociation_debiased_n = sum(1 for r in abf_valid if r["a_insufficient"] and r["bf_continue"] and r["d_stop"])
    strict_dissociation_debiased_rate = strict_dissociation_debiased_n / len(abf_valid) if abf_valid else 0.0

    # Definition 7: Lite-to-full gap = C_continue_rate - D_continue_rate (C parseable)
    c_valid_d = [r for r in c_valid]
    c_continue_n = sum(1 for r in c_valid_d if r["c_continue"])
    d_continue_among_cvalid = sum(1 for r in c_valid_d if r["d_continue"])
    lite_to_full_gap = (c_continue_n / len(c_valid_d) if c_valid_d else 0.0) - (d_continue_among_cvalid / len(c_valid_d) if c_valid_d else 0.0)

    # McNemar: B vs D (on B-parseable samples)
    bd_table = np.zeros((2, 2), dtype=int)
    for r in b_valid:
        b_cont = int(r["b_continue"])
        d_cont = int(r["d_continue"])
        bd_table[b_cont][d_cont] += 1
    bd_p = mcnemar_test(bd_table)

    # McNemar: C vs D (on C-parseable samples)
    cd_table = np.zeros((2, 2), dtype=int)
    for r in c_valid:
        c_cont = int(r["c_continue"])
        d_cont = int(r["d_continue"])
        cd_table[c_cont][d_cont] += 1
    cd_p = mcnemar_test(cd_table)

    # McNemar: C_flipped vs D
    cfd_table = np.zeros((2, 2), dtype=int)
    for r in cf_valid:
        cfd_table[int(r["cf_continue"])][int(r["d_continue"])] += 1
    cfd_p = mcnemar_test(cfd_table)

    # McNemar: C vs C_flipped (order sensitivity for C)
    ccf_valid = [r for r in results if r["c_parse_ok"] and r.get("cf_parse_ok")]
    ccf_table = np.zeros((2, 2), dtype=int)
    for r in ccf_valid:
        ccf_table[int(r["c_continue"])][int(r["cf_continue"])] += 1
    ccf_p = mcnemar_test(ccf_table)

    # McNemar: B_v2 vs D (on B_v2-parseable samples)
    bv2d_table = np.zeros((2, 2), dtype=int)
    for r in bv2_valid:
        bv2d_table[int(r["bv2_continue"])][int(r["d_continue"])] += 1
    bv2d_p = mcnemar_test(bv2d_table)

    # McNemar: C_v2 vs D (on C_v2-parseable samples)
    cv2d_table = np.zeros((2, 2), dtype=int)
    for r in cv2_valid:
        cv2d_table[int(r["cv2_continue"])][int(r["d_continue"])] += 1
    cv2d_p = mcnemar_test(cv2d_table)

    # McNemar: B vs B_v2 (debiasing effect)
    bbv2_valid = [r for r in results if r["b_parse_ok"] and r.get("bv2_parse_ok")]
    bbv2_table = np.zeros((2, 2), dtype=int)
    for r in bbv2_valid:
        bbv2_table[int(r["b_continue"])][int(r["bv2_continue"])] += 1
    bbv2_p = mcnemar_test(bbv2_table)

    # McNemar: B_flipped vs D (on B_flipped-parseable samples)
    bfd_table = np.zeros((2, 2), dtype=int)
    for r in bf_valid:
        bf_cont = int(r["bf_continue"])
        d_cont = int(r["d_continue"])
        bfd_table[bf_cont][d_cont] += 1
    bfd_p = mcnemar_test(bfd_table)

    # McNemar: B vs B_flipped (on both parseable — order sensitivity test)
    bbf_valid = [r for r in results if r["b_parse_ok"] and r.get("bf_parse_ok")]
    bbf_table = np.zeros((2, 2), dtype=int)
    for r in bbf_valid:
        b_cont = int(r["b_continue"])
        bf_cont = int(r["bf_continue"])
        bbf_table[b_cont][bf_cont] += 1
    bbf_p = mcnemar_test(bbf_table)

    # McNemar: E vs D (on E-parseable samples)
    ed_table = np.zeros((2, 2), dtype=int)
    for r in e_valid:
        e_cont = int(r["e_continue"])
        d_cont = int(r["d_continue"])
        ed_table[e_cont][d_cont] += 1
    ed_p = mcnemar_test(ed_table)

    # A3 rescued subset analysis
    a3_subset = [r for r in results if r["is_a3_rescued"]]
    a3_metrics = {}
    if a3_subset:
        def _rate(grp, key_ok, key_val):
            v = [r for r in grp if r.get(key_ok)]
            return sum(1 for r in v if r.get(key_val)) / len(v) if v else None
        a3_metrics = {
            "n": len(a3_subset),
            "n_label0": sum(1 for r in a3_subset if r["evidence_label"] == 0),
            "n_label1": sum(1 for r in a3_subset if r["evidence_label"] == 1),
            "a_insufficient_rate":   _rate(a3_subset, "a_parse_ok",   "a_insufficient"),
            "av2_insufficient_rate": _rate(a3_subset, "av2_parse_ok", "av2_insufficient"),
            "b_continue_rate":       _rate(a3_subset, "b_parse_ok",   "b_continue"),
            "bv2_continue_rate":     _rate(a3_subset, "bv2_parse_ok", "bv2_continue"),
            "bf_continue_rate":      _rate(a3_subset, "bf_parse_ok",  "bf_continue"),
            "c_continue_rate":       _rate(a3_subset, "c_parse_ok",   "c_continue"),
            "cf_continue_rate":      _rate(a3_subset, "cf_parse_ok",  "cf_continue"),
            "cv2_continue_rate":     _rate(a3_subset, "cv2_parse_ok", "cv2_continue"),
            "d_continue_rate":       sum(1 for r in a3_subset if r["d_continue"]) / len(a3_subset),
            "e_continue_rate":       _rate(a3_subset, "e_parse_ok",   "e_continue"),
            "strict_dissociation_n": sum(1 for r in a3_subset if r.get("a_parse_ok") and r.get("b_parse_ok") and r["a_insufficient"] and r["b_continue"] and r["d_stop"]),
        }

    # By evidence label
    label0 = [r for r in results if r["evidence_label"] == 0]
    label1 = [r for r in results if r["evidence_label"] == 1]

    def rates_for_group(group):
        if not group:
            return {}
        def _r(ok_key, val_key):
            v = [r for r in group if r.get(ok_key)]
            return sum(1 for r in v if r.get(val_key)) / len(v) if v else None
        b_r  = _r("b_parse_ok",  "b_continue")
        bf_r = _r("bf_parse_ok", "bf_continue")
        c_r  = _r("c_parse_ok",  "c_continue")
        cf_r = _r("cf_parse_ok", "cf_continue")
        return {
            "n": len(group),
            "a_insufficient_rate":   _r("a_parse_ok",   "a_insufficient"),
            "av2_insufficient_rate": _r("av2_parse_ok", "av2_insufficient"),
            "b_continue_rate":       b_r,
            "bv2_continue_rate":     _r("bv2_parse_ok", "bv2_continue"),
            "bf_continue_rate":      bf_r,
            "b_debiased_rate":       (b_r + bf_r) / 2 if (b_r is not None and bf_r is not None) else None,
            "c_continue_rate":       c_r,
            "cf_continue_rate":      cf_r,
            "cv2_continue_rate":     _r("cv2_parse_ok", "cv2_continue"),
            "c_debiased_rate":       (c_r + cf_r) / 2 if (c_r is not None and cf_r is not None) else None,
            "d_continue_rate":       sum(1 for r in group if r["d_continue"]) / len(group),
            "e_continue_rate":       _r("e_parse_ok",   "e_continue"),
        }

    return {
        "n_total": n,
        "parse_failures": {
            "A": a_pf, "A_v2": av2_pf,
            "B": b_pf, "B_v2": bv2_pf, "B_flipped": bf_pf,
            "C": c_pf, "C_flipped_noncanon": cf_noncanon, "C_v2_noncanon": cv2_noncanon,
            "E": e_pf,
        },
        "c_noncanon": c_noncanon,
        "c_noncanon_examples": c_noncanon_examples,
        "continue_rates": {
            "A_insufficient":    a_insufficient_rate,
            "A_v2_insufficient": av2_insufficient_rate,
            "B":                 b_continue_rate,
            "B_v2":              bv2_continue_rate,   # NOTE: contaminated by prompt contradiction
            "B_flipped":         bf_continue_rate,
            "B_debiased":        debiased_B,
            "C":                 c_continue_rate,
            "C_flipped":         cf_continue_rate,
            "C_v2":              cv2_continue_rate,   # NOTE: contaminated by prompt contradiction
            "C_debiased":        debiased_C,
            "D":                 d_continue_rate,
            "E":                 e_continue_rate,
        },
        "suppression_gap_B_minus_D":          suppression_gap,
        "suppression_gap_Bdebiased_minus_D":  (debiased_B - d_continue_rate) if debiased_B is not None else None,
        "lite_to_full_gap_C_minus_D":         lite_to_full_gap,
        "lite_to_full_gap_Cdebiased_minus_D": (debiased_C - d_continue_rate) if debiased_C is not None else None,
        "order_sensitivity_B_minus_Bflipped": b_continue_rate - bf_continue_rate,
        "order_sensitivity_C_minus_Cflipped": c_continue_rate - cf_continue_rate,
        "prompted_effect_E_minus_D":          e_continue_rate - d_continue_rate,
        # Strict dissociation (biased: uses B with position bias toward "A")
        "strict_dissociation_rate":           strict_dissociation_rate,
        "strict_dissociation_n":              strict_dissociation_n,
        "strict_dissociation_base_n":         len(abc_valid),
        # Strict dissociation (debiased: uses B_flipped with consistent prompts)
        "strict_dissociation_debiased_rate":  strict_dissociation_debiased_rate,
        "strict_dissociation_debiased_n":     strict_dissociation_debiased_n,
        "strict_dissociation_debiased_base_n": len(abf_valid),
        "mcnemar": {
            "B_vs_D":          {"table": bd_table.tolist(),   "p": bd_p},
            "Bflipped_vs_D":   {"table": bfd_table.tolist(),  "p": bfd_p},
            "B_vs_Bflipped":   {"table": bbf_table.tolist(),  "p": bbf_p},
            "Bv2_vs_D":        {"table": bv2d_table.tolist(), "p": bv2d_p},
            "B_vs_Bv2":        {"table": bbv2_table.tolist(), "p": bbv2_p},
            "C_vs_D":          {"table": cd_table.tolist(),   "p": cd_p},
            "Cflipped_vs_D":   {"table": cfd_table.tolist(),  "p": cfd_p},
            "C_vs_Cflipped":   {"table": ccf_table.tolist(),  "p": ccf_p},
            "Cv2_vs_D":        {"table": cv2d_table.tolist(), "p": cv2d_p},
            "E_vs_D":          {"table": ed_table.tolist(),   "p": ed_p},
        },
        "by_label": {
            "label0_insufficient": rates_for_group(label0),
            "label1_partial":      rates_for_group(label1),
        },
        "a3_rescued_subset": a3_metrics,
    }


def print_dry_run_report(results: List[dict]):
    """Print a concise dry-run report."""
    n = len(results)
    print("\n" + "=" * 70)
    print("DRY RUN REPORT")
    print("=" * 70)

    print(f"\nN = {n} samples")

    print("\nParse success rates:")
    a_ok = sum(1 for r in results if r["a_parse_ok"])
    b_ok = sum(1 for r in results if r["b_parse_ok"])
    c_canon = sum(1 for r in results if r["c_canonical"])
    c_noncanon_ex = [r["raw_c"] for r in results if not r["c_canonical"]]
    print(f"  Condition A (YES/NO): {a_ok}/{n} ({a_ok/n:.0%})")
    print(f"  Condition B (A/B):    {b_ok}/{n} ({b_ok/n:.0%})")
    print(f"  Condition C (A/B, agent ctx): canonical={c_canon}/{n} ({c_canon/n:.0%})")
    if c_noncanon_ex:
        print(f"    Non-canonical examples: {c_noncanon_ex[:5]}")

    bf_ok = sum(1 for r in results if r.get("bf_parse_ok"))
    e_ok = sum(1 for r in results if r.get("e_parse_ok"))
    print(f"  Condition B_flipped (A=ans,B=search): {bf_ok}/{n} ({bf_ok/n:.0%})")
    print(f"  Condition E (prompted ReAct):         {e_ok}/{n} ({e_ok/n:.0%})")

    print("\nSample outputs:")
    for i, r in enumerate(results[:3]):
        print(f"\n  [{i}] {r['sample_id'][:20]}...")
        print(f"       Q: {r['question'][:80]}")
        print(f"       A raw: {repr(r['raw_a'])} → {r['parsed_a']}")
        print(f"       B raw: {repr(r['raw_b'])} → {r['parsed_b']}")
        print(f"       Bf raw: {repr(r.get('raw_bf',''))} → {r.get('parsed_bf')}")
        print(f"       C greedy: {repr(r['raw_c'])} → {r['parsed_c']} (canonical={r['c_canonical']})")
        print(f"       D: stop={r['d_stop']}")
        print(f"       E raw: {repr(r.get('raw_e','')[:100])} → {r.get('parsed_e')}")

    print("\nContinue rates (preliminary):")
    a_v = [r for r in results if r["a_parse_ok"]]
    b_v = [r for r in results if r["b_parse_ok"]]
    bf_v = [r for r in results if r.get("bf_parse_ok")]
    c_v = [r for r in results if r["c_parse_ok"]]
    e_v = [r for r in results if r.get("e_parse_ok")]
    print(f"  A_insufficient: {sum(1 for r in a_v if r['a_insufficient'])/len(a_v):.0%}" if a_v else "  A: N/A")
    print(f"  B_continue:     {sum(1 for r in b_v if r['b_continue'])/len(b_v):.0%}" if b_v else "  B: N/A")
    print(f"  Bf_continue:    {sum(1 for r in bf_v if r['bf_continue'])/len(bf_v):.0%}" if bf_v else "  Bf: N/A")
    print(f"  C_continue:     {sum(1 for r in c_v if r['c_continue'])/len(c_v):.0%}" if c_v else "  C: N/A")
    print(f"  D_continue:     {sum(1 for r in results if r['d_continue'])/n:.0%}")
    print(f"  E_continue:     {sum(1 for r in e_v if r['e_continue'])/len(e_v):.0%}" if e_v else "  E: N/A")


def print_full_report(metrics: dict):
    """Print the full analysis report."""
    print("\n" + "=" * 70)
    print("AGENT-SPECIFIC DISSOCIATION TEST — FULL RESULTS")
    print("=" * 70)

    n = metrics["n_total"]
    pf = metrics["parse_failures"]
    cr = metrics["continue_rates"]

    print(f"\nN = {n}")
    pf_str = "  ".join(f"{k}={v}" for k, v in pf.items())
    print(f"\nParse failures: {pf_str}")

    def _fmt(v):
        return f"{v:.1%}" if v is not None else "N/A"

    print(f"\n{'Cond':<18} {'Rate':>8}   Notes")
    print("-" * 75)
    print(f"{'A (judge)':<18} {cr['A_insufficient']:>7.1%}   1-word YES/NO (may have default-NO bias)")
    print(f"{'A_v2 (reasoning)':<18} {cr['A_v2_insufficient']:>7.1%}   Reasoning + SUFFICIENT/INSUFFICIENT")
    print(f"{'B':<18} {cr['B']:>7.1%}   A=search, B=answer  [fixed, no agent ctx]")
    print(f"{'B_flipped':<18} {cr['B_flipped']:>7.1%}   A=answer, B=search  [fixed, no agent ctx]")
    print(f"{'B_debiased':<18} {_fmt(cr['B_debiased']):>8}   (B + B_flipped) / 2  ← clean estimate")
    print(f"{'B_v2':<18} {cr['B_v2']:>7.1%}   ⚠ contaminated: sys/user contradiction")
    print(f"{'C':<18} {cr['C']:>7.1%}   A=search, B=answer  [fixed, agent ctx]")
    print(f"{'C_flipped':<18} {cr['C_flipped']:>7.1%}   A=answer, B=search  [fixed, agent ctx]")
    print(f"{'C_debiased':<18} {_fmt(cr['C_debiased']):>8}   (C + C_flipped) / 2  ← clean estimate")
    print(f"{'C_v2':<18} {cr['C_v2']:>7.1%}   ⚠ contaminated: sys/user contradiction")
    print(f"{'D (full ReAct)':<18} {cr['D']:>7.1%}   Baseline (multi-turn scratchpad)")
    print(f"{'E (prompted)':<18} {cr['E']:>7.1%}   Assessment prompt before step-1")

    db = cr["B_debiased"]
    dc = cr["C_debiased"]
    _gap = lambda v: f"{(v - cr['D']):+.1%}" if v is not None else "N/A"
    print(f"\nKey metrics:")
    print(f"  Suppression gap (B_debiased - D): {_gap(db)}   ← headline number")
    print(f"  Lite-to-full gap (C_debiased - D): {_gap(dc)}   ← agent context effect")
    print(f"  Order sensitivity B (B - B_flipped): {metrics['order_sensitivity_B_minus_Bflipped']:+.1%}")
    print(f"  Order sensitivity C (C - C_flipped): {metrics['order_sensitivity_C_minus_Cflipped']:+.1%}")
    print(f"  Prompted effect (E - D):             {metrics['prompted_effect_E_minus_D']:+.1%}")
    sd = metrics["strict_dissociation_rate"]
    sdn = metrics["strict_dissociation_n"]
    sdb = metrics["strict_dissociation_base_n"]
    sdd = metrics["strict_dissociation_debiased_rate"]
    sddn = metrics["strict_dissociation_debiased_n"]
    sddb = metrics["strict_dissociation_debiased_base_n"]
    print(f"  Strict dissociation (B-based):         {sd:.1%} ({sdn}/{sdb})  ← biased high")
    print(f"  Strict dissociation (B_flipped-based): {sdd:.1%} ({sddn}/{sddb})  ← debiased")

    mc = metrics["mcnemar"]
    print(f"\nMcNemar tests:")
    for key, label in [
        ("B_vs_D",          "B         vs D       "),
        ("Bflipped_vs_D",   "B_flipped  vs D       "),
        ("B_vs_Bflipped",   "B         vs B_flipped"),
        ("C_vs_D",          "C         vs D       "),
        ("Cflipped_vs_D",   "C_flipped  vs D       "),
        ("C_vs_Cflipped",   "C         vs C_flipped"),
        ("E_vs_D",          "E         vs D       "),
    ]:
        t = mc.get(key, {})
        p = t.get("p", float("nan"))
        sig = " *" if p < 0.05 else "  "
        print(f"  {label}: p = {p:.4f}{sig}")

    print(f"\nBy evidence label:")
    for lname, ldata in metrics["by_label"].items():
        if not ldata:
            continue
        print(f"  {lname} (n={ldata['n']}):")
        if isinstance(ldata.get('a_insufficient_rate'), float):
            print(f"    A_insufficient: {ldata['a_insufficient_rate']:.1%}")
        if isinstance(ldata.get('b_continue_rate'), float):
            print(f"    B_continue:     {ldata['b_continue_rate']:.1%}")
        if isinstance(ldata.get('c_continue_rate'), float):
            print(f"    C_continue:     {ldata['c_continue_rate']:.1%}")
        if isinstance(ldata.get('d_continue_rate'), float):
            print(f"    D_continue:     {ldata['d_continue_rate']:.1%}")

    a3 = metrics.get("a3_rescued_subset", {})
    if a3:
        print(f"\nA3 rescued subset (n={a3['n']}; label0={a3['n_label0']}, label1={a3['n_label1']}):")
        for k, v in a3.items():
            if k.startswith("n"):
                continue
            if isinstance(v, float):
                print(f"    {k}: {v:.1%}")
            elif v is not None:
                print(f"    {k}: {v}")


def save_summary_md(metrics: dict, path: Path):
    """Save a markdown summary."""
    n = metrics["n_total"]
    pf = metrics["parse_failures"]
    cr = metrics["continue_rates"]
    sg = metrics["suppression_gap_B_minus_D"]
    ltf = metrics["lite_to_full_gap_C_minus_D"]
    sd_rate = metrics["strict_dissociation_rate"]
    sd_n = metrics["strict_dissociation_n"]
    sd_base = metrics["strict_dissociation_base_n"]
    mc = metrics["mcnemar"]

    def _p(v):
        return f"{v:.1%}" if v is not None else "N/A"

    db = cr.get("B_debiased")
    dc = cr.get("C_debiased")
    _pgap = lambda v: f"{(v - cr['D']):+.1%}" if v is not None else "N/A"
    _pdiff = lambda a, b: f"{(a - b):+.1%}" if (a is not None and b is not None) else "N/A"
    e_rate = cr.get("E", 0.0)
    bf_rate = cr.get("B_flipped", 0.0)
    cf_rate = cr.get("C_flipped", 0.0)
    os_b = metrics.get("order_sensitivity_B_minus_Bflipped", 0.0)
    os_c = metrics.get("order_sensitivity_C_minus_Cflipped", 0.0)
    pe_gap = metrics.get("prompted_effect_E_minus_D", 0.0)
    sdd_rate = metrics.get("strict_dissociation_debiased_rate", 0.0)
    sdd_n = metrics.get("strict_dissociation_debiased_n", 0)
    sdd_base = metrics.get("strict_dissociation_debiased_base_n", 0)

    lines = [
        "# Agent-Specific Dissociation Test — Summary",
        "",
        f"N = {n} HotpotQA samples (step-1 decision points from baseline traces)",
        "",
        "## Conditions",
        "",
        "| Cond | Description | Rate | vs D |",
        "|------|-------------|------|------|",
        f"| A (judge)    | Is evidence sufficient? NO=insufficient | {cr['A_insufficient']:.1%} | — |",
        f"| A_v2         | Reasoning + SUFFICIENT/INSUFFICIENT     | {cr['A_v2_insufficient']:.1%} | — |",
        f"| B            | A=search, B=answer [no agent ctx]       | {cr['B']:.1%} | {cr['B']-cr['D']:+.1%} |",
        f"| B_flipped    | A=answer, B=search [no agent ctx]       | {bf_rate:.1%} | {bf_rate-cr['D']:+.1%} |",
        f"| **B_debiased** | **(B + B_flipped) / 2**               | **{_p(db)}** | **{_pgap(db)}** |",
        f"| C            | A=search, B=answer [agent ctx]          | {cr['C']:.1%} | {cr['C']-cr['D']:+.1%} |",
        f"| C_flipped    | A=answer, B=search [agent ctx]          | {cf_rate:.1%} | {cf_rate-cr['D']:+.1%} |",
        f"| **C_debiased** | **(C + C_flipped) / 2**               | **{_p(dc)}** | **{_pgap(dc)}** |",
        f"| D (full)     | Full ReAct baseline                     | {cr['D']:.1%} | 0.0% |",
        f"| E (prompted) | Assessment prompt before step-1         | {e_rate:.1%} | {pe_gap:+.1%} |",
        "",
        "> B_v2 and C_v2 are excluded: contaminated by system/user prompt contradiction.",
        "",
        "## Key Metrics",
        "",
        f"- **Suppression gap** (B_debiased − D): **{_pgap(db)}** — agent scaffolding suppresses search",
        f"- **Context effect** (C_debiased − B_debiased): **{_pdiff(dc, db)}** — additional suppression from agent context",
        f"- **Prompted effect** (E − D): **{pe_gap:+.1%}** — verbal assessment prompt does not fix routing failure",
        f"- **Order sensitivity B** (B − B_flipped): **{os_b:+.1%}** — position bias in non-agent context",
        f"- **Order sensitivity C** (C − C_flipped): **{os_c:+.1%}** — position bias in agent context",
        f"- **Strict dissociation (biased, B)**: {sd_rate:.1%} ({sd_n}/{sd_base})",
        f"- **Strict dissociation (debiased, B_flipped)**: **{sdd_rate:.1%}** ({sdd_n}/{sdd_base})",
        "",
        "## Statistical Tests",
        "",
        "| Comparison | p-value | Sig? |",
        "|------------|---------|------|",
    ]
    for key, label in [
        ("B_vs_D",          "B vs D"),
        ("Bflipped_vs_D",   "B_flipped vs D"),
        ("B_vs_Bflipped",   "B vs B_flipped (order)"),
        ("C_vs_D",          "C vs D"),
        ("Cflipped_vs_D",   "C_flipped vs D"),
        ("C_vs_Cflipped",   "C vs C_flipped (order)"),
        ("E_vs_D",          "E vs D"),
    ]:
        t = mc.get(key, {})
        p = t.get("p", float("nan"))
        lines.append(f"| {label} | {p:.4f} | {'Yes *' if p < 0.05 else 'No'} |")

    lines += [
        "",
        "## Parse Failures / Non-canonical",
        "",
        f"- A: {pf['A']}/{n}  A_v2: {pf.get('A_v2',0)}/{n}",
        f"- B: {pf['B']}/{n}  B_flipped: {pf.get('B_flipped',0)}/{n}",
        f"- C non-canonical: {metrics.get('c_noncanon',0)}/{n}",
        f"- C_flipped non-canonical: {pf.get('C_flipped_noncanon',0)}/{n}",
        f"- E: {pf.get('E',0)}/{n}",
        "",
    ]

    a3 = metrics.get("a3_rescued_subset", {})
    if a3 and a3.get("n", 0) > 0:
        lines += [
            "## A3 Rescued Subset",
            "",
            f"N = {a3['n']} (label0={a3['n_label0']}, label1={a3['n_label1']})",
            "",
            "| Condition | Rate |",
            "|-----------|------|",
        ]
        for k in ["a_insufficient_rate", "b_continue_rate", "bf_continue_rate", "c_continue_rate", "d_continue_rate", "e_continue_rate"]:
            v = a3.get(k)
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.1%} |")
        if "strict_dissociation_n" in a3:
            lines.append(f"\nStrict dissociation in A3 subset: {a3['strict_dissociation_n']}/{a3['n']}")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved summary to: {path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Agent-Specific Dissociation Test")
    parser.add_argument(
        "--labels-path",
        default="results/phase1_probe/labels.jsonl",
        help="Phase 1 labels JSONL",
    )
    parser.add_argument(
        "--baseline-path",
        default="results/l20_rho020_n500/baseline_results.jsonl",
        help="Baseline traces JSONL (for step-0 observations)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/agent_specific_dissociation",
        help="Output directory",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run on first 20 samples only",
    )
    parser.add_argument(
        "--dry-run-n",
        type=int,
        default=20,
        help="Number of samples for dry run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    # Paths
    labels_path = Path(args.labels_path)
    baseline_path = Path(args.baseline_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading labels and baseline step-0 data...")
    labels = load_labels(str(labels_path))
    step0_data = load_baseline_step0(str(baseline_path))

    samples = build_sample_table(labels, step0_data)
    print(f"Built sample table: {len(samples)} samples with step-0 observations")

    label_counts = {0: sum(1 for s in samples if s["evidence_label"] == 0),
                    1: sum(1 for s in samples if s["evidence_label"] == 1)}
    d_stop_rate = sum(1 for s in samples if s["behavioral_stop_D"]) / len(samples)
    print(f"Label distribution: label0={label_counts[0]}, label1={label_counts[1]}")
    print(f"Condition D (full ReAct) stop rate: {d_stop_rate:.1%}")
    print(f"A3 rescued samples in table: {sum(1 for s in samples if s['sample_id'] in A3_RESCUED_VIA_SEARCH)}")

    if args.dry_run:
        np.random.seed(args.seed)
        idx = list(range(len(samples)))
        np.random.shuffle(idx)
        samples = [samples[i] for i in idx[:args.dry_run_n]]
        print(f"\nDRY RUN: using {len(samples)} samples")

    # Load model
    model, tokenizer = load_model(args.model)

    # Run
    results = run_conditions(model, tokenizer, samples, output_dir, dry_run=args.dry_run)

    if args.dry_run:
        print_dry_run_report(results)
        # Save dry-run summary
        dr_summary = compute_metrics(results)
        with open(output_dir / "dry_run_summary.json", "w") as f:
            json.dump(dr_summary, f, indent=2)
        print(f"\nDry run complete. Check output at {output_dir}/dry_run_raw.jsonl")
        print("If parse rates look good (>85%), proceed with full run (remove --dry-run flag).")
        return

    # Full run: compute and save metrics
    metrics = compute_metrics(results)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {output_dir}/metrics.json")

    print_full_report(metrics)
    save_summary_md(metrics, output_dir / "summary.md")

    print(f"\nAll outputs in: {output_dir}/")


if __name__ == "__main__":
    main()
