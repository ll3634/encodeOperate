#!/usr/bin/env python3
"""
Phase 2: Run Qwen2.5-7B (base) on HotpotQA N=500.

Captures step-1 (second decision point) activations and margins for:
  - Evidence sufficiency probe comparison (base vs instruct)
  - Action direction extraction (P20/P80 mean-diff at L20)
  - Critical cosine computation: cosine(evidence_dir, action_dir)

Usage:
    cd tmc/scripts/e2e_agent
    python scripts/phase2_base_agent.py \
        --hotpotqa-data data/hotpotqa/hotpot_dev_distractor_v1.json \
        --corpus-path data/hotpotqa/corpus.jsonl \
        --output-dir results/phase2_rlhf_tax \
        --model Qwen/Qwen2.5-7B \
        [--validate-only]  # Run only 10-sample validation
"""

import os, sys, json, re, argparse, time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, parse_action, ACTION_TOKENS, TOOL_DESCRIPTIONS
from steering.hook_utils import get_model_layers
from tools.search_tool import SearchTool
from tools.calculator_tool import CalculatorTool
from datasets.hotpotqa import HotpotQADataset, build_hotpotqa_corpus


# ─── Balanced 4-shot few-shot examples ────────────────────────────────────────
# 2 × STOP  (answer after 1st search)  +  2 × CONTINUE (search again)
# = perfectly balanced 50/50 prior.
# These questions are NOT in the N=500 test set (verified: seed=42, bridge type).
#
# Format: each example is a list of (role, content) tuples that will be injected
# as real ChatML turns, NOT embedded as text in the system prompt.
# This teaches the model that each assistant turn is ONE step.
FEW_SHOT_EXAMPLES = [
    # ── STOP example 1: answer found in first search ──
    {
        "question": "Who was known by his stage name Aladin and helped organizations "
                    "improve their performance as a consultant?",
        "turns": [
            ("search", "Aladin stage name management consultant",
             "[1] Eenasul Fateh: Eenasul Fateh (born 1959), also known by his stage name "
             "Aladin, is a Bangladeshi-British management consultant, magician, and "
             "motivational speaker. He has helped numerous organizations improve their "
             "performance and has written books on management and leadership.\n\n"
             "[2] Management consulting: Management consulting is the practice of helping "
             "organizations improve their performance by analyzing existing business problems."),
        ],
        "final_answer": "Eenasul Fateh",
    },
    # ── STOP example 2: both facts in first result ──
    {
        "question": "What is the name of the fight song of the university whose main "
                    "campus is in Lawrence, Kansas and whose branch campuses are in the "
                    "Kansas City metropolitan area?",
        "turns": [
            ("search", "university Lawrence Kansas main campus fight song",
             "[1] University of Kansas: The University of Kansas (KU) is a public research "
             "university. The main campus is in Lawrence, Kansas. KU also has branch campuses "
             "in the Kansas City metropolitan area.\n\n"
             "[2] Kansas Song: Kansas Song (also known as \"We're From Kansas\") is the "
             "official fight song of the University of Kansas Jayhawks.\n\n"
             "[3] Lawrence, Kansas: Lawrence is a city in northeastern Kansas, home to KU."),
        ],
        "final_answer": "Kansas Song",
    },
    # ── CONTINUE example 1: need second search for missing entity ──
    {
        "question": "What government position was held by the woman who portrayed "
                    "Corliss Archer in the film Kiss and Tell?",
        "turns": [
            ("search", "Kiss and Tell 1945 film Corliss Archer actress",
             "[1] Kiss and Tell (1945 film): Kiss and Tell is a 1945 American comedy film "
             "starring then 17-year-old Shirley Temple as Corliss Archer.\n\n"
             "[2] Shirley Temple: Shirley Temple Black (1928–2014) was an American actress, "
             "singer, dancer, businesswoman, and diplomat.\n\n"
             "[3] Corliss Archer: Corliss Archer is a fictional teenage character from a "
             "series of short stories by F. Hugh Herbert."),
            ("search", "Shirley Temple government position diplomat",
             "[1] Shirley Temple: Shirley Temple Black served as United States Ambassador "
             "to Ghana (1974–1976) and as Chief of Protocol of the United States (1976–1977).\n\n"
             "[2] Chief of Protocol: The Chief of Protocol is a senior diplomatic position "
             "in the United States Department of State."),
        ],
        "final_answer": "Chief of Protocol",
    },
    # ── CONTINUE example 2: need second search for capacity ──
    {
        "question": "The arena where the Lewiston Maineiacs played their home games "
                    "can seat how many people?",
        "turns": [
            ("search", "Lewiston Maineiacs home arena",
             "[1] Lewiston Maineiacs: The Lewiston Maineiacs were a junior ice hockey team "
             "based in Lewiston, Maine. The team played at the Androscoggin Bank Colisée.\n\n"
             "[2] QMJHL: The QMJHL is a major junior ice hockey league.\n\n"
             "[3] Lewiston, Maine: Lewiston is the second-largest city in Maine."),
            ("search", "Androscoggin Bank Colisée seating capacity",
             "[1] Androscoggin Bank Colisée: The Androscoggin Bank Colisée is a multi-purpose "
             "arena in Lewiston, Maine with a total capacity of 4,000 and 3,677 seated.\n\n"
             "[2] Maine arena venues: Maine has several multi-purpose arenas."),
        ],
        "final_answer": "3,677",
    },
]


def build_system_prompt() -> str:
    """Build a clean system prompt (no examples embedded).

    Identical for both base and instruct models.
    Examples are provided as separate multi-turn messages via build_fewshot_messages().
    """
    pb = PromptBuilder()
    tool_desc = pb.get_tool_descriptions()
    return (
        "You are a helpful assistant that answers questions using available tools.\n\n"
        f"Available tools:\n{tool_desc}\n\n"
        "You MUST respond in exactly one of the following formats.\n\n"
        "If you need to use a tool:\n"
        "Action: <tool_name>\nAction Input: <input>\n\n"
        "If you can answer directly:\n"
        "Final Answer: <answer>\n\n"
        "Do NOT write \"Thought:\" and do NOT output any other text before "
        "the first word of your response.\n"
        "Your first word must be either \"Action\" or \"Final\"."
    )


def build_fewshot_messages() -> list:
    """Build balanced few-shot examples as real ChatML conversation turns.

    Each example is encoded as:
      user:      <question>
      assistant: Action: search\\nAction Input: <query>
      user:      Observation: <results>
      [optional: assistant + user for second search]
      assistant: Final Answer: <answer>

    This teaches the model that each assistant turn is ONE step,
    and that <|im_end|> terminates each step.
    """
    msgs = []
    for ex in FEW_SHOT_EXAMPLES:
        msgs.append({"role": "user", "content": ex["question"]})
        for _action, action_input, observation in ex["turns"]:
            msgs.append({"role": "assistant",
                         "content": f"Action: search\nAction Input: {action_input}"})
            msgs.append({"role": "user",
                         "content": f"Observation: {observation}"})
        msgs.append({"role": "assistant",
                     "content": f"Final Answer: {ex['final_answer']}"})
    return msgs


def apply_chat_template(tokenizer, messages: list, add_generation_prompt: bool = True) -> str:
    """Apply chat template, with fallback to manual ChatML."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    except Exception as e:
        print(f"  [WARNING] apply_chat_template failed: {e}. Using manual ChatML.")
        lines = []
        for msg in messages:
            role, content = msg["role"], msg["content"]
            lines.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        if add_generation_prompt:
            lines.append("<|im_start|>assistant")
        return "\n".join(lines)


def _get_stop_token_ids(tokenizer) -> list:
    """Return eos token IDs including ChatML stop tokens for base models."""
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    # Add ChatML terminators so base models stop cleanly
    # Also add <|im_start|> — if the model begins a new turn, we want to stop
    for tok in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|im_start|>"]:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if ids:
            stop_ids.add(ids[0])
    return list(stop_ids)


def generate_step(model, tokenizer, messages: list, max_new_tokens: int = 256) -> str:
    """Generate one agent step. Returns raw model output text.

    Post-processing:
      - Truncates at "Observation:" to prevent the model from generating
        multi-step traces in a single turn.
      - Truncates at "Example " to prevent the model from hallucinating
        additional few-shot examples.
    """
    prompt = apply_chat_template(tokenizer, messages)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    stop_ids = _get_stop_token_ids(tokenizer)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,  # greedy
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_ids,
        )

    new_tokens = output_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ── Post-processing: clean up base model ChatML artifacts ──────────────
    # Base models don't understand ChatML turn structure. They hallucinate
    # role markers ("user", "assistant") as regular text tokens, contaminating
    # both the beginning and end of their output.

    # 0. Strip leading ChatML role markers.  Base models often prepend
    #    "user\n", "assistant\n", or garbled variants before their real content.
    #    We iteratively strip until we hit actual content.
    changed = True
    while changed:
        changed = False
        for prefix in ["user\n", "user ", "assistant\n", "assistant ",
                       "Assistantre\n", "Assistant\n"]:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
                changed = True
                break
    if text.strip().lower() in ("user", "assistant"):
        text = ""

    # 1. Observation: (system provides this, model shouldn't generate it)
    for sentinel in ["\nObservation:", "\nObservation :"]:
        idx = text.find(sentinel)
        if idx != -1:
            text = text[:idx].strip()

    # 2. If model wrote Action + Final Answer in one shot, keep only the first
    #    valid block. If "Action:" comes first, truncate at "Final Answer:".
    #    If "Final Answer:" comes first, that's fine (it's a stop decision).
    action_idx = text.lower().find("action:")
    final_idx = text.lower().find("final answer:")
    if action_idx != -1 and final_idx != -1 and action_idx < final_idx:
        text = text[:final_idx].strip()

    # 3. Base models often regurgitate content then insert ChatML markers
    #    before their real action.  Pattern: "<garbage>assistant\n user\n
    #    Action: search\n...".  Detect and keep only the last valid Action/
    #    Final Answer block.
    #    Strategy: find the LAST "Action:" or "Final Answer:" and keep from
    #    that point (if preceded by role-marker noise).
    tl = text.lower()
    last_action = tl.rfind("action:")
    last_final = tl.rfind("final answer:")
    last_valid = max(last_action, last_final)
    first_valid = min(
        tl.find("action:") if tl.find("action:") != -1 else len(text),
        tl.find("final answer:") if tl.find("final answer:") != -1 else len(text),
    )
    if last_valid > 0 and first_valid != last_valid:
        # There are multiple Action/Final blocks — the earlier ones are likely
        # regurgitated few-shot content.  Keep from the last valid block.
        text = text[last_valid:].strip()

    # 4. Strip trailing ChatML role markers (newline-prefixed at end of text).
    #    Pattern: "Action Input: some query\nuser" or "...assistant\n user"
    changed = True
    while changed:
        changed = False
        for suffix in ["\nuser", "\n user", "\nassistant", "\n assistant"]:
            if text.lower().endswith(suffix.lower()):
                text = text[:-len(suffix)].strip()
                changed = True
                break

    # 5. Strip trailing inline "assistant" / "user" appended directly to
    #    action input (e.g. "query textassistant", "theater user").
    text = re.sub(r'\s*(?:assistant|user)\s*$', '', text, flags=re.IGNORECASE).strip()

    return text


def extract_activation_and_margin(
    model, tokenizer, messages: list,
    layer_indices: list, device
) -> tuple:
    """Single forward pass: capture hidden states at all target layers + margin.

    Returns:
        (hidden_per_layer: dict[int, np.ndarray], margin: float)
        hidden_per_layer[l] has shape (hidden_dim,) = last token activation
        margin = logP("Action") - logP("Final")
    """
    prompt = apply_chat_template(tokenizer, messages)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    model_layers = get_model_layers(model)
    captured = {}

    def make_hook(l):
        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[l] = h[0, -1, :].detach().float().cpu().numpy()
        return hook_fn

    handles = [model_layers[l].register_forward_hook(make_hook(l))
               for l in layer_indices]

    try:
        with torch.no_grad():
            outputs = model(input_ids)
    finally:
        for h in handles:
            h.remove()

    # Compute margin from logits at last token position
    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    tool_ids, finish_ids = [], []
    for tok in ACTION_TOKENS["tool_call"]:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if ids:
            tool_ids.append(ids[0])
    for tok in ACTION_TOKENS["finish"]:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if ids:
            finish_ids.append(ids[0])

    tool_lp = torch.logsumexp(log_probs[tool_ids], dim=0).item() if tool_ids else -100.0
    fin_lp = torch.logsumexp(log_probs[finish_ids], dim=0).item() if finish_ids else -100.0
    margin = tool_lp - fin_lp

    return captured, margin


def extract_retrieved_doc_titles(observation: str) -> list:
    """Extract document titles from BM25 search output.

    Uses Method A strict: r'\\[\\d+\\]\\s*([^:]+):'
    Matches "[N] Title: text" format from SearchTool output.
    """
    pattern = r'\[\d+\]\s*([^:]+):'
    matches = re.findall(pattern, observation)
    return [m.strip() for m in matches]


def compute_evidence_label(observation: str, sf_titles: list) -> dict:
    """Method A strict: count how many supporting fact titles were retrieved.

    Label = 0 if 0 supporting docs retrieved
    Label = 1 if 1+ supporting docs retrieved

    This is IDENTICAL to Phase 1 labeling logic.
    """
    retrieved = extract_retrieved_doc_titles(observation)
    sf_set = {t.lower().strip() for t in sf_titles}

    # Count supporting titles found in retrieved docs (case-insensitive prefix match)
    n_retrieved = 0
    matched_retrieved = []
    for rt in retrieved:
        rt_lower = rt.lower().strip()
        for sf in sf_set:
            # Allow prefix match (doc titles can be partial)
            if sf == rt_lower or sf.startswith(rt_lower) or rt_lower.startswith(sf):
                n_retrieved += 1
                matched_retrieved.append(rt)
                break

    return {
        "n_sf_retrieved": n_retrieved,
        "n_sf_total": len(sf_titles),
        "label": 0 if n_retrieved == 0 else 1,
        "retrieved_doc_titles": retrieved,
        "matched_sf_titles": matched_retrieved,
    }


def run_episode(
    model, tokenizer,
    sample,  # HotpotQASample with id, question, answer, supporting_facts
    sf_map: dict,  # sample.id -> list of supporting fact titles
    search_tool: SearchTool,
    system_prompt: str,
    fewshot_msgs: list,  # multi-turn few-shot messages (same for base & instruct)
    layer_indices: list,
    device,
    max_steps: int = 4,
    max_new_tokens: int = 256,
) -> dict:
    """Run one complete agent episode using multi-turn ChatML format.

    Conversation structure per step:
      system:    <instructions>
      *fewshot:  <balanced 4-shot examples as real turns>
      user:      <question>
      assistant: Action: search / Action Input: <query>   ← model generates
      user:      Observation: <search results>            ← system provides
      assistant: Action: search / Final Answer: ...       ← model generates (decision point!)

    Returns dict with standard fields.
    """
    question = sample.question
    gold = sample.answer if hasattr(sample, 'answer') else sample.gold_answer
    sf_titles = sf_map.get(sample.id, [])

    result = {
        "sample_id": sample.id,
        "question": question,
        "gold_answer": gold,
        "sf_titles": sf_titles,
        "steps": [],
        "n_steps": 0,
        "final_answer": None,
        "is_correct": False,
        "behavioral_continue": False,
        "behavioral_stop": True,
        "label": None,
        "n_sf_retrieved": None,
        "n_sf_total": len(sf_titles),
        "margin_step1": None,
        "step1_hidden": {},
        "parsing_error": False,
        "parse_failure_reason": None,
    }

    # Base messages: system + few-shot + user question
    base_msgs = [
        {"role": "system", "content": system_prompt},
        *fewshot_msgs,
        {"role": "user", "content": question},
    ]

    # ── STEP 0: Initial query (should search) ────────────────────────────────
    raw0 = generate_step(model, tokenizer, base_msgs, max_new_tokens)
    parsed0 = parse_action(raw0)

    step0 = {
        "step_idx": 0,
        "raw_output": raw0[:500],
        "action": parsed0["action"],
        "action_input": parsed0["action_input"],
        "final_answer": parsed0["final_answer"],
        "observation": None,
    }

    if parsed0["final_answer"]:
        result["final_answer"] = parsed0["final_answer"]
        result["steps"].append(step0)
        result["n_steps"] = 1
        result["is_correct"] = _check_correct(parsed0["final_answer"], gold)
        return result

    if not parsed0["action"] or "search" not in parsed0["action"].lower():
        result["parsing_error"] = True
        result["parse_failure_reason"] = f"step0_not_search: action={parsed0['action']}"
        result["steps"].append(step0)
        result["n_steps"] = 1
        return result

    # Execute step-0 search
    query0 = parsed0["action_input"] or question
    obs0 = search_tool(query0)
    step0["observation"] = obs0
    result["steps"].append(step0)

    # ── STEP 1: Decision point (after first search) ──────────────────────────
    # Multi-turn format: assistant said Action, user returned Observation
    msgs1 = base_msgs + [
        {"role": "assistant", "content": f"Action: search\nAction Input: {query0}"},
        {"role": "user", "content": f"Observation: {obs0}"},
    ]

    # Capture activations + margin at step-1 decision point
    try:
        hidden1, margin1 = extract_activation_and_margin(
            model, tokenizer, msgs1, layer_indices, device)
        result["step1_hidden"] = hidden1
        result["margin_step1"] = float(margin1)
    except Exception as e:
        print(f"  [WARNING] Activation capture failed: {e}")
        hidden1, margin1 = {}, 0.0

    # Compute evidence label from step-0 observation
    label_info = compute_evidence_label(obs0, sf_titles)
    result["label"] = label_info["label"]
    result["n_sf_retrieved"] = label_info["n_sf_retrieved"]
    result["margin_step1"] = float(margin1)

    # Generate step-1 action
    raw1 = generate_step(model, tokenizer, msgs1, max_new_tokens)
    parsed1 = parse_action(raw1)

    step1 = {
        "step_idx": 1,
        "raw_output": raw1[:500],
        "action": parsed1["action"],
        "action_input": parsed1["action_input"],
        "final_answer": parsed1["final_answer"],
        "observation": None,
    }

    if parsed1["final_answer"]:
        result["behavioral_stop"] = True
        result["behavioral_continue"] = False
        result["final_answer"] = parsed1["final_answer"]
        result["is_correct"] = _check_correct(parsed1["final_answer"], gold)
        result["steps"].append(step1)
        result["n_steps"] = 2
        return result

    if not parsed1["action"] or "search" not in parsed1["action"].lower():
        result["parsing_error"] = True
        result["parse_failure_reason"] = f"step1_bad_action: action={parsed1['action']}"
        result["steps"].append(step1)
        result["n_steps"] = 2
        return result

    # Model chose to continue searching
    result["behavioral_continue"] = True
    result["behavioral_stop"] = False

    query1 = parsed1["action_input"] or question
    obs1 = search_tool(query1)
    step1["observation"] = obs1
    result["steps"].append(step1)

    # ── STEP 2: Final answer (after second search) ───────────────────────────
    msgs2 = msgs1 + [
        {"role": "assistant", "content": f"Action: search\nAction Input: {query1}"},
        {"role": "user", "content": f"Observation: {obs1}"},
    ]

    raw2 = generate_step(model, tokenizer, msgs2, max_new_tokens)
    parsed2 = parse_action(raw2)

    step2 = {
        "step_idx": 2,
        "raw_output": raw2[:500],
        "action": parsed2["action"],
        "action_input": parsed2["action_input"],
        "final_answer": parsed2["final_answer"],
        "observation": None,
    }

    if parsed2["final_answer"]:
        result["final_answer"] = parsed2["final_answer"]
        result["is_correct"] = _check_correct(parsed2["final_answer"], gold)
        result["steps"].append(step2)
        result["n_steps"] = 3
        return result

    # One more search attempt if needed
    if parsed2["action"] and "search" in parsed2["action"].lower():
        query2 = parsed2["action_input"] or question
        obs2 = search_tool(query2)
        step2["observation"] = obs2
        result["steps"].append(step2)

        msgs3 = msgs2 + [
            {"role": "assistant", "content": f"Action: search\nAction Input: {query2}"},
            {"role": "user", "content": f"Observation: {obs2}"},
        ]
        raw3 = generate_step(model, tokenizer, msgs3, max_new_tokens)
        parsed3 = parse_action(raw3)
        if parsed3["final_answer"]:
            result["final_answer"] = parsed3["final_answer"]
            result["is_correct"] = _check_correct(parsed3["final_answer"], gold)
        result["n_steps"] = 4
    else:
        result["parsing_error"] = True
        result["parse_failure_reason"] = "step2_no_answer"
        result["steps"].append(step2)
        result["n_steps"] = 3

    return result


def _check_correct(pred: str, gold: str) -> bool:
    """Simple exact match / contains check (mirrors instruct baseline scoring)."""
    if not pred or not gold:
        return False
    pred_n = pred.strip().lower()
    gold_n = gold.strip().lower()
    if pred_n == gold_n:
        return True
    if gold_n in pred_n or pred_n in gold_n:
        return True
    return False


def validate_10_samples(model, tokenizer, samples, sf_map, search_tool,
                        system_prompt, fewshot_msgs, layer_indices, device,
                        out_dir: Path, n_validate: int = 10):
    """Run on first n_validate samples. Report PF rate, print traces verbatim."""
    print("\n" + "=" * 60)
    print(f"  VALIDATION: {n_validate} samples")
    print("=" * 60)

    results = []
    for i, sample in enumerate(samples[:n_validate]):
        print(f"\n[{i+1}/{n_validate}] {sample.id[:20]} Q: {sample.question[:60]}...")
        r = run_episode(model, tokenizer, sample, sf_map, search_tool,
                        system_prompt, fewshot_msgs, layer_indices, device)
        results.append(r)

        # Show trace for first 2
        if i < 2:
            print(f"  Step 0 action: {r['steps'][0].get('action')} | "
                  f"input: {str(r['steps'][0].get('action_input',''))[:40]}")
            if len(r["steps"]) > 0 and r["steps"][0].get("raw_output"):
                print(f"  Step 0 raw: {r['steps'][0]['raw_output'][:200]}")
            if r["margin_step1"] is not None:
                print(f"  Step 1 margin: {r['margin_step1']:.2f}")
            print(f"  Final answer: {r['final_answer']}")
            print(f"  Gold: {r['gold_answer']}, Correct: {r['is_correct']}")

    pf_count = sum(1 for r in results if r["parsing_error"])
    valid_step1 = sum(1 for r in results if r["margin_step1"] is not None)

    print(f"\n=== Validation Summary ===")
    print(f"  Samples: {len(results)}")
    print(f"  Parse failures: {pf_count}/{len(results)} = {pf_count/len(results):.1%}")
    print(f"  Valid step-1 captures: {valid_step1}/{len(results)}")
    print(f"  Correct: {sum(1 for r in results if r['is_correct'])}/{len(results)}")

    if pf_count / len(results) > 0.50:
        print("\n  [WARNING] PF > 50% — consider revising the 5-shot prompt")
    elif pf_count / len(results) > 0.30:
        print("\n  [WARNING] PF > 30% — acceptable but consider prompt refinement")
    else:
        print("\n  [PASS] PF within acceptable range")

    def _json_default(x):
        if hasattr(x, 'tolist'):
            return x.tolist()
        if hasattr(x, 'item'):
            return x.item()
        try:
            return float(x)
        except (TypeError, ValueError):
            return str(x)

    (out_dir / "validation_results.json").write_text(
        json.dumps(results, indent=2, default=_json_default))
    return pf_count / len(results) < 0.50  # True if OK to proceed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotpotqa-data",
                        default="data/hotpotqa/hotpot_dev_distractor_v1.json")
    parser.add_argument("--corpus-path", "--corpus",
                        default="data/hotpotqa/corpus.jsonl")
    parser.add_argument("--output-dir", default="results/phase2_rlhf_tax")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--layers", nargs="+", type=int, default=[12, 16, 20, 24])
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ids", default=None,
                        help="Path to JSON file with list of sample IDs to use "
                             "(from a previous run, e.g. results/l20_rho020_n500/test_sample_ids.json)")
    parser.add_argument("--n-validate", type=int, default=10,
                        help="Number of samples to use for validation (default: 10)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Run validation only (--n-validate samples), then exit")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip validation and run full N directly")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  PHASE 2: Qwen2.5-7B (Base) Agent — HotpotQA N=500")
    print("=" * 65)

    # Load HotpotQA
    dataset = HotpotQADataset(args.hotpotqa_data)
    if args.test_ids:
        test_ids_path = Path(args.test_ids)
        if not test_ids_path.exists():
            print(f"[WARNING] --test-ids file not found: {args.test_ids}. "
                  f"Falling back to --n-samples={args.n_samples} with seed={args.seed}.")
            samples = dataset.get_subset(args.n_samples, seed=args.seed)
        else:
            id_list = json.loads(test_ids_path.read_text())
            id_set = set(id_list)
            all_samples = dataset.get_subset(None, seed=args.seed)
            samples = [s for s in all_samples if s.id in id_set]
            # Preserve the order from id_list
            id_order = {sid: i for i, sid in enumerate(id_list)}
            samples = sorted(samples, key=lambda s: id_order.get(s.id, 9999))
            print(f"Loaded {len(samples)}/{len(id_list)} samples from --test-ids {args.test_ids}")
    else:
        samples = dataset.get_subset(args.n_samples, seed=args.seed)
        print(f"Loaded {len(samples)} bridge samples (seed={args.seed})")

    # Build supporting facts map
    sf_map = {}  # sample_id -> list of sf titles
    raw_data = json.loads(Path(args.hotpotqa_data).read_text())
    for item in raw_data:
        titles = list(set(sf[0] for sf in item.get("supporting_facts", [])))
        sf_map[item["_id"]] = titles

    # Build corpus if needed
    corpus_path = Path(args.corpus_path)
    if not corpus_path.exists():
        print(f"Building corpus at {corpus_path}...")
        build_hotpotqa_corpus(args.hotpotqa_data, str(corpus_path))

    # Load search tool
    search_tool = SearchTool(str(corpus_path), top_k=3)

    # Build system prompt + balanced few-shot examples (multi-turn)
    system_prompt = build_system_prompt()
    fewshot_msgs = build_fewshot_messages()
    prompt_path = out_dir / "base_system_prompt.txt"
    prompt_path.write_text(system_prompt)
    print(f"System prompt saved to {prompt_path}")
    print(f"  Prompt length: {len(system_prompt)} chars, {len(system_prompt.split())} words")
    print(f"  Few-shot examples: {len(FEW_SHOT_EXAMPLES)} ({len(fewshot_msgs)} messages)")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded. Device: {device}")

    # Validate chat template
    test_msgs = [{"role": "user", "content": "test"}]
    test_prompt = apply_chat_template(tokenizer, test_msgs)
    has_chatml = "<|im_start|>" in test_prompt
    print(f"Chat template: {'ChatML detected ✓' if has_chatml else 'ChatML NOT detected — using fallback'}")

    # Validate on n_validate samples
    if not args.skip_validation:
        ok = validate_10_samples(
            model, tokenizer, samples, sf_map, search_tool,
            system_prompt, fewshot_msgs, args.layers, device, out_dir,
            n_validate=args.n_validate)
        if args.validate_only:
            print("\n[--validate-only] Done.")
            return
        if not ok:
            print("\n[ABORT] Validation failed (PF > 50%). Fix prompt before full run.")
            return

    # Full N=500 run
    print(f"\n{'='*65}")
    print(f"  FULL RUN: N={len(samples)}")
    print(f"{'='*65}")

    all_hidden_per_layer = {l: [] for l in args.layers}
    all_labels, all_sample_ids = [], []
    all_margins = []

    traces_path = out_dir / "base_traces.jsonl"
    labels_path = out_dir / "base_labels.jsonl"

    n_pf, n_no_step1, n_valid = 0, 0, 0
    n_correct, n_continue = 0, 0

    with open(traces_path, "w") as tf, open(labels_path, "w") as lf:
        for i, sample in enumerate(samples):
            t0 = time.time()
            r = run_episode(
                model, tokenizer, sample, sf_map, search_tool,
                system_prompt, fewshot_msgs, args.layers, device)
            elapsed = time.time() - t0

            # Write trace (excluding large hidden arrays)
            trace = {k: v for k, v in r.items() if k != "step1_hidden"}
            # Slim down steps
            trace["steps"] = [{
                k2: v2 for k2, v2 in step.items() if k2 != "raw_output"
            } for step in r["steps"]]
            tf.write(json.dumps(trace) + "\n")
            tf.flush()

            # Write label record (same format as phase1_probe/labels.jsonl)
            if r["margin_step1"] is not None and r["label"] is not None:
                lrec = {
                    "sample_id": r["sample_id"],
                    "question": r["question"],
                    "gold_answer": r["gold_answer"],
                    "is_correct": r["is_correct"],
                    "label": r["label"],
                    "n_sf_retrieved": r["n_sf_retrieved"],
                    "n_sf_total": r["n_sf_total"],
                    "sf_titles": r["sf_titles"],
                    "retrieved_doc_titles": (r["steps"][0].get("observation", "")
                                             if r["steps"] else ""),
                    "behavioral_continue": r["behavioral_continue"],
                    "behavioral_stop": r["behavioral_stop"],
                    "margin_before": r["margin_step1"],
                }
                lf.write(json.dumps(lrec) + "\n")
                lf.flush()

                # Accumulate activations
                if all(l in r["step1_hidden"] for l in args.layers):
                    for l in args.layers:
                        all_hidden_per_layer[l].append(r["step1_hidden"][l])
                    all_labels.append(r["label"])
                    all_sample_ids.append(r["sample_id"])
                    all_margins.append(r["margin_step1"])
                    n_valid += 1

            if r["parsing_error"]:
                n_pf += 1
            if r["margin_step1"] is None:
                n_no_step1 += 1
            if r["is_correct"]:
                n_correct += 1
            if r["behavioral_continue"]:
                n_continue += 1

            if (i + 1) % 25 == 0 or i < 3:
                print(f"  [{i+1:3d}/{len(samples)}] "
                      f"valid={n_valid} pf={n_pf} correct={n_correct} "
                      f"continue={n_continue} | {elapsed:.1f}s")

    # Save activations
    if n_valid > 0:
        y = np.array(all_labels, dtype=np.int32)
        margins = np.array(all_margins, dtype=np.float32)
        save_dict = {
            "y": y,
            "sample_ids": np.array(all_sample_ids),
            "margins": margins,
            "model": np.array(args.model),
            "n_samples": np.array(n_valid),
        }
        for l in args.layers:
            if all_hidden_per_layer[l]:
                save_dict[f"layer_{l}"] = np.array(all_hidden_per_layer[l], dtype=np.float32)

        npz_path = out_dir / "base_activations.npz"
        np.savez(str(npz_path), **save_dict)
        print(f"\nSaved activations: {npz_path} (N={n_valid})")

    # Print summary
    print(f"\n{'='*65}")
    print(f"  PHASE 2 BASE MODEL SUMMARY")
    print(f"{'='*65}")
    print(f"  Total samples:        {len(samples)}")
    print(f"  Valid step-1 captures:{n_valid}  ({n_valid/len(samples):.1%})")
    print(f"  Parse failures:       {n_pf}  ({n_pf/len(samples):.1%})")
    print(f"  Correct (EM):         {n_correct}  ({n_correct/len(samples):.1%})")
    print(f"  2nd search rate:      {n_continue}  ({n_continue/len(samples):.1%})")

    if n_valid > 0:
        y_arr = np.array(all_labels)
        n0 = int((y_arr == 0).sum())
        n1 = int((y_arr == 1).sum())
        print(f"  Label=0 (insufficient):{n0}  ({n0/n_valid:.1%})")
        print(f"  Label=1 (sufficient):  {n1}  ({n1/n_valid:.1%})")
        print(f"  Mean step-1 margin:   {np.mean(all_margins):.2f} ± {np.std(all_margins):.2f}")

    # Save summary JSON
    summary = {
        "model": args.model,
        "n_total": len(samples),
        "n_valid": n_valid,
        "n_parse_failures": n_pf,
        "n_correct": n_correct,
        "n_continue": n_continue,
        "accuracy": n_correct / len(samples),
        "pf_rate": n_pf / len(samples),
        "second_search_rate": n_continue / len(samples),
        "label_distribution": {
            "n_label0": int((np.array(all_labels) == 0).sum()) if all_labels else 0,
            "n_label1": int((np.array(all_labels) == 1).sum()) if all_labels else 0,
        } if all_labels else {},
        "margin_stats": {
            "mean": float(np.mean(all_margins)),
            "std": float(np.std(all_margins)),
            "min": float(np.min(all_margins)),
            "max": float(np.max(all_margins)),
        } if all_margins else {},
    }
    (out_dir / "base_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
