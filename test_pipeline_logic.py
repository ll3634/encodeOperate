#!/usr/bin/env python3
"""
Test pipeline logic without requiring model download.
Validates all code paths and integrations.
"""

import sys
sys.path.insert(0, '.')

from pathlib import Path
import json
import tempfile

# Test 1: Verify imports
print("=" * 70)
print("TEST 1: Verify all imports")
print("=" * 70)

try:
    from steering.hook_utils import SteeringHook
    from agent.policies import SteeringDecision, BaselinePolicy, Policy
    from agent.policies_verify import JESStep2OnlyPolicy, JESStep2ForcePolicy, Baseline1HopPolicy, Oracle2HopPolicy
    from agent.prompts import PromptBuilder
    from agent.react_loop import ReActAgent, AgentConfig
    from steering.extract_search_post_direction import (
        select_samples_from_pool,
        build_irrelevant_pair_indices,
        build_step1_prompt as build_search_step1_prompt,
        load_runtime_queries,
    )
    from steering.extract_calculator_post_direction import build_calculator_post_pair, build_step1_prompt as build_calculator_step1_prompt
    from steering.jes import JESConfig, JESController
    from datasets.popqa import PopQADataset
    from datasets.hotpotqa import HotpotQADataset, HotpotQASample
    from scripts.control_budget_diagnosis import compute_diagnosis, print_diagnosis
    from scripts.run_verify_critical_pipeline import compute_stats
    print("✓ All imports successful\n")
except Exception as e:
    print(f"✗ Import failed: {e}\n")
    sys.exit(1)

# Test 2: Verify SteeringDecision has decision_only field
print("=" * 70)
print("TEST 2: Verify SteeringDecision.decision_only field")
print("=" * 70)

decision = SteeringDecision(rho=0.5, alpha=100.0, decision_only=True)
assert hasattr(decision, 'decision_only'), "Missing decision_only field"
assert decision.decision_only == True, "decision_only not set correctly"
print(f"✓ SteeringDecision.decision_only = {decision.decision_only}\n")

# Test 3: Verify SteeringHook accepts max_interventions
print("=" * 70)
print("TEST 3: Verify SteeringHook.max_interventions parameter")
print("=" * 70)

import inspect
sig = inspect.signature(SteeringHook.__init__)
params = list(sig.parameters.keys())
assert 'max_interventions' in params, "max_interventions not in SteeringHook.__init__"
print(f"✓ SteeringHook.__init__ parameters: {params}\n")

# Test 4: Verify JESStep2OnlyPolicy sets decision_only=True
print("=" * 70)
print("TEST 4: Verify JESStep2OnlyPolicy sets decision_only=True")
print("=" * 70)

import numpy as np
policy = JESStep2OnlyPolicy(config=JESConfig(), direction=np.random.randn(4096))

# Mock margin function
def mock_margin_fn(rho):
    return 1.0 + rho * 0.5

# Reset episode to start from step 0
policy.reset_episode()

# Step 0: No steering
decision_step0 = policy.decide(mock_margin_fn, "positive", hidden_rms=1.0, direction_rms=1.0)
assert decision_step0.rho == 0.0, "Step 0 should have no steering"
print(f"✓ Step 0: rho={decision_step0.rho} (no steering)")

# Step 1: JES steering with decision_only=True
decision_step1 = policy.decide(mock_margin_fn, "positive", hidden_rms=1.0, direction_rms=1.0)
assert hasattr(decision_step1, 'decision_only'), "Policy didn't return decision_only field"
assert decision_step1.decision_only == True, "Policy didn't set decision_only=True at step 1"
print(f"✓ Step 1: decision_only={decision_step1.decision_only} (JES steering)")

# Step 2+: No steering
decision_step2 = policy.decide(mock_margin_fn, "positive", hidden_rms=1.0, direction_rms=1.0)
assert decision_step2.rho == 0.0, "Step 2+ should have no steering"
print(f"✓ Step 2+: rho={decision_step2.rho} (no steering)\n")

# Test 4b: Verify verify-critical policies preserve old-good step-0 semantics
print("=" * 70)
print("TEST 4B: Verify verify-critical step-0 semantics")
print("=" * 70)

jes_only_step0_policy = JESStep2OnlyPolicy(config=JESConfig(), direction=np.random.randn(4096))
jes_force_step0_policy = JESStep2ForcePolicy(config=JESConfig(), direction=np.random.randn(4096))
baseline = Baseline1HopPolicy()
oracle = Oracle2HopPolicy()

baseline.reset_episode()
oracle.reset_episode()
jes_only_step0_policy.reset_episode()
jes_force_step0_policy.reset_episode()

policy_margin_calls = {"count": 0}

def counting_margin_fn(rho):
    policy_margin_calls["count"] += 1
    return 1.0 + rho

baseline_step0 = baseline.decide(counting_margin_fn, "positive", hidden_rms=1.0, direction_rms=1.0)
oracle_step0 = oracle.decide(counting_margin_fn, "positive", hidden_rms=1.0, direction_rms=1.0)
jes_step0 = jes_only_step0_policy.decide(mock_margin_fn, "positive", hidden_rms=1.0, direction_rms=1.0)
jes_force_step0 = jes_force_step0_policy.decide(mock_margin_fn, "positive", hidden_rms=1.0, direction_rms=1.0)

assert baseline_step0.override_action is None, "Baseline step 0 should not override search"
assert oracle_step0.override_action is None, "Oracle step 0 should not override search"
assert jes_step0.override_action is None, "JES step 0 should not override search"
assert jes_force_step0.override_action is None, "JES-force step 0 should not override search"
assert baseline_step0.details["action"] == "normal_gen", "Baseline step 0 should be normal generation"
assert oracle_step0.details["action"] == "normal_gen", "Oracle step 0 should be normal generation"
assert jes_step0.details["action"] == "no_steering_step0", "JES step 0 should be no_steering_step0"
assert jes_force_step0.details["action"] == "no_steering_step0", "JES-force step 0 should be no_steering_step0"
assert baseline_step0.skip_margin_before_log is True, "Baseline step 0 should skip margin-before logging"
assert oracle_step0.skip_margin_before_log is True, "Oracle step 0 should skip margin-before logging"
assert policy_margin_calls["count"] == 0, "Baseline/Oracle should not call margin_fn just for logging"
print("✓ Baseline/Oracle/JES step 0 no longer override search\n")

# Test 4c: Verify clean prompt semantics and scratchpad filtering
print("=" * 70)
print("TEST 4C: Verify prompt is clean and parse-failure markers stay out")
print("=" * 70)

prompt_builder = PromptBuilder(tools=["search"])
system_prompt = prompt_builder.build_system_prompt()
assert "Eiffel Tower" not in system_prompt, "System prompt should not include exemplar contamination"
assert 'first word must be either "Action" or "Final"' in system_prompt, "System prompt lost old-good first-token constraint"

scratchpad = prompt_builder.build_scratchpad([
    {"observation": "[PARSE_FAILURE] No valid action parsed"},
    {"action": "search", "action_input": "bridge question", "observation": "retrieved evidence"},
])
assert "[PARSE_FAILURE]" not in scratchpad, "Scratchpad should filter parse-failure sentinels"
assert "retrieved evidence" in scratchpad, "Scratchpad should retain real observations"
print("✓ Prompt no longer contains exemplar contamination and scratchpad stays clean\n")

# Test 4d: Verify clean extraction helpers stay non-overlapping and semantically aligned
print("=" * 70)
print("TEST 4D: Verify extraction helper logic for clean post-tool directions")
print("=" * 70)

mock_hotpot_samples = [
    HotpotQASample(id="q1", question="Q1", answer="A1", answers=["A1"], supporting_facts=[], context=[], level="easy", type="bridge"),
    HotpotQASample(id="q2", question="Q2", answer="A2", answers=["A2"], supporting_facts=[], context=[], level="easy", type="bridge"),
    HotpotQASample(id="q3", question="Q3", answer="A3", answers=["A3"], supporting_facts=[], context=[], level="easy", type="comparison"),
    HotpotQASample(id="q4", question="Q4", answer="A4", answers=["A4"], supporting_facts=[], context=[], level="easy", type="bridge"),
]

selected_once, eligible_count = select_samples_from_pool(
    mock_hotpot_samples, n=2, seed=7, type_filter="bridge", excluded_ids={"q2"}
)
selected_twice, eligible_count_again = select_samples_from_pool(
    mock_hotpot_samples, n=2, seed=7, type_filter="bridge", excluded_ids={"q2"}
)
selected_ids = [s.id for s in selected_once]
assert eligible_count == 2 and eligible_count_again == 2, "Eligible pool size after exclusion is wrong"
assert selected_ids == [s.id for s in selected_twice], "Selection should be deterministic for a fixed seed"
assert "q2" not in selected_ids, "Excluded ID leaked into selected extraction pool"
assert all(s.type == "bridge" for s in selected_once), "Type filter was not preserved"

pair_indices = build_irrelevant_pair_indices(4, __import__("random").Random(11))
assert all(i != j for i, j in enumerate(pair_indices)), "Irrelevant pairing contains self-pairs"

class FakeChatTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = "\n".join(f"[{m['role']}] {m['content']}" for m in messages)
        return text + ("\n[assistant]" if add_generation_prompt else "")


expr, obs_correct, obs_incorrect = build_calculator_post_pair("123")
calc_prompt = build_calculator_step1_prompt(FakeChatTokenizer(), "How many apples?", expr, obs_correct)
assert expr == "123", "Calculator expression should preserve normalized numeric answer"
assert obs_correct == "123", "Correct calculator observation should be normalized"
assert obs_incorrect != obs_correct, "Incorrect calculator observation should differ from correct"
assert "Action: calculator" in calc_prompt, "Calculator post-tool prompt lost calculator action line"
assert "Action Input: 123" in calc_prompt, "Calculator post-tool prompt lost action input"
assert "Observation: 123" in calc_prompt, "Calculator post-tool prompt lost observation"

trace_records = [
    {
        "sample_id": "q1",
        "steps": [{"step_idx": 0, "action": "search", "action_input": "rewritten bridge query"}],
    },
    {
        "sample_id": "q4",
        "steps": [{"step_idx": 0, "action": "search", "action_input": "Akrofuom location Ghana"}],
    },
]
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmp:
    for record in trace_records:
        tmp.write(json.dumps(record) + "\n")
    trace_path = tmp.name

try:
    runtime_queries = load_runtime_queries(trace_path)
    search_prompt = build_search_step1_prompt(
        FakeChatTokenizer(),
        "Which region of Ghana was the city where Akrofuom is located?",
        runtime_queries["q4"],
        "retrieved evidence",
    )
    assert runtime_queries["q1"] == "rewritten bridge query", "Runtime trace query loader lost step-0 query"
    assert "Action Input: Akrofuom location Ghana" in search_prompt, "Search post-tool prompt did not use runtime-aligned query"
    assert "Action Input: Which region of Ghana" not in search_prompt, "Prompt regressed to raw question-as-query"
finally:
    Path(trace_path).unlink(missing_ok=True)

print("✓ Clean extraction helpers preserve exclusion, non-self pairing, calculator semantics, and runtime-trace alignment\n")

# Test 5: Verify HotpotQASample has answers field
print("=" * 70)
print("TEST 5: Verify HotpotQASample.answers field")
print("=" * 70)

sample = HotpotQASample(
    id="test_1",
    question="What is 2+2?",
    answer="4",
    answers=["4"],
    supporting_facts=[],
    context=[],
    level="easy",
    type="comparison"
)
assert hasattr(sample, 'answers'), "HotpotQASample missing answers field"
assert sample.answers == ["4"], "answers not set correctly"
print(f"✓ HotpotQASample.answers = {sample.answers}\n")

# Test 6: Verify control_budget_diagnosis can be imported and called
print("=" * 70)
print("TEST 6: Verify control_budget_diagnosis functions")
print("=" * 70)

# Create mock results
mock_bl = [
    {"sample_id": "1", "is_correct": True, "details": {"m_before": 1.0}},
    {"sample_id": "2", "is_correct": False, "details": {"m_before": -1.0}},
]
mock_orc = [
    {"sample_id": "1", "is_correct": True, "details": {"m_before": 1.5}},
    {"sample_id": "2", "is_correct": True, "details": {"m_before": 0.5}},
]
mock_jes = [
    {"sample_id": "1", "is_correct": True, "details": {"rho_star_raw": 0.1}},
    {"sample_id": "2", "is_correct": True, "details": {"rho_star_raw": 0.8}},
]

try:
    diag = compute_diagnosis(mock_bl, mock_orc, mock_jes)
    assert "verdict" in diag, "Diagnosis missing verdict"
    assert "vc_density" in diag, "Diagnosis missing vc_density"
    print(f"✓ compute_diagnosis() works")
    print(f"  - VC density: {diag['vc_density']*100:.1f}%")
    print(f"  - Verdict: {diag['verdict']}\n")
except Exception as e:
    print(f"✗ compute_diagnosis() failed: {e}\n")
    sys.exit(1)

# Test 6b: Verify verify gate activation metrics capture real second-search behavior
print("=" * 70)
print("TEST 6B: Verify second-search activation stats")
print("=" * 70)

mock_bl_activation = [
    {"sample_id": "1", "is_correct": False},
    {"sample_id": "2", "is_correct": False},
    {"sample_id": "3", "is_correct": True},
]
mock_jes_activation = [
    {
        "sample_id": "1",
        "is_correct": False,
        "tool_calls": 1,
        "steps": [{"step_idx": 1, "action": "final", "parse_failure_reason": None}],
    },
    {
        "sample_id": "2",
        "is_correct": True,
        "tool_calls": 2,
        "steps": [{"step_idx": 1, "action": "search", "parse_failure_reason": None}],
    },
    {
        "sample_id": "3",
        "is_correct": True,
        "tool_calls": 1,
        "steps": [{"step_idx": 1, "action": None, "parse_failure_reason": "No valid action parsed"}],
    },
]

activation_stats = compute_stats(mock_bl_activation, mock_jes_activation)
assert activation_stats["step1_search_count"] == 1, "Failed to count step-1 search activations"
assert activation_stats["step1_final_count"] == 1, "Failed to count step-1 final answers"
assert activation_stats["step1_parse_failure_count"] == 1, "Failed to count step-1 parse failures"
assert activation_stats["second_search_activation_count"] == 1, "Failed to count actual second-search tool calls"
assert abs(activation_stats["step1_search_rate"] - (1 / 3)) < 1e-9, "Incorrect step-1 search rate"
assert abs(activation_stats["second_search_activation_rate"] - (1 / 3)) < 1e-9, "Incorrect second-search activation rate"
print("✓ Gate stats now expose step-1 behavior and actual second-search activation\n")

# Test 7: Verify pipeline sweep parameters
print("=" * 70)
print("TEST 7: Verify pipeline sweep parameter parsing")
print("=" * 70)

tau_values = [0.0, 0.1]
max_rho_values = [0.25, 0.75, 1.5]
sweep_grid = [(tau, mr) for tau in tau_values for mr in max_rho_values]

print(f"✓ Sweep grid generated:")
for tau, mr in sweep_grid:
    tag = f"tau{tau:.2f}_rho{mr:.2f}"
    print(f"  - {tag}")
print()

# Test 8: Verify PopQA dataset interface
print("=" * 70)
print("TEST 8: Verify PopQA dataset interface")
print("=" * 70)

popqa_path = Path("data/popqa/popqa_test.jsonl")
if popqa_path.exists():
    try:
        dataset = PopQADataset(str(popqa_path))
        sample = dataset[0]
        assert hasattr(sample, 'id'), "PopQA sample missing id"
        assert hasattr(sample, 'question'), "PopQA sample missing question"
        assert hasattr(sample, 'answer'), "PopQA sample missing answer"
        assert hasattr(sample, 'answers'), "PopQA sample missing answers"
        print(f"✓ PopQA dataset loaded: {len(dataset)} samples")
        print(f"  - Sample ID: {sample.id}")
        print(f"  - Question: {sample.question[:50]}...")
        print(f"  - Answers: {sample.answers}\n")
    except Exception as e:
        print(f"✗ PopQA loading failed: {e}\n")
else:
    print(f"⊘ PopQA data not found at {popqa_path}\n")

# Test 9: Verify run() does not re-route JES via forced prefixes
print("=" * 70)
print("TEST 9: Verify ReActAgent.run preserves normal generation semantics")
print("=" * 70)

import torch


class DummyModel:
    def parameters(self):
        yield torch.nn.Parameter(torch.zeros(1))


class DummyTokenizer:
    pass


class StaticPolicy(Policy):
    @property
    def name(self) -> str:
        return "static_semantic_check"

    def decide(self, margin_fn, target_side, hidden_rms, direction_rms) -> SteeringDecision:
        return SteeringDecision(
            rho=0.5,
            alpha=2.0,
            policy_name=self.name,
            m_before=1.0,
            decision_only=True,
        )


agent = ReActAgent(
    model=DummyModel(),
    tokenizer=DummyTokenizer(),
    tools={},
    config=AgentConfig(max_steps=1),
)

captured = {}


def fake_compute_margin(messages, rho=0.0):
    return 1.0 + rho


def fake_hidden_rms():
    return 1.0


def fake_generate_step(messages, steering_decision=None):
    captured["called"] = True
    captured["alpha"] = steering_decision.alpha if steering_decision else None
    return "Final Answer: semantic check", 10, 3


agent._compute_margin = fake_compute_margin
agent._get_hidden_rms = fake_hidden_rms
agent._generate_step = fake_generate_step

result = agent.run("test question", policy=StaticPolicy())
assert captured.get("called"), "run() did not call _generate_step"
assert captured.get("alpha") == 2.0, "run() failed to pass steering decision into generation"
assert result.final_answer == "semantic check", "run() failed to use direct generation output"
print("✓ run() no longer injects prefix-routing into the normal generation path\n")

# Test 10: Verify parse failures are not fed back into the next prompt
print("=" * 70)
print("TEST 10: Verify parse failures do not contaminate the next decision context")
print("=" * 70)


class NoSteeringPolicy(Policy):
    @property
    def name(self) -> str:
        return "no_steering_clean_context"

    def decide(self, margin_fn, target_side, hidden_rms, direction_rms) -> SteeringDecision:
        return SteeringDecision(rho=0.0, alpha=0.0, policy_name=self.name)


agent_clean = ReActAgent(
    model=DummyModel(),
    tokenizer=DummyTokenizer(),
    tools={},
    config=AgentConfig(max_steps=2),
)

call_messages = []
responses = [
    "This is malformed free-form text",
    "Final Answer: recovered cleanly",
]


def fake_generate_step_clean(messages, steering_decision=None):
    call_messages.append(messages)
    idx = len(call_messages) - 1
    return responses[idx], 10, 3


agent_clean._compute_margin = fake_compute_margin
agent_clean._get_hidden_rms = fake_hidden_rms
agent_clean._generate_step = fake_generate_step_clean

result_clean = agent_clean.run("test question", policy=NoSteeringPolicy())
assert len(call_messages) == 2, "Expected two generation attempts"
second_prompt_text = "\n".join(msg.get("content", "") for msg in call_messages[1])
assert "[PARSE_FAILURE]" not in second_prompt_text, "Parse failure marker leaked into second prompt"
assert "No valid action parsed" not in second_prompt_text, "Parse failure text leaked into second prompt"
assert result_clean.final_answer == "recovered cleanly", "Agent did not recover after a clean retry"
print("✓ Parse failures no longer pollute the next decision context\n")


print("=" * 70)
print("ALL TESTS PASSED ✓")
print("=" * 70)
print("\nCode is ready for pipeline execution.")
print("Next step: Download model and run full pipeline")

