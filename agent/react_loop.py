#!/usr/bin/env python3
"""
Minimal ReAct Agent Loop.
No external frameworks (LangChain, AutoGen, etc.) - just a simple while loop.
"""

import time
import torch
import numpy as np
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field

from .prompts import PromptBuilder, parse_action, ACTION_TOKENS
from .policies import Policy, BaselinePolicy, SteeringDecision

import sys
sys.path.insert(0, '..')
from steering.hook_utils import (
    SteeringHook,
    KVGroupScalingHook,
    KVGroupDirectionalScalingHook,
    get_model_layers,
    compute_rms,
)


@dataclass
class AgentConfig:
    """Configuration for ReAct agent."""
    max_steps: int = 10
    max_tokens_per_step: int = 256
    temperature: float = 0.0
    layer: int = 12
    position: int = -1
    tools: List[str] = field(default_factory=lambda: ["search", "calculator"])
    # Scoring mode used when evaluating final_answer against gold_answer.
    # "any"     – legacy (tries exact/contains/fuzzy/math_equiv in order; prone to false positives)
    # "numeric" – strict last-number extraction + numeric comparison (recommended for GSM8K/MATH)
    score_mode: str = "any"


@dataclass
class StepResult:
    """Result of a single agent step."""
    step_idx: int
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None
    final_answer: Optional[str] = None
    margin_before: Optional[float] = None
    margin_after: Optional[float] = None
    steering: Optional[Dict] = None
    tokens_prompt: int = 0
    tokens_completion: int = 0
    wall_time_ms: float = 0.0
    corruption_applied: bool = False
    raw_model_text: Optional[str] = None  # Raw model output (truncated)
    parse_failure_reason: Optional[str] = None  # Reason for parsing failure

    def to_dict(self) -> dict:
        return {
            "step_idx": self.step_idx,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "final_answer": self.final_answer,
            "margin_before": self.margin_before,
            "margin_after": self.margin_after,
            "steering": self.steering,
            "tokens_prompt": self.tokens_prompt,
            "tokens_completion": self.tokens_completion,
            "wall_time_ms": self.wall_time_ms,
            "corruption_applied": self.corruption_applied,
            "raw_model_text": self.raw_model_text,
            "parse_failure_reason": self.parse_failure_reason,
        }


@dataclass
class EpisodeResult:
    """Result of a complete agent episode."""
    id: str
    question: str
    success: bool
    final_answer: Optional[str] = None
    gold_answer: Optional[str] = None
    policy: str = "baseline"
    steps: List[StepResult] = field(default_factory=list)
    failure_reason: Optional[str] = None  # decision_error, execution_error, parsing_error, timeout
    total_tool_calls: int = 0
    total_tokens: int = 0
    total_wall_time_ms: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "success": self.success,
            "final_answer": self.final_answer,
            "gold_answer": self.gold_answer,
            "policy": self.policy,
            "steps": [s.to_dict() for s in self.steps],
            "failure_reason": self.failure_reason,
            "totals": {
                "tool_calls": self.total_tool_calls,
                "total_tokens": self.total_tokens,
                "total_wall_time_ms": self.total_wall_time_ms,
            }
        }


class ReActAgent:
    """
    Minimal ReAct agent with hidden-state steering.
    
    Usage:
        agent = ReActAgent(model, tokenizer, tools, config)
        result = agent.run(question, policy=JESPolicy(...))
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        tools: Dict[str, Callable],
        config: AgentConfig = None,
        direction: np.ndarray = None,
        direction_rms: float = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools
        self.config = config or AgentConfig()
        self.direction = direction
        self.direction_rms = direction_rms
        self.prompt_builder = PromptBuilder(tools=list(tools.keys()))
        
        # Compute hidden RMS from model (cached)
        self._hidden_rms = None
        
    @property
    def device(self):
        return next(self.model.parameters()).device
    
    def _get_hidden_rms(self) -> float:
        """Get or compute hidden state RMS via calibration forward pass."""
        if self._hidden_rms is None:
            self._hidden_rms = self._calibrate_hidden_rms()
        return self._hidden_rms

    def _calibrate_hidden_rms(self) -> float:
        """
        Run a short calibration forward pass to measure hidden RMS at target layer.

        Matches the original causal_intervention.py methodology:
        - Compute RMS at the LAST token position only (not averaged over all positions)
        - Early positions (especially attention sinks) have ~100x higher RMS and
          would catastrophically inflate the mean
        """
        calibration_text = "The quick brown fox jumps over the lazy dog."
        messages = [{"role": "user", "content": calibration_text}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        layers = get_model_layers(self.model)
        num_layers = len(layers)
        actual_layer = self.config.layer if self.config.layer >= 0 else num_layers + self.config.layer

        captured = {}

        def capture_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            captured['hidden'] = hidden.detach()
            return output  # Don't modify

        handle = layers[actual_layer].register_forward_hook(capture_hook)
        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            handle.remove()

        hidden = captured['hidden']  # [batch, seq, hidden_dim]
        # RMS at LAST position only — matches original causal_intervention.py approach
        # (outputs.hidden_states[layer][0, -1, :])
        h_last = hidden[0, -1, :].float()  # [hidden_dim]
        rms = float(h_last.pow(2).mean().sqrt().item())
        print(f"  [Calibration] hidden_rms at layer {self.config.layer} = {rms:.4f}")
        return rms
    
    def _compute_margin(
        self,
        messages: List[Dict],
        rho: float = 0.0
    ) -> float:
        """
        Compute action margin: log P(tool_call) - log P(finish).
        
        Positive margin = prefer tool call
        Negative margin = prefer finish
        """
        # Build prompt
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Get alpha from rho.
        # NOTE: alpha_max clamping is the sole responsibility of JESController.rho_to_alpha().
        # react_loop must NOT re-clamp; doing so would require a second alpha_max definition
        # which creates two competing sources of truth.
        alpha = 0.0
        if rho != 0.0 and self.direction is not None:
            alpha = rho * (self._get_hidden_rms() / self.direction_rms)
        
        # Forward pass with optional steering
        if alpha != 0.0 and self.direction is not None:
            with SteeringHook(
                self.model, self.direction, alpha,
                layer=self.config.layer, position=self.config.position
            ):
                with torch.no_grad():
                    outputs = self.model(input_ids)
        else:
            with torch.no_grad():
                outputs = self.model(input_ids)
        
        # Get logits for next token
        logits = outputs.logits[0, -1, :]  # [vocab_size]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Get token IDs for action types
        tool_tokens = []
        finish_tokens = []
        for token_str in ACTION_TOKENS["tool_call"]:
            ids = self.tokenizer.encode(token_str, add_special_tokens=False)
            if ids:
                tool_tokens.append(ids[0])
        for token_str in ACTION_TOKENS["finish"]:
            ids = self.tokenizer.encode(token_str, add_special_tokens=False)
            if ids:
                finish_tokens.append(ids[0])

        # Compute margin as logsumexp difference
        if tool_tokens:
            tool_logprob = torch.logsumexp(log_probs[tool_tokens], dim=0).item()
        else:
            tool_logprob = -100.0

        if finish_tokens:
            finish_logprob = torch.logsumexp(log_probs[finish_tokens], dim=0).item()
        else:
            finish_logprob = -100.0

        margin = tool_logprob - finish_logprob
        return margin

    def _generate_step(
        self,
        messages: List[Dict],
        steering_decision: SteeringDecision = None,
    ) -> tuple:
        """Generate next action with optional steering."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = input_ids.shape[1]
        # Fix transformers warning: explicitly pass attention_mask
        attention_mask = torch.ones_like(input_ids)

        # alpha is already clamped by JESController.rho_to_alpha() (JESConfig.alpha_max).
        # Do NOT re-clamp here — that would require a second alpha_max definition.
        alpha = steering_decision.alpha if steering_decision else 0.0

        # Build generation kwargs - only include supported params
        gen_kwargs = {
            "max_new_tokens": self.config.max_tokens_per_step,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "attention_mask": attention_mask,
        }
        if self.config.temperature > 0:
            gen_kwargs["temperature"] = self.config.temperature
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        # Decision-only steering: limit hook to first forward pass only
        max_interventions = None
        if steering_decision and getattr(steering_decision, 'decision_only', False):
            max_interventions = 1

        # Check for KV-group scaling (circuit-level intervention)
        kv_cfg = getattr(steering_decision, 'kv_group_config', None) if steering_decision else None

        # Check for cross-axis (rank-1 dynamic) intervention (Arm B reconnection sweep)
        cross_cfg = getattr(steering_decision, 'cross_axis_config', None) if steering_decision else None

        # Generate with optional steering and/or KV-group scaling
        use_steering = (alpha != 0.0 and self.direction is not None)
        use_kv = (kv_cfg is not None)
        use_cross = (cross_cfg is not None)

        # Allow policy to override steering layer (e.g., L19 for mlp_L20 input)
        steer_layer = self.config.layer
        if steering_decision and getattr(steering_decision, 'steer_layer', None) is not None:
            steer_layer = steering_decision.steer_layer

        if use_steering and use_kv:
            # Combined: steering + KV ablation (necessity test)
            with SteeringHook(
                self.model, self.direction, alpha,
                layer=steer_layer, position=self.config.position,
                max_interventions=max_interventions,
            ):
                with KVGroupScalingHook(
                    self.model,
                    layer=kv_cfg["layer"],
                    kv_group=kv_cfg["kv_group"],
                    alpha=kv_cfg["alpha"],
                    max_interventions=max_interventions,
                ):
                    with torch.no_grad():
                        outputs = self.model.generate(input_ids, **gen_kwargs)
        elif use_kv:
            # Circuit-level intervention: scale specific KV group in attention
            kv_mode = kv_cfg.get("mode", "uniform")
            if kv_mode == "directional":
                with KVGroupDirectionalScalingHook(
                    self.model,
                    layer=kv_cfg["layer"],
                    kv_group=kv_cfg["kv_group"],
                    direction=kv_cfg["direction"],
                    alpha_parallel=kv_cfg.get("alpha_parallel", 1.0),
                    alpha_orth=kv_cfg.get("alpha_orth", 1.0),
                    max_interventions=max_interventions,
                ):
                    with torch.no_grad():
                        outputs = self.model.generate(input_ids, **gen_kwargs)
            else:
                with KVGroupScalingHook(
                    self.model,
                    layer=kv_cfg["layer"],
                    kv_group=kv_cfg["kv_group"],
                    alpha=kv_cfg["alpha"],
                    max_interventions=max_interventions,
                ):
                    with torch.no_grad():
                        outputs = self.model.generate(input_ids, **gen_kwargs)
        elif use_cross:
            from steering.hook_utils import CrossAxisHook
            with CrossAxisHook(
                self.model,
                u_in=cross_cfg["u_in"],
                u_out=cross_cfg["u_out"],
                alpha=cross_cfg["alpha"],
                layer=cross_cfg.get("layer", steer_layer),
                position=self.config.position,
                max_interventions=max_interventions,
            ):
                with torch.no_grad():
                    outputs = self.model.generate(input_ids, **gen_kwargs)
        elif use_steering:
            with SteeringHook(
                self.model, self.direction, alpha,
                layer=steer_layer, position=self.config.position,
                max_interventions=max_interventions,
            ):
                with torch.no_grad():
                    outputs = self.model.generate(input_ids, **gen_kwargs)
        else:
            with torch.no_grad():
                outputs = self.model.generate(input_ids, **gen_kwargs)

        completion_ids = outputs[0, prompt_len:]
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

        return completion_text, prompt_len, len(completion_ids)

    def _find_thought_boundary(self, gen_ids: list) -> list:
        """
        Given a list of generated token IDs, find the 'thought' tokens —
        everything generated before the first \\nAction or \\nFinal Answer line.

        Returns list of thought token IDs, or [] if no boundary is found
        (model jumped directly to Action/Final without a thought).
        """
        accumulated = ""
        for tok_idx, tid in enumerate(gen_ids):
            tok_text = self.tokenizer.decode([tid], skip_special_tokens=True)
            accumulated += tok_text
            action_pos = accumulated.find("\nAction")
            final_pos = accumulated.find("\nFinal")
            if action_pos >= 0 or final_pos >= 0:
                cut_char = min(
                    action_pos if action_pos >= 0 else len(accumulated),
                    final_pos if final_pos >= 0 else len(accumulated),
                )
                # Walk back to find which token introduced the boundary
                prefix_len = 0
                for j, t in enumerate(gen_ids[:tok_idx + 1]):
                    prefix_len += len(self.tokenizer.decode([t], skip_special_tokens=True))
                    if prefix_len > cut_char:
                        return gen_ids[:j]
                return gen_ids[:tok_idx]
        return []

    def _generate_step_timed(
        self,
        messages: List[Dict],
        steering_decision: SteeringDecision,
        timing: str,
    ) -> tuple:
        """
        Two-pass generation for intervention timing experiments.

        Pass 1: generate the full step output WITHOUT steering to capture the
                natural thought tokens.
        Pass 2: build input_ids_2 = [original_prompt | thought_prefix] and
                inject steering at position=-1 (last thought token), then let
                the model generate the action/answer from there.

        timing="p4" — inject at the last thought token (full thought prefix).
        timing="p2" — inject at 50% through the thought.

        Falls back to _generate_step() if no thought is produced (< 4 tokens).
        """
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = input_ids.shape[1]

        gen_kwargs = {
            "max_new_tokens": self.config.max_tokens_per_step,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "attention_mask": torch.ones_like(input_ids),
            "do_sample": False,
        }

        # Pass 1: free generation (no steering) to get the thought
        with torch.no_grad():
            outputs1 = self.model.generate(input_ids, **gen_kwargs)
        gen_ids = outputs1[0, prompt_len:].tolist()

        thought_ids = self._find_thought_boundary(gen_ids)

        if len(thought_ids) >= 4:
            # Model produced a thought — use thought tokens as prefix space
            prefix_space = thought_ids
        elif len(gen_ids) >= 4:
            # No thought boundary (e.g. DEFAULT_SYSTEM_PROMPT) — use full
            # generation as prefix space so p2/p4 inject post-decision rather
            # than collapsing to p0.
            prefix_space = gen_ids
        else:
            # Generation too short for any meaningful offset — fall back to p0
            return self._generate_step(messages, steering_decision)

        # Determine prefix based on timing
        if timing == "p4":
            prefix_ids = prefix_space
        else:  # "p2"
            half = max(1, len(prefix_space) // 2)
            prefix_ids = prefix_space[:half]

        # Build extended input: original prompt + thought prefix
        prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=self.device)
        input_ids_2 = torch.cat([input_ids, prefix_tensor], dim=1)
        new_prompt_len = input_ids_2.shape[1]

        gen_kwargs_2 = {
            "max_new_tokens": self.config.max_tokens_per_step,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "attention_mask": torch.ones_like(input_ids_2),
            "do_sample": False,
        }

        alpha = steering_decision.alpha if steering_decision else 0.0

        # Pass 2: inject at position=-1 of input_ids_2 (= last prefix token)
        if alpha != 0.0 and self.direction is not None:
            with SteeringHook(
                self.model, self.direction, alpha,
                layer=self.config.layer, position=-1,
                max_interventions=1,
            ):
                with torch.no_grad():
                    outputs2 = self.model.generate(input_ids_2, **gen_kwargs_2)
        else:
            with torch.no_grad():
                outputs2 = self.model.generate(input_ids_2, **gen_kwargs_2)

        new_completion_ids = outputs2[0, new_prompt_len:].tolist()

        # Reconstruct completion: thought prefix + model continuation from pass 2
        full_ids = prefix_ids + new_completion_ids
        completion_text = self.tokenizer.decode(full_ids, skip_special_tokens=True)

        return completion_text, prompt_len, len(new_completion_ids)

    def _build_final_prompt(self, messages: List[Dict], question: str) -> str:
        """Build a prompt that asks the model to give a final answer directly."""
        # Add instruction to generate final answer
        final_messages = messages.copy()
        final_messages.append({
            "role": "user",
            "content": "Based on the information above, provide your final answer. Just state the answer directly without any reasoning."
        })
        return self.tokenizer.apply_chat_template(
            final_messages, tokenize=False, add_generation_prompt=True
        )

    def _generate_final(self, prompt: str) -> tuple:
        """Generate final answer without steering."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        gen_kwargs = {
            "max_new_tokens": 128,  # Final answers are short
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "attention_mask": attention_mask,
            "do_sample": False,
        }

        with torch.no_grad():
            outputs = self.model.generate(input_ids, **gen_kwargs)

        completion_ids = outputs[0, prompt_len:]
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

        return completion_text, prompt_len, len(completion_ids)

    def generate_rewrite_query(self, question: str, prev_query: str, observation: str) -> str:
        """Generate a rewritten query for second search (used by Oracle2Hop)."""
        rewrite_prompt = f"""You are a search query optimizer. A search was performed but the answer was not found.

Question: {question}
First query tried: {prev_query}
First result (did not contain the answer): {observation[:400]}

Generate ONE better search query. Focus on:
1. The specific named entity in the question (person, place, work title)
2. The exact attribute being asked (director, author, publisher, genre, etc.)
3. Try a more specific or alternate phrasing if the entity is ambiguous

Output ONLY the search query, one line, nothing else:"""

        messages = [{"role": "user", "content": rewrite_prompt}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        gen_kwargs = {
            "max_new_tokens": 64,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "attention_mask": attention_mask,
            "do_sample": False,
        }

        with torch.no_grad():
            outputs = self.model.generate(input_ids, **gen_kwargs)

        completion_ids = outputs[0, prompt_len:]
        query = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        # Clean up: take first line only
        query = query.split('\n')[0].strip()
        return query

    def run(
        self,
        question: str,
        policy: Policy = None,
        gold_answer: str = None,
        episode_id: str = None,
        target_side: str = "positive",  # "positive" = should use tool
        context: str = None,
    ) -> EpisodeResult:
        """
        Run agent on a question.

        Args:
            question: The question to answer
            policy: Steering policy (default: BaselinePolicy)
            gold_answer: Ground truth answer for evaluation
            episode_id: Unique ID for this episode
            target_side: "positive" (should adopt tool) or "negative" (should reject)
            context: Optional context to include

        Returns:
            EpisodeResult with full trajectory
        """
        if policy is None:
            policy = BaselinePolicy()

        # IMPORTANT: some policies maintain per-episode state (e.g., step counters,
        # guard-trigger bookkeeping, tau schedules). Ensure it is reset at the
        # beginning of each episode. Without this, JES step-aware schedules and
        # guard statistics become invalid and can silently skew results.
        if hasattr(policy, "reset_episode") and callable(getattr(policy, "reset_episode")):
            policy.reset_episode()

        # Ensure oracle policies always have access to the question, even if
        # step 0 fails to parse / execute a search (which would prevent
        # set_context from being called).  Without this, the fallback query
        # in Oracle2HopPolicy.decide() degrades to "information".
        if hasattr(policy, '_question'):
            policy._question = question

        result = EpisodeResult(
            id=episode_id or "unknown",
            question=question,
            success=False,
            gold_answer=gold_answer,
            policy=policy.name,
        )

        steps = []
        messages = self.prompt_builder.build_full_prompt(question, steps, context)

        start_time = time.time()

        for step_idx in range(self.config.max_steps):
            step_start = time.time()
            step_result = StepResult(step_idx=step_idx)

            # Compute margin and get steering decision
            def margin_fn(rho):
                return self._compute_margin(messages, rho)

            steering_decision = policy.decide(
                margin_fn, target_side,
                self._get_hidden_rms(),
                self.direction_rms or 1.0
            )

            # Use cached m_before from policy to avoid redundant forward pass
            if steering_decision.m_before is not None:
                step_result.margin_before = steering_decision.m_before
            elif getattr(steering_decision, "skip_margin_before_log", False):
                step_result.margin_before = None
            else:
                step_result.margin_before = margin_fn(0.0)
            # IMPORTANT: Always log the actual applied rho/alpha (not only policy-specific details).
            # Some policies (e.g., forced / fixed-rho) store rho under different keys, and
            # downstream analysis relies on a consistent `rho_used` field.
            steering_payload = dict(steering_decision.details or {})
            steering_payload.setdefault("policy_name", steering_decision.policy_name)
            steering_payload.setdefault("rho_used", float(steering_decision.rho))
            steering_payload.setdefault("alpha_used", float(steering_decision.alpha))
            steering_payload["rho"] = float(steering_decision.rho)
            steering_payload["alpha"] = float(steering_decision.alpha)
            step_result.steering = steering_payload

            # Best-effort margin-after logging:
            # - JES computes and returns m_after via its internal cached margin_fn.
            # - Other policies may not provide it.
            # Keeping this field populated makes downstream audits more reliable.
            if step_result.margin_after is None and "m_after" in steering_payload:
                step_result.margin_after = steering_payload.get("m_after")

            # Check for override action (bypass model generation)
            if steering_decision.override_action:
                override_action = steering_decision.override_action.lower()
                step_result.raw_model_text = f"[OVERRIDE:{override_action}]"

                if override_action == "search" and steering_decision.override_input:
                    # Override: execute search directly with provided query
                    step_result.action = "search"
                    step_result.action_input = steering_decision.override_input
                    tool_name = "search"
                    if tool_name in self.tools:
                        try:
                            observation = self.tools[tool_name](steering_decision.override_input)
                            step_result.observation = str(observation)[:2000]
                            result.total_tool_calls += 1
                            if hasattr(self.tools[tool_name], 'was_corrupted'):
                                step_result.corruption_applied = self.tools[tool_name].was_corrupted

                            # Set context for Oracle2HopPolicy after first override search
                            if hasattr(policy, 'set_context') and step_result.step_idx == 0:
                                policy.set_context(
                                    question,
                                    steering_decision.override_input,
                                    step_result.observation,
                                )
                                if hasattr(policy, 'set_agent'):
                                    policy.set_agent(self)
                        except Exception as e:
                            step_result.observation = f"Error: {str(e)}"
                    step_result.wall_time_ms = (time.time() - step_start) * 1000
                    result.steps.append(step_result)
                    # IMPORTANT: rebuild messages via build_full_prompt after override search.
                    # Directly appending assistant+user messages would create consecutive same-role
                    # messages when combined with the previous step's scratchpad, breaking the
                    # chat template alternating-role structure.
                    # build_full_prompt consolidates all steps into one assistant scratchpad message.
                    steps.append({
                        "action": "search",
                        "action_input": steering_decision.override_input,
                        "observation": step_result.observation,
                    })
                    messages = self.prompt_builder.build_full_prompt(question, steps, context)
                    continue

                elif override_action == "final":
                    # Override: force Final answer generation (no tool call)
                    # Generate only the final answer part
                    final_prompt = self._build_final_prompt(messages, question)
                    completion, prompt_tokens, completion_tokens = self._generate_final(final_prompt)
                    step_result.tokens_prompt = prompt_tokens
                    step_result.tokens_completion = completion_tokens
                    step_result.raw_model_text = f"[OVERRIDE:final] {completion[:150]}"
                    step_result.action = None
                    # Strip "Final Answer:" prefix if model echoes it
                    clean_answer = completion.strip()
                    for _prefix in ("Final Answer:", "Final:"):
                        if clean_answer.startswith(_prefix):
                            clean_answer = clean_answer[len(_prefix):].strip()
                            break
                    step_result.final_answer = clean_answer
                    result.final_answer = clean_answer
                    step_result.wall_time_ms = (time.time() - step_start) * 1000
                    result.steps.append(step_result)
                    break

            # Normal path: route via the model's actual next-token distribution
            # and generation under steering. Do NOT pre-inject an "Action" or
            # "Final Answer" prefix here: that turns JES diagnosis into a
            # prefix-routing artifact and bypasses the actual steering effect.
            _timing = getattr(steering_decision, 'timing', 'p0')
            if _timing in ('p2', 'p4') and (steering_decision.alpha if steering_decision else 0.0) != 0.0:
                completion, prompt_tokens, completion_tokens = self._generate_step_timed(
                    messages, steering_decision, _timing
                )
            else:
                completion, prompt_tokens, completion_tokens = self._generate_step(
                    messages, steering_decision
                )
            step_result.tokens_prompt = prompt_tokens
            step_result.tokens_completion = completion_tokens
            # Store raw model text (truncated for logging)
            step_result.raw_model_text = completion[:200] if completion else None

            # Parse action
            parsed = parse_action(completion)
            step_result.action = parsed["action"]
            step_result.action_input = parsed["action_input"]
            step_result.final_answer = parsed["final_answer"]

            # Check for final answer
            if parsed["final_answer"]:
                result.final_answer = parsed["final_answer"]
                step_result.wall_time_ms = (time.time() - step_start) * 1000
                result.steps.append(step_result)
                break

            # Execute tool if action specified
            if parsed["action"] and parsed["action"].lower() in self.tools:
                tool_name = parsed["action"].lower()
                tool_input = parsed["action_input"] or ""
                try:
                    observation = self.tools[tool_name](tool_input)
                    step_result.observation = str(observation)[:2000]  # Truncate
                    result.total_tool_calls += 1
                    # Track corruption (Bug #1 fix)
                    if hasattr(self.tools[tool_name], 'was_corrupted'):
                        step_result.corruption_applied = self.tools[tool_name].was_corrupted

                    # Set context for Oracle2HopPolicy after first search
                    if hasattr(policy, 'set_context') and step_result.step_idx == 0:
                        policy.set_context(question, tool_input, step_result.observation)
                        if hasattr(policy, 'set_agent'):
                            policy.set_agent(self)

                except Exception as e:
                    step_result.observation = f"Error: {str(e)}"
                    result.failure_reason = "execution_error"
            elif parsed["action"]:
                step_result.observation = f"[PARSE_FAILURE] Unknown tool: {parsed['action']}"
                step_result.parse_failure_reason = f"Unknown tool: {parsed['action']}"
                result.failure_reason = "parsing_error"
            else:
                step_result.observation = "[PARSE_FAILURE] No valid action parsed"
                step_result.parse_failure_reason = "No action or final_answer parsed from model output"
                result.failure_reason = "parsing_error"

            step_result.wall_time_ms = (time.time() - step_start) * 1000
            result.steps.append(step_result)

            # Update messages with new step.
            # Keep parse failures out of the scratchpad so malformed generations
            # do not contaminate the next decision point.
            if step_result.parse_failure_reason is None:
                steps.append({
                    "action": parsed["action"],
                    "action_input": parsed["action_input"],
                    "observation": step_result.observation,
                })
            messages = self.prompt_builder.build_full_prompt(question, steps, context)

        # Check timeout
        if len(result.steps) >= self.config.max_steps and not result.final_answer:
            result.failure_reason = "timeout"

        # Compute totals
        result.total_tokens = sum(s.tokens_prompt + s.tokens_completion for s in result.steps)
        result.total_wall_time_ms = (time.time() - start_time) * 1000

        # Evaluate success if gold_answer is provided
        if gold_answer and result.final_answer:
            from eval.scorers import answer_scorer
            score_result = answer_scorer(result.final_answer, gold_answer, mode=self.config.score_mode)
            result.success = score_result["matched"]
        elif result.final_answer:
            # No gold answer to compare: do NOT mark as success.
            # Keeping success=False avoids silently producing "fake" correctness.
            result.success = False

        return result

