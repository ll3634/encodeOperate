#!/usr/bin/env python3
"""
Verify-Critical Mining Policies.

Policies for mining verify-critical samples:
1. Baseline1HopPolicy: Exactly 1 search via normal generation, then OVERRIDE Final
2. Oracle2HopPolicy: Exactly 2 searches (2nd via query-rewrite), then OVERRIDE Final
3. JESStep2OnlyPolicy: JES only at step2 decision token (decision-only steering)

Key principle: Baseline/Oracle use OVERRIDE (direction-free, parse-free).
JES is separate and uses small-rho decision-only steering.
"""

import numpy as np
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

from .policies import Policy, SteeringDecision

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from steering.jes import JESConfig, JESController


class Baseline1HopPolicy(Policy):
    """
    Exactly 1 search, then OVERRIDE to Final.

    - Step 0: Normal generation (model generates Action: search)
    - Step 1+: OVERRIDE to Final (bypass parsing, directly generate final answer)

    Direction-free: no steering involved, purely override-based.
    """

    def __init__(self):
        self._step_counter = 0
        self._question = None  # Set by react_loop via set_context or externally

    @property
    def name(self) -> str:
        return "baseline_1hop"

    def reset_episode(self):
        self._step_counter = 0
        self._question = None

    def set_context(self, question: str, first_query: str = None, first_observation: str = None):
        """Retained for API compatibility with react_loop."""
        self._question = question

    def decide(self, margin_fn: Callable[[float], float], target_side: str,
               hidden_rms: float, direction_rms: float) -> SteeringDecision:
        step = self._step_counter
        self._step_counter += 1

        if step == 0:
            # Step 0: Normal generation - model will search.
            return SteeringDecision(
                rho=0.0, alpha=0.0, policy_name=self.name,
                skip_margin_before_log=True,
                details={"step": step, "action": "normal_gen"}
            )
        else:
            # Step 1+: OVERRIDE to Final - no model decision, force final answer
            return SteeringDecision(
                rho=0.0, alpha=0.0, policy_name=self.name,
                skip_margin_before_log=True,
                override_action="final",
                details={"step": step, "action": "override_final"}
            )


class Oracle2HopPolicy(Policy):
    """
    Exactly 2 searches via OVERRIDE, then Final.

    - Step 0: Normal generation (model generates Action: search with query1)
    - Step 1: OVERRIDE to search with rewritten query2 (no model decision)
    - Step 2+: OVERRIDE to Final

    Direction-free: no steering involved, purely override-based.
    Query rewrite is done via a dedicated prompt, not ReAct parsing.
    """

    def __init__(self, agent_ref=None):
        """
        Args:
            agent_ref: Reference to ReActAgent for query rewriting.
                       Set via set_agent() before running.
        """
        self._step_counter = 0
        self._agent = agent_ref
        self._first_query = None
        self._first_observation = None
        self._question = None

    @property
    def name(self) -> str:
        return "oracle_2hop"

    def reset_episode(self):
        self._step_counter = 0
        self._first_query = None
        self._first_observation = None
        self._question = None

    def set_agent(self, agent):
        """Set agent reference for query rewriting."""
        self._agent = agent

    def set_context(self, question: str, first_query: str, first_observation: str):
        """Set context from first step for query rewriting."""
        self._question = question
        self._first_query = first_query
        self._first_observation = first_observation

    def decide(self, margin_fn: Callable[[float], float], target_side: str,
               hidden_rms: float, direction_rms: float) -> SteeringDecision:
        step = self._step_counter
        self._step_counter += 1

        if step == 0:
            # Step 0: Normal generation - model will generate first search.
            return SteeringDecision(
                rho=0.0, alpha=0.0, policy_name=self.name,
                skip_margin_before_log=True,
                details={"step": step, "action": "normal_gen"}
            )
        elif step == 1:
            # Step 1: OVERRIDE to search with rewritten query
            # Generate query2 via dedicated rewrite prompt
            query2 = None
            if self._agent and self._first_query and self._first_observation:
                query2 = self._agent.generate_rewrite_query(
                    self._question, self._first_query, self._first_observation
                )

            if not query2:
                # Fallback: if first search failed/parsed incorrectly,
                # use the original question as query (better than garbage)
                query2 = self._question or self._first_query or "information"

            return SteeringDecision(
                rho=0.0, alpha=0.0, policy_name=self.name,
                skip_margin_before_log=True,
                override_action="search",
                override_input=query2,
                details={"step": step, "action": "override_search", "query2": query2}
            )
        else:
            # Step 2+: OVERRIDE to Final
            return SteeringDecision(
                rho=0.0, alpha=0.0, policy_name=self.name,
                skip_margin_before_log=True,
                override_action="final",
                details={"step": step, "action": "override_final"}
            )


class JESStep2OnlyPolicy(Policy):
    """
    JES only at step 2 decision token (decision-only steering).

    - Step 0: No steering
    - Step 1: Apply JES (push toward tool_call when uncertain)
    - Step 2+: No steering (let model finalize)

    This is the "矛" (spear) policy for verify-critical mining.
    """

    def __init__(self, config: JESConfig = None, direction: np.ndarray = None):
        self.config = config or JESConfig()
        self.direction = direction
        self._controller = None
        self._step_counter = 0
        self._question = None

    @property
    def name(self) -> str:
        return "jes_step2_only"

    def reset_episode(self):
        self._step_counter = 0
        self._controller = None  # Reset controller state
        self._question = None

    def set_context(self, question: str, first_query: str = None, first_observation: str = None):
        """Retained for API compatibility with react_loop."""
        self._question = question

    def _get_controller(self, hidden_rms: float, direction_rms: float) -> JESController:
        """Lazy init controller with RMS values."""
        if self._controller is None or self._controller.hidden_rms != hidden_rms:
            self._controller = JESController(
                self.config,
                self.direction if self.direction is not None else np.zeros(1),
                hidden_rms,
                direction_rms
            )
        return self._controller

    def decide(self, margin_fn: Callable[[float], float], target_side: str,
               hidden_rms: float, direction_rms: float) -> SteeringDecision:
        step = self._step_counter
        self._step_counter += 1

        m0 = margin_fn(0.0)

        if step == 0:
            # Step 0: No steering (first search).
            return SteeringDecision(
                rho=0.0, alpha=0.0, policy_name=self.name, m_before=m0,
                details={"step": step, "action": "no_steering_step0"}
            )
        elif step == 1:
            # Step 1: Apply JES to push toward tool_call when uncertain
            # This is the critical decision point: "should I verify?"
            controller = self._get_controller(hidden_rms, direction_rms)
            result = controller.compute_steering(margin_fn, target_side)

            return SteeringDecision(
                rho=result.rho_used,
                alpha=result.alpha_used,
                policy_name=self.name,
                m_before=result.m_before,
                decision_only=True,
                details={
                    "step": step,
                    "action": "jes_steering",
                    **result.to_dict()
                }
            )
        else:
            # Step 2+: No steering (let model finalize after potential 2nd search)
            return SteeringDecision(
                rho=0.0, alpha=0.0, policy_name=self.name, m_before=m0,
                details={"step": step, "action": "no_steering_step2+"}
            )


class FixedRhoStep2OnlyPolicy(Policy):
    """
    Fixed-rho steering at step 1 only (no JES adaptation).

    - Step 0: No steering (first search)
    - Step 1: Apply fixed rho (positive = push toward search, negative = push toward Final)
    - Step 2+: No steering

    Use negative rho for *reverse steering* diagnostic: if the direction is
    semantically meaningful, rho < 0 should increase regressions; random
    directions should have a null or symmetric effect.
    """

    def __init__(self, rho: float, alpha_max: float = 8.0, steer_layer: Optional[int] = None):
        self.rho = rho
        self.alpha_max = alpha_max
        self._steer_layer = steer_layer
        self._step_counter = 0
        self._question = None

    @property
    def name(self) -> str:
        return f"fixed_rho_step2_{self.rho:+.2f}"

    def reset_episode(self):
        self._step_counter = 0
        self._question = None

    def set_context(self, question: str, first_query: str = None, first_observation: str = None):
        self._question = question

    def decide(self, margin_fn: Callable[[float], float], target_side: str,
               hidden_rms: float, direction_rms: float) -> SteeringDecision:
        step = self._step_counter
        self._step_counter += 1

        if step == 0:
            return SteeringDecision(
                rho=0.0, alpha=0.0, policy_name=self.name,
                skip_margin_before_log=True,
                details={"step": step, "action": "no_steering_step0"}
            )
        elif step == 1:
            m0 = margin_fn(0.0)
            alpha = self.rho * (hidden_rms / direction_rms)
            alpha = float(np.clip(alpha, -self.alpha_max, self.alpha_max))
            return SteeringDecision(
                rho=self.rho, alpha=alpha, policy_name=self.name,
                m_before=m0, decision_only=True,
                steer_layer=self._steer_layer,
                details={
                    "step": step, "action": "fixed_rho_steering",
                    "rho_used": self.rho, "alpha_used": alpha,
                    "steer_layer": self._steer_layer,
                    "m_before": m0,
                }
            )
        else:
            return SteeringDecision(
                rho=0.0, alpha=0.0, policy_name=self.name,
                skip_margin_before_log=True,
                details={"step": step, "action": "no_steering_step2+"}
            )


class FixedAlphaStep2OnlyPolicy(Policy):
    """
    Fixed-alpha steering at step 1 only (no rho→alpha conversion).

    Like FixedRhoStep2OnlyPolicy but takes alpha directly, bypassing
    the hidden_rms/direction_rms scaling and alpha_max clamp.
    This is needed when direction_rms is very small, causing even tiny
    rho values to be clipped to alpha_max.
    """

    def __init__(self, alpha: float, steer_layer: Optional[int] = None):
        self.alpha = alpha
        self.steer_layer = steer_layer
        self._step_counter = 0
        self._question = None

    @property
    def name(self) -> str:
        if self.steer_layer is not None:
            return f"fixed_alpha_step2_{self.alpha:+.2f}_L{self.steer_layer}"
        return f"fixed_alpha_step2_{self.alpha:+.2f}"

    def reset_episode(self):
        self._step_counter = 0
        self._question = None

    def set_context(self, question: str, first_query: str = None, first_observation: str = None):
        self._question = question

    def decide(self, margin_fn: Callable[[float], float], target_side: str,
               hidden_rms: float, direction_rms: float) -> SteeringDecision:
        step = self._step_counter
        self._step_counter += 1

        if step == 0:
            return SteeringDecision(
                rho=0.0, alpha=0.0, policy_name=self.name,
                skip_margin_before_log=True,
                details={"step": step, "action": "no_steering_step0"}
            )
        elif step == 1:
            m0 = margin_fn(0.0)
            # Back-compute rho for logging (informational only)
            if direction_rms > 0 and hidden_rms > 0:
                rho_equiv = self.alpha * direction_rms / hidden_rms
            else:
                rho_equiv = 0.0
            return SteeringDecision(
                rho=rho_equiv, alpha=self.alpha, policy_name=self.name,
                m_before=m0, decision_only=True,
                steer_layer=self.steer_layer,
                details={
                    "step": step, "action": "fixed_alpha_steering",
                    "alpha_used": self.alpha, "rho_equiv": rho_equiv,
                    "steer_layer": self.steer_layer,
                    "m_before": m0,
                }
            )
        else:
            return SteeringDecision(
                rho=0.0, alpha=0.0, policy_name=self.name,
                skip_margin_before_log=True,
                details={"step": step, "action": "no_steering_step2+"}
            )


class TimedRhoStep2OnlyPolicy(Policy):
    """
    Fixed-rho steering at step 1 only, with configurable injection timing.

    timing="p0" : inject at the last token of the input prompt (identical to
                  FixedRhoStep2OnlyPolicy with decision_only=True).
    timing="p2" : inject at 50% through the generated thought (two-pass).
    timing="p4" : inject at the last thought token, just before Action:/Final
                  Answer: (two-pass).

    Two-pass generation for p2/p4 is handled transparently by ReActAgent when
    it sees SteeringDecision.timing != "p0".
    """

    def __init__(self, rho: float, timing: str = "p0", alpha_max: float = 8.0):
        if timing not in ("p0", "p2", "p4"):
            raise ValueError(f"timing must be 'p0', 'p2', or 'p4'; got {timing!r}")
        self.rho = rho
        self.timing = timing
        self.alpha_max = alpha_max
        self._step_counter = 0

    @property
    def name(self) -> str:
        return f"timed_rho_{self.rho:+.2f}_{self.timing}"

    def reset_episode(self):
        self._step_counter = 0

    def set_context(self, question: str, first_query: str = None, first_observation: str = None):
        pass

    def decide(self, margin_fn: Callable[[float], float], target_side: str,
               hidden_rms: float, direction_rms: float) -> SteeringDecision:
        step = self._step_counter
        self._step_counter += 1

        if step == 0:
            return SteeringDecision(
                rho=0.0, alpha=0.0, policy_name=self.name,
                skip_margin_before_log=True,
                details={"step": step, "action": "no_steering_step0"}
            )
        elif step == 1:
            m0 = margin_fn(0.0)
            alpha = self.rho * (hidden_rms / direction_rms)
            alpha = float(np.clip(alpha, -self.alpha_max, self.alpha_max))
            return SteeringDecision(
                rho=self.rho, alpha=alpha, policy_name=self.name,
                m_before=m0, decision_only=True,
                timing=self.timing,
                details={
                    "step": step, "action": "timed_rho_steering",
                    "timing": self.timing,
                    "rho_used": self.rho, "alpha_used": alpha,
                    "m_before": m0,
                }
            )
        else:
            return SteeringDecision(
                rho=0.0, alpha=0.0, policy_name=self.name,
                skip_margin_before_log=True,
                details={"step": step, "action": "no_steering_step2+"}
            )


class FreeGenBaselinePolicy(Policy):
    """
    Fair baseline: no OVERRIDE, no steering. Model generates freely at every step.

    This ensures the generation path is identical to steered conditions,
    eliminating the OVERRIDE prompt/truncation confound.

    - Every step: normal generation with rho=0 (no steering hook)
    - Model decides search vs. final_answer on its own
    """

    def __init__(self):
        self._step_counter = 0
        self._question = None

    @property
    def name(self) -> str:
        return "freegen_baseline"

    def reset_episode(self):
        self._step_counter = 0
        self._question = None

    def set_context(self, question: str, first_query: str = None, first_observation: str = None):
        self._question = question

    def decide(self, margin_fn: Callable[[float], float], target_side: str,
               hidden_rms: float, direction_rms: float) -> SteeringDecision:
        step = self._step_counter
        self._step_counter += 1
        m0 = margin_fn(0.0)
        return SteeringDecision(
            rho=0.0, alpha=0.0, policy_name=self.name, m_before=m0,
            details={"step": step, "action": "freegen_no_steering"}
        )


class FixedRhoSteerPolicy(Policy):
    """
    Configurable-step fixed-rho steering. No OVERRIDE anywhere.

    Unlike FixedRhoStep2OnlyPolicy which only steers at step 1,
    this policy steers at a configurable step (default=0 for initial
    search decision). All other steps use normal generation with rho=0.

    - steer_step: which step to apply steering (0 = initial decision)
    - decision_only: if True, limit the steering hook to the first token of
                     generation only (the decision token). This matches the
                     behaviour of EveryStepJESPolicy and prevents the steering
                     vector from distorting answer content.
    - All steps: normal generation (no OVERRIDE)
    """

    def __init__(self, rho: float, steer_step: int = None, alpha_max: float = 8.0,
                 decision_only: bool = False):
        """
        Args:
            rho: steering magnitude
            steer_step: which step to steer. None = steer at ALL steps (natural JES).
                        0 = initial decision only, 1 = second-search only, etc.
            alpha_max: clamp for alpha
            decision_only: if True, limit hook to first forward pass (decision token).
        """
        self.rho = rho
        self.steer_step = steer_step  # None means every step
        self.alpha_max = alpha_max
        self._decision_only = decision_only
        self._step_counter = 0
        self._question = None

    @property
    def name(self) -> str:
        step_label = "all" if self.steer_step is None else str(self.steer_step)
        do_label = "_do" if self._decision_only else ""
        return f"steer_step{step_label}_rho{self.rho:+.2f}{do_label}"

    def reset_episode(self):
        self._step_counter = 0
        self._question = None

    def set_context(self, question: str, first_query: str = None, first_observation: str = None):
        self._question = question

    def decide(self, margin_fn: Callable[[float], float], target_side: str,
               hidden_rms: float, direction_rms: float) -> SteeringDecision:
        step = self._step_counter
        self._step_counter += 1

        m0 = margin_fn(0.0)

        if self.steer_step is None or step == self.steer_step:
            alpha = self.rho * (hidden_rms / direction_rms)
            alpha = float(np.clip(alpha, -self.alpha_max, self.alpha_max))
            return SteeringDecision(
                rho=self.rho, alpha=alpha, policy_name=self.name,
                m_before=m0,
                decision_only=self._decision_only,
                details={
                    "step": step, "action": "fixed_rho_steering",
                    "rho_used": self.rho, "alpha_used": alpha,
                    "m_before": m0,
                    "decision_only": self._decision_only,
                }
            )
        else:
            return SteeringDecision(
                rho=0.0, alpha=0.0, policy_name=self.name, m_before=m0,
                details={"step": step, "action": "no_steering"}
            )


class JESStep2ForcePolicy(Policy):
    """
    Alternative: JES at step 1, then force Final at step 2+.

    More aggressive than JESStep2OnlyPolicy:
    - Step 0: No steering
    - Step 1: Apply JES (push toward tool_call when uncertain)
    - Step 2+: Force Final (to cap cost)
    """

    def __init__(self, config: JESConfig = None, direction: np.ndarray = None,
                 rho_magnitude: float = 0.5, alpha_max: float = 2000.0):
        self.config = config or JESConfig()
        self.direction = direction
        self.rho_magnitude = rho_magnitude
        self.alpha_max = alpha_max
        self._controller = None
        self._step_counter = 0
        self._question = None

    @property
    def name(self) -> str:
        return "jes_step2_force"

    def reset_episode(self):
        self._step_counter = 0
        self._controller = None
        self._question = None

    def set_context(self, question: str, first_query: str = None, first_observation: str = None):
        """Retained for API compatibility with react_loop."""
        self._question = question

    def _get_controller(self, hidden_rms: float, direction_rms: float) -> JESController:
        if self._controller is None or self._controller.hidden_rms != hidden_rms:
            self._controller = JESController(
                self.config,
                self.direction if self.direction is not None else np.zeros(1),
                hidden_rms,
                direction_rms
            )
        return self._controller

    def decide(self, margin_fn: Callable[[float], float], target_side: str,
               hidden_rms: float, direction_rms: float) -> SteeringDecision:
        step = self._step_counter
        self._step_counter += 1

        m0 = margin_fn(0.0)

        if step == 0:
            # Step 0: No steering (first search).
            return SteeringDecision(
                rho=0.0, alpha=0.0, policy_name=self.name, m_before=m0,
                details={"step": step, "action": "no_steering_step0"}
            )
        elif step == 1:
            controller = self._get_controller(hidden_rms, direction_rms)
            result = controller.compute_steering(margin_fn, target_side)
            return SteeringDecision(
                rho=result.rho_used,
                alpha=result.alpha_used,
                policy_name=self.name,
                m_before=result.m_before,
                decision_only=True,
                details={"step": step, "action": "jes_steering", **result.to_dict()}
            )
        else:
            # Step 2+: Force Final
            rho_pos = float(self.rho_magnitude)
            rho_neg = -float(self.rho_magnitude)
            m_pos = margin_fn(rho_pos)
            m_neg = margin_fn(rho_neg)

            rho = rho_pos if m_pos <= m_neg else rho_neg
            alpha = rho * (hidden_rms / direction_rms)
            alpha = float(np.clip(alpha, -self.alpha_max, self.alpha_max))

            return SteeringDecision(
                rho=rho, alpha=alpha, policy_name=self.name, m_before=m0,
                details={"step": step, "action": "force_final", "rho_used": rho}
            )




class EveryStepJESPolicy(Policy):
    """
    Every-step adaptive JES: applies margin-based steering at EVERY step.

    JES itself decides whether to intervene:
    - If margin already satisfies tau → already_satisfied, rho=0, no push
    - If margin is uncertain (near boundary) → compute minimum rho* to push past tau
    - If margin is extreme opposite → may saturate (can't push enough)

    No OVERRIDE anywhere. No fixed step. Generalizes to n-hop scenarios.

    This is the "correct" JES design: the model's own uncertainty at each step
    determines whether and how much to steer.
    """

    def __init__(self, config: JESConfig = None, direction: np.ndarray = None):
        self.config = config or JESConfig()
        self.direction = direction
        self._controller = None
        self._step_counter = 0
        self._question = None

    @property
    def name(self) -> str:
        cfg = self.config
        return f"jes_every_tau{cfg.tau:.2f}_rho{cfg.max_rho:.2f}"

    def reset_episode(self):
        self._step_counter = 0
        self._controller = None
        self._question = None

    def set_context(self, question: str, first_query: str = None, first_observation: str = None):
        self._question = question

    def _get_controller(self, hidden_rms: float, direction_rms: float) -> JESController:
        if self._controller is None or self._controller.hidden_rms != hidden_rms:
            self._controller = JESController(
                self.config,
                self.direction if self.direction is not None else np.zeros(1),
                hidden_rms,
                direction_rms
            )
        return self._controller

    def decide(self, margin_fn: Callable[[float], float], target_side: str,
               hidden_rms: float, direction_rms: float) -> SteeringDecision:
        step = self._step_counter
        self._step_counter += 1

        controller = self._get_controller(hidden_rms, direction_rms)
        result = controller.compute_steering(margin_fn, target_side)

        return SteeringDecision(
            rho=result.rho_used,
            alpha=result.alpha_used,
            policy_name=self.name,
            m_before=result.m_before,
            decision_only=True,
            details={
                "step": step,
                "action": "jes_adaptive",
                "already_satisfied": result.already_satisfied,
                **result.to_dict()
            }
        )


class SteerPlusAblatePolicy(Policy):
    """
    Necessity test: L20 steering + L18 KV group ablation.

    Combines residual-stream steering (action_dir at L20) with KV-group
    ablation (alpha=0 at L18) to test whether the upstream attention
    circuit is necessary for steering to work.

    - Step 0: No intervention (first search happens naturally)
    - Step 1: Apply BOTH L20 steering AND L18 KV ablation at decision point
    - Step 2+: No intervention
    """

    def __init__(self, rho: float = -0.20, ablate_layer: int = 18,
                 ablate_kv_group: int = 2, alpha_max: float = 8.0):
        self._rho = rho
        self._ablate_layer = ablate_layer
        self._ablate_kv_group = ablate_kv_group
        self._alpha_max = alpha_max
        self._step_counter = 0
        self._question = None

    @property
    def name(self) -> str:
        return f"steer_rho{self._rho:+.2f}_ablate_L{self._ablate_layer}_KV{self._ablate_kv_group}"

    def reset_episode(self):
        self._step_counter = 0
        self._question = None

    def set_context(self, question: str, first_query: str = None, first_observation: str = None):
        self._question = question

    def decide(self, margin_fn, target_side, hidden_rms, direction_rms) -> SteeringDecision:
        step = self._step_counter
        self._step_counter += 1

        m0 = margin_fn(0.0)

        if step == 1:
            # Compute steering alpha from rho
            alpha = self._rho * (hidden_rms / direction_rms)
            alpha = float(np.clip(alpha, -self._alpha_max, self._alpha_max))
            return SteeringDecision(
                rho=self._rho, alpha=alpha, policy_name=self.name,
                m_before=m0,
                decision_only=True,
                steer_layer=19,  # Inject at L19 output = L20 input, so signal flows through mlp_L20
                kv_group_config={
                    "layer": self._ablate_layer,
                    "kv_group": self._ablate_kv_group,
                    "alpha": 0.0,  # ablation
                },
                details={
                    "step": step,
                    "action": "steer_plus_ablate",
                    "rho_used": self._rho,
                    "alpha_used": alpha,
                    "ablate_layer": self._ablate_layer,
                    "ablate_kv_group": self._ablate_kv_group,
                    "steer_layer": 19,
                    "m_before": m0,
                }
            )
        else:
            return SteeringDecision(
                rho=0.0, alpha=0.0, policy_name=self.name,
                m_before=m0 if step == 0 else None,
                skip_margin_before_log=(step != 0),
                details={"step": step, "action": "no_intervention"}
            )


class KVGroupScalingPolicy(Policy):
    """
    Circuit-level intervention: scale specific KV group output at attn_L18.

    - Step 0: No intervention (first search happens naturally)
    - Step 1: Apply KV group scaling at the decision point (decision_only=True)
    - Step 2+: No intervention

    This is used for the circuit-level behavioral intervention experiment.
    """

    def __init__(self, layer: int = 18, kv_group: int = 2, alpha: float = 2.0):
        self._layer = layer
        self._kv_group = kv_group
        self._alpha = alpha
        self._step_counter = 0
        self._question = None

    @property
    def name(self) -> str:
        return f"kv_scale_L{self._layer}_KV{self._kv_group}_a{self._alpha:.1f}"

    def reset_episode(self):
        self._step_counter = 0
        self._question = None

    def set_context(self, question: str, first_query: str = None, first_observation: str = None):
        self._question = question

    def decide(self, margin_fn, target_side, hidden_rms, direction_rms) -> SteeringDecision:
        step = self._step_counter
        self._step_counter += 1

        m0 = margin_fn(0.0)

        if step == 1:
            # Apply KV group scaling at decision point
            return SteeringDecision(
                rho=0.0, alpha=0.0, policy_name=self.name,
                m_before=m0,
                decision_only=True,
                kv_group_config={
                    "layer": self._layer,
                    "kv_group": self._kv_group,
                    "alpha": self._alpha,
                },
                details={
                    "step": step,
                    "action": "kv_group_scaling",
                    "layer": self._layer,
                    "kv_group": self._kv_group,
                    "scale_alpha": self._alpha,
                    "m_before": m0,
                }
            )
        else:
            return SteeringDecision(
                rho=0.0, alpha=0.0, policy_name=self.name,
                m_before=m0 if step == 0 else None,
                skip_margin_before_log=(step != 0),
                details={"step": step, "action": "no_intervention"}
            )


class KVGroupDirectionalScalingPolicy(Policy):
    """
    Circuit-level intervention: decompose a KV group's contribution to attn
    output along a reference direction (e.g. evidence_dir) and scale the
    parallel / orthogonal components independently.

    - Step 0: No intervention (first search happens naturally)
    - Step 1: Apply directional KV-group scaling at the decision point
    - Step 2+: No intervention

    Used for KV2 output decomposition patching: tests whether KV2 routes
    evidence information through the evidence-parallel channel or the
    evidence-orthogonal channel of its output.
    """

    def __init__(
        self,
        layer: int = 18,
        kv_group: int = 2,
        direction: np.ndarray = None,
        alpha_parallel: float = 1.0,
        alpha_orth: float = 1.0,
        tag: str = "",
    ):
        if direction is None:
            raise ValueError("direction (reference vector) is required")
        self._layer = layer
        self._kv_group = kv_group
        self._direction = np.asarray(direction, dtype=np.float32).reshape(-1)
        self._alpha_parallel = alpha_parallel
        self._alpha_orth = alpha_orth
        self._tag = tag
        self._step_counter = 0
        self._question = None

    @property
    def name(self) -> str:
        base = (f"kv_dir_scale_L{self._layer}_KV{self._kv_group}"
                f"_ap{self._alpha_parallel:.1f}_ao{self._alpha_orth:.1f}")
        return f"{base}_{self._tag}" if self._tag else base

    def reset_episode(self):
        self._step_counter = 0
        self._question = None

    def set_context(self, question: str, first_query: str = None,
                    first_observation: str = None):
        self._question = question

    def decide(self, margin_fn, target_side, hidden_rms, direction_rms) -> SteeringDecision:
        step = self._step_counter
        self._step_counter += 1

        m0 = margin_fn(0.0)

        if step == 1:
            return SteeringDecision(
                rho=0.0, alpha=0.0, policy_name=self.name,
                m_before=m0,
                decision_only=True,
                kv_group_config={
                    "mode": "directional",
                    "layer": self._layer,
                    "kv_group": self._kv_group,
                    "direction": self._direction,
                    "alpha_parallel": self._alpha_parallel,
                    "alpha_orth": self._alpha_orth,
                },
                details={
                    "step": step,
                    "action": "kv_group_directional_scaling",
                    "layer": self._layer,
                    "kv_group": self._kv_group,
                    "alpha_parallel": self._alpha_parallel,
                    "alpha_orth": self._alpha_orth,
                    "m_before": m0,
                }
            )
        else:
            return SteeringDecision(
                rho=0.0, alpha=0.0, policy_name=self.name,
                m_before=m0 if step == 0 else None,
                skip_margin_before_log=(step != 0),
                details={"step": step, "action": "no_intervention"}
            )



class CrossAxisStep2OnlyPolicy(Policy):
    """
    Rank-1 cross-axis intervention at step 1 only (decision-only).

    At the decision token of step 1, modifies the residual stream as:
        h' = h + alpha * (h . u_in) * u_out
    where u_in/u_out are unit-norm vectors.  Used for Arm B of the
    reconnection sweep: tests whether learning a rank-1 mapping from the
    evidence axis to the action axis makes evidence operative for the
    stop/continue decision.
    """

    def __init__(
        self,
        u_in: np.ndarray,
        u_out: np.ndarray,
        alpha: float,
        layer: int = 20,
    ):
        self.u_in = u_in
        self.u_out = u_out
        self.alpha = alpha
        self.layer = layer
        self._step_counter = 0
        self._question = None

    @property
    def name(self) -> str:
        return f"cross_axis_step2_a{self.alpha:+.3f}_L{self.layer}"

    def reset_episode(self):
        self._step_counter = 0
        self._question = None

    def set_context(self, question: str, first_query: str = None, first_observation: str = None):
        self._question = question

    def decide(self, margin_fn: Callable[[float], float], target_side: str,
               hidden_rms: float, direction_rms: float) -> SteeringDecision:
        step = self._step_counter
        self._step_counter += 1

        if step == 1:
            m0 = margin_fn(0.0)
            return SteeringDecision(
                rho=0.0, alpha=0.0, policy_name=self.name,
                m_before=m0, decision_only=True,
                cross_axis_config={
                    "layer": self.layer,
                    "u_in": self.u_in,
                    "u_out": self.u_out,
                    "alpha": self.alpha,
                },
                details={
                    "step": step, "action": "cross_axis_steering",
                    "alpha_used": self.alpha, "layer": self.layer,
                    "m_before": m0,
                }
            )
        else:
            return SteeringDecision(
                rho=0.0, alpha=0.0, policy_name=self.name,
                skip_margin_before_log=True,
                details={"step": step, "action": "no_steering"}
            )
