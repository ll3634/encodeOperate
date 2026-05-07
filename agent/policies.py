#!/usr/bin/env python3
"""
Steering policies for ReAct agent.
Defines how to compute intervention strength (rho/alpha) at each decision point.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

import sys
sys.path.insert(0, '..')
from steering.jes import JESConfig, JESController, JESResult
from steering.hook_utils import compute_rms


@dataclass
class SteeringDecision:
    """Result of a steering policy decision."""
    rho: float = 0.0
    alpha: float = 0.0
    policy_name: str = "baseline"
    m_before: Optional[float] = None  # Baseline margin (rho=0), cached to avoid recomputation
    skip_margin_before_log: bool = False  # If True, do not compute margin_fn(0.0) only for logging.
    details: Dict[str, Any] = None

    # Override fields: bypass model generation entirely
    override_action: Optional[str] = None  # "search", "final", or None (no override)
    override_input: Optional[str] = None   # For search: the query; for final: None (will generate)

    # Decision-only steering: if True, steering hook fires only for the first
    # forward pass (the decision token) and is automatically disabled for
    # subsequent autoregressive tokens in the same generation step.
    decision_only: bool = False

    # Injection timing for two-pass intervention experiments.
    # "p0" = inject at last token of input prompt (default, same as decision_only).
    # "p2" = inject at 50% through the generated thought (two-pass).
    # "p4" = inject at last thought token, just before Action:/Final Answer: (two-pass).
    timing: str = "p0"

    # KV-group scaling config for circuit-level intervention.
    # When set, _generate_step uses KVGroupScalingHook instead of SteeringHook.
    # Dict with keys: layer (int), kv_group (int), alpha (float).
    kv_group_config: Optional[Dict[str, Any]] = None

    # Override the steering layer (default: use AgentConfig.layer).
    # Used by necessity test to steer at L19 (mlp_L20 input) instead of L20 output.
    steer_layer: Optional[int] = None

    # Cross-axis (rank-1 dynamic) config for Arm B of the reconnection sweep.
    # When set, _generate_step uses CrossAxisHook in place of SteeringHook.
    # Dict with keys: layer (int), u_in (np.ndarray), u_out (np.ndarray), alpha (float).
    cross_axis_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class Policy(ABC):
    """Abstract base class for steering policies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Policy name for logging."""
        pass
    
    @abstractmethod
    def decide(
        self,
        margin_fn: Callable[[float], float],
        target_side: str,
        hidden_rms: float,
        direction_rms: float
    ) -> SteeringDecision:
        """
        Decide steering strength.
        
        Args:
            margin_fn: Function that computes margin given rho
            target_side: "positive" (should adopt) or "negative" (should reject)
            hidden_rms: RMS of hidden states
            direction_rms: RMS of direction vector
            
        Returns:
            SteeringDecision with rho and alpha
        """
        pass


class BaselinePolicy(Policy):
    """No intervention (rho=0)."""
    
    @property
    def name(self) -> str:
        return "baseline"
    
    def decide(self, margin_fn, target_side, hidden_rms, direction_rms) -> SteeringDecision:
        m0 = margin_fn(0.0)
        return SteeringDecision(rho=0.0, alpha=0.0, policy_name=self.name, m_before=m0)


class FixedRhoPolicy(Policy):
    """Fixed rho value."""
    
    def __init__(self, rho: float, alpha_max: float = 2000.0):
        self.rho = rho
        self.alpha_max = alpha_max
    
    @property
    def name(self) -> str:
        return f"fixed_rho_{self.rho:+.2f}"
    
    def decide(self, margin_fn, target_side, hidden_rms, direction_rms) -> SteeringDecision:
        alpha = self.rho * (hidden_rms / direction_rms)
        alpha = np.clip(alpha, -self.alpha_max, self.alpha_max)
        return SteeringDecision(
            rho=self.rho,
            alpha=alpha,
            policy_name=self.name,
            details={"fixed_rho": self.rho}
        )


class JESPolicy(Policy):
    """Just-Enough Steering: adaptive rho based on margin slope."""
    
    def __init__(self, config: JESConfig = None, direction: np.ndarray = None):
        self.config = config or JESConfig()
        self.direction = direction
        self._controller = None
    
    @property
    def name(self) -> str:
        return "jes"
    
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
    
    def decide(self, margin_fn, target_side, hidden_rms, direction_rms) -> SteeringDecision:
        controller = self._get_controller(hidden_rms, direction_rms)
        result = controller.compute_steering(margin_fn, target_side)

        return SteeringDecision(
            rho=result.rho_used,
            alpha=result.alpha_used,
            policy_name=self.name,
            m_before=result.m_before,  # Cache baseline margin to avoid recomputation
            details=result.to_dict()
        )


class ForcedPolicy(Policy):
    """
    Forced adopt or reject policy.
    Uses large fixed rho to force decision.
    """
    
    def __init__(self, force_adopt: bool = True, rho_magnitude: float = 0.5,
                 alpha_max: float = 2000.0):
        self.force_adopt = force_adopt
        self.rho_magnitude = rho_magnitude
        self.alpha_max = alpha_max
    
    @property
    def name(self) -> str:
        return "force_adopt" if self.force_adopt else "force_reject"
    
    def decide(self, margin_fn, target_side, hidden_rms, direction_rms) -> SteeringDecision:
        """Force tool adoption/rejection in a direction-sign-agnostic way.

        We choose the rho sign that maximizes (force_adopt) or minimizes (force_reject)
        the action margin, so this works even if the saved direction has flipped sign.
        """
        m0 = margin_fn(0.0)

        rho_pos = float(self.rho_magnitude)
        rho_neg = -float(self.rho_magnitude)

        m_pos = margin_fn(rho_pos)
        m_neg = margin_fn(rho_neg)

        if self.force_adopt:
            rho = rho_pos if m_pos >= m_neg else rho_neg
        else:
            rho = rho_pos if m_pos <= m_neg else rho_neg

        alpha = rho * (hidden_rms / direction_rms)
        alpha_clipped = float(np.clip(alpha, -self.alpha_max, self.alpha_max))

        return SteeringDecision(
            rho=rho,
            alpha=alpha_clipped,
            policy_name=self.name,
            m_before=m0,  # Cache baseline margin
            details={
                "forced": self.force_adopt,
                "rho_magnitude": self.rho_magnitude,
                "rho_selected": rho,
                "margin_at_+rho": m_pos,
                "margin_at_-rho": m_neg,
                "alpha_raw": float(alpha),
                "alpha_used": float(alpha_clipped),
            }
        )


class RandomControlPolicy(Policy):
    """
    Random orthogonal direction control.
    Uses same JES/fixed-rho logic but with random direction.
    """
    
    def __init__(self, base_policy: Policy):
        self.base_policy = base_policy
    
    @property
    def name(self) -> str:
        return f"random_control_{self.base_policy.name}"
    
    def decide(self, margin_fn, target_side, hidden_rms, direction_rms) -> SteeringDecision:
        result = self.base_policy.decide(margin_fn, target_side, hidden_rms, direction_rms)
        result.policy_name = self.name
        result.details["is_random_control"] = True
        return result

