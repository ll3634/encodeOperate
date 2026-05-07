#!/usr/bin/env python3
"""
JES (Just-Enough Steering) algorithm for E2E agent.
Adapted from jes_intervention.py for real-time agent use.
"""

import numpy as np
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field


@dataclass
class JESConfig:
    """JES hyperparameters."""
    tau: float = 0.2          # Target margin threshold
    eps: float = 0.02         # Probe epsilon for slope estimation
    max_rho: float = 0.25     # Maximum |rho| allowed
    slope_min: float = 0.05   # Minimum |slope| for stability
    alpha_max: float = 8.0    # Maximum |alpha| (safety clamp; prevents format corruption)


@dataclass
class JESResult:
    """Result of JES computation."""
    rho_used: float = 0.0
    alpha_used: float = 0.0
    slope: float = 0.0
    rho_star_raw: float = float('nan')
    achieved: bool = False
    clipped: bool = False
    saturated: bool = False
    unstable: bool = False
    slope_small: bool = False
    m_before: float = 0.0
    m_after: float = 0.0
    m_target: float = 0.0
    already_satisfied: bool = False
    eps_effective: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "rho_used": self.rho_used,
            "alpha_used": self.alpha_used,
            "slope": self.slope,
            "rho_star_raw": self.rho_star_raw if np.isfinite(self.rho_star_raw) else None,
            "achieved": self.achieved,
            "clipped": self.clipped,
            "saturated": self.saturated,
            "unstable": self.unstable,
            "slope_small": self.slope_small,
            "m_before": self.m_before,
            "m_after": self.m_after,
            "m_target": self.m_target,
            "already_satisfied": self.already_satisfied,
            "eps_effective": self.eps_effective,
        }


def compute_rho_star(
    m0: float,
    m_plus: float,
    m_minus: float,
    eps: float,
    m_target: float,
    rho_max: float,
    slope_min: float
) -> Dict[str, Any]:
    """
    Compute the just-enough rho to reach m_target.
    
    Args:
        m0: baseline margin
        m_plus: margin at rho = +eps
        m_minus: margin at rho = -eps
        eps: probe epsilon value
        m_target: target margin
        rho_max: maximum allowed |rho|
        slope_min: minimum |slope| to consider stable
        
    Returns:
        dict with slope, rho_star_raw, rho_used, clipped, slope_small, unstable
    """
    # Compute slope via central difference
    slope = (m_plus - m_minus) / (2 * eps)
    
    result = {
        "slope": slope,
        "rho_star_raw": float('nan'),
        "rho_used": 0.0,
        "clipped": False,
        "slope_small": False,
        "unstable": False,
    }
    
    # Check for invalid slope
    if not np.isfinite(slope):
        result["unstable"] = True
        return result
    
    # Check for slope too small
    if abs(slope) < slope_min:
        result["slope_small"] = True
        result["unstable"] = True
        return result
    
    # Compute raw rho*
    rho_star_raw = (m_target - m0) / slope
    result["rho_star_raw"] = rho_star_raw
    
    # Check for invalid rho_star
    if not np.isfinite(rho_star_raw):
        result["unstable"] = True
        return result
    
    # Clip to [-rho_max, +rho_max]
    rho_used = np.clip(rho_star_raw, -rho_max, rho_max)
    result["rho_used"] = float(rho_used)
    result["clipped"] = bool(abs(rho_star_raw) > rho_max)
    
    return result


class JESController:
    """
    JES controller for real-time agent steering.
    
    Usage:
        controller = JESController(config, direction, hidden_rms, direction_rms)
        
        # At each decision point:
        result = controller.compute_steering(
            margin_fn=lambda rho: compute_margin_at_rho(rho),
            target_side="positive"  # or "negative"
        )
        alpha = result.alpha_used
    """
    
    def __init__(
        self,
        config: JESConfig,
        direction: np.ndarray,
        hidden_rms: float,
        direction_rms: Optional[float] = None
    ):
        self.config = config
        self.direction = direction
        self.hidden_rms = hidden_rms
        self.direction_rms = direction_rms or float(np.sqrt(np.mean(direction ** 2)))
        
    def rho_to_alpha(self, rho: float) -> float:
        """Convert rho to alpha using RMS normalization."""
        alpha = rho * (self.hidden_rms / self.direction_rms)
        return np.clip(alpha, -self.config.alpha_max, self.config.alpha_max)
    
    def compute_steering(
        self,
        margin_fn: Callable[[float], float],
        target_side: str = "positive"
    ) -> JESResult:
        """
        Compute JES steering for current decision point.
        
        Args:
            margin_fn: Function that computes margin given rho value
            target_side: "positive" (want margin > tau) or "negative" (want margin < -tau)
            
        Returns:
            JESResult with computed rho and alpha
        """
        cfg = self.config
        result = JESResult()
        
        # Determine target margin
        if target_side == "positive":
            m_target = cfg.tau
        else:
            m_target = -cfg.tau
        result.m_target = m_target
        
        # Get baseline margin
        m0 = margin_fn(0.0)
        result.m_before = m0

        # Check if already satisfied
        if target_side == "positive" and m0 >= m_target:
            result.already_satisfied = True
            result.achieved = True
            result.m_after = m0
            return result
        elif target_side == "negative" and m0 <= m_target:
            result.already_satisfied = True
            result.achieved = True
            result.m_after = m0
            return result

        # Probe at ±eps (with adaptive fallback if slope is near-zero).
        #
        # Motivation: on some tasks/models, the action margin can be locally
        # *even* around rho=0 (first derivative ~ 0) while still changing at
        # moderate |rho|. In that case a tiny eps yields slope≈0 and JES would
        # incorrectly disable steering.
        eps_candidates = [float(cfg.eps)]
        # Geometric backoff, clamped to max_rho.
        for mult in (5.0, 25.0):
            eps_candidates.append(float(min(cfg.max_rho, cfg.eps * mult)))
        # De-dup while preserving order
        seen = set()
        eps_candidates = [e for e in eps_candidates if (e > 0 and not (e in seen or seen.add(e)))]

        rho_result = None
        for eps_try in eps_candidates:
            m_plus = margin_fn(eps_try)
            m_minus = margin_fn(-eps_try)
            trial = compute_rho_star(
                m0, m_plus, m_minus, eps_try,
                m_target, cfg.max_rho, cfg.slope_min
            )
            # Accept the first stable slope.
            if (not trial.get("unstable", False)) and np.isfinite(trial.get("slope", float("nan"))):
                rho_result = trial
                result.eps_effective = float(eps_try)
                break

        # If all probes unstable, fall back to no steering.
        if rho_result is None:
            result.slope = 0.0
            result.rho_star_raw = float('nan')
            result.rho_used = 0.0
            result.clipped = False
            result.slope_small = True
            result.unstable = True
            result.m_after = m0
            result.eps_effective = float(eps_candidates[0]) if eps_candidates else float(cfg.eps)
            return result

        result.slope = rho_result["slope"]
        result.rho_star_raw = rho_result["rho_star_raw"]
        result.rho_used = rho_result["rho_used"]
        result.clipped = rho_result["clipped"]
        result.slope_small = rho_result["slope_small"]
        result.unstable = rho_result["unstable"]

        if result.unstable:
            result.m_after = m0
            return result

        # Compute alpha
        result.alpha_used = self.rho_to_alpha(result.rho_used)

        # Get final margin (optional - can skip for efficiency)
        result.m_after = margin_fn(result.rho_used)

        # Check if achieved
        if target_side == "positive":
            result.achieved = result.m_after >= m_target
        else:
            result.achieved = result.m_after <= m_target

        # Check if saturated (clipped but not achieved)
        result.saturated = result.clipped and not result.achieved

        return result


def run_self_test():
    """Self-test for JES functions."""
    print("Running JES self-tests...")

    # Test 1: compute_rho_star normal case
    result = compute_rho_star(
        m0=1.0, m_plus=1.2, m_minus=0.8, eps=0.02,
        m_target=1.5, rho_max=0.25, slope_min=0.05
    )
    assert abs(result["slope"] - 10.0) < 1e-6
    assert abs(result["rho_used"] - 0.05) < 1e-6
    assert not result["clipped"]
    print("  Test 1 (normal): PASSED")

    # Test 2: clipping
    result = compute_rho_star(
        m0=0.0, m_plus=0.2, m_minus=-0.2, eps=0.02,
        m_target=5.0, rho_max=0.25, slope_min=0.05
    )
    assert result["clipped"]
    assert abs(result["rho_used"] - 0.25) < 1e-6
    print("  Test 2 (clipping): PASSED")

    # Test 3: slope too small
    result = compute_rho_star(
        m0=0.0, m_plus=0.0008, m_minus=-0.0008, eps=0.02,
        m_target=1.0, rho_max=0.25, slope_min=0.05
    )
    assert result["slope_small"]
    assert result["unstable"]
    print("  Test 3 (slope small): PASSED")

    print("All JES self-tests PASSED!")


if __name__ == "__main__":
    run_self_test()

