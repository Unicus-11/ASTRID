"""
reward.py
====================
Absolute estimated-queue cost remains the primary objective (per-step,
summed across approaches, never dropped to zero for a None estimate).
A genuine policy-requested BEGIN_TRANSITION carries a configurable
switch penalty. FORCE_TRANSITION_MAX_GREEN (the safety layer, not the
policy, decided to switch) is tracked separately and by default
penalized far less -- conflating the two teaches the policy to avoid
letting the safety cap ever fire rather than to control traffic well.

Optional additive potential-based shaping (Ng et al. 1999 style,
F = gamma*Phi(s') - Phi(s) with Phi(s) = -w_shaping*queue(s)) can be
layered on top via RewardConfig.w_shaping. It is OFF by default
(w_shaping=0.0): the absolute queue cost stays the sole primary signal
unless explicitly enabled, and even when enabled it never replaces the
absolute term, only adds a denser secondary one.

No delay/waiting term (D_t) is included: ControllerState carries no
legitimate delay signal today (only estimated_queue_m), and this module
does not fabricate one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from actions import TransitionEffect

BEGIN_TRANSITION = TransitionEffect.BEGIN_TRANSITION.value
FORCE_TRANSITION_MAX_GREEN = TransitionEffect.FORCE_TRANSITION_MAX_GREEN.value


@dataclass(frozen=True)
class RewardConfig:
    w_queue: float = 1.0
    w_switch_requested: float = 15.0   # genuine policy-chosen BEGIN_TRANSITION
    w_switch_forced: float = 1.0       # safety-cap FORCE_TRANSITION_MAX_GREEN; kept small on purpose
    enable_potential_shaping: bool = False
    w_shaping: float = 0.0
    shaping_gamma: float = 0.99        # should match the training discount factor when enabled


@dataclass
class StepRewardComponents:
    queue_m: float
    queue_cost: float
    requested_switch_cost: float
    forced_switch_cost: float
    is_requested_transition: bool
    is_forced_transition: bool
    shaping_reward: float
    reward: float  # = -(queue_cost + requested_switch_cost + forced_switch_cost) + shaping_reward


def total_estimated_queue_m(estimated_queue_m: Dict[str, Optional[float]]) -> float:
    """None (no estimate yet) counts as 0.0 -- the physical minimum, not
    a dropped term."""
    return sum(v if v is not None else 0.0 for v in estimated_queue_m.values())


def compute_step_components(
    estimated_queue_m: Dict[str, Optional[float]],
    resolved_action_value: str,
    prev_total_queue_m: Optional[float],
    config: RewardConfig = RewardConfig(),
) -> StepRewardComponents:
    """One simulation step's reward, broken into components for logging.
    `prev_total_queue_m` is this same quantity from the PRECEDING step
    (None on the first step of an episode) -- only used if shaping is
    enabled."""
    queue_m = total_estimated_queue_m(estimated_queue_m)
    queue_cost = config.w_queue * queue_m

    is_requested = resolved_action_value == BEGIN_TRANSITION
    is_forced = resolved_action_value == FORCE_TRANSITION_MAX_GREEN
    requested_switch_cost = config.w_switch_requested if is_requested else 0.0
    forced_switch_cost = config.w_switch_forced if is_forced else 0.0

    shaping_reward = 0.0
    if config.enable_potential_shaping and prev_total_queue_m is not None:
        # F(s,s') = gamma*Phi(next) - Phi(curr), Phi(s) = -w_shaping*queue(s)
        shaping_reward = config.w_shaping * (prev_total_queue_m - config.shaping_gamma * queue_m)

    total_cost = queue_cost + requested_switch_cost + forced_switch_cost
    reward = -total_cost + shaping_reward

    return StepRewardComponents(
        queue_m=queue_m,
        queue_cost=queue_cost,
        requested_switch_cost=requested_switch_cost,
        forced_switch_cost=forced_switch_cost,
        is_requested_transition=is_requested,
        is_forced_transition=is_forced,
        shaping_reward=shaping_reward,
        reward=reward,
    )