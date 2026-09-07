"""
actions.py
====================
The controller's action space and the ONLY logic allowed to turn an
action into an actual signal-program change.

ACTION_KEEP           -- keep the current stage (extend/hold its green).
ACTION_REQUEST_NEXT    -- request leaving the current stage for the next
                          one, subject to the constraints below.

Every requested transition is validated against signal_config.py before
it can reach SUMO. This module never:
  * jumps to an arbitrary phase index
  * skips a required yellow/transition phase
  * switches two conflicting greens on directly
  * invents a signal state string
  * violates MIN_GREEN_S / a transition phase's fixed duration

If the current phase is a yellow/transition phase (1, 3, 5, or 7), no
action has any effect: those phases are NOT controllable and must run to
completion for their exact sq.net.xml duration before anything else can
happen. The action layer reflects this by returning EFFECT_NONE for any
action requested during a transition phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from signal_config import (
    MAX_GREEN_S,
    MIN_GREEN_S,
    PHASE_BY_INDEX,
    STAGE_INDICES,
    TRANSITION_AFTER_STAGE,
)

ACTION_KEEP = "KEEP"
ACTION_REQUEST_NEXT = "REQUEST_NEXT"
VALID_ACTIONS = (ACTION_KEEP, ACTION_REQUEST_NEXT)


class TransitionEffect(str, Enum):
    NONE = "NONE"                # nothing changes this step
    HOLD_STAGE = "HOLD_STAGE"    # stay in the current green stage
    BEGIN_TRANSITION = "BEGIN_TRANSITION"  # move into the mandatory yellow phase
    FORCE_TRANSITION_MAX_GREEN = "FORCE_TRANSITION_MAX_GREEN"  # safety cap hit


@dataclass(frozen=True)
class ResolvedAction:
    effect: TransitionEffect
    next_phase_index: "int | None"  # only set for the two TRANSITION effects
    reason: str


class IllegalActionError(ValueError):
    """Raised when the requested action, if honored literally, would
    produce an illegal signal transition. The interface should never let
    this reach SUMO -- see resolve_action()."""


def validate_action_name(action: str) -> None:
    if action not in VALID_ACTIONS:
        raise IllegalActionError(
            f"Unknown action {action!r}; must be one of {VALID_ACTIONS}."
        )


def resolve_action(action: str, current_phase: int, phase_elapsed_s: float) -> ResolvedAction:
    """The single choke point between a requested action and any actual
    signal change. Returns what is ALLOWED to happen this step -- the
    caller (sumo_interface.py) must apply exactly this, nothing else.
    """
    validate_action_name(action)

    phase_def = PHASE_BY_INDEX.get(current_phase)
    if phase_def is None:
        raise IllegalActionError(f"current_phase={current_phase} is not a known phase index.")

    # Transition (yellow) phases are never controllable -- any action is
    # a no-op until the phase's own fixed sq.net.xml duration elapses.
    # Advancing it early would skip a required yellow phase; that is
    # exactly what this module exists to prevent.
    if phase_def.is_transition:
        return ResolvedAction(
            effect=TransitionEffect.NONE,
            next_phase_index=None,
            reason=(
                f"phase {current_phase} is a mandatory transition phase; "
                f"actions are ignored until it completes."
            ),
        )

    if current_phase not in STAGE_INDICES:
        raise IllegalActionError(
            f"current_phase={current_phase} is neither a known stage nor a "
            f"known transition phase."
        )

    min_green = MIN_GREEN_S[current_phase]
    max_green = MAX_GREEN_S[current_phase]

    # Safety cap: regardless of the requested action, a stage may never
    # run longer than MAX_GREEN_S (an ASTRID prototype assumption, see
    # signal_config.py) -- this prevents the placeholder/RL policy from
    # starving the other approach indefinitely.
    if phase_elapsed_s >= max_green:
        return ResolvedAction(
            effect=TransitionEffect.FORCE_TRANSITION_MAX_GREEN,
            next_phase_index=TRANSITION_AFTER_STAGE[current_phase],
            reason=f"phase_elapsed_s={phase_elapsed_s:.1f} >= MAX_GREEN_S={max_green:.1f}; forcing transition.",
        )

    if action == ACTION_KEEP:
        return ResolvedAction(
            effect=TransitionEffect.HOLD_STAGE,
            next_phase_index=None,
            reason="ACTION_KEEP: holding current stage.",
        )

    # action == ACTION_REQUEST_NEXT
    if phase_elapsed_s < min_green:
        # Requesting next before the minimum green has elapsed is not an
        # error -- it is simply not honored yet, matching real signal
        # controllers' minimum-green guarantees. This keeps the action
        # layer strict without making the policy's own request illegal.
        return ResolvedAction(
            effect=TransitionEffect.HOLD_STAGE,
            next_phase_index=None,
            reason=(
                f"ACTION_REQUEST_NEXT denied: phase_elapsed_s={phase_elapsed_s:.1f} "
                f"< MIN_GREEN_S={min_green:.1f}; holding stage instead."
            ),
        )

    return ResolvedAction(
        effect=TransitionEffect.BEGIN_TRANSITION,
        next_phase_index=TRANSITION_AFTER_STAGE[current_phase],
        reason="ACTION_REQUEST_NEXT honored: beginning mandatory transition phase.",
    )