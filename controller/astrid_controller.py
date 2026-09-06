"""
astrid_controller.py
====================
PLACEHOLDER / TEST POLICY -- NOT THE FINAL ASTRID CONTROLLER.

This is a simple, fully deterministic function of ControllerState. Its
ONLY purpose is to verify the full control pipeline (state -> policy ->
action -> validated transition -> SUMO) end to end. It contains no
learning, no neural network, and is not tuned against TEST/OOD.

Because controller_state.py's PlaceholderQueueEstimator currently
returns None for every approach (see that module's docstring), this
policy cannot yet make a queue-informed decision -- and it does not
pretend to. When every estimated_queue_m value is None, it falls back
to requesting the next stage as soon as the minimum green has elapsed,
which is the only safe, information-respecting default: it degenerates
to running the stages back-to-back at their sq.net.xml-derived minimum
durations, rather than inventing a queue-based decision from no data.

Once a real online queue estimate is wired into controller_state.py,
this function's `if any(v is not None ...)` branch is where a genuine
(still non-learned) heuristic -- e.g. "extend green while the opposing
stage's estimated queue is below some threshold" -- would go, ahead of
the eventual neural/RL policy described in astrid_controller.py's
module docstring and the project README.
"""

from __future__ import annotations

from actions import ACTION_KEEP, ACTION_REQUEST_NEXT
from controller_state import ControllerState
from signal_config import MIN_GREEN_S


def placeholder_policy(state: ControllerState) -> str:
    """PLACEHOLDER / TEST POLICY -- NOT THE FINAL ASTRID CONTROLLER.

    Deterministic, stateless function of ControllerState only. Returns
    one of ACTION_KEEP / ACTION_REQUEST_NEXT.
    """
    min_green = MIN_GREEN_S.get(state.current_phase)
    if min_green is None:
        # Currently in a transition phase; any action is a no-op anyway
        # (see actions.resolve_action) -- return a fixed, harmless value.
        return ACTION_KEEP

    have_real_queue_estimate = any(v is not None for v in state.estimated_queue_m.values())

    if not have_real_queue_estimate:
        # No real information yet (see PlaceholderQueueEstimator) -- the
        # only honest default is to not pretend to optimize anything.
        # Request the next stage as soon as the minimum green is
        # satisfied; this is intentionally equivalent to a simple
        # minimum-green round robin, not a smart decision.
        return ACTION_REQUEST_NEXT if state.phase_elapsed_s >= min_green else ACTION_KEEP

    # Reachable only once a real (non-placeholder) QueueEstimator is
    # wired in. Left as a documented hook, not implemented: a genuine
    # queue-aware rule belongs here, followed eventually by the real
    # neural/RL policy -- see README.md "Future neural/RL design".
    return ACTION_REQUEST_NEXT if state.phase_elapsed_s >= min_green else ACTION_KEEP