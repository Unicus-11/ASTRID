"""
rule_based_policy.py
======================
ASTRID Prototype -- deterministic, threshold-based control policy.

Drop-in replacement for nn_controller.nn_policy: same call interface
`policy(state: ControllerState) -> str`, same safety boundary (never
calls TraCI, only ever returns ACTION_KEEP / ACTION_REQUEST_NEXT, which
still passes through actions.resolve_action() as the sole legality
layer -- unchanged).

WHY THIS EXISTS
----------------
For the Monday prototype deadline: uses the already-validated HGB
estimated queues directly, with zero training, zero convergence risk.
The ML contribution (sensors -> HGB -> estimated_queue_m) is unchanged
and fully working; only the decision-on-top-of-it is simplified from
"learned policy" to "threshold rule" for this submission. RL/learned
control remains explicitly future work.

RULE
-----
While in a controllable green stage, after MIN_GREEN_S has elapsed:
    if max(estimated queue on the RED approaches)
       > max(estimated queue on the currently-GREEN approaches) + MARGIN_M
    -> request an early transition (let the more congested direction go)
    else
    -> keep the current stage

This is intentionally the simplest defensible rule: it only uses
information already flowing through the existing, verified pipeline
(ControllerState.estimated_queue_m), and it only ever acts within the
same MIN_GREEN_S/MAX_GREEN_S window the NN policy was already
constrained to -- actions.py's safety/legality enforcement is
completely unchanged and still has final say.

*** IMPORTANT, UNRESOLVED MAPPING CAVEAT (same one flagged earlier in
this project) ***
PHASE_APPROACH_EDGES below encodes signal_config.py's OWN stated
phase-to-direction mapping (phase 0,2 -> NS edges; phase 4,6 -> EW
edges), which is DIFFERENT from feature_builder.py's mapping (used
only to keep the online HGB features statistically consistent with
what the model was trained on). For control decisions -- unlike HGB
feature construction -- this file must use the REAL, physical
mapping, since we're now describing "which edges are actually
flowing", not "what the HGB was trained to call it".

**You must confirm this against the real signal_config.py before
trusting this file** -- if signal_config.py's actual dict differs from
what's written below, edit PHASE_APPROACH_EDGES to match it exactly.
Getting this backwards would make the rule optimize for the wrong
approach's queue.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from actions import ACTION_KEEP, ACTION_REQUEST_NEXT
from controller_state import ControllerState
from signal_config import APPROACH_EDGES, MIN_GREEN_S, STAGE_INDICES

# CONFIRM against real signal_config.py -- see module docstring caveat.
# As stated in this project's own phase-mapping investigation:
#   phase 0,2 -> NS approaches (edges 3i, 4i)
#   phase 4,6 -> EW approaches (edges 1i, 2i)
PHASE_APPROACH_EDGES: Dict[int, List[str]] = {
    0: ["3i", "4i"],
    2: ["3i", "4i"],
    4: ["1i", "2i"],
    6: ["1i", "2i"],
}

# How much bigger the red-side queue must be than the green-side queue
# before we bother requesting an early transition. Prevents flip-flopping
# on noisy HGB estimates for marginal differences. Tune this after
# watching a few runs -- there is no "correct" value from first
# principles, it trades off responsiveness vs. stability.
MARGIN_M = 15.0


def _max_queue(state: ControllerState, edges: List[str]) -> float:
    """Treats a missing/None estimate as 0.0 -- conservative: an
    unknown queue never itself triggers an early transition."""
    values = [state.estimated_queue_m.get(e) for e in edges]
    return max((v for v in values if v is not None), default=0.0)


def rule_based_policy(state: ControllerState) -> str:
    """Same interface as nn_controller.NNPolicy.__call__ /
    astrid_controller.placeholder_policy: policy(state) -> action
    string. Stateless -- safe to call every step; actions.py's own
    MIN_GREEN_S gating already prevents any early transition from being
    honored before it's legal."""

    # Transition (yellow) phases: never controllable, same as NNPolicy.
    if state.current_phase not in STAGE_INDICES:
        return ACTION_KEEP

    if state.phase_elapsed_s < MIN_GREEN_S[state.current_phase]:
        # Not legal to request yet anyway -- actions.py would just hold
        # the stage, but no need to even evaluate the rule.
        return ACTION_KEEP

    green_edges = PHASE_APPROACH_EDGES.get(state.current_phase)
    if green_edges is None:
        # Unknown phase index -- fail safe to KEEP rather than guess.
        return ACTION_KEEP

    red_edges = [e for e in APPROACH_EDGES if e not in green_edges]

    green_queue = _max_queue(state, green_edges)
    red_queue = _max_queue(state, red_edges)

    if red_queue > green_queue + MARGIN_M:
        return ACTION_REQUEST_NEXT

    return ACTION_KEEP