"""
controller_state.py
====================
The ASTRID controller's own view of the world -- small, interpretable,
and restricted to information a real deployment could actually have.

--------------------------------------------------------------------------
WHAT IS INCLUDED, AND WHY EACH FIELD IS SAFE
--------------------------------------------------------------------------
estimated_queue_m : Dict[str, float | None], one entry per approach edge
    Intended source: the existing ASTRID pipeline
    (camera + GPS -> feature_builder.py -> HistGradientBoosting queue
    estimator), i.e. exactly the "estimated queue" box in this project's
    architecture diagram. That pipeline is CSV/offline (it reads whole
    scenario files after a run), not a live per-simulation-step API, so
    wiring true HGB inference into a per-TraCI-step loop is a real
    online-inference integration task -- explicitly NOT done in this
    foundation stage (see QueueEstimator below). What IS enforced here is
    the boundary: this field is never filled from any SUMO ground-truth
    queue/halting/occupancy call. See QueueEstimator's docstring.

current_phase : int
    The controller's own actuator state (which SUMO signal phase is
    currently active). This is not privileged information -- a real
    signal controller always knows its own current phase.

phase_elapsed_s : float
    Seconds since the current phase began. Derived from the simulation
    clock and the controller's own last phase-change timestamp -- again
    self-knowledge of the actuator, not sensed traffic information.

--------------------------------------------------------------------------
INVESTIGATED BUT NOT YET INCLUDED (documented per PART 5's instruction)
--------------------------------------------------------------------------
queue growth/change
    Safe in principle -- ASTRID's Layer 2 feature pipeline already
    computes visible_queue_length_m_change_30s from camera history only.
    Not included here because it depends on the SAME not-yet-built
    online feature pipeline as estimated_queue_m; adding it before that
    exists would mean fabricating it.

estimated flow / arrivals (observed_flow_veh_per_hour)
    Safe in principle -- derived in feature_builder.py purely from
    camera-observable count/speed, no ground truth. Same reason as
    above: no online feature pipeline exists yet to compute it live.

observable speed (visible_mean_speed_mps / probe_mean_speed_mps)
    Safe in principle -- both are direct camera/GPS sensor outputs. Same
    reason as above.

probe_count
    Safe in principle -- a direct GPS sensor output (count of GPS-
    equipped vehicles observed), not a ground-truth quantity. Same
    reason as above.

All four of the above are DEFERRED, not rejected: they require an online
(per-TraCI-step) version of camera_simulator.py / gps_simulator.py /
observation_assembler.py / feature_builder.py that does not exist yet.
Building that is future work, not part of this foundation stage.

--------------------------------------------------------------------------
WHAT IS DELIBERATELY EXCLUDED, PERMANENTLY (not just deferred)
--------------------------------------------------------------------------
true_queue_length_m, exact vehicle positions, future arrivals, future
queue length, or any other SUMO/TraCI call that reads privileged
simulation ground truth. See sumo_interface.py and reward.py for where
ground truth IS legitimately used (reward calculation, evaluation,
diagnostics only) -- never here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol

from signal_config import APPROACH_EDGES


@dataclass(frozen=True)
class ControllerState:
    """Everything the ASTRID controller (placeholder or, eventually, the
    real RL policy) is allowed to see when choosing an action."""

    estimated_queue_m: Dict[str, Optional[float]]
    current_phase: int
    phase_elapsed_s: float

    def summary(self) -> str:
        queues = ", ".join(
            f"{edge}={'NA' if v is None else f'{v:.1f}m'}"
            for edge, v in self.estimated_queue_m.items()
        )
        return (
            f"phase={self.current_phase} elapsed={self.phase_elapsed_s:.1f}s "
            f"queues[{queues}]"
        )


class QueueEstimator(Protocol):
    """Abstraction over 'wherever estimated_queue_m comes from', so the
    controller and the sumo_interface loop never need to know whether
    that's a real online HGB pipeline (future work) or this stage's
    placeholder."""

    def estimate(self) -> Dict[str, Optional[float]]:
        ...


class PlaceholderQueueEstimator:
    """PLACEHOLDER / TEST ESTIMATOR -- NOT THE REAL HGB PIPELINE.

    Returns a fixed, non-informative value (None) for every approach.
    This exists purely so the closed-loop wiring (state -> policy ->
    action -> validated transition -> SUMO) can be exercised and
    verified in this foundation stage WITHOUT:
      (a) wiring a real online feature+HGB pipeline (out of scope here,
          and not something to build casually inside a controller
          module), or
      (b) reaching for a SUMO ground-truth queue/halting count as a
          shortcut, which would violate the information boundary this
          whole module exists to enforce.

    Returning None (never 0.0, never a fabricated number) makes it
    visually and programmatically obvious downstream that no real queue
    estimate is present yet -- see astrid_controller.py's placeholder
    policy, which treats None as "no information" rather than "queue is
    empty".
    """

    def estimate(self) -> Dict[str, Optional[float]]:
        return {edge: None for edge in APPROACH_EDGES}


def build_controller_state(
    queue_estimator: QueueEstimator,
    current_phase: int,
    phase_elapsed_s: float,
) -> ControllerState:
    return ControllerState(
        estimated_queue_m=queue_estimator.estimate(),
        current_phase=current_phase,
        phase_elapsed_s=phase_elapsed_s,
    )