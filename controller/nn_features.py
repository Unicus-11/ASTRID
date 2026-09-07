"""
nn_features.py
====================
Turns a controller_state.ControllerState into a fixed-size, normalized
feature vector for the NN controller. This is the ONLY place that
touches ControllerState for the NN path -- it does not read anything
from SUMO/TraCI directly, and it does not change controller_state.py,
signal_config.py, or the HGB queue-estimation pipeline in any way.

INPUT CONTRACT
--------------
Exactly the fields ControllerState already exposes:
  - estimated_queue_m[edge] for edge in ("1i", "2i", "3i", "4i")
    (west, east, south, north -- see signal_config.APPROACH_DIRECTION)
  - current_phase
  - phase_elapsed_s

None of these are ground truth -- see controller_state.py's own
docstring for why. This module adds no additional inputs and reads no
additional SUMO state. It never imports sumo_interface.py or traci.

HANDLING MISSING QUEUE ESTIMATES (None)
----------------------------------------
Today's PlaceholderQueueEstimator returns None for every approach (see
controller_state.py). A numeric NN needs numbers, so:
  - a missing estimate is encoded as 0.0 for its value slot
  - a separate "have_estimate" bit (1.0 / 0.0) is appended per approach,
    so the network can tell "no information" apart from "queue is
    genuinely zero" -- 0.0 alone would silently look like an empty
    queue, which controller_state.py's docstring explicitly says must
    never be faked.
Once a real online HGB estimator is wired into controller_state.py (see
that file's docstring -- out of scope for this change), these bits
simply start turning on with no code change needed here, and the NN
starts seeing real numbers instead of the 0.0 placeholder.
"""

from __future__ import annotations

import numpy as np

from controller_state import ControllerState
from signal_config import APPROACH_EDGES, STAGE_INDICES

# Normalization constants -- prototype-scale choices, documented so they
# are easy to revisit once real queue magnitudes are observed from the
# HGB pipeline. Not sourced from SUMO; purely for keeping NN inputs in a
# sane, bounded numeric range.
QUEUE_NORM_M = 200.0          # treat ~200m as a "long" queue for scaling
PHASE_ELAPSED_NORM_S = 60.0   # treat ~60s as a "long" elapsed time for scaling

# Fixed feature ordering. nn_model.py's FEATURE_DIM must agree with this
# list's length; anything that reads a feature vector by index must agree
# with this order.
FEATURE_NAMES = (
    "queue_1i_west", "have_1i",
    "queue_2i_east", "have_2i",
    "queue_3i_south", "have_3i",
    "queue_4i_north", "have_4i",
    "phase_is_NS_through",   # phase 0
    "phase_is_NS_left",      # phase 2
    "phase_is_EW_through",   # phase 4
    "phase_is_EW_left",      # phase 6
    "phase_is_transition",   # phase in {1, 3, 5, 7}
    "phase_elapsed_norm",
)
FEATURE_DIM = len(FEATURE_NAMES)


def build_nn_features(state: ControllerState) -> np.ndarray:
    """ControllerState -> shape-(FEATURE_DIM,) float32 vector.
    Pure function: no side effects, no SUMO/TraCI access, no mutation
    of `state`."""
    values = []
    for edge in APPROACH_EDGES:  # fixed order: ("1i", "2i", "3i", "4i")
        q = state.estimated_queue_m.get(edge)
        if q is None:
            values.append(0.0)
            values.append(0.0)
        else:
            values.append(float(min(q, QUEUE_NORM_M)) / QUEUE_NORM_M)
            values.append(1.0)

    values.extend(
        [
            1.0 if state.current_phase == 0 else 0.0,
            1.0 if state.current_phase == 2 else 0.0,
            1.0 if state.current_phase == 4 else 0.0,
            1.0 if state.current_phase == 6 else 0.0,
            1.0 if state.current_phase not in STAGE_INDICES else 0.0,
        ]
    )

    values.append(min(state.phase_elapsed_s, PHASE_ELAPSED_NORM_S) / PHASE_ELAPSED_NORM_S)

    vec = np.asarray(values, dtype=np.float32)
    assert vec.shape == (FEATURE_DIM,), f"feature vector shape mismatch: {vec.shape}"
    return vec