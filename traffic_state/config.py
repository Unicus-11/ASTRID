"""
============================================================
ASTRID — LAYER 1 CONFIGURATION
============================================================

Purpose
-------
Central configuration for the Layer 1 traffic-state estimation
pipeline.

This file contains MODEL RULES and CONSTANTS.

It does NOT contain scenario-specific data.

Scenario-specific data comes from:

    datasets/scenario_XXXX/sensor_dataset.json
============================================================
"""


# ============================================================
# SIMULATION
# ============================================================

SIMULATION_DURATION = 3600

TIME_STEP = 1.0


# ============================================================
# INTERSECTION
# ============================================================

APPROACHES = (
    "north",
    "south",
    "east",
    "west",
)


# ============================================================
# VEHICLE TYPES
# ============================================================

VEHICLE_TYPES = (
    "bike",
    "car",
    "bus",
    "hgv",
)


# ============================================================
# QUEUE DEFINITION
# ============================================================

# A vehicle travelling below this speed is considered queued.
#
# This follows the definition already used in the dataset
# generation / validation logic.

QUEUE_SPEED_THRESHOLD = 0.5       # m/s


# ============================================================
# SENSOR TYPES
# ============================================================

SENSORS = (
    "gps",
    "cctv",
)


# ============================================================
# SHOCKWAVE / QUEUE ESTIMATION
# ============================================================

# These will be used when we implement the shockwave model.
#
# Do not put scenario-specific values here.

MIN_SPEED = 0.0


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_FILENAME = "estimated_state.json"