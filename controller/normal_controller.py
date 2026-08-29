"""
============================================================
ASTRID NORMAL CONTROLLER
============================================================

Purpose
-------
Apply the normal Webster-based signal controller to the
CURRENT SUMO simulation.

Pipeline
--------

scenario.json
      |
      v
scenario_builder.py
      |
      v
SUMO route files
      |
      v
state_extractor.py
      |
      +---- normal_controller.py
      |
      +---- sensor_simulator.py
      |
      v
sensor_dataset.json


IMPORTANT
---------
This module does NOT:

    - start SUMO
    - create vehicles
    - create routes
    - run the simulation
    - collect sensor observations
    - create datasets

SUMO is already running when this controller is called.

The controller only calculates and applies the normal
signal timing for the selected scenario.

The scenario format is:

{
    "name": "scenario_0058",
    "seed": 1058,
    "demand": "high",
    "demand_rate": 2018,
    ...
}

Therefore:

    scenario["demand_rate"]

is the authoritative traffic demand.

The controller does NOT use a separate hard-coded demand
table.

============================================================
"""

import traci


# ============================================================
# CONSTANTS
# ============================================================

TLS_ID = "0"

# Webster lost time per cycle.
#
# This represents the total non-effective time caused by
# signal changes/intergreen periods.
#
# The existing ASTRID model uses:
#
#     L = 14 seconds
#
LOST_TIME = 14.0

# Saturation flow per lane (veh/hour).
SATURATION_FLOW_PER_LANE = 525.0

# Number of effective lanes serving each direction group.
LANES_NS = 6.0
LANES_EW = 6.0

# Minimum and maximum cycle length.
MIN_CYCLE = 60.0
MAX_CYCLE = 120.0

# Webster cycle calculation constant.
WEBSTER_NUMERATOR = 5.0

# Minimum yellow/intergreen phase duration used by the
# existing signal program.
CHANGE_PHASE_DURATION = 3.0


# ============================================================
# DEMAND
# ============================================================

def calculate_directional_demand(
    scenario: dict,
) -> tuple[float, float]:
    """
    Convert total scenario demand into NS and EW demand.

    Example:

        demand_rate = 2000 veh/h

        north = 0.25
        south = 0.25
        east  = 0.30
        west  = 0.20

    Then:

        NS = 2000 * (0.25 + 0.25)
           = 1000 veh/h

        EW = 2000 * (0.30 + 0.20)
           = 1000 veh/h
    """

    demand = float(
        scenario["demand_rate"]
    )

    approach = scenario[
        "approach_distribution"
    ]

    north = float(
        approach["north"]
    )

    south = float(
        approach["south"]
    )

    east = float(
        approach["east"]
    )

    west = float(
        approach["west"]
    )

    ns_demand = demand * (
        north + south
    )

    ew_demand = demand * (
        east + west
    )

    return (
        ns_demand,
        ew_demand,
    )


# ============================================================
# WEBSTER CALCULATION
# ============================================================

def calculate_webster_timing(
    ns_demand: float,
    ew_demand: float,
) -> dict:
    """
    Calculate the normal Webster signal timing.

    The calculation follows the existing ASTRID model:

        S_NS = 525 * 6
        S_EW = 525 * 6

        y_NS = NS / S_NS
        y_EW = EW / S_EW

        Y = y_NS + y_EW

    If Y < 1:

        C = (1.5L + 5) / (1 - Y)

    The result is then constrained to:

        60 <= C <= 120

    If Y >= 1:

        C = 120

    Green time is divided between NS and EW according to
    their relative critical flow ratios.
    """

    saturation_ns = (
        SATURATION_FLOW_PER_LANE
        * LANES_NS
    )

    saturation_ew = (
        SATURATION_FLOW_PER_LANE
        * LANES_EW
    )

    y_ns = (
        ns_demand
        / saturation_ns
    )

    y_ew = (
        ew_demand
        / saturation_ew
    )

    total_flow_ratio = (
        y_ns + y_ew
    )

    # --------------------------------------------------------
    # Cycle length
    # --------------------------------------------------------

    if total_flow_ratio < 1.0:

        cycle = (
            1.5 * LOST_TIME
            + WEBSTER_NUMERATOR
        ) / (
            1.0 - total_flow_ratio
        )

        cycle = max(
            MIN_CYCLE,
            min(
                cycle,
                MAX_CYCLE,
            ),
        )

    else:

        # Oversaturated demand.
        cycle = MAX_CYCLE

    # --------------------------------------------------------
    # Available green
    # --------------------------------------------------------

    green_total = (
        cycle
        - LOST_TIME
    )

    if green_total <= 0:

        raise ValueError(
            "Calculated green time is <= 0."
        )

    # --------------------------------------------------------
    # Green allocation
    # --------------------------------------------------------

    if total_flow_ratio > 0:

        green_ns = (
            green_total
            * (
                y_ns
                / total_flow_ratio
            )
        )

        green_ew = (
            green_total
            * (
                y_ew
                / total_flow_ratio
            )
        )

    else:

        # No demand.
        # Divide green time equally.
        green_ns = (
            green_total / 2.0
        )

        green_ew = (
            green_total / 2.0
        )

    return {

        "saturation_ns":
            saturation_ns,

        "saturation_ew":
            saturation_ew,

        "y_ns":
            y_ns,

        "y_ew":
            y_ew,

        "Y":
            total_flow_ratio,

        "cycle":
            cycle,

        "green_total":
            green_total,

        "green_ns":
            green_ns,

        "green_ew":
            green_ew,
    }


# ============================================================
# VALIDATE TRAFFIC LIGHT
# ============================================================

def validate_traffic_light():
    """
    Confirm that the expected traffic light exists and that
    the current program contains the eight phases expected by
    the ASTRID signal model.
    """

    traffic_lights = (
        traci.trafficlight.getIDList()
    )

    if TLS_ID not in traffic_lights:

        raise RuntimeError(
            f"Traffic light '{TLS_ID}' "
            f"was not found in SUMO.\n"
            f"Available traffic lights: "
            f"{traffic_lights}"
        )

    programs = (
        traci.trafficlight
        .getAllProgramLogics(TLS_ID)
    )

    if not programs:

        raise RuntimeError(
            f"No signal program found "
            f"for traffic light '{TLS_ID}'."
        )

    program = programs[0]

    phases = program.getPhases()

    if len(phases) < 8:

        raise RuntimeError(
            "The ASTRID normal controller expects "
            f"at least 8 signal phases, but SUMO "
            f"returned {len(phases)}."
        )

    return program, phases


# ============================================================
# APPLY SIGNAL TIMING
# ============================================================

def apply_signal_timing(
    green_ns: float,
    green_ew: float,
):
    """
    Modify the existing SUMO signal program.

    Phase structure assumed by the current ASTRID network:

        phase 0 -> EW green
        phase 1 -> change
        phase 2 -> EW green
        phase 3 -> change

        phase 4 -> NS green
        phase 5 -> change
        phase 6 -> NS green
        phase 7 -> change

    We preserve the phase states and only modify durations.
    """

    program, phases = (
        validate_traffic_light()
    )

    # --------------------------------------------------------
    # East-West
    # --------------------------------------------------------

    phases[0].duration = (
        green_ew
    )

    phases[1].duration = (
        CHANGE_PHASE_DURATION
    )

    phases[2].duration = (
        green_ew
    )

    phases[3].duration = (
        CHANGE_PHASE_DURATION
    )

    # --------------------------------------------------------
    # North-South
    # --------------------------------------------------------

    phases[4].duration = (
        green_ns
    )

    phases[5].duration = (
        CHANGE_PHASE_DURATION
    )

    phases[6].duration = (
        green_ns
    )

    phases[7].duration = (
        CHANGE_PHASE_DURATION
    )

    program.phases = phases

    traci.trafficlight.setProgramLogic(
        TLS_ID,
        program,
    )


# ============================================================
# MAIN CONTROLLER
# ============================================================

def apply_normal_controller(
    scenario: dict,
) -> dict:
    """
    Apply the normal Webster controller to the currently
    running SUMO simulation.

    Parameters
    ----------
    scenario:
        The complete scenario dictionary loaded from
        scenario.json.

    Returns
    -------
    dict
        The calculated signal timing.

    IMPORTANT
    ---------
    SUMO must already be connected through TraCI before
    this function is called.

    Example:

        traci.start([...])

        timing = apply_normal_controller(
            scenario
        )
    """

    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    required = {

        "name",

        "demand",
        "demand_rate",

        "approach_distribution",
    }

    missing = (
        required
        - set(scenario)
    )

    if missing:

        raise ValueError(
            "Scenario missing fields: "
            + ", ".join(
                sorted(missing)
            )
        )

    # --------------------------------------------------------
    # Validate demand
    # --------------------------------------------------------

    demand_rate = float(
        scenario["demand_rate"]
    )

    if demand_rate <= 0:

        raise ValueError(
            "scenario['demand_rate'] "
            "must be > 0."
        )

    # --------------------------------------------------------
    # Validate approach distribution
    # --------------------------------------------------------

    approach = scenario[
        "approach_distribution"
    ]

    required_approaches = {

        "north",
        "south",
        "east",
        "west",
    }

    missing_approaches = (
        required_approaches
        - set(approach)
    )

    if missing_approaches:

        raise ValueError(
            "approach_distribution missing: "
            + ", ".join(
                sorted(
                    missing_approaches
                )
            )
        )

    approach_total = sum(
        float(value)
        for value in approach.values()
    )

    if abs(
        approach_total - 1.0
    ) > 1e-5:

        raise ValueError(
            "approach_distribution "
            "must sum to 1.0."
        )

    # ========================================================
    # CALCULATE DEMAND
    # ========================================================

    ns_demand, ew_demand = (
        calculate_directional_demand(
            scenario
        )
    )

    # ========================================================
    # WEBSTER TIMING
    # ========================================================

    timing = (
        calculate_webster_timing(
            ns_demand,
            ew_demand,
        )
    )

    # ========================================================
    # APPLY TO SUMO
    # ========================================================

    apply_signal_timing(

        timing["green_ns"],

        timing["green_ew"],
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 60)
    print("NORMAL WEBSTER CONTROLLER")
    print("=" * 60)

    print(
        f"Scenario       : "
        f"{scenario['name']}"
    )

    print(
        f"Demand class   : "
        f"{scenario['demand']}"
    )

    print(
        f"Demand rate    : "
        f"{demand_rate:.1f} veh/h"
    )

    print(
        f"NS demand      : "
        f"{ns_demand:.1f} veh/h"
    )

    print(
        f"EW demand      : "
        f"{ew_demand:.1f} veh/h"
    )

    print(
        f"y_NS           : "
        f"{timing['y_ns']:.3f}"
    )

    print(
        f"y_EW           : "
        f"{timing['y_ew']:.3f}"
    )

    print(
        f"Y              : "
        f"{timing['Y']:.3f}"
    )

    print(
        f"Cycle          : "
        f"{timing['cycle']:.1f} s"
    )

    print(
        f"NS green       : "
        f"{timing['green_ns']:.1f} s"
    )

    print(
        f"EW green       : "
        f"{timing['green_ew']:.1f} s"
    )

    print(
        f"Change phase   : "
        f"{CHANGE_PHASE_DURATION:.1f} s"
    )

    print(
        "Signal timing applied."
    )

    print("=" * 60)

    return timing


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print(
        "normal_controller.py is a module."
    )

    print(
        "It must be called from a running "
        "TraCI/SUMO simulation."
    )

    print()
    print(
        "Example:"
    )

    print(
        "    apply_normal_controller(scenario)"
    )