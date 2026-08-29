"""
============================================================
ASTRID — SCENARIO GENERATOR
============================================================

Purpose
-------
Generate many different traffic scenarios for SUMO.

Each scenario changes:

    - traffic demand
    - approach distribution
    - vehicle composition
    - turning movements
    - GPS penetration
    - CCTV detection quality

Important
---------
We are NOT manually assigning queue length.

Queue length must EMERGE from the SUMO simulation because
it depends on traffic demand, vehicle mix, movements,
signal timing, and vehicle behaviour.

GPS and CCTV describe how much of the traffic we can observe:

    GPS penetration
        = fraction of vehicles equipped with GPS

    CCTV detection
        = probability that CCTV detects a vehicle

Sensor noise and observation errors belong in
sensor_simulator.py.

The neural network is NOT part of this stage.

The order is:

    create_scenarios.py
            ↓
    scenario.json
            ↓
    scenario_builder.py
            ↓
    sq.flow.xml + sq.vtype.xml
            ↓
    duarouter
            ↓
    sq.rou.xml
            ↓
    SUMO simulation
            ↓
    sensor_simulator.py
            ↓
    state_extractor.py
            ↓
    COMPLETE DATASET
            ↓
    neural network

============================================================
WHAT KIND OF SCENARIOS ARE GENERATED?
============================================================

The scenarios represent different traffic conditions.

LOW
    lighter traffic

MEDIUM
    normal/moderate traffic

HIGH
    heavy traffic

VERY_HIGH
    very heavy / congested traffic

Within each demand level, the generator also changes:

    - how traffic is distributed between approaches
    - vehicle composition
    - turning behaviour
    - GPS availability
    - CCTV detection quality

Therefore two "medium" scenarios can still be different.

Queue length is NOT generated here.
It will be measured after SUMO runs.

Main formulas

1. Total vehicles

>> total_vehicles = demand_rate

For example:

>> high demand → 2100 vehicles

2. Vehicle-type allocation

>> vehicles_of_type = total_vehicles × vehicle_distribution[type]


(Conceptually:

    allocation = total × probability

Then largest-remainder allocation converts the fractional
results into integers while preserving the required total.)

Example:

2100 × 0.70 = 1470 cars
2100 × 0.20 = 420 bikes
2100 × 0.05 = 105 buses
2100 × 0.05 = 105 HGVs

3. Approach allocation

>> approach_vehicles = total_vehicles × approach_distribution[direction]

Example:

2100 × 0.25 = 525 north
2100 × 0.25 = 525 south
...

4. Movement allocation

For each approach:

>> movement_vehicles =
    approach_vehicles × movement_probability

For example, if north has 525 vehicles:

straight = 525 × 0.50
left     = 525 × 0.30
right    = 525 × 0.20

You also wanted the largest-remainder allocation so that rounding does not cause:

>> sum(movements) ≠ approach_vehicles

Instead:

>> sum(all movements) = approach_vehicles

============================================================
"""

import json
import random
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

# This file is directly inside the ASTRID project folder.
PROJECT_DIR = Path(__file__).resolve().parent

SCENARIOS_DIR = PROJECT_DIR / "scenarios"


# ============================================================
# SIMULATION SETTINGS
# ============================================================

SIMULATION_END = 3600


# ============================================================
# DEMAND LEVELS
# ============================================================

# Demand is expressed as vehicles per hour.
#
# The generator randomly selects a value inside the
# corresponding range.
#
# Higher demand should generally create more congestion
# when the intersection cannot process vehicles fast enough.

DEMAND_RANGES = {
    "low": (700, 1200),
    "medium": (1000, 1800),
    "high": (1500, 2400),
    "very_high": (2200, 3200),
}


# ============================================================
# VEHICLE TYPE RANGES
# ============================================================

# Vehicle composition is varied for every scenario.
#
# The four values are probabilities and must sum to 1.

VEHICLE_RANGES = {

    "bike": (0.30, 0.60),

    "car": (0.20, 0.50),

    "bus": (0.05, 0.15),

    "hgv": (0.05, 0.15),
}


# ============================================================
# MOVEMENT RANGES
# ============================================================

# Turning behaviour is also varied.
#
# left + straight + right = 1.

MOVEMENT_RANGES = {

    "left": (0.15, 0.40),

    "straight": (0.35, 0.60),

    "right": (0.10, 0.30),
}


# ============================================================
# NORMALISE DISTRIBUTION
# ============================================================

def normalise_distribution(
    values: dict,
) -> dict:
    """
    Convert arbitrary positive values into probabilities
    whose total is exactly 1.0.
    """

    total = sum(values.values())

    if total <= 0:
        raise ValueError(
            "Distribution total must be greater than zero."
        )

    return {
        key: value / total
        for key, value in values.items()
    }


# ============================================================
# CREATE APPROACH DISTRIBUTION
# ============================================================

def create_approach_distribution(
    rng,
) -> dict:
    """
    Create a different traffic split for the four approaches.

    Example:

        north = 0.40
        south = 0.20
        east  = 0.25
        west  = 0.15

    This allows asymmetric traffic instead of always
    sending exactly 25% from every direction.
    """

    raw = {
        "north": rng.uniform(0.15, 0.40),
        "south": rng.uniform(0.15, 0.40),
        "east": rng.uniform(0.15, 0.40),
        "west": rng.uniform(0.15, 0.40),
    }

    return normalise_distribution(raw)


# ============================================================
# CREATE VEHICLE DISTRIBUTION
# ============================================================

def create_vehicle_distribution(
    rng,
) -> dict:
    """
    Create a different mixture of bikes, cars, buses and HGVs.
    """

    raw = {
        vehicle_type: rng.uniform(
            minimum,
            maximum,
        )

        for vehicle_type, (
            minimum,
            maximum,
        )

        in VEHICLE_RANGES.items()
    }

    return normalise_distribution(raw)


# ============================================================
# CREATE MOVEMENT DISTRIBUTION
# ============================================================

def create_movement_distribution(
    rng,
) -> dict:
    """
    Create a different left/straight/right distribution.
    """

    raw = {
        movement: rng.uniform(
            minimum,
            maximum,
        )

        for movement, (
            minimum,
            maximum,
        )

        in MOVEMENT_RANGES.items()
    }

    return normalise_distribution(raw)


# ============================================================
# CREATE DEMAND
# ============================================================

def create_demand(
    rng,
):
    """
    Select a demand class and then select a random traffic
    rate inside that class.

    Example:

        demand = "high"
        demand_rate = 2117 veh/hour
    """

    demand_class = rng.choice(
        list(DEMAND_RANGES.keys())
    )

    minimum, maximum = DEMAND_RANGES[
        demand_class
    ]

    demand_rate = rng.uniform(
        minimum,
        maximum,
    )

    return (
        demand_class,
        round(demand_rate),
    )


# ============================================================
# CREATE ONE SCENARIO
# ============================================================

def create_scenario(
    scenario_number: int,
    seed: int,
) -> dict:
    """
    Generate one complete scenario description.
    """

    rng = random.Random(seed)

    # --------------------------------------------------------
    # Traffic demand
    # --------------------------------------------------------

    demand_class, demand_rate = create_demand(
        rng
    )

    # --------------------------------------------------------
    # Traffic distributions
    # --------------------------------------------------------

    approach_distribution = (
        create_approach_distribution(rng)
    )

    vehicle_distribution = (
        create_vehicle_distribution(rng)
    )

    movement_distribution = (
        create_movement_distribution(rng)
    )

    # --------------------------------------------------------
    # Exact number of vehicles for this simulation
    # --------------------------------------------------------
    #
    # demand_rate is vehicles/hour.
    #
    # simulation duration is one hour here.
    #
    # Therefore:
    #
    # total_vehicles =
    #       demand_rate × simulation_hours
    #
    # This is calculated automatically.
    # It is NOT manually chosen as "1600".

    simulation_hours = (
        SIMULATION_END / 3600
    )

    total_vehicles = round(
        demand_rate
        * simulation_hours
    )

    # --------------------------------------------------------
    # Sensor conditions
    # --------------------------------------------------------

    gps_penetration = rng.uniform(
        0.20,
        0.50,
    )

    cctv_detection = rng.uniform(
        0.70,
        1.00,
    )

    # --------------------------------------------------------
    # Complete scenario
    # --------------------------------------------------------

    scenario = {

        "name":
            f"scenario_{scenario_number:04d}",

        "seed":
            seed,

        "simulation_end":
            SIMULATION_END,

        # Traffic level.
        "demand":
            demand_class,

        # Actual traffic rate generated for this scenario.
        "demand_rate":
            demand_rate,

        # Exact number of vehicles that the builder must create.
        "total_vehicles":
            total_vehicles,

        "vehicle_distribution":
            {
                key: round(value, 6)

                for key, value
                in vehicle_distribution.items()
            },

        "approach_distribution":
            {
                key: round(value, 6)

                for key, value
                in approach_distribution.items()
            },

        "movement_distribution":
            {
                key: round(value, 6)

                for key, value
                in movement_distribution.items()
            },

        "gps_penetration":
            round(
                gps_penetration,
                4,
            ),

        "cctv_detection":
            round(
                cctv_detection,
                4,
            ),
    }

    return scenario


# ============================================================
# GENERATE ALL SCENARIOS
# ============================================================

def main():

    SCENARIOS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Number of different traffic experiments.
    NUMBER_OF_SCENARIOS = 200

    for i in range(
        1,
        NUMBER_OF_SCENARIOS + 1,
    ):

        seed = 1000 + i

        scenario = create_scenario(
            scenario_number=i,
            seed=seed,
        )

        scenario_dir = (
            SCENARIOS_DIR
            / scenario["name"]
        )

        scenario_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            scenario_dir
            / "scenario.json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                scenario,
                f,
                indent=2,
            )

        print(
            f"Created {output_file}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()