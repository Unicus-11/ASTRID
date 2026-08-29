"""
Run/read the simulation state and produce synthetic sensor observations.

For example:

Step 0
├── Ground truth vehicles
├── GPS observations
├── CCTV observations
├── Traffic state
└── Signal state

Step 1
├── Ground truth vehicles
├── GPS observations
├── CCTV observations
├── Traffic state
└── Signal state

This doesn't create database.

The structure is:

SUMO
 ↓
TraCI
 ↓
sensor_simulator.py
 ↓
GPS + CCTV + ground truth
 ↓
state_extractor.py
 ↓
sensor_dataset.json

Our incoming roads are about 484.9 m long, so each camera
observes only the final 150 m approaching the junction.

IMPORTANT ARCHITECTURE
----------------------

There is only ONE copy of this file.

The same state_extractor.py is used repeatedly:

    scenario_001 → SUMO → dataset_001
    scenario_002 → SUMO → dataset_002
    scenario_003 → SUMO → dataset_003
                     ...
    scenario_200 → SUMO → dataset_200

This file does NOT create scenarios.

create_scenarios.py
    creates scenario.json

scenario_builder.py
    converts scenario.json into SUMO input files

state_extractor.py
    runs SUMO and extracts the resulting dataset

run_all_scenarios.py
    will later orchestrate all 200 scenarios
"""


import json
import sys
from pathlib import Path
from collections import Counter

import traci

from sensing.sensor_simulator import (
    SensorConfig,
    SensorSimulator,
    QUEUE_SPEED_THRESHOLD,
)

from controller.normal_controller import (
    apply_normal_controller,
)


# ============================================================
# PATHS
# ============================================================

# state_extractor.py is inside ASTRID/sensing/
# therefore parent.parent is the project root.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCENARIOS_DIR = PROJECT_ROOT / "scenarios"

DATASETS_DIR = PROJECT_ROOT / "datasets"

SUMO_DIR = (
    PROJECT_ROOT
    / "sumo"
    / "Squire_Junction_Multiple_Lanes"
)

NET_FILE = SUMO_DIR / "sq.net.xml"


# ============================================================
# CONSTANTS
# ============================================================

DIRECTION_EDGES = {

    "north": "4i",
    "south": "3i",
    "east": "2i",
    "west": "1i",
}

TLS_ID = "0"


# ============================================================
# LOAD SCENARIO
# ============================================================

def load_scenario(
    scenario_name: str,
) -> dict:

    scenario_dir = (
        SCENARIOS_DIR
        / scenario_name
    )

    scenario_path = (
        scenario_dir
        / "scenario.json"
    )

    if not scenario_path.exists():

        raise FileNotFoundError(
            f"Scenario does not exist:\n"
            f"{scenario_path}"
        )

    with open(
        scenario_path,
        "r",
        encoding="utf-8",
    ) as f:

        scenario = json.load(f)

    # The directory name and JSON name
    # must refer to the same scenario.

    if scenario.get("name") != scenario_name:

        raise ValueError(
            "Scenario mismatch.\n"
            f"Directory: {scenario_name}\n"
            f"JSON name: {scenario.get('name')}"
        )

    return scenario


# ============================================================
# VALIDATE SCENARIO
# ============================================================

def validate_scenario(
    scenario: dict,
):
    """
    Validate the current ASTRID scenario.json format.

    Source of truth:

        {
            "name": "...",
            "seed": ...,
            "simulation_end": ...,
            "demand": "...",
            "demand_rate": ...,
            "total_vehicles": ...,
            "vehicle_distribution": {...},
            "approach_distribution": {...},
            "movement_distribution": {...},
            "gps_penetration": ...,
            "cctv_detection": ...
        }

    The old demand structure is NOT supported:

        demand.class
        demand.profile
        demand.profile.base
        demand.profile.peak
    """

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    required = {
        "name",
        "seed",
        "simulation_end",
        "demand",
        "demand_rate",
        "total_vehicles",
        "vehicle_distribution",
        "approach_distribution",
        "movement_distribution",
        "gps_penetration",
        "cctv_detection",
    }

    missing = required - set(scenario)

    if missing:
        raise ValueError(
            "Scenario missing fields: "
            + ", ".join(sorted(missing))
        )

    # --------------------------------------------------------
    # Name
    # --------------------------------------------------------

    if not isinstance(
        scenario["name"],
        str,
    ) or not scenario["name"].strip():

        raise ValueError(
            "scenario.name must be a non-empty string."
        )

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    try:
        seed = int(scenario["seed"])
    except (TypeError, ValueError):

        raise ValueError(
            "seed must be an integer."
        )

    if seed < 0:

        raise ValueError(
            "seed must be >= 0."
        )

    # --------------------------------------------------------
    # Simulation duration
    # --------------------------------------------------------

    try:
        simulation_end = int(
            scenario["simulation_end"]
        )
    except (TypeError, ValueError):

        raise ValueError(
            "simulation_end must be an integer."
        )

    if simulation_end <= 0:

        raise ValueError(
            "simulation_end must be > 0."
        )

    # --------------------------------------------------------
    # Demand class
    # --------------------------------------------------------

    if not isinstance(
        scenario["demand"],
        str,
    ):

        raise ValueError(
            "demand must be a string."
        )

    allowed_demand_classes = {
        "low",
        "medium",
        "high",
        "very_high",
    }

    if scenario["demand"] not in allowed_demand_classes:

        raise ValueError(
            "Unknown demand class: "
            f"{scenario['demand']}. "
            f"Expected one of: "
            f"{sorted(allowed_demand_classes)}"
        )

    # --------------------------------------------------------
    # Demand rate
    # --------------------------------------------------------

    try:
        demand_rate = float(
            scenario["demand_rate"]
        )
    except (TypeError, ValueError):

        raise ValueError(
            "demand_rate must be numeric."
        )

    if demand_rate <= 0:

        raise ValueError(
            "demand_rate must be > 0."
        )

    # --------------------------------------------------------
    # Total vehicles
    # --------------------------------------------------------

    try:
        total_vehicles = int(
            scenario["total_vehicles"]
        )
    except (TypeError, ValueError):

        raise ValueError(
            "total_vehicles must be an integer."
        )

    if total_vehicles <= 0:

        raise ValueError(
            "total_vehicles must be > 0."
        )

    # --------------------------------------------------------
    # Demand consistency
    # --------------------------------------------------------
    #
    # demand_rate is vehicles/hour.
    #
    # Therefore the expected number of vehicles during
    # the simulation is:
    #
    #     demand_rate * simulation_time / 3600
    #
    # We check this rather than blindly trusting the JSON.
    #
    # Example:
    #
    #     demand_rate = 2100 veh/h
    #     simulation_end = 3600 s
    #
    #     expected = 2100 vehicles
    #
    # --------------------------------------------------------

    expected_total = round(
        demand_rate
        * simulation_end
        / 3600.0
    )

    if total_vehicles != expected_total:

        raise ValueError(
            "Scenario demand is inconsistent.\n"
            f"demand_rate   = {demand_rate}\n"
            f"simulation_end = {simulation_end}\n"
            f"total_vehicles = {total_vehicles}\n"
            f"expected_total = {expected_total}"
        )

    # --------------------------------------------------------
    # Vehicle distribution
    # --------------------------------------------------------

    validate_probability_distribution(
        scenario["vehicle_distribution"],
        "vehicle_distribution",
        [
            "bike",
            "car",
            "bus",
            "hgv",
        ],
    )

    # --------------------------------------------------------
    # Approach distribution
    # --------------------------------------------------------

    validate_probability_distribution(
        scenario["approach_distribution"],
        "approach_distribution",
        [
            "north",
            "south",
            "east",
            "west",
        ],
    )

    # --------------------------------------------------------
    # Movement distribution
    # --------------------------------------------------------

    validate_probability_distribution(
        scenario["movement_distribution"],
        "movement_distribution",
        [
            "left",
            "straight",
            "right",
        ],
    )

    # --------------------------------------------------------
    # GPS penetration
    # --------------------------------------------------------

    try:
        gps = float(
            scenario["gps_penetration"]
        )
    except (TypeError, ValueError):

        raise ValueError(
            "gps_penetration must be numeric."
        )

    if not 0.0 <= gps <= 1.0:

        raise ValueError(
            "gps_penetration must be "
            "between 0 and 1."
        )

    # --------------------------------------------------------
    # CCTV detection
    # --------------------------------------------------------

    try:
        cctv = float(
            scenario["cctv_detection"]
        )
    except (TypeError, ValueError):

        raise ValueError(
            "cctv_detection must be numeric."
        )

    if not 0.0 <= cctv <= 1.0:

        raise ValueError(
            "cctv_detection must be "
            "between 0 and 1."
        )


# ============================================================
# VALIDATE PROBABILITY DISTRIBUTION
# ============================================================

def validate_probability_distribution(
    distribution: dict,
    name: str,
    expected_keys=None,
):
    """
    Validate a probability distribution.

    Requirements:

        - must be a dictionary
        - must not be empty
        - required keys must exist
        - probabilities cannot be negative
        - probabilities must sum to 1.0
    """

    if not isinstance(
        distribution,
        dict,
    ):

        raise ValueError(
            f"{name} must be an object."
        )

    if not distribution:

        raise ValueError(
            f"{name} cannot be empty."
        )

    # --------------------------------------------------------
    # Required keys
    # --------------------------------------------------------

    if expected_keys is not None:

        missing = (
            set(expected_keys)
            - set(distribution)
        )

        if missing:

            raise ValueError(
                f"{name} missing keys: "
                f"{sorted(missing)}"
            )

    # --------------------------------------------------------
    # Validate individual probabilities
    # --------------------------------------------------------

    for key, value in distribution.items():

        try:
            value = float(value)

        except (TypeError, ValueError):

            raise ValueError(
                f"{name}[{key}] must be numeric."
            )

        if value < 0:

            raise ValueError(
                f"{name}[{key}] "
                "cannot be negative."
            )

    # --------------------------------------------------------
    # Validate total
    # --------------------------------------------------------

    total = sum(
        float(value)
        for value in distribution.values()
    )

    if abs(total - 1.0) > 1e-5:

        raise ValueError(
            f"{name} must sum to 1.0. "
            f"Current sum = {total}"
        )
# ============================================================
# VEHICLE FEATURES
# ============================================================

def calculate_vehicle_features(
    ground_truth,
):
    """
    Calculate vehicle-type and movement counts
    from the current ground-truth observation.
    """

    vehicle_types = Counter()

    movements = Counter()

    for vehicle in ground_truth:

        vehicle_type = vehicle.get(
            "vehicle_type"
        )

        movement = vehicle.get(
            "movement"
        )

        if vehicle_type:

            vehicle_types[
                vehicle_type
            ] += 1

        if movement:

            movements[
                movement
            ] += 1

    result = {

        "bike_count":
            vehicle_types["bike"],

        "car_count":
            vehicle_types["car"],

        "bus_count":
            vehicle_types["bus"],

        "hgv_count":
            vehicle_types["hgv"],
    }

    movement_names = [

        "north_to_east",
        "north_to_south",
        "north_to_west",

        "south_to_east",
        "south_to_north",
        "south_to_west",

        "east_to_north",
        "east_to_south",
        "east_to_west",

        "west_to_north",
        "west_to_south",
        "west_to_east",
    ]

    for movement in movement_names:

        result[
            f"{movement}_count"
        ] = movements[movement]

    return result


# ============================================================
# TRAFFIC STATE
# ============================================================

def calculate_traffic_state(
    previous_edge_vehicles,
):
    """
    Calculate traffic state on the four incoming roads.

    Queue definition:

        vehicle speed < QUEUE_SPEED_THRESHOLD
    """

    vehicle_ids = (
        traci.vehicle.getIDList()
    )

    edge_vehicles = {

        edge_id: []

        for edge_id
        in DIRECTION_EDGES.values()
    }

    # --------------------------------------------------------
    # Collect vehicles by incoming edge
    # --------------------------------------------------------

    for vehicle_id in vehicle_ids:

        edge_id = (
            traci.vehicle.getRoadID(
                vehicle_id
            )
        )

        if edge_id in edge_vehicles:

            edge_vehicles[
                edge_id
            ].append(
                vehicle_id
            )

    state = {}

    # --------------------------------------------------------
    # Calculate state for each approach
    # --------------------------------------------------------

    for direction, edge_id in (
        DIRECTION_EDGES.items()
    ):

        vehicles = edge_vehicles[
            edge_id
        ]

        speeds = [

            traci.vehicle.getSpeed(
                vehicle_id
            )

            for vehicle_id
            in vehicles
        ]

        # ----------------------------------------------------
        # Average speed
        # ----------------------------------------------------

        if speeds:

            average_speed = (
                sum(speeds)
                / len(speeds)
            )

        else:

            average_speed = 0.0

        # ----------------------------------------------------
        # Queue
        # ----------------------------------------------------

        queue = sum(

            1

            for speed in speeds

            if speed
            < QUEUE_SPEED_THRESHOLD
        )

        # ----------------------------------------------------
        # Arrivals
        # ----------------------------------------------------

        current_set = set(
            vehicles
        )

        arrivals = len(
            current_set
            -
            previous_edge_vehicles[
                edge_id
            ]
        )

        previous_edge_vehicles[
            edge_id
        ] = current_set

        # ----------------------------------------------------
        # Store state
        # ----------------------------------------------------

        state[direction] = {

            "vehicles":
                len(vehicles),

            "queue":
                queue,

            "speed":
                round(
                    average_speed,
                    2,
                ),

            "approach_arrivals":
                arrivals,
        }

    return state


# ============================================================
# CAMERA COUNTS
# ============================================================

def calculate_camera_counts(
    cctv,
):
    """
    Count CCTV detections by camera.
    """

    counts = {

        "north_camera": 0,
        "south_camera": 0,
        "east_camera": 0,
        "west_camera": 0,
    }

    for observation in cctv:

        camera_id = observation.get(
            "camera_id"
        )

        if camera_id in counts:

            counts[
                camera_id
            ] += 1

    return counts


# ============================================================
# RUN ONE SCENARIO
# ============================================================

def run_simulation(
    scenario_name: str,
    use_gui: bool = True,
):
    """
    Run exactly ONE scenario.

    The same function can later be called repeatedly
    by run_all_scenarios.py.
    """

    # ========================================================
    # LOAD + VALIDATE
    # ========================================================

    scenario = load_scenario(
        scenario_name
    )

    validate_scenario(
        scenario
    )

    scenario_dir = (
        SCENARIOS_DIR
        / scenario_name
    )

    # ========================================================
    # SUMO FILES
    # ========================================================

    route_file = (
        scenario_dir
        / "sq.rou.xml"
    )

    vtype_file = (
        scenario_dir
        / "sq.vtype.xml"
    )

    flow_file = (
        scenario_dir
        / "sq.flow.xml"
    )

    required_files = [

        NET_FILE,
        route_file,
        vtype_file,
        flow_file,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(

                f"Required SUMO file not found:\n"
                f"{file_path}\n\n"
                f"Run scenario_builder.py first."
            )

    # ========================================================
    # SENSOR CONFIGURATION
    # ========================================================

    sensor_config = SensorConfig(

        scenario_name=scenario[
            "name"
        ],

        seed=int(
            scenario["seed"]
        ),

        gps_penetration=float(
            scenario[
                "gps_penetration"
            ]
        ),

        cctv_detection=float(
            scenario[
                "cctv_detection"
            ]
        ),

        simulation_end=int(
            scenario[
                "simulation_end"
            ]
        ),
    )

    sensors = SensorSimulator(
        sensor_config
    )

    # ========================================================
    # DATASET OUTPUT
    # ========================================================

    dataset_dir = (
        DATASETS_DIR
        / scenario_name
    )

    dataset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_path = (
        dataset_dir
        / "sensor_dataset.json"
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print("=" * 70)
    print("ASTRID SCENARIO")
    print("=" * 70)

    print(
        f"Scenario:          "
        f"{scenario['name']}"
    )

    print(
        f"Seed:              "
        f"{scenario['seed']}"
    )

    print(
        f"Demand class:      "
        f"{scenario['demand']}"
    )

    print(
        f"Demand rate:       "
        f"{scenario['demand_rate']} veh/h"
    )

    print(
        f"Total vehicles:    "
        f"{scenario['total_vehicles']}"
    )

    print(
        f"GPS penetration:   "
        f"{scenario['gps_penetration']:.2%}"
    )

    print(
        f"CCTV detection:    "
        f"{scenario['cctv_detection']:.2%}"
    )

    print(
        f"Simulation end:    "
        f"{scenario['simulation_end']} s"
    )

    print("=" * 70)

    print()
    print("Scenario files:")

    print(
        f"  Network: {NET_FILE}"
    )

    print(
        f"  Flow:    {flow_file}"
    )

    print(
        f"  VType:   {vtype_file}"
    )

    print(
        f"  Route:   {route_file}"
    )

    # ========================================================
    # START SUMO
    # ========================================================

    sumo_binary = (
        "sumo-gui"
        if use_gui
        else "sumo"
    )

    traci.start([

        sumo_binary,

        "--net-file",
        str(NET_FILE),

        "--route-files",
        str(route_file),

        "--seed",
        str(
            scenario["seed"]
        ),

        "--begin",
        "0",

        "--end",
        str(
            scenario[
                "simulation_end"
            ]
        ),

        "--quit-on-end",
        "true",
    ])

    print()
    print(
        f"SUMO connected using "
        f"{sumo_binary}."
    )

    # ========================================================
    # NORMAL CONTROLLER
    # ========================================================

    apply_normal_controller(
        scenario
    )

    # ========================================================
    # PREVIOUS VEHICLES
    # ========================================================

    previous_edge_vehicles = {

        edge_id: set()

        for edge_id
        in DIRECTION_EDGES.values()
    }

    # ========================================================
    # DATASET
    # ========================================================

    dataset = []

    # ========================================================
    # SIMULATION LOOP
    # ========================================================

    try:

        while True:

            # ------------------------------------------------
            # Check simulation state
            # ------------------------------------------------

            current_time = (
                traci.simulation.getTime()
            )

            if current_time >= (
                sensor_config.simulation_end
            ):

                break

            if (
                traci.simulation
                .getMinExpectedNumber()
                <= 0
            ):

                break

            # ------------------------------------------------
            # Advance SUMO
            # ------------------------------------------------

            traci.simulationStep()

            simulation_time = (
                traci.simulation.getTime()
            )

            # ------------------------------------------------
            # Stop if simulation has reached end
            # ------------------------------------------------

            if simulation_time > (
                sensor_config.simulation_end
            ):

                break

            # ------------------------------------------------
            # Traffic state
            # ------------------------------------------------

            traffic_state = (
                calculate_traffic_state(
                    previous_edge_vehicles
                )
            )

            # ------------------------------------------------
            # Sensor observations
            # ------------------------------------------------

            sensor_data = (
                sensors.get_sensor_data()
            )

            # ------------------------------------------------
            # Camera counts
            # ------------------------------------------------

            camera_counts = (
                calculate_camera_counts(
                    sensor_data[
                        "cctv"
                    ]
                )
            )

            # ------------------------------------------------
            # Vehicle features
            # ------------------------------------------------

            vehicle_features = (
                calculate_vehicle_features(
                    sensor_data[
                        "ground_truth"
                    ]
                )
            )

            # ------------------------------------------------
            # Dataset record
            # ------------------------------------------------

            record = {

                "scenario":
                    scenario["name"],

                "seed":
                    scenario["seed"],

                "step":
                    len(dataset),

                "simulation_time":
                    simulation_time,

                "traffic":
                    traffic_state,

                "gps_count":
                    len(
                        sensor_data[
                            "gps"
                        ]
                    ),

                "cctv_count":
                    len(
                        sensor_data[
                            "cctv"
                        ]
                    ),

                "camera_counts":
                    camera_counts,

                "vehicle_features":
                    vehicle_features,

                "sensors":
                    sensor_data,
            }

            dataset.append(
                record
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if len(dataset) % 100 == 0:

                print(

                    f"Time: "
                    f"{simulation_time:7.1f}s | "

                    f"Records: "
                    f"{len(dataset):5d} | "

                    f"Vehicles: "
                    f"{len(sensor_data['ground_truth']):4d}"
                )

    finally:

        # ====================================================
        # SAVE DATASET
        # ====================================================

        with open(
            dataset_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                dataset,
                f,
                indent=2,
            )

        print()
        print(
            f"Dataset saved: "
            f"{len(dataset)} records"
        )

        print(
            f"Path: "
            f"{dataset_path}"
        )

        # ====================================================
        # CLOSE SUMO
        # ====================================================

        try:

            traci.close()

        except Exception:

            pass

        print(
            "SUMO closed."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) not in (2, 3):

        print(
            "Usage:"
        )

        print(
            "  python -m sensing.state_extractor "
            "<scenario_name>"
        )

        print()

        print(
            "GUI:"
        )

        print(
            "  python -m sensing.state_extractor "
            "scenario_0043"
        )

        print()

        print(
            "Without GUI:"
        )

        print(
            "  python -m sensing.state_extractor "
            "scenario_0043 --nogui"
        )

        sys.exit(1)

    scenario_name = sys.argv[1]

    use_gui = True

    if len(sys.argv) == 3:

        if sys.argv[2] == "--nogui":

            use_gui = False

        else:

            print(
                "Unknown option: "
                f"{sys.argv[2]}"
            )

            sys.exit(1)

    run_simulation(
        scenario_name,
        use_gui=use_gui,
    )