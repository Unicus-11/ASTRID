"""
ASTRID state extractor.

Usage:

    python state_extractor.py baseline

    python state_extractor.py realistic_gps

    python state_extractor.py sensor_degradation
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

# state_extractor.py is inside ASTRID/sensing/,
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
):

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
    scenario,
):

    required = {

        "name",
        "seed",
        "simulation_end",
        "total_vehicles",

        "vehicle_distribution",
        "approach_distribution",
        "movement_distribution",

        "gps_penetration",
        "cctv_detection",
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
    # Basic values
    # --------------------------------------------------------

    if int(scenario["seed"]) < 0:

        raise ValueError(
            "seed must be >= 0"
        )

    if int(scenario["total_vehicles"]) <= 0:

        raise ValueError(
            "total_vehicles must be > 0"
        )

    if int(scenario["simulation_end"]) <= 0:

        raise ValueError(
            "simulation_end must be > 0"
        )

    # --------------------------------------------------------
    # Distributions
    # --------------------------------------------------------

    distributions = {

        "vehicle_distribution":
            scenario["vehicle_distribution"],

        "approach_distribution":
            scenario["approach_distribution"],

        "movement_distribution":
            scenario["movement_distribution"],
    }

    for name, distribution in distributions.items():

        total = sum(
            float(value)
            for value in distribution.values()
        )

        if abs(total - 1.0) > 1e-9:

            raise ValueError(
                f"{name} must sum to 1.0. "
                f"Current sum = {total}"
            )

        for key, value in distribution.items():

            if float(value) < 0:

                raise ValueError(
                    f"{name}[{key}] cannot be negative."
                )

    # --------------------------------------------------------
    # Sensor configuration
    # --------------------------------------------------------

    gps = float(
        scenario["gps_penetration"]
    )

    cctv = float(
        scenario["cctv_detection"]
    )

    if not 0 <= gps <= 1:

        raise ValueError(
            "gps_penetration must be between 0 and 1"
        )

    if not 0 <= cctv <= 1:

        raise ValueError(
            "cctv_detection must be between 0 and 1"
        )


# ============================================================
# VEHICLE FEATURES
# ============================================================

def calculate_vehicle_features(
    ground_truth,
):

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
            vehicle_types[vehicle_type] += 1

        if movement:
            movements[movement] += 1

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

    vehicle_ids = traci.vehicle.getIDList()

    edge_vehicles = {

        edge_id: []

        for edge_id
        in DIRECTION_EDGES.values()
    }

    for vehicle_id in vehicle_ids:

        edge_id = traci.vehicle.getRoadID(
            vehicle_id
        )

        if edge_id in edge_vehicles:

            edge_vehicles[
                edge_id
            ].append(
                vehicle_id
            )

    state = {}

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

        if speeds:

            average_speed = (
                sum(speeds)
                / len(speeds)
            )

        else:

            average_speed = 0.0

        queue = sum(

            1

            for speed in speeds

            if speed
            < QUEUE_SPEED_THRESHOLD
        )

        current_set = set(
            vehicles
        )

        arrivals = len(
            current_set
            - previous_edge_vehicles[
                edge_id
            ]
        )

        previous_edge_vehicles[
            edge_id
        ] = current_set

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

    counts = {

        "north_camera": 0,
        "south_camera": 0,
        "east_camera": 0,
        "west_camera": 0,
    }

    for observation in cctv:

        camera_id = observation[
            "camera_id"
        ]

        if camera_id in counts:

            counts[camera_id] += 1

    return counts


# ============================================================
# RUN ONE SCENARIO
# ============================================================

def run_simulation(
    scenario_name: str,
):

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

    # ========================================================
    # CHECK GENERATED FILES
    # ========================================================

    required_files = [

        NET_FILE,
        route_file,
        vtype_file,
        flow_file,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(

                f"Required file not found:\n"
                f"{file_path}\n\n"
                f"Run scenario_builder.py first."
            )

    # ========================================================
    # SENSOR CONFIGURATION
    # ========================================================

    sensor_config = SensorConfig(

        scenario_name=scenario["name"],

        seed=int(
            scenario["seed"]
        ),

        gps_penetration=float(
            scenario["gps_penetration"]
        ),

        cctv_detection=float(
            scenario["cctv_detection"]
        ),

        simulation_end=int(
            scenario["simulation_end"]
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
        f"Scenario:          {scenario['name']}"
    )

    print(
        f"Seed:              {scenario['seed']}"
    )

    print(
        f"Total vehicles:    {scenario['total_vehicles']}"
    )

    print(
        f"Demand:            "
        f"{scenario.get('demand', 'undefined')}"
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
        f"{scenario['simulation_end']}"
    )

    print("=" * 70)

    print()
    print("Scenario files:")

    print(f"  Network: {NET_FILE}")
    print(f"  Flow:    {flow_file}")
    print(f"  VType:   {vtype_file}")
    print(f"  Route:   {route_file}")

    # ========================================================
    # START SUMO
    # ========================================================

    traci.start([

        "sumo-gui",

        "--net-file",
        str(NET_FILE),

        "--route-files",
        str(route_file),

        "--seed",
        str(scenario["seed"]),

        "--begin",
        "0",

        "--end",
        str(
            scenario["simulation_end"]
        ),

        "--quit-on-end",
        "true",
    ])

    print()
    print("SUMO connected.")

    # ========================================================
    # NORMAL CONTROLLER
    # ========================================================

    apply_normal_controller(
        scenario
    )

    # ========================================================
    # STATE
    # ========================================================

    previous_edge_vehicles = {

        edge_id: set()

        for edge_id
        in DIRECTION_EDGES.values()
    }

    dataset = []

    # ========================================================
    # SIMULATION LOOP
    # ========================================================

    try:

        while (
            traci.simulation.getMinExpectedNumber()
            > 0
        ):

            current_time = (
                traci.simulation.getTime()
            )

            if current_time >= (
                sensor_config.simulation_end
            ):
                break

            traci.simulationStep()

            simulation_time = (
                traci.simulation.getTime()
            )

            # ------------------------------------------------
            # Traffic state
            # ------------------------------------------------

            traffic_state = (
                calculate_traffic_state(
                    previous_edge_vehicles
                )
            )

            # ------------------------------------------------
            # Sensors
            # ------------------------------------------------

            sensor_data = (
                sensors.get_sensor_data()
            )

            # ------------------------------------------------
            # Camera counts
            # ------------------------------------------------

            camera_counts = (
                calculate_camera_counts(
                    sensor_data["cctv"]
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
                        sensor_data["gps"]
                    ),

                "cctv_count":
                    len(
                        sensor_data["cctv"]
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
        # SAVE
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
            f"Path: {dataset_path}"
        )

        traci.close()

        print("SUMO closed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python state_extractor.py <scenario_name>"
        )

        print()
        print(
            "Example:"
        )

        print(
            "python state_extractor.py baseline"
        )

        sys.exit(1)

    run_simulation(
        sys.argv[1]
    )