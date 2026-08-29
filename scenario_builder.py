"""
============================================================
ASTRID SCENARIO BUILDER
============================================================

Purpose
-------
Convert each scenario.json into the SUMO files required
for that scenario.

SOURCE OF TRUTH
---------------

scenario.json defines:

    demand
    demand_rate
    total_vehicles
    vehicle_distribution
    approach_distribution
    movement_distribution
    gps_penetration
    cctv_detection

The builder does NOT invent or recalculate demand.

DATA FLOW
---------

create_scenarios.py
        |
        v
scenario.json
        |
        v
scenario_builder.py
        |
        +--> sq.vtype.xml
        |
        +--> sq.flow.xml
        |
        v
    duarouter
        |
        v
    sq.rou.xml
        |
        v
       SUMO


IMPORTANT
---------

There is only ONE copy of this file.

There is only ONE copy of:

    sensor_simulator.py
    state_extractor.py

They are reused for every scenario.

Example:

    scenario_001 -> SUMO -> dataset_001
    scenario_002 -> SUMO -> dataset_002
    ...
    scenario_200 -> SUMO -> dataset_200
"""

import json
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

SCENARIOS_DIR = PROJECT_DIR / "scenarios"

SUMO_DIR = (
    PROJECT_DIR
    / "sumo"
    / "Squire_Junction_Multiple_Lanes"
)

ORIGINAL_NET = SUMO_DIR / "sq.net.xml"


# ============================================================
# MOVEMENTS
# ============================================================

MOVEMENTS = {

    "north": {
        "left": ("4i", "1o"),
        "straight": ("4i", "3o"),
        "right": ("4i", "2o"),
    },

    "south": {
        "left": ("3i", "2o"),
        "straight": ("3i", "4o"),
        "right": ("3i", "1o"),
    },

    "east": {
        "left": ("2i", "4o"),
        "straight": ("2i", "1o"),
        "right": ("2i", "3o"),
    },

    "west": {
        "left": ("1i", "3o"),
        "straight": ("1i", "2o"),
        "right": ("1i", "4o"),
    },
}


# ============================================================
# VEHICLE TYPES
# ============================================================

VEHICLE_TYPES = {

    "bike": {
        "accel": "3.5",
        "decel": "2.8",
        "length": "1.5",
        "maxSpeed": "27.77",
        "guiShape": "motorcycle",
        "stripWidth": "1",
    },

    "car": {
        "accel": "3.5",
        "decel": "2.8",
        "length": "4.5",
        "maxSpeed": "25.0",
        "guiShape": "passenger",
        "stripWidth": "3",
    },

    "hgv": {
        "accel": "2.5",
        "decel": "1.5",
        "length": "10.21",
        "maxSpeed": "19.44",
        "guiShape": "truck",
        "stripWidth": "5",
    },

    "bus": {
        "accel": "1.2",
        "decel": "0.9",
        "length": "11.54",
        "maxSpeed": "19.44",
        "guiShape": "bus",
        "stripWidth": "5",
    },
}


# ============================================================
# LOAD SCENARIO
# ============================================================

def load_scenario(path: Path) -> dict:

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# VALIDATION
# ============================================================

def validate_probability_distribution(
    distribution: dict,
    name: str,
    expected_keys=None,
):

    if not isinstance(distribution, dict):
        raise ValueError(
            f"{name} must be an object."
        )

    if not distribution:
        raise ValueError(
            f"{name} cannot be empty."
        )

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

    for key, value in distribution.items():

        value = float(value)

        if value < 0:
            raise ValueError(
                f"{name}[{key}] "
                f"cannot be negative."
            )

    total = sum(
        float(value)
        for value in distribution.values()
    )

    if abs(total - 1.0) > 1e-5:
        raise ValueError(
            f"{name} must sum to 1.0. "
            f"Current sum = {total}"
        )


def validate_scenario(
    scenario: dict,
):

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
    # Basic values
    # --------------------------------------------------------

    if not isinstance(scenario["name"], str):
        raise ValueError(
            "name must be a string."
        )

    if int(scenario["seed"]) < 0:
        raise ValueError(
            "seed must be >= 0."
        )

    if int(scenario["simulation_end"]) <= 0:
        raise ValueError(
            "simulation_end must be > 0."
        )

    if float(scenario["demand_rate"]) <= 0:
        raise ValueError(
            "demand_rate must be > 0."
        )

    if int(scenario["total_vehicles"]) <= 0:
        raise ValueError(
            "total_vehicles must be > 0."
        )

    # --------------------------------------------------------
    # Demand
    # --------------------------------------------------------

    if not isinstance(
        scenario["demand"],
        str,
    ):
        raise ValueError(
            "demand must be a string."
        )

    # --------------------------------------------------------
    # Demand consistency
    # --------------------------------------------------------

    expected_total = round(
        float(scenario["demand_rate"])
        * int(scenario["simulation_end"])
        / 3600.0
    )

    if int(scenario["total_vehicles"]) != expected_total:
        raise ValueError(
            "Scenario demand is inconsistent.\n"
            f"demand_rate   = {scenario['demand_rate']}\n"
            f"simulation_end = {scenario['simulation_end']}\n"
            f"total_vehicles = {scenario['total_vehicles']}\n"
            f"expected_total = {expected_total}"
        )

    # --------------------------------------------------------
    # Vehicle distribution
    # --------------------------------------------------------

    validate_probability_distribution(
        scenario["vehicle_distribution"],
        "vehicle_distribution",
        VEHICLE_TYPES.keys(),
    )

    # --------------------------------------------------------
    # Approach distribution
    # --------------------------------------------------------

    validate_probability_distribution(
        scenario["approach_distribution"],
        "approach_distribution",
        MOVEMENTS.keys(),
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
    # Sensors
    # --------------------------------------------------------

    gps = float(
        scenario["gps_penetration"]
    )

    cctv = float(
        scenario["cctv_detection"]
    )

    if not 0 <= gps <= 1:
        raise ValueError(
            "gps_penetration must be between 0 and 1."
        )

    if not 0 <= cctv <= 1:
        raise ValueError(
            "cctv_detection must be between 0 and 1."
        )
        
        
# ============================================================
# EXACT INTEGER ALLOCATION
# ============================================================

def allocate_counts(
    total: int,
    distribution: dict,
) -> dict:
    """
    Convert probabilities into integer counts.

    The final counts always sum exactly to `total`.
    """

    raw = {

        key:
        total * float(probability)

        for key, probability
        in distribution.items()
    }

    counts = {

        key:
        int(value)

        for key, value
        in raw.items()
    }

    remaining = (
        total
        - sum(counts.values())
    )

    remainders = sorted(

        (
            (
                raw[key]
                - counts[key],

                key,
            )

            for key in raw
        ),

        reverse=True,
    )

    for _, key in remainders[:remaining]:

        counts[key] += 1

    return counts


# ============================================================
# CREATE VEHICLE TYPE XML
# ============================================================

def create_vtype_xml(
    scenario: dict,
    output_file: Path,
):

    distribution = scenario[
        "vehicle_distribution"
    ]

    root = ET.Element(
        "routes"
    )

    type_distribution = ET.SubElement(
        root,
        "vTypeDistribution",
        {
            "id": "typedist1"
        },
    )

    for (
        vehicle_type,
        probability
    ) in distribution.items():

        attributes = (
            VEHICLE_TYPES[
                vehicle_type
            ].copy()
        )

        attributes["id"] = (
            vehicle_type
        )

        attributes["probability"] = (
            str(probability)
        )

        vtype = ET.SubElement(
            type_distribution,
            "vType",
            attributes,
        )

        ET.SubElement(
            vtype,
            "carFollowing-Krauss",
            {
                "sigma": "0.0",
                "tau": "1.0",
            },
        )

    tree = ET.ElementTree(
        root
    )

    ET.indent(
        tree,
        space="    ",
    )

    tree.write(
        output_file,
        encoding="UTF-8",
        xml_declaration=True,
    )

# ============================================================
# CREATE FLOW XML
# ============================================================

def create_flow_xml(
    scenario: dict,
    output_file: Path,
):
    """
    Create the SUMO flow file for one scenario.

    total_vehicles is taken directly from scenario.json.

    demand_rate is metadata describing traffic demand
    in vehicles/hour. It is NOT used to recalculate
    total_vehicles here.
    """

    root = ET.Element("routes")

    simulation_end = int(
        scenario["simulation_end"]
    )

    total_vehicles = int(
        scenario["total_vehicles"]
    )

    # --------------------------------------------------------
    # Allocate vehicles to approaches
    # --------------------------------------------------------

    approach_counts = allocate_counts(
        total_vehicles,
        scenario["approach_distribution"],
    )

    flow_id = 0

    # --------------------------------------------------------
    # Create flows for each approach
    # --------------------------------------------------------

    for direction in (
        "north",
        "south",
        "east",
        "west",
    ):

        approach_total = approach_counts.get(
            direction,
            0,
        )

        if approach_total <= 0:
            continue

        # ----------------------------------------------------
        # Allocate approach vehicles to movements
        # ----------------------------------------------------

        movement_counts = allocate_counts(
            approach_total,
            scenario["movement_distribution"],
        )

        for movement in (
            "left",
            "straight",
            "right",
        ):

            number = movement_counts.get(
                movement,
                0,
            )

            if number <= 0:
                continue

            # ------------------------------------------------
            # Get SUMO origin/destination edges
            # ------------------------------------------------

            try:

                from_edge, to_edge = MOVEMENTS[
                    direction
                ][movement]

            except KeyError as exc:

                raise ValueError(
                    f"Invalid movement configuration: "
                    f"{direction} -> {movement}"
                ) from exc

            # ------------------------------------------------
            # Create SUMO flow
            # ------------------------------------------------

            ET.SubElement(
                root,
                "flow",
                {
                    "id":
                        f"flow_{flow_id}",

                    "from":
                        from_edge,

                    "to":
                        to_edge,

                    "number":
                        str(number),

                    "begin":
                        "0",

                    "end":
                        str(simulation_end),

                    "type":
                        "typedist1",

                    "departLane":
                        "free",

                    "departSpeed":
                        "random",
                },
            )

            flow_id += 1

    # --------------------------------------------------------
    # Write XML
    # --------------------------------------------------------

    tree = ET.ElementTree(root)

    ET.indent(
        tree,
        space="    ",
    )

    tree.write(
        output_file,
        encoding="UTF-8",
        xml_declaration=True,
    )

    print(
        f"Created flow file: {output_file}"
    )


# ============================================================
# RUN DUAROUTER
# ============================================================

def create_route_file(
    scenario_dir: Path,
    seed: int,
):

    flow_file = (
        scenario_dir
        / "sq.flow.xml"
    )

    vtype_file = (
        scenario_dir
        / "sq.vtype.xml"
    )

    route_file = (
        scenario_dir
        / "sq.rou.xml"
    )

    scenario = load_scenario(
        scenario_dir
        / "scenario.json"
    )

    command = [
        "duarouter",

        "--net-file",
        str(ORIGINAL_NET),

        "--additional-files",
        str(vtype_file),

        "--route-files",
        str(flow_file),

        "--output-file",
        str(route_file),

        "--seed",
        str(seed),

        "--begin",
        "0",

        "--end",
        str(
            scenario["simulation_end"]
        ),
    ]

    print("Running duarouter...")

    subprocess.run(
        command,
        check=True,
    )

    print(
        f"Created route file: {route_file}"
    )


# ============================================================
# BUILD ONE SCENARIO
# ============================================================

def build_scenario(
    scenario_dir: Path,
):

    scenario_file = (
        scenario_dir
        / "scenario.json"
    )

    scenario = load_scenario(
        scenario_file
    )

    validate_scenario(
        scenario
    )

    scenario_name = scenario["name"]

    # --------------------------------------------------------
    # Check directory name
    # --------------------------------------------------------

    if scenario_name != scenario_dir.name:

        raise ValueError(
            "Scenario directory/name mismatch:\n"
            f"Directory = {scenario_dir.name}\n"
            f"JSON name = {scenario_name}"
        )

    total_vehicles = int(
        scenario["total_vehicles"]
    )

    # --------------------------------------------------------
    # Display scenario information
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        f"BUILDING SCENARIO: {scenario_name}"
    )
    print("=" * 70)

    print(
        f"Demand class:  {scenario['demand']}"
    )

    print(
        f"Demand rate:   "
        f"{scenario['demand_rate']} veh/h"
    )

    print(
        f"Total vehicles: {total_vehicles}"
    )

    print(
        f"Simulation:    "
        f"{scenario['simulation_end']} s"
    )

    print(
        f"GPS:            "
        f"{scenario['gps_penetration']:.2%}"
    )

    print(
        f"CCTV:           "
        f"{scenario['cctv_detection']:.2%}"
    )

    print(
        f"Seed:            "
        f"{scenario['seed']}"
    )

    # --------------------------------------------------------
    # Create vehicle types
    # --------------------------------------------------------

    create_vtype_xml(
        scenario,
        scenario_dir / "sq.vtype.xml",
    )

    # --------------------------------------------------------
    # Create traffic flows
    # --------------------------------------------------------

    create_flow_xml(
        scenario,
        scenario_dir / "sq.flow.xml",
    )

    # --------------------------------------------------------
    # Create routes
    # --------------------------------------------------------

    create_route_file(
        scenario_dir,
        int(scenario["seed"]),
    )

    print(
        f"Scenario '{scenario_name}' "
        f"built successfully."
    )


# ============================================================
# BUILD ALL SCENARIOS
# ============================================================

def main():

    if not SCENARIOS_DIR.exists():

        raise FileNotFoundError(
            "Scenario directory not found:\n"
            f"{SCENARIOS_DIR}"
        )

    scenario_dirs = sorted(
        path
        for path in SCENARIOS_DIR.iterdir()
        if (
            path.is_dir()
            and
            (path / "scenario.json").exists()
        )
    )

    if not scenario_dirs:

        raise RuntimeError(
            "No scenario.json files found."
        )

    print(
        f"Found {len(scenario_dirs)} scenarios."
    )

    # --------------------------------------------------------
    # Build every scenario
    # --------------------------------------------------------

    for index, scenario_dir in enumerate(
        scenario_dirs,
        start=1,
    ):

        print(
            f"\n[{index}/{len(scenario_dirs)}] "
            f"{scenario_dir.name}"
        )

        build_scenario(
            scenario_dir
        )

    print()
    print("=" * 70)
    print("ALL SCENARIOS BUILT")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()