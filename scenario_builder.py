"""  


scenario.json
     │
     ├── traffic amount
     ├── approach distribution
     ├── vehicle distribution
     ├── movement distribution
     ├── departure pattern
     ├── GPS penetration
     └── CCTV detection
             │
             ▼
     scenario_builder.py
             │
       ┌─────┴─────┐
       ▼           ▼
  scenario       scenario
  sq.flow.xml    sq.vtype.xml
       │           │
       └─────┬─────┘
             ▼
         duarouter
             ▼
     scenario/sq.rou.xml
     
     
basically 

scenario.json
     ↓
defines WHAT traffic we want
     ↓
scenario_builder.py
     ↓
creates sq.flow.xml + sq.vtype.xml
     ↓
duarouter
     ↓
sq.rou.xml
   
   
   
   Corrected MOVEMENTS to match sensor_simulator.py.
Largest-remainder allocation so the generated flow counts sum exactly to the intended total.
demand now affects vehicle volume using:
low → 0.75
medium → 1.00
high → 1.25   
     
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
# DEMAND PROFILES
# ============================================================


DEMAND_PROFILES = {
    "low": {
        "base": 700,
        "peak": 1100,
    },

    "medium": {
        "base": 1000,
        "peak": 1800,
    },

    "high": {
        "base": 1400,
        "peak": 2400,
    },

    "very_high": {
        "base": 1800,
        "peak": 3200,
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


def calculate_total_vehicles(
    scenario: dict,
) -> int:
    """
    Convert the scenario's demand level into an exact
    number of vehicles for the simulation duration.
    """

    demand_name = scenario["demand"]

    if demand_name not in DEMAND_PROFILES:
        raise ValueError(
            f"Unknown demand level: {demand_name}. "
            f"Expected one of: {sorted(DEMAND_PROFILES)}"
        )

    demand_rate = DEMAND_PROFILES[demand_name]

    simulation_end = int(
        scenario["simulation_end"]
    )

    total = (
        demand_rate
        * simulation_end
        / 3600
    )

    return round(total)


# ============================================================
# LOAD SCENARIO
# ============================================================

def load_scenario(path: Path) -> dict:

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# VALIDATION
# ============================================================

def validate_probability_distribution(
    distribution: dict,
    name: str,
    expected_keys=None,
):
    if not distribution:
        raise ValueError(f"{name} cannot be empty.")

    if expected_keys is not None:

        missing = set(expected_keys) - set(distribution)

        if missing:
            raise ValueError(
                f"{name} missing keys: {sorted(missing)}"
            )

    for key, value in distribution.items():

        value = float(value)

        if value < 0:
            raise ValueError(
                f"{name}[{key}] cannot be negative."
            )

    total = sum(float(v) for v in distribution.values())

    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"{name} must sum to 1.0. "
            f"Current sum = {total}"
        )


def validate_scenario(scenario: dict):

    required = {
        "name",
        "seed",
        "simulation_end",
        "demand",
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

    if int(scenario["seed"]) < 0:
        raise ValueError("seed must be >= 0.")

    if int(scenario["simulation_end"]) <= 0:
        raise ValueError("simulation_end must be > 0.")
    
    if scenario["demand"] not in DEMAND_PROFILES:
        raise ValueError(
            f"Unknown demand level: {scenario['demand']}. "
            f"Expected one of: {sorted(DEMAND_PROFILES)}"
        )

    if int(scenario["total_vehicles"]) <= 0:
        raise ValueError("total_vehicles must be > 0.")

    validate_probability_distribution(
        scenario["vehicle_distribution"],
        "vehicle_distribution",
        VEHICLE_TYPES.keys(),
    )

    validate_probability_distribution(
        scenario["approach_distribution"],
        "approach_distribution",
        MOVEMENTS.keys(),
    )

    validate_probability_distribution(
        scenario["movement_distribution"],
        "movement_distribution",
        ["left", "straight", "right"],
    )

    gps = float(scenario["gps_penetration"])
    cctv = float(scenario["cctv_detection"])

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

def allocate_counts(total: int, distribution: dict) -> dict:
    """
    Convert probabilities into integer counts whose sum is
    exactly `total`.

    Uses the largest-remainder method.
    """

    raw = {
        key: total * float(probability)
        for key, probability in distribution.items()
    }

    counts = {
        key: int(value)
        for key, value in raw.items()
    }

    remaining = total - sum(counts.values())

    remainders = sorted(
        (
            (raw[key] - counts[key], key)
            for key in raw
        ),
        reverse=True,
    )

    for _, key in remainders[:remaining]:
        counts[key] += 1

    return counts


# ============================================================
# CREATE VTYPE XML
# ============================================================

def create_vtype_xml(
    scenario: dict,
    output_file: Path,
):

    distribution = scenario["vehicle_distribution"]

    root = ET.Element("routes")

    type_distribution = ET.SubElement(
        root,
        "vTypeDistribution",
        {"id": "typedist1"},
    )

    for vehicle_type, probability in distribution.items():

        attributes = VEHICLE_TYPES[vehicle_type].copy()

        attributes["id"] = vehicle_type
        attributes["probability"] = str(probability)

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

    tree = ET.ElementTree(root)

    ET.indent(tree, space="    ")

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

    root = ET.Element("routes")

    simulation_end = int(
        scenario["simulation_end"]
    )
    
    total_vehicles = calculate_total_vehicles(
    scenario
)

    approach_counts = allocate_counts(
        total_vehicles,
        scenario["approach_distribution"],
    )

    flow_id = 0

    for direction in MOVEMENTS:

        approach_total = approach_counts[direction]

        movement_counts = allocate_counts(
            approach_total,
            scenario["movement_distribution"],
        )

        for movement in (
            "left",
            "straight",
            "right",
        ):

            number = movement_counts[movement]

            if number <= 0:
                continue

            from_edge, to_edge = (
                MOVEMENTS[direction][movement]
            )

            ET.SubElement(
                root,
                "flow",
                {
                    "id": f"flow_{flow_id}",
                    "from": from_edge,
                    "to": to_edge,
                    "number": str(number),
                    "begin": "0",
                    "end": str(simulation_end),
                    "type": "typedist1",
                    "departLane": "free",
                    "departSpeed": "random",
                },
            )

            flow_id += 1

    tree = ET.ElementTree(root)

    ET.indent(tree, space="    ")

    tree.write(
        output_file,
        encoding="UTF-8",
        xml_declaration=True,
    )


# ============================================================
# RUN DUAROUTER
# ============================================================

def create_route_file(
    scenario_dir: Path,
    seed: int,
):

    flow_file = scenario_dir / "sq.flow.xml"
    vtype_file = scenario_dir / "sq.vtype.xml"
    route_file = scenario_dir / "sq.rou.xml"

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
            # Read from scenario JSON
            load_scenario(
                scenario_dir / "scenario.json"
            )["simulation_end"]
        ),
    ]

    print("Running duarouter...")

    subprocess.run(
        command,
        check=True,
    )

    print(f"Created: {route_file}")


# ============================================================
# BUILD ONE SCENARIO
# ============================================================

def build_scenario(
    scenario_dir: Path,
):

    scenario_file = scenario_dir / "scenario.json"

    scenario = load_scenario(scenario_file)

    validate_scenario(scenario)

    scenario_name = scenario["name"]

    if scenario_name != scenario_dir.name:
        raise ValueError(
            f"Scenario directory/name mismatch:\n"
            f"Directory = {scenario_dir.name}\n"
            f"JSON name  = {scenario_name}"
        )

    print()
    print("=" * 70)
    print(f"BUILDING SCENARIO: {scenario_name}")
    print("=" * 70)

    print(
        f"Vehicles: {scenario['total_vehicles']}"
    )

    print(
        f"Simulation: {scenario['simulation_end']} s"
    )

    print(
        f"Seed: {scenario['seed']}"
    )

    # --------------------------------------------------------
    # Vehicle types
    # --------------------------------------------------------

    create_vtype_xml(
        scenario,
        scenario_dir / "sq.vtype.xml",
    )

    # --------------------------------------------------------
    # Traffic flows
    # --------------------------------------------------------

    create_flow_xml(
        scenario,
        scenario_dir / "sq.flow.xml",
    )

    # --------------------------------------------------------
    # Routes
    # --------------------------------------------------------

    create_route_file(
        scenario_dir,
        int(scenario["seed"]),
    )

    print(
        f"Scenario '{scenario_name}' built successfully."
    )


# ============================================================
# BUILD ALL SCENARIOS
# ============================================================

def main():

    if not SCENARIOS_DIR.exists():
        raise FileNotFoundError(
            f"Scenario directory not found:\n"
            f"{SCENARIOS_DIR}"
        )

    scenario_dirs = sorted(
        path
        for path in SCENARIOS_DIR.iterdir()
        if (
            path.is_dir()
            and (path / "scenario.json").exists()
        )
    )

    if not scenario_dirs:
        raise RuntimeError(
            "No scenario.json files found."
        )

    for scenario_dir in scenario_dirs:
        build_scenario(scenario_dir)

    print()
    print("=" * 70)
    print("ALL SCENARIOS BUILT")
    print("=" * 70)


if __name__ == "__main__":
    main()