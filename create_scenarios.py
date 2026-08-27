import json
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "scenarios"



# ============================================================
# SCENARIO DEFINITIONS
# ============================================================

SCENARIOS = [

    # ========================================================
    # 1. BASELINE
    # ========================================================

    {
        "name": "baseline",

        "seed": 42,

        "simulation_end": 3600,


        "vehicle_distribution": {
            "bike": 0.50,
            "car": 0.30,
            "bus": 0.10,
            "hgv": 0.10
        },

        "approach_distribution": {
            "north": 0.25,
            "south": 0.25,
            "east": 0.25,
            "west": 0.25
        },

        "movement_distribution": {
            "left": 0.30,
            "straight": 0.50,
            "right": 0.20
        },

        "demand": "medium",

        "gps_penetration": 0.30,

        "cctv_detection": 1.00
    },


    # ========================================================
    # 2. REALISTIC GPS
    # ========================================================

    {
        "name": "realistic_gps",

        "seed": 43,

        "simulation_end": 3600,

        "vehicle_distribution": {
            "bike": 0.50,
            "car": 0.30,
            "bus": 0.10,
            "hgv": 0.10
        },

        "approach_distribution": {
            "north": 0.25,
            "south": 0.25,
            "east": 0.25,
            "west": 0.25
        },

        "movement_distribution": {
            "left": 0.30,
            "straight": 0.50,
            "right": 0.20
        },

        "demand": "medium",

        # 39% GPS penetration
        "gps_penetration": 0.39,

        "cctv_detection": 1.00
    },


    # ========================================================
    # 3. EAST-WEST HEAVY
    # ========================================================

    {
        "name": "east_west_heavy",

        "seed": 44,

        "simulation_end": 3600,

        "vehicle_distribution": {
            "bike": 0.50,
            "car": 0.30,
            "bus": 0.10,
            "hgv": 0.10
        },

        "approach_distribution": {
            "north": 0.10,
            "south": 0.10,
            "east": 0.35,
            "west": 0.45
        },

        "movement_distribution": {
            "left": 0.30,
            "straight": 0.50,
            "right": 0.20
        },

        "demand": "high",

        "gps_penetration": 0.25,

        "cctv_detection": 1.00
    },


    # ========================================================
    # 4. NORTH-SOUTH HEAVY
    # ========================================================

    {
        "name": "north_south_heavy",

        "seed": 45,

        "simulation_end": 3600,


        "vehicle_distribution": {
            "bike": 0.50,
            "car": 0.30,
            "bus": 0.10,
            "hgv": 0.10
        },

        "approach_distribution": {
            "north": 0.40,
            "south": 0.35,
            "east": 0.15,
            "west": 0.10
        },

        "movement_distribution": {
            "left": 0.30,
            "straight": 0.50,
            "right": 0.20
        },

        "demand": "high",

        "gps_penetration": 0.25,

        "cctv_detection": 1.00
    },


    # ========================================================
    # 5. HIGH DEMAND
    # ========================================================

    {
        "name": "high_demand",

        "seed": 46,

        "simulation_end": 3600,


        "vehicle_distribution": {
            "bike": 0.40,
            "car": 0.40,
            "bus": 0.10,
            "hgv": 0.10
        },

        "approach_distribution": {
            "north": 0.25,
            "south": 0.25,
            "east": 0.25,
            "west": 0.25
        },

        "movement_distribution": {
            "left": 0.30,
            "straight": 0.50,
            "right": 0.20
        },

        "demand": "very_high",

        "gps_penetration": 0.25,

        "cctv_detection": 1.00
    },


    # ========================================================
    # 6. SENSOR DEGRADATION
    # ========================================================

    {
        "name": "sensor_degradation",

        "seed": 47,

        "simulation_end": 3600,


        "vehicle_distribution": {
            "bike": 0.50,
            "car": 0.30,
            "bus": 0.10,
            "hgv": 0.10
        },

        "approach_distribution": {
            "north": 0.25,
            "south": 0.25,
            "east": 0.25,
            "west": 0.25
        },

        "movement_distribution": {
            "left": 0.30,
            "straight": 0.50,
            "right": 0.20
        },

        "demand": "medium",

        "gps_penetration": 0.20,

        "cctv_detection": 0.70
    },


    # ========================================================
    # 7. HEAVY ALL SIDES
    # ========================================================

    {
        "name": "heavy_all_sides",

        "seed": 48,

        "simulation_end": 3600,


        "vehicle_distribution": {
            "bike": 0.40,
            "car": 0.40,
            "bus": 0.10,
            "hgv": 0.10
        },

        "approach_distribution": {
            "north": 0.25,
            "south": 0.25,
            "east": 0.25,
            "west": 0.25
        },

        "movement_distribution": {
            "left": 0.30,
            "straight": 0.50,
            "right": 0.20
        },

        "demand": "very_high",

        "gps_penetration": 0.25,

        "cctv_detection": 1.00
    }
]


# ============================================================
# CREATE SCENARIOS
# ============================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    for scenario in SCENARIOS:
        folder = OUTPUT_DIR / scenario["name"]
        folder.mkdir(exist_ok=True)

        output = folder / "scenario.json"

        with open(output, "w", encoding="utf-8") as f:
            json.dump(scenario, f, indent=2)

        print(f"Created {output}")

if __name__ == "__main__":
    main()