"""
============================================================
ASTRID RUN ALL SCENARIOS
============================================================

Purpose
-------
Run every generated ASTRID scenario sequentially.

Architecture
------------

scenarios/
    scenario_001/
        scenario.json
        sq.flow.xml
        sq.vtype.xml
        sq.rou.xml

    scenario_002/
        scenario.json
        sq.flow.xml
        sq.vtype.xml
        sq.rou.xml

    ...

    scenario_200/
        scenario.json
        sq.flow.xml
        sq.vtype.xml
        sq.rou.xml


This script does NOT contain the simulation logic.

It simply does:

    scenario_001
        ↓
    state_extractor.py
        ↓
    SUMO
        ↓
    dataset_001

then:

    scenario_002
        ↓
    state_extractor.py
        ↓
    SUMO
        ↓
    dataset_002

and so on.

There is only ONE copy of:

    state_extractor.py
    sensor_simulator.py
    normal_controller.py
    scenario_builder.py
    create_scenarios.py

They are reused for every scenario.

IMPORTANT
---------
This script runs scenarios SEQUENTIALLY.

It does NOT create 200 SUMO windows simultaneously.

One scenario finishes before the next scenario starts.
============================================================
"""

import subprocess
import sys
import argparse
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent

SCENARIOS_DIR = (
    PROJECT_ROOT
    / "scenarios"
)


# ============================================================
# FIND SCENARIOS
# ============================================================

def find_scenarios():
    """
    Find every scenario directory containing scenario.json.
    """

    if not SCENARIOS_DIR.exists():

        raise FileNotFoundError(
            f"Scenarios directory does not exist:\n"
            f"{SCENARIOS_DIR}"
        )

    scenarios = sorted(

        path.name

        for path
        in SCENARIOS_DIR.iterdir()

        if (
            path.is_dir()
            and
            (
                path / "scenario.json"
            ).exists()
        )
    )

    if not scenarios:

        raise RuntimeError(
            "No scenario.json files were found."
        )

    return scenarios


# ============================================================
# RUN ONE SCENARIO
# ============================================================

def run_scenario(
    scenario_name: str,
):
    """
    Run one scenario through state_extractor.py.

    state_extractor.py is responsible for:

        - loading the scenario
        - starting SUMO
        - applying normal_controller
        - running the simulation
        - collecting sensor observations
        - creating sensor_dataset.json
        - closing SUMO
    """

    print()
    print("=" * 70)
    print(
        f"RUNNING SCENARIO: {scenario_name}"
    )
    print("=" * 70)

    subprocess.run(

        [
            sys.executable,

            "-m",

            "sensing.state_extractor",

            scenario_name,
            "--nogui", # comment this  --nogui for sumo 
        ],

        cwd=PROJECT_ROOT,

        check=True,
    )


# ============================================================
# MAIN
# ============================================================
def main():

    parser = argparse.ArgumentParser(
        description="Run ASTRID scenarios sequentially."
    )

    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Scenario number to start from."
    )

    args = parser.parse_args()

    scenarios = find_scenarios()

    # --------------------------------------------------------
    # Filter scenarios from the requested starting number
    # --------------------------------------------------------

    selected_scenarios = []

    for scenario_name in scenarios:

        try:
            number = int(
                scenario_name.split("_")[-1]
            )

        except ValueError:
            continue

        if number >= args.start:
            selected_scenarios.append(
                scenario_name
            )

    if not selected_scenarios:

        raise RuntimeError(
            f"No scenarios found starting from "
            f"scenario_{args.start:04d}."
        )

    print()
    print("=" * 70)
    print("ASTRID — RUN ALL SCENARIOS")
    print("=" * 70)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Starting from: scenario_{args.start:04d}"
    )

    print(
        f"Scenarios    : {len(selected_scenarios)}"
    )

    print()

    for index, scenario_name in enumerate(
        selected_scenarios,
        start=1,
    ):

        print(
            f"[{index}/{len(selected_scenarios)}] "
            f"{scenario_name}"
        )

        run_scenario(
            scenario_name
        )

    print()
    print("=" * 70)
    print("ALL SELECTED SCENARIOS COMPLETED")
    print("=" * 70)

    print(
        f"Total scenarios: "
        f"{len(selected_scenarios)}"
    )
    
    
# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()