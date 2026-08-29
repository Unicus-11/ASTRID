"""
run_scenarios.py
================

ASTRID Prototype

Purpose:
    Run the generated SUMO scenarios through the real SUMO network
    and save RAW vehicle trajectories.

This script DOES:
    1. Read scenario_XXXX/scenario.json
    2. Read scenario_XXXX/flow.xml
    3. Read scenario_XXXX/vtype.xml
    4. Use the fixed real network: sq.net.xml
    5. Run SUMO for the configured simulation period
    6. Use the scenario's seed for reproducibility
    7. Record every active vehicle at every simulation timestep
    8. Save raw trajectory CSV files

This script DOES NOT:
    - calculate queue length
    - calculate density
    - calculate flow
    - calculate shockwave speed
    - create camera observations
    - create GPS observations
    - create ML features
    - train an ML model

The output is RAW SUMO GROUND TRUTH.

Run ONE scenario:
    python sumo/run_scenarios.py --scenario scenario_0001

Run ALL scenarios:
    python sumo/run_scenarios.py

Run with SUMO-GUI:
    python sumo/run_scenarios.py --scenario scenario_0001 --gui
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import traci


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUMO_DIR = PROJECT_ROOT / "sumo"

SCENARIOS_DIR = SUMO_DIR / "generated_scenarios"

NETWORK_DIR = SUMO_DIR / "Squire_Junction_Multiple_Lanes"

NETWORK_FILE = NETWORK_DIR / "sq.net.xml"


# ============================================================================
# NETWORK EDGE CLASSIFICATION
# ============================================================================

# Real incoming approach edges from Scenario_config.json:
#
#     1i = WEST
#     2i = EAST
#     3i = SOUTH
#     4i = NORTH
#
# Used only to explicitly identify whether a vehicle is currently
# on one of the four incoming approaches.

APPROACH_EDGES = {
    "1i",
    "2i",
    "3i",
    "4i",
}


def is_internal_edge(edge_id: str) -> bool:
    """
    SUMO internal junction edges start with ':'.

    Example:
        :0_0
        :0_1
    """

    return edge_id.startswith(":")


# ============================================================================
# SUMO COMMAND
# ============================================================================

def get_sumo_binary(use_gui: bool) -> str:
    """
    Find SUMO or SUMO-GUI from the system PATH.
    """

    binary_name = "sumo-gui" if use_gui else "sumo"

    binary = shutil.which(binary_name)

    if binary is None:
        print()
        print(f"ERROR: '{binary_name}' was not found in PATH.")
        print()
        print("Check that SUMO is installed and that this works:")
        print()
        print(f"    {binary_name} --version")
        print()

        sys.exit(1)

    return binary


# ============================================================================
# LOAD SCENARIO METADATA
# ============================================================================

def load_scenario_metadata(scenario_dir: Path) -> dict:
    """
    Load scenario.json produced by Scenario_builder.py.
    """

    scenario_file = scenario_dir / "scenario.json"

    if not scenario_file.exists():
        raise FileNotFoundError(
            f"Missing scenario metadata: {scenario_file}"
        )

    with open(scenario_file, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# VALIDATE SCENARIO FILES
# ============================================================================

def validate_scenario_files(scenario_dir: Path) -> None:
    """
    Make sure all files required by the SUMO run exist.
    """

    required_files = [
        scenario_dir / "scenario.json",
        scenario_dir / "flow.xml",
        scenario_dir / "vtype.xml",
        NETWORK_FILE,
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )


# ============================================================================
# BUILD TEMPORARY SUMO CONFIG
# ============================================================================

def build_sumo_config(
    scenario_dir: Path,
    scenario: dict,
    output_dir: Path,
) -> Path:
    """
    Create a scenario-specific SUMO configuration.

    The real network remains:
        sq.net.xml

    The generated scenario provides:
        flow.xml
        vtype.xml
    """

    config_file = output_dir / "scenario.sumo.cfg"

    begin = int(scenario["simulation_begin"])
    end = int(scenario["simulation_end"])

    config_text = f"""<?xml version="1.0" encoding="UTF-8"?>

<configuration>

    <input>
        <net-file value="{NETWORK_FILE.resolve()}"/>
        <route-files value="{(scenario_dir / "flow.xml").resolve()}"/>
        <additional-files value="{(scenario_dir / "vtype.xml").resolve()}"/>
    </input>

    <time>
        <begin value="{begin}"/>
        <end value="{end}"/>
        <step-length value="1.0"/>
    </time>

    <processing>
        <time-to-teleport value="-1"/>
    </processing>

    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
    </report>

</configuration>
"""

    with open(config_file, "w", encoding="utf-8") as f:
        f.write(config_text)

    return config_file


# ============================================================================
# RUN ONE SCENARIO
# ============================================================================

def run_one_scenario(
    scenario_dir: Path,
    use_gui: bool = False,
) -> dict:

    # ------------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------------

    validate_scenario_files(scenario_dir)

    scenario = load_scenario_metadata(scenario_dir)

    scenario_id = scenario["scenario_id"]

    seed = int(scenario["seed"])

    begin = int(scenario["simulation_begin"])

    end = int(scenario["simulation_end"])


    # ------------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------------

    output_dir = scenario_dir / "raw_output"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ------------------------------------------------------------------------
    # Output files
    # ------------------------------------------------------------------------

    trajectory_file = (
        output_dir / "vehicle_trajectories.csv"
    )

    summary_file = (
        output_dir / "simulation_summary.json"
    )


    # ------------------------------------------------------------------------
    # Build temporary SUMO config
    # ------------------------------------------------------------------------

    config_file = build_sumo_config(
        scenario_dir=scenario_dir,
        scenario=scenario,
        output_dir=output_dir,
    )


    # ------------------------------------------------------------------------
    # Find SUMO
    # ------------------------------------------------------------------------

    sumo_binary = get_sumo_binary(use_gui)


    # ------------------------------------------------------------------------
    # SUMO command
    # ------------------------------------------------------------------------

    sumo_cmd = [
        sumo_binary,

        "-c",
        str(config_file),

        "--seed",
        str(seed),

        "--step-length",
        "1.0",

        "--quit-on-end",
    ]


    # ------------------------------------------------------------------------
    # Print scenario information
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print(f"RUNNING {scenario_id}")
    print("=" * 80)

    print(
        f"Scenario seed        : {seed}"
    )

    print(
        f"Demand rate          : "
        f"{scenario['demand_rate_veh_per_hour']} veh/h"
    )

    print(
        f"Demand class         : "
        f"{scenario['demand_class']}"
    )

    print(
        f"Approach pattern     : "
        f"{scenario['approach_pattern']}"
    )

    print(
        f"Movement pattern     : "
        f"{scenario['movement_pattern']}"
    )

    print(
        f"Composition pattern  : "
        f"{scenario['composition_pattern']}"
    )

    print(
        f"Arrival pattern      : "
        f"{scenario['arrival_pattern']}"
    )

    print(
        f"Simulation            : "
        f"{begin} -> {end} s"
    )

    print()

    print(
        f"Network              : "
        f"{NETWORK_FILE}"
    )

    print(
        f"Flow                 : "
        f"{scenario_dir / 'flow.xml'}"
    )

    print(
        f"Vehicle types        : "
        f"{scenario_dir / 'vtype.xml'}"
    )

    print()

    print(
        f"Raw output           : "
        f"{trajectory_file}"
    )

    print("=" * 80)
    print()


    # ------------------------------------------------------------------------
    # CSV writer
    # ------------------------------------------------------------------------

    # FIX 1:
    # Added lane_length_m and distance_from_stop_line_m.
    #
    # FIX 2:
    # Added explicit edge classification columns.

    trajectory_columns = [
        "timestamp",
        "vehicle_id",
        "vehicle_type",

        "edge_id",
        "lane_id",

        "is_internal_edge",
        "is_approach_edge",

        "lane_position_m",
        "lane_length_m",
        "distance_from_stop_line_m",

        "speed_mps",
        "acceleration_mps2",
        "waiting_time_s",

        "x",
        "y",
        "angle_deg",
    ]


    vehicle_rows = 0

    unique_vehicle_ids = set()

    max_active_vehicles = 0

    total_departed = 0

    total_arrived = 0


    # ------------------------------------------------------------------------
    # Lane-length cache
    # ------------------------------------------------------------------------

    # Lane geometry does not change during a SUMO run, so once we obtain
    # a lane's length we can reuse it for every vehicle/time step.
    lane_length_cache = {}


    # ------------------------------------------------------------------------
    # Open CSV
    # ------------------------------------------------------------------------

    with open(
        trajectory_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow(
            trajectory_columns
        )


        # --------------------------------------------------------------------
        # Start SUMO + TraCI
        # --------------------------------------------------------------------

        traci.start(sumo_cmd)

        try:

            while traci.simulation.getTime() < end:

                # ------------------------------------------------------------
                # FIX 3:
                #
                # SUMO can finish naturally before the configured `end`
                # when there are no vehicles currently active and no future
                # vehicles expected.
                #
                # Check this before asking SUMO for another simulation step.
                # ------------------------------------------------------------

                if (
                    traci.simulation.getMinExpectedNumber()
                    == 0
                ):

                    print(
                        f"SUMO finished early at "
                        f"{traci.simulation.getTime()} s."
                    )

                    break


                # ------------------------------------------------------------
                # Advance SUMO by one second
                # ------------------------------------------------------------

                traci.simulationStep()

                current_time = (
                    traci.simulation.getTime()
                )


                # ------------------------------------------------------------
                # Vehicle IDs currently inside the simulation
                # ------------------------------------------------------------

                vehicle_ids = (
                    traci.vehicle.getIDList()
                )

                active_vehicle_count = (
                    len(vehicle_ids)
                )

                max_active_vehicles = max(
                    max_active_vehicles,
                    active_vehicle_count,
                )


                # ------------------------------------------------------------
                # Record every active vehicle
                # ------------------------------------------------------------

                for vehicle_id in vehicle_ids:

                    vehicle_rows += 1

                    unique_vehicle_ids.add(
                        vehicle_id
                    )


                    # --------------------------------------------------------
                    # Vehicle identity
                    # --------------------------------------------------------

                    vehicle_type = (
                        traci.vehicle.getTypeID(
                            vehicle_id
                        )
                    )


                    # --------------------------------------------------------
                    # SUMO network position
                    # --------------------------------------------------------

                    edge_id = (
                        traci.vehicle.getRoadID(
                            vehicle_id
                        )
                    )

                    lane_id = (
                        traci.vehicle.getLaneID(
                            vehicle_id
                        )
                    )


                    # --------------------------------------------------------
                    # FIX 2:
                    #
                    # Explicitly classify internal junction edges.
                    #
                    # Internal SUMO edges look like:
                    #     :0_0
                    #     :0_1
                    #
                    # They are NOT approach edges.
                    # --------------------------------------------------------

                    internal_edge = (
                        is_internal_edge(edge_id)
                    )

                    approach_edge = (
                        edge_id in APPROACH_EDGES
                    )


                    # --------------------------------------------------------
                    # Lane position
                    #
                    # TraCI returns distance from the START of the lane.
                    # --------------------------------------------------------

                    lane_position = (
                        traci.vehicle.getLanePosition(
                            vehicle_id
                        )
                    )


                    # --------------------------------------------------------
                    # FIX 1:
                    #
                    # Capture lane length while TraCI is already connected.
                    #
                    # This avoids needing another connection later.
                    # --------------------------------------------------------

                    if lane_id:

                        if lane_id not in lane_length_cache:

                            lane_length_cache[lane_id] = (
                                traci.lane.getLength(
                                    lane_id
                                )
                            )

                        lane_length = (
                            lane_length_cache[lane_id]
                        )

                    else:

                        lane_length = None


                    # --------------------------------------------------------
                    # FIX 1:
                    #
                    # Convert lane position into distance from the END of
                    # the lane.
                    #
                    # For incoming approach lanes, the end corresponds to
                    # the junction/stop-line side.
                    #
                    # IMPORTANT:
                    # This is a geometric measurement.
                    # It is NOT yet a queue-length calculation.
                    # --------------------------------------------------------

                    if (
                        approach_edge
                        and lane_length is not None
                    ):

                        distance_from_stop_line = (
                            lane_length
                            - lane_position
                        )

                    else:

                        distance_from_stop_line = None


                    # --------------------------------------------------------
                    # Kinematics
                    # --------------------------------------------------------

                    speed = (
                        traci.vehicle.getSpeed(
                            vehicle_id
                        )
                    )

                    acceleration = (
                        traci.vehicle.getAcceleration(
                            vehicle_id
                        )
                    )

                    waiting_time = (
                        traci.vehicle.getAccumulatedWaitingTime(
                            vehicle_id
                        )
                    )


                    # --------------------------------------------------------
                    # World coordinates
                    # --------------------------------------------------------

                    x, y = (
                        traci.vehicle.getPosition(
                            vehicle_id
                        )
                    )

                    angle = (
                        traci.vehicle.getAngle(
                            vehicle_id
                        )
                    )


                    # --------------------------------------------------------
                    # Write row
                    # --------------------------------------------------------

                    writer.writerow([
                        current_time,

                        vehicle_id,

                        vehicle_type,

                        edge_id,

                        lane_id,

                        int(internal_edge),

                        int(approach_edge),

                        round(
                            lane_position,
                            4,
                        ),

                        (
                            round(
                                lane_length,
                                4,
                            )
                            if lane_length is not None
                            else ""
                        ),

                        (
                            round(
                                distance_from_stop_line,
                                4,
                            )
                            if distance_from_stop_line is not None
                            else ""
                        ),

                        round(
                            speed,
                            4,
                        ),

                        round(
                            acceleration,
                            4,
                        ),

                        round(
                            waiting_time,
                            4,
                        ),

                        round(
                            x,
                            4,
                        ),

                        round(
                            y,
                            4,
                        ),

                        round(
                            angle,
                            4,
                        ),
                    ])


                # ------------------------------------------------------------
                # Track departures/arrivals
                # ------------------------------------------------------------

                departed = (
                    traci.simulation.getDepartedIDList()
                )

                arrived = (
                    traci.simulation.getArrivedIDList()
                )

                total_departed += len(
                    departed
                )

                total_arrived += len(
                    arrived
                )


                # ------------------------------------------------------------
                # Progress display
                # ------------------------------------------------------------

                if int(current_time) % 100 == 0:

                    print(
                        f"{scenario_id}: "
                        f"{int(current_time)}/{end} s | "
                        f"active vehicles={active_vehicle_count} | "
                        f"rows={vehicle_rows}"
                    )


        finally:

            traci.close()


    # ------------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------------

    summary = {

        "scenario_id":
            scenario_id,

        "seed":
            seed,

        "simulation_begin_s":
            begin,

        "simulation_end_s":
            end,

        "step_length_s":
            1.0,

        "configured_demand_rate_veh_per_hour":
            scenario[
                "demand_rate_veh_per_hour"
            ],

        "demand_class":
            scenario[
                "demand_class"
            ],

        "approach_pattern":
            scenario[
                "approach_pattern"
            ],

        "movement_pattern":
            scenario[
                "movement_pattern"
            ],

        "composition_pattern":
            scenario[
                "composition_pattern"
            ],

        "arrival_pattern":
            scenario[
                "arrival_pattern"
            ],

        "total_departed_vehicles":
            total_departed,

        "total_arrived_vehicles":
            total_arrived,

        "unique_vehicle_ids_recorded":
            len(unique_vehicle_ids),

        "trajectory_rows":
            vehicle_rows,

        "maximum_active_vehicles":
            max_active_vehicles,

        "trajectory_file":
            str(
                trajectory_file.resolve()
            ),

        "note": (
            "Raw SUMO ground truth only. "
            "No queue, density, flow, shockwave, "
            "camera or GPS features were calculated."
        ),
    }


    with open(
        summary_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )


    # ------------------------------------------------------------------------
    # Finished
    # ------------------------------------------------------------------------

    print()

    print(
        f"{scenario_id} COMPLETE"
    )

    print(
        f"Trajectory rows : "
        f"{vehicle_rows}"
    )

    print(
        f"Vehicles seen   : "
        f"{len(unique_vehicle_ids)}"
    )

    print(
        f"Departed        : "
        f"{total_departed}"
    )

    print(
        f"Arrived         : "
        f"{total_arrived}"
    )

    print()

    print(
        "Trajectory file:"
    )

    print(
        f"  {trajectory_file}"
    )

    print()

    print(
        "Summary file:"
    )

    print(
        f"  {summary_file}"
    )

    print()

    return summary


# ============================================================================
# FIND SCENARIOS
# ============================================================================

def find_scenarios() -> list[Path]:
    """
    Return scenario_XXXX folders in numerical order.
    """

    scenarios = sorted(
        SCENARIOS_DIR.glob(
            "scenario_*"
        )
    )

    scenarios = [
        path
        for path in scenarios
        if path.is_dir()
    ]

    return scenarios


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run ASTRID generated scenarios through SUMO "
            "and save raw vehicle trajectories."
        )
    )


    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help=(
            "Run one scenario, e.g. "
            "scenario_0001. "
            "If omitted, all scenarios are run."
        ),
    )


    parser.add_argument(
        "--gui",
        action="store_true",
        help=(
            "Use SUMO-GUI instead of command-line SUMO."
        ),
    )


    args = parser.parse_args()


    # ------------------------------------------------------------------------
    # Validate network
    # ------------------------------------------------------------------------

    if not NETWORK_FILE.exists():

        print()

        print(
            "ERROR: Network file does not exist:"
        )

        print(
            NETWORK_FILE
        )

        sys.exit(1)


    # ------------------------------------------------------------------------
    # Determine scenarios
    # ------------------------------------------------------------------------

    if args.scenario:

        scenario_dir = (
            SCENARIOS_DIR
            / args.scenario
        )

        if not scenario_dir.exists():

            print()

            print(
                f"ERROR: Scenario does not exist: "
                f"{args.scenario}"
            )

            sys.exit(1)

        scenarios = [
            scenario_dir
        ]

    else:

        scenarios = find_scenarios()


    if not scenarios:

        print()

        print(
            "ERROR: No generated scenarios found."
        )

        print()

        print(
            f"Expected folders inside: "
            f"{SCENARIOS_DIR}"
        )

        sys.exit(1)


    # ------------------------------------------------------------------------
    # Run scenarios
    # ------------------------------------------------------------------------

    print()

    print("=" * 80)

    print(
        "ASTRID SUMO SCENARIO RUNNER"
    )

    print("=" * 80)

    print(
        f"Scenarios found: "
        f"{len(scenarios)}"
    )

    print()


    successful = 0

    failed = []


    for scenario_dir in scenarios:

        try:

            run_one_scenario(
                scenario_dir,
                use_gui=args.gui,
            )

            successful += 1

        except Exception as exc:

            print()

            print("=" * 80)

            print(
                f"FAILED: "
                f"{scenario_dir.name}"
            )

            print("=" * 80)

            print(
                str(exc)
            )

            print()

            failed.append({
                "scenario_id":
                    scenario_dir.name,

                "error":
                    str(exc),
            })


    # ------------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------------

    print()

    print("=" * 80)

    print(
        "RUN COMPLETE"
    )

    print("=" * 80)

    print(
        f"Successful : "
        f"{successful}"
    )

    print(
        f"Failed     : "
        f"{len(failed)}"
    )

    print()


    if failed:

        print(
            "Failed scenarios:"
        )

        for item in failed:

            print(
                f"  {item['scenario_id']}: "
                f"{item['error']}"
            )

        print()


if __name__ == "__main__":
    main()