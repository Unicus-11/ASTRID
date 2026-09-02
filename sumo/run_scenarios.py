"""
run_scenarios.py
================
ASTRID Prototype -- run generated scenarios through SUMO, save raw
vehicle trajectories, signal-phase state, AND a demand-realization audit.

v0.3 changes (this investigation):
    Several scenarios do not realize the demand specified in their own
    scenario.json. Two distinct, separately-confirmed causes:

    1. GENERATION BUG (burst pattern only, scenario_0002/0012): fixed in
       scenario_builder.py v0.4 -- build_flow_xml() now writes flow entries
       in globally-sorted time order instead of per-movement order. That
       bug meant the flow.xml file itself didn't schedule the vehicles it
       claimed to (SUMO silently dropped out-of-order flow definitions).

    2. NETWORK CAPACITY / INSERTION CONSTRAINT (constant pattern,
       several scenarios, e.g. 0003's left-turn movements): the flow.xml
       correctly schedules the right number of vehicles, but SUMO can't
       insert all of them onto the network before the simulation ends
       (queues/lane/signal capacity limits, and time-to-teleport=-1 means
       blocked vehicles are never force-inserted -- they just wait).

    These need different fixes (or, for #2, an explicit decision to treat
    the scenario as intentionally demand-constrained) and this audit is
    what distinguishes them automatically per scenario, rather than
    requiring a manual investigation each time:

        requested_vehicle_count   = demand_rate_veh_per_hour * duration_hours
                                     (from scenario.json -- what was asked for)
        scheduled_vehicle_count   = integrated from the ACTUAL flow.xml used
                                     for this run (vehsPerHour * duration per
                                     <flow> element) -- independent check of
                                     whether the generation step itself
                                     produced a correct file
        departed_vehicle_count    = vehicles TraCI actually inserted (as before)
        pending_never_inserted    = vehicles SUMO LOADED from flow.xml but
                                     never managed to insert onto the network
                                     before the run ended (still queued,
                                     waiting for a gap) -- this is the
                                     "insertion-delay/pending-vehicle"
                                     diagnostic: it tells you whether unmet
                                     demand was discarded (scheduling bug) or
                                     genuinely stuck waiting (capacity limit)

    Status is classified as one of: no_demand, scheduling_shortfall,
    within_tolerance, capacity_constrained, unexplained_shortfall. See
    REALIZATION_STATUS_MEANINGS below for what each means and what to do
    about it.

    IMPORTANT: this script does NOT change simulation behavior, does NOT
    force more vehicles in, and does NOT alter demand. It only measures
    and reports what happened, per your own "smallest scientifically
    correct change" plan -- the decision of whether a capacity-constrained
    scenario should be lowered in demand, given a longer simulation window,
    or explicitly kept and relabeled as a demand-constrained experiment is
    yours to make from this report, not something this script decides.

v0.3.1 changes (simulation completion logic fix):
    The run previously stepped SUMO only across the scenario's own
    observation window (simulation_begin -> simulation_end, e.g. 0 ->
    3600s) and, within that same window, would also break out early the
    moment traci.simulation.getMinExpectedNumber() reported 0 -- using
    "no vehicles expected" as if it were the definition of a successfully
    realized/completed run. That conflated two different things: (a) the
    fixed 3600s window that DEFINES the dataset (what gets written to
    vehicle_trajectories.csv / tls_state.csv), and (b) whether vehicles
    generated inside that window actually got to finish their routes.
    Vehicles still in transit at simulation_end were simply cut off, with
    no chance to arrive, which understated departed/arrived counts and
    made capacity_constrained look worse than it was.

    Fixed by splitting the run into two explicit phases:

    Phase 1 -- DATA-COLLECTION PERIOD (0 -> simulation_end, e.g. 3600s):
        The only period for which trajectory rows and TLS-state rows are
        written to disk. This defines the dataset and is unchanged in
        length, demand, routes, signal timing, or teleport settings.

    Phase 2 -- CLEARANCE PERIOD (simulation_end -> simulation_end +
        CLEARANCE_DURATION_S, e.g. 3600 -> 7200s, MAXIMUM only):
        No new vehicles are scheduled here (flow.xml has nothing beyond
        simulation_end), and no trajectory/TLS rows are written -- this
        phase exists solely to let vehicles that already departed during
        Phase 1 finish clearing the network. The loop stops as soon as
        traci.simulation.getMinExpectedNumber() == 0 (a SECONDARY
        completion check -- "nothing left to clear"), or at
        simulation_end + CLEARANCE_DURATION_S, whichever comes first.
        getMinExpectedNumber() is never used to define successful demand
        realization -- that determination is made purely from
        requested_vehicle_count / scheduled_vehicle_count /
        departed_vehicle_count / pending_never_inserted_count, as before.

    departed_vehicle_ids and arrived_vehicle_ids are now tracked
    explicitly as sets (across both phases) instead of via running
    integer counters, and those sets are what feed the realization audit
    and the pending-never-inserted diagnostic.

This script still does NOT calculate queue/density/flow/shockwave, and
does NOT build camera/GPS/ML features. Raw ground truth + audit only.

Run:
    python sumo/run_scenarios.py --scenario scenario_0001
    python sumo/run_scenarios.py                       # all scenarios
    python sumo/run_scenarios.py --scenario scenario_0001 --gui
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import xml.etree.ElementTree as ET
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

APPROACH_EDGES = {"1i", "2i", "3i", "4i"}


def is_internal_edge(edge_id: str) -> bool:
    return edge_id.startswith(":")


# ============================================================================
# SIGNAL (TLS) CONSTANTS -- must match normal_controller.py
# ============================================================================

TLS_ID = "0"


def validate_traffic_light() -> None:
    tls_ids = traci.trafficlight.getIDList()
    if TLS_ID not in tls_ids:
        raise RuntimeError(f"Traffic light '{TLS_ID}' not found in SUMO. Available: {tls_ids}")


# ============================================================================
# SIMULATION TIMING -- data-collection period vs. clearance period
# ============================================================================
#
# Vehicles are generated only during the scenario's own observation window
# (scenario.json's simulation_begin -> simulation_end, e.g. 0 -> 3600s).
# That window is ALSO the only period for which trajectory/TLS rows are
# written to disk -- it defines the dataset and its length is untouched.
#
# CLEARANCE_DURATION_S extends the SUMO run (via TraCI stepping, NOT by
# scheduling any new vehicles -- flow.xml has nothing beyond
# simulation_end) so vehicles already generated inside the observation
# window get a chance to finish their route instead of being cut off. This
# is a MAXIMUM cap, not a target: the clearance loop stops as soon as no
# vehicles are left expected (a secondary completion check), or at
# simulation_end + CLEARANCE_DURATION_S, whichever comes first.
CLEARANCE_DURATION_S = 3600


# ============================================================================
# SUMO COMMAND
# ============================================================================

def get_sumo_binary(use_gui: bool) -> str:
    binary_name = "sumo-gui" if use_gui else "sumo"
    binary = shutil.which(binary_name)
    if binary is None:
        print()
        print(f"ERROR: '{binary_name}' was not found in PATH.")
        print()
        print("Check that SUMO is installed and that this works:")
        print(f"    {binary_name} --version")
        print()
        sys.exit(1)
    return binary


# ============================================================================
# LOAD SCENARIO METADATA
# ============================================================================

def load_scenario_metadata(scenario_dir: Path) -> dict:
    scenario_file = scenario_dir / "scenario.json"
    if not scenario_file.exists():
        raise FileNotFoundError(f"Missing scenario metadata: {scenario_file}")
    with open(scenario_file, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# VALIDATE SCENARIO FILES
# ============================================================================

def validate_scenario_files(scenario_dir: Path) -> None:
    required_files = [
        scenario_dir / "scenario.json",
        scenario_dir / "flow.xml",
        scenario_dir / "vtype.xml",
        NETWORK_FILE,
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(f"  - {p}" for p in missing))


# ============================================================================
# BUILD TEMPORARY SUMO CONFIG
# ============================================================================

def build_sumo_config(scenario_dir: Path, scenario: dict, output_dir: Path) -> Path:
    """The configured <end> is the MAXIMUM clearance time (simulation_end +
    CLEARANCE_DURATION_S), not the dataset's simulation_end, so that
    --quit-on-end / SUMO's own internal end check never cuts the run off
    before the clearance phase has had its full window to drain already-
    departed vehicles. Demand, routes, and signal timing are untouched --
    this only changes how long SUMO is told it is allowed to keep
    stepping."""
    config_file = output_dir / "scenario.sumo.cfg"
    begin = int(scenario["simulation_begin"])
    end = int(scenario["simulation_end"])
    max_clearance_time = end + CLEARANCE_DURATION_S

    config_text = f"""<?xml version="1.0" encoding="UTF-8"?>

<configuration>

    <input>
        <net-file value="{NETWORK_FILE.resolve()}"/>
        <route-files value="{(scenario_dir / "flow.xml").resolve()}"/>
        <additional-files value="{(scenario_dir / "vtype.xml").resolve()}"/>
    </input>

    <time>
        <begin value="{begin}"/>
        <end value="{max_clearance_time}"/>
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
# REALIZATION AUDIT
# ============================================================================

REALIZATION_STATUS_MEANINGS = {
    "no_demand": "requested_vehicle_count is ~0 -- nothing to realize, not a problem.",
    "scheduling_shortfall": (
        "The flow.xml file used for THIS run does not itself integrate to the requested demand "
        "(scheduled_vehicle_count is well below requested_vehicle_count). This is a scenario-"
        "GENERATION bug (check scenario_builder.py's build_flow_xml, e.g. the burst-ordering bug "
        "fixed in v0.4) -- independent of whether SUMO could insert the vehicles. Regenerate this "
        "scenario after fixing the generator; do not try to fix it by changing run_scenarios.py."
    ),
    "within_tolerance": "departed_vehicle_count is within tolerance of requested_vehicle_count. No action needed.",
    "capacity_constrained": (
        "flow.xml schedules ~the right number of vehicles (scheduled ~= requested), but a "
        "meaningful share of them were LOADED and still waiting for network insertion "
        "(pending_never_inserted > 0) even after the clearance period ended -- blocked by "
        "queue/lane/signal capacity, not discarded. time-to-teleport=-1 means they're never "
        "force-inserted. This is a real network-capacity limitation, not a bug. Options: lower "
        "demand, adjust movement splits/signal timing, lengthen the clearance window, or "
        "explicitly record this scenario as an intentionally demand-constrained experiment "
        "(requested != realized) and use the REALIZED numbers for any downstream analysis/labels."
    ),
    "unexplained_shortfall": (
        "scheduled_vehicle_count matches requested, and few/no vehicles were left pending "
        "insertion, but departed_vehicle_count is still short. Doesn't fit either known cause -- "
        "needs manual investigation (check for routing failures, invalid route edges, vType issues)."
    ),
}

REALIZATION_TOLERANCE = 0.95


def parse_scheduled_vehicle_count(flow_xml_path: Path) -> float:
    """Sums vehsPerHour * (end-begin)/3600 over every <flow> element in the
    ACTUAL route file used for this run. This is independent of write
    order -- it tells you what the file claims to schedule, regardless of
    whether SUMO's parser will honor all of it (that's a separate check,
    see the burst-ordering bug this file's docstring describes)."""
    tree = ET.parse(flow_xml_path)
    root = tree.getroot()
    total = 0.0
    for flow in root.findall("flow"):
        veh_per_hour = float(flow.get("vehsPerHour", 0) or 0)
        seg_begin = float(flow.get("begin", 0) or 0)
        seg_end = float(flow.get("end", 0) or 0)
        total += veh_per_hour * max(seg_end - seg_begin, 0.0) / 3600.0
    return total


def classify_realization(requested: float, scheduled: float, departed: int, pending_never_inserted: int) -> str:
    """Definition of realization success/failure. Deliberately does NOT
    consult traci.simulation.getMinExpectedNumber() anywhere -- that value
    is only ever used in the clearance loop as a secondary "can we stop
    stepping early" check, never as part of what counts as a realized
    scenario."""
    if requested <= 0:
        return "no_demand"
    if scheduled < requested * REALIZATION_TOLERANCE:
        return "scheduling_shortfall"
    if departed >= requested * REALIZATION_TOLERANCE:
        return "within_tolerance"
    shortfall = max(requested - departed, 0)
    if shortfall > 0 and pending_never_inserted >= shortfall * 0.5:
        return "capacity_constrained"
    return "unexplained_shortfall"


def build_realization_audit(requested: float, scheduled: float, departed: int, loaded: int,
                             pending_never_inserted: int) -> dict:
    status = classify_realization(requested, scheduled, departed, pending_never_inserted)
    return {
        "requested_vehicle_count": round(requested, 1),
        "scheduled_vehicle_count_from_flow_xml": round(scheduled, 1),
        "scheduled_vs_requested_ratio": round(scheduled / requested, 4) if requested > 0 else None,
        "departed_vehicle_count": departed,
        "realization_ratio_departed_over_requested": round(departed / requested, 4) if requested > 0 else None,
        "loaded_vehicle_count": loaded,
        "pending_never_inserted_count": pending_never_inserted,
        "status": status,
        "_interpretation": REALIZATION_STATUS_MEANINGS[status],
    }


# ============================================================================
# RUN ONE SCENARIO
# ============================================================================

def run_one_scenario(scenario_dir: Path, use_gui: bool = False) -> dict:

    validate_scenario_files(scenario_dir)
    scenario = load_scenario_metadata(scenario_dir)

    scenario_id = scenario["scenario_id"]
    seed = int(scenario["seed"])
    begin = int(scenario["simulation_begin"])
    end = int(scenario["simulation_end"])                 # main data-collection period end (e.g. 3600s)
    max_clearance_time = end + CLEARANCE_DURATION_S        # maximum wall-clock cap (e.g. 7200s)
    duration_hours = (end - begin) / 3600.0
    requested_vehicle_count = float(scenario["demand_rate_veh_per_hour"]) * duration_hours

    output_dir = scenario_dir / "raw_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_file = output_dir / "vehicle_trajectories.csv"
    tls_state_file = output_dir / "tls_state.csv"
    summary_file = output_dir / "simulation_summary.json"
    realization_audit_file = output_dir / "realization_audit.json"

    config_file = build_sumo_config(scenario_dir=scenario_dir, scenario=scenario, output_dir=output_dir)
    sumo_binary = get_sumo_binary(use_gui)

    scheduled_vehicle_count = parse_scheduled_vehicle_count(scenario_dir / "flow.xml")

    sumo_cmd = [
        sumo_binary,
        "-c", str(config_file),
        "--seed", str(seed),
        "--step-length", "1.0",
        "--quit-on-end",
    ]

    print()
    print("=" * 80)
    print(f"RUNNING {scenario_id}")
    print("=" * 80)
    print(f"Scenario seed        : {seed}")
    print(f"Demand rate          : {scenario['demand_rate_veh_per_hour']} veh/h")
    print(f"Demand class         : {scenario['demand_class']}")
    print(f"Approach pattern     : {scenario['approach_pattern']}")
    print(f"Movement pattern     : {scenario['movement_pattern']}")
    print(f"Composition pattern  : {scenario['composition_pattern']}")
    print(f"Arrival pattern      : {scenario['arrival_pattern']}")
    print(f"Data-collection period : {begin} -> {end} s (this defines the dataset)")
    print(f"Clearance period (max) : {end} -> {max_clearance_time} s "
          f"(no new vehicles, no data saved -- lets departed vehicles finish)")
    print(f"Requested vehicles   : {requested_vehicle_count:.0f}")
    print(f"Scheduled (flow.xml) : {scheduled_vehicle_count:.0f}"
          + ("  <-- MISMATCH with requested, check scenario_builder.py generation"
             if scheduled_vehicle_count < requested_vehicle_count * REALIZATION_TOLERANCE else ""))
    print()
    print(f"Network              : {NETWORK_FILE}")
    print(f"Flow                 : {scenario_dir / 'flow.xml'}")
    print(f"Vehicle types        : {scenario_dir / 'vtype.xml'}")
    print()
    print(f"Raw trajectories     : {trajectory_file}")
    print(f"Raw signal state     : {tls_state_file}")
    print(f"Realization audit    : {realization_audit_file}")
    print("=" * 80)
    print()

    trajectory_columns = [
        "timestamp", "vehicle_id", "vehicle_type",
        "edge_id", "lane_id",
        "is_internal_edge", "is_approach_edge",
        "lane_position_m", "lane_length_m", "distance_from_stop_line_m",
        "speed_mps", "acceleration_mps2", "waiting_time_s",
        "x", "y", "angle_deg",
    ]
    tls_columns = ["timestamp", "phase", "state"]

    vehicle_rows = 0
    unique_vehicle_ids = set()          # vehicles recorded in trajectory rows (data-collection period only)
    loaded_ids_ever = set()             # everything SUMO ever loaded from flow.xml, across both phases
    departed_vehicle_ids = set()        # everything TraCI actually inserted onto the network, across both phases
    arrived_vehicle_ids = set()         # everything that completed its route and arrived, across both phases
    max_active_vehicles = 0
    tls_rows = 0

    lane_length_cache: dict[str, float] = {}

    with open(trajectory_file, "w", newline="", encoding="utf-8") as csv_file, \
         open(tls_state_file, "w", newline="", encoding="utf-8") as tls_file:

        writer = csv.writer(csv_file)
        writer.writerow(trajectory_columns)

        tls_writer = csv.writer(tls_file)
        tls_writer.writerow(tls_columns)

        traci.start(sumo_cmd)

        try:
            validate_traffic_light()

            # ----------------------------------------------------------------
            # PHASE 1 -- DATA-COLLECTION PERIOD (begin -> end, e.g. 0 -> 3600s)
            # This is the ONLY period for which trajectory/TLS rows are
            # written. Runs the full window every time; not cut short by
            # getMinExpectedNumber() (that check belongs to the clearance
            # phase only).
            # ----------------------------------------------------------------
            while traci.simulation.getTime() < end:

                traci.simulationStep()
                current_time = traci.simulation.getTime()

                phase = traci.trafficlight.getPhase(TLS_ID)
                state = traci.trafficlight.getRedYellowGreenState(TLS_ID)
                tls_writer.writerow([current_time, phase, state])
                tls_rows += 1

                loaded_ids_ever.update(traci.simulation.getLoadedIDList())

                vehicle_ids = traci.vehicle.getIDList()
                active_vehicle_count = len(vehicle_ids)
                max_active_vehicles = max(max_active_vehicles, active_vehicle_count)

                for vehicle_id in vehicle_ids:
                    vehicle_rows += 1
                    unique_vehicle_ids.add(vehicle_id)

                    vehicle_type = traci.vehicle.getTypeID(vehicle_id)
                    edge_id = traci.vehicle.getRoadID(vehicle_id)
                    lane_id = traci.vehicle.getLaneID(vehicle_id)

                    internal_edge = is_internal_edge(edge_id)
                    approach_edge = edge_id in APPROACH_EDGES

                    lane_position = traci.vehicle.getLanePosition(vehicle_id)

                    if lane_id:
                        if lane_id not in lane_length_cache:
                            lane_length_cache[lane_id] = traci.lane.getLength(lane_id)
                        lane_length = lane_length_cache[lane_id]
                    else:
                        lane_length = None

                    if approach_edge and lane_length is not None:
                        distance_from_stop_line = lane_length - lane_position
                    else:
                        distance_from_stop_line = None

                    speed = traci.vehicle.getSpeed(vehicle_id)
                    acceleration = traci.vehicle.getAcceleration(vehicle_id)
                    waiting_time = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
                    x, y = traci.vehicle.getPosition(vehicle_id)
                    angle = traci.vehicle.getAngle(vehicle_id)

                    writer.writerow([
                        current_time, vehicle_id, vehicle_type,
                        edge_id, lane_id,
                        int(internal_edge), int(approach_edge),
                        round(lane_position, 4),
                        round(lane_length, 4) if lane_length is not None else "",
                        round(distance_from_stop_line, 4) if distance_from_stop_line is not None else "",
                        round(speed, 4), round(acceleration, 4), round(waiting_time, 4),
                        round(x, 4), round(y, 4), round(angle, 4),
                    ])

                departed = traci.simulation.getDepartedIDList()
                arrived = traci.simulation.getArrivedIDList()
                departed_vehicle_ids.update(departed)
                arrived_vehicle_ids.update(arrived)

                if int(current_time) % 100 == 0:
                    print(
                        f"{scenario_id}: [data-collection] {int(current_time)}/{end} s | "
                        f"active vehicles={active_vehicle_count} | rows={vehicle_rows}"
                    )

            print(f"{scenario_id}: data-collection period complete at "
                  f"{traci.simulation.getTime()}s ({vehicle_rows} trajectory rows, {tls_rows} TLS rows).")

            # ----------------------------------------------------------------
            # PHASE 2 -- CLEARANCE PERIOD (end -> max_clearance_time, max
            # 3600s more). No new vehicles are scheduled (flow.xml has
            # nothing beyond `end`); no trajectory/TLS rows are written.
            # Only purpose: let vehicles that already departed in Phase 1
            # finish clearing the network. getMinExpectedNumber() == 0 is
            # used here ONLY as a secondary "can we stop early" check --
            # never as the definition of successful demand realization.
            # ----------------------------------------------------------------
            while traci.simulation.getTime() < max_clearance_time:

                if traci.simulation.getMinExpectedNumber() == 0:
                    print(f"{scenario_id}: all vehicles cleared at "
                          f"{traci.simulation.getTime()}s (secondary completion check) -- "
                          f"stopping clearance phase early.")
                    break

                traci.simulationStep()
                current_time = traci.simulation.getTime()

                loaded_ids_ever.update(traci.simulation.getLoadedIDList())
                departed = traci.simulation.getDepartedIDList()
                arrived = traci.simulation.getArrivedIDList()
                departed_vehicle_ids.update(departed)
                arrived_vehicle_ids.update(arrived)

                if int(current_time) % 100 == 0:
                    print(
                        f"{scenario_id}: [clearance] {int(current_time)}/{max_clearance_time} s | "
                        f"still expected={traci.simulation.getMinExpectedNumber()}"
                    )
        finally:
            traci.close()

    # Vehicles SUMO loaded from flow.xml but that never actually got
    # inserted onto the network (across both phases) -- the
    # insertion-delay/pending diagnostic.
    pending_never_inserted = loaded_ids_ever - departed_vehicle_ids
    realization_audit = build_realization_audit(
        requested=requested_vehicle_count,
        scheduled=scheduled_vehicle_count,
        departed=len(departed_vehicle_ids),
        loaded=len(loaded_ids_ever),
        pending_never_inserted=len(pending_never_inserted),
    )
    with open(realization_audit_file, "w", encoding="utf-8") as f:
        json.dump(realization_audit, f, indent=2)

    summary = {
        "scenario_id": scenario_id,
        "seed": seed,
        "simulation_begin_s": begin,
        "simulation_end_s": end,
        "clearance_max_time_s": max_clearance_time,
        "step_length_s": 1.0,
        "configured_demand_rate_veh_per_hour": scenario["demand_rate_veh_per_hour"],
        "demand_class": scenario["demand_class"],
        "approach_pattern": scenario["approach_pattern"],
        "movement_pattern": scenario["movement_pattern"],
        "composition_pattern": scenario["composition_pattern"],
        "arrival_pattern": scenario["arrival_pattern"],
        "total_departed_vehicles": len(departed_vehicle_ids),
        "total_arrived_vehicles": len(arrived_vehicle_ids),
        "unique_vehicle_ids_recorded": len(unique_vehicle_ids),
        "trajectory_rows": vehicle_rows,
        "tls_state_rows": tls_rows,
        "tls_id": TLS_ID,
        "maximum_active_vehicles": max_active_vehicles,
        "trajectory_file": str(trajectory_file.resolve()),
        "tls_state_file": str(tls_state_file.resolve()),
        "realization_audit": realization_audit,
        "note": (
            "Raw SUMO ground truth + signal state + demand-realization audit only. No queue, "
            "density, flow, shockwave, camera or GPS features were calculated. Trajectory/TLS rows "
            "cover only the data-collection period (simulation_begin -> simulation_end); the "
            "clearance period (up to simulation_end + CLEARANCE_DURATION_S) is not saved to disk "
            "and exists only to let already-departed vehicles finish their routes before the "
            "departed/arrived counts used in the realization audit are finalized."
        ),
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"{scenario_id} COMPLETE")
    print(f"Trajectory rows (data-collection only) : {vehicle_rows}")
    print(f"TLS state rows (data-collection only)  : {tls_rows}")
    print(f"Vehicles seen in trajectories           : {len(unique_vehicle_ids)}")
    print(f"Departed (both phases)                  : {len(departed_vehicle_ids)}")
    print(f"Arrived (both phases)                   : {len(arrived_vehicle_ids)}")
    print()
    print(f"REALIZATION AUDIT: status={realization_audit['status']}")
    print(f"  requested={realization_audit['requested_vehicle_count']}  "
          f"scheduled={realization_audit['scheduled_vehicle_count_from_flow_xml']}  "
          f"departed={realization_audit['departed_vehicle_count']}  "
          f"pending_never_inserted={realization_audit['pending_never_inserted_count']}")
    print(f"  {realization_audit['_interpretation']}")
    print()
    print("Trajectory file:")
    print(f"  {trajectory_file}")
    print("TLS state file:")
    print(f"  {tls_state_file}")
    print("Realization audit:")
    print(f"  {realization_audit_file}")
    print()
    print("Summary file:")
    print(f"  {summary_file}")
    print()

    return summary


# ============================================================================
# FIND SCENARIOS
# ============================================================================

def find_scenarios(split_filter: str = "all") -> list[Path]:
    """split_filter: 'all', 'dev' (train+val+test), 'ood', or a specific
    split name ('train'/'val'/'test'/'ood'). Reads each scenario's own
    scenario.json to filter -- added so OOD scenarios can be run/inspected
    separately from development ones without hand-picking scenario IDs."""
    scenarios = sorted(SCENARIOS_DIR.glob("scenario_*"))
    scenarios = [path for path in scenarios if path.is_dir()]
    if split_filter == "all":
        return scenarios
    filtered = []
    for path in scenarios:
        try:
            meta = load_scenario_metadata(path)
        except FileNotFoundError:
            continue
        split = meta.get("split")
        if split_filter == "dev" and split in ("train", "val", "test"):
            filtered.append(path)
        elif split_filter == split:
            filtered.append(path)
    return filtered


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ASTRID generated scenarios through SUMO and save raw trajectories + signal state + realization audit."
    )
    parser.add_argument("--scenario", type=str, default=None,
                         help="Run one scenario, e.g. scenario_normal_balanced. If omitted, runs per --split.")
    parser.add_argument("--split", type=str, default="all",
                         choices=["all", "dev", "train", "val", "test", "ood"],
                         help="Which scenarios to run when --scenario is omitted. 'dev' = train+val+test "
                              "(everything except OOD). Default: all.")
    parser.add_argument("--gui", action="store_true", help="Use SUMO-GUI instead of command-line SUMO.")
    args = parser.parse_args()

    if not NETWORK_FILE.exists():
        print()
        print("ERROR: Network file does not exist:")
        print(NETWORK_FILE)
        sys.exit(1)

    if args.scenario:
        scenario_dir = SCENARIOS_DIR / args.scenario
        if not scenario_dir.exists():
            print()
            print(f"ERROR: Scenario does not exist: {args.scenario}")
            sys.exit(1)
        scenarios = [scenario_dir]
    else:
        scenarios = find_scenarios(args.split)
        if args.split == "ood":
            print("NOTE: running OOD-only. Remember: OOD scenarios must never feed training, "
                  "feature fitting, tuning, or model selection -- final generalization test only.\n")

    if not scenarios:
        print()
        print("ERROR: No generated scenarios found.")
        print(f"Expected folders inside: {SCENARIOS_DIR}")
        sys.exit(1)

    print()
    print("=" * 80)
    print("ASTRID SUMO SCENARIO RUNNER")
    print("=" * 80)
    print(f"Scenarios found: {len(scenarios)}")
    print()

    successful = 0
    failed = []
    audits: dict[str, dict] = {}

    for scenario_dir in scenarios:
        try:
            summary = run_one_scenario(scenario_dir, use_gui=args.gui)
            successful += 1
            audit = summary["realization_audit"]
            try:
                audit["_split"] = load_scenario_metadata(scenario_dir).get("split", "?")
            except FileNotFoundError:
                audit["_split"] = "?"
            audits[scenario_dir.name] = audit
        except Exception as exc:
            print()
            print("=" * 80)
            print(f"FAILED: {scenario_dir.name}")
            print("=" * 80)
            print(str(exc))
            print()
            failed.append({"scenario_id": scenario_dir.name, "error": str(exc)})

    print()
    print("=" * 80)
    print("RUN COMPLETE")
    print("=" * 80)
    print(f"Successful : {successful}")
    print(f"Failed     : {len(failed)}")
    print()

    if audits:
        print("REALIZATION AUDIT SUMMARY (all scenarios run this session):")
        print(f"{'scenario_id':30} {'split':6} {'status':22} {'requested':>10} {'scheduled':>10} {'departed':>9} {'pending':>8}")
        dev_count, ood_count = 0, 0
        for sid, a in audits.items():
            split = a.get("_split", "?")
            if split == "ood":
                ood_count += 1
            else:
                dev_count += 1
            print(f"{sid:30} {split:6} {a['status']:22} {a['requested_vehicle_count']:>10} "
                  f"{a['scheduled_vehicle_count_from_flow_xml']:>10} {a['departed_vehicle_count']:>9} "
                  f"{a['pending_never_inserted_count']:>8}")
        print(f"\nRan {dev_count} development + {ood_count} OOD scenario(s) this session.")
        problem_statuses = {"scheduling_shortfall", "capacity_constrained", "unexplained_shortfall"}
        flagged = {sid: a for sid, a in audits.items() if a["status"] in problem_statuses}
        if flagged:
            print(f"\n{len(flagged)} scenario(s) did not realize their configured demand within tolerance:")
            for sid, a in flagged.items():
                print(f"  {sid}: {a['status']} -- {a['_interpretation']}")

    if failed:
        print("Failed scenarios:")
        for item in failed:
            print(f"  {item['scenario_id']}: {item['error']}")
        print()


if __name__ == "__main__":
    main()
