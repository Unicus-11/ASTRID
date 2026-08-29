"""
camera_simulator.py
====================
ASTRID Prototype -- Camera Sensor Simulator

RESPONSIBILITY:
    Take the enriched trajectory data (same per-vehicle enrichment
    ground_truth.py uses -- distance-to-stopline, is_queued) and apply
    the ONE limit a real camera has: it cannot see past camera_range_m
    from the stop line. Everything computed here uses ONLY vehicles
    within that range -- this file has no access to what's happening
    further upstream, by construction, the same way a real camera
    wouldn't.

    This is OBSERVATION, not ground truth and not a feature. It does
    NOT know the true queue length, does NOT know whether the queue
    extends past the camera, and must not be given that information.

Reads (per scenario):
    sumo/generated_scenarios/scenario_XXXX/scenario.json
    sumo/generated_scenarios/scenario_XXXX/raw_output/vehicle_trajectories.csv
    sumo/generated_scenarios/scenario_XXXX/raw_output/lane_metadata.json

Writes:
    sumo/generated_scenarios/scenario_XXXX/observations/camera_timeseries.csv

Run:
    python sensors/camera_simulator.py --scenario scenario_0001
    python sensors/camera_simulator.py                # all scenarios
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

# trajectory_utils.py lives in dataset/, a sibling folder -- not a proper
# installable package here, so add it to sys.path explicitly. Keeps the
# queue definition and distance-to-stopline logic in exactly one place
# instead of duplicating it into sensors/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "dataset"))
from trajectory_utils import SAMPLING_INTERVAL_S, load_trajectories, load_lane_metadata, \
    attach_distance_to_stopline, flag_queued  # noqa: E402

SUMO_DIR = PROJECT_ROOT / "sumo"
SCENARIOS_DIR = SUMO_DIR / "generated_scenarios"
SCENARIO_CONFIG_FILE = SUMO_DIR / "scenario_config.json"


def load_network_config() -> dict:
    with open(SCENARIO_CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scenario_metadata(scenario_dir: Path) -> dict:
    with open(scenario_dir / "scenario.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# The camera limit
# ============================================================================

def build_camera_observation(
    df: pd.DataFrame,
    approach_edges: List[str],
    camera_range_m: float,
    sim_begin: int,
    sim_end: int,
) -> pd.DataFrame:
    """Per approach, per sampling interval: what a camera covering only
    [0, camera_range_m] from the stop line would report. A vehicle at
    distance_to_stopline_m > camera_range_m does not exist as far as this
    function is concerned -- same as a real camera."""

    visible = df[df["is_on_approach"] & (df["distance_to_stopline_m"] <= camera_range_m)]

    sample_times = list(range(sim_begin, sim_end + 1, SAMPLING_INTERVAL_S))
    rows = []

    for t in sample_times:
        snapshot = visible[visible["timestamp"] == t]
        for edge in approach_edges:
            on_edge = snapshot[snapshot["edge_id"] == edge]
            queued = on_edge[on_edge["is_queued"]]

            visible_count = len(on_edge)
            visible_mean_speed = float(on_edge["speed_mps"].mean()) if visible_count > 0 else 0.0
            visible_queue_count = len(queued)
            visible_queue_length_m = float(queued["distance_to_stopline_m"].max()) if visible_queue_count > 0 else 0.0

            # A camera reports "queue fills the entire visible range" as a
            # distinct, useful signal from "no vehicles queued near the edge
            # of view" -- this flag is what tells a downstream model
            # "there might be more queue I can't see", without claiming to
            # know how much.
            queue_reaches_camera_edge = visible_queue_count > 0 and visible_queue_length_m >= (camera_range_m - SAMPLING_INTERVAL_S)

            rows.append({
                "timestamp": t,
                "approach_edge": edge,
                "camera_range_m": camera_range_m,
                "visible_vehicle_count": visible_count,
                "visible_mean_speed_mps": round(visible_mean_speed, 4),
                "visible_queue_count": visible_queue_count,
                "visible_queue_length_m": round(visible_queue_length_m, 2),
                "queue_reaches_camera_edge": bool(queue_reaches_camera_edge),
            })

    return pd.DataFrame(rows)


# ============================================================================
# Orchestration
# ============================================================================

def process_scenario(scenario_dir: Path, cfg: dict) -> pd.DataFrame:
    scenario = load_scenario_metadata(scenario_dir)
    approach_edges = cfg["network"]["approaches"]
    camera_range_m = cfg["network"]["camera_range_m"]

    df = load_trajectories(scenario_dir)
    lane_metadata = load_lane_metadata()
    df = attach_distance_to_stopline(df, lane_metadata, approach_edges)
    df = flag_queued(df)

    camera_df = build_camera_observation(
        df, approach_edges, camera_range_m,
        int(scenario["simulation_begin"]), int(scenario["simulation_end"]),
    )

    out_dir = scenario_dir / "observations"
    out_dir.mkdir(parents=True, exist_ok=True)
    camera_df.to_csv(out_dir / "camera_timeseries.csv", index=False)

    edge_hit_counts = camera_df.groupby("approach_edge")["queue_reaches_camera_edge"].sum().to_dict()
    print(f"{scenario['scenario_id']}: camera observation written to {out_dir / 'camera_timeseries.csv'}")
    print(f"  intervals where visible queue reaches the {camera_range_m}m camera edge: {edge_hit_counts}")

    return camera_df


def find_scenarios() -> List[Path]:
    return sorted(p for p in SCENARIOS_DIR.glob("scenario_*") if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build camera-limited observation from raw SUMO trajectories.")
    parser.add_argument("--scenario", type=str, default=None)
    args = parser.parse_args()

    cfg = load_network_config()

    if args.scenario:
        scenario_dirs = [SCENARIOS_DIR / args.scenario]
        if not scenario_dirs[0].exists():
            print(f"ERROR: scenario not found: {scenario_dirs[0]}")
            sys.exit(1)
    else:
        scenario_dirs = find_scenarios()
        if not scenario_dirs:
            print(f"ERROR: no scenarios found in {SCENARIOS_DIR}")
            sys.exit(1)

    failed = []
    for scenario_dir in scenario_dirs:
        try:
            process_scenario(scenario_dir, cfg)
        except Exception as exc:
            print(f"FAILED: {scenario_dir.name}: {exc}")
            failed.append(scenario_dir.name)

    print(f"\nDone. {len(scenario_dirs) - len(failed)}/{len(scenario_dirs)} succeeded.")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()