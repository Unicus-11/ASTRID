"""
camera_simulator.py
====================
ASTRID Prototype -- Camera Sensor Simulator

RESPONSIBILITY:
    Take the raw trajectory data and apply the ONE limit a real camera
    has: it cannot see past camera_range_m from the stop line. Everything
    computed here uses ONLY vehicles within that range -- this file has
    no access to what's happening further upstream, by construction, the
    same way a real camera wouldn't.

    This is OBSERVATION, not ground truth and not a feature. It does NOT
    know the true queue length, does NOT know whether the queue extends
    past the camera, and must not be given that information.

v0.2 changes (this revision -- aligning with the current project):
    - BUG FIX: the previous implementation filtered on a column named
      "is_on_approach", which does not exist in the current raw schema.
      The canonical column (written by run_scenarios.py, defined in
      trajectory_utils.py) is "is_approach_edge". Using the wrong name
      meant this filter would have raised a KeyError against current
      data. Now imports and uses trajectory_utils.APPROACH_FLAG_COLUMN
      directly rather than hardcoding either name, so there is exactly
      one place this could ever drift from the raw schema again.
    - Distance-to-stopline now uses
      trajectory_utils.resolve_distance_to_stopline(), which prefers the
      raw per-row "distance_from_stop_line_m" column written directly by
      run_scenarios.py and falls back to the older network-metadata
      calculation only where that raw value is missing. The previous
      implementation called attach_distance_to_stopline() directly,
      which ignores the raw column entirely and always recomputes from
      sq.net.xml. This matches the same raw-preferred resolution
      ground_truth.py and gps_simulator.py already use, so camera, GPS,
      and ground truth all agree on what "distance to stopline" means
      for a given row.
    - Added an explicit observation-window filter
      (restrict_to_observation_window), mirroring ground_truth.py's and
      gps_simulator.py's own defensive filter: trajectory rows are
      filtered to [scenario.simulation_begin, scenario.simulation_end]
      by timestamp before anything else happens, so a future change to
      what vehicle_trajectories.csv contains (e.g. if a clearance period
      were ever saved) can never leak into camera observations.
    - No more hardcoded "scenario_0001"-style scenario naming in
      docstrings/examples -- the current project uses 12 fixed named
      scenarios (scenario_normal_balanced, scenario_low_demand, ...).
    - Corrected this docstring's "Reads" section: there is no
      lane_metadata.json file. Lane geometry comes from
      trajectory_utils.load_lane_metadata(), which parses sq.net.xml
      directly (used only as the fallback path inside
      resolve_distance_to_stopline).
    - Added explicit, itemized validation (schema, camera-range
      sanity, scenario-directory existence, timestamp window,
      sampling-grid alignment, expected approach edges, camera-range
      containment of the actual visible set used for aggregation,
      ground-truth non-leakage, empty-visibility safety) instead of
      relying on things happening to work, matching the validation
      style already established in gps_simulator.py. See validate_*
      functions.
    - build_camera_observation() now returns (camera_df, visible_df)
      -- the filtered per-row data actually used to build camera_df --
      alongside the aggregated output, so validation can check the real
      filtered set instead of re-deriving or trusting the aggregation.

ANTI-LEAKAGE RULE:
    This script never reads dataset/ground_truth.py's output and never
    computes true queue length, density, or flow. It derives camera
    observations only from raw_output/vehicle_trajectories.csv -- the
    same raw file ground truth is built from, not ground truth itself --
    and only from vehicles within camera_range_m of the stop line.

Reads (per scenario):
    sumo/generated_scenarios/<scenario_id>/scenario.json
    sumo/generated_scenarios/<scenario_id>/raw_output/vehicle_trajectories.csv
    (sq.net.xml, indirectly, only as a distance-to-stopline fallback)

Writes:
    sumo/generated_scenarios/<scenario_id>/observations/camera_timeseries.csv

Run:
    python sensors/camera_simulator.py --scenario scenario_normal_balanced
    python sensors/camera_simulator.py                # all scenarios
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Set

import pandas as pd

# trajectory_utils.py lives in dataset/, a sibling folder -- not a proper
# installable package here, so add it to sys.path explicitly. Keeps the
# queue definition, approach-flag name, and distance-to-stopline logic in
# exactly one place instead of duplicating it into sensors/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "dataset"))
from trajectory_utils import (  # noqa: E402
    SAMPLING_INTERVAL_S,
    APPROACH_FLAG_COLUMN,
    load_trajectories,
    load_lane_metadata,
    resolve_distance_to_stopline,
    flag_queued,
)

SUMO_DIR = PROJECT_ROOT / "sumo"
SCENARIOS_DIR = SUMO_DIR / "generated_scenarios"
SCENARIO_CONFIG_FILE = SUMO_DIR / "scenario_config.json"

REQUIRED_RAW_COLUMNS = ["timestamp", "vehicle_id", "edge_id", "speed_mps"]

# Fields this sensor is allowed to write. Any of these names appearing
# would mean a true, ground-truth-shaped quantity leaked into the
# observation -- used as a regression guard in validate_camera_observations().
# Camera output uses its own "visible_*" names, so these should never
# collide in normal operation; this exists purely to catch a future
# accidental import of ground_truth.py's naming.
FORBIDDEN_GROUND_TRUTH_COLUMNS = {
    "queue_length_m", "queue_count", "queue_beyond_camera",
    "density_veh_per_km", "flow_veh_per_hour", "vehicle_count",
    "mean_speed_mps",
}


# ============================================================================
# Config / metadata loading
# ============================================================================

def load_network_config() -> dict:
    with open(SCENARIO_CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scenario_metadata(scenario_dir: Path) -> dict:
    with open(scenario_dir / "scenario.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# Validation -- run BEFORE and AFTER building observations
# ============================================================================

def validate_camera_range(camera_range_m: float) -> None:
    if camera_range_m is None or not (camera_range_m > 0):
        raise ValueError(f"camera_range_m must be a positive number, got {camera_range_m}")


def validate_scenario_directory(scenario_dir: Path) -> None:
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario directory does not exist: {scenario_dir}")
    if not (scenario_dir / "scenario.json").exists():
        raise FileNotFoundError(f"Missing scenario.json in {scenario_dir}")
    trajectories_path = scenario_dir / "raw_output" / "vehicle_trajectories.csv"
    if not trajectories_path.exists():
        raise FileNotFoundError(
            f"Missing raw trajectories: {trajectories_path}. "
            f"Run sumo/run_scenarios.py for this scenario first."
        )


def validate_raw_trajectory_schema(df: pd.DataFrame) -> None:
    """Checks the raw trajectory file actually has what this sensor needs,
    rather than failing confusingly partway through construction."""
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw trajectory data is missing required column(s): {missing}")

    has_raw_distance = "distance_from_stop_line_m" in df.columns
    has_fallback_geometry = {"lane_id", "lane_position_m"}.issubset(df.columns)
    if not has_raw_distance and not has_fallback_geometry:
        raise ValueError(
            "Raw trajectory data has neither a usable 'distance_from_stop_line_m' column "
            "nor the ('lane_id', 'lane_position_m') columns needed to derive it as a fallback."
        )


def validate_camera_observations(
    camera_df: pd.DataFrame,
    visible_df: pd.DataFrame,
    camera_range_m: float,
    approach_edges: List[str],
    sim_begin: int,
    sim_end: int,
) -> None:
    """Post-construction checks on the camera output itself."""
    # Nothing farther than camera_range_m contributed to the rows actually
    # used to build the aggregation -- the core FOV guarantee.
    if not visible_df.empty:
        max_dist = visible_df["distance_to_stopline_m"].max()
        if max_dist > camera_range_m:
            raise ValueError(
                f"Camera observation was built from a row at distance_to_stopline_m={max_dist}, "
                f"which exceeds camera_range_m={camera_range_m}."
            )
        min_dist = visible_df["distance_to_stopline_m"].min()
        if min_dist < 0:
            raise ValueError(
                f"Camera observation was built from a row at distance_to_stopline_m={min_dist} < 0."
            )

    if camera_df.empty:
        # Empty camera observations (no vehicles ever within range) are a
        # valid, safe outcome, not an error -- nothing further to check.
        return

    # Timestamps must fall within the scenario's primary observation window.
    out_of_window = camera_df[(camera_df["timestamp"] < sim_begin) | (camera_df["timestamp"] > sim_end)]
    if not out_of_window.empty:
        raise ValueError(
            f"Camera observation contains {len(out_of_window)} row(s) with timestamps outside "
            f"[{sim_begin}, {sim_end}]."
        )

    # Timestamps must align with the sampling grid.
    misaligned = camera_df[(camera_df["timestamp"] - sim_begin) % SAMPLING_INTERVAL_S != 0]
    if not misaligned.empty:
        raise ValueError(
            f"Camera observation contains {len(misaligned)} row(s) not aligned to "
            f"SAMPLING_INTERVAL_S={SAMPLING_INTERVAL_S}."
        )

    # Every expected approach edge must be represented (even with count=0).
    observed_edges = set(camera_df["approach_edge"].unique())
    expected_edges = set(approach_edges)
    if observed_edges != expected_edges:
        raise ValueError(
            f"Camera observation approach edges {observed_edges} do not match expected {expected_edges}."
        )

    # No ground-truth column names leaked into the output.
    leaked = FORBIDDEN_GROUND_TRUTH_COLUMNS.intersection(camera_df.columns)
    if leaked:
        raise ValueError(f"Camera observation contains forbidden ground-truth-shaped column(s): {leaked}")

    # camera_range_m must be recorded exactly as configured, on every row.
    if (camera_df["camera_range_m"] != camera_range_m).any():
        raise ValueError("Camera observation contains a camera_range_m value inconsistent with configuration.")


# ============================================================================
# Observation-window filtering (defensive, mirrors ground_truth.py / gps_simulator.py)
# ============================================================================

def restrict_to_observation_window(df: pd.DataFrame, sim_begin: int, sim_end: int) -> pd.DataFrame:
    """Defensively restrict trajectory rows to the scenario's own primary
    observation period before anything else happens, so a future change
    to what vehicle_trajectories.csv contains can never leak into camera
    observations. Mirrors dataset/ground_truth.py's and
    sensors/gps_simulator.py's identical filter."""
    mask = (df["timestamp"] >= sim_begin) & (df["timestamp"] <= sim_end)
    return df.loc[mask].copy()


# ============================================================================
# The camera limit
# ============================================================================

def build_camera_observation(
    df: pd.DataFrame,
    approach_edges: List[str],
    camera_range_m: float,
    sim_begin: int,
    sim_end: int,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Per approach, per sampling interval: what a camera covering only
    [0, camera_range_m] from the stop line would report. A vehicle at
    distance_to_stopline_m > camera_range_m does not exist as far as this
    function is concerned -- same as a real camera.

    Returns (camera_df, visible_df) -- visible_df (the filtered per-row
    data used to build camera_df) is returned alongside for validation.
    """
    visible = df[df[APPROACH_FLAG_COLUMN] & (df["distance_to_stopline_m"] >= 0) &
                 (df["distance_to_stopline_m"] <= camera_range_m)]

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
            # of view" -- this flag is what tells a downstream model "there
            # might be more queue I can't see", without claiming to know how
            # much. queue_reaches_camera_edge=True means the visible queue
            # reaches approximately the limit of what the camera can see; it
            # does NOT mean the true queue length equals camera_range_m.
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

    return pd.DataFrame(rows), visible


# ============================================================================
# Orchestration
# ============================================================================

def process_scenario(scenario_dir: Path, cfg: dict) -> pd.DataFrame:
    validate_scenario_directory(scenario_dir)

    scenario = load_scenario_metadata(scenario_dir)
    approach_edges = cfg["network"]["approaches"]
    camera_range_m = cfg["network"]["camera_range_m"]
    validate_camera_range(camera_range_m)

    sim_begin = int(scenario["simulation_begin"])
    sim_end = int(scenario["simulation_end"])

    df = load_trajectories(scenario_dir)
    validate_raw_trajectory_schema(df)

    # Defensive filter: camera observations are built ONLY from the
    # scenario's own primary observation window, regardless of what the
    # raw trajectories file happens to contain (mirrors ground_truth.py
    # and gps_simulator.py).
    df = restrict_to_observation_window(df, sim_begin, sim_end)

    lane_metadata = load_lane_metadata()
    df = resolve_distance_to_stopline(df, lane_metadata, approach_edges)
    df = flag_queued(df)

    camera_df, visible_df = build_camera_observation(
        df, approach_edges, camera_range_m, sim_begin, sim_end,
    )
    validate_camera_observations(camera_df, visible_df, camera_range_m, approach_edges, sim_begin, sim_end)

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
    parser.add_argument("--scenario", type=str, default=None,
                         help="Run one scenario, e.g. scenario_normal_balanced. If omitted, runs all found scenarios.")
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