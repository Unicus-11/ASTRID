"""
trajectory_utils.py
====================
ASTRID Prototype -- shared per-vehicle trajectory enrichment.

Used by dataset/ground_truth.py, sensors/camera_simulator.py, and
sensors/gps_simulator.py. Factored out here so the queue definition and
distance-to-stopline calculation exist in exactly ONE place.

v0.3: lane lengths are read directly from sq.net.xml (parsed once,
cached) instead of a separately generated lane_metadata.json. The
network is fixed across every V0 scenario, so this is the same file
for all of them -- no need to capture or regenerate anything.
"""

from __future__ import annotations

import functools
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

import pandas as pd


# ============================================================================
# PATHS -- matches the real ASTRID folder layout:
#   <project_root>/dataset/trajectory_utils.py
#   <project_root>/sumo/scenario_config.json
#   <project_root>/sumo/Squire_Junction_Multiple_Lanes/sq.net.xml
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMO_DIR = PROJECT_ROOT / "sumo"
NETWORK_FILE = SUMO_DIR / "Squire_Junction_Multiple_Lanes" / "sq.net.xml"


# ============================================================================
# QUEUE DEFINITION -- ONE reproducible rule, used everywhere downstream.
# ============================================================================

QUEUE_SPEED_THRESHOLD_MPS = 1.0     # ~3.6 km/h -- standard "near-stationary" cutoff
QUEUE_MIN_DURATION_S = 3.0          # must be near-stationary this long to count as queued
SAMPLING_INTERVAL_S = 5             # state snapshot cadence, shared by ground truth + sensors


def load_trajectories(scenario_dir: Path) -> pd.DataFrame:
    path = scenario_dir / "raw_output" / "vehicle_trajectories.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No raw trajectories at {path}. Run sumo/run_scenarios.py for this scenario first."
        )
    return pd.read_csv(path)


@functools.lru_cache(maxsize=1)
def load_lane_metadata() -> Dict[str, float]:
    """Parse sq.net.xml once (cached) and return {lane_id: length_m}.

    The network is the same for every scenario, so this file only needs
    to be read once per process, not once per scenario. NOT loaded from
    a generated file -- read directly from the real SUMO network."""
    if not NETWORK_FILE.exists():
        raise FileNotFoundError(
            f"Network file not found: {NETWORK_FILE}\n"
            f"trajectory_utils.py expects it at "
            f"<project_root>/sumo/Squire_Junction_Multiple_Lanes/sq.net.xml -- "
            f"if your real path differs, update NETWORK_FILE at the top of this file."
        )

    tree = ET.parse(NETWORK_FILE)
    root = tree.getroot()

    lane_lengths: Dict[str, float] = {}
    for edge in root.findall("edge"):
        for lane in edge.findall("lane"):
            lane_id = lane.get("id")
            length = lane.get("length")
            if lane_id is not None and length is not None:
                lane_lengths[lane_id] = float(length)

    if not lane_lengths:
        raise ValueError(f"No <lane> elements with id/length found in {NETWORK_FILE} -- unexpected net.xml format.")

    return lane_lengths


def attach_distance_to_stopline(df: pd.DataFrame, lane_metadata: Dict[str, float],
                                 approach_edges: List[str]) -> pd.DataFrame:
    """Adds: lane_length_m, is_on_approach, is_internal_edge, distance_to_stopline_m.

    is_on_approach / is_internal_edge come directly from the trajectory's own
    edge_id column -- no lookup needed for those. lane_metadata is used ONLY
    for lane length."""
    df = df.copy()
    df["lane_length_m"] = df["lane_id"].map(lane_metadata)
    df["is_on_approach"] = df["edge_id"].isin(approach_edges)
    df["is_internal_edge"] = df["edge_id"].astype(str).str.startswith(":")

    df["distance_to_stopline_m"] = 0.0
    on_approach = df["is_on_approach"] & df["lane_length_m"].notna()
    df.loc[on_approach, "distance_to_stopline_m"] = (
        df.loc[on_approach, "lane_length_m"] - df.loc[on_approach, "lane_position_m"]
    ).clip(lower=0.0)

    missing_lane = df["is_on_approach"] & df["lane_length_m"].isna()
    if missing_lane.any():
        n = missing_lane.sum()
        bad_ids = sorted(df.loc[missing_lane, "lane_id"].unique())[:5]
        print(f"WARNING: {n} rows on an approach edge have a lane_id not found in sq.net.xml "
              f"(e.g. {bad_ids}) -- distance_to_stopline_m left at 0 for these rows. "
              f"Check NETWORK_FILE points at the right net.xml.")

    return df


def flag_queued(df: pd.DataFrame) -> pd.DataFrame:
    """Adds: low_speed_streak_s, is_queued. Same rule everywhere: speed <=
    QUEUE_SPEED_THRESHOLD_MPS, on an approach edge, sustained for at least
    QUEUE_MIN_DURATION_S consecutive seconds."""
    df = df.sort_values(["vehicle_id", "timestamp"]).copy()

    is_slow = (df["speed_mps"] <= QUEUE_SPEED_THRESHOLD_MPS) & df["is_on_approach"]
    group_break = (~is_slow).groupby(df["vehicle_id"]).cumsum()
    streak = is_slow.groupby([df["vehicle_id"], group_break]).cumcount() + 1
    streak = streak.where(is_slow, 0)

    df["low_speed_streak_s"] = streak
    df["is_queued"] = is_slow & (streak >= QUEUE_MIN_DURATION_S)
    return df


def enrich_trajectories(scenario_dir: Path, approach_edges: List[str]) -> pd.DataFrame:
    """Convenience wrapper: load + attach distance + flag queued in one call."""
    df = load_trajectories(scenario_dir)
    lane_metadata = load_lane_metadata()
    df = attach_distance_to_stopline(df, lane_metadata, approach_edges)
    df = flag_queued(df)
    return df