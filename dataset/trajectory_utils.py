"""
trajectory_utils.py
====================
ASTRID Prototype -- shared per-vehicle trajectory preprocessing.

This module is NOT a scenario generator, NOT a SUMO runner, and NOT a
dataset builder. It is a small, scenario-agnostic utility layer shared by:

    dataset/ground_truth.py        -- computes true per-approach traffic state
    sensors/camera_simulator.py    -- simulates a camera's limited view of traffic
    sensors/gps_simulator.py       -- simulates GPS-probe observations of traffic

All three need the same underlying per-vehicle preprocessing (loading raw
trajectories, knowing how far a vehicle is from the stop line, and knowing
whether a vehicle counts as "queued"). That logic lives here, in exactly
one place, so ground truth and every sensor simulator agree on it.

This module works identically for all 12 current scenarios (4 train, 2 val,
2 test, 4 OOD) and for any future scenario using the same network and the
same run_scenarios.py output format. It contains NO scenario-name,
split-name, or demand-level-specific logic, and it does not know or care
which split a scenario belongs to.

--------------------------------------------------------------------------
RAW COLUMNS this module expects in vehicle_trajectories.csv
(as written by sumo/run_scenarios.py):
--------------------------------------------------------------------------
    timestamp                  -- simulation second (int/float)
    vehicle_id                 -- SUMO vehicle id (string)
    vehicle_type                -- SUMO vType id (string)
    edge_id                     -- current SUMO edge id
    lane_id                     -- current SUMO lane id
    is_internal_edge            -- 1/0, True if edge_id starts with ":"
    is_approach_edge            -- 1/0, True if edge_id is one of the four
                                    approach edges (1i/2i/3i/4i)
    lane_position_m             -- vehicle's position along its current lane
    lane_length_m               -- lane length as recorded by run_scenarios.py
    distance_from_stop_line_m   -- raw, per-row distance-to-stopline computed
                                    directly from SUMO lane length/position
                                    at record time (may be blank/NaN for
                                    non-approach edges or older raw files)
    speed_mps, acceleration_mps2, waiting_time_s, x, y, angle_deg

CANONICAL FIELD NAME -- IMPORTANT:
    The single boolean flag for "this row is on one of the four approach
    edges" is ALWAYS named:

        is_approach_edge

    This is the name run_scenarios.py already writes to the raw CSV, and
    it is the ONLY name used anywhere in this module or its downstream
    consumers. The older, retired name "is_on_approach" must never be
    reintroduced -- it does not exist in the current raw schema.

--------------------------------------------------------------------------
DERIVED COLUMNS this module adds:
--------------------------------------------------------------------------
    lane_length_m (float, from network) -- looked up from sq.net.xml via
                                            load_lane_metadata(); distinct
                                            from any lane_length_m the raw
                                            CSV may already carry, and used
                                            only by the network-based
                                            fallback distance calculation
    distance_to_stopline_m      -- the single, resolved distance value
                                    downstream code should read. Prefers
                                    the raw distance_from_stop_line_m
                                    column; falls back to a network-based
                                    calculation (lane length - lane
                                    position) only where the raw value is
                                    missing/invalid.
    low_speed_streak_s          -- consecutive seconds a vehicle has been
                                    at/below QUEUE_SPEED_THRESHOLD_MPS
                                    while on an approach edge
    is_queued                   -- True once low_speed_streak_s reaches
                                    QUEUE_MIN_DURATION_S

--------------------------------------------------------------------------
PUBLIC API (names preserved for compatibility with the current
dataset/ground_truth.py, which imports these directly):
--------------------------------------------------------------------------
    SAMPLING_INTERVAL_S
    load_trajectories(scenario_dir)
    load_lane_metadata()
    attach_distance_to_stopline(df, lane_metadata, approach_edges)
    flag_queued(df)

Additional helpers (used by sensor simulators / new callers, not required
by the current ground_truth.py):
    resolve_distance_to_stopline(df, lane_metadata, approach_edges)
    enrich_trajectories(scenario_dir, approach_edges)
"""

from __future__ import annotations

import functools
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

import pandas as pd


# ============================================================================
# PATHS -- matches the real ASTRID folder layout:
#   <project_root>/dataset/trajectory_utils.py
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

# Raw column written by run_scenarios.py's trajectory writer, and the
# resolved column name every downstream consumer (ground truth, sensors)
# reads. Kept as named constants so the "prefer raw, fall back to network
# calc" logic below has a single source of truth for both names.
RAW_DISTANCE_COLUMN = "distance_from_stop_line_m"
RESOLVED_DISTANCE_COLUMN = "distance_to_stopline_m"

# Canonical boolean flag name -- see module docstring. Never rename this
# or introduce a second name ("is_on_approach") for the same concept.
APPROACH_FLAG_COLUMN = "is_approach_edge"
INTERNAL_FLAG_COLUMN = "is_internal_edge"


# ============================================================================
# 1. LOAD RAW TRAJECTORIES
# ============================================================================

def load_trajectories(scenario_dir: Path) -> pd.DataFrame:
    """Load raw_output/vehicle_trajectories.csv for any scenario directory.

    Works identically for every scenario (train/val/test/OOD) -- the
    scenario directory's own structure is the only input, nothing here
    is scenario-specific."""
    path = Path(scenario_dir) / "raw_output" / "vehicle_trajectories.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No raw trajectories at {path}. Run sumo/run_scenarios.py for this scenario first."
        )
    return pd.read_csv(path)


# ============================================================================
# 2. LOAD FIXED NETWORK LANE METADATA
# ============================================================================

@functools.lru_cache(maxsize=1)
def load_lane_metadata() -> Dict[str, float]:
    """Parse sq.net.xml once (cached) and return {lane_id: length_m}.

    The road network is fixed across every current and future V0
    scenario, so this is read directly from the real SUMO network file
    rather than from any per-scenario generated metadata file. Cached
    with lru_cache since the result is identical for the whole process."""
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


# ============================================================================
# Shared helper: canonical is_approach_edge / is_internal_edge flags
# ============================================================================

def _ensure_edge_flags(df: pd.DataFrame, approach_edges: List[str]) -> pd.DataFrame:
    """Guarantee df has canonical, boolean-typed is_approach_edge /
    is_internal_edge columns.

    run_scenarios.py already writes both of these to the raw CSV (as
    1/0 ints), so the normal path here is just a dtype cast. The
    edge_id-based derivation is kept ONLY as a defensive fallback for a
    trajectories file that predates those raw columns -- it is not the
    primary source of truth."""
    df = df.copy()

    if APPROACH_FLAG_COLUMN in df.columns:
        df[APPROACH_FLAG_COLUMN] = df[APPROACH_FLAG_COLUMN].astype(bool)
    else:
        df[APPROACH_FLAG_COLUMN] = df["edge_id"].isin(approach_edges)

    if INTERNAL_FLAG_COLUMN in df.columns:
        df[INTERNAL_FLAG_COLUMN] = df[INTERNAL_FLAG_COLUMN].astype(bool)
    else:
        df[INTERNAL_FLAG_COLUMN] = df["edge_id"].astype(str).str.startswith(":")

    return df


# ============================================================================
# 3. RESOLVE DISTANCE TO STOP LINE
# ============================================================================

def attach_distance_to_stopline(df: pd.DataFrame, lane_metadata: Dict[str, float],
                                 approach_edges: List[str]) -> pd.DataFrame:
    """Network-metadata-based distance-to-stopline calculation.

    Adds/overwrites: lane_length_m (from the network, not the raw CSV),
    is_approach_edge, is_internal_edge, distance_to_stopline_m.

    This is the OLD calculation path: distance = lane_length - lane_position,
    using lane lengths looked up from sq.net.xml. It does NOT look at the
    raw distance_from_stop_line_m column at all -- it exists so callers
    that specifically want (or need, as a fallback) the network-derived
    value have a single reproducible way to get it. For the raw-preferring
    resolution most callers should use, see resolve_distance_to_stopline()
    below.
    """
    
    # `lane_length_m` in the enriched dataframe is the network-derived value.
    # The raw CSV also contains `lane_length_m`, but downstream fallback
    # calculations intentionally use the fixed network metadata as the
    # authoritative geometry source.
    df = _ensure_edge_flags(df, approach_edges)
    df["lane_length_m"] = df["lane_id"].map(lane_metadata)

    df[RESOLVED_DISTANCE_COLUMN] = 0.0
    on_approach = df[APPROACH_FLAG_COLUMN] & df["lane_length_m"].notna()
    df.loc[on_approach, RESOLVED_DISTANCE_COLUMN] = (
        df.loc[on_approach, "lane_length_m"] - df.loc[on_approach, "lane_position_m"]
    ).clip(lower=0.0)

    missing_lane = df[APPROACH_FLAG_COLUMN] & df["lane_length_m"].isna()
    if missing_lane.any():
        n = missing_lane.sum()
        bad_ids = sorted(df.loc[missing_lane, "lane_id"].unique())[:5]
        print(f"WARNING: {n} rows on an approach edge have a lane_id not found in sq.net.xml "
              f"(e.g. {bad_ids}) -- distance_to_stopline_m left at 0 for these rows. "
              f"Check NETWORK_FILE points at the right net.xml.")

    return df


def resolve_distance_to_stopline(df: pd.DataFrame, lane_metadata: Dict[str, float],
                                  approach_edges: List[str]) -> pd.DataFrame:
    """Preferred distance-to-stopline resolution: raw column first, network
    calculation only as a fallback for rows where the raw value is
    missing or invalid. This never silently invents a distance for a row
    that has neither a usable raw value nor a resolvable lane length --
    such rows are left at 0 with a warning, same as the old network-only
    path, so a missing distance is always visible rather than masked.

    Adds/overwrites: is_approach_edge, is_internal_edge, distance_to_stopline_m.
    """
    df = _ensure_edge_flags(df, approach_edges)

    if RAW_DISTANCE_COLUMN in df.columns:
        raw_distance = pd.to_numeric(df[RAW_DISTANCE_COLUMN], errors="coerce")
        missing_mask = raw_distance.isna()

        if missing_mask.any():
            # Only rows missing a usable raw value fall back to the
            # network-based calculation; rows with a valid raw value keep it.
            fallback_df = attach_distance_to_stopline(df, lane_metadata, approach_edges)
            fallback_distance = fallback_df[RESOLVED_DISTANCE_COLUMN]
            df[RESOLVED_DISTANCE_COLUMN] = raw_distance.where(~missing_mask, fallback_distance)
        else:
            df[RESOLVED_DISTANCE_COLUMN] = raw_distance

        return df

    # No raw column at all (e.g. a trajectories file that predates
    # run_scenarios.py's raw distance column) -- fall back entirely.
    return attach_distance_to_stopline(df, lane_metadata, approach_edges)


# ============================================================================
# 4. QUEUE DEFINITION
# ============================================================================

def flag_queued(df: pd.DataFrame) -> pd.DataFrame:
    """Adds: low_speed_streak_s, is_queued.

    Single reproducible rule, shared by ground truth and every sensor
    simulator: a vehicle counts as queued once it has been at or below
    QUEUE_SPEED_THRESHOLD_MPS, on an approach edge, for at least
    QUEUE_MIN_DURATION_S consecutive seconds.

    Requires is_approach_edge to already be present on df (set by
    attach_distance_to_stopline / resolve_distance_to_stopline /
    enrich_trajectories) -- this function does not derive it itself,
    since it has no approach_edges list to derive it from."""
    if APPROACH_FLAG_COLUMN not in df.columns:
        raise ValueError(
            f"flag_queued() requires '{APPROACH_FLAG_COLUMN}' on the DataFrame. "
            f"Call attach_distance_to_stopline() or resolve_distance_to_stopline() "
            f"(or enrich_trajectories()) first."
        )

    df = df.sort_values(["vehicle_id", "timestamp"]).copy()

    is_slow = (df["speed_mps"] <= QUEUE_SPEED_THRESHOLD_MPS) & df[APPROACH_FLAG_COLUMN]
    group_break = (~is_slow).groupby(df["vehicle_id"]).cumsum()
    streak = is_slow.groupby([df["vehicle_id"], group_break]).cumcount() + 1
    streak = streak.where(is_slow, 0)

    df["low_speed_streak_s"] = streak
    df["is_queued"] = is_slow & (streak >= QUEUE_MIN_DURATION_S)
    return df


# ============================================================================
# 5. CONVENIENCE WRAPPER
# ============================================================================

def enrich_trajectories(scenario_dir: Path, approach_edges: List[str]) -> pd.DataFrame:
    """Common sequence used by sensor simulators (and available to any
    other caller): load raw trajectories -> load network lane metadata ->
    resolve distance to stop line (raw-preferred) -> flag queues.

    dataset/ground_truth.py does not call this wrapper directly -- it
    performs the same steps itself so it can also apply its own
    observation-window filtering in between loading and enrichment -- but
    it uses the same underlying attach_distance_to_stopline / flag_queued
    functions defined here."""
    df = load_trajectories(scenario_dir)
    lane_metadata = load_lane_metadata()
    df = resolve_distance_to_stopline(df, lane_metadata, approach_edges)
    df = flag_queued(df)
    return df