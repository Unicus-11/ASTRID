"""
gps_simulator.py
====================
ASTRID Prototype -- GPS / Probe Vehicle Sensor Simulator  (v0.6)

RESPONSIBILITY:
    Simulate a sparse fraction of vehicles being GPS-equipped probes.
    Unlike the camera, a probe's position is NOT limited by distance --
    a probe far upstream still reports in. The limit here is COUNT, not
    RANGE: only a fraction (penetration_rate) of vehicles are probes,
    and which ones are probes is decided once, deterministically, per
    (scenario seed + vehicle_id + penetration rate) -- never by
    dataframe row order.

    This is OBSERVATION, not ground truth. A downstream model is only
    ever allowed to see the probe subset produced here, never the full
    vehicle population, and never any ground-truth queue/density/flow
    label. See ANTI-LEAKAGE below.

v0.2-v0.4 changes: see prior revisions (bug fixes to the approach-edge
column name, distance-to-stopline resolution, probe-population ordering,
and exact-count top-K probe selection).

v0.5 changes: introduced a per-probe, per-second trajectory output
alongside the 5s aggregate, for Cheng et al.-style critical-point
analysis. That first version built the per-probe table from the SAME
approach-edge-only dataframe used for the 5s aggregate, which is exactly
what v0.6 below corrects.

v0.6 changes (this revision -- there is only ONE underlying GPS
simulation, not two):
    v0.5's per-probe trajectory was built from probe_df, the
    approach-edge-only rows used for the aggregate. That silently cut
    every probe's trajectory off the moment it left the approach edge --
    exactly the region (crossing the stop line, clearing the
    intersection, accelerating away on the internal/outgoing edge) where
    Cheng et al.'s Type II critical point is expected to sit. A probe's
    trajectory would just vanish from the table mid-maneuver.

    The corrected model is:

        selected probes (deterministic, unchanged)
                |
                v
        1-second, per-probe trajectory -- EVERY row belonging to a
        selected probe inside the observation window, on ANY edge
        (approach, internal, or outgoing) -- this is now the single
        underlying GPS observation.
                |
                +--> aggregate the approach-edge rows to the shared 5s
                |    grid --> gps_p{TAG}_timeseries.csv (UNCHANGED format,
                |    UNCHANGED meaning -- observation_assembler.py and
                |    Layer 2 keep reading it exactly as before)
                |
                +--> gps_p{TAG}_probe_trajectories.csv (full per-probe
                     trajectory, all edges, native 1s resolution -- for
                     Cheng et al.-style critical-point extraction only)

    The 5-second aggregate is no longer built directly from raw
    trajectory rows; it is now explicitly a downstream aggregation of
    the same 1-second per-probe trajectory the Cheng-oriented output
    also comes from -- one GPS simulation, two views of it, not two
    separate sensors that happen to share a probe set.

    Two field-level corrections that fall out of using the full,
    un-filtered trajectory:

    - edge_id vs. approach_edge are now genuinely different fields.
      edge_id is simply whatever SUMO edge the probe is on at that
      timestamp -- it changes as the probe moves from an approach edge,
      through the internal junction edge, onto an outgoing edge.
      approach_edge instead identifies WHICH of the four approach edges
      this probe's intersection crossing belongs to: it is set to
      edge_id while the probe is actually on an approach edge, and then
      carried forward (forward-filled, per probe) onto that probe's
      later internal/outgoing rows, so later processing can still
      attribute a post-stop-line row to the correct approach. A probe
      that never enters an approach edge during the observation window
      keeps approach_edge as NaN for its entire trajectory -- this is
      never fabricated.
    - distance_to_stopline_m is only physically meaningful while a
      probe is actually on an approach edge. trajectory_utils.py's
      distance resolution fills a network-based default of 0.0 for
      every other edge (that default exists for ITS OWN callers, e.g.
      ground_truth.py, which only ever look at approach-edge rows
      anyway). Left as 0.0 here it would misleadingly read as "sitting
      at the stop line" for a probe that is, say, three blocks past the
      intersection. This module blanks it back to NaN for any row where
      the probe is not on an approach edge, without touching
      trajectory_utils.py itself.

    Neither change alters probe selection, the 5s aggregate's schema,
    or anything ground-truth-related.

    Network-boundary note (documentation only -- no logic added here):
    the simulated approach is roughly 483m long. A probe trajectory that
    is still present at (or very near) that upstream boundary at the end
    of the observation window may reflect a genuinely long queue, OR may
    simply reflect that the queue outgrew the modeled network before the
    window ended -- this module does not attempt to tell those apart. It
    preserves the raw edge/position/timestamp information needed for a
    later stage to make that distinction; it does not classify or label
    it.

ANTI-LEAKAGE RULE:
    This script never reads dataset/ground_truth.py's output and never
    computes queue length, density, or flow -- for the population OR for
    an individual probe. It derives every GPS observation only from
    raw_output/vehicle_trajectories.csv -- the same raw file ground
    truth is built from, not ground truth itself. The full per-probe
    trajectory added in v0.5/v0.6 reports only facts a real onboard
    GPS/IMU device could contribute for its OWN vehicle (edge, lane,
    distance to stop-bar while on an approach, speed, acceleration) --
    never a queue length, density, or flow computed from the wider
    vehicle population, and never a label of any kind.

Reads (per scenario):
    sumo/generated_scenarios/<scenario_id>/scenario.json
    sumo/generated_scenarios/<scenario_id>/raw_output/vehicle_trajectories.csv
    (sq.net.xml, indirectly, only as a distance-to-stopline fallback)

Writes (per scenario, per penetration rate):
    sumo/generated_scenarios/<scenario_id>/observations/gps_p{NN}_timeseries.csv
        -- 5s population aggregate, approach-edge rows only, UNCHANGED
        format -- now explicitly a downstream aggregation of the 1s
        per-probe trajectory below, rather than an independent read of
        the raw file.
    sumo/generated_scenarios/<scenario_id>/observations/gps_p{NN}_probe_ids.json
    sumo/generated_scenarios/<scenario_id>/observations/gps_p{NN}_probe_trajectories.csv
        -- the underlying 1-second per-probe trajectory: EVERY row (any
        edge) belonging to a selected probe inside the observation
        window. For Cheng et al.-style critical-point analysis only.

Run:
    python sensors/gps_simulator.py --scenario scenario_normal_balanced --penetration 0.10
    python sensors/gps_simulator.py --penetration 0.05        # all scenarios, 5% penetration
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "dataset"))
from trajectory_utils import (  # noqa: E402
    SAMPLING_INTERVAL_S,
    APPROACH_FLAG_COLUMN,
    load_trajectories,
    load_lane_metadata,
    resolve_distance_to_stopline,
)

SUMO_DIR = PROJECT_ROOT / "sumo"
SCENARIOS_DIR = SUMO_DIR / "generated_scenarios"
SCENARIO_CONFIG_FILE = SUMO_DIR / "scenario_config.json"

# Keeps probe-selection hashing independent of any other use of the
# scenario's own seed elsewhere in the pipeline (SUMO's --seed, etc.).
GPS_SEED_OFFSET = 5000

# Documents the resolution of gps_p{TAG}_probe_trajectories.csv. NOT used
# to resample anything: run_scenarios.py already records every active
# vehicle every simulation step (step-length=1.0s), so the per-probe rows
# pulled out of the raw trajectory data are already at this resolution --
# this constant exists so feature_builder.py and any validation here can
# assert against it explicitly rather than assuming it.
PROBE_TRAJECTORY_SAMPLING_INTERVAL_S = 1

# Columns written to gps_p{TAG}_probe_trajectories.csv -- the full,
# un-filtered per-probe trajectory (any edge) at native 1-second
# resolution. edge_id and approach_edge are deliberately separate fields
# -- see the v0.6 note in the module docstring for what each one means.
PROBE_TRAJECTORY_COLUMNS = [
    "timestamp", "probe_id", "edge_id", "approach_edge", "lane_id",
    "speed_mps", "acceleration_mps2", "distance_to_stopline_m",
]

# Fields this sensor is allowed to write. Any of these names appearing
# would mean a ground-truth label leaked into the observation -- used as
# a regression guard in validate_gps_observations() /
# validate_probe_trajectories().
FORBIDDEN_GROUND_TRUTH_COLUMNS = {
    "queue_length_m", "queue_count", "queue_beyond_camera",
    "density_veh_per_km", "flow_veh_per_hour", "vehicle_count",
    "mean_speed_mps",  # ground truth's per-approach mean; GPS has its own probe_mean_speed_mps
}

REQUIRED_RAW_COLUMNS = ["timestamp", "vehicle_id", "edge_id", "lane_id", "speed_mps"]


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

def validate_penetration_rate(penetration_rate: float) -> None:
    if not (0.0 < penetration_rate <= 1.0):
        raise ValueError(f"penetration_rate must be in (0, 1], got {penetration_rate}")


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
    if "acceleration_mps2" not in df.columns:
        raise ValueError(
            "Raw trajectory data is missing 'acceleration_mps2' -- required for the per-probe "
            "trajectory output (gps_p{TAG}_probe_trajectories.csv), which Cheng et al.-style "
            "critical-point detection needs."
        )


def validate_probe_selection(probe_ids: Set[str], all_vehicle_ids: Set[str], seed: int,
                              penetration_rate: float) -> None:
    """Probe IDs must be a subset of real vehicle IDs, must be exactly
    round(len(all_vehicle_ids) * penetration_rate) in count (not merely
    close to it), and probe selection must be reproducible (same seed +
    vehicle_id set + rate -> same result, independent of dataframe order
    -- verified here by recomputing). UNCHANGED from prior revisions."""
    if not probe_ids.issubset(all_vehicle_ids):
        raise ValueError("Probe IDs are not a subset of the scenario's actual vehicle IDs.")

    expected_count = min(max(round(len(all_vehicle_ids) * penetration_rate), 0), len(all_vehicle_ids))
    if len(probe_ids) != expected_count:
        raise ValueError(
            f"Probe count {len(probe_ids)} does not exactly match "
            f"round({len(all_vehicle_ids)} * {penetration_rate}) = {expected_count}."
        )

    recomputed = select_probe_vehicles(all_vehicle_ids, seed, penetration_rate)
    if recomputed != probe_ids:
        raise ValueError(
            "Probe selection is not deterministic: recomputing from the same "
            "(seed, vehicle_id set, penetration_rate) produced a different result."
        )


def validate_probe_trajectories(
    traj_df: pd.DataFrame,
    probe_ids: Set[str],
    approach_edges: List[str],
    sim_begin: int,
    sim_end: int,
) -> None:
    """Post-construction checks on the full per-probe trajectory output
    (any edge, native 1s resolution). An empty result (e.g. very low
    penetration with few vehicles) is a valid, safe outcome -- not an
    error."""
    if traj_df.empty:
        return

    missing = [c for c in PROBE_TRAJECTORY_COLUMNS if c not in traj_df.columns]
    if missing:
        raise ValueError(f"Probe trajectory table is missing required column(s): {missing}")

    contributing_ids = set(traj_df["probe_id"].unique())
    if not contributing_ids.issubset(probe_ids):
        raise ValueError("Probe trajectory table was built from vehicle(s) outside the selected probe set.")

    out_of_window = traj_df[(traj_df["timestamp"] < sim_begin) | (traj_df["timestamp"] > sim_end)]
    if not out_of_window.empty:
        raise ValueError(
            f"Probe trajectory table contains {len(out_of_window)} row(s) with timestamps outside "
            f"[{sim_begin}, {sim_end}]."
        )

    # edge_id is intentionally UNRESTRICTED here -- a probe's full
    # trajectory legitimately includes internal (":..." ) and outgoing
    # edges, not just the four approach edges. Only approach_edge (when
    # not NaN) must be one of the real approach edges.
    observed_approaches = set(traj_df["approach_edge"].dropna().unique())
    if not observed_approaches.issubset(set(approach_edges)):
        raise ValueError(
            f"Probe trajectory table has approach_edge value(s) {observed_approaches} outside "
            f"expected {set(approach_edges)}."
        )

    leaked = FORBIDDEN_GROUND_TRUTH_COLUMNS.intersection(traj_df.columns)
    if leaked:
        raise ValueError(f"Probe trajectory table contains forbidden ground-truth-shaped column(s): {leaked}")

    # Each probe's own timestamps must be strictly increasing within this
    # table -- Cheng et al.'s method depends on being able to walk one
    # vehicle's trajectory forward in time without duplicate/out-of-order
    # rows. This does NOT require consecutive timestamps (a real gap in
    # when a probe was recorded is preserved, never fabricated) -- only
    # that recorded rows are in increasing order.
    non_increasing = (
        traj_df.sort_values(["probe_id", "timestamp"])
        .groupby("probe_id")["timestamp"]
        .apply(lambda s: bool((s.diff().dropna() <= 0).any()))
    )
    bad_probes = non_increasing[non_increasing].index.tolist()
    if bad_probes:
        raise ValueError(
            f"Probe trajectory table has non-increasing timestamps for probe_id(s): {bad_probes}"
        )


def validate_gps_observations(
    gps_df: pd.DataFrame,
    aggregate_source_df: pd.DataFrame,
    probe_ids: Set[str],
    approach_edges: List[str],
    sim_begin: int,
    sim_end: int,
) -> None:
    """Post-construction checks on the 5s aggregate GPS output.
    aggregate_source_df is the approach-edge subset of the per-probe
    trajectory table that gps_df was aggregated FROM (see
    build_gps_observations) -- kept for these cross-checks, same role
    "probe_df" played in prior revisions."""
    contributing_ids = set(aggregate_source_df["probe_id"].unique())
    if not contributing_ids.issubset(probe_ids):
        raise ValueError("GPS observation was built from vehicle(s) outside the selected probe set.")

    if gps_df.empty:
        # Empty probe observations (e.g. very low penetration + few vehicles)
        # are a valid, safe outcome, not an error -- nothing further to check.
        return

    out_of_window = gps_df[(gps_df["timestamp"] < sim_begin) | (gps_df["timestamp"] > sim_end)]
    if not out_of_window.empty:
        raise ValueError(
            f"GPS observation contains {len(out_of_window)} row(s) with timestamps outside "
            f"[{sim_begin}, {sim_end}]."
        )

    observed_edges = set(gps_df["approach_edge"].unique())
    expected_edges = set(approach_edges)
    if observed_edges != expected_edges:
        raise ValueError(
            f"GPS observation approach edges {observed_edges} do not match expected {expected_edges}."
        )

    leaked = FORBIDDEN_GROUND_TRUTH_COLUMNS.intersection(gps_df.columns)
    if leaked:
        raise ValueError(f"GPS observation contains forbidden ground-truth-shaped column(s): {leaked}")


# ============================================================================
# Probe selection -- deterministic per (seed, vehicle_id) SCORE, then an
# exact top-K cut, not an independent per-vehicle probability threshold.
# UNCHANGED from prior revisions -- see module docstring point 8.
# ============================================================================

def _probe_score(vehicle_id: str, seed: int) -> float:
    """Deterministic per-vehicle score in [0, 1), derived from a hash of
    (seed, vehicle_id). Same vehicle_id + seed always gives the same
    score, independent of how many vehicles exist, what order they're
    processed in, or what penetration_rate is requested -- rate is not
    part of the hash input, only the cutoff applied to the ranking below."""
    digest = hashlib.sha256(f"{seed}:{vehicle_id}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def select_probe_vehicles(vehicle_ids: Set[str], seed: int, penetration_rate: float) -> Set[str]:
    """Select EXACTLY round(len(vehicle_ids) * penetration_rate) vehicles
    as GPS probes -- an exact count, not an independent coin flip per
    vehicle. Ranks every vehicle_id by its deterministic _probe_score()
    and takes the lowest-scoring K. Ties are broken by vehicle_id (string
    order) so the ranking -- and therefore the selection -- is fully
    deterministic even in the astronomically unlikely case of a hash
    collision. Recomputing over the same (vehicle_ids, seed,
    penetration_rate) always returns the same set, regardless of the
    order vehicle_ids is iterated in (it's a set, and sorting fixes
    order for the ranking itself)."""
    if not vehicle_ids:
        return set()
    target_count = round(len(vehicle_ids) * penetration_rate)
    target_count = min(max(target_count, 0), len(vehicle_ids))
    ranked = sorted(vehicle_ids, key=lambda vid: (_probe_score(vid, seed), vid))
    return set(ranked[:target_count])


# ============================================================================
# THE single underlying GPS observation: full per-probe trajectory,
# any edge, native 1-second resolution.
# ============================================================================



# Cheng note:
# The full per-probe trajectory is preserved across approach, internal,
# and outgoing edges because Cheng's critical-point sequence may require
# trajectory information on both sides of the stop line, especially for
# confirming the post-green acceleration regime used to identify Type III.
# Type II itself is the point where the probe slows and joins the queue.

def build_probe_trajectories(probe_full_df: pd.DataFrame) -> pd.DataFrame:
    """Build the underlying 1-second, per-probe GPS trajectory.

    probe_full_df must already be restricted to (a) the observation
    window and (b) the selected probe vehicles -- but, unlike prior
    revisions, must NOT be pre-filtered to approach edges only. Every
    row a selected probe has anywhere in the window (approach, internal
    junction, or outgoing edge) is preserved, because Cheng et al.'s
    Type II critical point can fall on either side of the stop line and
    a trajectory that's cut off at the approach-edge boundary would
    truncate exactly the region that matters.

    Adds two derived fields on top of the raw per-row data:

    - approach_edge: which of the four approach edges this probe's
      current intersection crossing belongs to. Set to edge_id while the
      probe is actually on an approach edge (using the canonical
      is_approach_edge flag from trajectory_utils, not a recomputed
      edge-id check), then forward-filled per probe so later
      internal/outgoing rows are still attributable to that approach.
      Never back-filled: rows recorded before a probe's first approach
      entry (if any) correctly stay NaN, and a probe that never touches
      an approach edge in this window keeps approach_edge as NaN for its
      entire trajectory -- this is never fabricated.
    - distance_to_stopline_m: only meaningful while the probe is
      actually on an approach edge. trajectory_utils' resolution leaves
      a network-based fallback of 0.0 on every other edge type (fine for
      its other callers, which only ever look at approach-edge rows);
      here that would misread as "at the stop line" for a probe that has
      long since crossed it, so it is blanked to NaN for any row that is
      not on an approach edge.
    """
    if probe_full_df.empty:
        return pd.DataFrame(columns=PROBE_TRAJECTORY_COLUMNS)

    df = probe_full_df.sort_values(["vehicle_id", "timestamp"]).copy()

    is_approach = df[APPROACH_FLAG_COLUMN].astype(bool)

    # approach_edge: edge_id where currently on an approach edge, else
    # NaN, then carried forward per probe so post-stop-line rows still
    # know which approach they belong to. groupby().ffill() only fills
    # forward -- rows before a probe's first approach entry stay NaN.
    df["approach_edge"] = df["edge_id"].where(is_approach)
    df["approach_edge"] = df.groupby("vehicle_id")["approach_edge"].ffill()

    # distance_to_stopline_m is physically meaningful only on an approach
    # edge -- blank the network-fallback default everywhere else rather
    # than let it read as "at the stop line".
    df["distance_to_stopline_m"] = df["distance_to_stopline_m"].where(is_approach)

    out = df.rename(columns={"vehicle_id": "probe_id"})[PROBE_TRAJECTORY_COLUMNS]
    return out.sort_values(["probe_id", "timestamp"]).reset_index(drop=True)


def build_gps_observations(
    probe_trajectories_df: pd.DataFrame,
    approach_edges: List[str],
    sim_begin: int,
    sim_end: int,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Aggregate the approach-edge rows of the underlying 1s per-probe
    trajectory (built by build_probe_trajectories, above) onto the
    SHARED 5-second observation grid (SAMPLING_INTERVAL_S) that
    camera_timeseries.csv and observation_assembler.py's join both rely
    on. This is now explicitly a DOWNSTREAM VIEW of the single per-probe
    GPS trajectory, not an independent read of the raw file -- there is
    one underlying GPS simulation, this is one of two ways it gets
    consumed.

    Reports only facts a real GPS-equipped vehicle could contribute:
    that it was present on a given approach edge at a given time, its
    speed, and its distance to the stop line. A probe far upstream is
    simply a probe far upstream -- this function does not infer, cap, or
    reinterpret that distance as a queue-length estimate; that
    interpretation (including the ~483m road length vs. ~150m camera
    range distinction) belongs to later feature engineering, not here.

    Returns (gps_df, aggregate_source_df) -- aggregate_source_df is the
    approach-edge-only slice of probe_trajectories_df this was built
    from, returned alongside purely for validation.
    """
    approach_rows = probe_trajectories_df[probe_trajectories_df["edge_id"].isin(approach_edges)]

    sample_times = list(range(sim_begin, sim_end + 1, SAMPLING_INTERVAL_S))
    rows = []

    for t in sample_times:
        snapshot = approach_rows[approach_rows["timestamp"] == t]
        for edge in approach_edges:
            on_edge = snapshot[snapshot["edge_id"] == edge]
            count = len(on_edge)

            rows.append({
                "timestamp": t,
                "approach_edge": edge,
                "probe_count": count,
                "probe_mean_speed_mps": round(float(on_edge["speed_mps"].mean()), 4) if count > 0 else None,
                "probe_min_distance_to_stopline_m": round(float(on_edge["distance_to_stopline_m"].min()), 2) if count > 0 else None,
                "probe_max_distance_to_stopline_m": round(float(on_edge["distance_to_stopline_m"].max()), 2) if count > 0 else None,
            })

    return pd.DataFrame(rows), approach_rows


# ============================================================================
# Writing outputs
# ============================================================================

def write_outputs(
    out_dir: Path,
    tag: str,
    gps_df: pd.DataFrame,
    probe_ids: Set[str],
    all_vehicle_ids: Set[str],
    seed: int,
    penetration_rate_requested: float,
    probe_ids_observed: Set[str],
    probe_trajectories_df: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    gps_df.to_csv(out_dir / f"gps_{tag}_timeseries.csv", index=False)
    probe_trajectories_df.to_csv(out_dir / f"gps_{tag}_probe_trajectories.csv", index=False)

    realized_rate = (len(probe_ids) / len(all_vehicle_ids)) if all_vehicle_ids else 0.0
    observed_rate_of_probes = (len(probe_ids_observed) / len(probe_ids)) if probe_ids else 0.0
    with open(out_dir / f"gps_{tag}_probe_ids.json", "w", encoding="utf-8") as f:
        json.dump({
            "penetration_rate_requested": penetration_rate_requested,
            "penetration_rate_realized": realized_rate,
            "seed": seed,
            "total_vehicles_considered": len(all_vehicle_ids),
            "probe_vehicle_count": len(probe_ids),
            "probe_vehicle_ids": sorted(probe_ids),
            "probe_vehicle_count_observed_in_window": len(probe_ids_observed),
            "probe_vehicle_ids_observed_in_window": sorted(probe_ids_observed),
            "probe_observed_rate_in_window": round(observed_rate_of_probes, 4),
            "probe_trajectory_rows": int(len(probe_trajectories_df)),
            "probe_trajectory_sampling_interval_s": PROBE_TRAJECTORY_SAMPLING_INTERVAL_S,
            "_note": (
                "probe_vehicle_ids is the COMPLETE set of GPS-equipped vehicles for this scenario, "
                "selected from every vehicle_id in the raw trajectory file -- not just vehicle_ids "
                "appearing in the observation window. A probe absent from "
                "probe_vehicle_ids_observed_in_window is still a designated probe; it simply had no "
                "row at all (on any edge) during this scenario's simulation_begin-simulation_end "
                "viewing window. gps_{tag}_probe_trajectories.csv is the single underlying GPS "
                "observation -- every row (any edge) a selected probe has in the window, at native "
                "1-second resolution. gps_{tag}_timeseries.csv is a DOWNSTREAM AGGREGATION of that "
                "same per-probe trajectory (approach-edge rows only, onto the shared 5s observation "
                "grid) -- kept only because the current Layer 2 pipeline already consumes aggregate "
                "GPS statistics, not a second, independent sensor."
            ),
        }, f, indent=2)


# ============================================================================
# Observation-window filtering (defensive, mirrors ground_truth.py)
# ============================================================================

def restrict_to_observation_window(df: pd.DataFrame, sim_begin: int, sim_end: int) -> pd.DataFrame:
    """Defensively restrict trajectory rows to the scenario's own primary
    observation period before anything else happens, so a future change
    to what vehicle_trajectories.csv contains can never leak into GPS
    observations. Mirrors dataset/ground_truth.py's identical filter."""
    mask = (df["timestamp"] >= sim_begin) & (df["timestamp"] <= sim_end)
    return df.loc[mask].copy()


# ============================================================================
# Orchestration -- one scenario
# ============================================================================

def process_scenario(scenario_dir: Path, cfg: dict, penetration_rate: float) -> dict:
    validate_penetration_rate(penetration_rate)
    validate_scenario_directory(scenario_dir)

    scenario = load_scenario_metadata(scenario_dir)
    approach_edges = cfg["network"]["approaches"]
    seed = int(scenario["seed"]) + GPS_SEED_OFFSET
    sim_begin = int(scenario["simulation_begin"])
    sim_end = int(scenario["simulation_end"])

    df = load_trajectories(scenario_dir)
    validate_raw_trajectory_schema(df)

    # Probe selection uses the COMPLETE scenario vehicle population --
    # every vehicle_id that appears ANYWHERE in the raw trajectory file,
    # taken BEFORE the observation-window filter below. Penetration rate
    # is a property of the scenario's vehicle population ("11% of this
    # scenario's vehicles are GPS-equipped"), not of the viewing window:
    # a vehicle designated as a probe that happens not to have any row
    # inside the window is still a probe, it just has nothing to report.
    all_vehicle_ids = set(df["vehicle_id"].unique())
    probe_ids = select_probe_vehicles(all_vehicle_ids, seed, penetration_rate)
    validate_probe_selection(probe_ids, all_vehicle_ids, seed, penetration_rate)

    df = restrict_to_observation_window(df, sim_begin, sim_end)

    lane_metadata = load_lane_metadata()
    df = resolve_distance_to_stopline(df, lane_metadata, approach_edges)

    # The single underlying GPS observation: EVERY row (any edge --
    # approach, internal, or outgoing) belonging to a selected probe
    # inside the observation window. This is deliberately NOT filtered to
    # approach edges -- that filtering happens downstream, only for the
    # 5s aggregate, in build_gps_observations().
    probe_full_df = df[df["vehicle_id"].isin(probe_ids)]

    probe_trajectories_df = build_probe_trajectories(probe_full_df)
    validate_probe_trajectories(probe_trajectories_df, probe_ids, approach_edges, sim_begin, sim_end)

    # 5s aggregate is a downstream view of the per-probe trajectory above
    # (approach-edge rows only) -- not an independent read of the raw file.
    gps_df, aggregate_source_df = build_gps_observations(
        probe_trajectories_df, approach_edges, sim_begin, sim_end
    )
    validate_gps_observations(gps_df, aggregate_source_df, probe_ids, approach_edges, sim_begin, sim_end)

    # Of the designated probes, how many had ANY row at all (any edge) in
    # the observation window -- a separate, smaller-or-equal quantity
    # from probe_ids itself. Not fed back into probe selection anywhere.
    probe_ids_observed = set(probe_full_df["vehicle_id"].unique())

    tag = f"p{int(round(penetration_rate * 100)):02d}"
    out_dir = scenario_dir / "observations"
    write_outputs(out_dir, tag, gps_df, probe_ids, all_vehicle_ids, seed, penetration_rate,
                  probe_ids_observed, probe_trajectories_df)

    realized_rate = (len(probe_ids) / len(all_vehicle_ids)) if all_vehicle_ids else 0.0
    print(f"{scenario['scenario_id']}: GPS observation ({tag}) written to {out_dir}")
    if all_vehicle_ids:
        print(f"  probes designated: {len(probe_ids)}/{len(all_vehicle_ids)} scenario vehicles "
              f"(requested {penetration_rate:.0%}, realized {realized_rate:.1%})")
        print(f"  probes observed in {sim_begin}-{sim_end}s window: "
              f"{len(probe_ids_observed)}/{len(probe_ids)}")
        print(f"  probe trajectory rows (per-probe, per-second, all edges, for Cheng-style analysis): "
              f"{len(probe_trajectories_df)}")
    else:
        print("  no vehicles recorded for this scenario")

    return {
        "scenario_id": scenario["scenario_id"],
        "penetration_requested": penetration_rate,
        "penetration_realized": realized_rate,
        "total_vehicles": len(all_vehicle_ids),
        "probe_count": len(probe_ids),
        "probe_count_observed_in_window": len(probe_ids_observed),
        "probe_trajectory_rows": len(probe_trajectories_df),
    }


# ============================================================================
# Scenario discovery + CLI
# ============================================================================

def find_scenarios() -> List[Path]:
    return sorted(p for p in SCENARIOS_DIR.glob("scenario_*") if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build sparse GPS/probe observations from raw SUMO trajectories."
    )
    parser.add_argument("--scenario", type=str, default=None,
                         help="Run one scenario, e.g. scenario_normal_balanced. If omitted, runs all found scenarios.")
    parser.add_argument("--penetration", type=float, default=0.10,
                         help="Fraction of vehicles that are GPS-equipped probes, in (0, 1]. Default 0.10.")
    args = parser.parse_args()

    try:
        validate_penetration_rate(args.penetration)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

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
    succeeded = []
    for scenario_dir in scenario_dirs:
        try:
            result = process_scenario(scenario_dir, cfg, args.penetration)
            succeeded.append(result)
        except Exception as exc:
            print(f"FAILED: {scenario_dir.name}: {exc}")
            failed.append(scenario_dir.name)

    print(f"\nDone. {len(succeeded)}/{len(scenario_dirs)} succeeded.")
    if succeeded:
        print(f"{'scenario_id':30} {'requested':>10} {'realized':>10} {'vehicles':>9} {'probes':>7} {'traj_rows':>10}")
        for r in succeeded:
            print(f"{r['scenario_id']:30} {r['penetration_requested']:>10.0%} "
                  f"{r['penetration_realized']:>10.1%} {r['total_vehicles']:>9} {r['probe_count']:>7} "
                  f"{r['probe_trajectory_rows']:>10}")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()