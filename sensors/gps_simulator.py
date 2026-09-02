"""
gps_simulator.py
====================
ASTRID Prototype -- GPS / Probe Vehicle Sensor Simulator  (v0.2)

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

v0.2 changes (this revision -- aligning with the current project):
    - BUG FIX: the previous implementation filtered on a column named
      "is_on_approach", which does not exist in the current raw schema.
      The canonical column (written by run_scenarios.py, defined in
      trajectory_utils.py) is "is_approach_edge". Using the wrong name
      meant this filter would have raised a KeyError against current
      data.
    - Distance-to-stopline now uses trajectory_utils.resolve_distance_to_stopline(),
      which prefers the raw per-row "distance_from_stop_line_m" column
      written directly by run_scenarios.py (from SUMO's own lane
      length/position at record time) and falls back to the older
      network-metadata calculation only where that raw value is
      missing. The previous implementation called
      attach_distance_to_stopline() directly, which ignores the raw
      column entirely and always recomputes from sq.net.xml. This
      matches the same raw-preferred resolution ground_truth.py now
      uses, so GPS and ground truth agree on what "distance to
      stopline" means for a given row.
    - Removed the flag_queued() call and any notion of "queued" from
      this script. Queue determination is a shared *judgment* (a speed
      + duration threshold) applied identically for ground truth --
      computing it here added an unused, unrequested field and risked
      quietly turning a raw GPS ping into a queue-membership judgment,
      which belongs to feature engineering, not the sensor layer. GPS
      now reports only position/speed facts about probes.
    - Added an explicit observation-window filter
      (restrict_to_observation_window) mirroring ground_truth.py's own
      defensive filter: trajectory rows are filtered to
      [scenario.simulation_begin, scenario.simulation_end] by
      timestamp before anything else happens, so a future change to
      what vehicle_trajectories.csv contains (e.g. if a clearance
      period were ever saved) can never leak into GPS observations.
    - No more hardcoded "scenario_0001"-style scenario naming in
      docstrings/examples -- the current project uses 12 fixed named
      scenarios (scenario_normal_balanced, scenario_low_demand, ...).
      The code itself was already naming-agnostic (glob scenario_*),
      so no logic changes were needed there.
    - Corrected this docstring's "Reads" section: there is no
      lane_metadata.json file. Lane geometry comes from
      trajectory_utils.load_lane_metadata(), which parses
      sq.net.xml directly (used only as the fallback path inside
      resolve_distance_to_stopline).
    - Added explicit, itemized validation at every stage (schema,
      penetration rate, probe-subset containment, timestamp window,
      expected approach edges, determinism, ground-truth non-leakage,
      empty-probe safety, realized-vs-requested-rate reporting) instead
      of relying on things happening to work. See validate_* functions.
    - Restructured into the requested function shape: load config /
      load scenario metadata / deterministic probe selection / validate
      raw trajectory schema / build GPS observations / validate GPS
      observations / write outputs / process one scenario / find
      scenarios / main CLI.

v0.3 changes (this revision -- probe-population ordering fix):
    - BUG FIX: probe selection previously ran AFTER
      restrict_to_observation_window(), so the penetration-rate
      denominator was "vehicles with at least one row inside
      0-simulation_end" rather than "vehicles in this scenario."
      penetration_rate is a property of the scenario's vehicle
      population, not of the viewing window, so process_scenario() now
      computes all_vehicle_ids from the full raw trajectory file BEFORE
      restrict_to_observation_window() runs, and only THEN filters to
      the window before building observations. A probe that doesn't
      appear inside the window is still counted as a designated probe
      -- it just has nothing to report -- and that "designated vs.
      actually observed" distinction is now reported separately in both
      the console summary and gps_p{NN}_probe_ids.json
      (probe_vehicle_count vs. probe_vehicle_count_observed_in_window)
      instead of being collapsed into one number.
    - Note on scope: vehicle_trajectories.csv (as currently written by
      run_scenarios.py) is the only file with concrete, non-invented
      vehicle_id values -- scenario_builder.py never assigns them (SUMO
      generates IDs from flow.xml at simulation runtime), and
      run_scenarios.py's realization_audit.json records population
      COUNTS (loaded/departed/pending) but not the underlying ID lists.
      So this fix corrects the ordering for every vehicle that has at
      least one row anywhere in the trajectory file; it cannot recover
      probe status for a vehicle that has zero rows in the file at all
      (e.g. one SUMO never managed to insert onto the network) without
      inventing an ID that was never recorded anywhere.

v0.4 changes (this revision -- exact-count probe selection):
    - BUG FIX: select_probe_vehicles() previously flipped an independent
      per-vehicle coin (probe if hash(seed, vehicle_id) < penetration_rate).
      That gives the CORRECT rate in expectation, but for any one scenario
      the realized count is a binomial draw around N * penetration_rate,
      not exactly N * penetration_rate -- e.g. 1600 vehicles at 11% came
      out to 164, not 176, purely from per-vehicle sampling variance
      (std-dev ~= sqrt(1600 * 0.11 * 0.89) ~= 12.5, so 164 is well within
      normal range, just not what a fixed "11% penetration" experiment
      condition should mean). Fixed by ranking every vehicle_id by a
      deterministic score and taking the exact top
      round(len(vehicle_ids) * penetration_rate). This guarantees the
      probe count for a scenario is always exactly round(N * rate), not
      an expected value with sampling noise.
    - Nice side effect: because the per-vehicle score no longer depends
      on penetration_rate itself (only on seed + vehicle_id), the probe
      sets are NESTED/monotonic across rates for a fixed scenario+seed --
      the 5% probe set is always a subset of the 11% probe set, which is
      a subset of the 20% probe set, etc. Not required by this fix, just
      a useful property if penetration is later swept.
    - TRADE-OFF, stated explicitly: is_probe_vehicle() (a per-vehicle
      boolean, computable in isolation) is removed. Because "is this
      vehicle in the top K?" depends on every other vehicle's score, a
      single vehicle's probe status can no longer be determined without
      the full population -- select_probe_vehicles() always receives the
      complete vehicle_ids set already (see v0.3 above), so this does not
      change any call site, but it does remove the standalone per-vehicle
      helper. If anything outside this file imported is_probe_vehicle
      directly, it will need to call select_probe_vehicles() instead.

ANTI-LEAKAGE RULE:
    This script never reads dataset/ground_truth.py's output and never
    computes queue length, density, or flow. It derives GPS observations
    only from raw_output/vehicle_trajectories.csv -- the same raw file
    ground truth is built from, not ground truth itself.

Reads (per scenario):
    sumo/generated_scenarios/<scenario_id>/scenario.json
    sumo/generated_scenarios/<scenario_id>/raw_output/vehicle_trajectories.csv
    (sq.net.xml, indirectly, only as a distance-to-stopline fallback)

Writes (per scenario, per penetration rate):
    sumo/generated_scenarios/<scenario_id>/observations/gps_p{NN}_timeseries.csv
    sumo/generated_scenarios/<scenario_id>/observations/gps_p{NN}_probe_ids.json

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

# Fields this sensor is allowed to write. Any of these names appearing
# would mean a ground-truth label leaked into the observation -- used as
# a regression guard in validate_gps_observations().
FORBIDDEN_GROUND_TRUTH_COLUMNS = {
    "queue_length_m", "queue_count", "queue_beyond_camera",
    "density_veh_per_km", "flow_veh_per_hour", "vehicle_count",
    "mean_speed_mps",  # ground truth's per-approach mean; GPS has its own probe_mean_speed_mps
}

REQUIRED_RAW_COLUMNS = ["timestamp", "vehicle_id", "edge_id", "speed_mps"]


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


def validate_probe_selection(probe_ids: Set[str], all_vehicle_ids: Set[str], seed: int,
                              penetration_rate: float) -> None:
    """Probe IDs must be a subset of real vehicle IDs, must be exactly
    round(len(all_vehicle_ids) * penetration_rate) in count (not merely
    close to it), and probe selection must be reproducible (same seed +
    vehicle_id set + rate -> same result, independent of dataframe order
    -- verified here by recomputing)."""
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


def validate_gps_observations(
    gps_df: pd.DataFrame,
    probe_df: pd.DataFrame,
    probe_ids: Set[str],
    approach_edges: List[str],
    sim_begin: int,
    sim_end: int,
) -> None:
    """Post-construction checks on the GPS output itself."""
    # No non-probe vehicle contributed to the per-row data the output was aggregated from.
    contributing_ids = set(probe_df["vehicle_id"].unique())
    if not contributing_ids.issubset(probe_ids):
        raise ValueError("GPS observation was built from vehicle(s) outside the selected probe set.")

    if gps_df.empty:
        # Empty probe observations (e.g. very low penetration + few vehicles)
        # are a valid, safe outcome, not an error -- nothing further to check.
        return

    # Timestamps must fall within the scenario's primary observation window.
    out_of_window = gps_df[(gps_df["timestamp"] < sim_begin) | (gps_df["timestamp"] > sim_end)]
    if not out_of_window.empty:
        raise ValueError(
            f"GPS observation contains {len(out_of_window)} row(s) with timestamps outside "
            f"[{sim_begin}, {sim_end}]."
        )

    # Every expected approach edge must be represented (even with probe_count=0).
    observed_edges = set(gps_df["approach_edge"].unique())
    expected_edges = set(approach_edges)
    if observed_edges != expected_edges:
        raise ValueError(
            f"GPS observation approach edges {observed_edges} do not match expected {expected_edges}."
        )

    # No ground-truth column names leaked into the output.
    leaked = FORBIDDEN_GROUND_TRUTH_COLUMNS.intersection(gps_df.columns)
    if leaked:
        raise ValueError(f"GPS observation contains forbidden ground-truth-shaped column(s): {leaked}")


# ============================================================================
# Probe selection -- deterministic per (seed, vehicle_id) SCORE, then an
# exact top-K cut, not an independent per-vehicle probability threshold.
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
# GPS observation: sparse by vehicle count, NOT range-limited
# ============================================================================

def build_gps_observations(
    df: pd.DataFrame,
    probe_ids: Set[str],
    approach_edges: List[str],
    sim_begin: int,
    sim_end: int,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Aggregate raw probe rows into per-timestamp, per-approach GPS
    observations.

    Reports only facts a real GPS-equipped vehicle could contribute:
    that it was present on a given approach edge at a given time, its
    speed, and its distance to the stop line. A probe far upstream is
    simply a probe far upstream -- this function does not infer, cap, or
    reinterpret that distance as a queue-length estimate; that
    interpretation (including the ~483m road length vs. ~150m camera
    range distinction) belongs to later feature engineering.

    Returns (gps_df, probe_df) -- probe_df (the filtered per-row probe
    data used to build gps_df) is returned alongside for validation.
    """
    probe_df = df[df["vehicle_id"].isin(probe_ids) & df[APPROACH_FLAG_COLUMN]]

    sample_times = list(range(sim_begin, sim_end + 1, SAMPLING_INTERVAL_S))
    rows = []

    for t in sample_times:
        snapshot = probe_df[probe_df["timestamp"] == t]
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

    return pd.DataFrame(rows), probe_df


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
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    gps_df.to_csv(out_dir / f"gps_{tag}_timeseries.csv", index=False)

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
            "_note": (
                "probe_vehicle_ids is the COMPLETE set of GPS-equipped vehicles for this scenario, "
                "selected from every vehicle_id in the raw trajectory file -- not just vehicle_ids "
                "appearing in the observation window. A probe absent from "
                "probe_vehicle_ids_observed_in_window is still a designated probe; it simply had no "
                "row to report during this scenario's 0-simulation_end viewing window."
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
    # taken BEFORE the 0-simulation_end observation-window filter below.
    # Penetration rate is a property of the scenario's vehicle population
    # ("11% of this scenario's vehicles are GPS-equipped"), not of the
    # viewing window: a vehicle designated as a probe that happens not to
    # have any row inside the window is still a probe, it just has
    # nothing to report. Selecting probes only from window-filtered rows
    # would silently shrink the denominator whenever any vehicle's rows
    # fell entirely outside [sim_begin, sim_end], understating penetration.
    all_vehicle_ids = set(df["vehicle_id"].unique())
    probe_ids = select_probe_vehicles(all_vehicle_ids, seed, penetration_rate)
    validate_probe_selection(probe_ids, all_vehicle_ids, seed, penetration_rate)

    df = restrict_to_observation_window(df, sim_begin, sim_end)

    lane_metadata = load_lane_metadata()
    df = resolve_distance_to_stopline(df, lane_metadata, approach_edges)

    gps_df, probe_df = build_gps_observations(df, probe_ids, approach_edges, sim_begin, sim_end)
    validate_gps_observations(gps_df, probe_df, probe_ids, approach_edges, sim_begin, sim_end)

    # Of the designated probes, how many actually had a row inside the
    # observation window -- a separate, smaller-or-equal quantity from
    # probe_ids itself. Not fed back into probe selection anywhere.
    probe_ids_observed = set(probe_df["vehicle_id"].unique())

    tag = f"p{int(round(penetration_rate * 100)):02d}"
    out_dir = scenario_dir / "observations"
    write_outputs(out_dir, tag, gps_df, probe_ids, all_vehicle_ids, seed, penetration_rate,
                  probe_ids_observed)

    realized_rate = (len(probe_ids) / len(all_vehicle_ids)) if all_vehicle_ids else 0.0
    print(f"{scenario['scenario_id']}: GPS observation ({tag}) written to {out_dir}")
    if all_vehicle_ids:
        print(f"  probes designated: {len(probe_ids)}/{len(all_vehicle_ids)} scenario vehicles "
              f"(requested {penetration_rate:.0%}, realized {realized_rate:.1%})")
        print(f"  probes observed in {sim_begin}-{sim_end}s window: "
              f"{len(probe_ids_observed)}/{len(probe_ids)}")
    else:
        print("  no vehicles recorded for this scenario")

    return {
        "scenario_id": scenario["scenario_id"],
        "penetration_requested": penetration_rate,
        "penetration_realized": realized_rate,
        "total_vehicles": len(all_vehicle_ids),
        "probe_count": len(probe_ids),
        "probe_count_observed_in_window": len(probe_ids_observed),
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
        print(f"{'scenario_id':30} {'requested':>10} {'realized':>10} {'vehicles':>9} {'probes':>7}")
        for r in succeeded:
            print(f"{r['scenario_id']:30} {r['penetration_requested']:>10.0%} "
                  f"{r['penetration_realized']:>10.1%} {r['total_vehicles']:>9} {r['probe_count']:>7}")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()