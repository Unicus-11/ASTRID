"""
ground_truth.py
================
ASTRID Prototype -- Ground Truth Builder

RESPONSIBILITY:
    Read the RAW trajectory CSV + lane_metadata.json produced by
    sumo/run_scenarios.py, and compute the TRUE per-approach traffic
    state at every sampling interval: vehicle count, mean speed,
    density, flow, queue count, queue length (m), whether the queue
    extends past the camera range, and realized demand/composition.

    This is the answer key. Nothing here is limited to what a camera
    or GPS probe could see -- that limiting happens in sensors/.
    Nothing here is an ML feature -- that happens in a later
    feature_builder.py, deliberately not built yet.

Reads (per scenario):
    generated_scenarios/scenario_XXXX/scenario.json
    generated_scenarios/scenario_XXXX/raw_output/vehicle_trajectories.csv
    generated_scenarios/scenario_XXXX/raw_output/lane_metadata.json

Writes (per scenario):
    generated_scenarios/scenario_XXXX/ground_truth/state_timeseries.csv
    generated_scenarios/scenario_XXXX/ground_truth/realized_demand.json
    generated_scenarios/scenario_XXXX/ground_truth/realized_composition.json
    generated_scenarios/scenario_XXXX/ground_truth/summary.json

Run:
    python dataset/ground_truth.py --scenario scenario_0001
    python dataset/ground_truth.py                # all scenarios

Depends on: pandas (pip install pandas)

v0.2 changes (this revision):
    - run_scenarios.py (v0.3.1) now runs a data-collection period
      (simulation_begin -> simulation_end) followed by a separate,
      unsaved clearance period. vehicle_trajectories.csv should
      therefore already contain only data-collection-period rows, but
      this file no longer trusts that implicitly: every trajectory
      frame is explicitly filtered to
      [scenario.simulation_begin, scenario.simulation_end] by
      timestamp before any ground-truth quantity is computed, so a
      future change to the raw file's format/contents can never leak
      clearance-period rows into the ground truth.
    - vehicle_trajectories.csv now includes a raw
      "distance_from_stop_line_m" column (computed by run_scenarios.py
      itself from SUMO's own lane length/position at record time).
      That column is now the primary source for per-vehicle
      distance-to-stopline; the old network-metadata-based calculation
      (attach_distance_to_stopline) is used only as a fallback, and
      only for rows where the raw column is absent or empty (e.g. an
      older trajectories file, or a non-approach edge where the raw
      script did not compute a value). Everything downstream (queue
      flagging, queue length/count, etc.) is unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

from trajectory_utils import (
    SAMPLING_INTERVAL_S,
    load_trajectories,
    load_lane_metadata,
    attach_distance_to_stopline,
    flag_queued,
)


# ============================================================================
# PATHS -- mirrors sumo/run_scenarios.py's layout conventions
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMO_DIR = PROJECT_ROOT / "sumo"
SCENARIOS_DIR = SUMO_DIR / "generated_scenarios"
SCENARIO_CONFIG_FILE = SUMO_DIR / "scenario_config.json"

# Raw column written by run_scenarios.py's trajectory writer.
RAW_DISTANCE_COLUMN = "distance_from_stop_line_m"
# Column name used throughout the rest of this module's calculations
# (queue flagging, queue length, etc.) -- unchanged from the prior version.
RESOLVED_DISTANCE_COLUMN = "distance_to_stopline_m"


# ============================================================================
# Config / loading
# ============================================================================

def load_network_config() -> dict:
    with open(SCENARIO_CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scenario_metadata(scenario_dir: Path) -> dict:
    with open(scenario_dir / "scenario.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# Observation-window filtering
# ============================================================================

def restrict_to_observation_window(df: pd.DataFrame, sim_begin: int, sim_end: int) -> pd.DataFrame:
    """Defensively restrict trajectory rows to the scenario's own primary
    observation period (scenario.json's simulation_begin -> simulation_end).

    run_scenarios.py (v0.3.1) only ever writes data-collection-period rows
    to vehicle_trajectories.csv -- the clearance period that follows is not
    saved to disk. This filter does not depend on that being true, though:
    it re-derives the correct window from scenario.json and drops anything
    outside [sim_begin, sim_end] by timestamp, so a future change to the
    raw file's format or contents (e.g. if clearance-period rows were ever
    added) can never silently leak into the ground truth.
    """
    mask = (df["timestamp"] >= sim_begin) & (df["timestamp"] <= sim_end)
    return df.loc[mask].copy()


# ============================================================================
# Distance-to-stopline resolution -- raw column first, network calc fallback
# ============================================================================

def resolve_distance_to_stopline(
    df: pd.DataFrame, lane_metadata: dict, approach_edges: List[str]
) -> pd.DataFrame:
    """Populate RESOLVED_DISTANCE_COLUMN ("distance_to_stopline_m"), the
    column every downstream calculation (queue flagging, queue length,
    etc.) reads.

    Preference order, per row:
      1. The raw "distance_from_stop_line_m" column already computed by
         run_scenarios.py at record time (from SUMO's own lane length and
         lane position) -- used wherever it is present and numeric.
      2. The old network-metadata-based calculation
         (attach_distance_to_stopline), used ONLY as a fallback for rows
         where the raw column is missing, empty, or non-numeric (e.g. an
         older trajectories file that predates the raw column, or a
         non-approach edge/row the raw script left blank).
    """
    if RAW_DISTANCE_COLUMN in df.columns:
        raw_distance = pd.to_numeric(df[RAW_DISTANCE_COLUMN], errors="coerce")
        missing_mask = raw_distance.isna()

        if missing_mask.any():
            fallback_df = attach_distance_to_stopline(df.copy(), lane_metadata, approach_edges)
            fallback_distance = fallback_df[RESOLVED_DISTANCE_COLUMN]
            df[RESOLVED_DISTANCE_COLUMN] = raw_distance.where(~missing_mask, fallback_distance)
        else:
            df[RESOLVED_DISTANCE_COLUMN] = raw_distance

        return df

    # No raw column at all (e.g. trajectories file predates run_scenarios.py's
    # raw distance column) -- fall back to the old calculation entirely.
    return attach_distance_to_stopline(df, lane_metadata, approach_edges)


# ============================================================================
# Step 1: per-approach, per-interval state timeseries (the ground truth)
# ============================================================================

def build_state_timeseries(
    df: pd.DataFrame,
    approach_edges: List[str],
    approach_length_m: float,
    camera_range_m: float,
    sim_begin: int,
    sim_end: int,
) -> pd.DataFrame:
    approach_length_km = approach_length_m / 1000.0
    interval_hours = SAMPLING_INTERVAL_S / 3600.0

    # -- Flow: count vehicles that CROSS OUT of an approach edge within each interval --
    df_sorted = df.sort_values(["vehicle_id", "timestamp"])
    prev_edge = df_sorted.groupby("vehicle_id")["edge_id"].shift(1)

    # is_approach_edge arrives from the raw CSV as int64 (0/1). shift()
    # introduces NaN for each group's first row, which silently promotes
    # the column to float64 -- fillna(False) does NOT undo that promotion,
    # it just writes 0.0 in place of NaN. Bitwise `&` between that float64
    # series and the int64 `~is_approach_edge` series is what raises
    # "unsupported operand type(s) for &: 'float' and 'int'" (and even
    # without the error, `~` on an int column is a numeric bitwise-not,
    # not a logical negation, so it would be silently wrong too).
    # Casting to real bool dtype on both sides before shifting/negating
    # fixes the dtype mismatch and keeps the logic identical: a vehicle
    # "crossed out" of an approach edge if it WAS on one at the previous
    # timestamp and is NOT on one now.
    is_approach_bool = df_sorted["is_approach_edge"].astype(bool)
    was_on_approach = is_approach_bool.groupby(df_sorted["vehicle_id"]).shift(1).fillna(False).astype(bool)
    crossed_out = was_on_approach & (~is_approach_bool)
    crossing_events = df_sorted.loc[crossed_out, ["timestamp", "vehicle_id"]].copy()
    crossing_events["approach_edge"] = prev_edge[crossed_out]
    crossing_events = crossing_events[crossing_events["approach_edge"].isin(approach_edges)]

    sample_times = list(range(sim_begin, sim_end + 1, SAMPLING_INTERVAL_S))
    rows = []

    for t in sample_times:
        snapshot = df[df["timestamp"] == t]
        interval_start = max(sim_begin, t - SAMPLING_INTERVAL_S)
        crossings_in_window = crossing_events[
            (crossing_events["timestamp"] > interval_start) & (crossing_events["timestamp"] <= t)
        ]

        for edge in approach_edges:
            on_edge = snapshot[snapshot["edge_id"] == edge]
            queued = on_edge[on_edge["is_queued"]]

            vehicle_count = len(on_edge)
            mean_speed = float(on_edge["speed_mps"].mean()) if vehicle_count > 0 else 0.0
            density_veh_per_km = vehicle_count / approach_length_km if approach_length_km > 0 else 0.0

            queue_count = len(queued)
            queue_length_m = float(queued["distance_to_stopline_m"].max()) if queue_count > 0 else 0.0
            queue_beyond_camera = queue_length_m > camera_range_m

            edge_crossings = crossings_in_window[crossings_in_window["approach_edge"] == edge]
            flow_veh_per_hour = len(edge_crossings) / interval_hours if interval_hours > 0 else 0.0

            rows.append({
                "timestamp": t,
                "approach_edge": edge,
                "vehicle_count": vehicle_count,
                "mean_speed_mps": round(mean_speed, 4),
                "density_veh_per_km": round(density_veh_per_km, 4),
                "flow_veh_per_hour": round(flow_veh_per_hour, 2),
                "queue_count": queue_count,
                "queue_length_m": round(queue_length_m, 2),
                "queue_beyond_camera": bool(queue_beyond_camera),
            })

    return pd.DataFrame(rows)


# ============================================================================
# Step 4: realized demand + composition (requested vs. what SUMO actually produced)
# ============================================================================

def compute_realized_demand(df: pd.DataFrame, scenario: dict, approach_edges: List[str],
                             approach_name_to_edge: Dict[str, str]) -> dict:
    first_seen = df.sort_values("timestamp").groupby("vehicle_id").first()
    duration_hours = (scenario["simulation_end"] - scenario["simulation_begin"]) / 3600.0

    edge_to_name = {v: k for k, v in approach_name_to_edge.items()}
    per_approach = {}
    total_realized = 0

    for edge in approach_edges:
        name = edge_to_name.get(edge, edge)
        count = int((first_seen["edge_id"] == edge).sum())
        total_realized += count
        requested_share = scenario["approach_distribution"].get(name, 0.0)
        requested_veh_per_hour = scenario["demand_rate_veh_per_hour"] * requested_share
        per_approach[name] = {
            "edge": edge,
            "realized_vehicle_count": count,
            "realized_veh_per_hour": round(count / duration_hours, 1) if duration_hours > 0 else 0.0,
            "requested_veh_per_hour": round(requested_veh_per_hour, 1),
        }

    return {
        "requested_total_veh_per_hour": scenario["demand_rate_veh_per_hour"],
        "realized_total_vehicle_count": total_realized,
        "realized_total_veh_per_hour": round(total_realized / duration_hours, 1) if duration_hours > 0 else 0.0,
        "per_approach": per_approach,
        "_note": "realized_vehicle_count counts distinct vehicles first observed on each approach edge; "
                 "it will not exactly match the requested rate due to normal stochastic variation in SUMO's "
                 "flow emission process -- large, systematic gaps are worth investigating, small ones are not.",
    }


def compute_realized_composition(df: pd.DataFrame, approach_edges: List[str]) -> dict:
    first_seen = df.sort_values("timestamp").groupby("vehicle_id").first()

    overall = first_seen["vehicle_type"].value_counts(normalize=True).round(4).to_dict()

    per_approach = {}
    for edge in approach_edges:
        subset = first_seen[first_seen["edge_id"] == edge]
        if len(subset) == 0:
            per_approach[edge] = {}
            continue
        per_approach[edge] = subset["vehicle_type"].value_counts(normalize=True).round(4).to_dict()

    return {"overall": overall, "per_approach_edge": per_approach}


# ============================================================================
# Step 5: summary
# ============================================================================

def build_summary(state_df: pd.DataFrame, realized_demand: dict, realized_composition: dict,
                   scenario: dict) -> dict:
    per_approach_summary = {}
    for edge, group in state_df.groupby("approach_edge"):
        per_approach_summary[edge] = {
            "max_queue_length_m": float(group["queue_length_m"].max()),
            "max_queue_count": int(group["queue_count"].max()),
            "intervals_with_queue": int((group["queue_count"] > 0).sum()),
            "intervals_with_queue_beyond_camera": int(group["queue_beyond_camera"].sum()),
            "max_density_veh_per_km": float(group["density_veh_per_km"].max()),
            "max_flow_veh_per_hour": float(group["flow_veh_per_hour"].max()),
            "min_mean_speed_mps": float(group[group["vehicle_count"] > 0]["mean_speed_mps"].min())
                if (group["vehicle_count"] > 0).any() else None,
        }

    any_queue_beyond_camera = any(
        v["intervals_with_queue_beyond_camera"] > 0 for v in per_approach_summary.values()
    )

    return {
        "scenario_id": scenario["scenario_id"],
        "demand_class": scenario["demand_class"],
        "webster_diagnostic": scenario.get("_webster_diagnostic"),
        "per_approach": per_approach_summary,
        "any_queue_beyond_camera_range": any_queue_beyond_camera,
        "realized_demand": realized_demand,
        "realized_composition": realized_composition,
        "_note": "This is MEASURED from SUMO output -- unlike the Webster-Y or expected_regime_hint "
                 "diagnostics in scenario.json, these numbers are ground truth, not a planning estimate.",
    }


# ============================================================================
# Orchestration
# ============================================================================

def process_scenario(scenario_dir: Path, cfg: dict) -> dict:
    scenario = load_scenario_metadata(scenario_dir)
    approach_edges = cfg["network"]["approaches"]
    approach_name_to_edge = cfg["approach_name_to_edge"]
    approach_length_m = cfg["network"]["approach_length_m"]
    camera_range_m = cfg["network"]["camera_range_m"]

    sim_begin = int(scenario["simulation_begin"])
    sim_end = int(scenario["simulation_end"])

    df = load_trajectories(scenario_dir)

    # Defensive filter: ground truth is computed ONLY from the scenario's
    # own primary observation window. This must never include
    # run_scenarios.py's clearance period, regardless of what the raw
    # trajectories file happens to contain.
    df = restrict_to_observation_window(df, sim_begin, sim_end)

    lane_metadata = load_lane_metadata()

    # Prefer the raw per-row distance-to-stopline column written directly
    # by run_scenarios.py; fall back to the old network-metadata-based
    # calculation only where that raw value is missing.
    df = resolve_distance_to_stopline(df, lane_metadata, approach_edges)
    df = flag_queued(df)

    state_df = build_state_timeseries(
        df, approach_edges, approach_length_m, camera_range_m,
        sim_begin, sim_end,
    )
    realized_demand = compute_realized_demand(df, scenario, approach_edges, approach_name_to_edge)
    realized_composition = compute_realized_composition(df, approach_edges)
    summary = build_summary(state_df, realized_demand, realized_composition, scenario)

    out_dir = scenario_dir / "ground_truth"
    out_dir.mkdir(parents=True, exist_ok=True)

    state_df.to_csv(out_dir / "state_timeseries.csv", index=False)
    with open(out_dir / "realized_demand.json", "w", encoding="utf-8") as f:
        json.dump(realized_demand, f, indent=2)
    with open(out_dir / "realized_composition.json", "w", encoding="utf-8") as f:
        json.dump(realized_composition, f, indent=2)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"{scenario['scenario_id']}: ground truth written to {out_dir}")
    print(f"  max queue by approach: "
          f"{ {e: v['max_queue_length_m'] for e, v in summary['per_approach'].items()} }")
    print(f"  any queue beyond camera range ({camera_range_m}m): {summary['any_queue_beyond_camera_range']}")

    return summary


def find_scenarios() -> List[Path]:
    return sorted(p for p in SCENARIOS_DIR.glob("scenario_*") if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ground-truth traffic state from raw SUMO trajectories.")
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