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
    crossed_out = df_sorted["is_on_approach"].groupby(df_sorted["vehicle_id"]).shift(1).fillna(False) \
        & (~df_sorted["is_on_approach"])
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

    df = load_trajectories(scenario_dir)
    lane_metadata = load_lane_metadata()

    df = attach_distance_to_stopline(df, lane_metadata, approach_edges)
    df = flag_queued(df)

    state_df = build_state_timeseries(
        df, approach_edges, approach_length_m, camera_range_m,
        int(scenario["simulation_begin"]), int(scenario["simulation_end"]),
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