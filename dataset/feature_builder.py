"""
feature_builder.py
====================
ASTRID Prototype -- Feature Builder, LAYER 1 (camera-only state estimation)

Implements exactly dataset/FEATURE_PLAN_LAYER1.md:
    A. what's visible      -- camera-only counts / speed / queue
    B. how it's changing   -- 30s deltas
    C. vehicle mix         -- visible composition

Deliberately NOT included yet:
    - GPS / probe features
    - shockwave / LWR physics estimate
    - signal-phase features
    - future-time prediction (this predicts the CURRENT queue only)

Structural leakage guarantee: ground_truth/ is read in exactly ONE
function (build_labels), which writes to a separate labels file. The
rest of this file never opens it.

Reads (per scenario):
    scenario.json
    observations/camera_timeseries.csv     (from sensors/camera_simulator.py)
    raw_output/vehicle_trajectories.csv    (for vehicle-mix only)
    ground_truth/state_timeseries.csv      (ONLY inside build_labels)

Writes:
    features/features_layer1.csv
    features/labels_layer1.csv
    features/feature_manifest_layer1.json

Run:
    python dataset/feature_builder.py --scenario scenario_0001
    python dataset/feature_builder.py                # all scenarios
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

from trajectory_utils import (
    SAMPLING_INTERVAL_S,
    load_trajectories,
    load_lane_metadata,
    attach_distance_to_stopline,
    flag_queued,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMO_DIR = PROJECT_ROOT / "sumo"
SCENARIOS_DIR = SUMO_DIR / "generated_scenarios"
SCENARIO_CONFIG_FILE = SUMO_DIR / "scenario_config.json"

DELTA_WINDOW_S = 30   # "how it's changing" window, per the plan


def load_network_config() -> dict:
    with open(SCENARIO_CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scenario_metadata(scenario_dir: Path) -> dict:
    with open(scenario_dir / "scenario.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# A + B: what's visible, and how it's changing -- from camera_timeseries.csv
# ============================================================================

def add_change_features(camera_df: pd.DataFrame) -> pd.DataFrame:
    df = camera_df.sort_values(["approach_edge", "timestamp"]).copy()
    delta_steps = max(1, DELTA_WINDOW_S // SAMPLING_INTERVAL_S)

    g = df.groupby("approach_edge")
    df[f"queue_length_change_{DELTA_WINDOW_S}s"] = g["visible_queue_length_m"].diff(delta_steps)
    df[f"speed_change_{DELTA_WINDOW_S}s"] = g["visible_mean_speed_mps"].diff(delta_steps)
    return df


# ============================================================================
# C: vehicle mix -- needs raw per-vehicle data (not in camera_timeseries.csv),
# restricted to camera-visible vehicles only.
# ============================================================================

def compute_visible_composition(
    df: pd.DataFrame, approach_edges: List[str], vehicle_types: List[str],
    camera_range_m: float, sim_begin: int, sim_end: int,
) -> pd.DataFrame:
    visible = df[df["is_on_approach"] & (df["distance_to_stopline_m"] <= camera_range_m)]
    sample_times = range(sim_begin, sim_end + 1, SAMPLING_INTERVAL_S)
    rows = []
    for t in sample_times:
        snap = visible[visible["timestamp"] == t]
        for edge in approach_edges:
            on_edge = snap[snap["edge_id"] == edge]
            total = len(on_edge)
            row = {"timestamp": t, "approach_edge": edge}
            for vt in vehicle_types:
                row[f"visible_{vt}_frac"] = (on_edge["vehicle_type"] == vt).sum() / total if total > 0 else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# Labels -- ground truth is read ONLY here, and only here
# ============================================================================

def build_labels(scenario_dir: Path) -> pd.DataFrame:
    gt_path = scenario_dir / "ground_truth" / "state_timeseries.csv"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing {gt_path} -- run dataset/ground_truth.py first.")
    gt = pd.read_csv(gt_path)
    return gt[["timestamp", "approach_edge", "queue_length_m", "queue_beyond_camera"]].rename(
        columns={"queue_length_m": "true_queue_length_m", "queue_beyond_camera": "true_queue_beyond_camera"}
    )


# ============================================================================
# Feature manifest -- documents every column so nothing is a mystery later
# ============================================================================

def build_feature_manifest(vehicle_types: List[str]) -> dict:
    manifest = {
        "layer": 1,
        "purpose": "camera-only state estimation -- predict true_queue_length_m from what a camera alone can see",
        "timestamp": {"kind": "index"},
        "approach_edge": {"kind": "index"},
        "visible_vehicle_count": {"kind": "observed", "source": "camera"},
        "visible_mean_speed_mps": {"kind": "observed", "source": "camera"},
        "visible_queue_count": {"kind": "observed", "source": "camera"},
        "visible_queue_length_m": {"kind": "observed", "source": "camera"},
        "visible_occupancy_fraction": {"kind": "derived", "source": "camera"},
        "queue_reaches_camera_edge": {"kind": "observed", "source": "camera"},
        f"queue_length_change_{DELTA_WINDOW_S}s": {"kind": "derived", "source": "camera history (past only)"},
        f"speed_change_{DELTA_WINDOW_S}s": {"kind": "derived", "source": "camera history (past only)"},
    }
    for vt in vehicle_types:
        manifest[f"visible_{vt}_frac"] = {"kind": "observed", "source": "camera"}

    manifest["_labels_file"] = ("labels_layer1.csv -- true_queue_length_m, true_queue_beyond_camera. "
                                 "Never merge into features; join only at training time.")
    manifest["_not_in_layer1"] = ["GPS/probe features", "shockwave/LWR physics estimate",
                                   "signal-phase features", "future-time prediction"]
    return manifest


# ============================================================================
# Orchestration
# ============================================================================

def process_scenario(scenario_dir: Path, cfg: dict) -> None:
    scenario = load_scenario_metadata(scenario_dir)
    approach_edges = cfg["network"]["approaches"]
    camera_range_m = cfg["network"]["camera_range_m"]
    vehicle_types = list(cfg["vehicle_types"].keys())
    sim_begin, sim_end = int(scenario["simulation_begin"]), int(scenario["simulation_end"])

    camera_path = scenario_dir / "observations" / "camera_timeseries.csv"
    if not camera_path.exists():
        raise FileNotFoundError(f"Missing {camera_path} -- run sensors/camera_simulator.py first.")
    camera_df = pd.read_csv(camera_path)

    features = add_change_features(camera_df)
    features["visible_occupancy_fraction"] = (features["visible_queue_length_m"] / camera_range_m).clip(upper=1.0)

    raw = load_trajectories(scenario_dir)
    lane_metadata = load_lane_metadata()
    raw = attach_distance_to_stopline(raw, lane_metadata, approach_edges)
    raw = flag_queued(raw)  # not used directly in Layer 1 output, kept for consistency with later layers

    composition_df = compute_visible_composition(raw, approach_edges, vehicle_types, camera_range_m, sim_begin, sim_end)
    features = features.merge(composition_df, on=["timestamp", "approach_edge"], how="left")

    labels = build_labels(scenario_dir)
    manifest = build_feature_manifest(vehicle_types)

    out_dir = scenario_dir / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_dir / "features_layer1.csv", index=False)
    labels.to_csv(out_dir / "labels_layer1.csv", index=False)
    with open(out_dir / "feature_manifest_layer1.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"{scenario['scenario_id']}: Layer 1 features written to {out_dir}")
    print(f"  features: {features.shape[0]} rows x {features.shape[1]} cols | labels: {labels.shape[0]} rows")


def find_scenarios() -> List[Path]:
    return sorted(p for p in SCENARIOS_DIR.glob("scenario_*") if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Layer-1 (camera-only state estimation) features + labels.")
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