"""
gps_simulator.py
====================
ASTRID Prototype -- GPS / Probe Vehicle Sensor Simulator

RESPONSIBILITY:
    Simulate a sparse fraction of vehicles being GPS-equipped probes.
    Unlike the camera, a probe's position is NOT limited by distance --
    a probe far upstream still reports in. The limit here is COUNT, not
    RANGE: only a fraction (penetration_rate) of vehicles are probes,
    and which ones are probes is unknown to us in advance -- it's a
    property of the vehicle, decided once, deterministically, per
    scenario+penetration-rate seed.

    This is OBSERVATION, not ground truth. A downstream model is only
    ever allowed to see the probe subset produced here, never the full
    vehicle population.

Reads (per scenario):
    sumo/generated_scenarios/scenario_XXXX/scenario.json
    sumo/generated_scenarios/scenario_XXXX/raw_output/vehicle_trajectories.csv
    sumo/generated_scenarios/scenario_XXXX/raw_output/lane_metadata.json

Writes:
    sumo/generated_scenarios/scenario_XXXX/observations/gps_p{PENETRATION}_timeseries.csv
    sumo/generated_scenarios/scenario_XXXX/observations/gps_p{PENETRATION}_probe_ids.json

Run:
    python sensors/gps_simulator.py --scenario scenario_0001 --penetration 0.10
    python sensors/gps_simulator.py --penetration 0.05        # all scenarios, 5% penetration
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import List, Set

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "dataset"))
from trajectory_utils import SAMPLING_INTERVAL_S, load_trajectories, load_lane_metadata, \
    attach_distance_to_stopline, flag_queued  # noqa: E402

SUMO_DIR = PROJECT_ROOT / "sumo"
SCENARIOS_DIR = SUMO_DIR / "generated_scenarios"
SCENARIO_CONFIG_FILE = SUMO_DIR / "scenario_config.json"

GPS_SEED_OFFSET = 5000  # keeps probe-selection randomness independent of the scenario's own seed use


def load_network_config() -> dict:
    with open(SCENARIO_CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scenario_metadata(scenario_dir: Path) -> dict:
    with open(scenario_dir / "scenario.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# Probe selection -- deterministic per (seed, vehicle_id), not per ordering
# ============================================================================

def is_probe_vehicle(vehicle_id: str, seed: int, penetration_rate: float) -> bool:
    """Deterministic per-vehicle coin flip: same vehicle_id + seed always
    gives the same answer, independent of how many vehicles exist or what
    order they're processed in. Uses a hash instead of a seeded RNG stream
    so probe status can be recomputed for one vehicle without replaying
    the whole population."""
    digest = hashlib.sha256(f"{seed}:{vehicle_id}".encode()).hexdigest()
    draw = int(digest[:8], 16) / 0xFFFFFFFF  # -> float in [0, 1)
    return draw < penetration_rate


def select_probe_vehicles(vehicle_ids: Set[str], seed: int, penetration_rate: float) -> Set[str]:
    return {vid for vid in vehicle_ids if is_probe_vehicle(vid, seed, penetration_rate)}


# ============================================================================
# GPS observation: sparse, but NOT range-limited
# ============================================================================

def build_gps_observation(
    df: pd.DataFrame,
    probe_ids: Set[str],
    approach_edges: List[str],
    sim_begin: int,
    sim_end: int,
) -> pd.DataFrame:
    probe_df = df[df["vehicle_id"].isin(probe_ids) & df["is_on_approach"]]

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
                # the furthest-back probe is the closest thing this sensor has to
                # direct evidence of a queue extending past the camera -- when it
                # exists, it is a real (if sparse) sighting, not an estimate.
            })

    return pd.DataFrame(rows)


# ============================================================================
# Orchestration
# ============================================================================

def process_scenario(scenario_dir: Path, cfg: dict, penetration_rate: float) -> pd.DataFrame:
    scenario = load_scenario_metadata(scenario_dir)
    approach_edges = cfg["network"]["approaches"]
    seed = int(scenario["seed"]) + GPS_SEED_OFFSET

    df = load_trajectories(scenario_dir)
    lane_metadata = load_lane_metadata()
    df = attach_distance_to_stopline(df, lane_metadata, approach_edges)
    df = flag_queued(df)

    all_vehicle_ids = set(df["vehicle_id"].unique())
    probe_ids = select_probe_vehicles(all_vehicle_ids, seed, penetration_rate)

    gps_df = build_gps_observation(
        df, probe_ids, approach_edges,
        int(scenario["simulation_begin"]), int(scenario["simulation_end"]),
    )

    tag = f"p{int(round(penetration_rate * 100)):02d}"
    out_dir = scenario_dir / "observations"
    out_dir.mkdir(parents=True, exist_ok=True)
    gps_df.to_csv(out_dir / f"gps_{tag}_timeseries.csv", index=False)
    with open(out_dir / f"gps_{tag}_probe_ids.json", "w", encoding="utf-8") as f:
        json.dump({
            "penetration_rate_requested": penetration_rate,
            "penetration_rate_realized": len(probe_ids) / len(all_vehicle_ids) if all_vehicle_ids else 0.0,
            "seed": seed,
            "total_vehicles": len(all_vehicle_ids),
            "probe_vehicle_count": len(probe_ids),
            "probe_vehicle_ids": sorted(probe_ids),
        }, f, indent=2)

    print(f"{scenario['scenario_id']}: GPS observation ({tag}) written to {out_dir}")
    print(f"  probes: {len(probe_ids)}/{len(all_vehicle_ids)} vehicles "
          f"(requested {penetration_rate:.0%}, realized {len(probe_ids)/len(all_vehicle_ids):.1%})"
          if all_vehicle_ids else "  no vehicles recorded")

    return gps_df


def find_scenarios() -> List[Path]:
    return sorted(p for p in SCENARIOS_DIR.glob("scenario_*") if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sparse GPS/probe observation from raw SUMO trajectories.")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--penetration", type=float, default=0.10,
                         help="Fraction of vehicles that are GPS-equipped probes (default 0.10 = 10%%).")
    args = parser.parse_args()

    if not (0.0 < args.penetration <= 1.0):
        print(f"ERROR: --penetration must be in (0, 1], got {args.penetration}")
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
    for scenario_dir in scenario_dirs:
        try:
            process_scenario(scenario_dir, cfg, args.penetration)
        except Exception as exc:
            print(f"FAILED: {scenario_dir.name}: {exc}")
            failed.append(scenario_dir.name)

    print(f"\nDone. {len(scenario_dirs) - len(failed)}/{len(scenario_dirs)} succeeded.")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()