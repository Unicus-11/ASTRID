"""
observation_assembler.py
====================
ASTRID Prototype -- Observation Assembler

RESPONSIBILITY:
    Combine the already-built sensor observations -- camera_timeseries.csv
    (sensors/camera_simulator.py) and gps_p{TAG}_timeseries.csv
    (sensors/gps_simulator.py) -- into ONE aligned observation table, keyed
    on (timestamp, approach_edge).

    This is a pure data-alignment layer. It does NOT compute any derived
    signal (queue growth rate, shockwave speed, jam density, hidden-queue
    estimate, or any other physics-informed feature) and does NOT read or
    fill anything from dataset/ground_truth.py. Ground truth stays the
    target/reference for a later stage, never an input into the sensor
    observation table. Feature engineering is a separate, later module
    (feature_builder.py), deliberately not built here.

    Camera and GPS quantities are kept separate on purpose:
        visible_vehicle_count  -- ALL vehicles within camera_range_m
        probe_count             -- only the selected penetration-rate subset
    These are never merged into a single generic "vehicle_count".

v0.2 changes (this revision):
    - Default --penetration changed from 0.10 to 0.11 -- the project's
      primary ASTRID experiment is 11% GPS penetration, so running this
      script with no --penetration flag now looks for
      gps_p11_timeseries.csv (matching the file gps_simulator.py's own
      primary run actually produces) instead of gps_p10_timeseries.csv,
      which was never generated.
    - Fixed a validation-order bug: probe_count's NaN->0 fill previously
      happened INSIDE assemble_observations(), before
      validate_assembled_observations() ran. That meant the check
      `merged["probe_count"].isna().any()` could never fire -- by the
      time it ran, every NaN had already been overwritten with 0, so a
      genuine camera/GPS grid misalignment (a (timestamp, approach_edge)
      key present in one file but not the other) would have silently
      looked identical to "GPS observed zero probes here", instead of
      being caught. assemble_observations() now performs ONLY the outer
      join and returns the raw merged frame with NaNs intact;
      process_scenario() runs validate_assembled_observations() against
      that raw frame (so a real gap is still visible as NaN and gets
      caught), and only fills probe_count -> 0 afterward, once alignment
      is confirmed, immediately before writing the file. Order is now:
      merge -> validate coverage -> fill probe_count -> save.

Reads (per scenario -- must already exist; run camera_simulator.py and
gps_simulator.py first):
    sumo/generated_scenarios/<scenario_id>/scenario.json
    sumo/generated_scenarios/<scenario_id>/observations/camera_timeseries.csv
    sumo/generated_scenarios/<scenario_id>/observations/gps_p{TAG}_timeseries.csv

Writes:
    sumo/generated_scenarios/<scenario_id>/observations/assembled_observations_p{TAG}.csv

    (Tagged by penetration rate, matching the GPS file's own p{TAG}
    convention -- an untagged "assembled_observations.csv" would be
    silently overwritten if a second penetration rate is ever assembled
    for the same scenario.)

Run:
    python sensors/observation_assembler.py --scenario scenario_normal_balanced --penetration 0.11
    python sensors/observation_assembler.py                # all scenarios, default 11% penetration
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "dataset"))
from trajectory_utils import SAMPLING_INTERVAL_S  # noqa: E402

SUMO_DIR = PROJECT_ROOT / "sumo"
SCENARIOS_DIR = SUMO_DIR / "generated_scenarios"
SCENARIO_CONFIG_FILE = SUMO_DIR / "scenario_config.json"

# Primary ASTRID experiment penetration rate. Kept as a named constant
# (rather than only living inside argparse's default=) so
# process_scenario() and any future caller share one source of truth for
# "what does 'no --penetration given' mean".
DEFAULT_PENETRATION_RATE = 0.11

# Schemas as actually written by the current sensor simulators -- checked
# explicitly rather than assumed, so a future change to either sensor's
# output columns fails loudly here instead of silently producing a
# malformed assembled table.
CAMERA_REQUIRED_COLUMNS = [
    "timestamp", "approach_edge", "camera_range_m",
    "visible_vehicle_count", "visible_mean_speed_mps",
    "visible_queue_count", "visible_queue_length_m",
    "queue_reaches_camera_edge",
]
GPS_REQUIRED_COLUMNS = [
    "timestamp", "approach_edge", "probe_count",
    "probe_mean_speed_mps",
    "probe_min_distance_to_stopline_m", "probe_max_distance_to_stopline_m",
]

# Ground-truth-shaped names that must never appear in an assembled sensor
# table -- regression guard, mirrors the same list used in
# camera_simulator.py / gps_simulator.py.
FORBIDDEN_GROUND_TRUTH_COLUMNS = {
    "queue_length_m", "queue_count", "queue_beyond_camera",
    "density_veh_per_km", "flow_veh_per_hour", "vehicle_count",
    "mean_speed_mps", "true_queue_length_m", "true_density", "true_flow",
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
# Validation -- BEFORE assembly (inputs) and AFTER (output)
# ============================================================================

def validate_sensor_files_exist(scenario_dir: Path, gps_tag: str) -> "tuple[Path, Path]":
    camera_path = scenario_dir / "observations" / "camera_timeseries.csv"
    gps_path = scenario_dir / "observations" / f"gps_{gps_tag}_timeseries.csv"

    if not camera_path.exists():
        raise FileNotFoundError(
            f"Missing {camera_path} -- run sensors/camera_simulator.py "
            f"--scenario {scenario_dir.name} first."
        )
    if not gps_path.exists():
        raise FileNotFoundError(
            f"Missing {gps_path} -- run sensors/gps_simulator.py "
            f"--scenario {scenario_dir.name} --penetration <rate> first."
        )
    return camera_path, gps_path


def validate_source_schema(df: pd.DataFrame, required_columns: List[str], source_name: str) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{source_name} is missing required column(s): {missing}")


def validate_no_duplicate_keys(df: pd.DataFrame, source_name: str) -> None:
    dupes = df[df.duplicated(subset=["timestamp", "approach_edge"], keep=False)]
    if not dupes.empty:
        raise ValueError(
            f"{source_name} contains {len(dupes)} row(s) with duplicate "
            f"(timestamp, approach_edge) keys -- cannot join unambiguously."
        )


def validate_expected_edges(df: pd.DataFrame, approach_edges: List[str], source_name: str) -> None:
    observed = set(df["approach_edge"].unique())
    expected = set(approach_edges)
    if observed != expected:
        raise ValueError(
            f"{source_name} approach edges {observed} do not match expected {expected}."
        )


def validate_assembled_observations(
    merged: pd.DataFrame,
    approach_edges: List[str],
    sim_begin: int,
    sim_end: int,
) -> None:
    """Post-join checks on the assembled table, run BEFORE probe_count's
    NaN->0 fill (see process_scenario()) -- so a NaN here still means
    exactly one thing: this (timestamp, approach_edge) key was present in
    one source file but not the other, i.e. the camera and GPS grids do
    not actually align. Once this passes, probe_count's NaNs are known to
    be genuine "GPS observed zero probes" rows, not coverage gaps, and
    can be safely filled to 0."""
    if merged.empty:
        raise ValueError("Assembled observation table is empty -- nothing to write.")

    # Timestamps must fall within the scenario's primary observation window.
    out_of_window = merged[(merged["timestamp"] < sim_begin) | (merged["timestamp"] > sim_end)]
    if not out_of_window.empty:
        raise ValueError(
            f"Assembled observation contains {len(out_of_window)} row(s) with timestamps "
            f"outside [{sim_begin}, {sim_end}]."
        )

    # Timestamps must align with the sampling grid.
    misaligned = merged[(merged["timestamp"] - sim_begin) % SAMPLING_INTERVAL_S != 0]
    if not misaligned.empty:
        raise ValueError(
            f"Assembled observation contains {len(misaligned)} row(s) not aligned to "
            f"SAMPLING_INTERVAL_S={SAMPLING_INTERVAL_S}."
        )

    # Every expected approach edge must be represented.
    observed_edges = set(merged["approach_edge"].unique())
    expected_edges = set(approach_edges)
    if observed_edges != expected_edges:
        raise ValueError(
            f"Assembled observation approach edges {observed_edges} do not match "
            f"expected {expected_edges}."
        )

    # No duplicate keys after the join.
    dupes = merged[merged.duplicated(subset=["timestamp", "approach_edge"], keep=False)]
    if not dupes.empty:
        raise ValueError(
            f"Assembled observation contains {len(dupes)} row(s) with duplicate "
            f"(timestamp, approach_edge) keys after the join."
        )

    # Exact expected row count: every (timestamp, approach_edge) combination
    # must exist exactly once.
    sample_times = list(range(sim_begin, sim_end + 1, SAMPLING_INTERVAL_S))
    expected_row_count = len(sample_times) * len(approach_edges)
    if len(merged) != expected_row_count:
        raise ValueError(
            f"Assembled observation has {len(merged)} row(s), expected exactly "
            f"{expected_row_count} ({len(sample_times)} timestamps x "
            f"{len(approach_edges)} approach edges)."
        )

    # Camera/GPS alignment -- checked BEFORE any probe_count fill has
    # happened, so a NaN here is unambiguous evidence that the two source
    # files don't cover the same (timestamp, approach_edge) grid, not a
    # false positive caused by our own fillna().
    if merged["visible_vehicle_count"].isna().any():
        raise ValueError(
            "Assembled observation has row(s) with no camera data -- camera and GPS "
            "observation grids do not align."
        )
    if merged["probe_count"].isna().any():
        raise ValueError(
            "Assembled observation has row(s) with no GPS data (probe_count is NaN) "
            "-- camera and GPS observation grids do not align."
        )

    # No ground-truth-shaped column leaked in.
    leaked = FORBIDDEN_GROUND_TRUTH_COLUMNS.intersection(merged.columns)
    if leaked:
        raise ValueError(f"Assembled observation contains forbidden ground-truth-shaped column(s): {leaked}")

    # scenario_id must be present and single-valued.
    if "scenario_id" not in merged.columns or merged["scenario_id"].nunique() != 1:
        raise ValueError("Assembled observation must carry exactly one scenario_id value.")


# ============================================================================
# Assembly
# ============================================================================

def assemble_observations(camera_df: pd.DataFrame, gps_df: pd.DataFrame) -> pd.DataFrame:
    """Outer join on the shared observation key (timestamp, approach_edge).
    Never joins by row position/order -- pandas merge is key-based.

    validate="one_to_one" additionally guarantees, at merge time, that
    neither side has a duplicate (timestamp, approach_edge) key -- this is
    redundant with validate_no_duplicate_keys() above but kept as a second,
    independent guard directly on the join itself.

    Deliberately does NOT fill probe_count here. A NaN at this point means
    "GPS has no row for this (timestamp, approach_edge) at all" -- a grid
    misalignment between camera and GPS -- and must still be visible as
    NaN when validate_assembled_observations() runs, so that check can
    actually catch a coverage mismatch instead of a mismatch that's
    already been erased. The probe_count=0 fill (a real "GPS worked, no
    probe was present" observation) happens later in process_scenario(),
    only after validation has confirmed the two grids actually match."""
    merged = camera_df.merge(
        gps_df, on=["timestamp", "approach_edge"], how="outer", validate="one_to_one"
    )
    return merged.sort_values(["approach_edge", "timestamp"]).reset_index(drop=True)


# ============================================================================
# Orchestration
# ============================================================================

def process_scenario(scenario_dir: Path, cfg: dict, penetration_rate: float) -> pd.DataFrame:
    approach_edges = cfg["network"]["approaches"]
    scenario = load_scenario_metadata(scenario_dir)
    sim_begin = int(scenario["simulation_begin"])
    sim_end = int(scenario["simulation_end"])

    gps_tag = f"p{int(round(penetration_rate * 100)):02d}"
    camera_path, gps_path = validate_sensor_files_exist(scenario_dir, gps_tag)

    camera_df = pd.read_csv(camera_path)
    gps_df = pd.read_csv(gps_path)

    validate_source_schema(camera_df, CAMERA_REQUIRED_COLUMNS, "camera_timeseries.csv")
    validate_source_schema(gps_df, GPS_REQUIRED_COLUMNS, f"gps_{gps_tag}_timeseries.csv")
    validate_no_duplicate_keys(camera_df, "camera_timeseries.csv")
    validate_no_duplicate_keys(gps_df, f"gps_{gps_tag}_timeseries.csv")
    validate_expected_edges(camera_df, approach_edges, "camera_timeseries.csv")
    validate_expected_edges(gps_df, approach_edges, f"gps_{gps_tag}_timeseries.csv")

    # merge -> validate coverage -> fill probe_count -> save
    merged = assemble_observations(camera_df, gps_df)

    # Scenario-identifying metadata (not a derived feature) -- needed
    # downstream (e.g. to keep OOD scenarios out of training) without
    # re-reading scenario.json for every assembled row. Added before
    # validation since validate_assembled_observations() checks scenario_id.
    merged["scenario_id"] = scenario["scenario_id"]
    merged["split"] = scenario.get("split", "")
    merged["design_method"] = scenario.get("design_method", "")
    
        # Metadata only: records the GPS sensing condition used for this
    # observation dataset. It must NOT be passed to the ML model as a
    # predictive feature unless we intentionally design an experiment where
    # the model is allowed to know the GPS penetration rate.
    merged["gps_penetration_rate_requested"] = penetration_rate

    # Validated while probe_count NaNs are still NaN -- a coverage gap
    # between camera and GPS is caught here, before it could be masked by
    # the fill below.
    validate_assembled_observations(merged, approach_edges, sim_begin, sim_end)

    # Only now, with alignment confirmed, is probe_count=0 filled in --
    # a real "GPS worked, no probe was present" observation, distinct
    # from the grid-misalignment case just ruled out above. The other
    # probe_* columns are left as NaN when probe_count is 0: there is
    # genuinely no speed or distance to report for a probe that wasn't
    # there, and inventing a value would fabricate an observation.
    merged["probe_count"] = merged["probe_count"].fillna(0).astype(int)

    out_dir = scenario_dir / "observations"
    out_path = out_dir / f"assembled_observations_{gps_tag}.csv"
    merged.to_csv(out_path, index=False)

    both_signals = int((
        (merged["queue_reaches_camera_edge"] == True)  # noqa: E712
        & (merged["probe_count"] > 0)
    ).sum())

    print(f"{scenario['scenario_id']}: assembled observation ({gps_tag}) written to {out_path}")
    print(f"  rows: {len(merged)} | intervals where camera queue-at-edge + a probe both fired: {both_signals}")

    return merged


def find_scenarios() -> List[Path]:
    return sorted(p for p in SCENARIOS_DIR.glob("scenario_*") if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble camera + GPS sensor observations into one aligned table (no feature engineering)."
    )
    parser.add_argument("--scenario", type=str, default=None,
                         help="Run one scenario, e.g. scenario_normal_balanced. If omitted, runs all found scenarios.")
    default_pct_str = f"{DEFAULT_PENETRATION_RATE:.0%}".replace("%", "%%")
    parser.add_argument("--penetration", type=float, default=DEFAULT_PENETRATION_RATE,
                         help=f"Which GPS penetration run to assemble (must match a completed gps_simulator.py "
                              f"run). Default {default_pct_str} -- ASTRID's primary experiment.")
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
            process_scenario(scenario_dir, cfg, args.penetration)
        except Exception as exc:
            print(f"FAILED: {scenario_dir.name}: {exc}")
            failed.append(scenario_dir.name)

    print(f"\nDone. {len(scenario_dirs) - len(failed)}/{len(scenario_dirs)} succeeded.")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()