"""
observation_builder.py
====================
ASTRID Prototype -- Observation Builder

RESPONSIBILITY:
    Merge camera_timeseries.csv + gps_{tag}_timeseries.csv (already
    produced by camera_simulator.py and gps_simulator.py) into ONE
    observation table per approach per timestamp.

    This is still NOT feature engineering: no derived signals (no queue
    growth rate, no shockwave estimate, no history windows) are added
    here. It only places the two observation sources side by side, plus
    the signal-agnostic context (camera_range_m, penetration_rate) a
    downstream feature_builder.py will need. feature_builder.py is a
    separate, later module -- deliberately not built yet.

Reads (per scenario, must already exist -- run camera_simulator.py and
gps_simulator.py first):
    sumo/generated_scenarios/scenario_XXXX/observations/camera_timeseries.csv
    sumo/generated_scenarios/scenario_XXXX/observations/gps_p{TAG}_timeseries.csv

Writes:
    sumo/generated_scenarios/scenario_XXXX/observations/observation_p{TAG}.csv

Run:
    python sensors/observation_builder.py --scenario scenario_0001 --penetration 0.10
    python sensors/observation_builder.py --penetration 0.10        # all scenarios
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMO_DIR = PROJECT_ROOT / "sumo"
SCENARIOS_DIR = SUMO_DIR / "generated_scenarios"


def process_scenario(scenario_dir: Path, penetration_rate: float) -> pd.DataFrame:
    obs_dir = scenario_dir / "observations"
    tag = f"p{int(round(penetration_rate * 100)):02d}"

    camera_path = obs_dir / "camera_timeseries.csv"
    gps_path = obs_dir / f"gps_{tag}_timeseries.csv"

    if not camera_path.exists():
        raise FileNotFoundError(f"Missing {camera_path} -- run sensors/camera_simulator.py first.")
    if not gps_path.exists():
        raise FileNotFoundError(f"Missing {gps_path} -- run sensors/gps_simulator.py --penetration {penetration_rate} first.")

    camera_df = pd.read_csv(camera_path)
    gps_df = pd.read_csv(gps_path)

    merged = camera_df.merge(
        gps_df, on=["timestamp", "approach_edge"], how="outer", validate="one_to_one"
    )

    # Probe columns are NaN when no probe was present in that interval --
    # that's a real "zero probes observed", not missing data, so fill
    # explicitly rather than leaving it ambiguous for a later consumer.
    merged["probe_count"] = merged["probe_count"].fillna(0).astype(int)
    merged["gps_penetration_rate_requested"] = penetration_rate

    merged = merged.sort_values(["approach_edge", "timestamp"]).reset_index(drop=True)

    out_path = obs_dir / f"observation_{tag}.csv"
    merged.to_csv(out_path, index=False)

    intervals_with_probe_evidence_of_hidden_queue = int((
        (merged["queue_reaches_camera_edge"] == True)  # noqa: E712
        & (merged["probe_count"] > 0)
    ).sum())

    print(f"{scenario_dir.name}: observation ({tag}) written to {out_path}")
    print(f"  rows: {len(merged)} | intervals where camera queue-at-edge + a probe both fired: "
          f"{intervals_with_probe_evidence_of_hidden_queue}")

    return merged


def find_scenarios() -> List[Path]:
    return sorted(p for p in SCENARIOS_DIR.glob("scenario_*") if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge camera + GPS observations into one table.")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--penetration", type=float, default=0.10,
                         help="Which GPS penetration run to merge in (must match a completed gps_simulator.py run).")
    args = parser.parse_args()

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
            process_scenario(scenario_dir, args.penetration)
        except Exception as exc:
            print(f"FAILED: {scenario_dir.name}: {exc}")
            failed.append(scenario_dir.name)

    print(f"\nDone. {len(scenario_dirs) - len(failed)}/{len(scenario_dirs)} succeeded.")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()