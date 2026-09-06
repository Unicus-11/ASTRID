"""
online_traffic_observer.py
============================
ASTRID Prototype -- ONLINE (live TraCI) camera + GPS sensor emulation.

This is the online counterpart to sensors/camera_simulator.py and
sensors/gps_simulator.py. Those two scripts operate on a completed
scenario's raw_output/vehicle_trajectories.csv; this module reproduces
the SAME observation semantics from live TraCI calls, one simulation
step at a time, with NO access to future timestamps and NO ground-truth
queue/halting calls.

WHAT IS EXACTLY REPRODUCED FROM THE OFFLINE PIPELINE
-----------------------------------------------------
- camera_simulator.build_camera_observation() samples a SINGLE INSTANT
  at each 5s grid point (`snapshot = visible[visible["timestamp"]==t]`),
  not a windowed average. A live TraCI query at that same instant
  reproduces this exactly.
- distance_to_stopline_m: run_scenarios.py itself computes this live as
  `lane_length - lane_position` via TraCI. Reproduced verbatim here.
- is_queued / low_speed_streak_s: trajectory_utils.flag_queued's rule
  (>= QUEUE_MIN_DURATION_S consecutive seconds at/below
  QUEUE_SPEED_THRESHOLD_MPS, while on an approach edge) is reproduced
  via a per-vehicle streak counter updated every simulation second --
  matching run_scenarios.py's 1s recording cadence.

WHAT IS NOT EXACTLY REPRODUCIBLE, AND WHY (disclosed, not hidden)
-------------------------------------------------------------------
gps_simulator.select_probe_vehicles() selects an EXACT TOP-K of
vehicles, ranked by a deterministic hash, where K is computed from the
COMPLETE scenario vehicle population (every vehicle_id that will ever
appear in the ENTIRE scenario run). Online, at any point during a live
run, the future vehicle population is unknown -- using it would violate
the no-future-information rule this whole project enforces elsewhere.

Substitute used here: the SAME deterministic hash function and seed
offset (imported from gps_simulator, not reimplemented), but applied as
a PER-VEHICLE THRESHOLD rule instead of an exact top-K cut:

    is_probe(vehicle_id) := _probe_score(vehicle_id, seed) < penetration_rate

Since _probe_score(...) is uniform on [0, 1), this converges to the
same realized penetration rate in expectation, but will NOT select the
identical probe SET the offline training data used. This is a
deliberate, minimal substitution -- not a silent workaround -- because
exact reproduction is impossible without future information.

GROUND-TRUTH PROHIBITION
-------------------------
This module NEVER calls:
    lane.getLastStepHaltingNumber()
    vehicle.getIDList() as a queue count
    any TraCI API that reports a pre-aggregated true queue/halting state
Every output field is built by iterating individual vehicles and
applying the SAME sensor-limiting rules (camera range, GPS penetration)
the offline pipeline uses.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SENSORS_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
for p in (str(SENSORS_DIR), str(DATASET_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from trajectory_utils import (  # noqa: E402
    QUEUE_SPEED_THRESHOLD_MPS,
    QUEUE_MIN_DURATION_S,
    SAMPLING_INTERVAL_S,
)
from gps_simulator import _probe_score, GPS_SEED_OFFSET  # noqa: E402  (reused, not duplicated)


class OnlineSensorObserver:
    """Maintains per-vehicle queue-streak state at 1 Hz, and produces a
    single-instant camera+GPS aggregate snapshot per approach edge at
    each SAMPLING_INTERVAL_S-aligned tick.

    Parameters mirror scenario_config.json / scenario.json fields that
    camera_simulator.py / gps_simulator.py read offline -- NOT invented
    here, just supplied by the caller since this module has no file
    access of its own.
    """

    def __init__(
        self,
        traci_module,
        approach_edges: Iterable[str],
        camera_range_m: float,
        gps_penetration_rate: float,
        scenario_seed: int,
    ) -> None:
        if camera_range_m is None or not (camera_range_m > 0):
            raise ValueError(f"camera_range_m must be positive, got {camera_range_m}")
        if not (0.0 < gps_penetration_rate <= 1.0):
            raise ValueError(f"gps_penetration_rate must be in (0,1], got {gps_penetration_rate}")

        self._traci = traci_module
        self._approach_edges = list(approach_edges)
        self._camera_range_m = float(camera_range_m)
        self._gps_penetration_rate = float(gps_penetration_rate)
        self._gps_seed = int(scenario_seed) + GPS_SEED_OFFSET

        # vehicle_id -> consecutive low-speed seconds while on an approach edge
        self._low_speed_streak: Dict[str, int] = {}

    # -- probe membership -------------------------------------------------

    def _is_probe(self, vehicle_id: str) -> bool:
        """Per-vehicle threshold substitute for gps_simulator's exact
        top-K selection -- see module docstring for why."""
        return _probe_score(vehicle_id, self._gps_seed) < self._gps_penetration_rate

    # -- 1 Hz bookkeeping ---------------------------------------------------

    def update_per_second_state(self) -> None:
        """Call exactly once per simulation second (i.e. once per
        SumoInterface.step()). Updates the low-speed streak used by
        is_queued, mirroring trajectory_utils.flag_queued's run-length
        rule at the SAME 1 Hz resolution run_scenarios.py records at."""
        active_ids = set(self._traci.vehicle.getIDList())

        # Drop bookkeeping for vehicles that have left the simulation.
        for vid in list(self._low_speed_streak.keys()):
            if vid not in active_ids:
                del self._low_speed_streak[vid]

        for vehicle_id in active_ids:
            edge_id = self._traci.vehicle.getRoadID(vehicle_id)
            on_approach = edge_id in self._approach_edges
            speed = self._traci.vehicle.getSpeed(vehicle_id)

            is_slow_on_approach = on_approach and (speed <= QUEUE_SPEED_THRESHOLD_MPS)
            if is_slow_on_approach:
                self._low_speed_streak[vehicle_id] = self._low_speed_streak.get(vehicle_id, 0) + 1
            else:
                self._low_speed_streak[vehicle_id] = 0

    def _is_queued(self, vehicle_id: str) -> bool:
        return self._low_speed_streak.get(vehicle_id, 0) >= QUEUE_MIN_DURATION_S

    def _distance_to_stopline(self, vehicle_id: str, lane_id: str) -> Optional[float]:
        """Verbatim reproduction of run_scenarios.py's own live
        computation: lane_length - lane_position, clipped at 0."""
        if not lane_id:
            return None
        lane_length = self._traci.lane.getLength(lane_id)
        lane_position = self._traci.vehicle.getLanePosition(vehicle_id)
        return max(lane_length - lane_position, 0.0)

    # -- 5s-grid snapshot ---------------------------------------------------

    def sample_five_second_snapshot(self, current_time: float) -> Dict[str, dict]:
        """Call only when current_time lands on the shared
        SAMPLING_INTERVAL_S grid. Returns, per approach edge, the same
        fields camera_timeseries.csv / gps_p{TAG}_timeseries.csv carry --
        computed from a live snapshot, not history."""
        per_edge_visible: Dict[str, List[dict]] = {e: [] for e in self._approach_edges}
        per_edge_probes: Dict[str, List[dict]] = {e: [] for e in self._approach_edges}

        for vehicle_id in self._traci.vehicle.getIDList():
            edge_id = self._traci.vehicle.getRoadID(vehicle_id)
            if edge_id not in self._approach_edges:
                continue

            lane_id = self._traci.vehicle.getLaneID(vehicle_id)
            distance = self._distance_to_stopline(vehicle_id, lane_id)
            speed = self._traci.vehicle.getSpeed(vehicle_id)

            if distance is not None and 0.0 <= distance <= self._camera_range_m:
                per_edge_visible[edge_id].append({
                    "speed_mps": speed,
                    "distance_to_stopline_m": distance,
                    "is_queued": self._is_queued(vehicle_id),
                })

            if self._is_probe(vehicle_id) and distance is not None:
                per_edge_probes[edge_id].append({
                    "speed_mps": speed,
                    "distance_to_stopline_m": distance,
                })

        out: Dict[str, dict] = {}
        for edge in self._approach_edges:
            visible = per_edge_visible[edge]
            queued = [v for v in visible if v["is_queued"]]

            visible_count = len(visible)
            visible_mean_speed = (
                sum(v["speed_mps"] for v in visible) / visible_count if visible_count > 0 else 0.0
            )
            visible_queue_count = len(queued)
            visible_queue_length_m = (
                max(v["distance_to_stopline_m"] for v in queued) if visible_queue_count > 0 else 0.0
            )
            queue_reaches_camera_edge = (
                visible_queue_count > 0
                and visible_queue_length_m >= (self._camera_range_m - SAMPLING_INTERVAL_S)
            )

            probes = per_edge_probes[edge]
            probe_count = len(probes)
            probe_mean_speed = (
                sum(p["speed_mps"] for p in probes) / probe_count if probe_count > 0 else None
            )
            probe_min_dist = min((p["distance_to_stopline_m"] for p in probes), default=None)
            probe_max_dist = max((p["distance_to_stopline_m"] for p in probes), default=None)

            out[edge] = {
                "timestamp": current_time,
                "camera_range_m": self._camera_range_m,
                "visible_vehicle_count": visible_count,
                "visible_mean_speed_mps": round(visible_mean_speed, 4),
                "visible_queue_count": visible_queue_count,
                "visible_queue_length_m": round(visible_queue_length_m, 2),
                "queue_reaches_camera_edge": bool(queue_reaches_camera_edge),
                "probe_count": probe_count,
                "probe_mean_speed_mps": round(probe_mean_speed, 4) if probe_mean_speed is not None else None,
                "probe_min_distance_to_stopline_m": round(probe_min_dist, 2) if probe_min_dist is not None else None,
                "probe_max_distance_to_stopline_m": round(probe_max_dist, 2) if probe_max_dist is not None else None,
            }
        return out