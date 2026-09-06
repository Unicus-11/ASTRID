"""
comparison.py
====================
Runs the SAME scenario twice through the real, unmodified SumoInterface
pipeline:
  * "normal"  -- astrid_controller.placeholder_policy (the fixed
                 controller already in this codebase)
  * "astrid"  -- forest_controller.ForestPolicy loaded from
                 --forest-model-path (the trained Random Forest
                 controller)

and writes one JSON log per scenario to frontend/output/, plus an
index.json listing every scenario written, for frontend/output/index.html
to fetch and replay.

SUMO is not browser-accessible, so the dashboard REPLAYS this saved log
(play/pause/step/speed controls) rather than streaming a live
simulation. Both controllers run on the SAME scenario/seed, so any
difference in the logged outcomes is attributable to the controller,
not scenario randomness.

Per-step vehicle speed / accumulated waiting time / arrived-count / type /
lane position is read directly via TraCI, purely for REPORTING -- read-only,
never fed back into either controller (both still only ever see
estimated_queue_m through the unmodified ControllerState/actions.py/
SumoInterface pipeline). Same read-only pattern as
eval_common.RecordingQueueEstimator.

PER-VEHICLE QUEUE LOGGING (queue_vehicles)
-------------------------------------------
Each frame also carries "queue_vehicles": a list of vehicles currently on
one of the four approach edges, within --vehicle-log-range-m of that
edge's stop line, each as:
    {"id": "10.0", "type": "car", "edge": "4i", "lane": 1, "dist_to_stop_m": 12.4}

"type" is read via traci.vehicle.getTypeID() -- the vType id string from
the scenario's own route file (this project's .rou.xml/.vtype.xml define
"bike", "car", "hgv", "bus"), NOT traci's generic vehicle CLASS taxonomy.
"edge" is one of APPROACH_EDGES; "lane" is the lane index parsed from the
laneID (0/1/2, matching signal_config.py's 3-lanes-per-approach layout).
"dist_to_stop_m" is lane length minus lane position, the same computation
online_traffic_observer.py's _distance_to_stopline already uses.

This list is scoped the same way this project's own sensor code scopes
"visible" vehicles -- only vehicles near the stop line, not the whole
active population -- both because that's what the dashboard actually
renders, and because logging every active vehicle every second would
make the JSON file impractically large. Lane lengths are cached per lane
(read once, reused) rather than re-queried every vehicle every step.
--vehicle-log-every-n-steps lets you throttle this (repeat the last
computed list between ticks) if file size becomes a problem on long runs;
default is every step (1) for full fidelity.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, List

REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in ("controller", "sensors", "dataset", "models", "models/results"):
    p = str(REPO_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from astrid_controller import placeholder_policy  # noqa: E402 (unmodified)
from controller_state import ControllerState  # noqa: E402
from forest_controller import ForestPolicy, load_forest_policy  # noqa: E402
from signal_config import APPROACH_EDGES, SIMULATION_END_S  # noqa: E402
from sumo_interface import LoopConfig, SumoInterface  # noqa: E402
from eval_common import RecordingQueueEstimator, build_estimator, total_estimated_queue_m  # noqa: E402


def run_and_log(policy_fn: Callable[[ControllerState], str], scenario_dir: Path, args) -> dict:
    estimator = RecordingQueueEstimator(
        build_estimator(scenario_dir, Path(args.sumo_config_json), Path(args.model_path),
                         Path(args.manifest_path), args.penetration)
    )
    interface = SumoInterface(LoopConfig(
        sumo_binary=args.sumo_binary,
        config_path=args.sumo_cfg_path,
        max_steps=args.max_steps,
        queue_estimator=estimator,
        policy_fn=policy_fn,
        print_every_s=float("inf"),
    ))
    interface.start()
    frames: List[dict] = []
    total_arrived = 0
    lane_length_cache: dict = {}

    def lane_length(lane_id: str) -> float:
        if lane_id not in lane_length_cache:
            lane_length_cache[lane_id] = interface.traci.lane.getLength(lane_id)
        return lane_length_cache[lane_id]

    def compute_queue_vehicles(vids) -> list:
        out = []
        for v in vids:
            lane_id = interface.traci.vehicle.getLaneID(v)
            if "_" not in lane_id:
                continue  # junction-internal lane, not one of the 4 approach edges
            edge_id, _, lane_idx_str = lane_id.rpartition("_")
            if edge_id not in APPROACH_EDGES:
                continue
            dist_to_stop = max(lane_length(lane_id) - interface.traci.vehicle.getLanePosition(v), 0.0)
            if dist_to_stop > args.vehicle_log_range_m:
                continue
            out.append({
                "id": v,
                "type": interface.traci.vehicle.getTypeID(v),
                "edge": edge_id,
                "lane": int(lane_idx_str),
                "dist_to_stop_m": round(dist_to_stop, 2),
            })
        return out

    try:
        n_steps = 0
        last_queue_vehicles: list = []
        while True:
            if args.max_steps is not None and n_steps >= args.max_steps:
                break
            if interface.traci.simulation.getTime() >= SIMULATION_END_S:
                break
            trace = interface.step()

            vids = interface.traci.vehicle.getIDList()
            speeds = [interface.traci.vehicle.getSpeed(v) for v in vids]
            waits = [interface.traci.vehicle.getAccumulatedWaitingTime(v) for v in vids]
            arrived_this_step = interface.traci.simulation.getArrivedNumber()
            total_arrived += arrived_this_step

            if n_steps % args.vehicle_log_every_n_steps == 0:
                last_queue_vehicles = compute_queue_vehicles(vids)

            frames.append({
                "t": trace.simulation_time,
                "phase": trace.sumo_current_phase,
                "action": trace.controller_action,
                "resolved": trace.resolved_action,
                "vehicles": trace.active_vehicle_count,
                "queues": {e: estimator.last_estimate.get(e) for e in APPROACH_EDGES},
                "mean_speed_mps": (sum(speeds) / len(speeds)) if speeds else 0.0,
                "mean_wait_s": (sum(waits) / len(waits)) if waits else 0.0,
                "arrived": arrived_this_step,  # per-step count, for a rolling throughput line in the dashboard
                "queue_vehicles": last_queue_vehicles,  # see module docstring
            })
            n_steps += 1
    finally:
        interface.close()

    n = len(frames) or 1
    duration_hours = ((frames[-1]["t"] - frames[0]["t"]) / 3600.0) if len(frames) > 1 else (SIMULATION_END_S / 3600.0)
    kpis = {
        "avg_wait_s": sum(f["mean_wait_s"] for f in frames) / n,
        "avg_speed_kmh": (sum(f["mean_speed_mps"] for f in frames) / n) * 3.6,
        "avg_queue_m": sum(total_estimated_queue_m(f["queues"]) for f in frames) / n,
        "max_queue_m": max((total_estimated_queue_m(f["queues"]) for f in frames), default=0.0),
        "throughput_veh_per_hr": (total_arrived / duration_hours) if duration_hours > 0 else 0.0,
        "requested_transitions": sum(1 for f in frames if f["resolved"] == "BEGIN_TRANSITION"),
        "forced_transitions": sum(1 for f in frames if f["resolved"] == "FORCE_TRANSITION_MAX_GREEN"),
    }
    return {"frames": frames, "kpis": kpis}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run normal vs ASTRID controllers and log a dashboard-ready comparison.")
    parser.add_argument("--scenario-dirs", type=str, nargs="+", required=True)
    parser.add_argument("--sumo-config-json", type=str, required=True)
    parser.add_argument("--sumo-cfg-path", type=str, required=True,
                         help="Full path to sq.sumo.cfg (shared by every scenario in this junction).")
    parser.add_argument("--model-path", type=str,
                         default="models/artifacts/layer2_p11/hist_gradient_boosting_layer2_p11_tuned/hist_gradient_boosting.joblib")
    parser.add_argument("--manifest-path", type=str, default="dataset/assembled/layer2_p11/manifest.json")
    parser.add_argument("--penetration", type=float, default=0.11)
    parser.add_argument("--sumo-binary", type=str, default="sumo")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--forest-model-path", type=str, default="controller/forest_models/best_model.joblib")
    parser.add_argument("--output-dir", type=str, default="frontend/output")
    parser.add_argument("--vehicle-log-range-m", type=float, default=120.0,
                         help="Only vehicles within this distance of an approach's stop line are logged per-vehicle.")
    parser.add_argument("--vehicle-log-every-n-steps", type=int, default=1,
                         help="Throttle: recompute queue_vehicles every N steps (repeats the last list "
                              "in between) to shrink the output file on long runs. 1 = every step.")
    args = parser.parse_args()

    template = load_forest_policy(Path(args.forest_model_path))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for scenario_dir in [Path(s) for s in args.scenario_dirs]:
        print(f"[{scenario_dir.name}] running normal (placeholder_policy)...")
        normal_result = run_and_log(placeholder_policy, scenario_dir, args)

        print(f"[{scenario_dir.name}] running astrid (random forest)...")
        astrid_policy = ForestPolicy(model=template.model)  # fresh scheduling state per run
        astrid_result = run_and_log(astrid_policy, scenario_dir, args)

        payload = {"scenario": scenario_dir.name, "normal": normal_result, "astrid": astrid_result}
        out_path = out_dir / f"{scenario_dir.name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        print(f"[{scenario_dir.name}] wrote {out_path} "
              f"(normal avg_wait={normal_result['kpis']['avg_wait_s']:.1f}s, "
              f"astrid avg_wait={astrid_result['kpis']['avg_wait_s']:.1f}s)")
        index.append(scenario_dir.name)

    with open(out_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump({"scenarios": index}, f)
    print(f"[done] wrote {out_dir / 'index.json'}")


if __name__ == "__main__":
    main()