"""
sumo_worker.py
====================
Step-by-step live TraCI worker. One process = one sumo-gui window =
one controller leg (Normal fixed-time OR ASTRID RF), driven entirely by
JSON-line commands on stdin, replying with one JSON line per event on
stdout.

This is comparison_2.py's run_and_log() turned "inside out": the exact
same per-step body (estimator call pattern, queue_vehicles sampling,
frame shape) is preserved verbatim, but instead of a single blocking
while-loop that only appends to a `frames` list, each iteration is
performed on demand in response to a "step" command, so an external
process (bridge_server.py) can pause/step/throttle it in real time.
policy_fn, the estimator, and build_scenario_sumocfg are imported and
called exactly as run_and_log() calls them -- nothing about them is
reimplemented differently. The only new code here is the
request/response loop.

Protocol (newline-delimited JSON on stdin/stdout):
  stdin  -> {"cmd": "step"}   : advance exactly one simulated second
  stdin  -> {"cmd": "quit"}   : close TraCI and exit
  stdout <- {"type": "ready", "role": ..., "sim_end_s": ...}
  stdout <- {"type": "frame", "frame": {...}}
  stdout <- {"type": "done"}
  stdout <- {"type": "error", "message": "..."}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in ("controller", "sensors", "dataset", "models", "models/results", "ppo"):
    p = str(REPO_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
THIS_DIR = str(Path(__file__).resolve().parent)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from astrid_controller import placeholder_policy  # noqa: E402
from forest_controller import ForestPolicy, load_forest_policy  # noqa: E402
from signal_config import APPROACH_EDGES, SIMULATION_END_S  # noqa: E402
from sumo_interface import LoopConfig, SumoInterface  # noqa: E402
from eval_common import RecordingQueueEstimator, build_estimator  # noqa: E402
from comparison_2 import build_scenario_sumocfg  # noqa: E402 (reused verbatim, not reimplemented)


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["normal", "astrid"], required=True)
    parser.add_argument("--scenario-dir", type=str, required=True)
    parser.add_argument("--sumo-config-json", type=str, required=True)
    parser.add_argument("--model-path", type=str,
                         default="models/artifacts/layer2_p11/hist_gradient_boosting_layer2_p11_tuned/hist_gradient_boosting.joblib")
    parser.add_argument("--manifest-path", type=str, default="dataset/assembled/layer2_p11/manifest.json")
    parser.add_argument("--penetration", type=float, default=0.11)
    parser.add_argument("--forest-model-path", type=str, default="controller/forest_models/best_model.joblib")
    parser.add_argument("--sumo-binary", type=str, default="sumo-gui")
    parser.add_argument("--vehicle-log-range-m", type=float, default=120.0)
    parser.add_argument("--vehicle-log-every-n-steps", type=int, default=1)
    args = parser.parse_args()

    scenario_dir = Path(args.scenario_dir)

    try:
        if args.role == "normal":
            policy_fn = placeholder_policy
        else:
            template = load_forest_policy(Path(args.forest_model_path))
            policy_fn = ForestPolicy(model=template.model)

        estimator = RecordingQueueEstimator(
            build_estimator(scenario_dir, Path(args.sumo_config_json), Path(args.model_path),
                             Path(args.manifest_path), args.penetration)
        )
        scenario_sumocfg = build_scenario_sumocfg(scenario_dir)
        interface = SumoInterface(LoopConfig(
            sumo_binary=args.sumo_binary,
            config_path=str(scenario_sumocfg),
            max_steps=None,
            queue_estimator=estimator,
            policy_fn=policy_fn,
            print_every_s=float("inf"),
        ))
        interface.start()
    except Exception as exc:  # noqa: BLE001
        emit({"type": "error", "message": f"startup failed: {exc}"})
        sys.exit(1)

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
                continue
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

    n_steps = 0
    last_queue_vehicles: list = []
    finished = False

    emit({"type": "ready", "role": args.role, "sim_end_s": SIMULATION_END_S})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            continue

        if cmd.get("cmd") == "quit":
            break

        if cmd.get("cmd") != "step":
            continue

        if finished:
            emit({"type": "done"})
            continue

        if interface.traci.simulation.getTime() >= SIMULATION_END_S:
            finished = True
            emit({"type": "done"})
            continue

        try:
            trace = interface.step()

            vids = interface.traci.vehicle.getIDList()
            speeds = [interface.traci.vehicle.getSpeed(v) for v in vids]
            waits = [interface.traci.vehicle.getAccumulatedWaitingTime(v) for v in vids]
            arrived_this_step = interface.traci.simulation.getArrivedNumber()

            if n_steps % args.vehicle_log_every_n_steps == 0:
                last_queue_vehicles = compute_queue_vehicles(vids)

            frame = {
                "t": trace.simulation_time,
                "phase": trace.sumo_current_phase,
                "action": trace.controller_action,
                "resolved": trace.resolved_action,
                "vehicles": trace.active_vehicle_count,
                "queues": {e: estimator.last_estimate.get(e) for e in APPROACH_EDGES},
                "mean_speed_mps": (sum(speeds) / len(speeds)) if speeds else 0.0,
                "mean_wait_s": (sum(waits) / len(waits)) if waits else 0.0,
                "arrived": arrived_this_step,
                "queue_vehicles": last_queue_vehicles,
            }
            n_steps += 1
            emit({"type": "frame", "frame": frame})
        except Exception as exc:  # noqa: BLE001
            emit({"type": "error", "message": f"step failed: {exc}"})
            break

    try:
        interface.close()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()