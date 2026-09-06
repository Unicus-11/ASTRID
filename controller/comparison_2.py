"""
comparison.py  (PATCHED -- adds PPO leg)
====================
Runs the SAME scenario through THREE controllers now:
  * "normal" -- astrid_controller.placeholder_policy (unchanged)
  * "astrid" -- forest_controller.ForestPolicy (unchanged)
  * "ppo"    -- the frozen 50k-timestep PPO checkpoint (NEW)

PPO IS NOT ROUTED THROUGH sumo_interface.py'S policy_fn MECHANISM.
--------------------------------------------------------------------
policy_fn (used by "normal"/"astrid") only receives a ControllerState
with 3 fields (estimated_queue_m, current_phase, phase_elapsed_s). PPO
was trained on a 25-dim observation (per-edge speed/count/occupancy,
time-since-switch, phase-relative min/max-green ratios) that
ControllerState cannot supply. Rather than enlarge ControllerState (and
risk that reimplementation silently drifting from what PPO actually
trained on -- an observation-integrity violation), the PPO leg drives
ppo/ppo_env.py's ASTRIDSignalEnv DIRECTLY: the exact same environment
class used for training/validation/evaluate_ppo.py. This guarantees the
observation PPO acts on here is identical in construction to training.

FRAME-RATE PARITY
------------------
sumo_interface.step() advances SUMO 1 second per call, so "normal" and
"astrid" log one frame per simulated second. PPO only re-decides every
control_interval_s (5s), but ASTRIDSignalEnv.step() still advances SUMO
internally 1 second at a time (unchanged) -- it just didn't use to
expose those intermediate ticks. It now accepts an `on_substep`
callback (ppo_env.py patch) that fires once per inner simulationStep(),
which this file uses to log one frame per simulated second for PPO too,
same as the other two legs. The controller's actual decision (and its
resolved KEEP/SWITCH/forced-switch outcome) only changes once every 5
such frames -- that's a real, correct property of how PPO was trained,
not a logging gap.

ESTIMATOR RE-INVOCATION HAZARD -- AVOIDED
------------------------------------------
The frozen HGB estimator's `estimate()` recomputes only on
SAMPLING_INTERVAL_S-aligned ticks (=5s, exactly == PPO's
control_interval_s) and holds its last value in between. This file
NEVER calls env._estimator.estimate() directly (that would double-fire
its internal per-second bookkeeping -- the same hazard
eval_common.RecordingQueueEstimator exists to avoid on the RF side).
Instead it reads `info["queue_estimates"]` (added in the ppo_env.py
patch) once per completed 5s control interval and backfills it onto
that interval's 5 logged frames -- which is exactly the real value in
effect for all of them, since the estimator itself doesn't change
mid-interval.

WARM-UP ASYMMETRY -- DOCUMENTED, NOT HIDDEN
---------------------------------------------
ASTRIDSignalEnv runs its normal `warmup_seconds` (default 300s, the
same value used throughout training/evaluate_ppo.py) before any PPO
frame is logged, matching the conditions PPO actually trained/was
selected under. "normal"/"astrid" have no such warm-up in
sumo_interface.py and log from t=0. This means the "ppo" entry's first
frame has t=300 while "normal"/"astrid" start at t=0 -- same scenario,
same demand, same duration of PPO's own reward-bearing control window,
but NOT the same absolute t=0 reference point. This was a deliberate
choice (forcing warmup_seconds=0 for PPO here would evaluate it under
colder HGB feature history than it was ever trained/selected with,
which is a worse kind of unfairness) -- flagged here and in the
per-scenario console output rather than silently picked either way.

SEED PARITY
-----------
PPO's own SUMO process is launched fresh by ASTRIDSignalEnv (it builds
its own temporary sumocfg), so its sumo_seed/estimator_seed are
explicitly set from scenario.json's own "seed" field -- the same
integer eval_common.build_estimator already uses for the RF leg's HGB
sensor-noise draw. This does not touch normal/astrid's own SUMO
process at all (unchanged).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in ("controller", "sensors", "dataset", "models", "models/results", "ppo"):
    p = str(REPO_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from astrid_controller import placeholder_policy  # noqa: E402 (unmodified)
from controller_state import ControllerState  # noqa: E402
from forest_controller import ForestPolicy, load_forest_policy  # noqa: E402
from signal_config import APPROACH_EDGES, SIMULATION_END_S  # noqa: E402
from sumo_interface import LoopConfig, SumoInterface  # noqa: E402
from eval_common import RecordingQueueEstimator, build_estimator, total_estimated_queue_m  # noqa: E402

# PPO leg imports -- only used if --ppo-model-path is supplied.
import ppo_config as ppo_cfg  # noqa: E402
from ppo_env import ASTRIDSignalEnv  # noqa: E402


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
                "arrived": arrived_this_step,
                "queue_vehicles": last_queue_vehicles,
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


def run_and_log_ppo(model, scenario_dir: Path, args) -> dict:
    """PPO leg. Drives ASTRIDSignalEnv directly (see module docstring
    for why this bypasses sumo_interface.py's policy_fn mechanism).
    `model` is a loaded stable_baselines3.PPO instance, already
    PPO.load()-ed by the caller (never trained/fine-tuned here)."""

    with open(scenario_dir / "scenario.json", "r", encoding="utf-8") as f:
        scenario = json.load(f)
    scenario_seed = int(scenario["seed"])

    run_cfg = ppo_cfg.PPORunConfig()
    run_cfg.sumo_binary = args.sumo_binary
    # Start real PPO control at t=0, matching astrid/normal, per explicit
    # request. NOTE: PPO's HGB queue estimator normally gets 300s of
    # rolling sensor history before it's used (see ppo_config.py /
    # ppo_env.py) -- with warmup=0 it acts on a cold estimator for
    # roughly the first 300s. Untested regime, not necessarily bad, just
    # different from how PPO was trained/validated. This override is
    # LOCAL to this evaluation script only -- train_ppo.py and
    # evaluate_ppo.py still use the real DEFAULT_WARMUP_SECONDS.
    run_cfg.warmup_seconds = 0

    env = ASTRIDSignalEnv(run_cfg, [scenario_dir.name], seed=0)
    obs, _ = env.reset(options={
        "scenario_id": scenario_dir.name,
        "sumo_seed": scenario_seed,
        "estimator_seed": scenario_seed,
    })

    frames: List[dict] = []
    total_arrived = 0
    lane_length_cache: dict = {}

    def lane_length(lane_id: str) -> float:
        if lane_id not in lane_length_cache:
            lane_length_cache[lane_id] = env._traci.lane.getLength(lane_id)
        return lane_length_cache[lane_id]

    def compute_queue_vehicles(vids) -> list:
        out = []
        for v in vids:
            lane_id = env._traci.vehicle.getLaneID(v)
            if "_" not in lane_id:
                continue
            edge_id, _, lane_idx_str = lane_id.rpartition("_")
            if edge_id not in ppo_cfg.APPROACH_EDGES:
                continue
            dist_to_stop = max(lane_length(lane_id) - env._traci.vehicle.getLanePosition(v), 0.0)
            if dist_to_stop > args.vehicle_log_range_m:
                continue
            out.append({
                "id": v,
                "type": env._traci.vehicle.getTypeID(v),
                "edge": edge_id,
                "lane": int(lane_idx_str),
                "dist_to_stop_m": round(dist_to_stop, 2),
            })
        return out

    # Mutable closure state for the on_substep hook below.
    substep_state = {"n": 0, "last_queue_vehicles": []}

    def on_substep(traci_conn) -> None:
        idx = substep_state["n"]
        substep_state["n"] += 1

        vids = traci_conn.vehicle.getIDList()
        speeds = [traci_conn.vehicle.getSpeed(v) for v in vids]
        waits = [traci_conn.vehicle.getAccumulatedWaitingTime(v) for v in vids]
        arrived_this_step = traci_conn.simulation.getArrivedNumber()

        nonlocal total_arrived
        total_arrived += arrived_this_step

        if idx % args.vehicle_log_every_n_steps == 0:
            substep_state["last_queue_vehicles"] = compute_queue_vehicles(vids)

        current_phase = traci_conn.trafficlight.getPhase(ppo_cfg.TLS_ID)
        # Live, phase-only label for non-decision ticks. The FIRST frame
        # of each control interval is overridden below (after step()
        # returns) with the actual decision outcome -- a switch that
        # happens exactly on this tick would otherwise be mislabeled as
        # a continuing "NONE"/"HOLD_STAGE" rather than the event itself.
        resolved = "HOLD_STAGE" if current_phase in ppo_cfg.STAGE_INDICES else "NONE"

        frames.append({
            "t": traci_conn.simulation.getTime(),
            "phase": current_phase,
            "action": None,   # backfilled below
            "resolved": resolved,
            "vehicles": len(vids),
            "queues": None,   # backfilled below from info["queue_estimates"]
            "mean_speed_mps": (sum(speeds) / len(speeds)) if speeds else 0.0,
            "mean_wait_s": (sum(waits) / len(waits)) if waits else 0.0,
            "arrived": arrived_this_step,
            "queue_vehicles": substep_state["last_queue_vehicles"],
        })

    prev_forced = 0
    done = False

    try:
        while not done:
            if args.max_steps is not None and len(frames) >= args.max_steps:
                break

            action, _ = model.predict(obs, deterministic=True)
            action_label = "SWITCH" if int(action) == 1 else "KEEP"

            frames_before = len(frames)
            obs, _, terminated, truncated, info = env.step(int(action), on_substep=on_substep)
            done = terminated or truncated

            new_frames = frames[frames_before:]
            if not new_frames:
                continue

            forced_now = info["forced_switch_count"]
            if forced_now > prev_forced:
                decision_resolved = "FORCE_TRANSITION_MAX_GREEN"
            elif info["switched"]:
                decision_resolved = "BEGIN_TRANSITION"
            else:
                decision_resolved = new_frames[0]["resolved"]  # already correct (HOLD_STAGE/NONE)
            prev_forced = forced_now

            new_frames[0]["resolved"] = decision_resolved
            for f in new_frames:
                f["action"] = action_label
                if f["queues"] is None:
                    f["queues"] = info["queue_estimates"]
    finally:
        env.close()

    n = len(frames) or 1
    duration_hours = ((frames[-1]["t"] - frames[0]["t"]) / 3600.0) if len(frames) > 1 else (run_cfg.episode_seconds / 3600.0)
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
    parser = argparse.ArgumentParser(description="Run normal vs ASTRID (vs PPO) controllers and log a dashboard-ready comparison.")
    parser.add_argument("--scenario-dirs", type=str, nargs="+", required=True)
    parser.add_argument("--sumo-config-json", type=str, required=True)
    parser.add_argument("--sumo-cfg-path", type=str, required=True,
                         help="Full path to sq.sumo.cfg (shared by every scenario in this junction).")
    parser.add_argument("--model-path", type=str,
                         default="models/artifacts/layer2_p11/hist_gradient_boosting_layer2_p11_tuned/hist_gradient_boosting.joblib")
    parser.add_argument("--manifest-path", type=str, default="dataset/assembled/layer2_p11/manifest.json")
    parser.add_argument("--penetration", type=float, default=0.11)
    parser.add_argument("--sumo-binary", type=str, default="sumo")
    parser.add_argument("--max-steps", type=int, default=None,
                         help="Max logged frames per leg (1 frame == 1 simulated second, for ALL three legs).")
    parser.add_argument("--forest-model-path", type=str, default="controller/forest_models/best_model.joblib")
    parser.add_argument("--ppo-model-path", type=str, default=None,
                         help="Path to the frozen PPO checkpoint (e.g. "
                              "ppo_models/ppo_astrid_v1/best_model/model.zip). "
                              "If omitted, the PPO leg is skipped entirely (normal/astrid only, unchanged).")
    parser.add_argument("--ppo-only", action="store_true",
                         help="Skip normal/astrid entirely (no forest model load, no re-run of scenarios "
                              "you've already generated) and only run+write the PPO leg. Requires "
                              "--ppo-model-path. Use this to add PPO results for scenarios whose "
                              "normal/astrid JSON already exists, without touching those files.")
    parser.add_argument("--output-dir", type=str, default="frontend/output")
    parser.add_argument("--vehicle-log-range-m", type=float, default=120.0,
                         help="Only vehicles within this distance of an approach's stop line are logged per-vehicle.")
    parser.add_argument("--vehicle-log-every-n-steps", type=int, default=1,
                         help="Throttle: recompute queue_vehicles every N steps (repeats the last list "
                              "in between) to shrink the output file on long runs. 1 = every step.")
    args = parser.parse_args()

    if args.ppo_only and not args.ppo_model_path:
        raise ValueError("--ppo-only requires --ppo-model-path.")

    ppo_model = None
    if args.ppo_model_path:
        from stable_baselines3 import PPO
        ppo_model = PPO.load(args.ppo_model_path)
        print(f"[ppo] loaded frozen checkpoint from {args.ppo_model_path} (inference-only, no .learn() call).")

    out_dir = Path(args.output_dir)

    # PPO always writes into its own subfolder -- never merged into the
    # normal/astrid combined JSON, and never overwrites files already
    # produced by a normal/astrid-only run.
    ppo_out_dir = out_dir / "ppo_model"
    if ppo_model is not None:
        ppo_out_dir.mkdir(parents=True, exist_ok=True)

    template = None
    if not args.ppo_only:
        template = load_forest_policy(Path(args.forest_model_path))
        out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    ppo_index = []

    for scenario_dir in [Path(s) for s in args.scenario_dirs]:
        summary = ""

        if not args.ppo_only:
            print(f"[{scenario_dir.name}] running normal (placeholder_policy)...")
            normal_result = run_and_log(placeholder_policy, scenario_dir, args)

            print(f"[{scenario_dir.name}] running astrid (random forest)...")
            astrid_policy = ForestPolicy(model=template.model)
            astrid_result = run_and_log(astrid_policy, scenario_dir, args)

            payload = {"scenario": scenario_dir.name, "normal": normal_result, "astrid": astrid_result}
            out_path = out_dir / f"{scenario_dir.name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

            summary = (
                f"normal avg_wait={normal_result['kpis']['avg_wait_s']:.1f}s, "
                f"astrid avg_wait={astrid_result['kpis']['avg_wait_s']:.1f}s"
            )
            print(f"[{scenario_dir.name}] wrote {out_path} ({summary})")
            index.append(scenario_dir.name)

        if ppo_model is not None:
            print(f"[{scenario_dir.name}] running ppo (frozen 50k checkpoint, warmup=0, "
                  f"decisions start at t=0)...")
            ppo_result = run_and_log_ppo(ppo_model, scenario_dir, args)
            ppo_payload = {"scenario": scenario_dir.name, "ppo": ppo_result}
            ppo_out_path = ppo_out_dir / f"{scenario_dir.name}.json"
            with open(ppo_out_path, "w", encoding="utf-8") as f:
                json.dump(ppo_payload, f)
            print(f"[{scenario_dir.name}] wrote {ppo_out_path} "
                  f"(ppo avg_wait={ppo_result['kpis']['avg_wait_s']:.1f}s)")
            ppo_index.append(scenario_dir.name)

    if not args.ppo_only:
        with open(out_dir / "index.json", "w", encoding="utf-8") as f:
            json.dump({"scenarios": index}, f)
        print(f"[done] wrote {out_dir / 'index.json'}")

    if ppo_model is not None:
        with open(ppo_out_dir / "index.json", "w", encoding="utf-8") as f:
            json.dump({"scenarios": ppo_index}, f)
        print(f"[done] wrote {ppo_out_dir / 'index.json'}")


if __name__ == "__main__":
    main()