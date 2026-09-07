"""
eval_common.py
====================
Shared, deterministic-episode evaluation helpers used by BOTH
train_rl.py (validation-based checkpoint selection) and eval_rl.py
(held-out comparison). One implementation, so "the metric used to pick
the best checkpoint" and "the metric used to report final results"
cannot silently drift apart.

RecordingQueueEstimator wraps an existing QueueEstimator (e.g.
OnlineHGBQueueEstimator, unmodified) and caches whatever it returns.
This is necessary because SumoInterface.step() calls
queue_estimator.estimate() itself exactly once per step -- calling
estimate() again from outside would double-invoke
OnlineHGBQueueEstimator's internal 1Hz bookkeeping and corrupt its
timing/streak state. Wrapping lets this module read the SAME value the
policy was handed that step, without a second call.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from controller_state import ControllerState, QueueEstimator
from reward import RewardConfig, total_estimated_queue_m
from signal_config import APPROACH_EDGES, SIMULATION_END_S, TLS_ID
from sumo_interface import LoopConfig, SumoInterface


class RecordingQueueEstimator:
    """QueueEstimator wrapper that caches the last estimate returned,
    for external (read-only) inspection after SumoInterface.step()."""

    def __init__(self, inner: QueueEstimator) -> None:
        self._inner = inner
        self.last_estimate: Dict[str, Optional[float]] = {e: None for e in APPROACH_EDGES}

    def estimate(self) -> Dict[str, Optional[float]]:
        self.last_estimate = self._inner.estimate()
        return self.last_estimate


def build_estimator(scenario_dir: Path, sumo_config_json: Path, model_path: Path,
                     manifest_path: Path, penetration: float):
    """Same construction as run_online_hgb_demo.py. `traci` is imported
    lazily so this module stays importable without SUMO installed."""
    import traci
    from online_hgb_queue_estimator import OnlineHGBQueueEstimator

    with open(scenario_dir / "scenario.json", "r", encoding="utf-8") as f:
        scenario = json.load(f)
    with open(sumo_config_json, "r", encoding="utf-8") as f:
        network_cfg = json.load(f)

    return OnlineHGBQueueEstimator(
        traci_module=traci,
        model_path=model_path,
        manifest_path=manifest_path,
        approach_edges=APPROACH_EDGES,
        tls_id=TLS_ID,
        camera_range_m=network_cfg["network"]["camera_range_m"],
        gps_penetration_rate=penetration,
        scenario_seed=int(scenario["seed"]),
        sim_begin_s=int(scenario["simulation_begin"]),
    )


@dataclass
class EpisodeMetrics:
    scenario: str
    n_steps: int
    mean_total_queue_m: float
    max_total_queue_m: float
    requested_transitions: int      # policy-chosen BEGIN_TRANSITION
    forced_transitions: int         # safety-cap FORCE_TRANSITION_MAX_GREEN
    total_transitions: int
    mean_cost: float                # RewardConfig-weighted per-step cost, for reference only


def run_evaluation_episode(
    policy_fn: Callable[[ControllerState], str],
    scenario_dir: Path,
    sumo_config_json: Path,
    model_path: Path,
    manifest_path: Path,
    penetration: float,
    sumo_binary: str,
    sumo_cfg_name: str,
    max_steps: Optional[int],
    reward_config: RewardConfig = RewardConfig(),
) -> EpisodeMetrics:
    """Runs exactly one episode against the real (unmodified)
    SumoInterface/LoopConfig with the given policy_fn, and reports
    traffic-control metrics -- not RL reward -- derived only from
    information the controller already legitimately has (HGB-estimated
    queue, and resolved_action, both already produced by the existing
    pipeline)."""
    estimator = RecordingQueueEstimator(
        build_estimator(scenario_dir, sumo_config_json, model_path, manifest_path, penetration)
    )
    interface = SumoInterface(LoopConfig(
        sumo_binary=sumo_binary,
        config_path=str(scenario_dir / sumo_cfg_name),
        max_steps=max_steps,
        queue_estimator=estimator,
        policy_fn=policy_fn,
        print_every_s=float("inf"),
    ))
    interface.start()
    queue_samples: List[float] = []
    cost_samples: List[float] = []
    requested = 0
    forced = 0
    n_steps = 0
    try:
        while True:
            if max_steps is not None and n_steps >= max_steps:
                break
            if interface.traci.simulation.getTime() >= SIMULATION_END_S:
                break
            trace = interface.step()
            q = total_estimated_queue_m(estimator.last_estimate)
            queue_samples.append(q)
            is_requested = trace.resolved_action == "BEGIN_TRANSITION"
            is_forced = trace.resolved_action == "FORCE_TRANSITION_MAX_GREEN"
            requested += int(is_requested)
            forced += int(is_forced)
            cost_samples.append(
                reward_config.w_queue * q
                + (reward_config.w_switch_requested if is_requested else 0.0)
                + (reward_config.w_switch_forced if is_forced else 0.0)
            )
            n_steps += 1
    finally:
        interface.close()

    return EpisodeMetrics(
        scenario=scenario_dir.name,
        n_steps=n_steps,
        mean_total_queue_m=(sum(queue_samples) / n_steps) if n_steps else 0.0,
        max_total_queue_m=max(queue_samples) if queue_samples else 0.0,
        requested_transitions=requested,
        forced_transitions=forced,
        total_transitions=requested + forced,
        mean_cost=(sum(cost_samples) / n_steps) if n_steps else 0.0,
    )