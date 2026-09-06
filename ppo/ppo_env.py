"""
ppo/ppo_env.py  (PATCHED v2)
=================
Changes from v1 patch (seed determinism + collision tracking, kept):

3. REWARD FIX: waiting was SUMMED across the 4 approach edges while
   queue was AVERAGED across the same 4 edges. That's an unintentional
   ~4x weighting of waiting vs queue in the reward the policy actually
   optimizes, even though queue is the PRIMARY metric in
   train_ppo.py's composite_score() used for checkpoint selection.
   Policy and selection objective were quietly misaligned. Waiting is
   now averaged too, and the variable is genuinely "avg_waiting_s" (it
   was already misleadingly named that while holding a sum). Rescale
   ppo_config.RewardWeights.waiting_scale_s alongside this (see
   ppo_config.py patch) since the raw magnitude is now ~4x smaller.

4. OBSERVATION: added 2 phase-relative features so the policy doesn't
   have to infer "is this phase long enough yet" purely from a raw
   phase index plus a single GLOBAL_MAX_GREEN_S-normalized elapsed
   time. These use each phase's OWN min/max green
   (cfg.MIN_GREEN_S / cfg.MAX_GREEN_S), matching exactly what
   SignalSafetyController.apply_action() actually checks. OBS_DIM goes
   from 23 -> 25. Only meaningful for a fresh training run (not for
   resuming an old checkpoint -- the obs shape changed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import ppo_config as cfg
from ppo_controller import KEEP, SWITCH, SignalSafetyController


PROJECT_ROOT = cfg.PROJECT_ROOT

for import_dir in (
    PROJECT_ROOT / "controller",
    PROJECT_ROOT / "sensors",
    PROJECT_ROOT / "dataset",
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "models" / "results",
):
    import_dir = str(import_dir)
    if import_dir not in sys.path:
        sys.path.insert(0, import_dir)

from online_hgb_queue_estimator import OnlineHGBQueueEstimator  # noqa: E402

try:
    import traci
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "traci is not importable. Add SUMO's tools/ directory to "
        "PYTHONPATH (usually via SUMO_HOME/tools)."
    ) from exc


NETWORK_FILE = (
    cfg.SUMO_DIR
    / "Squire_Junction_Multiple_Lanes"
    / "sq.net.xml"
)

N_PHASES = len(cfg.PHASE_SEQUENCE)
FEATURES_PER_EDGE = 5

# PATCH: +2 for the new phase-relative features (was +3, now +5).
OBS_DIM = len(cfg.APPROACH_EDGES) * FEATURES_PER_EDGE + 5


class ASTRIDSignalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        run_cfg: cfg.PPORunConfig,
        scenario_ids: List[str],
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.run_cfg = run_cfg
        self.scenario_ids = list(scenario_ids)

        if not self.scenario_ids:
            raise ValueError("ASTRIDSignalEnv requires at least one scenario ID.")

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)

        self._estimator: Optional[OnlineHGBQueueEstimator] = None
        self._controller: Optional[SignalSafetyController] = None
        self._current_scenario: Optional[str] = None
        self._sumo_started = False
        self._traci_label: Optional[str] = None
        self._traci = None
        self._init_seed = seed

        self._episode_collisions = 0
        self._episode_teleports = 0

        super().reset(seed=seed)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._close_sumo()

        options = options or {}
        scenario_id = options.get("scenario_id")

        if scenario_id is None:
            index = int(self.np_random.integers(len(self.scenario_ids)))
            scenario_id = self.scenario_ids[index]

        if scenario_id not in self.scenario_ids:
            raise ValueError(
                f"Scenario '{scenario_id}' is not configured for this "
                f"environment.\nAvailable scenarios: {self.scenario_ids}"
            )

        self._current_scenario = scenario_id
        scenario_dir = cfg.SCENARIOS_DIR / scenario_id
        self._validate_scenario_directory(scenario_dir)
        sumocfg = self._build_sumo_config(scenario_dir)

        sumo_seed_override = options.get("sumo_seed")
        if sumo_seed_override is not None:
            sumo_seed = int(sumo_seed_override)
        else:
            sumo_seed = int(self.np_random.integers(0, 2_147_483_647))

        estimator_seed_override = options.get("estimator_seed")
        if estimator_seed_override is not None:
            estimator_seed = int(estimator_seed_override)
        else:
            estimator_seed = int(self.np_random.integers(0, 1_000_000))

        self._traci_label = f"ppo_env_{id(self)}"

        sumo_cmd = [
            self.run_cfg.sumo_binary,
            "-c", str(sumocfg),
            "--no-step-log", "true",
            "--start", "true",
            "--seed", str(sumo_seed),
            "--quit-on-end",
        ]

        try:
            traci.start(sumo_cmd, label=self._traci_label)
            self._traci = traci.getConnection(self._traci_label)
            self._sumo_started = True
        except Exception:
            self._traci = None
            self._sumo_started = False
            self._traci_label = None
            raise

        self._estimator = OnlineHGBQueueEstimator(
            traci_module=self._traci,
            model_path=cfg.HGB_MODEL_PATH,
            manifest_path=cfg.HGB_MANIFEST_PATH,
            approach_edges=cfg.APPROACH_EDGES,
            tls_id=cfg.TLS_ID,
            camera_range_m=cfg.CAMERA_RANGE_M,
            gps_penetration_rate=cfg.GPS_PENETRATION_RATE,
            scenario_seed=estimator_seed,
            sim_begin_s=0,
        )

        self._controller = SignalSafetyController(self._traci, cfg.TLS_ID)

        self._episode_collisions = 0
        self._episode_teleports = 0

        warmup_end = float(self.run_cfg.warmup_seconds)
        while self._traci.simulation.getTime() < warmup_end:
            self._traci.simulationStep()
            self._episode_collisions += self._traci.simulation.getCollidingVehiclesNumber()
            self._episode_teleports += self._traci.simulation.getStartingTeleportNumber()
            self._estimator.estimate()

        self._controller.reset(self._traci.simulation.getTime())

        obs = self._build_observation()
        info = {"scenario_id": scenario_id}
        return obs, info

    def step(self, action: int, on_substep=None):
        """
        PATCH: optional `on_substep` callback.

        PPO only re-decides every control_interval_s (5s), but SUMO is
        still advanced 1 second at a time internally (see the loop
        below -- unchanged). External callers (e.g. an evaluation/
        comparison harness that wants one logged frame per SUMO second,
        matching a 1s-granularity baseline controller) can pass
        `on_substep(traci_connection)` to be invoked once per inner
        simulationStep(), AFTER that step, with this env's own TraCI
        connection. This does not change training in any way: PPO
        itself never passes on_substep (train_ppo.py / evaluate_ppo.py
        call step(action) with no second argument), and the callback
        being None (default) skips this entirely.
        """
        if self._traci is None:
            raise RuntimeError("SUMO/TraCI is not running. Call reset() before step().")
        if self._controller is None:
            raise RuntimeError("Signal controller has not been initialized. Call reset() before step().")
        if self._estimator is None:
            raise RuntimeError("HGB estimator has not been initialized. Call reset() before step().")

        action = int(action)
        if action not in (KEEP, SWITCH):
            raise ValueError(f"Invalid PPO action {action}. Expected KEEP={KEEP} or SWITCH={SWITCH}.")

        current_time = self._traci.simulation.getTime()
        switched = self._controller.apply_action(action, current_time, self.run_cfg.control_interval_s)

        arrived_this_interval = 0
        collisions_this_interval = 0
        teleports_this_interval = 0
        speed_samples: List[float] = []

        target_time = self._traci.simulation.getTime() + self.run_cfg.control_interval_s

        while self._traci.simulation.getTime() < target_time:
            self._traci.simulationStep()

            arrived_this_interval += self._traci.simulation.getArrivedNumber()
            collisions_this_interval += self._traci.simulation.getCollidingVehiclesNumber()
            teleports_this_interval += self._traci.simulation.getStartingTeleportNumber()

            edge_speeds = [
                self._traci.edge.getLastStepMeanSpeed(edge) for edge in cfg.APPROACH_EDGES
            ]
            speed_samples.append(float(np.mean(edge_speeds)))

            if on_substep is not None:
                on_substep(self._traci)

        self._episode_collisions += collisions_this_interval
        self._episode_teleports += teleports_this_interval

        estimates = self._estimator.estimate()
        obs = self._build_observation(estimates)

        reward, info = self._compute_reward(
            estimates=estimates,
            arrived_this_interval=arrived_this_interval,
            speed_samples=speed_samples,
            switched=switched,
        )

        # PATCH: expose the exact per-edge dict already used for the
        # reward this step, so an external logger (e.g. a comparison
        # harness) never needs to call self._estimator.estimate() a
        # second time -- doing so would double-invoke its internal
        # per-second bookkeeping and corrupt its timing/streak state
        # (same hazard eval_common.RecordingQueueEstimator exists to
        # avoid on the RF/baseline side).
        info["queue_estimates"] = dict(estimates)
        info["collisions"] = collisions_this_interval
        info["teleports"] = teleports_this_interval
        info["cumulative_collisions"] = self._episode_collisions
        info["cumulative_teleports"] = self._episode_teleports

        current_time = self._traci.simulation.getTime()
        terminated = False
        truncated = current_time >= self.run_cfg.episode_seconds
        info["scenario_id"] = self._current_scenario

        return obs, reward, terminated, truncated, info

    def close(self):
        self._close_sumo()

    @staticmethod
    def _validate_scenario_directory(scenario_dir: Path) -> None:
        if not scenario_dir.exists():
            raise FileNotFoundError(f"Scenario directory does not exist:\n  {scenario_dir}")

        required_files = ("scenario.json", "flow.xml", "vtype.xml")
        missing = [name for name in required_files if not (scenario_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Scenario '{scenario_dir.name}' is missing:\n"
                + "\n".join(f"  {name}" for name in missing)
            )

        if not NETWORK_FILE.exists():
            raise FileNotFoundError(f"Shared SUMO network does not exist:\n  {NETWORK_FILE}")

    def _build_sumo_config(self, scenario_dir: Path) -> Path:
        flow_file = scenario_dir / "flow.xml"
        vtype_file = scenario_dir / "vtype.xml"
        raw_output_dir = scenario_dir / "raw_output"
        raw_output_dir.mkdir(parents=True, exist_ok=True)
        sumocfg = raw_output_dir / "ppo_episode.sumo.cfg"

        begin = float(getattr(cfg, "SIMULATION_BEGIN_S", 0))
        end = float(self.run_cfg.episode_seconds)

        network_path = NETWORK_FILE.resolve()
        flow_path = flow_file.resolve()
        vtype_path = vtype_file.resolve()

        config_text = f"""<?xml version="1.0" encoding="UTF-8"?>

<configuration>
    <input>
        <net-file value="{network_path}"/>
        <route-files value="{flow_path}"/>
        <additional-files value="{vtype_path}"/>
    </input>
    <time>
        <begin value="{begin}"/>
        <end value="{end}"/>
        <step-length value="1.0"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
</configuration>
"""
        sumocfg.write_text(config_text, encoding="utf-8")
        return sumocfg

    def _close_sumo(self) -> None:
        if not self._sumo_started:
            return

        connection = self._traci
        self._traci = None
        self._sumo_started = False
        self._traci_label = None

        if connection is None:
            return

        try:
            connection.close()
        except Exception:
            pass

    def _build_observation(
        self, estimates: Optional[Dict[str, Optional[float]]] = None,
    ) -> np.ndarray:
        if self._traci is None:
            raise RuntimeError("TraCI connection is not active.")

        if estimates is None:
            if self._estimator is not None:
                estimates = self._estimator.estimate()
            else:
                estimates = {edge: 0.0 for edge in cfg.APPROACH_EDGES}

        if self._controller is not None:
            phase = self._controller.current_phase()
            current_phase_elapsed = self._controller.current_phase_elapsed_s()
        else:
            phase = 0
            current_phase_elapsed = 0.0

        current_time = self._traci.simulation.getTime()

        if self._controller is not None:
            time_since_switch = self._controller.time_since_last_switch(current_time)
        else:
            time_since_switch = 0.0

        phase_def = cfg.PHASE_BY_INDEX.get(phase)
        if phase_def is not None and not phase_def.is_transition:
            current_stage = phase_def.stage
        else:
            current_stage = None

        values: List[float] = []

        for edge in cfg.APPROACH_EDGES:
            queue_estimate = estimates.get(edge)
            queue_value = 0.0 if queue_estimate is None else float(queue_estimate)
            values.append(queue_value)

            edge_stage = cfg.APPROACH_STAGE.get(edge)
            is_green = current_stage is not None and edge_stage == current_stage
            values.append(1.0 if is_green else 0.0)

            values.append(float(self._traci.edge.getLastStepVehicleNumber(edge)))
            values.append(float(self._traci.edge.getLastStepMeanSpeed(edge)))
            values.append(float(self._traci.edge.getLastStepOccupancy(edge)))

        # 6. Normalized phase index
        values.append(float(phase / max(N_PHASES - 1, 1)))

        # 7. Normalized current phase elapsed time (global scale, kept
        # for backward-style continuity with earlier runs)
        values.append(float(min(current_phase_elapsed / cfg.GLOBAL_MAX_GREEN_S, 1.0)))

        # 8. Normalized time since controller switch
        values.append(float(min(time_since_switch / cfg.GLOBAL_MAX_GREEN_S, 1.0)))

        # ------------------------------------------------------------------
        # PATCH: phase-relative features (9, 10).
        #
        # SignalSafetyController.apply_action() gates legality using THIS
        # phase's own MIN_GREEN_S / MAX_GREEN_S, not a global constant.
        # Give the policy that same ratio directly instead of making it
        # re-derive "is this phase long enough" from phase-index +
        # globally-normalized elapsed time.
        # ------------------------------------------------------------------
        if phase in cfg.STAGE_INDICES:
            phase_max_green = cfg.MAX_GREEN_S[phase]
            phase_min_green = cfg.MIN_GREEN_S[phase]
            # 9. How far through this phase's OWN max-green budget we are.
            progress_to_max = (
                min(current_phase_elapsed / phase_max_green, 1.0)
                if phase_max_green > 0 else 0.0
            )
            # 10. How far through this phase's OWN min-green requirement
            # we are (1.0 once a SWITCH request would actually be legal).
            progress_to_min = (
                min(current_phase_elapsed / phase_min_green, 1.0)
                if phase_min_green > 0 else 1.0
            )
        else:
            # In a mandatory yellow/transition phase: PPO's action is a
            # no-op here anyway (ppo_controller.apply_action returns
            # early), so both ratios are neutral placeholders.
            progress_to_max = 0.0
            progress_to_min = 1.0

        values.append(float(progress_to_max))
        values.append(float(progress_to_min))

        observation = np.asarray(values, dtype=np.float32)

        if observation.shape != self.observation_space.shape:
            raise RuntimeError(
                f"Generated observation has shape {observation.shape}, "
                f"but Gym space expects {self.observation_space.shape}."
            )

        return observation

    def _compute_reward(
        self,
        estimates: Dict[str, Optional[float]],
        arrived_this_interval: int,
        speed_samples: List[float],
        switched: bool,
    ) -> Tuple[float, dict]:
        if self._traci is None:
            raise RuntimeError("TraCI connection is not active.")

        w = self.run_cfg.reward_weights

        queue_values = [float(v) for v in estimates.values() if v is not None]
        avg_queue_m = float(np.mean(queue_values)) if queue_values else 0.0

        # ------------------------------------------------------------------
        # PATCH: was sum(...) across 4 edges while queue above is a mean
        # across the same 4 edges -- an unintentional ~4x over-weighting
        # of waiting vs queue in the reward. Now genuinely averaged, so
        # the variable name ("avg_waiting_s") matches what it holds, and
        # queue/waiting sit on a comparable footing before the
        # w_queue/w_waiting weights and scale constants are applied.
        # Rescale RewardWeights.waiting_scale_s accordingly (see
        # ppo_config.py patch note).
        # ------------------------------------------------------------------
        waiting_values = [
            self._traci.edge.getWaitingTime(edge) for edge in cfg.APPROACH_EDGES
        ]
        avg_waiting_s = float(np.mean(waiting_values)) if waiting_values else 0.0

        avg_speed_mps = float(np.mean(speed_samples)) if speed_samples else 0.0

        queue_component = w.w_queue * (-avg_queue_m / w.queue_scale_m)
        waiting_component = w.w_waiting * (-avg_waiting_s / w.waiting_scale_s)
        speed_component = w.w_speed * (avg_speed_mps / w.speed_scale_mps)
        throughput_component = w.w_throughput * (arrived_this_interval / self.run_cfg.control_interval_s)
        switch_component = -w.w_switch * (1.0 if switched else 0.0)

        reward = (
            queue_component + waiting_component + speed_component
            + throughput_component + switch_component
        )

        info = {
            "avg_queue_m": avg_queue_m,
            "avg_waiting_s": avg_waiting_s,
            "avg_speed_mps": avg_speed_mps,
            "arrived_this_interval": arrived_this_interval,
            "switched": switched,
            "switch_count": self._controller.switch_count if self._controller is not None else 0,
            "forced_switch_count": self._controller.forced_switch_count if self._controller is not None else 0,
        }

        return float(reward), info
