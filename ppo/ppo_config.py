"""
ppo/ppo_config.py  (PATCHED v2)
====================
Changes from previous version:

- ent_coef: 0.0 -> 0.01. With a 2-action space and a reward already
  dominated by one term (see waiting_scale_s note below), zero entropy
  bonus let the policy collapse toward one behavior early and then get
  destabilized when that behavior turned out suboptimal -- a plausible
  contributor to the mid-training dip you saw (reward improving to
  ~10-12k then degrading). A small entropy bonus keeps exploration
  alive longer without preventing convergence.

- waiting_scale_s: 200.0 -> 50.0. ppo_env.py's reward calc previously
  SUMMED waiting across 4 edges but AVERAGED queue across the same 4
  edges -- ppo_env.py v2 now averages both. That makes the raw
  avg_waiting_s magnitude ~4x smaller than before, so waiting_scale_s
  is reduced ~4x to keep the waiting reward TERM (post-division) in
  the same rough range it was calibrated for, rather than silently
  shrinking waiting's influence to near-zero relative to queue.

Central config for the ASTRID PPO signal-control experiment. Now that
controller/signal_config.py has been supplied, this file imports every
phase/timing/edge fact from it directly instead of guessing or
re-deriving them. signal_config.py's own docstring flags that
dataset/feature_builder.py's PHASE_GREEN_GROUP has the NS/EW labels
backwards relative to the real sq.net.xml program -- PPO's env now uses
signal_config.py's mapping (verified against the actual <tlLogic>), not
feature_builder.py's, for anything that touches real signal control.
The frozen HGB estimator internally still uses feature_builder.py's
(buggy-but-self-consistent) mapping for its own feature, which is
correct to leave alone -- it must match what the model was trained on,
not the "correct" geography.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMO_DIR = PROJECT_ROOT / "sumo"
SCENARIOS_DIR = SUMO_DIR / "generated_scenarios"
NETWORK_DIR = SUMO_DIR / "Squire_Junction_Multiple_Lanes"
NETWORK_FILE = NETWORK_DIR / "sq.net.xml"
CONTROLLER_DIR = PROJECT_ROOT / "controller"

if str(CONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_DIR))

import signal_config as sc  # noqa: E402

TLS_ID = sc.TLS_ID
APPROACH_EDGES: List[str] = list(sc.APPROACH_EDGES)
PHASE_SEQUENCE = sc.PHASE_SEQUENCE
STAGE_INDICES = sc.STAGE_INDICES
TRANSITION_AFTER_STAGE = sc.TRANSITION_AFTER_STAGE
PHASE_BY_INDEX = sc.PHASE_BY_INDEX
APPROACH_STAGE = sc.APPROACH_STAGE
STAGE_APPROACHES = sc.STAGE_APPROACHES

MIN_GREEN_S: Dict[int, float] = dict(sc.MIN_GREEN_S)
MAX_GREEN_S: Dict[int, float] = dict(sc.MAX_GREEN_S)
GLOBAL_MAX_GREEN_S = max(MAX_GREEN_S.values())

N_PHASES = len(PHASE_SEQUENCE)

SIMULATION_BEGIN_S = sc.SIMULATION_BEGIN_S
SIMULATION_END_S = sc.SIMULATION_END_S

SCENARIO_SUMOCFG_NAME = sc.SUMO_CONFIG_FILE


def scenario_sumocfg_path(scenario_id: str) -> Path:
    return SCENARIOS_DIR / scenario_id / SCENARIO_SUMOCFG_NAME


HGB_MODEL_PATH = (
    PROJECT_ROOT / "models" / "artifacts" / "layer2_p11"
    / "hist_gradient_boosting_layer2_p11_tuned" / "hist_gradient_boosting.joblib"
)
HGB_MANIFEST_PATH = PROJECT_ROOT / "dataset" / "assembled" / "layer2_p11" / "manifest.json"

SAMPLING_INTERVAL_S = 5
CONTROL_INTERVAL_S = 5

CAMERA_RANGE_M = 150.0
GPS_PENETRATION_RATE = 0.11

TRAIN_SCENARIOS = (
    "scenario_high_demand",
    "scenario_left_turn_heavy",
    "scenario_low_demand",
    "scenario_normal_balanced",
)

VALIDATION_SCENARIOS = (
    "scenario_north_heavy",
    "scenario_straight_heavy",
)

TEST_SCENARIOS = (
    "scenario_east_west_heavy",
    "scenario_south_heavy",
)

OOD_SCENARIOS = (
    "scenario_burst_demand_OOD",
    "scenario_heavy_vehicle_OOD",
    "scenario_north_extreme_OOD",
    "scenario_very_high_demand_OOD",
)

SCENARIO_SPLITS: Dict[str, List[str]] = {
    "train": [
        "scenario_high_demand",
        "scenario_left_turn_heavy",
        "scenario_low_demand",
        "scenario_normal_balanced",
    ],
    "validation": [
        "scenario_north_heavy",
        "scenario_straight_heavy",
    ],
    "test": [
        "scenario_east_west_heavy",
        "scenario_south_heavy",
    ],
    "ood": [
        "scenario_burst_demand_OOD",
        "scenario_heavy_vehicle_OOD",
        "scenario_north_extreme_OOD",
        "scenario_very_high_demand_OOD",
    ],
}

DEFAULT_WARMUP_SECONDS = 300
DEFAULT_EPISODE_SECONDS = SIMULATION_END_S


@dataclass
class RewardWeights:
    w_queue: float = 1.0
    w_waiting: float = 1.0
    w_speed: float = 0.5
    w_throughput: float = 0.5
    w_switch: float = 0.3

    queue_scale_m: float = 100.0
    # PATCH: 200.0 -> 50.0. See module docstring: ppo_env.py now
    # averages waiting across edges (was summed), so the raw magnitude
    # is ~4x smaller and the scale is reduced to match, keeping the
    # waiting reward term in the range it was originally calibrated for.
    waiting_scale_s: float = 50.0
    speed_scale_mps: float = 13.9


@dataclass
class PPOHyperparams:
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    # PATCH: 0.0 -> 0.01. Keeps exploration alive on this 2-action space
    # instead of letting the policy collapse early (see module docstring).
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: List[int] = field(default_factory=lambda: [128, 128])
    total_timesteps: int = 200_000
    seed: int = 42


@dataclass
class PPORunConfig:
    warmup_seconds: int = DEFAULT_WARMUP_SECONDS
    episode_seconds: int = DEFAULT_EPISODE_SECONDS
    control_interval_s: int = CONTROL_INTERVAL_S
    sumo_binary: str = "sumo"
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    hyperparams: PPOHyperparams = field(default_factory=PPOHyperparams)
    train_scenarios: List[str] = field(default_factory=lambda: SCENARIO_SPLITS["train"])
    validation_scenarios: List[str] = field(default_factory=lambda: SCENARIO_SPLITS["validation"])
    test_scenarios: List[str] = field(default_factory=lambda: SCENARIO_SPLITS["test"])
    ood_scenarios: List[str] = field(default_factory=lambda: SCENARIO_SPLITS["ood"])
    model_out_dir: Path = PROJECT_ROOT / "ppo" / "ppo_models"