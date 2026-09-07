"""
teacher_policy.py
====================
Wraps rule_teacher.rule_based_teacher with the SAME decision-interval /
KEEP no-op / transition-phase scheduling nn_controller.NNPolicy used
(duplicated here rather than imported -- this file has no NN
dependency by design), so the (features, label) pairs collected for
supervised training come from the same decision cadence the deployed
controller will actually be consulted at, not every simulation step.

Records one (features, label) pair per decision into self.dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from actions import ACTION_KEEP
from controller_state import ControllerState
from nn_features import build_nn_features
from rule_teacher import RuleTeacherConfig, rule_based_teacher
from signal_config import STAGE_INDICES

DECISION_INTERVAL_S = 5.0
ACTION_TO_INDEX = {"KEEP": 0, "REQUEST_NEXT": 1}


@dataclass
class TeacherDataCollector:
    teacher_config: RuleTeacherConfig = field(default_factory=RuleTeacherConfig)
    decision_interval_s: float = DECISION_INTERVAL_S

    dataset: List[Tuple[np.ndarray, int]] = field(default_factory=list, init=False)
    _last_phase_seen: Optional[int] = field(default=None, init=False)
    _last_decision_bucket: int = field(default=-1, init=False)

    def reset_episode(self) -> None:
        self._last_phase_seen = None
        self._last_decision_bucket = -1

    def __call__(self, state: ControllerState) -> str:
        if state.current_phase not in STAGE_INDICES:
            self._last_phase_seen = state.current_phase
            self._last_decision_bucket = -1
            return ACTION_KEEP

        if self._last_phase_seen != state.current_phase:
            self._last_phase_seen = state.current_phase
            self._last_decision_bucket = -1

        bucket = int(state.phase_elapsed_s // self.decision_interval_s)
        if bucket <= self._last_decision_bucket:
            return ACTION_KEEP

        self._last_decision_bucket = bucket
        action = rule_based_teacher(state, self.teacher_config)
        features = build_nn_features(state)
        self.dataset.append((features, ACTION_TO_INDEX[action]))
        return action