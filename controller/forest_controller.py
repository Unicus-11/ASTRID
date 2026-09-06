"""
forest_controller.py
====================
Deployment-facing ASTRID controller wrapping a trained scikit-learn
classifier (RandomForestClassifier / HistGradientBoostingClassifier /
etc, chosen by train_classifiers.py) behind EXACTLY the same call
interface as nn_controller.NNPolicy: `policy(state) -> str`. Drop-in
replacement for LoopConfig.policy_fn -- actions.py and
sumo_interface.py are unaware this exists, and this file has no NN
dependency.

Reuses NNPolicy's decision-interval / KEEP no-op / transition-phase
scheduling (duplicated, not imported -- same reasoning as elsewhere:
the deployment-facing scheduling logic should not gain a branch for
"which model kind is this").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib

from actions import ACTION_KEEP, ACTION_REQUEST_NEXT
from controller_state import ControllerState
from nn_features import build_nn_features
from signal_config import STAGE_INDICES

DECISION_INTERVAL_S = 5.0
INDEX_TO_ACTION = {0: ACTION_KEEP, 1: ACTION_REQUEST_NEXT}


@dataclass
class ForestPolicy:
    model: object  # any scikit-learn classifier exposing .predict()
    decision_interval_s: float = DECISION_INTERVAL_S
    _last_phase_seen: Optional[int] = field(default=None, init=False)
    _last_decision_bucket: int = field(default=-1, init=False)

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
        features = build_nn_features(state).reshape(1, -1)
        action_index = int(self.model.predict(features)[0])
        return INDEX_TO_ACTION[action_index]


def load_forest_policy(model_path: Path) -> ForestPolicy:
    return ForestPolicy(model=joblib.load(str(model_path)))