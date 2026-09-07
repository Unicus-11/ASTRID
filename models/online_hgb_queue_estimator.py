"""
online_hgb_queue_estimator.py
================================
ASTRID Prototype -- ONLINE HGB inference bridge.

Implements controller_state.QueueEstimator using the already-selected,
already-trained Layer-2 p11 HistGradientBoosting model
(models/artifacts/layer2_p11/hist_gradient_boosting_layer2_p11_tuned/
hist_gradient_boosting.joblib), loaded INFERENCE-ONLY via
models/persistence.py. This module never calls .fit() and never
touches the .joblib file's contents.

Model output is true_queue_length_m during training; it is renamed
estimated_queue_length_m the instant it leaves this module (see
estimate() below) and that name is never changed back.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for sub in ("sensors", "dataset", "models"):
    p = str(PROJECT_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from persistence import load_model  # noqa: E402
from online_traffic_observer import OnlineSensorObserver  # noqa: E402
from online_layer2_features import (  # noqa: E402
    OnlineLayer2FeatureState,
    load_manifest_feature_columns,
)
from trajectory_utils import SAMPLING_INTERVAL_S  # noqa: E402
from feature_builder import FORBIDDEN_GROUND_TRUTH_COLUMNS  # noqa: E402


class OnlineHGBQueueEstimator:
    """controller_state.QueueEstimator implementation. `estimate()` is
    called once per SumoInterface.step() (i.e. once per simulation
    second); internally it updates 1 Hz bookkeeping every call but only
    recomputes the HGB prediction on SAMPLING_INTERVAL_S-aligned ticks,
    holding the last prediction in between."""

    def __init__(
        self,
        traci_module,
        model_path: Path,
        manifest_path: Path,
        approach_edges: Iterable[str],
        tls_id: str,
        camera_range_m: float,
        gps_penetration_rate: float,
        scenario_seed: int,
        sim_begin_s: int = 0,
    ) -> None:
        self._traci = traci_module
        self._tls_id = tls_id
        self._approach_edges = list(approach_edges)
        self._sim_begin_s = int(sim_begin_s)

        self._model = load_model(Path(model_path))  # inference-only; never .fit() here
        self._feature_columns = load_manifest_feature_columns(Path(manifest_path))

        self._observer = OnlineSensorObserver(
            traci_module=traci_module,
            approach_edges=self._approach_edges,
            camera_range_m=camera_range_m,
            gps_penetration_rate=gps_penetration_rate,
            scenario_seed=scenario_seed,
        )
        self._feature_state = OnlineLayer2FeatureState(self._approach_edges)

        self._last_estimate: Dict[str, Optional[float]] = {e: None for e in self._approach_edges}

    def estimate(self) -> Dict[str, Optional[float]]:
        self._observer.update_per_second_state()

        current_time = self._traci.simulation.getTime()
        aligned = (int(round(current_time)) - self._sim_begin_s) % SAMPLING_INTERVAL_S == 0
        if not aligned:
            return dict(self._last_estimate)

        current_phase = self._traci.trafficlight.getPhase(self._tls_id)
        phase_elapsed_s = self._traci.trafficlight.getSpentDuration(self._tls_id)
        raw_obs = self._observer.sample_five_second_snapshot(current_time)

        rows = []
        for edge in self._approach_edges:
            feat = self._feature_state.update_and_build(
                edge, raw_obs[edge], current_phase, phase_elapsed_s, current_time,
            )
            leaked = FORBIDDEN_GROUND_TRUTH_COLUMNS.intersection(feat.keys())
            if leaked:
                raise RuntimeError(
                    f"Ground-truth-shaped column(s) {leaked} present in an online feature row -- "
                    f"refusing to feed this to the HGB model."
                )
            missing = set(self._feature_columns) - set(feat.keys())
            if missing:
                raise RuntimeError(
                    f"Manifest expects feature(s) {missing} that the online feature builder does "
                    f"not produce -- feature_columns and online_layer2_features.py have drifted "
                    f"out of sync."
                )
            rows.append([feat[name] for name in self._feature_columns])

        X = pd.DataFrame(rows, columns=self._feature_columns).astype("float64")
        predictions = self._model.predict(X)  # true_queue_length_m predictions from HGB

        for edge, pred in zip(self._approach_edges, predictions):
         # Queue length has a physical lower bound of zero.
         estimated_queue_m = max(0.0, float(pred))

         # Relabeled immediately after model inference.
         self._last_estimate[edge] = estimated_queue_m

        return dict(self._last_estimate)