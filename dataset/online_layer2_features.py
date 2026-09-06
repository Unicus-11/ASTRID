"""
online_layer2_features.py
============================
ASTRID Prototype -- ONLINE Layer 2 feature construction.

Consumes one 5s-grid raw observation (from
sensors.online_traffic_observer.OnlineSensorObserver) per approach edge
and produces the SAME Layer 2 feature set dataset/feature_builder.py
computes offline, using ONLY past observations (no future timestamps).

PHASE-MAPPING NOTE (do not "fix" -- see chat diagnosis)
---------------------------------------------------------
PHASE_GREEN_GROUP / PHASE_IS_GREEN / GROUP_EDGES below are copied
VERBATIM from dataset/feature_builder.py, NOT imported from
signal_config.py. The selected HGB model was trained against
feature_builder.py's mapping; signal_config.py may describe the
opposite physical phase-to-direction assignment, but changing this
mapping here would silently shift the online feature distribution away
from what the model was trained on. That inconsistency is a real,
open issue -- flagged, not silently resolved.

MANIFEST-DRIVEN FEATURE ORDER
-------------------------------
This module never hardcodes the feature list. It loads
dataset/assembled/layer2_p11/manifest.json's feature_columns to
determine the order (and count) actually expected by the trained model.
The manifest.json content was not supplied in the conversation this
module was written from, so `_extract_feature_columns` makes a
best-effort, clearly-erroring attempt at a couple of plausible JSON
shapes rather than guessing silently -- if it raises, inspect the real
manifest and report back the top-level keys.
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from trajectory_utils import SAMPLING_INTERVAL_S  # noqa: E402
from feature_builder import FORBIDDEN_GROUND_TRUTH_COLUMNS  # noqa: E402 (reused, not duplicated)

DELTA_WINDOW_S = 30
_DELTA_STEPS = DELTA_WINDOW_S // SAMPLING_INTERVAL_S
if DELTA_WINDOW_S % SAMPLING_INTERVAL_S != 0:
    raise RuntimeError(
        f"DELTA_WINDOW_S={DELTA_WINDOW_S} is not a multiple of "
        f"SAMPLING_INTERVAL_S={SAMPLING_INTERVAL_S}; online history buffers below assume it is."
    )
_HISTORY_LEN = _DELTA_STEPS + 1  # current sample + N steps back

# Verbatim from dataset/feature_builder.py -- see module docstring.
PHASE_GREEN_GROUP = {0: "EW", 1: "EW", 2: "EW", 3: "EW", 4: "NS", 5: "NS", 6: "NS", 7: "NS"}
PHASE_IS_GREEN = {0: True, 1: False, 2: True, 3: False, 4: True, 5: False, 6: True, 7: False}
GROUP_EDGES = {"EW": ["1i", "2i"], "NS": ["3i", "4i"]}
EDGE_GROUP = {e: g for g, edges in GROUP_EDGES.items() for e in edges}

_HISTORY_COLUMNS = (
    "visible_queue_length_m",
    "visible_mean_speed_mps",
    "probe_count",
    "probe_max_distance_to_stopline_m",
)


def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or not (denominator > 1e-9):
        return None
    return numerator / denominator


def load_manifest_feature_columns(manifest_path: Path, split: str = "train") -> List[str]:
    """Loads the authoritative feature ordering from
    dataset/assembled/layer2_p11/manifest.json.

    Real manifest shape (confirmed from the actual file):
        {"splits": {"train": {"feature_columns": [...]}, "val": {...},
                     "test": {...}, "ood": {...}}}

    `split` selects which split's feature_columns to use as authoritative
    (default "train"). If other splits are present, their feature_columns
    are checked for exact agreement -- a mismatch across splits would
    mean the manifest itself is internally inconsistent about what the
    model was trained/evaluated on, which should stop this pipeline
    rather than silently pick one split's list.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    splits = manifest.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise ValueError(
            f"Expected a non-empty top-level 'splits' dict in {manifest_path}, "
            f"found top-level keys: {list(manifest.keys())}"
        )
    if split not in splits:
        raise ValueError(
            f"Split '{split}' not found in {manifest_path}'s splits. "
            f"Available splits: {list(splits.keys())}"
        )

    columns = splits[split].get("feature_columns")
    if not columns:
        raise ValueError(
            f"Split '{split}' in {manifest_path} has no non-empty 'feature_columns' list."
        )

    for other_split, entry in splits.items():
        other_cols = entry.get("feature_columns")
        if other_cols and other_cols != columns:
            raise ValueError(
                f"feature_columns mismatch between split '{split}' and split "
                f"'{other_split}' in {manifest_path} -- manifest is internally "
                f"inconsistent about the trained feature set. Refusing to guess "
                f"which one is authoritative."
            )

    forbidden = FORBIDDEN_GROUND_TRUTH_COLUMNS.intersection(columns)
    if forbidden:
        raise ValueError(
            f"Manifest at {manifest_path} declares forbidden ground-truth-shaped feature "
            f"column(s): {forbidden}. Refusing to build an online estimator against this manifest."
        )
    return list(columns)


class OnlineLayer2FeatureState:
    """Per-approach-edge history/state needed to reproduce
    feature_builder.py's Layer 2 features online, using only past
    observations."""

    def __init__(self, approach_edges: Iterable[str]) -> None:
        self._approach_edges = list(approach_edges)
        self._history: Dict[str, Dict[str, Deque[float]]] = {
            edge: {col: deque(maxlen=_HISTORY_LEN) for col in _HISTORY_COLUMNS}
            for edge in self._approach_edges
        }
        self._red_streak_start_time: Dict[str, Optional[float]] = {e: None for e in self._approach_edges}
        self._last_known_propagation_rate: Dict[str, Optional[float]] = {e: None for e in self._approach_edges}

    def _change_30s(self, edge: str, column: str, current_value: Optional[float]) -> Optional[float]:
        hist = self._history[edge][column]
        # hist currently holds up to _HISTORY_LEN-1 PAST values (this
        # tick's value is appended by the caller after this is read).
        if len(hist) < _DELTA_STEPS or current_value is None or hist[0] is None:
            return None
        value_30s_ago = hist[-_DELTA_STEPS] if len(hist) >= _DELTA_STEPS else None
        if value_30s_ago is None:
            return None
        return current_value - value_30s_ago

    def update_and_build(
        self,
        edge: str,
        raw_obs: dict,
        current_phase: int,
        phase_elapsed_s: float,
        current_time: float,
    ) -> Dict[str, object]:
        """Advances this edge's history by one 5s tick and returns the
        full Layer 2 feature dict (name -> value) for this tick."""
        hist = self._history[edge]

        visible_queue_length_m = float(raw_obs["visible_queue_length_m"])
        visible_mean_speed_mps = float(raw_obs["visible_mean_speed_mps"])
        probe_count = float(raw_obs["probe_count"])
        probe_max_dist = raw_obs["probe_max_distance_to_stopline_m"]
        probe_max_dist = float(probe_max_dist) if probe_max_dist is not None else None

        change_vq = self._change_30s(edge, "visible_queue_length_m", visible_queue_length_m)
        change_speed = self._change_30s(edge, "visible_mean_speed_mps", visible_mean_speed_mps)
        change_probe_count = self._change_30s(edge, "probe_count", probe_count)
        change_probe_max_dist = self._change_30s(edge, "probe_max_distance_to_stopline_m", probe_max_dist)

        # Push this tick's raw values into history AFTER reading deltas.
        hist["visible_queue_length_m"].append(visible_queue_length_m)
        hist["visible_mean_speed_mps"].append(visible_mean_speed_mps)
        hist["probe_count"].append(probe_count)
        hist["probe_max_distance_to_stopline_m"].append(probe_max_dist)

        camera_range_m = float(raw_obs["camera_range_m"])
        visible_occupancy_fraction = _safe_div(visible_queue_length_m, camera_range_m)
        if visible_occupancy_fraction is not None:
            visible_occupancy_fraction = min(max(visible_occupancy_fraction, 0.0), 1.0)

        # -- signal features (feature_builder.py's mapping, verbatim) --
        group = EDGE_GROUP.get(edge)
        is_green_for_approach = (
            (PHASE_GREEN_GROUP.get(current_phase) == group) and PHASE_IS_GREEN.get(current_phase, False)
        )

        is_red = is_green_for_approach is False
        if is_red:
            if self._red_streak_start_time[edge] is None:
                self._red_streak_start_time[edge] = current_time
            red_duration_s = current_time - self._red_streak_start_time[edge]
        else:
            self._red_streak_start_time[edge] = None
            red_duration_s = None

        # -- physics-derived --
        camera_range_km = camera_range_m / 1000.0
        estimated_density_k_veh_per_km = _safe_div(float(raw_obs["visible_vehicle_count"]), camera_range_km)
        speed_kmh = visible_mean_speed_mps * 3.6
        observed_flow_veh_per_hour = (
            estimated_density_k_veh_per_km * speed_kmh if estimated_density_k_veh_per_km is not None else None
        )

        propagation_rate = None
        if change_vq is not None and raw_obs["queue_reaches_camera_edge"] is False:
            propagation_rate = change_vq / DELTA_WINDOW_S
        if propagation_rate is not None:
            self._last_known_propagation_rate[edge] = propagation_rate

        hidden_extension = None
        if (
            raw_obs["queue_reaches_camera_edge"] is True
            and is_red
            and red_duration_s is not None
            and red_duration_s >= 0
            and self._last_known_propagation_rate[edge] is not None
        ):
            rate = max(self._last_known_propagation_rate[edge], 0.0)
            hidden_extension = rate * red_duration_s

        return {
            "camera_range_m": camera_range_m,
            "visible_vehicle_count": float(raw_obs["visible_vehicle_count"]),
            "visible_mean_speed_mps": visible_mean_speed_mps,
            "visible_queue_count": float(raw_obs["visible_queue_count"]),
            "visible_queue_length_m": visible_queue_length_m,
            "queue_reaches_camera_edge": 1.0 if raw_obs["queue_reaches_camera_edge"] else 0.0,
            "probe_count": probe_count,
            "probe_mean_speed_mps": raw_obs["probe_mean_speed_mps"],
            "probe_min_distance_to_stopline_m": raw_obs["probe_min_distance_to_stopline_m"],
            "probe_max_distance_to_stopline_m": probe_max_dist,
            "visible_queue_length_m_change_30s": change_vq,
            "visible_mean_speed_mps_change_30s": change_speed,
            "visible_occupancy_fraction": visible_occupancy_fraction,
            "probe_count_change_30s": change_probe_count,
            "probe_max_distance_to_stopline_m_change_30s": change_probe_max_dist,
            "current_phase": float(current_phase),
            "phase_elapsed_s": float(phase_elapsed_s),
            "is_green_for_approach": None if is_green_for_approach is None else (1.0 if is_green_for_approach else 0.0),
            "red_duration_s": red_duration_s,
            "estimated_density_k_veh_per_km": estimated_density_k_veh_per_km,
            "observed_flow_veh_per_hour": observed_flow_veh_per_hour,
            "estimated_queue_front_propagation_m_per_s": propagation_rate,
            "estimated_hidden_queue_extension_m": hidden_extension,
        }