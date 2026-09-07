"""
test_online_hgb_pipeline.py
==============================
Tests per task §19. Uses a minimal fake TraCI stub -- no real SUMO
install required for these. The actual .joblib inference test is
skipped if the real artifact isn't present at the expected path (I was
never given its bytes, only its path).
"""

from __future__ import annotations

import json

from pathlib import Path

import pytest

from online_traffic_observer import OnlineSensorObserver
from online_layer2_features import OnlineLayer2FeatureState, load_manifest_feature_columns
from feature_builder import FORBIDDEN_GROUND_TRUTH_COLUMNS


APPROACH_EDGES = ["1i", "2i", "3i", "4i"]


class _FakeVehicle:
    def __init__(self, vid, edge_id, lane_id, lane_pos, lane_len, speed):
        self.vid, self.edge_id, self.lane_id = vid, edge_id, lane_id
        self.lane_pos, self.lane_len, self.speed = lane_pos, lane_len, speed


class FakeTraCI:
    """Minimal stand-in exposing only the calls this pipeline makes."""

    def __init__(self):
        self._vehicles = {}
        self._phase = 0
        self._spent_duration = 0.0

    def set_vehicles(self, vehicles):
        self._vehicles = {v.vid: v for v in vehicles}

    def set_phase(self, phase, spent_duration):
        self._phase, self._spent_duration = phase, spent_duration

    class vehicle:
        pass

    class lane:
        pass

    class trafficlight:
        pass


def make_fake_traci(vehicles, phase, spent_duration):
    fake = FakeTraCI()
    fake.set_vehicles(vehicles)
    fake.set_phase(phase, spent_duration)

    fake.vehicle.getIDList = lambda: list(fake._vehicles.keys())
    fake.vehicle.getRoadID = lambda vid: fake._vehicles[vid].edge_id
    fake.vehicle.getLaneID = lambda vid: fake._vehicles[vid].lane_id
    fake.vehicle.getLanePosition = lambda vid: fake._vehicles[vid].lane_pos
    fake.vehicle.getSpeed = lambda vid: fake._vehicles[vid].speed
    fake.lane.getLength = lambda lane_id: next(
        v.lane_len for v in fake._vehicles.values() if v.lane_id == lane_id
    )
    fake.trafficlight.getPhase = lambda tls_id: fake._phase
    fake.trafficlight.getSpentDuration = lambda tls_id: fake._spent_duration
    return fake


@pytest.fixture
def fake_manifest(tmp_path):
    columns = [
        "camera_range_m", "visible_vehicle_count", "visible_mean_speed_mps",
        "visible_queue_count", "visible_queue_length_m", "queue_reaches_camera_edge",
        "probe_count", "probe_mean_speed_mps", "probe_min_distance_to_stopline_m",
        "probe_max_distance_to_stopline_m", "visible_queue_length_m_change_30s",
        "visible_mean_speed_mps_change_30s", "visible_occupancy_fraction",
        "probe_count_change_30s", "probe_max_distance_to_stopline_m_change_30s",
        "current_phase", "phase_elapsed_s", "is_green_for_approach", "red_duration_s",
        "estimated_density_k_veh_per_km", "observed_flow_veh_per_hour",
        "estimated_queue_front_propagation_m_per_s", "estimated_hidden_queue_extension_m",
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "splits": {
            "train": {"feature_columns": columns},
            "val": {"feature_columns": columns},
        }
    }))
    return manifest_path, columns

def test_manifest_feature_columns_dimension_and_order(fake_manifest):
    manifest_path, expected_columns = fake_manifest
    loaded = load_manifest_feature_columns(manifest_path)
    assert loaded == expected_columns
    assert len(loaded) == 23


def test_manifest_rejects_ground_truth_columns(tmp_path):
    bad = tmp_path / "manifest_bad.json"
    bad.write_text(json.dumps({"feature_columns": ["visible_vehicle_count", "true_queue_length_m"]}))
    with pytest.raises(ValueError):
        load_manifest_feature_columns(bad)


def test_no_future_leakage_and_correct_30s_delta():
    observer = OnlineSensorObserver(
        traci_module=make_fake_traci([], 0, 0.0),
        approach_edges=APPROACH_EDGES, camera_range_m=150.0,
        gps_penetration_rate=0.11, scenario_seed=42,
    )
    state = OnlineLayer2FeatureState(APPROACH_EDGES)

    values = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0]  # t=0..30, step 5s
    results = []
    for i, val in enumerate(values):
        t = i * 5
        raw_obs = {
            "camera_range_m": 150.0, "visible_vehicle_count": 5,
            "visible_mean_speed_mps": 8.0, "visible_queue_count": 1,
            "visible_queue_length_m": val, "queue_reaches_camera_edge": False,
            "probe_count": 1, "probe_mean_speed_mps": 8.0,
            "probe_min_distance_to_stopline_m": 20.0, "probe_max_distance_to_stopline_m": 20.0,
        }
        feat = state.update_and_build("1i", raw_obs, current_phase=0, phase_elapsed_s=float(t), current_time=float(t))
        results.append(feat)

    # First 6 ticks (t=0..25) must have no 30s-delta value yet (no future leakage).
    for feat in results[:6]:
        assert feat["visible_queue_length_m_change_30s"] is None

    # At t=30, delta must be current(22.0) - value_at_t=0(10.0) = 12.0.
    assert results[6]["visible_queue_length_m_change_30s"] == pytest.approx(12.0)


def test_missing_gps_values_stay_none_not_zero():
    observer = OnlineSensorObserver(
        traci_module=make_fake_traci([], 0, 0.0),
        approach_edges=APPROACH_EDGES, camera_range_m=150.0,
        gps_penetration_rate=0.11, scenario_seed=1,
    )
    snap = observer.sample_five_second_snapshot(0.0)
    for edge in APPROACH_EDGES:
        assert snap[edge]["probe_count"] == 0
        assert snap[edge]["probe_mean_speed_mps"] is None
        assert snap[edge]["visible_mean_speed_mps"] == 0.0  # camera uses 0.0, not None


def test_ground_truth_columns_never_appear_in_feature_dict():
    state = OnlineLayer2FeatureState(APPROACH_EDGES)
    raw_obs = {
        "camera_range_m": 150.0, "visible_vehicle_count": 0, "visible_mean_speed_mps": 0.0,
        "visible_queue_count": 0, "visible_queue_length_m": 0.0, "queue_reaches_camera_edge": False,
        "probe_count": 0, "probe_mean_speed_mps": None,
        "probe_min_distance_to_stopline_m": None, "probe_max_distance_to_stopline_m": None,
    }
    feat = state.update_and_build("1i", raw_obs, current_phase=0, phase_elapsed_s=0.0, current_time=0.0)
    assert FORBIDDEN_GROUND_TRUTH_COLUMNS.isdisjoint(feat.keys())


@pytest.mark.skipif(
    not Path("models/artifacts/layer2_p11/hist_gradient_boosting_layer2_p11_tuned/hist_gradient_boosting.joblib").exists(),
    reason="Real HGB artifact not present in this environment.",
)
def test_hgb_artifact_loads_and_predicts():
    from persistence import load_model
    from online_layer2_features import load_manifest_feature_columns
    import pandas as pd

    model = load_model(Path(
        "models/artifacts/layer2_p11/hist_gradient_boosting_layer2_p11_tuned/hist_gradient_boosting.joblib"
    ))
    columns = load_manifest_feature_columns(
        Path("dataset/assembled/layer2_p11/manifest.json")
    )
    assert len(columns) == 23
    X = pd.DataFrame([[0.0] * len(columns)], columns=columns)
    preds = model.predict(X)
    assert len(preds) == 1
    assert isinstance(float(preds[0]), float)