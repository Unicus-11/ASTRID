"""
dataset_qa.py
==============
ASTRID Prototype -- Dataset QA / Audit tool.

RESPONSIBILITY:
    Audit the per-scenario feature and label CSVs produced by
    feature_builder.py, BEFORE dataset assembly. This module only reads
    and reports -- it never cleans, repairs, rewrites, fills, or otherwise
    mutates any generated file.

This is deliberately a read-only auditor. It imports constants and
project logic from the existing pipeline modules (trajectory_utils.py,
feature_builder.py) rather than re-deriving or guessing them, so QA stays
in sync with whatever those files actually define.

Source of truth used here (imported, not re-implemented):
    trajectory_utils.SAMPLING_INTERVAL_S
    feature_builder.DELTA_WINDOW_S
    feature_builder.FORBIDDEN_GROUND_TRUTH_COLUMNS
    feature_builder.CAMERA_REQUIRED_COLUMNS
    feature_builder.GPS_REQUIRED_COLUMNS
    feature_builder.SCENARIOS_DIR / PROJECT_ROOT

Layer structure audited (per feature_builder.py, v0.5):
    Layer 1 = camera-only observed columns + past-only history deltas
              (visible_queue_length_m_change_30s,
               visible_mean_speed_mps_change_30s) + visible_occupancy_fraction.
    Layer 2 = everything Layer 1 builds, PLUS GPS/probe columns, probe
              history deltas, signal-phase columns, and physics-derived
              columns (see feature_builder.add_physics_features /
              build_layer2_features).

Run:
    python dataset/dataset_qa.py
    python dataset/dataset_qa.py --scenario scenario_normal_balanced --layer layer2 --penetration 0.11
    python dataset/dataset_qa.py --strict
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ----------------------------------------------------------------------
# Import project constants / logic directly -- never re-declare them.
# ----------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trajectory_utils import SAMPLING_INTERVAL_S  # noqa: E402
import feature_builder as fb  # noqa: E402

SCENARIOS_DIR = fb.SCENARIOS_DIR
DELTA_WINDOW_S = fb.DELTA_WINDOW_S
FORBIDDEN_GROUND_TRUTH_COLUMNS = fb.FORBIDDEN_GROUND_TRUTH_COLUMNS
CAMERA_REQUIRED_COLUMNS = fb.CAMERA_REQUIRED_COLUMNS
GPS_REQUIRED_COLUMNS = fb.GPS_REQUIRED_COLUMNS

def _derive_layer_expected_columns() -> "tuple[set, set]":
    """Derive the Layer 1 / Layer 2 expected column sets from the
    AUTHORITATIVE definitions in feature_builder.py itself
    (build_feature_manifest()), rather than hand-maintaining a second,
    parallel list that can silently drift out of sync when
    feature_builder.py changes. build_feature_manifest() is called with
    has_tls=True so the TLS-conditional columns are still included in the
    "expected" set (their absence at audit time is then judged by
    check_layer_integrity as a WARNING, since tls_state.csv is documented
    as optional)."""
    l1_manifest = fb.build_feature_manifest("layer1", has_tls=True)
    l2_manifest = fb.build_feature_manifest("layer2", has_tls=True)

    def _feature_keys(manifest: dict) -> set:
        return {k for k in manifest.keys() if not k.startswith("_") and k not in ("layer", "sampling_interval_s")}

    l1_keys = _feature_keys(l1_manifest)
    l2_keys = _feature_keys(l2_manifest)
    return l1_keys, (l2_keys - l1_keys)


LAYER1_EXPECTED_COLUMNS, LAYER2_ADDITIONAL_COLUMNS = _derive_layer_expected_columns()

# --------------------------------------------------------------------------
# MANUALLY-ENCODED CONTRACT -- keep this section small and in one place.
# --------------------------------------------------------------------------
# Some project rules (which columns are allowed to be NaN and why, which
# columns must be non-negative) are not exposed as introspectable Python
# objects in feature_builder.py -- they exist only as behavior inside
# add_signal_features() / add_physics_features() / build_labels(), or as
# prose in build_feature_manifest()'s per-column "note" strings. Those
# cannot be imported, so they are necessarily re-stated here by hand.
# Where feature_builder.py DOES expose the underlying fact as data (the
# manifest's "note" field), _cross_check_na_reasons_against_manifest()
# below verifies this dict hasn't silently drifted from that text, so a
# future edit to feature_builder.py's manifest notes surfaces here as a
# WARNING instead of staying invisible. Each line below cites exactly
# which function in feature_builder.py it mirrors, so a future change to
# that function's actual NA/sign behavior is a one-place, visible diff:
#   - "*_change_{DELTA_WINDOW_S}s"                -> add_change_features()
#   - "current_phase" / "phase_elapsed_s" /
#     "is_green_for_approach" / "red_duration_s"  -> add_signal_features()
#   - "estimated_queue_front_propagation_m_per_s" /
#     "estimated_hidden_queue_extension_m"        -> add_physics_features()
#   - "true_queue_length_future_m"                -> build_labels()
EXPECTED_NA_COLUMNS = {
    f"visible_queue_length_m_change_{DELTA_WINDOW_S}s": "no history within first window (past-only delta)",
    f"visible_mean_speed_mps_change_{DELTA_WINDOW_S}s": "no history within first window (past-only delta)",
    f"probe_count_change_{DELTA_WINDOW_S}s": "no history within first window (past-only delta)",
    f"probe_max_distance_to_stopline_m_change_{DELTA_WINDOW_S}s": "no history within first window (past-only delta)",
    "current_phase": "left NA when raw_output/tls_state.csv is absent",
    "phase_elapsed_s": "left NA when raw_output/tls_state.csv is absent",
    "is_green_for_approach": "left NA when raw_output/tls_state.csv is absent",
    "red_duration_s": "left NA outside a red streak / when tls_state.csv is absent",
    "estimated_queue_front_propagation_m_per_s": "NA while queue_reaches_camera_edge is True (censored)",
    "estimated_hidden_queue_extension_m": "NA unless currently censored AND red AND a prior rate exists",
    "true_queue_length_future_m": "NA at the tail of the horizon shift (no future ground truth)",
}

# Columns that must never be negative, per their definitions in
# feature_builder.py / ground_truth.py. See the same citation note above --
# this is behavioral (not introspectable) and must be kept in sync by hand.
NONNEGATIVE_COLUMNS = [
    "visible_vehicle_count", "visible_queue_count", "visible_queue_length_m",
    "camera_range_m", "probe_count", "probe_min_distance_to_stopline_m",
    "probe_max_distance_to_stopline_m", "estimated_density_k_veh_per_km",
    "observed_flow_veh_per_hour", "phase_elapsed_s", "red_duration_s",
    "estimated_hidden_queue_extension_m", "true_queue_length_m",
    "true_queue_length_future_m",
]

# TASK 3: label-only nonnegative set. Deliberately excludes
# true_queue_beyond_camera (boolean -- feature_builder.build_labels()
# writes it straight from ground_truth's queue_beyond_camera flag, never
# as a numeric quantity). Everything else here is the SAME physical
# quantity as its NONNEGATIVE_COLUMNS entry above, just audited against
# labels_{layer}.csv instead of features_{layer}.csv.
LABEL_NONNEGATIVE_COLUMNS = ["true_queue_length_m", "true_queue_length_future_m"]


def _cross_check_na_reasons_against_manifest() -> List[str]:
    """Best-effort drift detector: for each EXPECTED_NA_COLUMNS entry,
    check whether feature_builder.build_feature_manifest()'s own per-
    column 'note' text is still consistent with the reason recorded here
    (loose substring match on key structural words, since the manifest
    note is free prose, not a machine contract). Returns a list of
    human-readable warnings; never raises, since this is advisory."""
    warnings: List[str] = []
    try:
        l2_manifest = fb.build_feature_manifest("layer2", has_tls=True)
    except Exception:
        return warnings  # manifest generation itself failing is reported elsewhere, not here

    keyword_map = {
        "history": ["history", "past"],
        "tls_state.csv is absent": ["tls", "signal controller"],
        "queue_reaches_camera_edge is True": ["censor", "queue_reaches_camera_edge"],
        "horizon shift": ["horizon", "shift"],
    }
    for col, reason in EXPECTED_NA_COLUMNS.items():
        entry = l2_manifest.get(col)
        if not isinstance(entry, dict):
            continue
        note = (entry.get("note") or "").lower()
        if not note:
            continue
        matched_any_keyword_group = False
        for phrase, keywords in keyword_map.items():
            if phrase in reason:
                matched_any_keyword_group = True
                if not any(kw in note for kw in keywords):
                    warnings.append(
                        f"EXPECTED_NA_COLUMNS['{col}'] reason ({reason!r}) no longer matches "
                        f"feature_builder's own manifest note for that column ({note!r}) -- "
                        f"possible drift, review manually."
                    )
        if not matched_any_keyword_group:
            continue
    return warnings



FEATURE_KEY_COLUMNS = ["timestamp", "approach_edge"]

# Timestamps in this project are simulation seconds and are expected to be
# integer-valued, but are read from CSV as floats -- comparisons against
# SAMPLING_INTERVAL_S / DELTA_WINDOW_S therefore use a small tolerance and
# a rounding convention rather than exact float equality, so a harmless
# float representation artifact (e.g. 29.999999999999996 instead of 30.0)
# is never misreported as a genuine gap or lookup miss.
TS_ABS_TOL = 1e-6
TS_ROUND_NDIGITS = 6


def _round_ts(value) -> float:
    """Canonical rounding for a timestamp value, used consistently
    wherever timestamps are placed in a set/index for equality lookups
    (e.g. 't - DELTA_WINDOW_S' membership tests), so float noise can't
    turn a real match into a spurious miss."""
    return round(float(value), TS_ROUND_NDIGITS)


def _isclose_ts(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=TS_ABS_TOL)


# ============================================================================
# Result plumbing
# ============================================================================

STATUS_ORDER = {"PASS": 0, "WARNING": 1, "FAIL": 2}


@dataclass
class CheckResult:
    check: str
    status: str  # PASS | WARNING | FAIL
    message: str


@dataclass
class ScenarioLayerReport:
    scenario_id: str
    layer: str
    tag: Optional[str]  # e.g. "p11" for layer2, None for layer1
    row_count: int = 0
    n_approaches: int = 0
    timestamp_min: Optional[float] = None
    timestamp_max: Optional[float] = None
    missing_summary: Dict[str, dict] = field(default_factory=dict)
    checks: List[CheckResult] = field(default_factory=list)

    def add(self, check: str, status: str, message: str) -> None:
        self.checks.append(CheckResult(check, status, message))

    @property
    def overall_status(self) -> str:
        if not self.checks:
            return "WARNING"
        return max((c.status for c in self.checks), key=lambda s: STATUS_ORDER[s])

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario_id,
            "layer": self.layer,
            "penetration_tag": self.tag,
            "row_count": self.row_count,
            "n_approaches": self.n_approaches,
            "timestamp_range": [self.timestamp_min, self.timestamp_max],
            "missing_summary": self.missing_summary,
            "overall_status": self.overall_status,
            "checks": [c.__dict__ for c in self.checks],
        }


# ============================================================================
# Discovery
# ============================================================================

def discover_scenarios(scenario_filter: Optional[str] = None) -> List[Path]:
    """Find scenario directories under SCENARIOS_DIR. Does not hard-code
    any specific scenario name; relies on the scenario_* naming
    convention already used throughout the project (see
    feature_builder.find_scenarios / ground_truth.find_scenarios)."""
    if not SCENARIOS_DIR.exists():
        return []
    if scenario_filter:
        candidate = SCENARIOS_DIR / scenario_filter
        return [candidate] if candidate.exists() else []
    return sorted(p for p in SCENARIOS_DIR.glob("scenario_*") if p.is_dir())


def discover_layer_targets(
    scenario_dir: Path, layer_filter: Optional[str], penetration: Optional[float]
) -> List[dict]:
    """For a scenario dir, find which features_{layer}[_{tag}].csv files
    actually exist, returning [{"layer": ..., "tag": ...}, ...].

    Layer 2 files are tagged p{NN} (feature_builder.process_scenario's
    output_tag convention); Layer 1 is untagged."""
    features_dir = scenario_dir / "features"
    targets: List[dict] = []
    if not features_dir.exists():
        return targets

    if layer_filter in (None, "layer1"):
        if (features_dir / "features_layer1.csv").exists():
            targets.append({"layer": "layer1", "tag": None})

    if layer_filter in (None, "layer2"):
        if penetration is not None:
            tag = f"p{int(round(penetration * 100)):02d}"
            if (features_dir / f"features_layer2_{tag}.csv").exists():
                targets.append({"layer": "layer2", "tag": tag})
        else:
            for p in sorted(features_dir.glob("features_layer2_p*.csv")):
                tag = p.stem.replace("features_layer2_", "")
                targets.append({"layer": "layer2", "tag": tag})

    return targets


# ============================================================================
# Loading
# ============================================================================

def _output_tag(layer: str, tag: Optional[str]) -> str:
    return f"{layer}_{tag}" if layer == "layer2" else layer


def load_data(scenario_dir: Path, layer: str, tag: Optional[str]) -> dict:
    """Safely load features/labels/manifest for one (scenario, layer, tag).
    Returns a dict with dataframes (or None) and any load-time errors --
    never raises, so a single bad scenario doesn't abort the whole audit."""
    out_tag = _output_tag(layer, tag)
    features_dir = scenario_dir / "features"
    result = {
        "features_df": None, "labels_df": None, "manifest": None,
        "load_errors": [],
        "features_path": features_dir / f"features_{out_tag}.csv",
        "labels_path": features_dir / f"labels_{out_tag}.csv",
        "manifest_path": features_dir / f"feature_manifest_{out_tag}.json",
    }

    if result["features_path"].exists():
        try:
            result["features_df"] = pd.read_csv(result["features_path"])
        except Exception as exc:
            result["load_errors"].append(f"Failed to read features CSV: {exc}")
    else:
        result["load_errors"].append(f"Missing features file: {result['features_path']}")

    if result["labels_path"].exists():
        try:
            result["labels_df"] = pd.read_csv(result["labels_path"])
        except Exception as exc:
            result["load_errors"].append(f"Failed to read labels CSV: {exc}")
    else:
        result["load_errors"].append(f"Missing labels file: {result['labels_path']}")

    if result["manifest_path"].exists():
        try:
            with open(result["manifest_path"], "r", encoding="utf-8") as f:
                result["manifest"] = json.load(f)
        except Exception as exc:
            result["load_errors"].append(f"Failed to read manifest JSON: {exc}")
    # Manifest is optional metadata -- its absence is not a load error.

    return result


# ============================================================================
# 1. Structural checks
# ============================================================================

def check_schema(df: pd.DataFrame, layer: str, report: ScenarioLayerReport) -> None:
    if df is None:
        report.add("schema", "FAIL", "Features dataframe is None -- cannot check schema.")
        return

    if df.empty:
        report.add("schema", "FAIL", "Features file is empty (0 rows).")
        return

    for col in FEATURE_KEY_COLUMNS:
        if col not in df.columns:
            report.add("schema", "FAIL", f"Missing required key column '{col}'.")

    index_like = [c for c in df.columns if c == "Unnamed: 0" or c.startswith("Unnamed:")]
    if index_like:
        report.add("schema", "FAIL", f"Accidental index column(s) detected: {index_like}")

    required = CAMERA_REQUIRED_COLUMNS if layer == "layer1" else (CAMERA_REQUIRED_COLUMNS + GPS_REQUIRED_COLUMNS)
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        report.add("schema", "FAIL", f"Missing required source column(s) for {layer}: {missing_required}")

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        # A completely empty column is only expected for the conditional
        # signal columns when tls_state.csv was genuinely absent for the
        # whole scenario -- otherwise it's suspicious.
        suspicious = [c for c in empty_cols if c not in EXPECTED_NA_COLUMNS]
        if suspicious:
            report.add("schema", "WARNING", f"Completely empty column(s): {suspicious}")
        expected_empty = [c for c in empty_cols if c in EXPECTED_NA_COLUMNS]
        if expected_empty:
            report.add("schema", "PASS", f"Completely empty but structurally expected column(s): {expected_empty}")

    if not empty_cols and not missing_required and not index_like:
        report.add("schema", "PASS", "Required columns present, no accidental index column.")


def check_keys(df: pd.DataFrame, report: ScenarioLayerReport, source: str = "features") -> None:
    """Duplicate-key check, reusable for both features_df and labels_df.
    'source' distinguishes which file the FAIL/PASS refers to in the
    report, since duplicate label keys (e.g. labels: A, B, B, C) can
    silently break a features/labels row-count match even when the
    feature/label KEY SETS are identical -- key-set equality alone does
    not imply row-count equality unless both sides are already known to
    be duplicate-free."""
    check_name = "keys" if source == "features" else f"keys_{source}"
    if df is None or df.empty:
        return
    if not all(c in df.columns for c in FEATURE_KEY_COLUMNS):
        report.add(check_name, "FAIL", f"Cannot check key uniqueness in {source} -- key columns missing.")
        return

    dupes = df[df.duplicated(subset=FEATURE_KEY_COLUMNS, keep=False)]
    if not dupes.empty:
        report.add(
            check_name, "FAIL",
            f"{len(dupes)} row(s) in {source} with duplicate (timestamp, approach_edge) keys.",
        )
    else:
        report.add(check_name, "PASS", f"No duplicate (timestamp, approach_edge) keys in {source}.")


def check_label_schema(labels_df: pd.DataFrame, report: ScenarioLayerReport) -> None:
    """Structural audit of labels_{layer}.csv, mirroring check_schema()'s
    treatment of features_df. Without this, a malformed labels file could
    pass QA entirely simply because check_schema() was only ever called
    on features_df. Required columns per feature_builder.build_labels():
    timestamp, approach_edge, true_queue_length_m, true_queue_beyond_camera
    always; true_queue_length_future_m + prediction_horizon_s only when a
    horizon was requested (detected here by column presence, not assumed)."""
    if labels_df is None:
        report.add("label_schema", "FAIL", "Labels dataframe is None -- cannot check schema.")
        return
    if labels_df.empty:
        report.add("label_schema", "FAIL", "Labels file is empty (0 rows).")
        return

    always_required = ["timestamp", "approach_edge", "true_queue_length_m", "true_queue_beyond_camera"]
    missing = [c for c in always_required if c not in labels_df.columns]
    if missing:
        report.add("label_schema", "FAIL", f"Labels file is missing required column(s): {missing}")

    index_like = [c for c in labels_df.columns if c == "Unnamed: 0" or c.startswith("Unnamed:")]
    if index_like:
        report.add("label_schema", "FAIL", f"Accidental index column(s) detected in labels: {index_like}")

    has_future = "true_queue_length_future_m" in labels_df.columns
    has_horizon = "prediction_horizon_s" in labels_df.columns
    if has_future != has_horizon:
        report.add(
            "label_schema", "FAIL",
            f"'true_queue_length_future_m' present={has_future} but 'prediction_horizon_s' present={has_horizon} "
            f"-- feature_builder.build_labels() always writes these together when horizon_s is set.",
        )

    if not missing and not index_like:
        report.add("label_schema", "PASS", "Labels file has required columns, no accidental index column.")


def check_label_missingness(labels_df: pd.DataFrame, report: ScenarioLayerReport) -> None:
    """Missingness audit for labels_df, mirroring check_missingness()'s
    treatment of features_df. Distinguishes the ONE expected-NA label
    column (true_queue_length_future_m, whose tail rows legitimately lack
    a future ground-truth observation, per EXPECTED_NA_COLUMNS) from
    unexpected missing values in the actual current-time targets, which
    would mean some rows have no usable label at all."""
    if labels_df is None or labels_df.empty:
        return

    n_rows = len(labels_df)
    for col in labels_df.columns:
        n_missing = int(labels_df[col].isna().sum())
        if n_missing == 0:
            continue
        frac = n_missing / n_rows
        reason = EXPECTED_NA_COLUMNS.get(col)
        report.missing_summary[f"label:{col}"] = {"n_missing": n_missing, "fraction": round(frac, 4), "expected_reason": reason}
        if col in ("true_queue_length_m", "true_queue_beyond_camera"):
            report.add("label_missingness", "FAIL", f"Current-time label column '{col}' has {n_missing} missing value(s) ({frac*100:.2f}%) -- these rows have no usable target.")
        elif reason is None:
            report.add("label_missingness", "WARNING", f"Label column '{col}' has {n_missing} missing value(s) ({frac*100:.2f}%) with no known structural reason.")
        else:
            report.add("label_missingness", "PASS", f"Label column '{col}' has {n_missing} missing value(s) ({frac*100:.2f}%), expected: {reason}.")

    if not any(k.startswith("label:") for k in report.missing_summary):
        report.add("label_missingness", "PASS", "No missing values detected in any label column.")


def check_timestamps(df: pd.DataFrame, report: ScenarioLayerReport, source: str = "features") -> None:
    """Ordering + CONTINUITY check. A gap that is merely a multiple of
    SAMPLING_INTERVAL_S is not sufficient -- e.g. 10 -> 20 passes a
    "% SAMPLING_INTERVAL_S == 0" test but silently skips the row at 15.
    On this project's shared, gap-free observation grid (see
    observation_assembler.validate_assembled_observations, which asserts
    an EXACT expected_row_count of len(sample_times) x len(approach_edges)
    with no gaps permitted), consecutive timestamps within an
    approach_edge are expected to differ by EXACTLY SAMPLING_INTERVAL_S.

    Each consecutive diff is classified explicitly rather than lumped
    into one bucket, so a duplicate timestamp (diff == 0, already caught
    separately by check_keys) is never misreported as a "gap":
        diff == 0            -> duplicate/overlap (reported by check_keys; not double-counted here)
        diff < 0              -> ordering problem
        diff == interval       -> correct
        diff > interval, on-grid -> missing sample(s)
        diff not a multiple    -> off-grid misalignment

    TASK 1: 'source' distinguishes feature timestamps ("timestamps") from
    label timestamps ("timestamps_labels") so both files' continuity is
    audited, without either overwriting the other's check name. This is
    unrelated to check_label_shift_consistency(), which verifies the
    FUTURE-LABEL SHIFT is internally correct -- not raw timestamp
    continuity within the labels file.

    TASK 4: diff classification now goes through _round_ts()/_isclose_ts()
    instead of exact float equality, so CSV float noise (e.g.
    29.999999999999996 vs 30.0) can never be misreported as an ordering
    problem, an off-grid step, or a missing sample.
    """
    check_name = "timestamps" if source == "features" else f"timestamps_{source}"
    if df is None or df.empty or "timestamp" not in df.columns:
        return

    ts_numeric = pd.to_numeric(df["timestamp"], errors="coerce")
    if ts_numeric.isna().any():
        n_bad = int(ts_numeric.isna().sum())
        report.add(check_name, "FAIL", f"{n_bad} row(s) have non-parseable timestamp values.")
        return

    if source == "features":
        report.timestamp_min = float(ts_numeric.min())
        report.timestamp_max = float(ts_numeric.max())

    ordering_issues = 0
    duplicate_ts = 0
    gap_violations = 0
    misaligned_step = 0
    if "approach_edge" in df.columns:
        for edge, group in df.assign(_ts=ts_numeric).groupby("approach_edge"):
            raw_order = group["_ts"].reset_index(drop=True)
            if not raw_order.is_monotonic_increasing:
                ordering_issues += 1

            ts_sorted = raw_order.sort_values().reset_index(drop=True)
            diffs = ts_sorted.diff().dropna()
            for d in diffs:
                d_r = _round_ts(d)
                if _isclose_ts(d_r, 0.0):
                    duplicate_ts += 1  # already flagged by check_keys; tracked, not re-reported as a gap
                elif d_r < 0 and not _isclose_ts(d_r, 0.0):
                    ordering_issues += 1
                elif _isclose_ts(d_r, SAMPLING_INTERVAL_S):
                    continue
                else:
                    remainder = math.fmod(d_r, SAMPLING_INTERVAL_S)
                    on_grid = _isclose_ts(remainder, 0.0) or _isclose_ts(remainder, SAMPLING_INTERVAL_S)
                    if on_grid:
                        gap_violations += 1
                    else:
                        misaligned_step += 1

    if ordering_issues:
        report.add(check_name, "FAIL", f"{ordering_issues} approach_edge group(s) have non-increasing or out-of-order timestamps.")
    else:
        report.add(check_name, "PASS", "Timestamps are ordered within each approach_edge.")

    if misaligned_step:
        report.add(check_name, "FAIL", f"{misaligned_step} timestamp gap(s) not even aligned to SAMPLING_INTERVAL_S={SAMPLING_INTERVAL_S}s.")
    if gap_violations:
        report.add(
            check_name, "FAIL",
            f"{gap_violations} timestamp gap(s) larger than one SAMPLING_INTERVAL_S={SAMPLING_INTERVAL_S}s step "
            f"-- indicates missing sample(s), not just misalignment.",
        )
    if duplicate_ts:
        report.add(check_name, "PASS", f"{duplicate_ts} zero-diff (duplicate) timestamp pair(s) detected -- see 'keys' check for the FAIL on this.")
    if not misaligned_step and not gap_violations:
        report.add(check_name, "PASS", f"Timestamps are contiguous at exactly SAMPLING_INTERVAL_S={SAMPLING_INTERVAL_S}s within each approach_edge.")


def check_feature_label_key_compat(features_df: pd.DataFrame, labels_df: pd.DataFrame, report: ScenarioLayerReport) -> None:
    if features_df is None or labels_df is None or features_df.empty or labels_df.empty:
        return
    if not all(c in features_df.columns for c in FEATURE_KEY_COLUMNS):
        return
    if not all(c in labels_df.columns for c in FEATURE_KEY_COLUMNS):
        report.add("feature_label_keys", "FAIL", "Labels file missing (timestamp, approach_edge) key columns.")
        return

    f_keys = set(map(tuple, features_df[FEATURE_KEY_COLUMNS].itertuples(index=False, name=None)))
    l_keys = set(map(tuple, labels_df[FEATURE_KEY_COLUMNS].itertuples(index=False, name=None)))

    only_in_features = f_keys - l_keys
    only_in_labels = l_keys - f_keys

    if only_in_features or only_in_labels:
        report.add(
            "feature_label_keys", "FAIL",
            f"Feature/label key mismatch: {len(only_in_features)} key(s) only in features, "
            f"{len(only_in_labels)} key(s) only in labels.",
        )
    else:
        report.add("feature_label_keys", "PASS", "Feature and label keys match exactly.")

    # Key-SET equality alone does not imply row-count equality if either
    # side has duplicate keys (e.g. labels: A, B, B, C vs features: A, B,
    # C -- sets match, row counts don't). check_keys() is run separately
    # on both frames to catch duplicates directly; this check additionally
    # verifies the row counts agree as a cheap, direct cross-check.
    if len(features_df) != len(labels_df):
        report.add(
            "feature_label_keys", "FAIL",
            f"Row count mismatch: features has {len(features_df)} row(s), labels has {len(labels_df)} row(s) "
            f"-- possible duplicate key(s) in one file (see the 'keys' / 'keys_labels' checks).",
        )
    else:
        report.add("feature_label_keys", "PASS", f"Feature and label row counts match ({len(features_df)}).")


# ============================================================================
# 2. Missing-value checks
# ============================================================================

def check_missingness(df: pd.DataFrame, report: ScenarioLayerReport) -> None:
    """Column-level missingness summary for features_df. The detailed
    natural-vs-internal-gap distinction for history columns (TASK 5) lives
    in the dedicated check_temporal_history_gaps() below, which can also
    point at the actual rows responsible -- this function stays focused on
    a straightforward per-column missing-value tally."""
    if df is None or df.empty:
        return

    n_rows = len(df)
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        frac = n_missing / n_rows
        report.missing_summary[col] = {
            "n_missing": n_missing,
            "fraction": round(frac, 4),
            "expected_reason": EXPECTED_NA_COLUMNS.get(col),
        }

    for col, info in report.missing_summary.items():
        if info["expected_reason"] is not None:
            continue
        if info["fraction"] >= 0.5:
            report.add("missingness", "WARNING", f"Column '{col}' is {info['fraction']*100:.1f}% missing with no known structural reason.")
        elif info["fraction"] > 0:
            report.add("missingness", "PASS", f"Column '{col}' has {info['n_missing']} missing value(s) ({info['fraction']*100:.2f}%).")

    if not report.missing_summary:
        report.add("missingness", "PASS", "No missing values detected in any column.")


def check_temporal_history_gaps(df: pd.DataFrame, report: ScenarioLayerReport) -> None:
    """TASK 5: distinguishes, per approach_edge, the two structurally
    different reasons a *_change_{DELTA_WINDOW_S}s row can be NaN:

      (a) NATURAL initial-history absence: t - DELTA_WINDOW_S is before
          this approach's own first observed timestamp. Expected,
          structural, not a dataset problem.
      (b) INTERNAL missing-sample gap: t - DELTA_WINDOW_S falls ON this
          approach's own canonical timestamp grid (t_min, t_min+interval,
          ...) but is absent from the actual observed timestamps -- a
          dataset integrity problem.

    Unlike a purely aggregate count, this check identifies the ACTUAL rows
    responsible for case (b) where practical: for every row whose required
    history timestamp is an internal gap, it looks up that exact
    (approach_edge, timestamp) row in each *_change_ column and confirms
    whether it is actually NaN there, reporting concrete
    (approach_edge, timestamp, column) examples rather than only a count.
    Never modifies data -- this is audit-only, per the module's read-only
    contract.

    TASK 4: all timestamp membership tests here go through _round_ts() so
    a canonical-grid timestamp built by repeated float addition (t +=
    SAMPLING_INTERVAL_S) cannot drift out of exact-equality range with the
    same timestamp as it appears in the data.
    """
    if df is None or df.empty or "timestamp" not in df.columns or "approach_edge" not in df.columns:
        return

    change_cols = [
        c for c, reason in EXPECTED_NA_COLUMNS.items()
        if c in df.columns and "history" in reason
    ]
    if not change_cols:
        return

    total_natural = 0
    total_internal_expected = 0
    total_internal_confirmed_nan = 0
    examples: List[str] = []

    for edge, group in df.groupby("approach_edge"):
        ts_numeric = pd.to_numeric(group["timestamp"], errors="coerce")
        valid = ts_numeric.notna()
        if not valid.any():
            continue

        g = group.loc[valid].copy()
        g["_ts_rounded"] = ts_numeric[valid].map(_round_ts)
        rounded_set = set(g["_ts_rounded"])
        t_min = min(rounded_set)
        t_max = max(rounded_set)

        canonical_grid = set()
        t = t_min
        while t <= t_max + TS_ABS_TOL:
            canonical_grid.add(_round_ts(t))
            t += SAMPLING_INTERVAL_S

        row_by_ts = {row["_ts_rounded"]: row for _, row in g.iterrows()}

        for t_r in rounded_set:
            prior_r = _round_ts(t_r - DELTA_WINDOW_S)
            if prior_r in rounded_set:
                continue  # required history timestamp exists -- no gap of either kind
            if prior_r < t_min - TS_ABS_TOL or prior_r not in canonical_grid:
                total_natural += 1
                continue

            # prior_r should exist on this approach's own grid but doesn't
            # -- an internal missing-sample gap.
            total_internal_expected += 1
            row = row_by_ts.get(t_r)
            if row is None:
                continue
            for col in change_cols:
                if col not in row.index:
                    continue
                if pd.isna(row[col]):
                    total_internal_confirmed_nan += 1
                    if len(examples) < 8:
                        examples.append(f"({edge}, t={t_r}, {col})")

    if total_internal_expected == 0:
        report.add(
            "temporal_history_gaps", "PASS",
            f"No internal missing-sample gaps found on any approach_edge's canonical timestamp grid "
            f"({total_natural} natural initial-history row(s) identified and correctly excluded).",
        )
        return

    if total_internal_confirmed_nan > 0:
        report.add(
            "temporal_history_gaps", "FAIL",
            f"{total_internal_expected} row(s) require a t-{DELTA_WINDOW_S}s history timestamp that should "
            f"exist on that approach_edge's own canonical SAMPLING_INTERVAL_S grid but is absent from the "
            f"actual data (an internal gap, not natural initial-history insufficiency -- "
            f"{total_natural} of those were natural and correctly excluded). "
            f"{total_internal_confirmed_nan} instance(s) confirmed as NaN in the corresponding temporal "
            f"feature column(s). Example (approach_edge, timestamp, column) rows: {examples}",
        )
    else:
        report.add(
            "temporal_history_gaps", "WARNING",
            f"{total_internal_expected} internal missing-sample gap(s) detected on the canonical timestamp "
            f"grid, but the corresponding *_change_{DELTA_WINDOW_S}s column(s) did not show NaN at those "
            f"specific rows -- unexpected; worth reviewing manually.",
        )


# ============================================================================
# 3. Temporal consistency checks
# ============================================================================

def check_temporal_features(df: pd.DataFrame, report: ScenarioLayerReport) -> None:
    """INTERNAL temporal consistency check -- not an independent
    provenance/leakage verification. It reconstructs
    feature(t) == raw(t) - raw(t - DELTA_WINDOW_S) using timestamps
    (never a fixed row offset, per feature_builder.add_change_features /
    _resolve_delta_steps) against the *_change_{DELTA_WINDOW_S}s column
    and its own base column, both already inside this same generated
    features file. It can catch a broken/mislabeled delta computation; it
    cannot by itself confirm the base column was sourced from the correct
    upstream observation file -- that is a separate, out-of-scope
    provenance concern.

    No forward-difference / "future leakage" heuristic is used here: a
    legitimate past-only change can coincidentally equal the future
    forward-difference (e.g. a locally linear trend), which would make
    such a heuristic produce false positives. Leakage is instead checked
    structurally in check_leakage() and via the exact past-only
    reconstruction below.

    TASK 4: prior-timestamp lookup now goes through a rounded-timestamp
    index map instead of raw float `in` membership, so a valid prior
    timestamp represented as e.g. 29.999999999999996 instead of 30.0 is
    still found rather than silently skipped as "no history"."""
    if df is None or df.empty or "timestamp" not in df.columns or "approach_edge" not in df.columns:
        return

    if DELTA_WINDOW_S % SAMPLING_INTERVAL_S != 0:
        report.add(
            "temporal_features", "FAIL",
            f"DELTA_WINDOW_S={DELTA_WINDOW_S} is not an exact multiple of "
            f"SAMPLING_INTERVAL_S={SAMPLING_INTERVAL_S} -- feature_builder would itself raise on this.",
        )
        return

    change_cols = [c for c in df.columns if c.endswith(f"_change_{DELTA_WINDOW_S}s")]
    if not change_cols:
        return

    base_cols = {c: c[: -len(f"_change_{DELTA_WINDOW_S}s")] for c in change_cols}
    n_mismatches = 0
    n_checked = 0

    for edge, group in df.sort_values(["approach_edge", "timestamp"]).groupby("approach_edge"):
        g = group.set_index("timestamp")
        # Rounded-timestamp -> actual index value, for tolerant lookups.
        ts_index_map = {_round_ts(idx): idx for idx in g.index}
        for change_col, base_col in base_cols.items():
            if base_col not in g.columns:
                continue
            for ts, row in g.iterrows():
                observed_change = row[change_col]
                if pd.isna(observed_change):
                    continue
                prior_key = _round_ts(ts - DELTA_WINDOW_S)
                if prior_key not in ts_index_map:
                    continue
                actual_prior_ts = ts_index_map[prior_key]
                prior_val = g.loc[actual_prior_ts, base_col]
                current_val = row[base_col]
                if pd.isna(prior_val) or pd.isna(current_val):
                    continue
                expected_change = current_val - prior_val
                n_checked += 1
                if not math.isclose(float(observed_change), float(expected_change), abs_tol=1e-6, rel_tol=1e-6):
                    n_mismatches += 1

    if n_checked == 0:
        report.add("temporal_features", "WARNING", "No verifiable *_change_{}s rows found (insufficient overlapping history to check).".format(DELTA_WINDOW_S))
    elif n_mismatches > 0:
        report.add(
            "temporal_features", "FAIL",
            f"{n_mismatches}/{n_checked} '_change_{DELTA_WINDOW_S}s' values do not equal "
            f"value(t) - value(t-{DELTA_WINDOW_S}s) using actual timestamps.",
        )
    else:
        report.add("temporal_features", "PASS", f"All {n_checked} checked '_change_{DELTA_WINDOW_S}s' values match value(t) - value(t-{DELTA_WINDOW_S}s) (internal consistency only).")


# ============================================================================
# 4. Label alignment checks
# ============================================================================

def check_label_shift_consistency(labels_df: pd.DataFrame, report: ScenarioLayerReport) -> None:
    """SHIFT-CONSISTENCY check, not independent verification against
    ground_truth/state_timeseries.csv. It checks that
    true_queue_length_future_m (when present) equals true_queue_length_m
    at t + horizon, WITHIN THE SAME GENERATED labels_{layer}.csv file --
    using the horizon recorded in that file itself
    (feature_builder.build_labels writes 'prediction_horizon_s' alongside
    the shifted column, so the horizon is never hardcoded here). This can
    catch a broken/mis-shifted horizon; it does not re-read raw ground
    truth to confirm true_queue_length_m itself is correct, since
    dataset_qa.py audits the generated features/labels output, not
    ground_truth/ directly.

    TASK 4: target-timestamp lookup now goes through a rounded-timestamp
    index map (same approach as check_temporal_features) instead of raw
    float `in` membership, so float noise on t + horizon_s can't produce
    a spurious "no overlapping timestamp" skip."""
    if labels_df is None or labels_df.empty:
        return
    if "true_queue_length_future_m" not in labels_df.columns:
        report.add("label_shift_consistency", "PASS", "No future-horizon label column present -- nothing to check.")
        return
    if "prediction_horizon_s" not in labels_df.columns:
        report.add("label_shift_consistency", "WARNING", "'true_queue_length_future_m' present but 'prediction_horizon_s' missing -- cannot verify horizon.")
        return

    horizons = labels_df["prediction_horizon_s"].dropna().unique()
    if len(horizons) == 0:
        report.add("label_shift_consistency", "WARNING", "'prediction_horizon_s' column present but entirely empty.")
        return
    if len(horizons) > 1:
        report.add("label_shift_consistency", "WARNING", f"Multiple distinct prediction_horizon_s values found: {sorted(horizons)}")
    horizon_s = float(horizons[0])

    if not all(c in labels_df.columns for c in ["timestamp", "approach_edge", "true_queue_length_m"]):
        report.add("label_shift_consistency", "FAIL", "Missing timestamp/approach_edge/true_queue_length_m -- cannot verify label alignment.")
        return

    n_checked = 0
    n_mismatches = 0
    for edge, group in labels_df.sort_values(["approach_edge", "timestamp"]).groupby("approach_edge"):
        g = group.set_index("timestamp")
        ts_index_map = {_round_ts(idx): idx for idx in g.index}
        for ts, row in g.iterrows():
            future_label = row["true_queue_length_future_m"]
            if pd.isna(future_label):
                continue
            target_key = _round_ts(ts + horizon_s)
            if target_key not in ts_index_map:
                continue
            actual_target_ts = ts_index_map[target_key]
            actual_future_truth = g.loc[actual_target_ts, "true_queue_length_m"]
            if pd.isna(actual_future_truth):
                continue
            n_checked += 1
            if not math.isclose(float(future_label), float(actual_future_truth), abs_tol=1e-6, rel_tol=1e-6):
                n_mismatches += 1

    if n_checked == 0:
        report.add("label_shift_consistency", "WARNING", "Could not verify any future-label rows against ground truth at t + horizon (no overlapping timestamps).")
    elif n_mismatches > 0:
        report.add("label_shift_consistency", "FAIL", f"{n_mismatches}/{n_checked} future-label rows do not match true_queue_length_m at t + {horizon_s}s.")
    else:
        report.add("label_shift_consistency", "PASS", f"All {n_checked} checked future-label rows correctly align with ground truth at t + {horizon_s}s.")


# ============================================================================
# 5. Leakage checks
# ============================================================================

def check_leakage(features_df: pd.DataFrame, report: ScenarioLayerReport) -> None:
    if features_df is None:
        return

    leaked = FORBIDDEN_GROUND_TRUTH_COLUMNS.intersection(features_df.columns)
    if leaked:
        report.add("leakage", "FAIL", f"Forbidden ground-truth-shaped column(s) present in features: {sorted(leaked)}")
    else:
        report.add("leakage", "PASS", "No forbidden ground-truth-shaped columns found in features (per feature_builder.FORBIDDEN_GROUND_TRUTH_COLUMNS).")

    suspicious_prefixes = ("true_", "target_", "label_")
    suspicious = [c for c in features_df.columns if c.startswith(suspicious_prefixes)]
    if suspicious:
        report.add("leakage", "FAIL", f"Feature column(s) with target/ground-truth-like naming: {suspicious}")

    # Any *_future_* named column in a features file is inherently suspect.
    future_named = [c for c in features_df.columns if "future" in c.lower()]
    if future_named:
        report.add("leakage", "FAIL", f"Feature column(s) with future-oriented naming: {future_named}")


# ============================================================================
# 6. Layer checks
# ============================================================================

# TLS-dependent Layer-2 columns: per feature_builder.add_signal_features(),
# these are the ONLY columns left NA (not necessarily absent, but
# structurally unreliable) when raw_output/tls_state.csv is unavailable.
# GPS-derived and physics-derived columns have no such dependency -- their
# absence is a genuine defect, not something TLS availability can excuse.
TLS_DEPENDENT_COLUMNS = {"current_phase", "phase_elapsed_s", "is_green_for_approach", "red_duration_s"}
GPS_DEPENDENT_COLUMNS = set(GPS_REQUIRED_COLUMNS) | {
    f"probe_count_change_{DELTA_WINDOW_S}s",
    f"probe_max_distance_to_stopline_m_change_{DELTA_WINDOW_S}s",
}
# estimated_hidden_queue_extension_m indirectly depends on red_duration_s /
# is_green_for_approach (see add_physics_features), so it is TLS-dependent
# too; estimated_density_k_veh_per_km / observed_flow_veh_per_hour /
# estimated_queue_front_propagation_m_per_s are camera/GPS-only physics
# and are NOT TLS-dependent.
PHYSICS_TLS_DEPENDENT_COLUMNS = {"estimated_hidden_queue_extension_m"}
PHYSICS_INDEPENDENT_COLUMNS = {
    "estimated_density_k_veh_per_km", "observed_flow_veh_per_hour",
    "estimated_queue_front_propagation_m_per_s",
}


def check_layer_integrity(features_df: pd.DataFrame, layer: str, report: ScenarioLayerReport) -> None:
    if features_df is None or features_df.empty:
        return
    cols = set(features_df.columns)

    if layer == "layer1":
        unintended_l2 = cols.intersection(LAYER2_ADDITIONAL_COLUMNS)
        if unintended_l2:
            report.add("layer_integrity", "FAIL", f"Layer 1 features contain Layer-2-only column(s): {sorted(unintended_l2)}")
        else:
            report.add("layer_integrity", "PASS", "Layer 1 contains no Layer-2-only columns.")

        missing_l1 = LAYER1_EXPECTED_COLUMNS - cols
        if missing_l1:
            report.add("layer_integrity", "FAIL", f"Layer 1 is missing expected column(s): {sorted(missing_l1)}")

    elif layer == "layer2":
        missing_l1_in_l2 = LAYER1_EXPECTED_COLUMNS - cols
        if missing_l1_in_l2:
            report.add("layer_integrity", "FAIL", f"Layer 2 is missing Layer-1-inherited column(s): {sorted(missing_l1_in_l2)}")
        else:
            report.add("layer_integrity", "PASS", "Layer 2 retains all expected Layer-1 columns.")

        missing_l2_additions = LAYER2_ADDITIONAL_COLUMNS - cols
        tls_missing = missing_l2_additions & (TLS_DEPENDENT_COLUMNS | PHYSICS_TLS_DEPENDENT_COLUMNS)
        non_tls_missing = missing_l2_additions - tls_missing

        if non_tls_missing:
            report.add(
                "layer_integrity", "FAIL",
                f"Layer 2 is missing expected addition(s) with NO TLS dependency "
                f"(GPS-derived or TLS-independent physics -- must always be present): {sorted(non_tls_missing)}",
            )
        if tls_missing:
            report.add(
                "layer_integrity", "WARNING",
                f"Layer 2 is missing TLS-dependent column(s), legitimately absent/NA if "
                f"raw_output/tls_state.csv was unavailable for this scenario: {sorted(tls_missing)}",
            )
        if not missing_l2_additions:
            report.add("layer_integrity", "PASS", "Layer 2 contains all expected Layer-2 additions.")


# ============================================================================
# 7. Numerical / physical sanity checks
# ============================================================================

# Columns expected to always hold numeric values in a well-formed
# generated dataset -- derived from CAMERA_REQUIRED_COLUMNS /
# GPS_REQUIRED_COLUMNS (minus known non-numeric fields like
# approach_edge / queue_reaches_camera_edge, which are string/bool) plus
# the physics-derived columns. Used to catch malformed values (e.g. a
# stray non-numeric string) that pd.to_numeric(errors="coerce") would
# otherwise silently turn into NaN and hide from every downstream check.
EXPECTED_NUMERIC_COLUMNS = sorted(
    (set(NONNEGATIVE_COLUMNS) | {
        "timestamp", "visible_mean_speed_mps", "probe_mean_speed_mps",
        "visible_occupancy_fraction",
    })
)

# TASK 2: label-only numeric column set. Deliberately excludes
# true_queue_beyond_camera -- per feature_builder.build_labels(), that
# column is written straight from ground_truth's boolean
# queue_beyond_camera flag, never as a numeric quantity, so it is audited
# by check_censoring-style boolean logic elsewhere, not here.
EXPECTED_NUMERIC_LABEL_COLUMNS = [
    "timestamp", "true_queue_length_m", "true_queue_length_future_m", "prediction_horizon_s",
]


def check_numeric_schema(
    df: pd.DataFrame, report: ScenarioLayerReport, source: str = "features",
    columns: Optional[List[str]] = None,
) -> None:
    """Detect malformed values in columns that should always be numeric.
    A naive pd.to_numeric(errors='coerce') silently turns an unexpected
    string (e.g. 'banana' in visible_vehicle_count) into NaN, which then
    looks identical to a legitimate missing value everywhere else in this
    audit. This check compares each column's ORIGINAL non-null count
    against its post-coercion non-null count: any row that had a real
    (non-null) value before coercion but became NaN after it is a
    malformed value, not a missing one, and is reported as a FAIL.

    TASK 2: 'source'/'columns' let this same logic run against
    labels_{layer}.csv (EXPECTED_NUMERIC_LABEL_COLUMNS) as well as
    features_{layer}.csv (EXPECTED_NUMERIC_COLUMNS, the default), with the
    report distinguishing "numeric_schema" (features) from
    "numeric_schema_labels" (labels)."""
    check_name = "numeric_schema" if source == "features" else f"numeric_schema_{source}"
    cols = columns if columns is not None else EXPECTED_NUMERIC_COLUMNS
    if df is None or df.empty:
        return

    any_malformed = False
    for col in cols:
        if col not in df.columns:
            continue
        original = df[col]
        # Already a native numeric dtype -- coercion cannot introduce new
        # NaNs beyond what pandas itself already parsed, so this column
        # is trivially clean.
        if pd.api.types.is_numeric_dtype(original):
            continue
        coerced = pd.to_numeric(original, errors="coerce")
        originally_present = original.notna()
        newly_nan = originally_present & coerced.isna()
        if newly_nan.any():
            any_malformed = True
            bad_values = sorted(set(original[newly_nan].astype(str)))[:5]
            report.add(
                check_name, "FAIL",
                f"Column '{col}' expected to be numeric has {int(newly_nan.sum())} non-numeric value(s) "
                f"that cannot be parsed, e.g. {bad_values} -- these are malformed, not legitimately missing.",
            )

    if not any_malformed:
        report.add(check_name, "PASS", f"All expected-numeric {source} columns contain only numeric or legitimately-null values.")


def check_physics_sanity(
    df: pd.DataFrame, report: ScenarioLayerReport, source: str = "features",
    nonneg_columns: Optional[List[str]] = None,
) -> None:
    """TASK 3: 'source'/'nonneg_columns' let the nonnegative-value check
    run against labels_{layer}.csv (LABEL_NONNEGATIVE_COLUMNS) as well as
    features_{layer}.csv (NONNEGATIVE_COLUMNS, the default) -- reported as
    "physics_sanity" vs "physics_sanity_labels". The additional
    feature-specific structural checks (infinite values, occupancy-
    fraction range, queue-vs-camera-range, probe min/max ordering) only
    apply to the features file and are skipped for labels, since they
    reference feature-only columns."""
    check_name = "physics_sanity" if source == "features" else f"physics_sanity_{source}"
    cols = nonneg_columns if nonneg_columns is not None else NONNEGATIVE_COLUMNS
    if df is None or df.empty:
        return

    for col in cols:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        negative = numeric < 0
        if negative.any():
            report.add(check_name, "FAIL", f"Column '{col}' has {int(negative.sum())} negative value(s), which is physically invalid.")

    if source == "features":
        # Infinite / non-finite values anywhere in numeric columns.
        numeric_df = df.select_dtypes(include="number")
        inf_mask = numeric_df.isin([float("inf"), float("-inf")])
        inf_cols = [c for c in numeric_df.columns if inf_mask[c].any()]
        if inf_cols:
            report.add(check_name, "FAIL", f"Infinite value(s) detected in column(s): {inf_cols}")

        if "visible_occupancy_fraction" in df.columns:
            occ = pd.to_numeric(df["visible_occupancy_fraction"], errors="coerce").dropna()
            out_of_range = occ[(occ < 0) | (occ > 1)]
            if not out_of_range.empty:
                report.add(check_name, "FAIL", f"'visible_occupancy_fraction' has {len(out_of_range)} value(s) outside [0, 1].")

        if "visible_queue_length_m" in df.columns and "camera_range_m" in df.columns:
            ql = pd.to_numeric(df["visible_queue_length_m"], errors="coerce")
            rng = pd.to_numeric(df["camera_range_m"], errors="coerce")
            exceeds = (ql > rng) & ql.notna() & rng.notna()
            if exceeds.any():
                report.add(check_name, "FAIL", f"'visible_queue_length_m' exceeds 'camera_range_m' in {int(exceeds.sum())} row(s) -- camera cannot see beyond its own range.")

        if "probe_min_distance_to_stopline_m" in df.columns and "probe_max_distance_to_stopline_m" in df.columns:
            lo = pd.to_numeric(df["probe_min_distance_to_stopline_m"], errors="coerce")
            hi = pd.to_numeric(df["probe_max_distance_to_stopline_m"], errors="coerce")
            both_present = lo.notna() & hi.notna()
            inverted = both_present & (lo > hi)
            if inverted.any():
                report.add(check_name, "FAIL", f"'probe_min_distance_to_stopline_m' > 'probe_max_distance_to_stopline_m' in {int(inverted.sum())} row(s).")

    already_failed = any(c.check == check_name and c.status == "FAIL" for c in report.checks)
    if not already_failed:
        report.add(check_name, "PASS", f"No negative{' , infinite, or out-of-range' if source == 'features' else ''} values detected in checked {source} columns.")


# ============================================================================
# 8. Censoring checks (queue_reaches_camera_edge semantics)
# ============================================================================

_TRUE_STRINGS = {"true", "1", "1.0", "yes"}
_FALSE_STRINGS = {"false", "0", "0.0", "no"}


def _as_bool_series(col: pd.Series) -> pd.Series:
    """Normalize a boolean-intended column to real pandas bool (with NA),
    regardless of whether CSV round-tripping left it as native bool,
    0/1 ints, or the string forms "True"/"False" that a plain
    pd.read_csv can produce depending on how the column was written.
    Unrecognized values become NA rather than being silently coerced to
    False, so a genuinely malformed column is still visible as missing
    rather than misread."""
    if col.dtype == bool:
        return col
    if pd.api.types.is_numeric_dtype(col):
        return col.map(lambda v: pd.NA if pd.isna(v) else bool(v))

    def _parse(v):
        if pd.isna(v):
            return pd.NA
        s = str(v).strip().lower()
        if s in _TRUE_STRINGS:
            return True
        if s in _FALSE_STRINGS:
            return False
        return pd.NA

    return col.map(_parse)


def check_censoring(df: pd.DataFrame, report: ScenarioLayerReport) -> None:
    if df is None or df.empty or "queue_reaches_camera_edge" not in df.columns:
        return

    censored_raw = _as_bool_series(df["queue_reaches_camera_edge"])
    if censored_raw.isna().any() and not df["queue_reaches_camera_edge"].isna().any():
        # This column gates downstream physics logic (queue-front
        # propagation, hidden-queue extension) -- an unparseable value
        # means QA cannot trust the censoring state for that row, which
        # is a data-integrity FAIL, not merely a WARNING.
        report.add(
            "censoring", "FAIL",
            f"'queue_reaches_camera_edge' has {int(censored_raw.isna().sum())} value(s) that could not be "
            f"interpreted as boolean (unexpected representation) -- censoring state cannot be trusted for these rows.",
        )
    censored = censored_raw == True  # noqa: E712

    # estimated_queue_front_propagation_m_per_s must be NA exactly while
    # censored (per add_physics_features: "Valid ONLY while
    # queue_reaches_camera_edge is False this interval; NA when censored").
    if "estimated_queue_front_propagation_m_per_s" in df.columns:
        prop = df["estimated_queue_front_propagation_m_per_s"]
        violating = censored & prop.notna()
        if violating.any():
            report.add(
                "censoring", "FAIL",
                f"'estimated_queue_front_propagation_m_per_s' has {int(violating.sum())} non-NA value(s) "
                f"while queue_reaches_camera_edge is True (should be NA when censored, per feature_builder.py).",
            )
        else:
            report.add("censoring", "PASS", "'estimated_queue_front_propagation_m_per_s' is NA whenever queue_reaches_camera_edge is True, as intended.")

    # estimated_hidden_queue_extension_m should only be populated while
    # censored AND red (per add_physics_features mask: is_censored & is_red
    # & has_red_duration & has_rate).
    if "estimated_hidden_queue_extension_m" in df.columns and "is_green_for_approach" in df.columns:
        ext = df["estimated_hidden_queue_extension_m"]
        green_raw = _as_bool_series(df["is_green_for_approach"])
        malformed_green = green_raw.isna() & df["is_green_for_approach"].notna()
        if malformed_green.any():
            # Also gates downstream physics logic (red/green determines
            # whether estimated_hidden_queue_extension_m should be
            # populated at all) -- unparseable values here are a FAIL,
            # not a WARNING, for the same reason as queue_reaches_camera_edge.
            report.add(
                "censoring", "FAIL",
                f"'is_green_for_approach' has {int(malformed_green.sum())} value(s) that could not be "
                f"interpreted as boolean -- red/green state cannot be trusted for these rows.",
            )
        is_red = green_raw == False  # noqa: E712
        populated_but_not_censored_red = ext.notna() & ~(censored & is_red)
        if populated_but_not_censored_red.any():
            report.add(
                "censoring", "FAIL",
                f"'estimated_hidden_queue_extension_m' has {int(populated_but_not_censored_red.sum())} value(s) "
                f"populated outside the censored+red condition it requires (per feature_builder.py).",
            )
        else:
            report.add("censoring", "PASS", "'estimated_hidden_queue_extension_m' is only populated when censored and red, as intended.")

        ext_numeric = pd.to_numeric(ext, errors="coerce")
        negative_ext = ext_numeric < 0
        if negative_ext.any():
            report.add("censoring", "FAIL", f"'estimated_hidden_queue_extension_m' has {int(negative_ext.sum())} negative value(s) -- extension cannot shrink the queue.")

    # Semantic guard: never assert true queue length == camera_range_m from
    # queue_reaches_camera_edge alone; this only checks that no feature
    # column claims to directly equal camera_range_m at censoring, which
    # would indicate the forbidden assumption crept into a derived column.
    if "visible_queue_length_m" in df.columns and "camera_range_m" in df.columns:
        ql = pd.to_numeric(df["visible_queue_length_m"], errors="coerce")
        rng = pd.to_numeric(df["camera_range_m"], errors="coerce")
        # Not an error for visible_queue_length_m to equal camera_range_m
        # at the moment of censoring (that's what "reaches the edge"
        # means observationally) -- only flag if it EXCEEDS range, already
        # covered in check_physics_sanity. Nothing further asserted here
        # to avoid encoding a wrong equivalence ourselves.
        pass


# ============================================================================
# 9. Manifest / metadata checks
# ============================================================================

def check_manifest(manifest: Optional[dict], features_df: pd.DataFrame, layer: str, tag: Optional[str], report: ScenarioLayerReport) -> None:
    if manifest is None:
        report.add("manifest", "WARNING", "No feature manifest JSON found -- skipping manifest checks (optional metadata).")
        return
    if features_df is None:
        return

    if manifest.get("layer") != layer:
        report.add("manifest", "FAIL", f"Manifest 'layer' field ({manifest.get('layer')!r}) does not match expected layer ({layer!r}).")

    if "sampling_interval_s" in manifest and manifest["sampling_interval_s"] != SAMPLING_INTERVAL_S:
        report.add("manifest", "FAIL", f"Manifest sampling_interval_s={manifest['sampling_interval_s']} does not match project SAMPLING_INTERVAL_S={SAMPLING_INTERVAL_S}.")

    # feature_builder.build_feature_manifest() (the authoritative writer
    # for this project) puts feature names directly as top-level keys,
    # alongside a handful of reserved metadata keys ("layer",
    # "sampling_interval_s", and underscore-prefixed "_..." keys). That
    # flat shape is asserted here because it matches the actual function
    # in feature_builder.py -- but this is read defensively: if a
    # manifest is ever found with a different top-level shape (e.g. a
    # nested "features"/"provenance" object), that's treated as a schema
    # mismatch WARNING rather than silently comparing the wrong thing.
    RESERVED_KEYS = {"layer", "sampling_interval_s"}
    top_level_keys = set(manifest.keys())
    non_reserved = {k for k in top_level_keys if not k.startswith("_") and k not in RESERVED_KEYS}

    looks_flat = non_reserved and all(not isinstance(manifest[k], dict) or "kind" in manifest[k] or "source" in manifest[k] for k in non_reserved)
    if not looks_flat and non_reserved:
        report.add(
            "manifest", "WARNING",
            "Manifest top-level structure does not match the expected flat "
            "feature-name-as-key schema written by feature_builder.build_feature_manifest() "
            "-- skipping feature-list cross-check.",
        )
        return

    manifest_feature_keys = non_reserved
    actual_cols = set(features_df.columns)

    documented_missing = manifest_feature_keys - actual_cols
    if documented_missing:
        report.add("manifest", "WARNING", f"Manifest documents column(s) not present in the features file: {sorted(documented_missing)}")

    undocumented = actual_cols - manifest_feature_keys - set(FEATURE_KEY_COLUMNS)
    if undocumented:
        report.add("manifest", "WARNING", f"Features file has column(s) not documented in the manifest: {sorted(undocumented)}")

    if not documented_missing and not undocumented:
        report.add("manifest", "PASS", "Manifest feature list matches the features file's columns.")

    if layer == "layer2" and "_k_jam_veh_per_km" in manifest:
        k_jam = manifest["_k_jam_veh_per_km"]
        if not isinstance(k_jam, (int, float)) or k_jam <= 0:
            report.add("manifest", "FAIL", f"Manifest '_k_jam_veh_per_km'={k_jam!r} is not a positive number.")

    if "_removed_features" in manifest:
        removed = set(manifest["_removed_features"].keys())
        reintroduced = removed.intersection(actual_cols)
        if reintroduced:
            report.add("manifest", "FAIL", f"Column(s) documented as REMOVED in the manifest are present in features: {sorted(reintroduced)}")


# ============================================================================
# Orchestration for one (scenario, layer, tag)
# ============================================================================

def audit_scenario_layer(scenario_dir: Path, layer: str, tag: Optional[str]) -> ScenarioLayerReport:
    scenario_id = scenario_dir.name
    report = ScenarioLayerReport(scenario_id=scenario_id, layer=layer, tag=tag)

    data = load_data(scenario_dir, layer, tag)
    for err in data["load_errors"]:
        report.add("load", "FAIL", err)

    features_df = data["features_df"]
    labels_df = data["labels_df"]
    manifest = data["manifest"]

    if features_df is not None:
        report.row_count = len(features_df)
        if "approach_edge" in features_df.columns:
            report.n_approaches = int(features_df["approach_edge"].nunique())

    check_schema(features_df, layer, report)
    check_keys(features_df, report, source="features")
    check_timestamps(features_df, report, source="features")
    check_label_schema(labels_df, report)
    check_keys(labels_df, report, source="labels")
    check_timestamps(labels_df, report, source="labels")  # TASK 1
    check_feature_label_key_compat(features_df, labels_df, report)
    check_missingness(features_df, report)
    check_label_missingness(labels_df, report)
    check_temporal_history_gaps(features_df, report)  # TASK 5
    check_temporal_features(features_df, report)
    check_label_shift_consistency(labels_df, report)
    check_leakage(features_df, report)
    check_layer_integrity(features_df, layer, report)
    check_numeric_schema(features_df, report, source="features")
    check_numeric_schema(labels_df, report, source="labels", columns=EXPECTED_NUMERIC_LABEL_COLUMNS)  # TASK 2
    check_physics_sanity(features_df, report, source="features")
    check_physics_sanity(labels_df, report, source="labels", nonneg_columns=LABEL_NONNEGATIVE_COLUMNS)  # TASK 3
    check_censoring(features_df, report)
    check_manifest(manifest, features_df, layer, tag, report)

    for drift_warning in _cross_check_na_reasons_against_manifest():
        report.add("contract_drift", "WARNING", drift_warning)

    return report


# ============================================================================
# 10. Scenario-level reporting
# ============================================================================

def build_report(reports: List[ScenarioLayerReport]) -> dict:
    overall = "PASS"
    for r in reports:
        s = r.overall_status
        if STATUS_ORDER[s] > STATUS_ORDER[overall]:
            overall = s

    return {
        "overall_status": overall,
        "n_scenario_layers_audited": len(reports),
        "results": [r.to_dict() for r in reports],
    }


def print_human_report(reports: List[ScenarioLayerReport]) -> None:
    for r in reports:
        tag_str = f" ({r.tag})" if r.tag else ""
        print(f"\n=== {r.scenario_id} :: {r.layer}{tag_str} ===")
        print(f"  rows={r.row_count} approaches={r.n_approaches} "
              f"timestamp_range=[{r.timestamp_min}, {r.timestamp_max}] status={r.overall_status}")
        if r.missing_summary:
            worst = sorted(r.missing_summary.items(), key=lambda kv: -kv[1]["fraction"])[:5]
            print(f"  top missingness: {[(c, i['fraction']) for c, i in worst]}")
        for c in r.checks:
            if c.status != "PASS":
                print(f"  [{c.status}] {c.check}: {c.message}")


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit ASTRID per-scenario feature/label datasets before assembly. Read-only -- never modifies data."
    )
    parser.add_argument("--scenario", type=str, default=None, help="Audit only this scenario (directory name). Default: all discovered scenarios.")
    parser.add_argument("--layer", type=str, default=None, choices=["layer1", "layer2"], help="Audit only this layer. Default: both.")
    parser.add_argument("--penetration", type=float, default=None, help="For layer2, audit only this penetration rate (e.g. 0.11). Default: all found penetration tags.")
    parser.add_argument("--strict", action="store_true", help="Treat WARNING as a failing condition for the process exit code.")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to write the full JSON report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scenario_dirs = discover_scenarios(args.scenario)
    if not scenario_dirs:
        print(f"ERROR: no scenarios found under {SCENARIOS_DIR}" + (f" matching '{args.scenario}'" if args.scenario else ""))
        sys.exit(1)

    reports: List[ScenarioLayerReport] = []
    for scenario_dir in scenario_dirs:
        targets = discover_layer_targets(scenario_dir, args.layer, args.penetration)
        if not targets:
            r = ScenarioLayerReport(scenario_id=scenario_dir.name, layer=args.layer or "unknown", tag=None)
            r.add("discovery", "WARNING", "No matching features_*.csv file(s) found for this scenario/layer/penetration combination.")
            reports.append(r)
            continue
        for t in targets:
            reports.append(audit_scenario_layer(scenario_dir, t["layer"], t["tag"]))

    print_human_report(reports)

    full_report = build_report(reports)
    print(f"\n=== OVERALL STATUS: {full_report['overall_status']} "
          f"({full_report['n_scenario_layers_audited']} scenario/layer combination(s) audited) ===")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2, default=str)
        print(f"Full JSON report written to {args.json_out}")

    exit_status = full_report["overall_status"]
    if exit_status == "FAIL" or (args.strict and exit_status == "WARNING"):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()