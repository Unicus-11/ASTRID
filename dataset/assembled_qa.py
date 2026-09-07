
"""
assembled_qa.py
================
ASTRID Prototype -- Read-only QA for assembled ML-ready datasets.

Verifies dataset/assembled/{layer1,layer2_p11}/{train,validation,test,ood}.csv
+ manifest.json, produced by dataset/assemble_dataset.py. Never modifies,
regenerates, recomputes, or imputes anything.

Run:
    python dataset/assembled_qa.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ASSEMBLED_DIR = Path(__file__).resolve().parent / "assembled"
LAYERS = {"layer1": None, "layer2_p11": "p11"}
SPLIT_FILES = {"train": "train.csv", "val": "validation.csv", "test": "test.csv", "ood": "ood.csv"}

KEY_COLUMNS = ["timestamp", "approach_edge"]
REQUIRED_LABEL_COLUMNS = [
    "true_queue_length_m",
    "true_queue_beyond_camera",
]

OPTIONAL_LABEL_COLUMNS = [
    "true_queue_length_future_m",
    "prediction_horizon_s",
]
EXPECTED_N_APPROACHES = 4

FAILS: List[str] = []
WARNINGS: List[str] = []


def fail(msg: str) -> None:
    FAILS.append(msg)


def warn(msg: str) -> None:
    WARNINGS.append(msg)


# ============================================================================
# Loading
# ============================================================================

def load_layer(layer_dir: Path) -> Dict[str, dict]:
    """Load all four split CSVs + manifest for one layer. Missing files
    are recorded as None, not raised, so remaining checks can still run."""
    result = {"manifest": None, "manifest_path": layer_dir / "manifest.json", "splits": {}}

    if result["manifest_path"].exists():
        try:
            with open(result["manifest_path"], "r", encoding="utf-8") as f:
                result["manifest"] = json.load(f)
        except Exception as exc:
            fail(f"{layer_dir.name}: failed to parse manifest.json: {exc}")
    else:
        fail(f"{layer_dir.name}: missing manifest.json")

    for role, fname in SPLIT_FILES.items():
        path = layer_dir / fname
        entry = {"path": path, "df": None}
        if not path.exists():
            fail(f"{layer_dir.name}/{fname}: file missing")
        else:
            try:
                entry["df"] = pd.read_csv(path)
            except Exception as exc:
                fail(f"{layer_dir.name}/{fname}: failed to read CSV: {exc}")
        result["splits"][role] = entry

    return result


# ============================================================================
# 2. Schema consistency
# ============================================================================

def check_schema(layer_name: str, splits: Dict[str, dict]) -> None:
    col_sets = {}
    for role, entry in splits.items():
        df = entry["df"]
        if df is None:
            continue
        cols = list(df.columns)
        if len(cols) != len(set(cols)):
            dupes = [c for c in set(cols) if cols.count(c) > 1]
            fail(f"{layer_name}/{SPLIT_FILES[role]}: duplicate column name(s): {dupes}")
        col_sets[role] = set(cols)

        missing_keys = [c for c in KEY_COLUMNS if c not in cols]
        if missing_keys:
            fail(f"{layer_name}/{SPLIT_FILES[role]}: missing key column(s): {missing_keys}")
        missing_labels = [c for c in REQUIRED_LABEL_COLUMNS if c not in cols]
        if missing_labels:
            fail(f"{layer_name}/{SPLIT_FILES[role]}: missing required label column(s): {missing_labels}")

    if len(col_sets) > 1:
        reference_role = next(iter(col_sets))
        reference = col_sets[reference_role]
        for role, cols in col_sets.items():
            if cols != reference:
                fail(
                    f"{layer_name}: column mismatch between {SPLIT_FILES[role]} and "
                    f"{SPLIT_FILES[reference_role]} -- only in {role}: {sorted(cols - reference)}, "
                    f"only in {reference_role}: {sorted(reference - cols)}"
                )


# ============================================================================
# 3. Feature/label separation (manifest-driven)
# ============================================================================

def check_feature_label_separation(layer_name: str, manifest: Optional[dict], splits: Dict[str, dict]) -> None:
    if manifest is None:
        return
    manifest_splits = manifest.get("splits", {})
    for role, entry in splits.items():
        df = entry["df"]
        if df is None:
            continue
        m = manifest_splits.get(role)
        if m is None:
            warn(f"{layer_name}/{SPLIT_FILES[role]}: no manifest entry for split '{role}'")
            continue

        feature_cols = m.get("feature_columns", [])
        label_cols = m.get("label_columns", [])
        metadata_cols = m.get("metadata_columns", [])

        overlap_labels = set(feature_cols) & set(label_cols)
        if overlap_labels:
            fail(f"{layer_name}/{SPLIT_FILES[role]}: label column(s) also listed as features: {sorted(overlap_labels)}")

        overlap_keys = set(feature_cols) & set(KEY_COLUMNS)
        if overlap_keys:
            fail(f"{layer_name}/{SPLIT_FILES[role]}: key column(s) also listed as features: {sorted(overlap_keys)}")

        overlap_meta = set(feature_cols) & set(metadata_cols)
        if overlap_meta:
            fail(f"{layer_name}/{SPLIT_FILES[role]}: metadata column(s) also listed as features: {sorted(overlap_meta)}")

        all_declared = set(feature_cols) | set(label_cols) | set(metadata_cols)
        missing_in_csv = all_declared - set(df.columns)
        if missing_in_csv:
            fail(f"{layer_name}/{SPLIT_FILES[role]}: manifest column(s) not present in CSV: {sorted(missing_in_csv)}")

        declared = (
            set(KEY_COLUMNS)
            | set(feature_cols)
            | set(label_cols)
            | set(metadata_cols)
        )

        undeclared = set(df.columns) - declared

        if undeclared:
            fail(
                f"{layer_name}/{SPLIT_FILES[role]}: "
                f"CSV contains undeclared column(s): {sorted(undeclared)}"
            )


# ============================================================================
# 4. Scenario split integrity
# ============================================================================

EXPECTED_SPLIT_VALUE = {"train": "train", "val": "val", "test": "test", "ood": "ood"}


def check_split_integrity(layer_name: str, splits: Dict[str, dict]) -> Dict[str, set]:
    scenario_ids_by_role: Dict[str, set] = {}
    all_scenario_role_pairs: Dict[str, str] = {}

    for role, entry in splits.items():
        df = entry["df"]
        if df is None:
            continue
        if "split" not in df.columns or "scenario_id" not in df.columns:
            fail(f"{layer_name}/{SPLIT_FILES[role]}: missing 'split' or 'scenario_id' column -- cannot verify split integrity.")
            continue

        expected = EXPECTED_SPLIT_VALUE[role]
        bad_split = df[df["split"] != expected]
        if not bad_split.empty:
            fail(
                f"{layer_name}/{SPLIT_FILES[role]}: {len(bad_split)} row(s) have split != '{expected}' "
                f"(found: {sorted(bad_split['split'].unique())})"
            )

        ids = set(df["scenario_id"].unique())
        scenario_ids_by_role[role] = ids
        for sid in ids:
            if sid in all_scenario_role_pairs and all_scenario_role_pairs[sid] != role:
                fail(f"{layer_name}: scenario_id '{sid}' appears in both '{all_scenario_role_pairs[sid]}' and '{role}'")
            all_scenario_role_pairs[sid] = role

        print(f"  [{layer_name}/{SPLIT_FILES[role]}] scenario IDs ({len(ids)}): {sorted(ids)}")

    return scenario_ids_by_role


# ============================================================================
# 5. Row counts
# ============================================================================

def report_row_counts(layer_name: str, splits: Dict[str, dict]) -> Dict[str, int]:
    counts = {}
    for role, entry in splits.items():
        df = entry["df"]
        counts[role] = len(df) if df is not None else 0
        print(f"  [{layer_name}/{SPLIT_FILES[role]}] rows: {counts[role]}")
    return counts


# ============================================================================
# 6. Key integrity
# ============================================================================
def check_key_integrity(layer_name: str, splits: Dict[str, dict]) -> None:
    for role, entry in splits.items():
        df = entry["df"]
        if df is None or df.empty:
            continue

        assembled_key_columns = ["scenario_id", "timestamp", "approach_edge"]

        if not all(c in df.columns for c in assembled_key_columns):
            fail(
                f"{layer_name}/{SPLIT_FILES[role]}: missing one or more assembled "
                f"key columns: {assembled_key_columns}"
            )
        else:
            dupes = df[df.duplicated(subset=assembled_key_columns, keep=False)]
            if not dupes.empty:
                fail(
                    f"{layer_name}/{SPLIT_FILES[role]}: "
                    f"{len(dupes)} row(s) with duplicate "
                    f"(scenario_id, timestamp, approach_edge) keys."
                )

        if "approach_edge" not in df.columns:
            continue

        n_approaches = df["approach_edge"].nunique()
        if n_approaches != EXPECTED_N_APPROACHES:
            fail(
                f"{layer_name}/{SPLIT_FILES[role]}: expected "
                f"{EXPECTED_N_APPROACHES} distinct approach_edge values, "
                f"found {n_approaches}: "
                f"{sorted(df['approach_edge'].unique())}"
            )

        if "timestamp" not in df.columns:
            continue

        ts_numeric = pd.to_numeric(df["timestamp"], errors="coerce")
        if ts_numeric.isna().any():
            fail(
                f"{layer_name}/{SPLIT_FILES[role]}: "
                f"{int(ts_numeric.isna().sum())} non-numeric timestamp value(s)."
            )
            continue

        # Row order is not guaranteed across concatenated scenarios.
        # Verify that each scenario/approach has a valid 5-second timestamp grid.
        expected_step = 5.0

        if "scenario_id" not in df.columns:
            continue

        for (scenario_id, edge), group in (
            df.assign(_ts=ts_numeric)
            .groupby(["scenario_id", "approach_edge"])
        ):
            sorted_ts = sorted(group["_ts"].unique())

            if len(sorted_ts) < 2:
                continue

            diffs = pd.Series(sorted_ts).diff().dropna()

            bad_diffs = diffs[
                ~diffs.apply(lambda x: abs(x - expected_step) < 1e-6)
            ]

            if not bad_diffs.empty:
                fail(
                    f"{layer_name}/{SPLIT_FILES[role]}: "
                    f"scenario '{scenario_id}', approach '{edge}' has "
                    f"{len(bad_diffs)} timestamp interval(s) not equal to "
                    f"{expected_step}s; examples: "
                    f"{bad_diffs.head(5).tolist()}"
                )
# ============================================================================
# 7. NaN preservation
# ============================================================================

def report_nan_counts(layer_name: str, splits: Dict[str, dict]) -> Dict[str, Dict[str, int]]:
    nan_counts: Dict[str, Dict[str, int]] = {}
    for role, entry in splits.items():
        df = entry["df"]
        if df is None or df.empty:
            nan_counts[role] = {}
            continue
        counts = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().sum() > 0}
        nan_counts[role] = counts

        # Sanity: a sparse-sensor column with NaNs should not have been
        # silently zero-filled -- if probe_count == 0 exists but the
        # corresponding value columns show ZERO NaNs, that's suspicious
        # (values may have been coerced to 0 during assembly).
        if "probe_count" in df.columns:
            zero_probe_rows = df["probe_count"] == 0
            for col in ["probe_mean_speed_mps", "probe_min_distance_to_stopline_m", "probe_max_distance_to_stopline_m"]:
                if col not in df.columns:
                    continue
                if zero_probe_rows.any():
                    still_populated = zero_probe_rows & df[col].notna()
                    if still_populated.any():
                        warn(
                            f"{layer_name}/{SPLIT_FILES[role]}: '{col}' has {int(still_populated.sum())} "
                            f"non-NaN value(s) where probe_count == 0 -- possible fill-in during assembly."
                        )
    return nan_counts


def compare_nan_counts_across_layers(layer_nan_counts: Dict[str, Dict[str, Dict[str, int]]]) -> None:
    if "layer1" not in layer_nan_counts or "layer2_p11" not in layer_nan_counts:
        return
    for role in SPLIT_FILES:
        l1 = layer_nan_counts["layer1"].get(role, {})
        l2 = layer_nan_counts["layer2_p11"].get(role, {})
        shared_cols = set(l1) & set(l2)
        for col in shared_cols:
            if l1[col] != l2[col]:
                # informational only -- not required to match
                pass


# ============================================================================
# 8. Data-type sanity
# ============================================================================

NUMERIC_LIKELY_COLUMNS = [
    "timestamp", "visible_vehicle_count", "visible_mean_speed_mps",
    "visible_queue_count", "visible_queue_length_m", "camera_range_m",
    "probe_count", "probe_mean_speed_mps",
    "probe_min_distance_to_stopline_m", "probe_max_distance_to_stopline_m",
    "true_queue_length_m", "true_queue_length_future_m", "prediction_horizon_s",
]

_TRUE_STRINGS = {"true", "1", "1.0", "yes"}
_FALSE_STRINGS = {"false", "0", "0.0", "no"}


def check_data_types(layer_name: str, splits: Dict[str, dict]) -> None:
    for role, entry in splits.items():
        df = entry["df"]
        if df is None or df.empty:
            continue

        for col in NUMERIC_LIKELY_COLUMNS:
            if col not in df.columns:
                continue
            original = df[col]
            if pd.api.types.is_numeric_dtype(original):
                continue
            coerced = pd.to_numeric(original, errors="coerce")
            newly_nan = original.notna() & coerced.isna()
            if newly_nan.any():
                bad = sorted(set(original[newly_nan].astype(str)))[:5]
                fail(
                    f"{layer_name}/{SPLIT_FILES[role]}: column '{col}' expected numeric has "
                    f"{int(newly_nan.sum())} non-numeric value(s), e.g. {bad} (not coerced, reported only)."
                )

        if "true_queue_beyond_camera" in df.columns:
            col = df["true_queue_beyond_camera"]
            if col.dtype == bool:
                continue
            unrecognized = col.dropna().map(
                lambda v: str(v).strip().lower() not in (_TRUE_STRINGS | _FALSE_STRINGS)
            )
            if unrecognized.any():
                bad = sorted(set(col[col.notna()][unrecognized].astype(str)))[:5]
                fail(
                    f"{layer_name}/{SPLIT_FILES[role]}: 'true_queue_beyond_camera' has "
                    f"{int(unrecognized.sum())} value(s) not interpretable as boolean, e.g. {bad}."
                )


# ============================================================================
# 9. Manifest consistency
# ============================================================================

def check_manifest_consistency(layer_name: str, manifest: Optional[dict], splits: Dict[str, dict]) -> None:
    if manifest is None:
        return
    manifest_splits = manifest.get("splits", {})
    for role, entry in splits.items():
        df = entry["df"]
        m = manifest_splits.get(role)
        if m is None or df is None:
            continue

        if "n_rows" in m and m["n_rows"] != len(df):
            fail(f"{layer_name}/{SPLIT_FILES[role]}: manifest n_rows={m['n_rows']} != actual CSV rows={len(df)}")

        if "scenario_id" in df.columns:
            actual_ids = set(df["scenario_id"].unique())
            if "n_scenarios" in m and m["n_scenarios"] != len(actual_ids):
                fail(
                    f"{layer_name}/{SPLIT_FILES[role]}: manifest n_scenarios={m['n_scenarios']} != "
                    f"actual unique scenario_ids={len(actual_ids)}"
                )
            if "scenarios" in m:
                manifest_ids = set(m["scenarios"])
                if manifest_ids != actual_ids:
                    fail(
                        f"{layer_name}/{SPLIT_FILES[role]}: manifest scenario list does not match CSV -- "
                        f"only in manifest: {sorted(manifest_ids - actual_ids)}, "
                        f"only in CSV: {sorted(actual_ids - manifest_ids)}"
                    )


# ============================================================================
# 10. Layer comparison
# ============================================================================

def check_layer_comparison(
    scenario_ids_by_layer: Dict[str, Dict[str, set]],
    feature_cols_by_layer: Dict[str, Dict[str, List[str]]],
) -> None:
    if "layer1" not in scenario_ids_by_layer or "layer2_p11" not in scenario_ids_by_layer:
        return

    for role in SPLIT_FILES:
        l1_ids = scenario_ids_by_layer["layer1"].get(role, set())
        l2_ids = scenario_ids_by_layer["layer2_p11"].get(role, set())
        if l1_ids != l2_ids:
            fail(
                f"layer1 vs layer2_p11 ({role}): scenario membership mismatch -- "
                f"only in layer1: {sorted(l1_ids - l2_ids)}, only in layer2: {sorted(l2_ids - l1_ids)}"
            )

    for role in SPLIT_FILES:
        l1_features = set(feature_cols_by_layer.get("layer1", {}).get(role, []))
        l2_features = set(feature_cols_by_layer.get("layer2_p11", {}).get(role, []))
        if not l1_features or not l2_features:
            continue
        missing_in_l2 = l1_features - l2_features
        if missing_in_l2:
            fail(f"layer2_p11 ({role}): missing Layer-1-inherited feature column(s) present in layer1: {sorted(missing_in_l2)}")


# ============================================================================
# Orchestration
# ============================================================================

def audit_layer(layer_name: str, layer_dir: Path) -> dict:
    print(f"\n=== {layer_name} ({layer_dir}) ===")
    if not layer_dir.exists():
        fail(f"{layer_name}: directory does not exist: {layer_dir}")
        return {"scenario_ids_by_role": {}, "feature_cols_by_role": {}}

    data = load_layer(layer_dir)
    manifest = data["manifest"]
    splits = data["splits"]

    check_schema(layer_name, splits)
    check_feature_label_separation(layer_name, manifest, splits)
    scenario_ids_by_role = check_split_integrity(layer_name, splits)
    report_row_counts(layer_name, splits)
    check_key_integrity(layer_name, splits)
    nan_counts = report_nan_counts(layer_name, splits)
    check_data_types(layer_name, splits)
    check_manifest_consistency(layer_name, manifest, splits)

    feature_cols_by_role = {}
    if manifest is not None:
        for role, m in manifest.get("splits", {}).items():
            feature_cols_by_role[role] = m.get("feature_columns", [])

    return {
        "scenario_ids_by_role": scenario_ids_by_role,
        "feature_cols_by_role": feature_cols_by_role,
        "nan_counts": nan_counts,
    }


def main() -> None:
    layer_results = {}
    for layer_name in LAYERS:
        layer_dir = ASSEMBLED_DIR / layer_name
        layer_results[layer_name] = audit_layer(layer_name, layer_dir)

    print("\n=== Cross-layer checks ===")
    check_layer_comparison(
        {k: v["scenario_ids_by_role"] for k, v in layer_results.items()},
        {k: v["feature_cols_by_role"] for k, v in layer_results.items()},
    )
    compare_nan_counts_across_layers({k: v["nan_counts"] for k, v in layer_results.items()})

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if WARNINGS:
        print(f"\nWARNINGS ({len(WARNINGS)}):")
        for w in WARNINGS:
            print(f"  [WARNING] {w}")
    if FAILS:
        print(f"\nFAILURES ({len(FAILS)}):")
        for f in FAILS:
            print(f"  [FAIL] {f}")

    if not FAILS and not WARNINGS:
        print("\nOVERALL: PASS -- no issues found.")
    elif not FAILS:
        print(f"\nOVERALL: PASS WITH WARNINGS ({len(WARNINGS)} warning(s)).")
    else:
        print(f"\nOVERALL: FAIL ({len(FAILS)} failure(s), {len(WARNINGS)} warning(s)).")

    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
