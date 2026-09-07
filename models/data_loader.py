"""
data_loader.py
================
Reusable, manifest-driven loader for the assembled ASTRID datasets.

Responsibilities
-----------------
* Load train / validation / test / ood splits for a given layer
  (layer1 or layer2_p11).
* Treat each split's manifest.json entry as the SOLE authority for which
  columns are features, which are labels, and which are metadata/keys.
* Return features (X) and target (y) as separate objects so that model
  code never has direct access to a combined dataframe containing label
  or metadata columns.
* Never modify, impute, re-split, or otherwise alter the underlying CSV
  data. Missing values are preserved exactly as assembled.

This module intentionally does nothing "clever" -- it is meant to be the
single, boring, correct way every future model implementation gets its
data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from experiment_config import Layer, Split, TARGET_COLUMN, DEFAULT_DATA_ROOT

# ---------------------------------------------------------------------------
# Manifest <-> filesystem naming
# ---------------------------------------------------------------------------

# The manifest.json "splits" dict is keyed by these role names...
_MANIFEST_SPLIT_KEY = {
    Split.TRAIN: "train",
    Split.VALIDATION: "val",
    Split.TEST: "test",
    Split.OOD: "ood",
}

# ...while the CSV files on disk use these names.
_SPLIT_FILE_NAME = {
    Split.TRAIN: "train.csv",
    Split.VALIDATION: "validation.csv",
    Split.TEST: "test.csv",
    Split.OOD: "ood.csv",
}

_KEY_COLUMNS = ["timestamp", "approach_edge"]

# Values pandas.read_csv leaves as an `object` dtype column when a
# boolean-like CSV column (e.g. "True"/"False") also contains blanks
# (NaN). These map 1:1 to numeric 1.0 / 0.0; anything not in this map is
# left untouched so no non-boolean data is silently altered.
_BOOLEAN_LIKE_TO_NUMERIC = {
    True: 1.0,
    False: 0.0,
    "True": 1.0,
    "False": 0.0,
    "true": 1.0,
    "false": 0.0,
}


class ManifestError(RuntimeError):
    """Raised when a manifest.json is missing, malformed, or inconsistent
    with the CSV it describes."""


class DataLeakageGuardError(RuntimeError):
    """Raised when the manifest itself would allow a label/metadata/key
    column to be treated as a model feature."""


# ---------------------------------------------------------------------------
# dtype normalization (NOT imputation -- values, not missingness, change)
# ---------------------------------------------------------------------------

def _normalize_boolean_like_feature_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Convert feature columns that pandas has loaded as `object` dtype
    purely because they mix booleans with blank/NaN values (e.g.
    Layer 2's `is_green_for_approach`) into numeric 1.0/0.0.

    This is a dtype fix, not imputation or feature engineering:
    * existing NaNs are preserved exactly as NaN (never filled in)
    * True -> 1.0, False -> 0.0 (and their string equivalents, in case
      the CSV serialized them as text)
    * any column whose non-null values are NOT entirely boolean-like is
      left completely untouched
    * no column is added, removed, or renamed; only in-place dtype/value
      normalization of existing feature columns

    This exists because some downstream models (e.g. XGBoost) reject
    `object`-dtype columns outright, even though the underlying data is
    simple boolean-with-missing-values.
    """
    for col in X.columns:
        series = X[col]
        if series.dtype != object:
            continue

        non_null = series.dropna()
        if non_null.empty:
            continue

        if not set(non_null.unique()).issubset(_BOOLEAN_LIKE_TO_NUMERIC.keys()):
            continue

        X[col] = series.map(
            lambda v: _BOOLEAN_LIKE_TO_NUMERIC[v] if pd.notna(v) else np.nan
        ).astype(float)

    return X


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def _layer_dir(layer: Layer, data_root: Path) -> Path:
    return Path(data_root) / layer.value


def load_manifest(layer: Layer, data_root: Path = DEFAULT_DATA_ROOT) -> dict:
    """Load and parse manifest.json for a layer. This is the authoritative
    description of every split's feature/label/metadata columns."""
    manifest_path = _layer_dir(layer, data_root) / "manifest.json"
    if not manifest_path.exists():
        raise ManifestError(f"manifest.json not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _split_manifest_entry(manifest: dict, split: Split) -> dict:
    key = _MANIFEST_SPLIT_KEY[split]
    splits = manifest.get("splits", {})
    if key not in splits:
        raise ManifestError(f"manifest.json has no entry for split '{key}'")
    return splits[key]


def get_feature_columns(
    layer: Layer, split: Split, data_root: Path = DEFAULT_DATA_ROOT
) -> List[str]:
    """The authoritative feature-column list for one (layer, split), taken
    directly from manifest.json."""
    manifest = load_manifest(layer, data_root)
    entry = _split_manifest_entry(manifest, split)
    cols = entry.get("feature_columns")
    if not cols:
        raise ManifestError(
            f"manifest.json declares no feature_columns for split '{split.value}'"
        )
    return list(cols)


# ---------------------------------------------------------------------------
# Split data container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitData:
    """Everything about one (layer, split), with features and target kept
    strictly separate from metadata/keys.

    Attributes
    ----------
    X : feature matrix only. Columns are exactly manifest's
        feature_columns, in manifest order. Never contains labels,
        metadata, or key columns.
    y : the target column (true_queue_length_m), as a Series aligned to X.
    metadata : the metadata columns declared in the manifest
        (scenario_id, split, design_method, ...), aligned to X/y by
        position. Provided so callers can identify which scenario each
        row belongs to without it ever entering the feature matrix.
    keys : the key columns (timestamp, approach_edge), aligned to X/y by
        position.
    layer : which layer this split was loaded from.
    split : which split (train/validation/test/ood) this is.
    feature_columns : the exact feature column list used (== list(X.columns)).
    label_columns : all label columns declared for this split in the
        manifest (not just the target column actually used).
    """

    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame
    keys: pd.DataFrame
    layer: Layer
    split: Split
    feature_columns: List[str]
    label_columns: List[str]

    def __len__(self) -> int:
        return len(self.X)

    def scenario_ids(self) -> List[str]:
        """Convenience: unique scenario_ids contributing to this split."""
        if "scenario_id" not in self.metadata.columns:
            return []
        return sorted(self.metadata["scenario_id"].unique().tolist())


def load_split(
    layer: Layer,
    split: Split,
    data_root: Path = DEFAULT_DATA_ROOT,
    target_column: str = TARGET_COLUMN,
) -> SplitData:
    """Load one (layer, split) using manifest.json as the authoritative
    definition of feature/label/metadata columns.

    Guarantees:
    * X contains ONLY the manifest's declared feature_columns.
    * y is exactly target_column, untouched (no imputation, no dtype
      coercion beyond what pandas.read_csv does by default).
    * Row order and values are preserved exactly as assembled.
    * Rows/scenarios can be traced via SplitData.metadata / .keys.
    * Boolean-like feature columns that pandas loaded as `object` dtype
      (e.g. "True"/"False" mixed with blanks) are normalized to numeric
      1.0/0.0, with existing NaNs preserved exactly -- this is a dtype
      fix only, never imputation, and never applied to metadata/key/
      label columns.
    """
    manifest = load_manifest(layer, data_root)
    entry = _split_manifest_entry(manifest, split)

    feature_cols: List[str] = list(entry.get("feature_columns", []))
    label_cols: List[str] = list(entry.get("label_columns", []))
    metadata_cols: List[str] = list(entry.get("metadata_columns", []))

    if target_column not in label_cols:
        raise ManifestError(
            f"target column '{target_column}' is not listed as a label_column "
            f"for {layer.value}/{split.value} in manifest.json"
        )

    overlap = set(feature_cols) & (
        set(label_cols) | set(metadata_cols) | set(_KEY_COLUMNS)
    )
    if overlap:
        raise DataLeakageGuardError(
            f"manifest.json declares column(s) as BOTH feature and "
            f"label/metadata/key for {layer.value}/{split.value}: {sorted(overlap)}"
        )

    csv_path = _layer_dir(layer, data_root) / _SPLIT_FILE_NAME[split]
    if not csv_path.exists():
        raise FileNotFoundError(f"Assembled split CSV not found: {csv_path}")

    # Read exactly as assembled -- no na_values overrides, no dtype
    # coercion, no imputation.
    df = pd.read_csv(csv_path)

    required = set(feature_cols) | {target_column} | set(metadata_cols) | set(_KEY_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ManifestError(
            f"{csv_path.name}: missing column(s) declared in manifest.json: {sorted(missing)}"
        )

    X = df.loc[:, feature_cols].copy()
    X = _normalize_boolean_like_feature_columns(X)

    y = df.loc[:, target_column].copy()
    metadata = df.loc[:, [c for c in metadata_cols if c in df.columns]].copy()
    keys = df.loc[:, [c for c in _KEY_COLUMNS if c in df.columns]].copy()

    return SplitData(
        X=X,
        y=y,
        metadata=metadata,
        keys=keys,
        layer=layer,
        split=split,
        feature_columns=feature_cols,
        label_columns=label_cols,
    )


# ---------------------------------------------------------------------------
# High-level loader
# ---------------------------------------------------------------------------

class DatasetLoader:
    """Convenience wrapper around load_split() for loading every split of
    a layer. Stateless aside from `layer`/`data_root`/`target_column` --
    holds no cached dataframes, so data is always read fresh from disk and
    is never mutated in place.
    """

    def __init__(
        self,
        layer: Layer,
        data_root: Path = DEFAULT_DATA_ROOT,
        target_column: str = TARGET_COLUMN,
    ):
        self.layer = layer
        self.data_root = Path(data_root)
        self.target_column = target_column

    def load(self, split: Split) -> SplitData:
        return load_split(self.layer, split, self.data_root, self.target_column)

    def load_all(self) -> Dict[Split, SplitData]:
        """Load train, validation, test, and ood in one call."""
        return {split: self.load(split) for split in Split}