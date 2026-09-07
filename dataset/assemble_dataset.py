"""
assemble_dataset.py
====================
ASTRID Prototype -- Dataset Assembler.

RESPONSIBILITY:
    Combine already-generated, already-QA-audited per-scenario
    features_*.csv / labels_*.csv into split ML-ready datasets
    (train / validation / test / ood), split at the SCENARIO level using
    scenario.json's own "split" field as the sole source of truth.

    Never recomputes a feature or label value, never imputes, never
    re-splits at the row level, never invents a scenario assignment.

Reads (per scenario, already produced by feature_builder.py):
    features/features_layer1.csv, features/labels_layer1.csv
    features/features_layer2_p{TAG}.csv, features/labels_layer2_p{TAG}.csv
    scenario.json (for split / design_method / scenario_id -- authoritative)

Writes:
    dataset/assembled/layer1/{train,validation,test,ood}.csv + manifest.json
    dataset/assembled/layer2_p{TAG}/{train,validation,test,ood}.csv + manifest.json

Run:
    python dataset/assemble_dataset.py
    python dataset/assemble_dataset.py --layer layer2 --penetration 0.11
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_builder as fb  # noqa: E402
from trajectory_utils import SAMPLING_INTERVAL_S  # noqa: E402

SCENARIOS_DIR = fb.SCENARIOS_DIR
DELTA_WINDOW_S = fb.DELTA_WINDOW_S
FORBIDDEN_GROUND_TRUTH_COLUMNS = fb.FORBIDDEN_GROUND_TRUTH_COLUMNS
DEFAULT_PENETRATION_RATE = fb.DEFAULT_PENETRATION_RATE  # 0.11, matches feature_builder/observation_assembler

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "assembled"

KEY_COLUMNS = ["timestamp", "approach_edge"]

# Metadata/provenance columns. Layer 2's assembled_observations_p{tag}.csv
# (written by observation_assembler.py) already carries these and
# feature_builder.py passes them straight through; Layer 1 has none of
# them natively (camera_timeseries.csv only) -- added here from
# scenario.json so both layers carry identical, consistent provenance.
METADATA_COLUMNS = ["scenario_id", "split", "design_method", "gps_penetration_rate_requested"]

# scenario.json's own split values (authoritative, from scenario_builder.py)
# mapped to the requested output filenames.
SPLIT_ROLES = ["train", "val", "test", "ood"]
SPLIT_FILE_NAMES = {"train": "train.csv", "val": "validation.csv", "test": "test.csv", "ood": "ood.csv"}

KNOWN_LABEL_COLUMNS = [
    "true_queue_length_m", "true_queue_beyond_camera",
    "true_queue_length_future_m", "prediction_horizon_s",
]


def _tag_for(penetration_rate: float) -> str:
    return f"p{int(round(penetration_rate * 100)):02d}"


# ============================================================================
# Discovery / split membership
# ============================================================================

def discover_scenarios() -> List[Path]:
    if not SCENARIOS_DIR.exists():
        return []
    return sorted(p for p in SCENARIOS_DIR.glob("scenario_*") if p.is_dir())


def get_scenario_split(scenario_dir: Path) -> dict:
    """Read the AUTHORITATIVE split/design_method/scenario_id straight
    from scenario.json -- never re-derived or re-decided here."""
    path = scenario_dir / "scenario.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} -- run sumo/scenario_builder.py first.")
    with open(path, "r", encoding="utf-8") as f:
        scenario = json.load(f)
    split = scenario.get("split")
    if split not in SPLIT_ROLES:
        raise ValueError(f"{scenario_dir.name}: scenario.json split={split!r} is not one of {SPLIT_ROLES}.")
    return {
        "scenario_id": scenario["scenario_id"],
        "split": split,
        "design_method": scenario.get("design_method", ""),
    }


def group_scenarios_by_split(scenario_dirs: List[Path]) -> Dict[str, List[Path]]:
    """This IS the split-generation mechanism -- every role's membership
    traces directly to scenario_builder.py's already-written scenario.json
    'split' field. No second, independent split logic exists here."""
    groups: Dict[str, List[Path]] = {role: [] for role in SPLIT_ROLES}
    seen_ids = set()
    for scenario_dir in scenario_dirs:
        info = get_scenario_split(scenario_dir)
        sid = info["scenario_id"]
        if sid in seen_ids:
            raise ValueError(f"Duplicate scenario_id encountered: {sid}")
        seen_ids.add(sid)
        groups[info["split"]].append(scenario_dir)
    return groups


def validate_scenario_membership(role_to_scenarios: Dict[str, List[Path]]) -> None:
    """Defensive assertion: no scenario directory assigned to more than
    one role. Should be structurally impossible given
    group_scenarios_by_split(), but cheap to check directly."""
    seen: Dict[str, str] = {}
    for role, dirs in role_to_scenarios.items():
        for d in dirs:
            if d.name in seen:
                raise ValueError(f"Scenario '{d.name}' assigned to both '{seen[d.name]}' and '{role}'.")
            seen[d.name] = role


# ============================================================================
# Per-scenario loading
# ============================================================================

def _output_tag(layer: str, tag: Optional[str]) -> str:
    return f"{layer}_{tag}" if layer == "layer2" else layer


def find_feature_label_files(scenario_dir: Path, layer: str, tag: Optional[str]) -> "tuple[Path, Path]":
    out_tag = _output_tag(layer, tag)
    features_dir = scenario_dir / "features"
    return (
        features_dir / f"features_{out_tag}.csv",
        features_dir / f"labels_{out_tag}.csv",
    )


def validate_feature_label_alignment(features_df: pd.DataFrame, labels_df: pd.DataFrame, scenario_id: str) -> None:
    """Minimal alignment guard -- NOT a second QA framework. Just enough to
    refuse silently assembling corrupt data: key columns present, key sets
    identical, no duplicate keys on either side."""
    for name, df in (("features", features_df), ("labels", labels_df)):
        missing = [c for c in KEY_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{scenario_id}: {name} file is missing key column(s) {missing}.")
        if df.duplicated(subset=KEY_COLUMNS).any():
            raise ValueError(f"{scenario_id}: {name} file has duplicate {KEY_COLUMNS} keys.")

    f_keys = set(map(tuple, features_df[KEY_COLUMNS].itertuples(index=False, name=None)))
    l_keys = set(map(tuple, labels_df[KEY_COLUMNS].itertuples(index=False, name=None)))
    if f_keys != l_keys:
        raise ValueError(
            f"{scenario_id}: feature/label key mismatch -- "
            f"{len(f_keys - l_keys)} key(s) only in features, {len(l_keys - f_keys)} only in labels."
        )


def load_scenario_dataset(scenario_dir: Path, layer: str, tag: Optional[str]) -> pd.DataFrame:
    """Load one scenario's already-generated features + labels, merge them
    on (timestamp, approach_edge), and ensure provenance metadata columns
    are present. Never recomputes any feature or label value."""
    features_path, labels_path = find_feature_label_files(scenario_dir, layer, tag)
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path} -- run feature_builder.py for this scenario/layer first.")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing {labels_path} -- run feature_builder.py for this scenario/layer first.")

    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)
    info = get_scenario_split(scenario_dir)

    for col, value in (("scenario_id", info["scenario_id"]), ("split", info["split"]), ("design_method", info["design_method"])):
        if col not in features_df.columns:
            features_df.insert(0, col, value)
        else:
            mismatched = features_df[col].astype(str) != str(value)
            if mismatched.any():
                raise ValueError(f"{scenario_dir.name}: features column '{col}' disagrees with scenario.json.")

    validate_feature_label_alignment(features_df, labels_df, scenario_dir.name)

    overlap = (set(features_df.columns) & set(labels_df.columns)) - set(KEY_COLUMNS)
    if overlap:
        raise ValueError(f"{scenario_dir.name}: features/labels share unexpected column(s): {sorted(overlap)}")

    combined = features_df.merge(labels_df, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(combined) != len(features_df) or len(combined) != len(labels_df):
        raise ValueError(
            f"{scenario_dir.name}: feature/label merge lost or duplicated row(s) "
            f"(features={len(features_df)}, labels={len(labels_df)}, merged={len(combined)})."
        )
    return combined


# ============================================================================
# Column classification
# ============================================================================

def identify_metadata_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in METADATA_COLUMNS if c in df.columns]


def identify_label_columns(df: pd.DataFrame, metadata_cols: List[str]) -> List[str]:
    return [c for c in KNOWN_LABEL_COLUMNS if c in df.columns]


def identify_feature_columns(df: pd.DataFrame, label_cols: List[str], metadata_cols: List[str]) -> List[str]:
    """Candidate ML feature columns = everything except keys, labels, and
    metadata. The leakage check runs ONLY against this candidate set --
    not the full dataframe -- since label columns like
    true_queue_length_m are themselves in FORBIDDEN_GROUND_TRUTH_COLUMNS
    (they're forbidden as FEATURES, not forbidden from existing in the
    assembled file at all)."""
    excluded = set(KEY_COLUMNS) | set(label_cols) | set(metadata_cols)
    candidate_cols = [c for c in df.columns if c not in excluded]

    leaked = FORBIDDEN_GROUND_TRUTH_COLUMNS.intersection(candidate_cols)
    if leaked:
        raise ValueError(f"Forbidden ground-truth-shaped column(s) present in assembled feature columns: {sorted(leaked)}")

    return candidate_cols


# ============================================================================
# Assembly
# ============================================================================

def assemble_split(scenario_dirs: List[Path], layer: str, tag: Optional[str]) -> "tuple[pd.DataFrame, List[str]]":
    frames, contributing_ids = [], []
    for scenario_dir in scenario_dirs:
        frames.append(load_scenario_dataset(scenario_dir, layer, tag))
        contributing_ids.append(scenario_dir.name)
    if not frames:
        return pd.DataFrame(), contributing_ids
    return pd.concat(frames, axis=0, ignore_index=True), contributing_ids


# ============================================================================
# Writing
# ============================================================================

def write_assembled_dataset(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def write_manifest(out_path: Path, layer: str, tag: Optional[str], split_info: Dict[str, dict]) -> None:
    manifest = {
        "layer": layer,
        "penetration_tag": tag,
        "sampling_interval_s": SAMPLING_INTERVAL_S,
        "delta_window_s": DELTA_WINDOW_S,
        "splits": split_info,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)


# ============================================================================
# Orchestration for one (layer, tag)
# ============================================================================

def assemble_layer(layer: str, tag: Optional[str], output_dir: Path) -> None:
    scenario_dirs = discover_scenarios()
    if not scenario_dirs:
        print(f"ERROR: no scenarios found under {SCENARIOS_DIR}")
        return

    role_to_scenarios = group_scenarios_by_split(scenario_dirs)
    validate_scenario_membership(role_to_scenarios)

    layer_dir_name = layer if layer == "layer1" else f"{layer}_{tag}"
    layer_out_dir = output_dir / layer_dir_name

    split_info: Dict[str, dict] = {}
    for role in SPLIT_ROLES:
        scenario_dirs_for_role = role_to_scenarios[role]
        if not scenario_dirs_for_role:
            split_info[role] = {"n_scenarios": 0, "n_rows": 0, "scenarios": [],
                                 "feature_columns": [], "label_columns": [], "metadata_columns": []}
            print(f"{layer_dir_name} :: {role}: no scenarios found -- skipping.")
            continue

        combined, contributing_ids = assemble_split(scenario_dirs_for_role, layer, tag)

        metadata_cols = identify_metadata_columns(combined)
        label_cols = identify_label_columns(combined, metadata_cols)
        feature_cols = identify_feature_columns(combined, label_cols, metadata_cols)
        horizons = (sorted(combined["prediction_horizon_s"].dropna().unique().tolist())
                    if "prediction_horizon_s" in combined.columns else [])

        out_path = layer_out_dir / SPLIT_FILE_NAMES[role]
        write_assembled_dataset(combined, out_path)

        split_info[role] = {
            "n_scenarios": len(contributing_ids),
            "n_rows": len(combined),
            "scenarios": contributing_ids,
            "key_columns": KEY_COLUMNS,
            "feature_columns": feature_cols,
            "label_columns": label_cols,
            "metadata_columns": metadata_cols,
            "prediction_horizon_s": horizons,
            "output_file": str(out_path),
        }
        print(f"{layer_dir_name} :: {role}: {len(contributing_ids)} scenario(s), {len(combined)} row(s) -> {out_path}")

    write_manifest(layer_out_dir / "manifest.json", layer, tag, split_info)
    print(f"{layer_dir_name}: manifest written to {layer_out_dir / 'manifest.json'}")


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble per-scenario ASTRID features/labels into split ML-ready datasets.")
    parser.add_argument("--layer", type=str, default=None, choices=["layer1", "layer2"], help="Assemble only this layer. Default: both.")
    parser.add_argument("--penetration", type=float, default=DEFAULT_PENETRATION_RATE, help=f"Layer 2 GPS penetration to assemble (default {DEFAULT_PENETRATION_RATE}).")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Root output directory (default dataset/assembled).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    tag = _tag_for(args.penetration)

    for layer in ([args.layer] if args.layer else ["layer1", "layer2"]):
        assemble_layer(layer, tag if layer == "layer2" else None, output_dir)


if __name__ == "__main__":
    main()