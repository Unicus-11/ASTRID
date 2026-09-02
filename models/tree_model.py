"""
tree_model.py
====================
ASTRID Prototype -- Tree Baseline (Layer 1, Step 3)

Random Forest trained on Layer 1 camera-only features, predicting
true_queue_length_m. Must beat persistence.py's baseline on val AND
test -- if it doesn't, STOP and fix the dataset, don't reach for a
bigger model.

Validated by SCENARIO, not by row: train/val/test/ood were assigned
per-scenario back in scenario_builder.py, and assemble_dataset.py never
splits one scenario's rows across files. Using these three files as-is
already gives scenario-grouped validation -- nothing extra needed here.

Reads:
    training_data/{layer}/train_features.csv, train_labels.csv
    training_data/{layer}/val_features.csv,   val_labels.csv
    training_data/{layer}/test_features.csv,  test_labels.csv
    training_data/{layer}/persistence_report.json   (for the comparison gate)

Writes:
    training_data/{layer}/tree_model.joblib
    training_data/{layer}/tree_report.json

Run:
    python models/tree_model.py --layer layer1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DATA_DIR = PROJECT_ROOT / "training_data"

# Index / constant columns -- not predictive features.
NON_FEATURE_COLUMNS = ["scenario_id", "timestamp", "camera_range_m"]
LABEL_COLUMNS = ["true_queue_length_m", "true_queue_beyond_camera",
                  "true_queue_length_future_m", "prediction_horizon_s"]


def load_split(layer_dir: Path, split: str):
    f_path = layer_dir / f"{split}_features.csv"
    l_path = layer_dir / f"{split}_labels.csv"
    if not f_path.exists() or not l_path.exists():
        return None
    try:
        features = pd.read_csv(f_path)
        labels = pd.read_csv(l_path)
    except pd.errors.EmptyDataError:
        return None  # split was assembled but has no scenarios in it yet
    if features.empty or labels.empty:
        return None
    return features.merge(labels, on=["scenario_id", "timestamp", "approach_edge"], how="inner")


def build_xy(df: pd.DataFrame, feature_columns: Optional[List[str]] = None):
    work = df.copy()
    work["queue_reaches_camera_edge"] = work["queue_reaches_camera_edge"].astype(int)
    work = pd.get_dummies(work, columns=["approach_edge"], prefix="approach")

    y = work["true_queue_length_m"]
    X = work.drop(columns=[c for c in NON_FEATURE_COLUMNS + LABEL_COLUMNS if c in work.columns])

    if feature_columns is not None:
        # Align val/test to the train-time feature set (e.g. an approach
        # one-hot column missing because that approach never appears here).
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_columns]

    before = len(X)
    valid = X.notna().all(axis=1) & y.notna()
    dropped = before - int(valid.sum())
    return X[valid], y[valid], list(X.columns), dropped


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    error = pred - y
    return {
        "rows": len(y),
        "mae_m": round(float(error.abs().mean()), 2),
        "rmse_m": round(float(np.sqrt((error ** 2).mean())), 2),
    }


def resolve_tag(layer: str, penetration) -> str:
    """Mirrors dataset/assemble_dataset.py's naming exactly."""
    if layer == "layer1":
        return "layer1"
    if layer == "layer2":
        if penetration is None:
            raise ValueError("--penetration is required for --layer layer2")
        return f"layer2_p{int(round(penetration * 100)):02d}"
    raise ValueError(f"Unknown layer '{layer}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + evaluate a Random Forest baseline.")
    parser.add_argument("--layer", type=str, default="layer1", choices=["layer1", "layer2"])
    parser.add_argument("--penetration", type=float, default=None, help="Required for --layer layer2.")
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args()

    tag = resolve_tag(args.layer, args.penetration)
    layer_dir = TRAINING_DATA_DIR / tag
    train_df = load_split(layer_dir, "train")
    val_df = load_split(layer_dir, "val")
    test_df = load_split(layer_dir, "test")

    if train_df is None or len(train_df) == 0:
        raise RuntimeError(f"No training data in {layer_dir} -- run dataset/assemble_dataset.py first.")

    X_train, y_train, feature_columns, dropped_train = build_xy(train_df)
    print(f"Train: {len(X_train)} rows ({dropped_train} dropped for missing values)")
    print(f"Features ({len(feature_columns)}): {feature_columns}")

    model = RandomForestRegressor(n_estimators=args.n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    report = {"tag": tag, "n_estimators": args.n_estimators, "feature_columns": feature_columns}
    report["train"] = evaluate(model, X_train, y_train)

    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        if split_df is None or len(split_df) == 0:
            report[split_name] = {"rows": 0, "note": "split empty or not assembled"}
            continue
        X, y, _, dropped = build_xy(split_df, feature_columns)
        report[split_name] = evaluate(model, X, y)
        report[split_name]["dropped_for_missing_values"] = dropped

    # -- The gate: must beat persistence --
    persistence_path = layer_dir / "persistence_report.json"
    if persistence_path.exists():
        with open(persistence_path, "r", encoding="utf-8") as f:
            persistence = json.load(f)
        report["persistence_comparison"] = {}
        for split_name in ["val", "test"]:
            if report.get(split_name, {}).get("rows") and persistence.get(split_name, {}).get("rows"):
                tree_mae = report[split_name]["mae_m"]
                pers_mae = persistence[split_name]["mae_m"]
                report["persistence_comparison"][split_name] = {
                    "tree_mae_m": tree_mae, "persistence_mae_m": pers_mae,
                    "tree_beats_persistence": tree_mae < pers_mae,
                }
    else:
        print("NOTE: no persistence_report.json -- run models/persistence.py first to enable the comparison gate.")

    layer_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, layer_dir / "tree_model.joblib")
    with open(layer_dir / "tree_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 70)
    print(f"TREE MODEL -- {tag}")
    print("=" * 70)
    for split_name in ["train", "val", "test"]:
        r = report.get(split_name, {})
        if r.get("rows"):
            print(f"{split_name:6}: MAE={r['mae_m']:6.2f}m  RMSE={r['rmse_m']:6.2f}m  (n={r['rows']})")
        else:
            print(f"{split_name:6}: (empty)")

    if "persistence_comparison" in report:
        print()
        for split_name, comp in report["persistence_comparison"].items():
            verdict = "BEATS persistence" if comp["tree_beats_persistence"] else "DOES NOT beat persistence -- STOP, check the data"
            print(f"{split_name}: tree MAE={comp['tree_mae_m']}m vs persistence MAE={comp['persistence_mae_m']}m -- {verdict}")

    print(f"\nSaved model : {layer_dir / 'tree_model.joblib'}")
    print(f"Saved report: {layer_dir / 'tree_report.json'}")


if __name__ == "__main__":
    main()
    