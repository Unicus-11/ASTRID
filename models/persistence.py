"""
persistence.py
====================
ASTRID Prototype -- Persistence Baseline (Layer 1, Step 2)

THE DUMB GUESS: predicted queue = what the camera already sees
(visible_queue_length_m), used directly as the prediction for
true_queue_length_m. No fitting, no training. This is the number every
real model has to beat -- see tree_model.py's gate check.

Reads:
    training_data/{layer}/{split}_features.csv
    training_data/{layer}/{split}_labels.csv

Writes:
    training_data/{layer}/persistence_report.json

Run:
    python models/persistence.py --layer layer1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DATA_DIR = PROJECT_ROOT / "training_data"

SPLITS_TO_EVALUATE = ["val", "test", "ood"]  # never evaluate a baseline against train


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


def evaluate_persistence(df: pd.DataFrame) -> dict:
    predicted = df["visible_queue_length_m"]
    actual = df["true_queue_length_m"]
    error = predicted - actual

    mae = float(error.abs().mean())
    rmse = float(np.sqrt((error ** 2).mean()))

    # The case that actually matters: how wrong is the dumb guess when the
    # true queue extends past the camera (the camera is guaranteed wrong there)?
    beyond = df["true_queue_beyond_camera"] == True  # noqa: E712
    mae_beyond_camera = float(error[beyond].abs().mean()) if beyond.any() else None

    return {
        "rows": len(df),
        "mae_m": round(mae, 2),
        "rmse_m": round(rmse, 2),
        "rows_with_queue_beyond_camera": int(beyond.sum()),
        "mae_m_when_queue_beyond_camera": round(mae_beyond_camera, 2) if mae_beyond_camera is not None else None,
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
    parser = argparse.ArgumentParser(description="Evaluate the persistence baseline (predicted = visible queue).")
    parser.add_argument("--layer", type=str, default="layer1", choices=["layer1", "layer2"])
    parser.add_argument("--penetration", type=float, default=None, help="Required for --layer layer2.")
    args = parser.parse_args()

    tag = resolve_tag(args.layer, args.penetration)
    layer_dir = TRAINING_DATA_DIR / tag
    if not layer_dir.exists():
        raise FileNotFoundError(f"Missing {layer_dir} -- run dataset/assemble_dataset.py --layer {args.layer} "
                                 f"{'--penetration ' + str(args.penetration) if args.penetration else ''} first.")

    report = {"tag": tag, "method": "persistence (predicted_queue = visible_queue_length_m)"}
    for split in SPLITS_TO_EVALUATE:
        df = load_split(layer_dir, split)
        report[split] = evaluate_persistence(df) if df is not None and len(df) else {"rows": 0, "note": "split empty or not assembled"}

    print("=" * 70)
    print(f"PERSISTENCE BASELINE -- {tag}")
    print("=" * 70)
    for split in SPLITS_TO_EVALUATE:
        r = report[split]
        if not r.get("rows"):
            print(f"{split:6}: (empty)")
            continue
        print(f"{split:6}: MAE={r['mae_m']:6.2f}m  RMSE={r['rmse_m']:6.2f}m  "
              f"(n={r['rows']}, {r['rows_with_queue_beyond_camera']} rows with hidden queue, "
              f"MAE there={r['mae_m_when_queue_beyond_camera']})")

    out_path = layer_dir / "persistence_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()