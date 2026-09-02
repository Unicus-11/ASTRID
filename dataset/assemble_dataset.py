"""
assemble_dataset.py
====================
ASTRID Prototype -- Dataset Assembler

RESPONSIBILITY:
    Combine every scenario's features_{tag}.csv + labels_{tag}.csv into ONE
    set of files per split (train/val/test/ood), using the `split` field
    already assigned per scenario in sumo/generated_scenarios/manifest.json.

    `tag` is "layer1" for Layer 1, or "layer2_p{NN}" for Layer 2 at a given
    GPS penetration rate (e.g. "layer2_p10") -- matching feature_builder.py's
    own output naming exactly, since Layer 2 can be built at several
    penetration rates (0.05, 0.10, 0.15, 0.25, 0.50, ...) and each is a
    separate file, not one file overwritten repeatedly.

    Features and labels stay in SEPARATE files even after assembly
    (train_features.csv / train_labels.csv, not train.csv) -- same
    leakage discipline as every earlier stage: join by
    (scenario_id, timestamp, approach_edge) only at training time.

    Scenarios missing features/labels (not yet processed by
    feature_builder.py) are skipped with a warning, not an error.

Reads:
    sumo/generated_scenarios/manifest.json
    sumo/generated_scenarios/scenario_XXXX/features/features_{tag}.csv
    sumo/generated_scenarios/scenario_XXXX/features/labels_{tag}.csv

Writes:
    training_data/{tag}/train_features.csv, train_labels.csv
    training_data/{tag}/val_features.csv,   val_labels.csv
    training_data/{tag}/test_features.csv,  test_labels.csv
    training_data/{tag}/ood_features.csv,   ood_labels.csv
    training_data/{tag}/assembly_manifest.json

Run:
    python dataset/assemble_dataset.py --layer layer1
    python dataset/assemble_dataset.py --layer layer2 --penetration 0.10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMO_DIR = PROJECT_ROOT / "sumo"
SCENARIOS_DIR = SUMO_DIR / "generated_scenarios"
OUTPUT_ROOT = PROJECT_ROOT / "training_data"

SPLITS = ["train", "val", "test", "ood"]


def resolve_tag(layer: str, penetration: Optional[float]) -> str:
    """Matches feature_builder.py's own output naming exactly."""
    if layer == "layer1":
        return "layer1"
    if layer == "layer2":
        if penetration is None:
            raise ValueError("--penetration is required when --layer layer2 "
                              "(Layer 2 is built separately per GPS penetration rate).")
        return f"layer2_p{int(round(penetration * 100)):02d}"
    raise ValueError(f"Unknown layer '{layer}' (expected 'layer1' or 'layer2')")


def load_manifest() -> dict:
    path = SCENARIOS_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} -- run scenario_builder.py first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assemble(layer: str, penetration: Optional[float] = None) -> Dict[str, dict]:
    tag = resolve_tag(layer, penetration)
    manifest = load_manifest()

    split_features: Dict[str, List[pd.DataFrame]] = {s: [] for s in SPLITS}
    split_labels: Dict[str, List[pd.DataFrame]] = {s: [] for s in SPLITS}
    included: Dict[str, List[str]] = {s: [] for s in SPLITS}
    skipped: List[dict] = []

    for entry in manifest["scenarios"]:
        scenario_id = entry["scenario_id"]
        split = entry["split"]
        if split not in SPLITS:
            skipped.append({"scenario_id": scenario_id, "reason": f"unknown split '{split}'"})
            continue

        scenario_dir = SCENARIOS_DIR / scenario_id
        features_path = scenario_dir / "features" / f"features_{tag}.csv"
        labels_path = scenario_dir / "features" / f"labels_{tag}.csv"

        if not features_path.exists() or not labels_path.exists():
            skipped.append({"scenario_id": scenario_id,
                             "reason": f"features/labels for '{tag}' not built yet -- "
                                       f"run dataset/feature_builder.py --scenario {scenario_id} "
                                       f"--layer {layer}" + (f" --penetration {penetration}" if penetration else "")})
            continue

        f_df = pd.read_csv(features_path)
        l_df = pd.read_csv(labels_path)

        if len(f_df) != len(l_df):
            skipped.append({"scenario_id": scenario_id,
                             "reason": f"row count mismatch: features={len(f_df)} labels={len(l_df)} -- "
                                       f"not assembled, investigate before training"})
            continue

        f_df.insert(0, "scenario_id", scenario_id)
        l_df.insert(0, "scenario_id", scenario_id)

        split_features[split].append(f_df)
        split_labels[split].append(l_df)
        included[split].append(scenario_id)

    out_dir = OUTPUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, dict] = {}
    for split in SPLITS:
        if split_features[split]:
            f_all = pd.concat(split_features[split], ignore_index=True)
            l_all = pd.concat(split_labels[split], ignore_index=True)
        else:
            f_all = pd.DataFrame()
            l_all = pd.DataFrame()

        f_all.to_csv(out_dir / f"{split}_features.csv", index=False)
        l_all.to_csv(out_dir / f"{split}_labels.csv", index=False)

        report[split] = {
            "scenarios_included": included[split],
            "scenario_count": len(included[split]),
            "feature_rows": len(f_all),
            "label_rows": len(l_all),
        }

    assembly_manifest = {
        "tag": tag,
        "layer": layer,
        "penetration": penetration,
        "splits": report,
        "skipped": skipped,
        "_note": "features and labels are separate files by design -- join on "
                 "(scenario_id, timestamp, approach_edge) only at training time.",
    }
    with open(out_dir / "assembly_manifest.json", "w", encoding="utf-8") as f:
        json.dump(assembly_manifest, f, indent=2)

    return report, skipped, out_dir, tag


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble per-scenario features/labels into train/val/test/ood.")
    parser.add_argument("--layer", type=str, default="layer1", choices=["layer1", "layer2"])
    parser.add_argument("--penetration", type=float, default=None,
                         help="Required for --layer layer2 (e.g. 0.10 for p10). Ignored for layer1.")
    args = parser.parse_args()

    report, skipped, out_dir, tag = assemble(args.layer, args.penetration)

    print("=" * 70)
    print(f"ASSEMBLED DATASET -- {tag}")
    print("=" * 70)
    for split in SPLITS:
        r = report[split]
        print(f"{split:6}: {r['scenario_count']:2} scenarios | "
              f"{r['feature_rows']:>7} feature rows | {r['label_rows']:>7} label rows")

    if skipped:
        print()
        print(f"Skipped {len(skipped)} scenario(s):")
        for s in skipped:
            print(f"  {s['scenario_id']}: {s['reason']}")

    print()
    print(f"Written to: {out_dir}")

    if report["train"]["scenario_count"] == 0:
        print()
        print("WARNING: train split is empty -- nothing to train on yet.")
        sys.exit(1)


if __name__ == "__main__":
    main()