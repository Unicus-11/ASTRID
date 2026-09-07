"""
evaluate_gps_penetration_sensitivity.py
==========================================
GPS PENETRATION SENSITIVITY / ROBUSTNESS EXPERIMENT -- Layer 2.

This is NOT a hyperparameter-tuning experiment. It measures how the
SAME, ALREADY-FITTED, FROZEN original baseline HistGradientBoosting
model performs when GPS probe penetration changes:

    p05 = 5%
    p11 = 11%   <- existing primary experiment/reference
    p25 = 25%
    p50 = 50%

(10% is intentionally excluded.)

Model handling
---------------
The exact existing artifact

    models/artifacts/layer2_p11/hist_gradient_boosting_layer2_p11_baseline/
        hist_gradient_boosting.joblib

is loaded ONCE via persistence.load_model() and reused, unmodified, for
prediction against all four penetration-level datasets. It is never
refit, never retrained per penetration level, and no hyperparameters are
tuned anywhere in this script. This script creates no new model
artifact.

A note on DatasetLoader and penetration levels
------------------------------------------------
experiment_config.Layer only declares two members: LAYER1 and
LAYER2_P11 -- there is no LAYER2_P05 / LAYER2_P25 / LAYER2_P50 enum
member. Neither data_loader.DatasetLoader nor data_loader.load_split
actually require a *real* Layer enum member at runtime, though -- the
only thing either of them ever does with the `layer` argument is read
`layer.value` to build `data_root / layer.value`, and store it on
SplitData for bookkeeping. So rather than hand-rolling a second,
parallel CSV-reading path for p05/p25/p50 (which the task explicitly
asks to avoid), this script defines a minimal duck-typed stand-in
(_PenetrationLayer, just a frozen dataclass holding `.value`) and passes
that to DatasetLoader for the three penetration levels that have no
enum member. p11 uses the real Layer.LAYER2_P11 enum member. Neither
data_loader.py nor experiment_config.py is modified by this.

Feature-integrity guarantees
------------------------------
Before any prediction, for every (penetration level, split) this script
verifies:
    1. the split's model-feature columns match the fitted baseline
       model's own expected feature columns (via
       `feature_names_in_` when the fitted estimator exposes it);
    2. feature order matches;
    3. 'gps_penetration_rate_requested' is not a feature column;
    4. no metadata/scenario column is a feature column;
    5. no label column is a feature column.
Any mismatch STOPS the script with a reported diff instead of silently
reindexing/altering the data.

Run:
    python models/results/evaluate_gps_penetration_sensitivity.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# models/results/evaluate_gps_penetration_sensitivity.py -> models/ is
# the parent's parent.
_MODELS_DIR = Path(__file__).resolve().parent.parent
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from data_loader import DatasetLoader  # noqa: E402
from experiment_config import Layer, Split, DEFAULT_OUTPUT_ROOT  # noqa: E402
from metrics import compute_all_metrics  # noqa: E402
from persistence import load_model  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed configuration
# ---------------------------------------------------------------------------

_RESULTS_DIR = Path(__file__).resolve().parent
_OUTPUT_CSV = _RESULTS_DIR / "gps_penetration_sensitivity_results.csv"

# The exact, already-fitted original baseline artifact. Loaded once,
# never refit, never modified.
BASELINE_ARTIFACT_PATH = (
    DEFAULT_OUTPUT_ROOT / "layer2_p11"
    / "hist_gradient_boosting_layer2_p11_baseline"
    / "hist_gradient_boosting.joblib"
)

REFERENCE_TAG = "p11"


@dataclass(frozen=True)
class _PenetrationLayer:
    """Minimal duck-typed stand-in for experiment_config.Layer, used only
    for penetration levels (p05/p25/p50) that have no Layer enum member.
    data_loader.DatasetLoader/load_split only ever access `.value` on
    whatever `layer` they're given -- this satisfies exactly that, and
    nothing else. Does not modify experiment_config.py or data_loader.py."""

    value: str


# Penetration tag -> the `layer` object to hand to DatasetLoader.
# p11 uses the real, existing Layer.LAYER2_P11 member (existing primary
# experiment/reference); the others use the duck-typed stand-in above.
PENETRATION_LAYERS: Dict[str, Any] = {
    "p05": _PenetrationLayer("layer2_p05"),
    "p11": Layer.LAYER2_P11,
    "p25": _PenetrationLayer("layer2_p25"),
    "p50": _PenetrationLayer("layer2_p50"),
}

PENETRATION_ORDER: List[str] = ["p05", "p11", "p25", "p50"]
PENETRATION_PCT: Dict[str, int] = {"p05": 5, "p11": 11, "p25": 25, "p50": 50}
PENETRATION_RATE: Dict[str, float] = {"p05": 0.05, "p11": 0.11, "p25": 0.25, "p50": 0.50}

EXPECTED_SCENARIOS = {
    "TEST": ["scenario_east_west_heavy", "scenario_south_heavy"],
    "OOD": [
        "scenario_burst_demand_OOD",
        "scenario_heavy_vehicle_OOD",
        "scenario_north_extreme_OOD",
        "scenario_very_high_demand_OOD",
    ],
}

# Columns that must never reach the model as features, regardless of
# which penetration dataset they came from.
FORBIDDEN_MODEL_INPUT_COLUMNS = {
    "scenario_id",
    "split",
    "design_method",
    "gps_penetration_rate_requested",
    "true_queue_length_m",
    "true_queue_beyond_camera",
}

EVAL_SPLITS = ["TEST", "OOD"]
_SPLIT_ENUM = {"TEST": Split.TEST, "OOD": Split.OOD}


# ---------------------------------------------------------------------------
# Feature-schema verification
# ---------------------------------------------------------------------------

def resolve_expected_features(baseline_model: Any) -> "tuple[List[str], str]":
    """The reference feature schema every penetration dataset is checked
    against. Prefer the fitted estimator's own feature_names_in_ (set by
    sklearn automatically when fit on a DataFrame); fall back to the
    p11 dataset's manifest-declared feature_columns -- the dataset the
    artifact was actually trained on -- if the estimator doesn't expose
    that attribute."""
    if hasattr(baseline_model, "feature_names_in_"):
        return list(baseline_model.feature_names_in_), "baseline_model.feature_names_in_"

    reference_split = DatasetLoader(layer=Layer.LAYER2_P11).load(Split.TEST)
    return (
        list(reference_split.feature_columns),
        "layer2_p11 manifest.json feature_columns "
        "(fitted estimator has no feature_names_in_ attribute)",
    )


def check_feature_schema(
    expected_features: List[str],
    split_data,
    penetration_tag: str,
    split_label: str,
) -> None:
    """Verify column set + column order match, and that no forbidden
    metadata/label column has leaked into the feature matrix. Raises
    SystemExit (stops the run) on any mismatch instead of silently
    reindexing or dropping columns."""
    actual = list(split_data.X.columns)

    same_columns = set(actual) == set(expected_features)
    same_order = actual == expected_features
    missing = [c for c in expected_features if c not in actual]
    extra = [c for c in actual if c not in expected_features]
    forbidden_present = sorted(FORBIDDEN_MODEL_INPUT_COLUMNS & set(actual))

    print(f"  [{penetration_tag} / {split_label}] "
          f"same columns: {'YES' if same_columns else 'NO'}   "
          f"same order: {'YES' if same_order else 'NO'}   "
          f"missing columns: {missing}   extra columns: {extra}")

    if forbidden_present:
        print(f"  [{penetration_tag} / {split_label}] "
              f"FORBIDDEN column(s) present in feature matrix: {forbidden_present}")

    if not same_columns or not same_order or forbidden_present:
        raise SystemExit(
            f"STOPPING: feature schema problem for {penetration_tag}/{split_label}. "
            f"same_columns={same_columns} same_order={same_order} "
            f"missing={missing} extra={extra} forbidden={forbidden_present}. "
            f"Refusing to silently alter, reorder, or filter the data -- fix the "
            f"dataset/manifest and re-run."
        )


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _fmt(v: float, d: int = 6) -> str:
    return f"{v:.{d}f}"


def check_scenario_composition(split_data, penetration_tag: str, split_label: str) -> None:
    actual = split_data.scenario_ids()
    expected = EXPECTED_SCENARIOS[split_label]
    match = actual == expected
    print(f"  [{penetration_tag} / {split_label}] scenarios: {actual}   "
          f"matches expected: {'YES' if match else 'NO'}")
    if not match:
        print(f"    WARNING: expected {expected}, got {actual} -- "
              f"scenario membership shifted for this penetration level.")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    print("GPS PENETRATION SENSITIVITY / ROBUSTNESS EXPERIMENT (Layer 2)")
    print("Fixed model, four penetration levels. No tuning, no retraining.")
    print("=" * 84)
    print()

    print("ORIGINAL BASELINE -- loaded from existing artifact (NOT refit)")
    print("-" * 84)
    print(f"  artifact path = {BASELINE_ARTIFACT_PATH}")
    if not BASELINE_ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Baseline artifact not found: {BASELINE_ARTIFACT_PATH}")
    baseline_model = load_model(BASELINE_ARTIFACT_PATH)
    print("  status = loaded once; reused unmodified across all penetration levels")
    print()

    expected_features, feature_source = resolve_expected_features(baseline_model)
    print("REFERENCE FEATURE SCHEMA")
    print("-" * 84)
    print(f"  source = {feature_source}")
    print(f"  n_features = {len(expected_features)}")
    print(f"  columns (in order) = {expected_features}")
    print()

    print("FEATURE-INTEGRITY CHECK (per penetration level, per split)")
    print("-" * 84)

    per_split_data: Dict[str, Dict[str, Any]] = {tag: {} for tag in PENETRATION_ORDER}

    for tag in PENETRATION_ORDER:
        layer_obj = PENETRATION_LAYERS[tag]
        loader = DatasetLoader(layer=layer_obj)
        for split_label in EVAL_SPLITS:
            split_data = loader.load(_SPLIT_ENUM[split_label])
            check_feature_schema(expected_features, split_data, tag, split_label)
            per_split_data[tag][split_label] = split_data
    print()

    print("SCENARIO COMPOSITION CHECK (per penetration level, per split)")
    print("-" * 84)
    for tag in PENETRATION_ORDER:
        for split_label in EVAL_SPLITS:
            check_scenario_composition(per_split_data[tag][split_label], tag, split_label)
    print()

    # ---- Predict with the SAME frozen model on every penetration/split ----
    metric_rows: List[Dict[str, Any]] = []
    metrics_by_tag_split: Dict[str, Dict[str, Dict[str, float]]] = {tag: {} for tag in PENETRATION_ORDER}

    for tag in PENETRATION_ORDER:
        for split_label in EVAL_SPLITS:
            split_data = per_split_data[tag][split_label]
            preds = baseline_model.predict(split_data.X)
            m = compute_all_metrics(split_data.y, preds)
            metrics_by_tag_split[tag][split_label] = m
            metric_rows.append({
                "penetration_rate": PENETRATION_RATE[tag],
                "penetration_tag": tag,
                "split": split_label,
                "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
                "n": m["n"],
            })

    results_df = pd.DataFrame(metric_rows)

    # ---- Main output table ----
    print("MAIN OUTPUT TABLE")
    print("-" * 84)
    table_rows = []
    for tag in PENETRATION_ORDER:
        test_m = metrics_by_tag_split[tag]["TEST"]
        ood_m = metrics_by_tag_split[tag]["OOD"]
        table_rows.append({
            "penetration": f"{PENETRATION_PCT[tag]}%",
            "TEST MAE": test_m["mae"],
            "TEST RMSE": test_m["rmse"],
            "TEST R2": test_m["r2"],
            "OOD MAE": ood_m["mae"],
            "OOD RMSE": ood_m["rmse"],
            "OOD R2": ood_m["r2"],
        })
    table_df = pd.DataFrame(table_rows).round(6)
    print(table_df.to_string(index=False))
    print()

    # ---- Comparisons vs p11 reference ----
    print(f"COMPARISON vs {REFERENCE_TAG.upper()} REFERENCE ({PENETRATION_PCT[REFERENCE_TAG]}%)")
    print("-" * 84)
    ref_test = metrics_by_tag_split[REFERENCE_TAG]["TEST"]
    ref_ood = metrics_by_tag_split[REFERENCE_TAG]["OOD"]

    comparison_rows = []
    for tag in PENETRATION_ORDER:
        if tag == REFERENCE_TAG:
            continue
        test_m = metrics_by_tag_split[tag]["TEST"]
        ood_m = metrics_by_tag_split[tag]["OOD"]

        test_mae_pct = (ref_test["mae"] - test_m["mae"]) / ref_test["mae"] * 100.0
        test_rmse_pct = (ref_test["rmse"] - test_m["rmse"]) / ref_test["rmse"] * 100.0
        test_r2_delta = test_m["r2"] - ref_test["r2"]

        ood_mae_pct = (ref_ood["mae"] - ood_m["mae"]) / ref_ood["mae"] * 100.0
        ood_rmse_pct = (ref_ood["rmse"] - ood_m["rmse"]) / ref_ood["rmse"] * 100.0
        ood_r2_delta = ood_m["r2"] - ref_ood["r2"]

        comparison_rows.append({
            "penetration": f"{PENETRATION_PCT[tag]}%",
            "TEST_mae_change_pct": test_mae_pct,
            "TEST_rmse_change_pct": test_rmse_pct,
            "TEST_r2_change": test_r2_delta,
            "OOD_mae_change_pct": ood_mae_pct,
            "OOD_rmse_change_pct": ood_rmse_pct,
            "OOD_r2_change": ood_r2_delta,
        })

        print(f"  {PENETRATION_PCT[tag]}% vs {PENETRATION_PCT[REFERENCE_TAG]}%:")
        print(f"    TEST: MAE {test_mae_pct:+.2f}%   RMSE {test_rmse_pct:+.2f}%   R2 {test_r2_delta:+.6f}   "
              f"({'better' if test_mae_pct > 0 else 'worse'} by MAE)")
        print(f"    OOD : MAE {ood_mae_pct:+.2f}%   RMSE {ood_rmse_pct:+.2f}%   R2 {ood_r2_delta:+.6f}   "
              f"({'better' if ood_mae_pct > 0 else 'worse'} by MAE)")
    print()
    print("  (Positive % = lower error than p11, i.e. this penetration level performs")
    print("   BETTER than the 11% reference. Negative % = worse than p11.)")
    print()

    # ---- Save long-form CSV ----
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(_OUTPUT_CSV, index=False)

    # ---- Final interpretation (derived from the numbers above, not assumed) ----
    print("FINAL INTERPRETATION")
    print("-" * 84)

    test_by_tag = {tag: metrics_by_tag_split[tag]["TEST"] for tag in PENETRATION_ORDER}
    ood_by_tag = {tag: metrics_by_tag_split[tag]["OOD"] for tag in PENETRATION_ORDER}

    best_test_tag = min(PENETRATION_ORDER, key=lambda t: test_by_tag[t]["mae"])
    best_ood_tag = min(PENETRATION_ORDER, key=lambda t: ood_by_tag[t]["mae"])

    p05_worse_test = test_by_tag["p05"]["mae"] > ref_test["mae"]
    p05_worse_ood = ood_by_tag["p05"]["mae"] > ref_ood["mae"]

    test_mae_by_pct = [test_by_tag[t]["mae"] for t in PENETRATION_ORDER]  # p05,p11,p25,p50 order
    ood_mae_by_pct = [ood_by_tag[t]["mae"] for t in PENETRATION_ORDER]
    test_monotonic_improving = all(
        test_mae_by_pct[i] >= test_mae_by_pct[i + 1] for i in range(len(test_mae_by_pct) - 1)
    )
    ood_monotonic_improving = all(
        ood_mae_by_pct[i] >= ood_mae_by_pct[i + 1] for i in range(len(ood_mae_by_pct) - 1)
    )

    print(f"1. Does lower GPS penetration hurt TEST performance? "
          f"{'YES -- p05 TEST MAE is higher than p11.' if p05_worse_test else 'NO -- p05 TEST MAE is not higher than p11 in this run.'}")
    print(f"2. Does lower GPS penetration hurt OOD performance?  "
          f"{'YES -- p05 OOD MAE is higher than p11.' if p05_worse_ood else 'NO -- p05 OOD MAE is not higher than p11 in this run.'}")
    print(f"3. Does increasing GPS penetration monotonically improve accuracy (by MAE, "
          f"p05->p11->p25->p50)?\n"
          f"   TEST: {'YES' if test_monotonic_improving else 'NO'}   "
          f"OOD: {'YES' if ood_monotonic_improving else 'NO'}")
    print(f"4. Best TEST MAE: {PENETRATION_PCT[best_test_tag]}% "
          f"(MAE={_fmt(test_by_tag[best_test_tag]['mae'])})")
    print(f"5. Best OOD MAE:  {PENETRATION_PCT[best_ood_tag]}% "
          f"(MAE={_fmt(ood_by_tag[best_ood_tag]['mae'])})")

    p25_test_gain = (ref_test["mae"] - test_by_tag["p25"]["mae"]) / ref_test["mae"] * 100.0
    p50_test_gain = (ref_test["mae"] - test_by_tag["p50"]["mae"]) / ref_test["mae"] * 100.0
    p25_ood_gain = (ref_ood["mae"] - ood_by_tag["p25"]["mae"]) / ref_ood["mae"] * 100.0
    p50_ood_gain = (ref_ood["mae"] - ood_by_tag["p50"]["mae"]) / ref_ood["mae"] * 100.0
    material_gain = max(p25_test_gain, p50_test_gain, p25_ood_gain, p50_ood_gain)

    print(f"6. Is 11% sufficient, or does higher GPS penetration materially improve "
          f"estimation?\n"
          f"   p25 vs p11: TEST {p25_test_gain:+.2f}%, OOD {p25_ood_gain:+.2f}% (MAE)\n"
          f"   p50 vs p11: TEST {p50_test_gain:+.2f}%, OOD {p50_ood_gain:+.2f}% (MAE)\n"
          f"   Largest MAE gain from going above 11% penetration in this run: "
          f"{material_gain:+.2f}%.")
    print()
    print("No model was chosen or retuned based on these results. This was an")
    print("evaluation of the existing frozen baseline artifact only.")
    print()

    print(f"Results CSV written to: {_OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()