"""
analyze_trial7_by_scenario.py
================================
SCENARIO-LEVEL DIAGNOSTIC -- Original baseline vs Tuned Trial 7 (Layer 2,
layer2_p11), on TEST and OOD.

This is an ANALYSIS-ONLY script. It performs NO hyperparameter tuning
and does not modify any existing tuning script, evaluation script, or
result CSV. Its sole purpose is to break the already-computed aggregate
TEST/OOD metrics down by individual scenario, so that "Trial 7 is worse
on held-out data" can be explained by WHICH scenarios drive that
degradation, rather than guessed at from scenario names.

Original baseline
------------------
The original baseline is NOT reconstructed or refit. It is loaded
directly from its already-trained artifact:

    models/artifacts/layer2_p11/hist_gradient_boosting_layer2_p11_baseline/
        hist_gradient_boosting.joblib

via persistence.load_model(). It is never fit, never modified, and no
hyperparameters are assumed for it -- whatever hyperparameters produced
that artifact are irrelevant here because the artifact itself is used
as-is.

Trial 7
-------
Trial 7 is fixed (already selected; not re-derived or re-searched here)
and is constructed directly via sklearn.ensemble.HistGradientBoostingRegressor
(NOT via model_implementations/hist_gradient_boosting.py, which does not
expose/forward early_stopping), with early_stopping=False and
random_state=42. It is fit on TRAIN only.

Both models are scored per-scenario AND in aggregate via
metrics.compute_all_metrics() -- no metric math is reimplemented.

Scenario-level train/validation/test/ood split (verified, unchanged):
    TRAIN      : scenario_high_demand, scenario_left_turn_heavy,
                 scenario_low_demand, scenario_normal_balanced
    VALIDATION : scenario_north_heavy, scenario_straight_heavy
    TEST       : scenario_east_west_heavy, scenario_south_heavy
    OOD        : scenario_burst_demand_OOD, scenario_heavy_vehicle_OOD,
                 scenario_north_extreme_OOD, scenario_very_high_demand_OOD
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# models/results/analyze_trial7_by_scenario.py -> models/ is the
# parent's parent.
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

LAYER = Layer.LAYER2_P11
MODEL_RANDOM_STATE = 42

_RESULTS_DIR = Path(__file__).resolve().parent
_OUTPUT_CSV = _RESULTS_DIR / "hist_gradient_boosting_trial7_scenario_analysis.csv"

# Original baseline -- loaded from its existing trained artifact.
# NEVER refit, NEVER reconstructed from assumed hyperparameters.
ORIGINAL_BASELINE_PATH = (
    DEFAULT_OUTPUT_ROOT / "layer2_p11"
    / "hist_gradient_boosting_layer2_p11_baseline"
    / "hist_gradient_boosting.joblib"
)

# Selected Trial 7 configuration (already chosen via Rounds 2-5; NOT
# re-derived or re-searched here).
TRIAL7_CONFIG: Dict[str, Any] = {
    "learning_rate": 0.025034,
    "max_iter": 369,
    "max_leaf_nodes": 52,
    "min_samples_leaf": 35,
    "l2_regularization": 0.502603,
    "max_depth": 10,
}

# Previously reported final-evaluation aggregate numbers, used only as a
# consistency check -- NEVER substituted for freshly computed values.
EXPECTED_TRIAL7_AGGREGATE = {
    "TEST": {"mae": 19.152881, "rmse": 46.303357, "r2": 0.934579},
    "OOD": {"mae": 34.302741, "rmse": 63.247887, "r2": 0.903417},
}

REFERENCE_BASELINE_AGGREGATE = {
    "TEST": {"mae": 18.706834, "rmse": 44.630782, "r2": 0.939220},
    "OOD": {"mae": 34.757319, "rmse": 63.107135, "r2": 0.903846},
}


# ---------------------------------------------------------------------------
# Model construction / fitting
# ---------------------------------------------------------------------------

def build_trial7_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=TRIAL7_CONFIG["learning_rate"],
        max_iter=TRIAL7_CONFIG["max_iter"],
        max_leaf_nodes=TRIAL7_CONFIG["max_leaf_nodes"],
        min_samples_leaf=TRIAL7_CONFIG["min_samples_leaf"],
        l2_regularization=TRIAL7_CONFIG["l2_regularization"],
        max_depth=TRIAL7_CONFIG["max_depth"],
        early_stopping=False,
        random_state=MODEL_RANDOM_STATE,
    )


def load_original_baseline():
    """Load the existing trained baseline artifact as-is. Never fit,
    never reconstructed from assumed hyperparameters."""
    if not ORIGINAL_BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Original baseline artifact not found: {ORIGINAL_BASELINE_PATH}"
        )
    return load_model(ORIGINAL_BASELINE_PATH)


# ---------------------------------------------------------------------------
# Scenario-level scoring
# ---------------------------------------------------------------------------

def score_by_scenario(
    split_label: str,
    split_data,
    baseline_preds,
    trial7_preds,
) -> List[Dict[str, Any]]:
    """Compute per-scenario MAE/RMSE/R2/n for both models on one split,
    using the split's own metadata.scenario_id to slice rows -- no new
    splitting logic, just row selection within the already-loaded split."""
    rows = []
    scenario_ids = split_data.metadata["scenario_id"]

    for scenario in sorted(scenario_ids.unique()):
        mask = (scenario_ids == scenario).to_numpy()

        y_true_scn = split_data.y[mask]
        baseline_preds_scn = baseline_preds[mask]
        trial7_preds_scn = trial7_preds[mask]

        baseline_m = compute_all_metrics(y_true_scn, baseline_preds_scn)
        trial7_m = compute_all_metrics(y_true_scn, trial7_preds_scn)

        mae_change_pct = (baseline_m["mae"] - trial7_m["mae"]) / baseline_m["mae"] * 100.0
        rmse_change_pct = (baseline_m["rmse"] - trial7_m["rmse"]) / baseline_m["rmse"] * 100.0
        r2_change = trial7_m["r2"] - baseline_m["r2"]

        rows.append({
            "split": split_label,
            "scenario": scenario,
            "n": baseline_m["n"],
            "baseline_mae": baseline_m["mae"],
            "trial7_mae": trial7_m["mae"],
            "mae_change_pct": mae_change_pct,
            "baseline_rmse": baseline_m["rmse"],
            "trial7_rmse": trial7_m["rmse"],
            "rmse_change_pct": rmse_change_pct,
            "baseline_r2": baseline_m["r2"],
            "trial7_r2": trial7_m["r2"],
            "r2_change": r2_change,
        })
    return rows


def aggregate_row(split_label: str, split_data, baseline_preds, trial7_preds) -> Dict[str, Any]:
    baseline_m = compute_all_metrics(split_data.y, baseline_preds)
    trial7_m = compute_all_metrics(split_data.y, trial7_preds)

    mae_change_pct = (baseline_m["mae"] - trial7_m["mae"]) / baseline_m["mae"] * 100.0
    rmse_change_pct = (baseline_m["rmse"] - trial7_m["rmse"]) / baseline_m["rmse"] * 100.0
    r2_change = trial7_m["r2"] - baseline_m["r2"]

    return {
        "split": split_label,
        "scenario": "__AGGREGATE__",
        "n": baseline_m["n"],
        "baseline_mae": baseline_m["mae"],
        "trial7_mae": trial7_m["mae"],
        "mae_change_pct": mae_change_pct,
        "baseline_rmse": baseline_m["rmse"],
        "trial7_rmse": trial7_m["rmse"],
        "rmse_change_pct": rmse_change_pct,
        "baseline_r2": baseline_m["r2"],
        "trial7_r2": trial7_m["r2"],
        "r2_change": r2_change,
    }, baseline_m, trial7_m


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _fmt(v: float, d: int = 6) -> str:
    return f"{v:.{d}f}"


def print_aggregate_consistency_check(split_label: str, computed: Dict[str, float],
                                       expected: Dict[str, float], role: str) -> None:
    print(f"  [{role}] {split_label} -- recomputed vs previously reported:")
    for key in ("mae", "rmse", "r2"):
        diff = computed[key] - expected[key]
        print(f"    {key.upper():5s}: recomputed={_fmt(computed[key])}  "
              f"previously_reported={_fmt(expected[key])}  diff={diff:+.6f}")


def scenario_extreme(df: pd.DataFrame, split_label: str, metric_change_col: str) -> pd.Series:
    """Row with the most NEGATIVE change (largest degradation) for one
    split/metric among scenario-level rows only (aggregate excluded)."""
    subset = df[(df["split"] == split_label) & (df["scenario"] != "__AGGREGATE__")]
    return subset.loc[subset[metric_change_col].idxmin()]


def scenario_best(df: pd.DataFrame, split_label: str, metric_change_col: str) -> pd.Series:
    """Row with the most POSITIVE change (largest improvement)."""
    subset = df[(df["split"] == split_label) & (df["scenario"] != "__AGGREGATE__")]
    return subset.loc[subset[metric_change_col].idxmax()]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    print("SCENARIO-LEVEL DIAGNOSTIC -- Original baseline vs Tuned Trial 7")
    print("Analysis-only. No tuning is performed by this script.")
    print("=" * 84)
    print()
    print(f"Layer: {LAYER.value}")
    print()

    print("ORIGINAL BASELINE -- loaded from existing artifact (NOT refit)")
    print("-" * 84)
    print(f"  artifact path = {ORIGINAL_BASELINE_PATH}")
    print()

    print("TRIAL 7 CONFIGURATION (fixed -- not re-tuned)")
    print("-" * 84)
    for k, v in TRIAL7_CONFIG.items():
        print(f"  {k:20s} = {v}")
    print("  early_stopping       = False")
    print("  random_state         = 42")
    print()

    # ---- Load TRAIN, TEST, OOD (VALIDATION intentionally not used) ----
    loader = DatasetLoader(layer=LAYER)
    train_split = loader.load(Split.TRAIN)
    test_split = loader.load(Split.TEST)
    ood_split = loader.load(Split.OOD)

    print(f"TRAIN scenarios: {train_split.scenario_ids()}")
    print(f"TEST scenarios : {test_split.scenario_ids()}")
    print(f"OOD scenarios  : {ood_split.scenario_ids()}")
    print(f"TRAIN rows: {len(train_split)}   TEST rows: {len(test_split)}   OOD rows: {len(ood_split)}")
    print()

    # ---- Load original baseline (no fitting); fit Trial 7 on TRAIN only ----
    baseline_model = load_original_baseline()

    trial7_model = build_trial7_model()
    trial7_model.fit(train_split.X, train_split.y)

    # ---- Predict once per split, per model ----
    baseline_test_preds = baseline_model.predict(test_split.X)
    trial7_test_preds = trial7_model.predict(test_split.X)

    baseline_ood_preds = baseline_model.predict(ood_split.X)
    trial7_ood_preds = trial7_model.predict(ood_split.X)

    # ---- Per-scenario rows ----
    all_rows: List[Dict[str, Any]] = []
    all_rows += score_by_scenario("TEST", test_split, baseline_test_preds, trial7_test_preds)
    all_rows += score_by_scenario("OOD", ood_split, baseline_ood_preds, trial7_ood_preds)

    # ---- Aggregate rows (recomputed from predictions, not hardcoded) ----
    test_agg_row, test_baseline_m, test_trial7_m = aggregate_row(
        "TEST", test_split, baseline_test_preds, trial7_test_preds
    )
    ood_agg_row, ood_baseline_m, ood_trial7_m = aggregate_row(
        "OOD", ood_split, baseline_ood_preds, trial7_ood_preds
    )
    all_rows.append(test_agg_row)
    all_rows.append(ood_agg_row)

    results_df = pd.DataFrame(all_rows)

    # ---- Print full scenario-level table ----
    print("SCENARIO-LEVEL COMPARISON TABLE")
    print("-" * 84)
    display_cols = [
        "split", "scenario", "n",
        "baseline_mae", "trial7_mae", "mae_change_pct",
        "baseline_rmse", "trial7_rmse", "rmse_change_pct",
        "baseline_r2", "trial7_r2", "r2_change",
    ]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(results_df[display_cols].to_string(index=False))
    print()

    # ---- Aggregate sanity check vs previously reported final-eval numbers ----
    print("AGGREGATE CONSISTENCY CHECK (recomputed here vs previously reported)")
    print("-" * 84)
    print_aggregate_consistency_check("TEST", test_trial7_m, EXPECTED_TRIAL7_AGGREGATE["TEST"], "TRIAL7")
    print_aggregate_consistency_check("OOD", ood_trial7_m, EXPECTED_TRIAL7_AGGREGATE["OOD"], "TRIAL7")
    print_aggregate_consistency_check("TEST", test_baseline_m, REFERENCE_BASELINE_AGGREGATE["TEST"], "BASELINE")
    print_aggregate_consistency_check("OOD", ood_baseline_m, REFERENCE_BASELINE_AGGREGATE["OOD"], "BASELINE")
    print()
    print("  (BASELINE diffs above should be essentially zero, since the original")
    print("   baseline is loaded directly from its trained artifact -- not refit or")
    print("   reconstructed from assumed hyperparameters.)")
    print()

    # ---- Scenario-level findings ----
    print("SCENARIO-LEVEL FINDINGS")
    print("-" * 84)

    test_worst_mae = scenario_extreme(results_df, "TEST", "mae_change_pct")
    test_worst_rmse = scenario_extreme(results_df, "TEST", "rmse_change_pct")
    ood_worst_mae = scenario_extreme(results_df, "OOD", "mae_change_pct")
    ood_worst_rmse = scenario_extreme(results_df, "OOD", "rmse_change_pct")

    test_best_mae = scenario_best(results_df, "TEST", "mae_change_pct")
    ood_best_mae = scenario_best(results_df, "OOD", "mae_change_pct")

    print(f"1. Largest TEST MAE change : {test_worst_mae['scenario']} "
          f"({test_worst_mae['mae_change_pct']:+.2f}%) -- "
          f"{'degradation' if test_worst_mae['mae_change_pct'] < 0 else 'improvement'}")
    print(f"   Largest TEST MAE improvement: {test_best_mae['scenario']} "
          f"({test_best_mae['mae_change_pct']:+.2f}%)")
    print(f"2. Largest TEST RMSE change: {test_worst_rmse['scenario']} "
          f"({test_worst_rmse['rmse_change_pct']:+.2f}%) -- "
          f"{'degradation' if test_worst_rmse['rmse_change_pct'] < 0 else 'improvement'}")
    print(f"3. Largest OOD MAE change  : {ood_worst_mae['scenario']} "
          f"({ood_worst_mae['mae_change_pct']:+.2f}%) -- "
          f"{'degradation' if ood_worst_mae['mae_change_pct'] < 0 else 'improvement'}")
    print(f"   Largest OOD MAE improvement : {ood_best_mae['scenario']} "
          f"({ood_best_mae['mae_change_pct']:+.2f}%)")
    print(f"4. Largest OOD RMSE change : {ood_worst_rmse['scenario']} "
          f"({ood_worst_rmse['rmse_change_pct']:+.2f}%) -- "
          f"{'degradation' if ood_worst_rmse['rmse_change_pct'] < 0 else 'improvement'}")

    scenario_only = results_df[results_df["scenario"] != "__AGGREGATE__"]
    n_improved_mae = (scenario_only["mae_change_pct"] > 0).sum()
    n_degraded_mae = (scenario_only["mae_change_pct"] < 0).sum()
    n_total = len(scenario_only)

    if n_degraded_mae == 0:
        behavior = "CONSISTENTLY IMPROVES"
    elif n_improved_mae == 0:
        behavior = "CONSISTENTLY WORSENS"
    else:
        behavior = "MIXED"
    print(f"5. Behavior across all {n_total} TEST+OOD scenarios (by MAE): {behavior} "
          f"({n_improved_mae} improved, {n_degraded_mae} degraded)")
    print()

    # ---- Save CSV ----
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df[display_cols].to_csv(_OUTPUT_CSV, index=False)
    print(f"Scenario-level analysis CSV written to: {_OUTPUT_CSV.resolve()}")
    print()
    print("This was an analysis-only run. No tuning was performed, no existing")
    print("tuning/evaluation scripts or result CSVs were modified, the original")
    print("baseline artifact was never fit or modified, and the existing Trial 7")
    print("artifact was not overwritten (no artifact is saved by this script).")


if __name__ == "__main__":
    main()