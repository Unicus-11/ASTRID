"""
error_analysis_random_forest.py
====================
Read-only, analysis-only error inspection for the already-trained Layer 2
Random Forest baseline model.

This mirrors models/results/error_analysis.py's methodology exactly (same
grouping, same bins, same metrics function) but points at the Random
Forest artifact instead of the HistGradientBoosting one, so the two
reports are directly comparable.

This script:
    * loads the ALREADY-TRAINED model via persistence.load_model() --
      no retraining, no fitting
    * loads Layer 2 test/ood splits via the existing DatasetLoader --
      no new splitting logic, no dataset modification
    * calls model.predict(X) once per split
    * builds an in-memory row-level DataFrame per split for analysis only
    * reuses metrics.compute_all_metrics() for every MAE/RMSE/R2 number
      (grouped or overall) -- no metric formula is reimplemented
    * prints a structured terminal report

Nothing here writes to disk. No CSV/JSON is created. No infrastructure
file (data_loader.py, evaluate.py, experiment_runner.py, metrics.py,
persistence.py, model implementations, or error_analysis.py) is modified.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# models/results/error_analysis_random_forest.py -> models/ is the parent's parent.
_MODELS_DIR = Path(__file__).resolve().parent.parent
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from data_loader import DatasetLoader, SplitData  # noqa: E402
from experiment_config import ExperimentConfig, Layer, Split  # noqa: E402
from persistence import load_model  # noqa: E402
from metrics import compute_all_metrics  # noqa: E402

RANDOM_STATE = 42
LAYER = Layer.LAYER2_P11

# Only difference from error_analysis.py's model-identifying constants:
# this points at the Random Forest baseline artifact instead of the
# HistGradientBoosting one. Artifact path this resolves to:
#   models/artifacts/layer2_p11/random_forest_layer2_p11_baseline/random_forest.joblib
EXPERIMENT_NAME = "random_forest_layer2_p11_baseline"
MODEL_FILE_NAME = "random_forest.joblib"
BASELINE_MODEL_NAME = "random_forest"  # must match the "model" column value in baseline_results.csv

_RESULTS_DIR = Path(__file__).resolve().parent
_BASELINE_CSV_PATH = _RESULTS_DIR / "baseline_results.csv"

# ---------------------------------------------------------------------------
# ASSUMPTION: true_queue_length_m analysis bins.
# Kept IDENTICAL to error_analysis.py so the Hist vs Random Forest reports
# are directly comparable. The script also prints the actual distribution
# (describe()) of true_queue_length_m for both splits so the reader can
# judge whether these bins are appropriate; the underlying dataset is
# never modified.
# ---------------------------------------------------------------------------
TRUE_QUEUE_BIN_EDGES = [0, 25, 50, 100, 200, 400, np.inf]
TRUE_QUEUE_BIN_LABELS = ["0-25m", "25-50m", "50-100m", "100-200m", "200-400m", ">400m"]

# visible_queue_length_m: quartile bins computed from the ACTUAL observed
# test-split distribution at runtime, same approach as error_analysis.py.
VISIBLE_QUEUE_N_QUANTILES = 4


# ---------------------------------------------------------------------------
# Loading (no retraining, no dataset modification)
# ---------------------------------------------------------------------------

def load_trained_model():
    config = ExperimentConfig(
        layer=LAYER,
        random_state=RANDOM_STATE,
        experiment_name=EXPERIMENT_NAME,
    )
    model_path = config.output_dir() / MODEL_FILE_NAME
    model = load_model(model_path)
    return model, model_path, config


def load_splits() -> dict:
    loader = DatasetLoader(layer=LAYER)
    return {
        "test": loader.load(Split.TEST),
        "ood": loader.load(Split.OOD),
    }


# ---------------------------------------------------------------------------
# Row-level DataFrame construction (in-memory only)
# ---------------------------------------------------------------------------

_EXTRA_FEATURE_COLUMNS = [
    "queue_reaches_camera_edge",
    "visible_queue_length_m",
    "estimated_density_k_veh_per_km",
    "observed_flow_veh_per_hour",
]


def build_row_level_df(split_data: SplitData, predictions: np.ndarray) -> pd.DataFrame:
    """Combine SplitData.keys / .metadata / .X / .y with predictions into
    one in-memory analysis DataFrame. Does not mutate split_data."""
    extra_cols = [c for c in _EXTRA_FEATURE_COLUMNS if c in split_data.X.columns]

    df = pd.DataFrame(
        {
            "timestamp": split_data.keys["timestamp"].values,
            "approach_edge": split_data.keys["approach_edge"].values,
            "scenario_id": split_data.metadata["scenario_id"].values,
            "true_queue_length_m": split_data.y.values,
            "prediction": predictions,
        }
    )
    df["residual"] = df["prediction"] - df["true_queue_length_m"]
    df["absolute_error"] = df["residual"].abs()

    for col in extra_cols:
        df[col] = split_data.X[col].values

    return df


# ---------------------------------------------------------------------------
# Grouped metrics (reuses metrics.compute_all_metrics -- no reimplementation)
# ---------------------------------------------------------------------------

def grouped_report(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Group df by group_col and compute MAE/RMSE/R2/n via
    metrics.compute_all_metrics(), plus mean_residual and max_abs_error."""
    rows = []
    for group_value, group_df in df.groupby(group_col, observed=True, dropna=False):
        m = compute_all_metrics(group_df["true_queue_length_m"], group_df["prediction"])
        rows.append(
            {
                group_col: group_value,
                "n": m["n"],
                "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
                "mean_residual": group_df["residual"].mean(),
                "max_abs_error": group_df["absolute_error"].max(),
            }
        )
    result = pd.DataFrame(rows).set_index(group_col)
    return result.sort_values("mae", ascending=False)


def overall_report(df: pd.DataFrame) -> dict:
    m = compute_all_metrics(df["true_queue_length_m"], df["prediction"])
    return {
        "n": m["n"],
        "mae": m["mae"],
        "rmse": m["rmse"],
        "r2": m["r2"],
        "mean_residual": df["residual"].mean(),
        "max_abs_error": df["absolute_error"].max(),
    }


# ---------------------------------------------------------------------------
# Binning helpers (analysis-only columns, added to the in-memory copy)
# ---------------------------------------------------------------------------

def add_true_queue_bin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["true_queue_bin"] = pd.cut(
        df["true_queue_length_m"],
        bins=TRUE_QUEUE_BIN_EDGES,
        labels=TRUE_QUEUE_BIN_LABELS,
        right=False,
        include_lowest=True,
    )
    return df


def add_visible_queue_bin(df: pd.DataFrame) -> "tuple[pd.DataFrame, Optional[list]]":
    if "visible_queue_length_m" not in df.columns:
        return df, None
    df = df.copy()
    try:
        binned, edges = pd.qcut(
            df["visible_queue_length_m"],
            q=VISIBLE_QUEUE_N_QUANTILES,
            duplicates="drop",
            retbins=True,
        )
        df["visible_queue_bin"] = binned.astype(str)
        return df, list(edges)
    except ValueError:
        # Not enough distinct values to form meaningful quantile bins.
        return df, None


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _fmt(value, decimals: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NA"
    return f"{value:.{decimals}f}"


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def print_overall(label: str, stats: dict) -> None:
    print_section(label)
    print(f"n    = {stats['n']}")
    print(f"MAE  = {_fmt(stats['mae'])}")
    print(f"RMSE = {_fmt(stats['rmse'])}")
    print(f"R2   = {_fmt(stats['r2'])}")
    print(f"Mean residual (pred - true) = {_fmt(stats['mean_residual'])}")
    print(f"Max abs error                = {_fmt(stats['max_abs_error'])}")


def print_grouped(label: str, grouped_df: pd.DataFrame) -> None:
    print_section(label)
    if grouped_df.empty:
        print("(no data)")
        return
    display = grouped_df.copy()
    for col in ["mae", "rmse", "r2", "mean_residual", "max_abs_error"]:
        display[col] = display[col].apply(_fmt)
    print(display.to_string())


def print_distribution_summary(label: str, series: pd.Series) -> None:
    print_section(label)
    print(series.describe().to_string())


def load_baseline_consistency_row() -> Optional[pd.DataFrame]:
    if not _BASELINE_CSV_PATH.exists():
        return None
    baseline_df = pd.read_csv(_BASELINE_CSV_PATH)
    match = baseline_df[
        (baseline_df["model"] == BASELINE_MODEL_NAME)
        & (baseline_df["layer"] == LAYER.value)
    ]
    return match if not match.empty else None


# ---------------------------------------------------------------------------
# Key patterns summary (derived from computed tables, no new claims)
# ---------------------------------------------------------------------------

def print_key_patterns(
    test_by_scenario: pd.DataFrame,
    ood_by_scenario: pd.DataFrame,
    test_by_approach: pd.DataFrame,
    test_by_queue_bin: pd.DataFrame,
    test_by_congestion: pd.DataFrame,
    test_overall: dict,
    ood_overall: dict,
) -> None:
    print_section("KEY ERROR PATTERNS")

    if not test_by_scenario.empty:
        worst_test_scenario = test_by_scenario.index[0]
        print(
            f"- Highest test MAE by scenario: '{worst_test_scenario}' "
            f"(MAE={_fmt(test_by_scenario.iloc[0]['mae'])}, "
            f"n={int(test_by_scenario.iloc[0]['n'])})."
        )

    if not ood_by_scenario.empty:
        worst_ood_scenario = ood_by_scenario.index[0]
        print(
            f"- Highest OOD MAE by scenario: '{worst_ood_scenario}' "
            f"(MAE={_fmt(ood_by_scenario.iloc[0]['mae'])}, "
            f"n={int(ood_by_scenario.iloc[0]['n'])})."
        )

    if not test_by_approach.empty:
        worst_approach = test_by_approach.index[0]
        best_approach = test_by_approach.index[-1]
        print(
            f"- Test MAE varies by approach: highest at '{worst_approach}' "
            f"({_fmt(test_by_approach.iloc[0]['mae'])}), lowest at "
            f"'{best_approach}' ({_fmt(test_by_approach.iloc[-1]['mae'])})."
        )

    if not test_by_queue_bin.empty:
        by_bin_sorted = test_by_queue_bin.reindex(
            [b for b in TRUE_QUEUE_BIN_LABELS if b in test_by_queue_bin.index]
        )
        if not by_bin_sorted.empty:
            first_mae = by_bin_sorted["mae"].iloc[0]
            last_mae = by_bin_sorted["mae"].iloc[-1]
            trend = "increases" if last_mae > first_mae else "does not increase"
            print(
                f"- Across increasing true_queue_length_m bins, MAE {trend} "
                f"from {_fmt(first_mae)} ({by_bin_sorted.index[0]}) to "
                f"{_fmt(last_mae)} ({by_bin_sorted.index[-1]})."
            )
            worst_bin_row = by_bin_sorted["mae"].idxmax()
            residual_at_worst = by_bin_sorted.loc[worst_bin_row, "mean_residual"]
            direction = "under-" if residual_at_worst < 0 else "over-"
            print(
                f"- Largest-error bin is '{worst_bin_row}'; mean residual "
                f"there is {_fmt(residual_at_worst)}, indicating systematic "
                f"{direction}prediction in that range (sign-based observation only)."
            )

    if not test_by_congestion.empty and len(test_by_congestion) == 2:
        rows_sorted = test_by_congestion.sort_index()
        try:
            mae_false = rows_sorted.loc[False, "mae"] if False in rows_sorted.index else rows_sorted.loc["False", "mae"]
            mae_true = rows_sorted.loc[True, "mae"] if True in rows_sorted.index else rows_sorted.loc["True", "mae"]
            direction = "higher" if mae_true > mae_false else "lower"
            print(
                f"- MAE when queue_reaches_camera_edge=True is {direction} "
                f"than when False ({_fmt(mae_true)} vs {_fmt(mae_false)})."
            )
        except KeyError:
            pass

    ood_gap_mae = ood_overall["mae"] - test_overall["mae"]
    ood_gap_r2 = ood_overall["r2"] - test_overall["r2"]
    print(
        f"- OOD overall MAE is {_fmt(ood_gap_mae)} higher than test overall MAE, "
        f"and OOD R2 is {_fmt(ood_gap_r2)} relative to test R2 "
        f"(negative values indicate worse OOD generalization)."
    )
    print(
        "\nThese are descriptive, data-derived observations only -- no causal "
        "explanation is asserted, and none of this analysis selects or "
        "changes the current model-selection outcome."
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    print("RANDOM FOREST LAYER 2 ERROR ANALYSIS")
    print("=" * len("RANDOM FOREST LAYER 2 ERROR ANALYSIS"))

    model, model_path, config = load_trained_model()
    splits = load_splits()

    print_section("MODEL")
    print(f"Model type      : {type(model).__name__}")
    print(f"Model path      : {model_path}")
    print(f"Layer           : {config.layer.value}")
    print(f"Target column   : {config.target_column}")
    print(f"Random state    : {config.random_state}")
    print(f"Experiment name : {config.experiment_name}")

    # ---- Build row-level DataFrames (in-memory only) ----
    test_split = splits["test"]
    ood_split = splits["ood"]

    test_preds = model.predict(test_split.X)
    ood_preds = model.predict(ood_split.X)

    test_df = build_row_level_df(test_split, test_preds)
    ood_df = build_row_level_df(ood_split, ood_preds)

    # ---- Overall (consistency check against baseline_results.csv) ----
    test_overall = overall_report(test_df)
    ood_overall = overall_report(ood_df)

    print_overall("TEST OVERALL", test_overall)
    print_overall("OOD OVERALL", ood_overall)

    baseline_row = load_baseline_consistency_row()
    print_section("CONSISTENCY CHECK vs baseline_results.csv")
    if baseline_row is not None:
        r = baseline_row.iloc[0]
        print(
            f"baseline_results.csv -> test_mae={_fmt(r['test_mae'])}, "
            f"test_rmse={_fmt(r['test_rmse'])}, test_r2={_fmt(r['test_r2'])}, "
            f"ood_mae={_fmt(r['ood_mae'])}, ood_rmse={_fmt(r['ood_rmse'])}, "
            f"ood_r2={_fmt(r['ood_r2'])}"
        )
        print(
            "(compare against TEST OVERALL / OOD OVERALL above -- values "
            "should match, since both come from the same trained model and splits)"
        )
    else:
        print(f"No matching row found in {_BASELINE_CSV_PATH} for consistency check.")

    # ---- Distribution context for binning assumptions ----
    print_distribution_summary("TRUE QUEUE LENGTH DISTRIBUTION (test)", test_df["true_queue_length_m"])
    print_distribution_summary("TRUE QUEUE LENGTH DISTRIBUTION (ood)", ood_df["true_queue_length_m"])
    if "visible_queue_length_m" in test_df.columns:
        print_distribution_summary(
            "VISIBLE QUEUE LENGTH DISTRIBUTION (test)", test_df["visible_queue_length_m"]
        )

    # ---- 1. Scenario ----
    test_by_scenario = grouped_report(test_df, "scenario_id")
    print_grouped("TEST BY SCENARIO", test_by_scenario)

    ood_by_scenario = grouped_report(ood_df, "scenario_id")
    print_grouped("OOD BY SCENARIO", ood_by_scenario)

    # ---- 2. Approach ----
    test_by_approach = grouped_report(test_df, "approach_edge")
    print_grouped("TEST BY APPROACH", test_by_approach)

    ood_by_approach = grouped_report(ood_df, "approach_edge")
    print_grouped("OOD BY APPROACH", ood_by_approach)

    # ---- 3. True queue-length range ----
    test_df_binned = add_true_queue_bin(test_df)
    test_by_queue_bin = grouped_report(test_df_binned, "true_queue_bin")
    print_grouped("TEST BY TRUE QUEUE LENGTH", test_by_queue_bin)

    ood_df_binned = add_true_queue_bin(ood_df)
    ood_by_queue_bin = grouped_report(ood_df_binned, "true_queue_bin")
    print_grouped("OOD BY TRUE QUEUE LENGTH", ood_by_queue_bin)

    # ---- 4. High-congestion conditions ----
    test_by_congestion = pd.DataFrame()
    if "queue_reaches_camera_edge" in test_df.columns:
        test_by_congestion = grouped_report(test_df, "queue_reaches_camera_edge")
        print_grouped("TEST BY CAMERA-EDGE CENSORING (queue_reaches_camera_edge)", test_by_congestion)
    else:
        print_section("TEST BY CAMERA-EDGE CENSORING")
        print("queue_reaches_camera_edge not present in feature columns -- skipped.")

    test_df_vqbin, visible_bin_edges = add_visible_queue_bin(test_df)
    if visible_bin_edges is not None:
        test_by_visible_bin = grouped_report(test_df_vqbin, "visible_queue_bin")
        print_grouped("TEST BY VISIBLE QUEUE LENGTH (quartile bins)", test_by_visible_bin)
        print(f"Quartile bin edges (visible_queue_length_m): {[round(e, 2) for e in visible_bin_edges]}")
    else:
        print_section("TEST BY VISIBLE QUEUE LENGTH")
        print("visible_queue_length_m not present or insufficiently distinct for quartile bins -- skipped.")

    # ---- 5. queue_reaches_camera_edge already covered above (item 4) ----

    # ---- 6. OOD scenario type (scenario_id itself is sufficient) ----
    print_section("OOD BY SCENARIO (scenario_id used directly as type)")
    print(ood_by_scenario.to_string())

    # ---- Key patterns ----
    print_key_patterns(
        test_by_scenario=test_by_scenario,
        ood_by_scenario=ood_by_scenario,
        test_by_approach=test_by_approach,
        test_by_queue_bin=test_by_queue_bin,
        test_by_congestion=test_by_congestion,
        test_overall=test_overall,
        ood_overall=ood_overall,
    )


if __name__ == "__main__":
    main()