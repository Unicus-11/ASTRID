"""
diagnose_gps_penetration_p11_p25_p50.py
==========================================
DIAGNOSTIC-ONLY -- why does the frozen p11-trained original baseline
HistGradientBoosting model do much better at 25% GPS penetration than at
50%, even though 50% has more GPS probes than both 11% and 25%?

Motivating result (from evaluate_gps_penetration_sensitivity.py), same
frozen model, no retraining:

    GPS penetration   TEST MAE    OOD MAE
    5%                41.670430   61.775240
    11%               18.706834   34.757319
    25%               10.544564   19.404971
    50%               16.520118   28.930702

This script performs NO tuning, NO retraining at p25/p50, NO dataset
changes, NO GPS-observation regeneration, and does not modify any
existing evaluation/tuning script. TEST/OOD results are used only to
diagnose the already-observed 25%->50% drop, never to select or adjust
a model. The final model remains the frozen original baseline artifact
at every step.

Penetration/DatasetLoader note (same approach as
evaluate_gps_penetration_sensitivity.py): experiment_config.Layer only
declares LAYER1 and LAYER2_P11 -- there is no LAYER2_P25/LAYER2_P50
member. DatasetLoader/load_split only ever read `layer.value`, so p25
and p50 are passed a minimal duck-typed `_PenetrationLayer(value=...)`
stand-in instead of a real Layer member. p11 uses the real
Layer.LAYER2_P11. Neither experiment_config.py nor data_loader.py is
modified.

Run:
    python models/results/diagnose_gps_penetration_p11_p25_p50.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

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
_MISSINGNESS_CSV = _RESULTS_DIR / "gps_penetration_feature_missingness_p11_p25_p50.csv"
_DISTRIBUTIONS_CSV = _RESULTS_DIR / "gps_penetration_feature_distributions_p11_p25_p50.csv"
_SCENARIO_METRICS_CSV = _RESULTS_DIR / "gps_penetration_scenario_metrics_p11_p25_p50.csv"
_P25_VS_P50_CSV = _RESULTS_DIR / "gps_penetration_p25_vs_p50_scenario_comparison.csv"
# Optional extra diagnostic CSV (Part 7 -- probe coverage vs. error).
_PROBE_ERROR_CSV = _RESULTS_DIR / "gps_penetration_probe_coverage_vs_error_p11_p25_p50.csv"
# Optional extra diagnostic CSV (Part 4 -- per-scenario GPS diagnostics).
_SCENARIO_GPS_CSV = _RESULTS_DIR / "gps_penetration_scenario_gps_diagnostics_p11_p25_p50.csv"

BASELINE_ARTIFACT_PATH = (
    DEFAULT_OUTPUT_ROOT / "layer2_p11"
    / "hist_gradient_boosting_layer2_p11_baseline"
    / "hist_gradient_boosting.joblib"
)


@dataclass(frozen=True)
class _PenetrationLayer:
    """Duck-typed stand-in for experiment_config.Layer, used for p25/p50
    which have no Layer enum member. DatasetLoader/load_split only ever
    access `.value` -- see module docstring."""

    value: str


PENETRATION_LAYERS: Dict[str, Any] = {
    "p11": Layer.LAYER2_P11,
    "p25": _PenetrationLayer("layer2_p25"),
    "p50": _PenetrationLayer("layer2_p50"),
}

PENETRATION_ORDER: List[str] = ["p11", "p25", "p50"]
PENETRATION_PCT: Dict[str, int] = {"p11": 11, "p25": 25, "p50": 50}
EVAL_SPLITS: List[str] = ["TEST", "OOD"]
_SPLIT_ENUM = {"TEST": Split.TEST, "OOD": Split.OOD}

GPS_FEATURES: List[str] = [
    "probe_count",
    "probe_mean_speed_mps",
    "probe_min_distance_to_stopline_m",
    "probe_max_distance_to_stopline_m",
    "probe_count_change_30s",
    "probe_max_distance_to_stopline_m_change_30s",
]

FEATURE_TO_MISSING_COL: Dict[str, str] = {
    "probe_mean_speed_mps": "missing_probe_mean_speed_pct",
    "probe_min_distance_to_stopline_m": "missing_probe_min_distance_pct",
    "probe_max_distance_to_stopline_m": "missing_probe_max_distance_pct",
    "probe_count_change_30s": "missing_probe_count_change_pct",
    "probe_max_distance_to_stopline_m_change_30s": "missing_probe_max_distance_change_pct",
}

SCENARIO_SPLIT_MAP: Dict[str, str] = {
    "scenario_east_west_heavy": "TEST",
    "scenario_south_heavy": "TEST",
    "scenario_burst_demand_OOD": "OOD",
    "scenario_heavy_vehicle_OOD": "OOD",
    "scenario_north_extreme_OOD": "OOD",
    "scenario_very_high_demand_OOD": "OOD",
}

FORBIDDEN_MODEL_INPUT_COLUMNS = {
    "scenario_id",
    "split",
    "design_method",
    "gps_penetration_rate_requested",
    "true_queue_length_m",
    "true_queue_beyond_camera",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_splits() -> Dict[str, Dict[str, Any]]:
    splits: Dict[str, Dict[str, Any]] = {}
    for tag in PENETRATION_ORDER:
        loader = DatasetLoader(layer=PENETRATION_LAYERS[tag])
        splits[tag] = {
            "TEST": loader.load(Split.TEST),
            "OOD": loader.load(Split.OOD),
        }
    return splits


# ---------------------------------------------------------------------------
# Part 1 -- feature schema check
# ---------------------------------------------------------------------------

def verify_feature_schema(splits: Dict[str, Dict[str, Any]]) -> List[str]:
    print("A. FEATURE SCHEMA CHECK")
    print("-" * 84)

    reference_features = splits["p11"]["TEST"].feature_columns
    for tag in PENETRATION_ORDER:
        cols = splits[tag]["TEST"].feature_columns
        print(f"  {tag}: n_features = {len(cols)}")
        print(f"    columns (in order) = {cols}")

    mismatches = []
    for tag in PENETRATION_ORDER:
        for split_label in EVAL_SPLITS:
            cols = splits[tag][split_label].feature_columns
            same_columns = set(cols) == set(reference_features)
            same_order = cols == reference_features
            if not (same_columns and same_order):
                mismatches.append((tag, split_label, cols, same_columns, same_order))

    forbidden_present = FORBIDDEN_MODEL_INPUT_COLUMNS & set(reference_features)

    print()
    print(f"  same feature schema across p11/p25/p50 (TEST & OOD): "
          f"{'YES' if not mismatches else 'NO'}")
    print(f"  forbidden columns present in model features: "
          f"{sorted(forbidden_present) if forbidden_present else 'NONE'}")
    print()

    if mismatches or forbidden_present:
        for tag, split_label, cols, same_columns, same_order in mismatches:
            missing = [c for c in reference_features if c not in cols]
            extra = [c for c in cols if c not in reference_features]
            print(f"  MISMATCH [{tag}/{split_label}] same_columns={same_columns} "
                  f"same_order={same_order} missing={missing} extra={extra}")
        raise SystemExit(
            "STOPPING: feature schema differs across p11/p25/p50 (or a forbidden "
            "column is present as a model feature). Refusing to proceed -- fix the "
            "dataset/manifest and re-run."
        )

    return reference_features


# ---------------------------------------------------------------------------
# Part 2 -- GPS feature missingness
# ---------------------------------------------------------------------------

def compute_missingness(splits: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for tag in PENETRATION_ORDER:
        for split_label in EVAL_SPLITS:
            X = splits[tag][split_label].X
            n_rows = len(X)
            for feat in GPS_FEATURES:
                series = X[feat]
                n_missing = int(series.isna().sum())
                n_nonmissing = n_rows - n_missing
                missing_pct = (n_missing / n_rows * 100.0) if n_rows else float("nan")

                n_zero: Optional[int] = None
                zero_pct: Optional[float] = None
                if feat == "probe_count":
                    # NaN is never counted as zero here -- only actual 0 values.
                    n_zero = int((series == 0).sum())
                    zero_pct = (n_zero / n_rows * 100.0) if n_rows else float("nan")

                rows.append({
                    "penetration": PENETRATION_PCT[tag],
                    "penetration_tag": tag,
                    "split": split_label,
                    "feature": feat,
                    "n_rows": n_rows,
                    "n_missing": n_missing,
                    "missing_pct": missing_pct,
                    "n_nonmissing": n_nonmissing,
                    "n_zero": n_zero,
                    "zero_pct": zero_pct,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 3 -- GPS feature distributions
# ---------------------------------------------------------------------------

def compute_distributions(splits: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for tag in PENETRATION_ORDER:
        for split_label in EVAL_SPLITS:
            X = splits[tag][split_label].X
            for feat in GPS_FEATURES:
                s = X[feat].dropna()
                if len(s) == 0:
                    stats = {"count": 0, "mean": None, "std": None, "min": None,
                             "p25": None, "median": None, "p75": None, "max": None}
                else:
                    stats = {
                        "count": int(s.count()),
                        "mean": float(s.mean()),
                        "std": float(s.std()),
                        "min": float(s.min()),
                        "p25": float(s.quantile(0.25)),
                        "median": float(s.median()),
                        "p75": float(s.quantile(0.75)),
                        "max": float(s.max()),
                    }
                rows.append({
                    "penetration": PENETRATION_PCT[tag],
                    "penetration_tag": tag,
                    "split": split_label,
                    "feature": feat,
                    **stats,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 4 -- per-scenario GPS diagnostics
# ---------------------------------------------------------------------------

def compute_scenario_gps_diagnostics(splits: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for scenario, split_label in SCENARIO_SPLIT_MAP.items():
        for tag in PENETRATION_ORDER:
            split_data = splits[tag][split_label]
            scenario_ids = split_data.metadata["scenario_id"]
            mask = (scenario_ids == scenario).to_numpy()
            X_scn = split_data.X[mask]
            n_rows = len(X_scn)

            probe_count = X_scn["probe_count"]
            mean_probe_count = float(probe_count.mean()) if n_rows else float("nan")
            median_probe_count = float(probe_count.median()) if n_rows else float("nan")
            zero_probe_pct = (float((probe_count == 0).sum()) / n_rows * 100.0) if n_rows else float("nan")

            def missing_pct(col: str) -> float:
                return (float(X_scn[col].isna().sum()) / n_rows * 100.0) if n_rows else float("nan")

            rows.append({
                "scenario": scenario,
                "split": split_label,
                "penetration": PENETRATION_PCT[tag],
                "penetration_tag": tag,
                "n_rows": n_rows,
                "mean_probe_count": mean_probe_count,
                "median_probe_count": median_probe_count,
                "zero_probe_pct": zero_probe_pct,
                "missing_probe_mean_speed_pct": missing_pct("probe_mean_speed_mps"),
                "missing_probe_min_distance_pct": missing_pct("probe_min_distance_to_stopline_m"),
                "missing_probe_max_distance_pct": missing_pct("probe_max_distance_to_stopline_m"),
                "missing_probe_count_change_pct": missing_pct("probe_count_change_30s"),
                "missing_probe_max_distance_change_pct": missing_pct("probe_max_distance_to_stopline_m_change_30s"),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 5 -- per-scenario (and aggregate) model performance
# ---------------------------------------------------------------------------

def predict_all(splits: Dict[str, Dict[str, Any]], baseline_model: Any) -> Dict[Any, np.ndarray]:
    preds_cache: Dict[Any, np.ndarray] = {}
    for tag in PENETRATION_ORDER:
        for split_label in EVAL_SPLITS:
            split_data = splits[tag][split_label]
            preds_cache[(tag, split_label)] = baseline_model.predict(split_data.X)
    return preds_cache


def compute_aggregate_metrics(
    splits: Dict[str, Dict[str, Any]], preds_cache: Dict[Any, np.ndarray]
) -> pd.DataFrame:
    rows = []
    for tag in PENETRATION_ORDER:
        for split_label in EVAL_SPLITS:
            split_data = splits[tag][split_label]
            m = compute_all_metrics(split_data.y, preds_cache[(tag, split_label)])
            rows.append({
                "penetration": PENETRATION_PCT[tag], "penetration_tag": tag,
                "split": split_label, "mae": m["mae"], "rmse": m["rmse"],
                "r2": m["r2"], "n": m["n"],
            })
    return pd.DataFrame(rows)


def compute_scenario_model_performance(
    splits: Dict[str, Dict[str, Any]], preds_cache: Dict[Any, np.ndarray]
) -> pd.DataFrame:
    rows = []
    for scenario, split_label in SCENARIO_SPLIT_MAP.items():
        for tag in PENETRATION_ORDER:
            split_data = splits[tag][split_label]
            scenario_ids = split_data.metadata["scenario_id"]
            mask = (scenario_ids == scenario).to_numpy()

            y_scn = split_data.y[mask]
            preds_scn = preds_cache[(tag, split_label)][mask]
            m = compute_all_metrics(y_scn, preds_scn)

            rows.append({
                "split": split_label,
                "scenario": scenario,
                "penetration": PENETRATION_PCT[tag],
                "penetration_tag": tag,
                "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
                "n": m["n"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 6 -- p25 vs p50 compact comparison
# ---------------------------------------------------------------------------

def compute_p25_vs_p50(scenario_metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, split_label in SCENARIO_SPLIT_MAP.items():
        p25_row = scenario_metrics_df[
            (scenario_metrics_df["scenario"] == scenario)
            & (scenario_metrics_df["penetration_tag"] == "p25")
        ].iloc[0]
        p50_row = scenario_metrics_df[
            (scenario_metrics_df["scenario"] == scenario)
            & (scenario_metrics_df["penetration_tag"] == "p50")
        ].iloc[0]

        mae_change_pct = (p25_row["mae"] - p50_row["mae"]) / p25_row["mae"] * 100.0
        rmse_change_pct = (p25_row["rmse"] - p50_row["rmse"]) / p25_row["rmse"] * 100.0
        r2_change = p50_row["r2"] - p25_row["r2"]

        rows.append({
            "split": split_label,
            "scenario": scenario,
            "p25_mae": p25_row["mae"],
            "p50_mae": p50_row["mae"],
            "mae_change_pct_p50_vs_p25": mae_change_pct,
            "p25_rmse": p25_row["rmse"],
            "p50_rmse": p50_row["rmse"],
            "rmse_change_pct_p50_vs_p25": rmse_change_pct,
            "p25_r2": p25_row["r2"],
            "p50_r2": p50_row["r2"],
            "r2_change_p50_vs_p25": r2_change,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 7 -- probe coverage vs. error (descriptive only, not causal)
# ---------------------------------------------------------------------------

def compute_probe_error_relationship(
    splits: Dict[str, Dict[str, Any]], preds_cache: Dict[Any, np.ndarray]
) -> pd.DataFrame:
    rows = []
    for scenario, split_label in SCENARIO_SPLIT_MAP.items():
        for tag in PENETRATION_ORDER:
            split_data = splits[tag][split_label]
            scenario_ids = split_data.metadata["scenario_id"]
            mask = (scenario_ids == scenario).to_numpy()

            X_scn = split_data.X[mask].reset_index(drop=True)
            y_scn = split_data.y[mask].to_numpy()
            preds_scn = preds_cache[(tag, split_label)][mask]
            abs_error = pd.Series(np.abs(y_scn - preds_scn))

            def corr_with(col: str) -> Optional[float]:
                sub = pd.concat([X_scn[col], abs_error], axis=1).dropna()
                if len(sub) < 2:
                    return None
                return float(sub.iloc[:, 0].corr(sub.iloc[:, 1]))

            probe_count = X_scn["probe_count"]
            zero_mask = (probe_count == 0).to_numpy()
            nonzero_mask = (probe_count > 0).to_numpy()

            mean_ae_zero = float(abs_error[zero_mask].mean()) if zero_mask.sum() > 0 else None
            mean_ae_nonzero = float(abs_error[nonzero_mask].mean()) if nonzero_mask.sum() > 0 else None

            rows.append({
                "scenario": scenario,
                "split": split_label,
                "penetration": PENETRATION_PCT[tag],
                "penetration_tag": tag,
                "n_rows": len(X_scn),
                "corr_probe_count_vs_abs_error": corr_with("probe_count"),
                "corr_probe_mean_speed_vs_abs_error": corr_with("probe_mean_speed_mps"),
                "corr_probe_max_distance_vs_abs_error": corr_with("probe_max_distance_to_stopline_m"),
                "n_zero_probe": int(zero_mask.sum()),
                "n_nonzero_probe": int(nonzero_mask.sum()),
                "mean_abs_error_probe_count_zero": mean_ae_zero,
                "mean_abs_error_probe_count_nonzero": mean_ae_nonzero,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 8 -- feature-distribution shift, p25 -> p50 (measured diffs, no
# invented thresholds)
# ---------------------------------------------------------------------------

def print_distribution_shift(missingness_df: pd.DataFrame, distributions_df: pd.DataFrame) -> None:
    print("PART 8: FEATURE-DISTRIBUTION SHIFT -- MEASURED DIFFERENCES (p25 -> p50)")
    print("-" * 84)
    for split_label in EVAL_SPLITS:
        for feat in GPS_FEATURES:
            m25 = missingness_df[
                (missingness_df["split"] == split_label)
                & (missingness_df["penetration_tag"] == "p25")
                & (missingness_df["feature"] == feat)
            ].iloc[0]
            m50 = missingness_df[
                (missingness_df["split"] == split_label)
                & (missingness_df["penetration_tag"] == "p50")
                & (missingness_df["feature"] == feat)
            ].iloc[0]
            d25 = distributions_df[
                (distributions_df["split"] == split_label)
                & (distributions_df["penetration_tag"] == "p25")
                & (distributions_df["feature"] == feat)
            ].iloc[0]
            d50 = distributions_df[
                (distributions_df["split"] == split_label)
                & (distributions_df["penetration_tag"] == "p50")
                & (distributions_df["feature"] == feat)
            ].iloc[0]

            missing_pct_diff = m50["missing_pct"] - m25["missing_pct"]
            mean_diff = (d50["mean"] - d25["mean"]) if d50["mean"] is not None and d25["mean"] is not None else None
            median_diff = (d50["median"] - d25["median"]) if d50["median"] is not None and d25["median"] is not None else None
            std_diff = (d50["std"] - d25["std"]) if d50["std"] is not None and d25["std"] is not None else None

            print(f"  [{split_label}] {feat}:")
            print(f"    missing_pct   : p25={m25['missing_pct']:.4f}%  p50={m50['missing_pct']:.4f}%  diff={missing_pct_diff:+.4f}pp")
            print(f"    mean          : p25={d25['mean']}  p50={d50['mean']}  diff={mean_diff}")
            print(f"    median        : p25={d25['median']}  p50={d50['median']}  diff={median_diff}")
            print(f"    std           : p25={d25['std']}  p50={d50['std']}  diff={std_diff}")

            if feat == "probe_count":
                z25, z50 = m25["zero_pct"], m50["zero_pct"]
                print(f"    zero_probe_pct: p25={z25:.4f}%  p50={z50:.4f}%  diff={(z50 - z25):+.4f}pp")
    print()


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float], d: int = 6) -> str:
    if v is None:
        return "None"
    return f"{v:.{d}f}"


def print_missingness_table(missingness_df: pd.DataFrame) -> None:
    print("B. MISSINGNESS SUMMARY")
    print("-" * 84)
    display_cols = ["penetration", "split", "feature", "n_rows", "n_missing",
                     "missing_pct", "n_nonmissing", "n_zero", "zero_pct"]
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(missingness_df[display_cols].to_string(index=False))
    print()


def print_distributions_table(distributions_df: pd.DataFrame) -> None:
    print("C. DISTRIBUTION SUMMARY")
    print("-" * 84)
    display_cols = ["penetration", "split", "feature", "count", "mean", "std",
                     "min", "p25", "median", "p75", "max"]
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(distributions_df[display_cols].round(6).to_string(index=False))
    print()


def print_scenario_gps_table(scenario_gps_df: pd.DataFrame) -> None:
    print("PART 4: PER-SCENARIO GPS DIAGNOSTICS")
    print("-" * 84)
    with pd.option_context("display.width", 240, "display.max_columns", None):
        print(scenario_gps_df.round(4).to_string(index=False))
    print()


def print_aggregate_table(aggregate_df: pd.DataFrame) -> None:
    print("AGGREGATE TEST/OOD METRICS (frozen model, p11/p25/p50) -- context only")
    print("-" * 84)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(aggregate_df.round(6).to_string(index=False))
    print()


def print_scenario_performance_table(scenario_metrics_df: pd.DataFrame) -> None:
    print("D. SCENARIO PERFORMANCE")
    print("-" * 84)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(scenario_metrics_df.round(6).to_string(index=False))
    print()


def print_p25_vs_p50_table(comparison_df: pd.DataFrame) -> None:
    print("E. p25 VS p50 COMPARISON")
    print("-" * 84)
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(comparison_df.round(6).to_string(index=False))
    print()


def print_probe_error_table(probe_error_df: pd.DataFrame) -> None:
    print("PART 7: PROBE COVERAGE VS. ABSOLUTE ERROR (descriptive only, not causal)")
    print("-" * 84)
    with pd.option_context("display.width", 240, "display.max_columns", None):
        print(probe_error_df.round(6).to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# F. Final diagnostic conclusion
# ---------------------------------------------------------------------------

def print_final_conclusion(
    comparison_df: pd.DataFrame,
    missingness_df: pd.DataFrame,
    distributions_df: pd.DataFrame,
) -> None:
    print("F. FINAL DIAGNOSTIC CONCLUSION")
    print("-" * 84)

    test_rows = comparison_df[comparison_df["split"] == "TEST"]
    ood_rows = comparison_df[comparison_df["split"] == "OOD"]

    test_all_degraded = bool((test_rows["mae_change_pct_p50_vs_p25"] < 0).all())
    ood_all_degraded = bool((ood_rows["mae_change_pct_p50_vs_p25"] < 0).all())

    print(f"1. Is the p50 degradation present across both TEST scenarios? "
          f"{'SUPPORTED BY THE DIAGNOSTIC' if test_all_degraded else 'NOT SUPPORTED BY THE DIAGNOSTIC'} "
          f"-- per-scenario MAE change p50 vs p25 (TEST): "
          f"{dict(zip(test_rows['scenario'], test_rows['mae_change_pct_p50_vs_p25'].round(2)))}")
    print(f"2. Is the p50 degradation present across OOD scenarios? "
          f"{'SUPPORTED BY THE DIAGNOSTIC' if ood_all_degraded else 'NOT SUPPORTED BY THE DIAGNOSTIC'} "
          f"-- per-scenario MAE change p50 vs p25 (OOD): "
          f"{dict(zip(ood_rows['scenario'], ood_rows['mae_change_pct_p50_vs_p25'].round(2)))}")

    # Question 3: is one scenario responsible for most of the total
    # degradation? Measured as: does the single worst scenario's
    # degradation account for more than half of the summed degradation
    # across all scenarios that degraded? (a majority share, not an
    # arbitrary invented cutoff)
    degraded = comparison_df[comparison_df["mae_change_pct_p50_vs_p25"] < 0].copy()
    if len(degraded) == 0:
        q3 = "NOT SUPPORTED BY THE DIAGNOSTIC -- no scenario degraded from p25 to p50 by MAE."
    else:
        degraded["magnitude"] = -degraded["mae_change_pct_p50_vs_p25"]
        total_magnitude = degraded["magnitude"].sum()
        worst = degraded.loc[degraded["magnitude"].idxmax()]
        worst_share = worst["magnitude"] / total_magnitude * 100.0 if total_magnitude else 0.0
        verdict = "SUPPORTED BY THE DIAGNOSTIC" if worst_share > 50.0 else "NOT SUPPORTED BY THE DIAGNOSTIC"
        q3 = (f"{verdict} -- {worst['scenario']} ({worst['split']}) accounts for "
              f"{worst_share:.1f}% of total summed MAE degradation magnitude across "
              f"degraded scenarios ({worst['magnitude']:.2f} of {total_magnitude:.2f} percentage-points).")
    print(f"3. Is one scenario responsible for most of the p50 degradation? {q3}")

    # Question 4/5: measured missingness / distribution diffs, p25 -> p50.
    missingness_diffs = []
    dist_mean_diffs = []
    dist_median_diffs = []
    dist_std_diffs = []
    for split_label in EVAL_SPLITS:
        for feat in GPS_FEATURES:
            m25 = missingness_df[(missingness_df["split"] == split_label) & (missingness_df["penetration_tag"] == "p25") & (missingness_df["feature"] == feat)].iloc[0]
            m50 = missingness_df[(missingness_df["split"] == split_label) & (missingness_df["penetration_tag"] == "p50") & (missingness_df["feature"] == feat)].iloc[0]
            missingness_diffs.append(abs(m50["missing_pct"] - m25["missing_pct"]))

            d25 = distributions_df[(distributions_df["split"] == split_label) & (distributions_df["penetration_tag"] == "p25") & (distributions_df["feature"] == feat)].iloc[0]
            d50 = distributions_df[(distributions_df["split"] == split_label) & (distributions_df["penetration_tag"] == "p50") & (distributions_df["feature"] == feat)].iloc[0]
            if d25["mean"] is not None and d50["mean"] is not None:
                dist_mean_diffs.append(abs(d50["mean"] - d25["mean"]))
                dist_median_diffs.append(abs(d50["median"] - d25["median"]))
                dist_std_diffs.append(abs(d50["std"] - d25["std"]))

    max_missingness_diff = max(missingness_diffs) if missingness_diffs else 0.0
    max_mean_diff = max(dist_mean_diffs) if dist_mean_diffs else 0.0
    max_median_diff = max(dist_median_diffs) if dist_median_diffs else 0.0
    max_std_diff = max(dist_std_diffs) if dist_std_diffs else 0.0

    q4_verdict = "SUPPORTED BY THE DIAGNOSTIC" if max_missingness_diff > 0.01 else "NOT SUPPORTED BY THE DIAGNOSTIC"
    print(f"4. Does p50 have materially different GPS-feature missingness from p25? "
          f"{q4_verdict} -- largest |missing_pct| difference across all GPS features/splits "
          f"= {max_missingness_diff:.4f} percentage points (see Part 8 for the full breakdown).")

    q5_verdict = ("SUPPORTED BY THE DIAGNOSTIC"
                   if (max_mean_diff > 1e-6 or max_median_diff > 1e-6 or max_std_diff > 1e-6)
                   else "NOT SUPPORTED BY THE DIAGNOSTIC")
    print(f"5. Does p50 have materially different GPS-feature distributions from p25? "
          f"{q5_verdict} -- largest |mean| diff = {max_mean_diff:.6f}, "
          f"largest |median| diff = {max_median_diff:.6f}, largest |std| diff = {max_std_diff:.6f} "
          f"(see Part 8 for the full per-feature breakdown).")

    both_findings = (max_missingness_diff > 0.01) or (max_mean_diff > 1e-6) or (max_median_diff > 1e-6) or (max_std_diff > 1e-6)
    q6_verdict = ("SUPPORTED BY THE DIAGNOSTIC" if (test_all_degraded or ood_all_degraded) and both_findings
                  else "NOT SUPPORTED BY THE DIAGNOSTIC")
    print(f"6. Does the degradation coincide with GPS-feature missingness/distribution changes "
          f"(including zero-probe frequency, per Part 8)? {q6_verdict} -- this states co-occurrence "
          f"only, not causation; see Part 7 for the probe-coverage-vs-error breakdown and Part 8 "
          f"for the underlying feature shifts.")

    q7_verdict = ("SUPPORTED BY THE DIAGNOSTIC -- coincidence between measured GPS-feature shifts "
                  "and the MAE/RMSE degradation is present in the data above, though this diagnostic "
                  "establishes coincidence, not a mechanism."
                  if (test_all_degraded or ood_all_degraded) and both_findings
                  else "NOT SUPPORTED BY THE DIAGNOSTIC -- the measured evidence here does not, by "
                  "itself, explain the p25->p50 performance drop; the cause is UNRESOLVED and would "
                  "need further investigation.")
    print(f"7. Is the observed evidence sufficient to explain the p25 -> p50 performance drop? "
          f"{q7_verdict}")
    print()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    print("DIAGNOSTIC -- why does 25% GPS penetration outperform 50%, at fixed model?")
    print("Diagnostic-only. No tuning, no retraining, no dataset changes.")
    print("=" * 84)
    print()
    print("Motivating result (frozen model, from evaluate_gps_penetration_sensitivity.py):")
    print("  5%  : TEST MAE=41.670430   OOD MAE=61.775240")
    print("  11% : TEST MAE=18.706834   OOD MAE=34.757319")
    print("  25% : TEST MAE=10.544564   OOD MAE=19.404971")
    print("  50% : TEST MAE=16.520118   OOD MAE=28.930702")
    print()

    print("ORIGINAL BASELINE -- loaded from existing artifact (NOT refit)")
    print("-" * 84)
    print(f"  artifact path = {BASELINE_ARTIFACT_PATH}")
    if not BASELINE_ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Baseline artifact not found: {BASELINE_ARTIFACT_PATH}")
    baseline_model = load_model(BASELINE_ARTIFACT_PATH)
    print("  status = loaded once; reused unmodified across p11/p25/p50")
    print()

    splits = load_all_splits()

    # ---- Part 1 ----
    verify_feature_schema(splits)

    # ---- Part 2 ----
    missingness_df = compute_missingness(splits)
    print_missingness_table(missingness_df)

    # ---- Part 3 ----
    distributions_df = compute_distributions(splits)
    print_distributions_table(distributions_df)

    # ---- Part 4 ----
    scenario_gps_df = compute_scenario_gps_diagnostics(splits)
    print_scenario_gps_table(scenario_gps_df)

    # ---- Part 5 ----
    preds_cache = predict_all(splits, baseline_model)
    aggregate_df = compute_aggregate_metrics(splits, preds_cache)
    print_aggregate_table(aggregate_df)
    scenario_metrics_df = compute_scenario_model_performance(splits, preds_cache)
    print_scenario_performance_table(scenario_metrics_df)

    # ---- Part 6 ----
    comparison_df = compute_p25_vs_p50(scenario_metrics_df)
    print_p25_vs_p50_table(comparison_df)

    # ---- Part 7 ----
    probe_error_df = compute_probe_error_relationship(splits, preds_cache)
    print_probe_error_table(probe_error_df)

    # ---- Part 8 ----
    print_distribution_shift(missingness_df, distributions_df)

    # ---- Part 10.F ----
    print_final_conclusion(comparison_df, missingness_df, distributions_df)

    print("This was a diagnostic-only run. No model was tuned or retrained; the")
    print("frozen original baseline artifact remains the final model at 11%, 25%,")
    print("and 50% GPS penetration alike.")
    print()

    # ---- Part 9: save CSVs ----
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    missingness_df.to_csv(_MISSINGNESS_CSV, index=False)
    distributions_df.to_csv(_DISTRIBUTIONS_CSV, index=False)
    scenario_metrics_df.to_csv(_SCENARIO_METRICS_CSV, index=False)
    comparison_df.to_csv(_P25_VS_P50_CSV, index=False)
    # Optional extra diagnostic CSVs.
    scenario_gps_df.to_csv(_SCENARIO_GPS_CSV, index=False)
    probe_error_df.to_csv(_PROBE_ERROR_CSV, index=False)

    print("CSV files written:")
    for path in (
        _MISSINGNESS_CSV, _DISTRIBUTIONS_CSV, _SCENARIO_METRICS_CSV, _P25_VS_P50_CSV,
        _SCENARIO_GPS_CSV, _PROBE_ERROR_CSV,
    ):
        print(f"  {path.resolve()}")


if __name__ == "__main__":
    main()