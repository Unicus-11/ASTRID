"""
final_hybrid_evaluation.py
====================
Read-only, analysis-only FINAL evaluation of the frozen `congestion_flag`
hybrid routing rule over the two already-trained Layer 2 baseline models
(Random Forest and HistGradientBoosting).

FROZEN ROUTING RULE -- DO NOT MODIFY:
    queue_reaches_camera_edge == True  -> HistGradientBoosting
    queue_reaches_camera_edge == False -> Random Forest

This rule was selected on the Layer 2 VALIDATION split in
validation_router_investigation.py, where it was compared against RF
alone, Hist alone, and several other candidate rules (a signal-state
lookup, an approach_edge lookup, and a family of visible_queue_length_m
threshold rules) and won by validation MAE. This script does not repeat,
re-run, or re-derive that selection -- it takes `congestion_flag` as
given, applies it completely unchanged, and reports how it does on TEST
and OOD, the two splits validation_router_investigation.py never touched.

This script intentionally:
    * loads NO validation data anywhere (Split.VALIDATION is never
      imported, referenced, or loaded in this file)
    * does NOT search for, compare against, or construct any other
      candidate rule, threshold, or routing condition
    * does NOT recalibrate, refit, or adjust `congestion_flag` in any way
    * does NOT train, fit, or modify either model
    * does NOT modify data_loader.py, evaluate.py, experiment_runner.py,
      metrics.py, persistence.py, any model implementation, or either of
      the two analysis scripts (evaluate_hybrid_router.py,
      validation_router_investigation.py) that preceded it
    * writes nothing to disk -- no CSV/JSON output, print()-only

Structurally, this file mirrors evaluate_hybrid_router.py's loading and
routing conventions exactly (same ExperimentConfig / load_model /
DatasetLoader / compute_all_metrics pattern), trimmed to the single
frozen rule and extended with explicit hybrid-vs-RF / hybrid-vs-Hist
metric deltas, per the final-evaluation requirements this script exists
to satisfy.

ROUTING-SIGNAL / LEAKAGE GUARANTEE:
    true_queue_length_m (the target column) is read from SplitData.y and
    used ONLY inside compute_all_metrics() calls, for evaluation. The
    routing decision (build_route_mask, below) is built exclusively from
    split.X["queue_reaches_camera_edge"] -- an observed, camera-derived
    feature that is already part of each model's own input -- and never
    reads split.y at any point. build_hybrid_predictions() does not even
    accept y as an argument, so this is enforced structurally, not just
    by convention.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# models/results/final_hybrid_evaluation.py -> models/ is the parent's
# parent.
_MODELS_DIR = Path(__file__).resolve().parent.parent
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from data_loader import DatasetLoader, SplitData  # noqa: E402
from experiment_config import ExperimentConfig, Layer, Split  # noqa: E402
from persistence import load_model  # noqa: E402
from metrics import compute_all_metrics  # noqa: E402

RANDOM_STATE = 42
LAYER = Layer.LAYER2_P11

# Routing feature. Must already be one of the manifest's declared
# Layer 2 feature_columns (an observed camera field, never a label or
# metadata column) -- see the leakage guarantee in the module docstring.
# This is the FROZEN rule's only input; it is not searched over here.
ROUTING_COLUMN = "queue_reaches_camera_edge"
ROUTING_RULE_NAME = "congestion_flag"
ROUTING_RULE_DIRECTION = f"{ROUTING_COLUMN}==True -> Hist, {ROUTING_COLUMN}==False -> RF"

# ---------------------------------------------------------------------------
# Model identity, mirrors evaluate_hybrid_router.py / error_analysis.py /
# error_analysis_random_forest.py exactly -- same artifact paths, nothing
# re-derived or guessed.
# ---------------------------------------------------------------------------
RF_EXPERIMENT_NAME = "random_forest_layer2_p11_baseline"
RF_MODEL_FILE_NAME = "random_forest.joblib"

HIST_EXPERIMENT_NAME = "hist_gradient_boosting_layer2_p11_baseline"
HIST_MODEL_FILE_NAME = "hist_gradient_boosting.joblib"


# ---------------------------------------------------------------------------
# Loading (no retraining, no dataset modification, no validation data)
# ---------------------------------------------------------------------------

def load_trained_model(experiment_name: str, model_file_name: str):
    """Load one already-trained, already-persisted model artifact.
    Identical mechanism to evaluate_hybrid_router.py -- ExperimentConfig
    resolves the artifact directory, persistence.load_model()
    deserializes it. No fitting happens here."""
    config = ExperimentConfig(
        layer=LAYER,
        random_state=RANDOM_STATE,
        experiment_name=experiment_name,
    )
    model_path = config.output_dir() / model_file_name
    model = load_model(model_path)
    return model, model_path, config


def load_splits() -> Dict[str, SplitData]:
    """Loads ONLY Split.TEST and Split.OOD. Split.VALIDATION is
    intentionally never imported or referenced anywhere in this module --
    the rule it was used to select is taken as frozen input here."""
    loader = DatasetLoader(layer=LAYER)
    return {
        "test": loader.load(Split.TEST),
        "ood": loader.load(Split.OOD),
    }


# ---------------------------------------------------------------------------
# Frozen routing rule (reads ONLY split.X[ROUTING_COLUMN] -- never split.y)
# ---------------------------------------------------------------------------

def build_route_mask(X: pd.DataFrame) -> pd.Series:
    """True where the row is routed to HistGradientBoosting
    (queue_reaches_camera_edge == True); False where it is routed to
    Random Forest (queue_reaches_camera_edge == False).

    This is the exact, frozen `congestion_flag` rule from
    validation_router_investigation.py -- no direction search, no
    threshold search, no recalibration happens here or anywhere in this
    file. Built ONLY from X -- the observed camera feature -- never
    from y. data_loader.py's _normalize_boolean_like_feature_columns()
    already coerces this column to numeric 1.0/0.0 (with real missing
    values left as NaN, never fabricated), so the comparison below
    covers both that normalized numeric form and a raw boolean/string
    form defensively.
    """
    if ROUTING_COLUMN not in X.columns:
        raise KeyError(
            f"'{ROUTING_COLUMN}' is not a feature column for this split -- "
            f"the frozen hybrid rule has no basis to route on. Check manifest.json."
        )
    col = X[ROUTING_COLUMN]
    is_true = col.isin([True, 1, 1.0, "True", "true"])
    return is_true


def build_hybrid_predictions(
    X: pd.DataFrame, rf_preds: np.ndarray, hist_preds: np.ndarray
) -> "tuple[np.ndarray, pd.Series]":
    """Row-wise selection between two already-computed prediction arrays,
    using ONLY the frozen routing feature in X. Does not accept or touch
    y anywhere in this function -- routing cannot see the target by
    construction, not merely by convention.

    Rows where ROUTING_COLUMN is missing (NaN -- a real, observed gap in
    the camera feature, not fabricated) default to the Random Forest
    prediction, matching the rule's own False branch, since
    queue_reaches_camera_edge missing is not
    queue_reaches_camera_edge==True. This default is applied to a
    COUNTED, REPORTED subset (see n_missing_routing_col in main()) rather
    than silently.
    """
    route_to_hist = build_route_mask(X).to_numpy()
    hybrid = np.where(route_to_hist, hist_preds, rf_preds)
    return hybrid, build_route_mask(X)


# ---------------------------------------------------------------------------
# Metrics (reuses metrics.compute_all_metrics -- no reimplementation)
# ---------------------------------------------------------------------------

def eval_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return compute_all_metrics(y_true, y_pred)


def metric_diff(hybrid_metrics: dict, other_metrics: dict) -> dict:
    """hybrid - other, for mae/rmse/r2. Negative mae/rmse means the
    hybrid is better (lower error); positive r2 diff means the hybrid is
    better (higher r2). Purely arithmetic on already-computed metrics --
    no new predictions or rule evaluation happens here."""
    return {
        "mae_diff": hybrid_metrics["mae"] - other_metrics["mae"],
        "rmse_diff": hybrid_metrics["rmse"] - other_metrics["rmse"],
        "r2_diff": hybrid_metrics["r2"] - other_metrics["r2"],
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _fmt(value, decimals: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NA"
    return f"{value:.{decimals}f}"


def _fmt_signed(value, decimals: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NA"
    return f"{value:+.{decimals}f}"


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def print_comparison_table(
    label: str,
    rf_metrics: dict,
    hist_metrics: dict,
    hybrid_metrics: dict,
) -> None:
    print_section(label)
    header = f"{'metric':<8} {'RF':>14} {'HistGradientBoosting':>22} {'Hybrid (congestion_flag)':>26}"
    print(header)
    print("-" * len(header))
    for key, title in (("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R2")):
        print(
            f"{title:<8} {_fmt(rf_metrics[key]):>14} "
            f"{_fmt(hist_metrics[key]):>22} {_fmt(hybrid_metrics[key]):>26}"
        )
    print(f"{'n':<8} {rf_metrics['n']:>14} {hist_metrics['n']:>22} {hybrid_metrics['n']:>26}")


def print_diff_table(label: str, diff_vs_rf: dict, diff_vs_hist: dict) -> None:
    print_section(f"{label} -- Hybrid vs standalone models (hybrid minus other; MAE/RMSE: negative=better, R2: positive=better)")
    header = f"{'metric':<10} {'vs RF':>14} {'vs HistGradientBoosting':>24}"
    print(header)
    print("-" * len(header))
    print(f"{'MAE':<10} {_fmt_signed(diff_vs_rf['mae_diff']):>14} {_fmt_signed(diff_vs_hist['mae_diff']):>24}")
    print(f"{'RMSE':<10} {_fmt_signed(diff_vs_rf['rmse_diff']):>14} {_fmt_signed(diff_vs_hist['rmse_diff']):>24}")
    print(f"{'R2':<10} {_fmt_signed(diff_vs_rf['r2_diff']):>14} {_fmt_signed(diff_vs_hist['r2_diff']):>24}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    title = "LAYER 2 FINAL HYBRID EVALUATION -- FROZEN 'congestion_flag' RULE (TEST + OOD)"
    print(title)
    print("=" * len(title))
    print(
        f"\nRouting rule '{ROUTING_RULE_NAME}' ({ROUTING_RULE_DIRECTION}) was selected "
        "using the VALIDATION split ONLY, in validation_router_investigation.py, "
        "where it was compared against RF alone, Hist alone, and several other "
        "candidate rules and won on validation MAE.\n"
        "This script does not load, reference, or use validation data anywhere. "
        "It applies the rule completely UNCHANGED and reports its performance on "
        "TEST and OOD -- the only two splits evaluated here."
    )

    rf_model, rf_model_path, rf_config = load_trained_model(RF_EXPERIMENT_NAME, RF_MODEL_FILE_NAME)
    hist_model, hist_model_path, hist_config = load_trained_model(HIST_EXPERIMENT_NAME, HIST_MODEL_FILE_NAME)

    print_section("MODELS")
    print(f"RF   model type : {type(rf_model).__name__}")
    print(f"RF   model path : {rf_model_path}")
    print(f"Hist model type : {type(hist_model).__name__}")
    print(f"Hist model path : {hist_model_path}")
    print(f"Layer           : {rf_config.layer.value}")
    print(f"Target column   : {rf_config.target_column}")
    print(f"Random state    : {rf_config.random_state}")
    print(f"Routing rule    : {ROUTING_RULE_NAME} ({ROUTING_RULE_DIRECTION}) -- FROZEN, not re-derived here")

    splits = load_splits()

    for split_name in ("test", "ood"):
        split = splits[split_name]

        # Both models score every row -- the frozen rule only decides
        # which of these two ALREADY-COMPUTED prediction arrays is kept
        # per row. Neither predict() call is influenced by routing.
        rf_preds = rf_model.predict(split.X)
        hist_preds = hist_model.predict(split.X)

        hybrid_preds, route_mask = build_hybrid_predictions(split.X, rf_preds, hist_preds)

        n_missing_routing_col = int(split.X[ROUTING_COLUMN].isna().sum())
        n_routed_hist = int(route_mask.sum())
        n_routed_rf = int(len(route_mask) - n_routed_hist)

        # y (true_queue_length_m) is used HERE for the first and only
        # time in this loop iteration -- strictly for scoring, after
        # every prediction (rf_preds, hist_preds, hybrid_preds) has
        # already been produced without it.
        rf_metrics = eval_predictions(split.y, rf_preds)
        hist_metrics = eval_predictions(split.y, hist_preds)
        hybrid_metrics = eval_predictions(split.y, hybrid_preds)

        print_section(f"{split_name.upper()} SPLIT -- ROUTING SUMMARY")
        print(f"rows routed to Random Forest ({ROUTING_COLUMN}==False)      : {n_routed_rf}")
        print(f"rows routed to HistGradientBoosting ({ROUTING_COLUMN}==True): {n_routed_hist}")
        print(
            f"rows with missing {ROUTING_COLUMN} (defaulted to RF)         : {n_missing_routing_col}"
        )

        print_comparison_table(
            f"{split_name.upper()} -- RF | HistGradientBoosting | Hybrid (frozen congestion_flag)",
            rf_metrics,
            hist_metrics,
            hybrid_metrics,
        )

        diff_vs_rf = metric_diff(hybrid_metrics, rf_metrics)
        diff_vs_hist = metric_diff(hybrid_metrics, hist_metrics)
        print_diff_table(split_name.upper(), diff_vs_rf, diff_vs_hist)

    print_section("NOTE")
    print(
        f"'{ROUTING_RULE_NAME}' ({ROUTING_RULE_DIRECTION}) was selected on the VALIDATION "
        "split in validation_router_investigation.py and is applied here completely "
        "unchanged. This script never loads Split.VALIDATION, never searches for or "
        "compares against any other rule/threshold/condition, never recalibrates "
        f"{ROUTING_RULE_NAME}, and never trains, fits, or modifies either model. "
        "true_queue_length_m (SplitData.y) is read only inside eval_predictions()/"
        "compute_all_metrics() calls above, strictly for scoring -- build_route_mask() "
        "and build_hybrid_predictions() never receive y as an argument, so the routing "
        "decision structurally cannot depend on the target."
    )


if __name__ == "__main__":
    main()