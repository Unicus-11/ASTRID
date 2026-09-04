"""
evaluate_hybrid_router.py
====================
Read-only, analysis-only evaluation of a simple hybrid routing rule over
the two already-trained Layer 2 baseline models (Random Forest and
HistGradientBoosting).

Routing rule (fixed, not learned, not tuned here):
    queue_reaches_camera_edge == False -> use Random Forest's prediction
    queue_reaches_camera_edge == True  -> use HistGradientBoosting's prediction

This mirrors error_analysis.py / error_analysis_random_forest.py's loading
conventions exactly (same ExperimentConfig / load_model / DatasetLoader /
compute_all_metrics pattern), just applied to two models at once instead
of one.

This script:
    * loads BOTH ALREADY-TRAINED models via persistence.load_model() --
      no retraining, no fitting
    * loads Layer 2 test/ood splits via the existing DatasetLoader --
      no new splitting logic, no dataset modification
    * calls model.predict(X) once per model per split (both models see
      every row -- the routing rule only decides which prediction is
      KEPT, it never changes what either model is asked to predict)
    * builds the hybrid prediction as a row-wise selection between the
      two models' predictions, using ONLY queue_reaches_camera_edge
      (an observed camera feature) as the routing signal
    * reuses metrics.compute_all_metrics() for every MAE/RMSE/R2 number
      -- no metric formula is reimplemented
    * prints a compact RF vs HistGradientBoosting vs Hybrid Router
      comparison table for TEST and OOD

ROUTING-SIGNAL / LEAKAGE GUARANTEE:
    true_queue_length_m (the target column) is read from SplitData.y and
    used ONLY inside compute_all_metrics() calls, for evaluation. The
    routing decision (_route_mask, below) is built exclusively from
    split.X["queue_reaches_camera_edge"] -- an observed, camera-derived
    feature that is already part of the model's own input -- and never
    reads split.y at any point. build_hybrid_predictions() does not even
    accept y as an argument, so this is enforced structurally, not just
    by convention.

Nothing here writes to disk. No CSV/JSON is created. No infrastructure
file (data_loader.py, evaluate.py, experiment_runner.py, metrics.py,
persistence.py, model implementations, or either error_analysis*.py) is
modified. No model is retrained, re-fit, or tuned; this is evaluation of
two existing artifacts plus a fixed routing rule between them.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# models/results/evaluate_hybrid_router.py -> models/ is the parent's parent.
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
ROUTING_COLUMN = "queue_reaches_camera_edge"

# ---------------------------------------------------------------------------
# Model identity, mirrors error_analysis.py / error_analysis_random_forest.py
# exactly -- same artifact paths, nothing re-derived or guessed.
# ---------------------------------------------------------------------------
RF_EXPERIMENT_NAME = "random_forest_layer2_p11_baseline"
RF_MODEL_FILE_NAME = "random_forest.joblib"
RF_BASELINE_MODEL_NAME = "random_forest"  # matches baseline_results.csv "model" column

HIST_EXPERIMENT_NAME = "hist_gradient_boosting_layer2_p11_baseline"
HIST_MODEL_FILE_NAME = "hist_gradient_boosting.joblib"
HIST_BASELINE_MODEL_NAME = "hist_gradient_boosting"

_RESULTS_DIR = Path(__file__).resolve().parent
_BASELINE_CSV_PATH = _RESULTS_DIR / "baseline_results.csv"


# ---------------------------------------------------------------------------
# Loading (no retraining, no dataset modification)
# ---------------------------------------------------------------------------

def load_trained_model(experiment_name: str, model_file_name: str):
    """Load one already-trained, already-persisted model artifact.
    Identical mechanism to error_analysis.py / error_analysis_random_forest.py
    -- ExperimentConfig resolves the artifact directory, persistence.load_model()
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
    loader = DatasetLoader(layer=LAYER)
    return {
        "test": loader.load(Split.TEST),
        "ood": loader.load(Split.OOD),
    }


# ---------------------------------------------------------------------------
# Routing (reads ONLY split.X[ROUTING_COLUMN] -- never split.y)
# ---------------------------------------------------------------------------

def build_route_mask(X: pd.DataFrame) -> pd.Series:
    """True where the row should be routed to HistGradientBoosting
    (queue_reaches_camera_edge == True); False where it should be routed
    to Random Forest (queue_reaches_camera_edge == False).

    Built ONLY from X -- the observed camera feature -- never from y.
    data_loader.py's _normalize_boolean_like_feature_columns() already
    coerces this column to numeric 1.0/0.0 (with real missing values left
    as NaN, never fabricated), so the comparisons below cover both that
    normalized numeric form and a raw boolean/string form defensively, in
    case this script is ever pointed at an unnormalized frame.
    """
    if ROUTING_COLUMN not in X.columns:
        raise KeyError(
            f"'{ROUTING_COLUMN}' is not a feature column for this split -- "
            f"the hybrid router has no basis to route on. Check manifest.json."
        )
    col = X[ROUTING_COLUMN]
    is_true = col.isin([True, 1, 1.0, "True", "true"])
    return is_true


def build_hybrid_predictions(
    X: pd.DataFrame, rf_preds: np.ndarray, hist_preds: np.ndarray
) -> "tuple[np.ndarray, pd.Series]":
    """Row-wise selection between two already-computed prediction arrays,
    using ONLY the routing feature in X. Does not accept or touch y
    anywhere in this function -- routing cannot see the target by
    construction, not merely by convention.

    Rows where ROUTING_COLUMN is missing (NaN -- a real, observed gap in
    the camera feature, not fabricated) default to the Random Forest
    prediction, matching the rule's own False branch, since
    queue_reaches_camera_edge missing is not queue_reaches_camera_edge
    True. This default is applied to a COUNTED, REPORTED subset (see
    n_missing_routing_col in main()) rather than silently.
    """
    route_to_hist = build_route_mask(X).to_numpy()
    hybrid = np.where(route_to_hist, hist_preds, rf_preds)
    return hybrid, build_route_mask(X)


# ---------------------------------------------------------------------------
# Metrics (reuses metrics.compute_all_metrics -- no reimplementation)
# ---------------------------------------------------------------------------

def eval_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return compute_all_metrics(y_true, y_pred)


def load_baseline_consistency_row(model_name: str) -> Optional[pd.DataFrame]:
    """Optional cross-check: standalone RF / Hist metrics computed here
    should match the corresponding row already recorded in
    baseline_results.csv, since both come from the same trained artifact
    and the same DatasetLoader splits. Read-only; nothing is written."""
    if not _BASELINE_CSV_PATH.exists():
        return None
    baseline_df = pd.read_csv(_BASELINE_CSV_PATH)
    match = baseline_df[
        (baseline_df["model"] == model_name) & (baseline_df["layer"] == LAYER.value)
    ]
    return match if not match.empty else None


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


def print_comparison_table(
    label: str,
    rf_metrics: dict,
    hist_metrics: dict,
    hybrid_metrics: dict,
) -> None:
    print_section(label)
    header = f"{'metric':<8} {'RF':>14} {'HistGradientBoosting':>22} {'Hybrid Router':>16}"
    print(header)
    print("-" * len(header))
    for key, title in (("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R2")):
        print(
            f"{title:<8} {_fmt(rf_metrics[key]):>14} "
            f"{_fmt(hist_metrics[key]):>22} {_fmt(hybrid_metrics[key]):>16}"
        )
    print(f"{'n':<8} {rf_metrics['n']:>14} {hist_metrics['n']:>22} {hybrid_metrics['n']:>16}")


def print_consistency_check(model_label: str, baseline_model_name: str, computed: dict) -> None:
    print_section(f"CONSISTENCY CHECK vs baseline_results.csv ({model_label})")
    row = load_baseline_consistency_row(baseline_model_name)
    if row is None:
        print(f"No matching row found in {_BASELINE_CSV_PATH} for '{baseline_model_name}'.")
        return
    r = row.iloc[0]
    print(
        f"baseline_results.csv -> test_mae={_fmt(r.get('test_mae'))}, "
        f"test_rmse={_fmt(r.get('test_rmse'))}, test_r2={_fmt(r.get('test_r2'))}, "
        f"ood_mae={_fmt(r.get('ood_mae'))}, ood_rmse={_fmt(r.get('ood_rmse'))}, "
        f"ood_r2={_fmt(r.get('ood_r2'))}"
    )
    print("(compare against the standalone RF / Hist rows in the table above)")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    print("LAYER 2 HYBRID ROUTER EVALUATION (RF vs HistGradientBoosting vs Hybrid)")
    print("=" * len("LAYER 2 HYBRID ROUTER EVALUATION (RF vs HistGradientBoosting vs Hybrid)"))

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
    print(f"Routing rule    : {ROUTING_COLUMN}==False -> RF, {ROUTING_COLUMN}==True -> Hist")

    splits = load_splits()

    for split_name in ("test", "ood"):
        split = splits[split_name]

        # Both models score every row -- the routing rule only decides
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
            f"{split_name.upper()} -- RF | HistGradientBoosting | Hybrid Router",
            rf_metrics,
            hist_metrics,
            hybrid_metrics,
        )

    # Optional read-only cross-check of the standalone models' numbers
    # against the already-recorded baseline_results.csv, same pattern
    # error_analysis.py / error_analysis_random_forest.py already use.
    test_split = splits["test"]
    rf_test_metrics = eval_predictions(test_split.y, rf_model.predict(test_split.X))
    hist_test_metrics = eval_predictions(test_split.y, hist_model.predict(test_split.X))
    print_consistency_check("Random Forest", RF_BASELINE_MODEL_NAME, rf_test_metrics)
    print_consistency_check("HistGradientBoosting", HIST_BASELINE_MODEL_NAME, hist_test_metrics)

    print_section("NOTE")
    print(
        "true_queue_length_m (SplitData.y) is read only inside eval_predictions()/\n"
        "compute_all_metrics() calls above, strictly for scoring. build_route_mask()\n"
        "and build_hybrid_predictions() never receive y as an argument, so the routing\n"
        "decision structurally cannot depend on the target."
    )


if __name__ == "__main__":
    main()