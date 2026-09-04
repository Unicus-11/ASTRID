"""
evaluate_hist_gradient_boosting_trial7.py
============================================
FINAL HELD-OUT EVALUATION -- Trial 7 configuration (Layer 2, layer2_p11)

Trial 7 was selected as the winning HistGradientBoosting configuration
after five rounds of validation-only hyperparameter tuning (Rounds 2-5,
see tune_hist_gradient_boosting_round{2,3,4,5}.py and their results
CSVs). This script performs NO further tuning and evaluates NO
alternative candidates -- it exists solely to:

    1. Fit the single, already-selected Trial 7 configuration on TRAIN.
    2. Evaluate it once on TEST and once on OOD (both untouched until
       now -- neither was loaded nor inspected at any point during
       Rounds 2-5).
    3. Compare TEST/OOD results against the original (untuned)
       HistGradientBoosting baseline.
    4. Persist the trained model artifact and the evaluation results.

VALIDATION is not reloaded or re-used here -- it already did its job
(candidate selection) in Rounds 2-5 and plays no further role.

Methodology guarantees:
    * TRAIN   -> fit only
    * TEST    -> evaluated once, after fitting, never used to adjust
                 the model or hyperparameters
    * OOD     -> evaluated once, after fitting, never used to adjust
                 the model or hyperparameters
    * early_stopping = False
    * random_state = 42
    * constructs sklearn.ensemble.HistGradientBoostingRegressor
      directly (same approach as Rounds 2-5; NOT routed through
      model_implementations/hist_gradient_boosting.py, which does not
      expose early_stopping)
    * reuses metrics.compute_all_metrics() for MAE/RMSE/R2 -- no metric
      math is reimplemented
    * reuses persistence.save_model() for artifact persistence -- no
      custom serialization logic

This script does not modify any tuning script (Rounds 1-5) or any of
their result CSVs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# models/results/evaluate_hist_gradient_boosting_trial7.py -> models/ is
# the parent's parent.
_MODELS_DIR = Path(__file__).resolve().parent.parent
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from data_loader import DatasetLoader  # noqa: E402
from experiment_config import Layer, Split  # noqa: E402
from metrics import compute_all_metrics  # noqa: E402
from persistence import save_model  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed configuration
# ---------------------------------------------------------------------------

LAYER = Layer.LAYER2_P11
MODEL_RANDOM_STATE = 42

_RESULTS_DIR = Path(__file__).resolve().parent
_OUTPUT_CSV = _RESULTS_DIR / "hist_gradient_boosting_trial7_test_ood_results.csv"
_ARTIFACT_PATH = _RESULTS_DIR / "artifacts" / "hist_gradient_boosting_trial7.joblib"

# Selected Trial 7 configuration (already chosen via Rounds 2-5
# validation-only tuning; NOT re-derived or re-searched here).
TRIAL_7_CONFIG: Dict[str, Any] = {
    "learning_rate": 0.025034,
    "max_iter": 369,
    "max_leaf_nodes": 52,
    "min_samples_leaf": 35,
    "l2_regularization": 0.502603,
    "max_depth": 10,
}

# Original (untuned) HistGradientBoosting baseline, provided for
# comparison. NOT re-measured by this script -- taken as given.
BASELINE = {
    "TEST": {"mae": 18.706834, "rmse": 44.630782, "r2": 0.939220},
    "OOD": {"mae": 34.757319, "rmse": 63.107135, "r2": 0.903846},
}


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _fmt(value: float, decimals: int = 6) -> str:
    return f"{value:.{decimals}f}"


def print_config() -> None:
    print("SELECTED TRIAL 7 CONFIGURATION (fixed -- not re-tuned)")
    print("-" * 60)
    print(f"  learning_rate      = {TRIAL_7_CONFIG['learning_rate']}")
    print(f"  max_iter           = {TRIAL_7_CONFIG['max_iter']}")
    print(f"  max_leaf_nodes     = {TRIAL_7_CONFIG['max_leaf_nodes']}")
    print(f"  min_samples_leaf   = {TRIAL_7_CONFIG['min_samples_leaf']}")
    print(f"  l2_regularization  = {TRIAL_7_CONFIG['l2_regularization']}")
    print(f"  max_depth          = {TRIAL_7_CONFIG['max_depth']}")
    print(f"  early_stopping     = False")
    print(f"  random_state       = {MODEL_RANDOM_STATE}")
    print()


def print_metrics(label: str, m: Dict[str, float]) -> None:
    print(f"{label} METRICS")
    print("-" * 60)
    print(f"  MAE  = {_fmt(m['mae'])}")
    print(f"  RMSE = {_fmt(m['rmse'])}")
    print(f"  R2   = {_fmt(m['r2'])}")
    print(f"  N    = {m['n']}")
    print()


def pct_improvement(baseline_value: float, new_value: float) -> float:
    """(baseline - new) / baseline * 100 -- positive = improvement
    (lower error) for MAE/RMSE."""
    return (baseline_value - new_value) / baseline_value * 100.0


def print_comparison(split_label: str, baseline: Dict[str, float], new: Dict[str, float]) -> None:
    mae_pct = pct_improvement(baseline["mae"], new["mae"])
    rmse_pct = pct_improvement(baseline["rmse"], new["rmse"])
    r2_delta = new["r2"] - baseline["r2"]

    print(f"COMPARISON vs ORIGINAL BASELINE -- {split_label}")
    print("-" * 60)
    print(f"  MAE : baseline={_fmt(baseline['mae'])}  trial7={_fmt(new['mae'])}  "
          f"-> {'improvement' if mae_pct > 0 else 'degradation'} of {abs(mae_pct):.2f}%")
    print(f"  RMSE: baseline={_fmt(baseline['rmse'])}  trial7={_fmt(new['rmse'])}  "
          f"-> {'improvement' if rmse_pct > 0 else 'degradation'} of {abs(rmse_pct):.2f}%")
    print(f"  R2  : baseline={_fmt(baseline['r2'])}  trial7={_fmt(new['r2'])}  "
          f"-> change of {r2_delta:+.6f}")
    print()

    return mae_pct, rmse_pct, r2_delta


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    print("FINAL HELD-OUT EVALUATION -- Trial 7 (Layer 2, layer2_p11)")
    print("No further tuning is performed by this script.")
    print("=" * 78)
    print()
    print(f"Layer: {LAYER.value}")
    print("TEST and OOD are used ONLY for this final evaluation -- neither")
    print("was loaded or inspected at any point during Rounds 2-5 tuning.")
    print()

    print_config()

    # ---- Load TRAIN, TEST, OOD via the existing manifest-driven loader ----
    # (VALIDATION already did its job during Rounds 2-5 selection and is
    # deliberately not reloaded here.)
    loader = DatasetLoader(layer=LAYER)
    train_split = loader.load(Split.TRAIN)
    test_split = loader.load(Split.TEST)
    ood_split = loader.load(Split.OOD)

    print(f"TRAIN rows: {len(train_split)}   TEST rows: {len(test_split)}   OOD rows: {len(ood_split)}")
    print()

    # ---- Fit Trial 7 on TRAIN only ----
    model = HistGradientBoostingRegressor(
        learning_rate=TRIAL_7_CONFIG["learning_rate"],
        max_iter=TRIAL_7_CONFIG["max_iter"],
        max_leaf_nodes=TRIAL_7_CONFIG["max_leaf_nodes"],
        min_samples_leaf=TRIAL_7_CONFIG["min_samples_leaf"],
        l2_regularization=TRIAL_7_CONFIG["l2_regularization"],
        max_depth=TRIAL_7_CONFIG["max_depth"],
        early_stopping=False,
        random_state=MODEL_RANDOM_STATE,
    )
    model.fit(train_split.X, train_split.y)

    # ---- Evaluate once on TEST, once on OOD ----
    test_preds = model.predict(test_split.X)
    test_metrics = compute_all_metrics(test_split.y, test_preds)

    ood_preds = model.predict(ood_split.X)
    ood_metrics = compute_all_metrics(ood_split.y, ood_preds)

    print_metrics("TEST", test_metrics)
    print_metrics("OOD", ood_metrics)

    # ---- Compare against original baseline ----
    test_mae_pct, test_rmse_pct, test_r2_delta = print_comparison(
        "TEST", BASELINE["TEST"], test_metrics
    )
    ood_mae_pct, ood_rmse_pct, ood_r2_delta = print_comparison(
        "OOD", BASELINE["OOD"], ood_metrics
    )

    # ---- Summary table ----
    print("SUMMARY TABLE")
    print("-" * 78)
    summary_rows = [
        ("Original HistGradientBoosting", "TEST", BASELINE["TEST"]["mae"], BASELINE["TEST"]["rmse"], BASELINE["TEST"]["r2"]),
        ("Tuned Trial 7", "TEST", test_metrics["mae"], test_metrics["rmse"], test_metrics["r2"]),
        ("Original HistGradientBoosting", "OOD", BASELINE["OOD"]["mae"], BASELINE["OOD"]["rmse"], BASELINE["OOD"]["r2"]),
        ("Tuned Trial 7", "OOD", ood_metrics["mae"], ood_metrics["rmse"], ood_metrics["r2"]),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["model", "split", "mae", "rmse", "r2"])
    print(summary_df.to_string(index=False))
    print()

    both_improved = (test_mae_pct > 0 and test_rmse_pct > 0 and test_r2_delta > 0
                      and ood_mae_pct > 0 and ood_rmse_pct > 0 and ood_r2_delta > 0)
    print("CONCLUSION")
    print("-" * 78)
    print(f"  TEST: {'IMPROVED' if (test_mae_pct > 0 and test_rmse_pct > 0) else 'DID NOT IMPROVE'} "
          f"over baseline (MAE {test_mae_pct:+.2f}%, RMSE {test_rmse_pct:+.2f}%, R2 {test_r2_delta:+.6f})")
    print(f"  OOD:  {'IMPROVED' if (ood_mae_pct > 0 and ood_rmse_pct > 0) else 'DID NOT IMPROVE'} "
          f"over baseline (MAE {ood_mae_pct:+.2f}%, RMSE {ood_rmse_pct:+.2f}%, R2 {ood_r2_delta:+.6f})")
    print(f"  Overall, tuning {'DID' if both_improved else 'DID NOT'} produce a model that beats the "
          f"original baseline on BOTH held-out splits on all three metrics.")
    print()

    # ---- Save results CSV ----
    results_rows = [
        {
            "model": "original_baseline", "split": "TEST",
            "mae": BASELINE["TEST"]["mae"], "rmse": BASELINE["TEST"]["rmse"], "r2": BASELINE["TEST"]["r2"],
            "n": None, "mae_improvement_pct": None, "rmse_improvement_pct": None, "r2_change": None,
        },
        {
            "model": "trial7", "split": "TEST",
            "mae": test_metrics["mae"], "rmse": test_metrics["rmse"], "r2": test_metrics["r2"],
            "n": test_metrics["n"], "mae_improvement_pct": test_mae_pct,
            "rmse_improvement_pct": test_rmse_pct, "r2_change": test_r2_delta,
        },
        {
            "model": "original_baseline", "split": "OOD",
            "mae": BASELINE["OOD"]["mae"], "rmse": BASELINE["OOD"]["rmse"], "r2": BASELINE["OOD"]["r2"],
            "n": None, "mae_improvement_pct": None, "rmse_improvement_pct": None, "r2_change": None,
        },
        {
            "model": "trial7", "split": "OOD",
            "mae": ood_metrics["mae"], "rmse": ood_metrics["rmse"], "r2": ood_metrics["r2"],
            "n": ood_metrics["n"], "mae_improvement_pct": ood_mae_pct,
            "rmse_improvement_pct": ood_rmse_pct, "r2_change": ood_r2_delta,
        },
    ]
    results_df = pd.DataFrame(results_rows)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(_OUTPUT_CSV, index=False)

    # ---- Save trained model artifact via the project's persistence module ----
    saved_path = save_model(model, _ARTIFACT_PATH)

    print(f"Results CSV written to : {_OUTPUT_CSV.resolve()}")
    print(f"Model artifact saved to: {Path(saved_path).resolve()}")
    print()
    print("This was a final evaluation only. No hyperparameters were changed")
    print("based on TEST or OOD results, and no further tuning was performed.")


if __name__ == "__main__":
    main()