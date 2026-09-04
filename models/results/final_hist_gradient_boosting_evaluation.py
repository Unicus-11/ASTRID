"""
final_hist_gradient_boosting_evaluation.py
=============================================
Final held-out (TEST + OOD) evaluation of HistGradientBoosting on ASTRID
Layer 2 (p11), comparing three configurations:

    1. Original baseline  -- existing .joblib artifact, loaded, NOT retrained.
    2. Baseline-control -- original baseline hyperparameters,
                           early_stopping=False
    3. Tuned (Trial 26)    -- frozen winning validation-tuning config,
                              early_stopping=False, trained on TRAIN only.

This is the FINAL evaluation stage: TEST and OOD are read here. No further
hyperparameter search happens in this file. Nothing about the dataset,
features, target, splits, or the existing baseline artifact is touched.

Known gap in model_implementations/hist_gradient_boosting.py:
    HistGradientBoostingModel.__init__ does not accept or forward
    early_stopping to the underlying HistGradientBoostingRegressor at
    all, so every model built through it silently uses sklearn's own
    default ('auto'), never True/False explicitly. Since the
    baseline-control and tuned configs in this task require
    early_stopping=False, a small local subclass
    (_EarlyStoppingHGB below) is defined ONLY in this script to add that
    one missing knob. The shipped model file is not modified.

Run:
    python models/results/final_hist_gradient_boosting_evaluation.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# models/ is the parent of models/results/ -- add it to sys.path so the
# shared modules (experiment_config, data_loader, evaluate, persistence,
# experiment_runner) and model_implementations/ can be imported without a
# second data-loading or evaluation implementation.
_MODELS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_MODELS_DIR))

from experiment_config import ExperimentConfig, Layer, Split, DEFAULT_OUTPUT_ROOT  # noqa: E402
from data_loader import DatasetLoader  # noqa: E402
from evaluate import evaluate_model  # noqa: E402
from persistence import load_model  # noqa: E402
from experiment_runner import run_experiment, ExperimentResult  # noqa: E402
from model_implementations.hist_gradient_boosting import HistGradientBoostingModel  # noqa: E402


# ---------------------------------------------------------------------------
# Local fix: expose early_stopping, which the shipped wrapper omits.
# ---------------------------------------------------------------------------

class _EarlyStoppingHGB(HistGradientBoostingModel):
    """HistGradientBoostingModel + an explicit, forwarded early_stopping
    flag. See module docstring for why this exists. Only used in this
    script -- does not modify model_implementations/hist_gradient_boosting.py."""

    def __init__(self, *, early_stopping: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.early_stopping = early_stopping
        self._model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
            early_stopping=self.early_stopping,
        )


# ---------------------------------------------------------------------------
# Frozen configurations
# ---------------------------------------------------------------------------

LAYER = Layer.LAYER2_P11

ORIGINAL_BASELINE_PATH = (
    DEFAULT_OUTPUT_ROOT / "layer2_p11"
    / "hist_gradient_boosting_layer2_p11_baseline" / "hist_gradient_boosting.joblib"
)

BASELINE_CONTROL_PARAMS: Dict = dict(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.0,
    max_depth=None,
    early_stopping=False,
    random_state=42,
)

# Trial 26, full precision from hist_gradient_boosting_tuning_results.csv
# (validation_mae=5.075883918524597 matches the reported winning trial).
TUNED_PARAMS: Dict = dict(
    learning_rate=0.03419168411765895,
    max_iter=270,
    max_leaf_nodes=43,
    min_samples_leaf=40,
    l2_regularization=0.942853570557981,
    max_depth=10,
    early_stopping=False,
    random_state=42,
)

BASELINE_CONTROL_EXPERIMENT_NAME = "hist_gradient_boosting_layer2_p11_baseline_control"
TUNED_EXPERIMENT_NAME = "hist_gradient_boosting_layer2_p11_tuned"


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_existing_baseline() -> Dict[str, float]:
    """Load the existing baseline artifact and evaluate it on TEST/OOD.
    Does NOT fit/retrain it. Reuses the shared loader + evaluator only."""
    if not ORIGINAL_BASELINE_PATH.exists():
        raise FileNotFoundError(f"Original baseline artifact not found: {ORIGINAL_BASELINE_PATH}")

    fitted_model = load_model(ORIGINAL_BASELINE_PATH)

    loader = DatasetLoader(layer=LAYER)
    validation_split = loader.load(Split.VALIDATION)  # required by evaluate_model's signature; not fit on
    test_split = loader.load(Split.TEST)
    ood_split = loader.load(Split.OOD)

    report = evaluate_model(
        fitted_model,
        validation_split=validation_split,
        test_split=test_split,
        ood_split=ood_split,
    )

    return {
        "model": "Original baseline",
        "model_path": str(ORIGINAL_BASELINE_PATH),
        "test_mae": report.test.metrics["mae"],
        "test_rmse": report.test.metrics["rmse"],
        "test_r2": report.test.metrics["r2"],
        "test_n": report.test.metrics["n"],
        "ood_mae": report.ood.metrics["mae"],
        "ood_rmse": report.ood.metrics["rmse"],
        "ood_r2": report.ood.metrics["r2"],
        "ood_n": report.ood.metrics["n"],
    }


def train_and_evaluate(label: str, experiment_name: str, hyperparams: Dict) -> Dict[str, float]:
    """Fit a fresh model on TRAIN only (via the shared experiment_runner
    protocol) and evaluate on TEST/OOD. Never touches VALIDATION/TEST/OOD
    for fitting -- that guarantee lives in experiment_runner.run_experiment()."""
    config = ExperimentConfig(layer=LAYER, experiment_name=experiment_name, random_state=42)

    result: ExperimentResult = run_experiment(
        config=config,
        model=lambda: _EarlyStoppingHGB(**hyperparams),
        model_name="hist_gradient_boosting",
        save_model_flag=True,
    )

    return {
        "model": label,
        "model_path": str(result.model_path),
        "test_mae": result.test_metrics["mae"],
        "test_rmse": result.test_metrics["rmse"],
        "test_r2": result.test_metrics["r2"],
        "test_n": result.test_metrics["n"],
        "ood_mae": result.ood_metrics["mae"],
        "ood_rmse": result.ood_metrics["rmse"],
        "ood_r2": result.ood_metrics["r2"],
        "ood_n": result.ood_metrics["n"],
    }


# ---------------------------------------------------------------------------
# Comparison reporting
# ---------------------------------------------------------------------------

def _delta_row(name: str, ref: Dict, other: Dict) -> Dict:
    """other vs ref. MAE/RMSE: positive = improvement (error went down).
    R2: positive = improvement (R2 went up)."""
    return {
        "comparison": name,
        "test_mae_improvement": ref["test_mae"] - other["test_mae"],
        "test_rmse_improvement": ref["test_rmse"] - other["test_rmse"],
        "test_r2_improvement": other["test_r2"] - ref["test_r2"],
        "ood_mae_improvement": ref["ood_mae"] - other["ood_mae"],
        "ood_rmse_improvement": ref["ood_rmse"] - other["ood_rmse"],
        "ood_r2_improvement": other["ood_r2"] - ref["ood_r2"],
    }


def _verdict(delta_row: Dict, split: str) -> str:
    mae_ok = delta_row[f"{split}_mae_improvement"] > 0
    rmse_ok = delta_row[f"{split}_rmse_improvement"] > 0
    r2_ok = delta_row[f"{split}_r2_improvement"] > 0
    if mae_ok and rmse_ok and r2_ok:
        return "YES (all three metrics improved)"
    if not mae_ok and not rmse_ok and not r2_ok:
        return "NO (all three metrics worse)"
    return "MIXED (metrics disagree)"


def main() -> None:
    print("=" * 70)
    print("Final held-out evaluation: HistGradientBoosting, Layer2 (p11)")
    print("=" * 70)

    print("\n[1/3] Loading + evaluating ORIGINAL baseline (no retraining)...")
    original = evaluate_existing_baseline()

    print("\n[2/3] Training + evaluating BASELINE-CONTROL...")
    baseline_control = train_and_evaluate(
        "Baseline-control", BASELINE_CONTROL_EXPERIMENT_NAME, BASELINE_CONTROL_PARAMS
    )

    print("\n[3/3] Training + evaluating TUNED (Trial 26)...")
    tuned = train_and_evaluate("Tuned HistGradientBoosting", TUNED_EXPERIMENT_NAME, TUNED_PARAMS)

    results = [original, baseline_control, tuned]
    df = pd.DataFrame(results)[
        ["model", "test_mae", "test_rmse", "test_r2", "test_n",
         "ood_mae", "ood_rmse", "ood_r2", "ood_n", "model_path"]
    ]
    df_display = df.drop(columns=["model_path"]).round(6)

    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(df_display.to_string(index=False))

    print("\nArtifact paths:")
    for r in results:
        print(f"  {r['model']}: {r['model_path']}")

    tuned_vs_control = _delta_row("Tuned vs Baseline-control", baseline_control, tuned)
    tuned_vs_original = _delta_row("Tuned vs Original baseline", original, tuned)

    print("\n" + "=" * 70)
    print("IMPROVEMENT (positive = tuned model is better)")
    print("=" * 70)
    deltas_df = pd.DataFrame([tuned_vs_control, tuned_vs_original]).round(6)
    print(deltas_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("VERDICT (evidence only -- no model is declared final here)")
    print("=" * 70)
    print(f"Did tuning improve TEST vs baseline-control?  {_verdict(tuned_vs_control, 'test')}")
    print(f"Did tuning improve OOD  vs baseline-control?  {_verdict(tuned_vs_control, 'ood')}")
    print(f"Did tuning improve TEST vs original baseline? {_verdict(tuned_vs_original, 'test')}")
    print(f"Did tuning improve OOD  vs original baseline? {_verdict(tuned_vs_original, 'ood')}")
    print(
        "\nNote: 'vs Baseline-control' is the apples-to-apples comparison "
        "(identical training protocol, early_stopping=False on both sides). "
        "'vs Original baseline' is contextual only -- that artifact's "
        "early_stopping setting was never explicitly controlled."
    )

    out_csv = Path(__file__).resolve().parent / "final_hist_gradient_boosting_evaluation_results.csv"
    df_display.to_csv(out_csv, index=False)
    print(f"\nResults table written to: {out_csv}")


if __name__ == "__main__":
    main()