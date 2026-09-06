"""
train_evaluate_hist_gradient_boosting_p50_baseline.py
=========================================================
SEPARATE EXPERIMENT -- training-distribution mismatch probe.

The GPS penetration sensitivity experiment showed the FROZEN,
p11-trained original baseline HistGradientBoosting model doing much
worse at 50% GPS penetration than at 25%, even though 50% has more
probes than either 11% or 25%:

    (frozen p11-trained model, evaluated on p50)
    TEST MAE  = 16.520118   TEST RMSE = 39.883303   TEST R2 = 0.951463
    OOD  MAE  = 28.930702   OOD  RMSE = 57.671497   OOD  R2 = 0.919697

This script asks a narrower, separate question: is that p50 result
partly a TRAINING-distribution mismatch effect (the model has simply
never seen p50-like GPS feature statistics during training), rather
than something else? To find out, it trains a BRAND NEW
HistGradientBoosting model using ONLY the p50 TRAIN split, with the
project's existing original-baseline configuration/procedure UNCHANGED
(no tuning), and evaluates it on p50 VALIDATION/TEST/OOD.

This script does NOT:
    * modify or replace the existing p11 baseline artifact
      (it never opens, loads, or writes that file at all -- the p11
      frozen-on-p50 numbers above are used only as literal, already
      published reference numbers);
    * modify evaluate_gps_penetration_sensitivity.py or any other
      existing evaluation/tuning script;
    * use TEST or OOD for any tuning/selection decision;
    * change the scenario-level train/validation/test/OOD split;
    * change the project's current final-model selection -- the
      original p11 HGB remains the selected final model regardless of
      what this experiment finds;
    * introduce any hyperparameter tuning.

Model configuration
---------------------
Uses the SAME original baseline HistGradientBoosting configuration this
project already defined and used elsewhere for exactly this purpose --
BASELINE_CONTROL_PARAMS in final_hist_gradient_boosting_evaluation.py
("original baseline hyperparameters, early_stopping=False"):

    learning_rate      = 0.05
    max_iter           = 300
    max_leaf_nodes     = 31
    max_depth          = None
    min_samples_leaf   = 20
    l2_regularization  = 0.0
    early_stopping     = False   (explicit, never silently left to
                                   sklearn's own default)
    random_state       = 42

Nothing here is re-derived, tuned, or guessed -- only the training data
changes (p50 TRAIN instead of p11 TRAIN).

Because model_implementations.hist_gradient_boosting.HistGradientBoostingModel
does not accept/forward early_stopping (the same documented gap
final_hist_gradient_boosting_evaluation.py already worked around), this
script re-applies that project's own one-knob local-subclass patch
(_EarlyStoppingHGB) rather than modifying the shipped model file. It is
re-implemented locally (not imported) because that script's version is
explicitly private to it ("Only used in this script").

Orchestration reuses experiment_runner.run_experiment() verbatim -- the
existing, shared load -> fit(TRAIN only) -> evaluate(VALIDATION/TEST/OOD)
-> persist pipeline. No new fit/evaluate/persistence logic is written
here.

Run:
    python models/results/train_evaluate_hist_gradient_boosting_p50_baseline.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from sklearn.ensemble import HistGradientBoostingRegressor

_MODELS_DIR = Path(__file__).resolve().parent.parent
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from data_loader import DatasetLoader  # noqa: E402
from experiment_config import ExperimentConfig, Layer, Split, DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT  # noqa: E402
from experiment_runner import run_experiment, ExperimentResult  # noqa: E402
from model_implementations.hist_gradient_boosting import HistGradientBoostingModel  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed configuration
# ---------------------------------------------------------------------------

_RESULTS_DIR = Path(__file__).resolve().parent
_REPORT_PATH = _RESULTS_DIR / "hist_gradient_boosting_p50_baseline_report.md"
_METRICS_CSV = _RESULTS_DIR / "hist_gradient_boosting_p50_baseline_metrics.csv"

EXPERIMENT_NAME = "hist_gradient_boosting_layer2_p50_baseline"
MODEL_NAME = "hist_gradient_boosting"

# The p11 artifact path is referenced ONLY as text in the report/printed
# output, for the reader's context. It is never opened, loaded, fit, or
# written by this script.
P11_ARTIFACT_PATH = (
    DEFAULT_OUTPUT_ROOT / "layer2_p11"
    / "hist_gradient_boosting_layer2_p11_baseline"
    / "hist_gradient_boosting.joblib"
)

# Same original baseline configuration already established and used in
# final_hist_gradient_boosting_evaluation.py's BASELINE_CONTROL_PARAMS.
# Not re-derived, not tuned -- reused verbatim.
ORIGINAL_BASELINE_CONFIG: Dict[str, Any] = dict(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.0,
    max_depth=None,
    early_stopping=False,
    random_state=42,
)

# Already-published reference numbers: the FROZEN p11-trained baseline
# evaluated on p50 (from evaluate_gps_penetration_sensitivity.py's
# output). Not recomputed here.
P11_TRAINED_ON_P50: Dict[str, Dict[str, float]] = {
    "TEST": {"mae": 16.520118, "rmse": 39.883303, "r2": 0.951463},
    "OOD": {"mae": 28.930702, "rmse": 57.671497, "r2": 0.919697},
}

EXPECTED_SCENARIOS: Dict[str, List[str]] = {
    "train": ["scenario_high_demand", "scenario_left_turn_heavy",
              "scenario_low_demand", "scenario_normal_balanced"],
    "validation": ["scenario_north_heavy", "scenario_straight_heavy"],
    "test": ["scenario_east_west_heavy", "scenario_south_heavy"],
    "ood": ["scenario_burst_demand_OOD", "scenario_heavy_vehicle_OOD",
            "scenario_north_extreme_OOD", "scenario_very_high_demand_OOD"],
}
_SPLIT_ENUM = {"train": Split.TRAIN, "validation": Split.VALIDATION,
               "test": Split.TEST, "ood": Split.OOD}


@dataclass(frozen=True)
class _PenetrationLayer:
    """Duck-typed stand-in for experiment_config.Layer -- there is no
    Layer.LAYER2_P50 member. DatasetLoader/load_split and
    ExperimentConfig.output_dir() only ever read `.value` off whatever
    `layer` they're given, so this minimal object satisfies that without
    touching experiment_config.py or data_loader.py. Same technique
    already used in evaluate_gps_penetration_sensitivity.py and
    diagnose_gps_penetration_p11_p25_p50.py."""

    value: str


P50_LAYER = _PenetrationLayer("layer2_p50")


class _EarlyStoppingHGB(HistGradientBoostingModel):
    """HistGradientBoostingModel + an explicit, forwarded early_stopping
    flag. Same minimal patch already used in
    final_hist_gradient_boosting_evaluation.py, re-implemented locally
    because that script's version is private to it. Does not modify
    model_implementations/hist_gradient_boosting.py."""

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
# Verification (before any training happens)
# ---------------------------------------------------------------------------

def verify_feature_schema(p50_loader: DatasetLoader) -> List[str]:
    """Reference schema = p11's manifest-declared feature_columns (what
    the currently-selected final model actually expects). Verified
    against every p50 split. Stops on any mismatch."""
    reference_features = DatasetLoader(layer=Layer.LAYER2_P11).load(Split.TRAIN).feature_columns

    print("FEATURE SCHEMA VERIFICATION")
    print("-" * 84)
    print(f"  reference (p11) n_features = {len(reference_features)}")

    mismatches = []
    for split_key, split_enum in _SPLIT_ENUM.items():
        split_data = p50_loader.load(split_enum)
        cols = split_data.feature_columns
        same_columns = set(cols) == set(reference_features)
        same_order = cols == reference_features
        print(f"  p50/{split_key}: n_features={len(cols)}  same_columns={same_columns}  same_order={same_order}")
        if not (same_columns and same_order):
            mismatches.append((split_key, cols))

    if mismatches:
        for split_key, cols in mismatches:
            missing = [c for c in reference_features if c not in cols]
            extra = [c for c in cols if c not in reference_features]
            print(f"  MISMATCH [p50/{split_key}]: missing={missing} extra={extra}")
        raise SystemExit(
            "STOPPING: p50 feature schema does not match the expected model "
            "feature schema (p11). Refusing to train -- fix the dataset/manifest "
            "and re-run."
        )
    print("  RESULT: p50 feature schema matches the expected model feature schema (p11). OK.")
    print()
    return reference_features


def verify_scenario_composition(p50_loader: DatasetLoader) -> Dict[str, List[str]]:
    print("SCENARIO SPLIT VERIFICATION (p50)")
    print("-" * 84)
    actual: Dict[str, List[str]] = {}
    mismatches = []
    for split_key, split_enum in _SPLIT_ENUM.items():
        scenarios = p50_loader.load(split_enum).scenario_ids()
        actual[split_key] = scenarios
        expected = EXPECTED_SCENARIOS[split_key]
        match = scenarios == expected
        print(f"  {split_key.upper():10s}: {scenarios}   matches expected: {'YES' if match else 'NO'}")
        if not match:
            mismatches.append((split_key, scenarios, expected))

    if mismatches:
        for split_key, scenarios, expected in mismatches:
            print(f"  MISMATCH [{split_key}]: expected={expected} actual={scenarios}")
        raise SystemExit(
            "STOPPING: p50 scenario-level split does not match the expected "
            "train/validation/test/OOD composition. Refusing to train."
        )
    print("  RESULT: p50 scenario composition matches the verified split exactly. OK.")
    print()
    return actual


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(v: float, d: int = 6) -> str:
    return f"{v:.{d}f}"


def build_comparison_rows(p50_trained: ExperimentResult) -> List[Dict[str, Any]]:
    rows = []
    for split_label, ref in P11_TRAINED_ON_P50.items():
        new_metrics = p50_trained.test_metrics if split_label == "TEST" else p50_trained.ood_metrics
        mae_change_pct = (ref["mae"] - new_metrics["mae"]) / ref["mae"] * 100.0
        rmse_change_pct = (ref["rmse"] - new_metrics["rmse"]) / ref["rmse"] * 100.0
        r2_change = new_metrics["r2"] - ref["r2"]
        rows.append({
            "split": split_label,
            "p11_trained_on_p50_mae": ref["mae"],
            "p50_trained_mae": new_metrics["mae"],
            "mae_change_pct": mae_change_pct,
            "p11_trained_on_p50_rmse": ref["rmse"],
            "p50_trained_rmse": new_metrics["rmse"],
            "rmse_change_pct": rmse_change_pct,
            "p11_trained_on_p50_r2": ref["r2"],
            "p50_trained_r2": new_metrics["r2"],
            "r2_change": r2_change,
        })
    return rows


def write_report(
    scenario_check: Dict[str, List[str]],
    reference_features: List[str],
    p50_trained: ExperimentResult,
    comparison_rows: List[Dict[str, Any]],
) -> None:
    p50_data_root = DEFAULT_DATA_ROOT / "layer2_p50"
    lines: List[str] = []
    lines.append("# HistGradientBoosting -- p50-trained baseline (training-distribution mismatch probe)")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Purpose")
    lines.append(
        "Separate experiment from the GPS penetration sensitivity analysis. Trains a "
        "new HistGradientBoosting model on p50 TRAIN only (original baseline "
        "configuration, unchanged) and evaluates it on p50 VALIDATION/TEST/OOD, to "
        "check whether the frozen p11-trained model's weaker p50 result is a "
        "training-distribution mismatch effect. Does not change the project's final "
        "model selection -- the original p11 HGB remains selected regardless of this "
        "experiment's outcome."
    )
    lines.append("")
    lines.append("## Dataset")
    lines.append(f"- Path: `{p50_data_root}`")
    lines.append(f"- n_features (verified against p11 schema): {len(reference_features)}")
    lines.append("")
    lines.append("## Scenario split verification")
    for split_key in ("train", "validation", "test", "ood"):
        lines.append(f"- **{split_key.upper()}**: {scenario_check[split_key]}")
    lines.append("")
    lines.append("## Model configuration (original baseline, unchanged; only training data is p50)")
    lines.append("```python")
    for k, v in ORIGINAL_BASELINE_CONFIG.items():
        lines.append(f"{k} = {v!r}")
    lines.append("```")
    lines.append("")
    lines.append("## Artifact")
    lines.append(f"- Saved to: `{p50_trained.model_path}`")
    lines.append(f"- p11 baseline artifact (referenced only, never modified): `{P11_ARTIFACT_PATH}`")
    lines.append("")
    lines.append("## Metrics -- p50-trained model")
    lines.append("| split | mae | rmse | r2 | n |")
    lines.append("|---|---:|---:|---:|---:|")
    for split_label, m in (
        ("VALIDATION", p50_trained.validation_metrics),
        ("TEST", p50_trained.test_metrics),
        ("OOD", p50_trained.ood_metrics),
    ):
        lines.append(f"| {split_label} | {_fmt(m['mae'])} | {_fmt(m['rmse'])} | {_fmt(m['r2'])} | {m['n']} |")
    lines.append("")
    lines.append("## Comparison -- A) frozen p11-trained model on p50, vs B) p50-trained model on p50")
    lines.append("| split | A: p11-trained MAE | B: p50-trained MAE | MAE change % | "
                  "A: p11-trained RMSE | B: p50-trained RMSE | RMSE change % | "
                  "A: p11-trained R2 | B: p50-trained R2 | R2 change |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in comparison_rows:
        lines.append(
            f"| {row['split']} | {_fmt(row['p11_trained_on_p50_mae'])} | {_fmt(row['p50_trained_mae'])} | "
            f"{row['mae_change_pct']:+.2f}% | {_fmt(row['p11_trained_on_p50_rmse'])} | "
            f"{_fmt(row['p50_trained_rmse'])} | {row['rmse_change_pct']:+.2f}% | "
            f"{_fmt(row['p11_trained_on_p50_r2'])} | {_fmt(row['p50_trained_r2'])} | "
            f"{row['r2_change']:+.6f} |"
        )
    lines.append("")
    lines.append(
        "Positive % / positive R2 change = the p50-trained model is better than the "
        "frozen p11-trained model, when both are evaluated on p50."
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("- No hyperparameter tuning was performed.")
    lines.append("- TEST/OOD were used only for final reporting, never for selection.")
    lines.append("- The p11 baseline artifact was never opened, loaded, or written by this script.")
    lines.append(
        "- This experiment does not change the project's current final-model selection; "
        "the original p11 HistGradientBoosting remains the selected final model."
    )

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    print("TRAIN + EVALUATE HistGradientBoosting -- p50 baseline")
    print("Separate experiment. No tuning. Does not touch the p11 artifact.")
    print("=" * 84)
    print()

    p50_loader = DatasetLoader(layer=P50_LAYER)

    reference_features = verify_feature_schema(p50_loader)
    scenario_check = verify_scenario_composition(p50_loader)

    print("MODEL CONFIGURATION (original baseline, unchanged; training data = p50 only)")
    print("-" * 84)
    for k, v in ORIGINAL_BASELINE_CONFIG.items():
        print(f"  {k:20s} = {v}")
    print()

    config = ExperimentConfig(
        layer=P50_LAYER,
        experiment_name=EXPERIMENT_NAME,
        random_state=ORIGINAL_BASELINE_CONFIG["random_state"],
        notes="Training-distribution mismatch probe: original baseline config, trained on p50 TRAIN only.",
    )
    print(f"Artifact will be saved under: {config.output_dir()}")
    print()

    print("Fitting on p50 TRAIN only, evaluating on p50 VALIDATION/TEST/OOD "
          "via experiment_runner.run_experiment() ...")
    p50_trained: ExperimentResult = run_experiment(
        config=config,
        model=lambda: _EarlyStoppingHGB(**ORIGINAL_BASELINE_CONFIG),
        model_name=MODEL_NAME,
        save_model_flag=True,
    )
    print("  done.")
    print()

    print("METRICS -- p50-trained model")
    print("-" * 84)
    for split_label, m in (
        ("VALIDATION", p50_trained.validation_metrics),
        ("TEST", p50_trained.test_metrics),
        ("OOD", p50_trained.ood_metrics),
    ):
        print(f"  {split_label:10s}: MAE={_fmt(m['mae'])}  RMSE={_fmt(m['rmse'])}  "
              f"R2={_fmt(m['r2'])}  N={m['n']}")
    print()

    comparison_rows = build_comparison_rows(p50_trained)
    print("COMPARISON -- A) frozen p11-trained model on p50  vs  B) p50-trained model on p50")
    print("-" * 84)
    for row in comparison_rows:
        print(f"  [{row['split']}]")
        print(f"    MAE : A={_fmt(row['p11_trained_on_p50_mae'])}  B={_fmt(row['p50_trained_mae'])}  "
              f"change={row['mae_change_pct']:+.2f}%")
        print(f"    RMSE: A={_fmt(row['p11_trained_on_p50_rmse'])}  B={_fmt(row['p50_trained_rmse'])}  "
              f"change={row['rmse_change_pct']:+.2f}%")
        print(f"    R2  : A={_fmt(row['p11_trained_on_p50_r2'])}  B={_fmt(row['p50_trained_r2'])}  "
              f"change={row['r2_change']:+.6f}")
    print()
    print("  (Positive % / positive R2 change = p50-trained model is better than the frozen")
    print("   p11-trained model, when both are evaluated on p50.)")
    print()

    write_report(scenario_check, reference_features, p50_trained, comparison_rows)

    import pandas as pd
    metrics_rows = [
        {"split": "VALIDATION", **p50_trained.validation_metrics},
        {"split": "TEST", **p50_trained.test_metrics},
        {"split": "OOD", **p50_trained.ood_metrics},
    ]
    pd.DataFrame(metrics_rows).to_csv(_METRICS_CSV, index=False)

    print("FINAL VERIFICATION SUMMARY")
    print("-" * 84)
    print("  [OK] p50 dataset loaded successfully via DatasetLoader")
    print("  [OK] p50 train/validation/test/OOD scenario composition verified against the "
          "expected split")
    print("  [OK] p50 feature schema verified against the expected model feature schema (p11)")
    print("  [OK] model trained successfully on p50 TRAIN only "
         "(training split details are recorded in the report)")
    print(f"  [OK] artifact saved to: {p50_trained.model_path}")
    print(f"  [OK] VALIDATION metrics: MAE={_fmt(p50_trained.validation_metrics['mae'])}  "
          f"RMSE={_fmt(p50_trained.validation_metrics['rmse'])}  R2={_fmt(p50_trained.validation_metrics['r2'])}")
    print(f"  [OK] TEST metrics      : MAE={_fmt(p50_trained.test_metrics['mae'])}  "
          f"RMSE={_fmt(p50_trained.test_metrics['rmse'])}  R2={_fmt(p50_trained.test_metrics['r2'])}")
    print(f"  [OK] OOD metrics       : MAE={_fmt(p50_trained.ood_metrics['mae'])}  "
          f"RMSE={_fmt(p50_trained.ood_metrics['rmse'])}  R2={_fmt(p50_trained.ood_metrics['r2'])}")
    print(f"  [OK] p11 baseline artifact NOT modified -- never opened, loaded, or written by "
          f"this script (reference path only: {P11_ARTIFACT_PATH})")
    print("  [OK] final model selection unchanged -- the original p11 HistGradientBoosting "
          "remains the selected final model")
    print()
    print(f"Report written to: {_REPORT_PATH.resolve()}")
    print(f"Metrics CSV written to: {_METRICS_CSV.resolve()}")


if __name__ == "__main__":
    main()