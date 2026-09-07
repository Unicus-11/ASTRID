"""
experiment_runner.py
======================
Shared experiment orchestrator.

This module implements ONE fixed procedure for running an experiment with
ANY future model:

    ExperimentConfig
          |
    DatasetLoader
          |
    load train / validation / test / ood
          |
    fit model on TRAIN only
          |
    evaluate trained model on VALIDATION
          |
    evaluate trained model on TEST
          |
    evaluate trained model on OOD
          |
    collect metrics and experiment metadata
          |
    return a structured ExperimentResult

It contains NO model-specific logic (no Random Forest, XGBoost, CNN,
etc.), no feature engineering, no preprocessing, no imputation, no
dataset splitting, and no duplicated metric/persistence logic -- all of
that is delegated to the existing shared modules:

    experiment_config.py -> Layer, Split, ExperimentConfig
    data_loader.py       -> DatasetLoader, SplitData
    train.py              -> RegressionModel, build_training_inputs, train_model
    evaluate.py            -> evaluate_model, EvaluationReport
    persistence.py         -> (used indirectly, via train.train_model)

Model-selection convention
---------------------------
`ExperimentResult` carries validation, test, AND ood metrics, because the
project's evaluation protocol requires OOD to always be evaluated and
reported. However, OOD must never be used to select or tune a model.
Anything in this file (or downstream) that compares/selects between
experiments should read `EvaluationReport.selection_metrics()` (defined
in evaluate.py, which deliberately excludes OOD) or, equivalently, only
the `.validation` / `.test` fields of the result -- never `.ood`.
`results_to_dataframe()` includes ood_* columns for visibility/reporting
only; it performs no ranking or selection of any kind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import pandas as pd

from data_loader import DatasetLoader, SplitData
from experiment_config import ExperimentConfig, Layer, Split
from evaluate import EvaluationReport, evaluate_model
from train import RegressionModel, build_training_inputs, train_model

# A model can be supplied either as an already-constructed object that
# conforms to RegressionModel, or as a zero-argument factory that
# produces one. Factories are useful when running the same model class
# across multiple experiments/layers, since a fitted estimator should
# not generally be reused across independent training runs.
ModelOrFactory = Union[RegressionModel, Callable[[], RegressionModel]]


# ---------------------------------------------------------------------------
# Structured result
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """The structured output of a single run_experiment() call.

    Every future model implementation produces exactly this shape of
    result, so results from different model families can be compared
    directly through results_to_dataframe().
    """

    # Identity / metadata
    model_name: str
    experiment_name: str
    layer: Layer
    target_column: str
    random_state: int

    # Metric blocks -- each is the dict returned by
    # metrics.compute_all_metrics() (mae, rmse, r2, n), obtained via
    # evaluate.py's SplitEvaluation.metrics. Never computed here directly.
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    ood_metrics: Dict[str, float]

    # Provenance, preserved for traceability
    feature_columns: List[str]
    scenario_ids: Dict[str, List[str]] = field(default_factory=dict)  # split-name -> scenario_ids

    # Where the trained model was persisted, if it was.
    model_path: Optional[Path] = None

    def to_dict(self) -> dict:
        """Flat-ish dict representation, convenient for logging or JSON
        serialization. Does not compute anything new -- just reshapes
        fields already on this object."""
        return {
            "model_name": self.model_name,
            "experiment_name": self.experiment_name,
            "layer": self.layer.value,
            "target_column": self.target_column,
            "random_state": self.random_state,
            "validation": self.validation_metrics,
            "test": self.test_metrics,
            "ood": self.ood_metrics,
            "feature_columns": self.feature_columns,
            "scenario_ids": self.scenario_ids,
            "model_path": str(self.model_path) if self.model_path else None,
        }


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

def _resolve_model(model_or_factory: ModelOrFactory) -> RegressionModel:
    """Accept either a ready-to-fit model instance or a zero-argument
    factory that constructs one, and return a model instance.

    This is the ONLY place model construction is touched, and it never
    inspects or branches on what kind of model it is -- it only checks
    whether the object already looks like a fitted-interface model
    (has .fit and .predict) versus something callable that should be
    invoked to produce one.
    """
    if hasattr(model_or_factory, "fit") and hasattr(model_or_factory, "predict"):
        return model_or_factory  # already a RegressionModel instance
    if callable(model_or_factory):
        produced = model_or_factory()
        if not (hasattr(produced, "fit") and hasattr(produced, "predict")):
            raise TypeError(
                "model factory did not return an object with fit()/predict(); "
                f"got {type(produced)!r}"
            )
        return produced
    raise TypeError(
        "model_or_factory must be a RegressionModel instance (with fit/predict) "
        "or a zero-argument callable that returns one; "
        f"got {type(model_or_factory)!r}"
    )


# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------

def run_experiment(
    config: ExperimentConfig,
    model: ModelOrFactory,
    model_name: str,
    save_model_flag: bool = True,
) -> ExperimentResult:
    """Run one full experiment for a single model against a single layer,
    following the fixed protocol:

        train      -> fitting only
   validation -> model selection / tuning
   test       -> final in-distribution evaluation
   ood        -> final out-of-distribution robustness evaluation Parameters
    ----------
    config : ExperimentConfig
        Shared experiment settings (layer, target column, paths, seed,
        experiment name). Comes from experiment_config.py -- this
        function does not construct or modify one.
    model : RegressionModel or Callable[[], RegressionModel]
        Either an already-constructed model conforming to the
        RegressionModel protocol in train.py, or a zero-argument factory
        that produces one. This function contains no knowledge of what
        kind of model it is.
    model_name : str
        Human-readable identifier for this model (e.g. "random_forest"),
        used in the result and in the saved-model filename. Distinct from
        config.experiment_name, which identifies the overall experiment
        (e.g. "baseline_layer1_sweep").
    save_model_flag : bool
        If True (default), the fitted model is persisted via
        persistence.save_model() (called inside train.train_model()) to
        config.output_dir() / f"{model_name}.joblib". If False, no model
        artifact is written and ExperimentResult.model_path is None.

    Returns
    -------
    ExperimentResult
    """
    # 1. Load all four splits for this layer through the shared loader.
    #    No re-splitting, no imputation, no leakage guard bypass -- all
    #    of that is handled inside data_loader.py.
    loader = DatasetLoader(
        layer=config.layer,
        data_root=config.data_root,
        target_column=config.target_column,
    )
    splits: Dict[Split, SplitData] = loader.load_all()

    train_split = splits[Split.TRAIN]
    validation_split = splits[Split.VALIDATION]
    test_split = splits[Split.TEST]
    ood_split = splits[Split.OOD]

    # 2. Resolve the model (instance or factory) but do not construct or
    #    configure it beyond that -- hyperparameters are entirely the
    #    caller's responsibility.
    resolved_model = _resolve_model(model)

    # 3. Fit on TRAIN only, via the shared training interface.
    training_inputs = build_training_inputs(train_split, config)

    save_path: Optional[Path] = None
    if save_model_flag:
        save_path = config.output_dir() / f"{model_name}.joblib"

    fitted_model = train_model(resolved_model, training_inputs, save_path=save_path)

    # 4. Evaluate on validation / test / ood, via the shared evaluation
    #    interface, which itself uses metrics.py for every number.
    report: EvaluationReport = evaluate_model(
        fitted_model,
        validation_split=validation_split,
        test_split=test_split,
        ood_split=ood_split,
    )

    # 5. Collect metadata alongside the metrics -- nothing here is
    #    recomputed; it is only reshaped from what the splits/report
    #    already carry.
    scenario_ids = {
        "train": train_split.scenario_ids(),
        "validation": validation_split.scenario_ids(),
        "test": test_split.scenario_ids(),
        "ood": ood_split.scenario_ids(),
    }

    return ExperimentResult(
        model_name=model_name,
        experiment_name=config.experiment_name or "unnamed_experiment",
        layer=config.layer,
        target_column=config.target_column,
        random_state=config.random_state,
        validation_metrics=report.validation.metrics,
        test_metrics=report.test.metrics,
        ood_metrics=report.ood.metrics,
        feature_columns=list(train_split.feature_columns),
        scenario_ids=scenario_ids,
        model_path=save_path,
    )


# ---------------------------------------------------------------------------
# Comparison support
# ---------------------------------------------------------------------------

def results_to_dataframe(results: List[ExperimentResult]) -> pd.DataFrame:
    """Convert a list of ExperimentResult into a single comparison
    DataFrame, one row per experiment.

    This is purely a reshaping/formatting step: it does not rank, sort,
    filter, or select a "best" model, and it does not use OOD (or
    anything else) to make any decision -- ood_* columns are included for
    visibility only, alongside validation_*/test_*, so a human (or later,
    separate selection code operating only on the validation_*/test_*
    columns) can compare candidates.
    """
    rows = []
    for r in results:
        rows.append(
            {
                "model": r.model_name,
                "experiment_name": r.experiment_name,
                "layer": r.layer.value,
                "target": r.target_column,
                "random_state": r.random_state,
                "validation_mae": r.validation_metrics.get("mae"),
                "validation_rmse": r.validation_metrics.get("rmse"),
                "validation_r2": r.validation_metrics.get("r2"),
                "validation_n": r.validation_metrics.get("n"),
                "test_mae": r.test_metrics.get("mae"),
                "test_rmse": r.test_metrics.get("rmse"),
                "test_r2": r.test_metrics.get("r2"),
                "test_n": r.test_metrics.get("n"),
                "ood_mae": r.ood_metrics.get("mae"),
                "ood_rmse": r.ood_metrics.get("rmse"),
                "ood_r2": r.ood_metrics.get("r2"),
                "ood_n": r.ood_metrics.get("n"),
                "n_features": len(r.feature_columns),
                "model_path": str(r.model_path) if r.model_path else None,
            }
        )
    return pd.DataFrame(rows)