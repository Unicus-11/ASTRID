"""
train.py
=========
Common training entry point / interface.

This module defines HOW a future model implementation plugs into the
shared experiment structure -- it does not implement any specific model.
Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, and a CNN should
all be able to conform to the same small interface defined here.

No hyperparameter search, no model comparison logic, and no dataset
modification happens in this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Protocol, runtime_checkable

import pandas as pd

from data_loader import SplitData
from experiment_config import ExperimentConfig
from persistence import save_model


@runtime_checkable
class RegressionModel(Protocol):
    """The minimal interface every future model must satisfy to plug into
    this training/evaluation infrastructure. This mirrors the standard
    scikit-learn estimator interface, which XGBoost, LightGBM, and
    CatBoost's scikit-learn wrappers already satisfy; a CNN implementation
    would need only a thin wrapper exposing these same two methods.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RegressionModel":
        ...

    def predict(self, X: pd.DataFrame) -> Any:
        ...


@dataclass
class TrainingInputs:
    """Everything a model implementation's training step receives.
    Bundled into one object so every future model gets the exact same
    inputs, prepared the exact same way.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    feature_columns: List[str]
    config: ExperimentConfig


def build_training_inputs(train_split: SplitData, config: ExperimentConfig) -> TrainingInputs:
    """Assemble TrainingInputs from an already-loaded training SplitData
    and the experiment config. Performs no fitting -- purely a data-
    shaping step, kept separate so it's reusable and testable on its own.
    """
    if train_split.split.value != "train":
        raise ValueError(
            f"build_training_inputs expects the 'train' split, got '{train_split.split.value}'"
        )
    return TrainingInputs(
        X_train=train_split.X,
        y_train=train_split.y,
        feature_columns=list(train_split.feature_columns),
        config=config,
    )


def train_model(
    model: RegressionModel,
    inputs: TrainingInputs,
    save_path: Optional[Path] = None,
) -> RegressionModel:
    """Fit `model` (any object conforming to RegressionModel) on the given
    TrainingInputs, and optionally persist the result.

    This function does not know or care what kind of model it was given
    -- it only calls .fit(X, y) and, if requested, saves the result via
    persistence.save_model(). Model construction and hyperparameters
    remain entirely the caller's responsibility; no model comparison or
    hyperparameter search logic lives here.
    """
    fitted = model.fit(inputs.X_train, inputs.y_train)
    # Some estimators return None from fit(); fall back to the original
    # object reference in that case (mirrors the common convention of
    # fit() returning self, without assuming it).
    fitted_model = fitted if fitted is not None else model

    if save_path is not None:
        save_model(fitted_model, save_path)

    return fitted_model