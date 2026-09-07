"""
lightbgm_model.py
====================
LightGBM regression model implementation for predicting
true_queue_length_m from the feature columns supplied by the shared
data-loading pipeline (data_loader.py).

This file contains ONLY the model itself, kept consistent in style and
interface with random_forest.py / extra_trees.py / xgboost_model.py. It
does not load data, split data, compute metrics, persist anything, or
compare experiments -- all of that is handled by data_loader.py,
metrics.py, persistence.py, and experiment_runner.py. This class exists
solely to satisfy the RegressionModel interface expected by train.py /
evaluate.py / experiment_runner.py:

    fit(X, y) -> self
    predict(X) -> array-like

No imputation or other preprocessing is applied here -- missing values in
X are passed through to LGBMRegressor exactly as supplied by the data
loader (LightGBM natively handles NaNs internally).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from lightgbm import LGBMRegressor


class LightGBMModel:
    """Thin, lightgbm-backed wrapper around LGBMRegressor that conforms
    to the RegressionModel protocol used throughout this project
    (train.py, evaluate.py, experiment_runner.py).

    Sensible baseline defaults are used; nothing here is tuned, and no
    hyperparameter search happens in this file.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = -1,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_jobs = n_jobs
        self.random_state = random_state

        self._model = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMModel":
        """Fit the underlying LGBMRegressor on the given feature matrix
        and target. X is expected to already contain only the
        manifest-declared feature columns (data_loader.py's
        responsibility) -- no imputation or preprocessing happens here."""
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> Any:
        """Predict true_queue_length_m for the given feature matrix."""
        return self._model.predict(X)