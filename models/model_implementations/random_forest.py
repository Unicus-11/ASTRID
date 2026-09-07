"""
random_forest.py
==================
First concrete model implementation: a Random Forest regressor for
predicting true_queue_length_m from the feature columns supplied by the
shared data-loading pipeline (data_loader.py).

This file contains ONLY the model itself. It does not load data, split
data, compute metrics, persist anything, or compare experiments -- all of
that is handled by data_loader.py, metrics.py, persistence.py, and
experiment_runner.py. This class exists solely to satisfy the
RegressionModel interface expected by train.py / evaluate.py /
experiment_runner.py:

    fit(X, y) -> self
    predict(X) -> array-like

Feature/label separation, scenario-level splits, and column selection are
all decided upstream (in the assembled dataset's manifest.json, read by
data_loader.py) -- this model simply fits on whatever X/y it is given.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class RandomForestModel:
    """Thin, sklearn-backed wrapper around RandomForestRegressor that
    conforms to the RegressionModel protocol used throughout this
    project (train.py, evaluate.py, experiment_runner.py).

    Sensible defaults are used for a first baseline; nothing here is
    tuned, and no hyperparameter search happens in this file.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state

        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        """Fit the underlying RandomForestRegressor on the given feature
        matrix and target. X is expected to already contain only the
        manifest-declared feature columns (data_loader.py's
        responsibility) -- no imputation or preprocessing happens here."""
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> Any:
        """Predict true_queue_length_m for the given feature matrix."""
        return self._model.predict(X)