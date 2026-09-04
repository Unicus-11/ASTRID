"""
hist_gradient_boosting.py
============================
HistGradientBoosting regression model implementation for predicting
true_queue_length_m from the feature columns supplied by the shared
data-loading pipeline (data_loader.py).

This file contains ONLY the model itself, kept consistent in style and
interface with random_forest.py / extra_trees.py / xgboost_model.py /
lightbgm_model.py / catboost_model.py. It does not load data, split
data, compute metrics, persist anything, or compare experiments -- all
of that is handled by data_loader.py, metrics.py, persistence.py, and
experiment_runner.py. This class exists solely to satisfy the
RegressionModel interface expected by train.py / evaluate.py /
experiment_runner.py:

    fit(X, y) -> self
    predict(X) -> array-like

No imputation or other preprocessing is applied here -- missing values in
X are passed through to HistGradientBoostingRegressor exactly as
supplied by the data loader (HistGradientBoostingRegressor natively
handles NaNs internally).
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


class HistGradientBoostingModel:
    """Thin, sklearn-backed wrapper around HistGradientBoostingRegressor
    that conforms to the RegressionModel protocol used throughout this
    project (train.py, evaluate.py, experiment_runner.py).

    Sensible baseline defaults are used; nothing here is tuned, and no
    hyperparameter search or early stopping happens in this file.
    """

    def __init__(
        self,
        max_iter: int = 300,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 31,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        random_state: int = 42,
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state

        self._model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HistGradientBoostingModel":
        """Fit the underlying HistGradientBoostingRegressor on the given
        feature matrix and target. X is expected to already contain only
        the manifest-declared feature columns (data_loader.py's
        responsibility) -- no imputation or preprocessing happens here."""
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> Any:
        """Predict true_queue_length_m for the given feature matrix."""
        return self._model.predict(X)