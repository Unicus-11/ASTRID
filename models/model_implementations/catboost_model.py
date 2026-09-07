"""
catboost_model.py
====================
CatBoost regression model implementation for predicting
true_queue_length_m from the feature columns supplied by the shared
data-loading pipeline (data_loader.py).

This file contains ONLY the model itself, kept consistent in style and
interface with random_forest.py / extra_trees.py / xgboost_model.py /
lightbgm_model.py. It does not load data, split data, compute metrics,
persist anything, or compare experiments -- all of that is handled by
data_loader.py, metrics.py, persistence.py, and experiment_runner.py.
This class exists solely to satisfy the RegressionModel interface
expected by train.py / evaluate.py / experiment_runner.py:

    fit(X, y) -> self
    predict(X) -> array-like

No imputation or other preprocessing is applied here -- missing values in
X are passed through to CatBoostRegressor exactly as supplied by the
data loader (CatBoost natively handles NaNs internally).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from catboost import CatBoostRegressor


class CatBoostModel:
    """Thin, catboost-backed wrapper around CatBoostRegressor that
    conforms to the RegressionModel protocol used throughout this
    project (train.py, evaluate.py, experiment_runner.py).

    Sensible baseline defaults are used; nothing here is tuned, and no
    hyperparameter search or early stopping happens in this file.
    """

    def __init__(
        self,
        iterations: int = 300,
        learning_rate: float = 0.05,
        depth: int = 6,
        loss_function: str = "RMSE",
        random_seed: int = 42,
        verbose: bool = False,
        thread_count: int = -1,
    ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.loss_function = loss_function
        self.random_seed = random_seed
        self.verbose = verbose
        self.thread_count = thread_count

        self._model = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            loss_function=self.loss_function,
            random_seed=self.random_seed,
            verbose=self.verbose,
            thread_count=self.thread_count,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CatBoostModel":
        """Fit the underlying CatBoostRegressor on the given feature
        matrix and target. X is expected to already contain only the
        manifest-declared feature columns (data_loader.py's
        responsibility) -- no imputation, preprocessing, or early
        stopping happens here."""
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> Any:
        """Predict true_queue_length_m for the given feature matrix."""
        return self._model.predict(X)