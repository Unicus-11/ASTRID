"""
mlp.py
=======
MLP (Multi-Layer Perceptron) regression model implementation for
predicting true_queue_length_m from the feature columns supplied by the
shared data-loading pipeline (data_loader.py).

Unlike the tree-based models already in this project (HistGradientBoosting,
Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost), scikit-learn's
MLPRegressor:

  * does NOT accept NaN values in X (Layer 2's assembled features can
    contain NaNs -- e.g. is_green_for_approach for approaches with no
    signal), and
  * is sensitive to feature scale, since it is fit with gradient-based
    optimization rather than being scale-invariant like a tree split.

This file therefore wraps MLPRegressor together with a SimpleImputer and
a StandardScaler into a single fit/predict object, so that:

  * Imputation and scaling statistics are learned ONLY from whatever X is
    passed to fit() -- which, per train.py / experiment_runner.py, is
    always the TRAIN split's X. They are never fit again later, and
    never derived from validation/test/ood.
  * The exact same fitted imputer + scaler are then applied, unchanged,
    to whatever X is passed to predict() (validation, test, ood, or any
    future data) -- so there is no leakage of validation/test/ood
    statistics into preprocessing, and no leakage of the target column
    into preprocessing (the imputer/scaler never see y at all).
  * Because the imputer, scaler, and MLPRegressor are all plain
    attributes of this one object, persistence.save_model() -- a single
    joblib.dump() of the whole fitted model object, already used
    unchanged by every other model in this project -- persists all three
    together automatically. No changes to persistence.py, train.py,
    evaluate.py, or experiment_runner.py were required.

This file contains ONLY the model itself, kept consistent in style and
interface with random_forest.py / extra_trees.py / xgboost_model.py /
lightbgm_model.py / catboost_model.py / hist_gradient_boosting.py. It
does not load data, split data, compute metrics, persist anything, or
compare experiments -- all of that is handled by data_loader.py,
metrics.py, persistence.py, and experiment_runner.py. This class exists
solely to satisfy the RegressionModel interface expected by train.py /
evaluate.py / experiment_runner.py:

    fit(X, y) -> self
    predict(X) -> array-like

No hyperparameter search or tuning happens in this file -- this is a
fixed baseline configuration only, matching the project's convention for
its other six baseline models.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class MLPModel:
    """Thin, sklearn-backed wrapper around MLPRegressor that conforms to
    the RegressionModel protocol used throughout this project (train.py,
    evaluate.py, experiment_runner.py).

    Preprocessing pipeline (both steps learned on the fit() input only):

      1. SimpleImputer(strategy="median") -- fills missing feature
         values (e.g. Layer 2's NaN-containing columns) using the
         per-column median computed from the training split. This is a
         deliberate, training-only preprocessing step specific to this
         model, distinct from data_loader.py's guarantee that X is
         handed to models with missing values preserved exactly as
         assembled -- MLPRegressor requires it. It is fit once, on
         TRAIN, and never refit on validation/test/ood, and it never has
         access to the target column.
      2. StandardScaler -- zero-mean/unit-variance scaling, fit on the
         already-imputed training features only, then applied unchanged
         to any future predict() input.
      3. MLPRegressor -- the actual baseline model.

    Sensible, fixed baseline defaults are used; nothing here is tuned,
    and no hyperparameter search or sweep happens in this file.
    `early_stopping` is fixed to False, so the network trains on the
    complete TRAIN split for the full max_iter, matching how the other
    six fixed baselines in this project use all of TRAIN.
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (100, 50),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state

        # Preprocessing objects. Fit exclusively inside fit(), on
        # whatever X is passed there (the TRAIN split, per
        # train.build_training_inputs / experiment_runner.run_experiment)
        # -- never pre-fit here, and never re-fit anywhere else.
        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()

        # early_stopping=False (the MLPRegressor default) so the network
        # trains on the complete TRAIN split for up to max_iter iterations,
        # with no internal held-out slice of TRAIN carved off for its own
        # convergence check -- this keeps the first MLP baseline directly
        # comparable to the other six fixed baselines, all of which also
        # train on 100% of TRAIN. n_iter_no_change / validation_fraction
        # are omitted entirely since they only take effect when
        # early_stopping=True.
        self._model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=False,
            random_state=self.random_state,
        )

        # Feature-column order captured at fit() time, so predict() can
        # fail loudly instead of silently misaligning columns.
        self._feature_columns: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLPModel":
        """Fit imputer -> scaler -> MLPRegressor, in that order, on this
        X/y. Under the shared experiment infrastructure this is always
        called once, on the TRAIN split's (X, y) (see
        train.train_model() / experiment_runner.run_experiment()).

        Imputer and scaler statistics are derived only from X. The
        target y is passed only to the final MLPRegressor.fit() call,
        never to the preprocessing steps -- so no target information can
        leak into imputation or scaling.
        """
        self._feature_columns = list(X.columns)

        X_imputed = self._imputer.fit_transform(X)
        X_scaled = self._scaler.fit_transform(X_imputed)

        self._model.fit(X_scaled, y)
        return self

    def predict(self, X: pd.DataFrame) -> Any:
        """Predict true_queue_length_m for the given feature matrix.

        Applies the already-fitted imputer and scaler (learned in fit(),
        from TRAIN only) unchanged, then delegates to the fitted
        MLPRegressor. Used as-is for validation, test, and ood
        evaluation by evaluate.py, so all three splits are preprocessed
        identically to how TRAIN was.
        """
        if self._feature_columns is not None and list(X.columns) != self._feature_columns:
            raise ValueError(
                "MLPModel.predict() received a different set/order of feature "
                "columns than it was fit on.\n"
                f"  fit on:  {self._feature_columns}\n"
                f"  got:     {list(X.columns)}"
            )

        X_imputed = self._imputer.transform(X)
        X_scaled = self._scaler.transform(X_imputed)
        return self._model.predict(X_scaled)