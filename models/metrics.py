"""
metrics.py
===========
Common regression metrics shared by every model implementation.

Contains no model-specific evaluation logic -- only generic regression
metrics computed from (y_true, y_pred) arrays.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ArrayLike = Sequence[float]


def _as_array(values: ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=float)


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(_as_array(y_true), _as_array(y_pred)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root Mean Squared Error."""
    y_true_arr, y_pred_arr = _as_array(y_true), _as_array(y_pred)
    return float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Coefficient of determination (R^2)."""
    return float(r2_score(_as_array(y_true), _as_array(y_pred)))


def compute_all_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
    """Return the standard set of regression metrics for a set of
    predictions, as a plain dict suitable for logging/serialization.

    Every future model (Random Forest, XGBoost, CNN, etc.) should report
    results through this single function so metrics are computed
    identically across all of them.
    """
    y_true_arr, y_pred_arr = _as_array(y_true), _as_array(y_pred)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got "
            f"{y_true_arr.shape} vs {y_pred_arr.shape}"
        )
    if y_true_arr.size == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    return {
        "mae": mae(y_true_arr, y_pred_arr),
        "rmse": rmse(y_true_arr, y_pred_arr),
        "r2": r2(y_true_arr, y_pred_arr),
        "n": int(y_true_arr.size),
    }