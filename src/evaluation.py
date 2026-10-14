"""Common regression metrics for all project experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(y_true, y_pred) -> dict[str, float]:
    """Return shared metrics; Mean Error is prediction minus actual."""
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional.")
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same length.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("y_true and y_pred must contain only finite values.")

    return {
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "R2": float(r2_score(actual, predicted)),
        "Mean Error": float(np.mean(predicted - actual)),
    }
