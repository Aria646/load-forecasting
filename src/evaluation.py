"""Evaluation metrics for regression forecasts."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def smape(y_true, y_pred) -> float:
    """Return symmetric mean absolute percentage error in percent."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true) + np.abs(y_pred)
    ratio = np.divide(2.0 * np.abs(y_pred - y_true), denominator, out=np.zeros_like(denominator), where=denominator != 0)
    return float(np.mean(ratio) * 100)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return MAE, RMSE and sMAPE."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "sMAPE": smape(y_true, y_pred),
    }
