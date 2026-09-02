"""Feature-engineering helpers for electricity load forecasting."""

from __future__ import annotations

import pandas as pd


def add_time_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Create basic calendar features from a timestamp column."""
    result = df.copy()
    ts = pd.to_datetime(result[timestamp_col])
    result["hour"] = ts.dt.hour
    result["day_of_week"] = ts.dt.dayofweek
    result["month"] = ts.dt.month
    result["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    return result


def add_lag_features(df: pd.DataFrame, target_col: str, lags=(1, 24, 168)) -> pd.DataFrame:
    """Add lagged target values. Adjust lags if the final interval is not hourly."""
    result = df.copy()
    for lag in lags:
        result[f"{target_col}_lag_{lag}"] = result[target_col].shift(lag)
    return result
