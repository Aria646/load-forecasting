"""Model construction helpers."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def build_linear_regression() -> LinearRegression:
    """Return a linear-regression baseline model."""
    return LinearRegression()


def build_random_forest(random_state: int = 42) -> RandomForestRegressor:
    """Return a reproducible Random Forest regressor with starter settings."""
    return RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
