"""Data cleaning and preprocessing helpers for the load-forecasting project."""

from __future__ import annotations

import pandas as pd


def parse_timestamp(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return a copy of *df* with a parsed and sorted timestamp column."""
    result = df.copy()
    result[column] = pd.to_datetime(result[column], errors="coerce")
    return result.sort_values(column).reset_index(drop=True)


def report_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise missing values by column."""
    missing = df.isna().sum()
    return pd.DataFrame({"missing_count": missing, "missing_pct": missing / len(df) * 100})
