"""Visualisation helpers for forecast results."""

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_actual_vs_predicted(y_true, y_pred, title: str = "Actual vs Predicted Load"):
    """Create an actual-versus-predicted line chart and return the axes."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_true, label="Actual")
    ax.plot(y_pred, label="Predicted")
    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Load")
    ax.legend()
    fig.tight_layout()
    return ax
