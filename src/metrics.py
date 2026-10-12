"""Forecast-accuracy metrics shared by Mini-Projects 1 and 5.

All functions take an array of actual values and an array of predictions of the
same length and return a single float.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _paired_arrays(actual: ArrayLike, predicted: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to float arrays and check they are non-empty and the same length."""
    a = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    if a.size == 0:
        raise ValueError("actual and predicted must not be empty")
    if a.shape != p.shape:
        raise ValueError(f"length mismatch: {a.size} actual vs {p.size} predicted")
    return a, p


def mae(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Mean absolute error, in the units of the data."""
    a, p = _paired_arrays(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Root mean squared error, in the units of the data (penalises large misses)."""
    a, p = _paired_arrays(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Mean absolute percentage error, in percent.

    Raises
    ------
    ValueError
        If any actual value is zero, because the percentage error is undefined there.
    """
    a, p = _paired_arrays(actual, predicted)
    if np.any(a == 0):
        raise ValueError("MAPE is undefined when an actual value is zero")
    return float(np.mean(np.abs((a - p) / a)) * 100.0)


def all_metrics(actual: ArrayLike, predicted: ArrayLike) -> dict[str, float]:
    """Return MAE, RMSE and MAPE together as a dictionary."""
    return {"MAE": mae(actual, predicted), "RMSE": rmse(actual, predicted), "MAPE": mape(actual, predicted)}
