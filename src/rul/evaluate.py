"""Metrics for RUL prediction.

- RMSE: typical error in cycles, with large misses weighted heavily.
- MAE: average absolute miss in cycles; easier to explain, less sensitive to outliers.
- NASA score: asymmetric exponential penalty from the PHM08 challenge. Predicting
  failure too late (overestimating RUL) is penalized more than too early, because
  a late prediction means an engine can fail in service, while an early one only
  costs a maintenance slot.
- Early-warning alerts: treat "predicted RUL <= threshold" as a maintenance alert
  and measure how many engines that are truly close to failure get flagged.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

# Time constants from Saxena et al. (2008): late errors grow faster (10 < 13).
NASA_EARLY_TAU = 13.0
NASA_LATE_TAU = 10.0


def _as_arrays(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    true, pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape or true.ndim != 1:
        raise ValueError(
            f"expected two 1-D arrays of equal length, got {true.shape} and {pred.shape}"
        )
    if len(true) == 0:
        raise ValueError("cannot score empty arrays")
    return true, pred


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error in cycles."""
    true, pred = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error in cycles."""
    true, pred = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(pred - true)))


def nasa_score(y_true: ArrayLike, y_pred: ArrayLike, reduce: str = "sum") -> float:
    """PHM08 scoring function; lower is better, 0 is perfect.

    With d = predicted - true: exp(-d / 13) - 1 if d < 0 (early),
    exp(d / 10) - 1 if d >= 0 (late).

    Args:
        reduce: ``"sum"`` (the literature convention for the 100 test engines)
            or ``"mean"`` (comparable across sets of different size, used in CV).
    """
    true, pred = _as_arrays(y_true, y_pred)
    d = pred - true
    scores = np.where(d < 0, np.exp(-d / NASA_EARLY_TAU), np.exp(d / NASA_LATE_TAU)) - 1.0
    if reduce == "sum":
        return float(scores.sum())
    if reduce == "mean":
        return float(scores.mean())
    raise ValueError(f"reduce must be 'sum' or 'mean', got {reduce!r}")


def alert_metrics(
    y_true: ArrayLike, y_pred: ArrayLike, threshold: float = 30.0
) -> dict[str, float]:
    """Precision and recall of the alert "predicted RUL <= threshold".

    Recall answers "of the engines that really have <= threshold cycles left,
    how many did we flag?" (missed ones risk failure in service). Precision
    answers "of the flagged engines, how many really needed it?" (false alarms
    cost unnecessary maintenance).
    """
    true, pred = _as_arrays(y_true, y_pred)
    actual, flagged = true <= threshold, pred <= threshold
    hits = int(np.sum(actual & flagged))
    return {
        "alert_threshold": float(threshold),
        "alert_precision": hits / int(flagged.sum()) if flagged.any() else float("nan"),
        "alert_recall": hits / int(actual.sum()) if actual.any() else float("nan"),
        "alerts_raised": int(flagged.sum()),
        "engines_needing_alert": int(actual.sum()),
    }
