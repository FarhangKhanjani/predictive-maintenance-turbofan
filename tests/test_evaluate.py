"""Tests for rul.evaluate: metric values and the asymmetry of the NASA score."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rul.evaluate import alert_metrics, mae, nasa_score, rmse


def test_rmse_and_mae_on_known_values() -> None:
    assert rmse([10, 20], [13, 16]) == pytest.approx(math.sqrt((9 + 16) / 2))
    assert mae([10, 20], [13, 16]) == pytest.approx(3.5)


def test_perfect_predictions_score_zero() -> None:
    y = [5, 50, 120]
    assert rmse(y, y) == mae(y, y) == nasa_score(y, y) == 0.0


def test_nasa_score_matches_formula() -> None:
    # d = -13 (early) and d = +10 (late)
    assert nasa_score([50], [37]) == pytest.approx(math.e - 1)
    assert nasa_score([50], [60]) == pytest.approx(math.e - 1)


def test_nasa_score_penalizes_late_more_than_early() -> None:
    early = nasa_score([100], [80])  # predicted failure 20 cycles too soon
    late = nasa_score([100], [120])  # predicted failure 20 cycles too late
    assert late > early


def test_nasa_score_sum_and_mean() -> None:
    true, pred = np.array([10, 20, 30, 40]), np.array([15, 18, 30, 50])
    assert nasa_score(true, pred, "mean") == pytest.approx(nasa_score(true, pred) / 4)
    with pytest.raises(ValueError, match="reduce"):
        nasa_score(true, pred, "median")


def test_mismatched_or_empty_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="equal length"):
        rmse([1, 2], [1])
    with pytest.raises(ValueError, match="empty"):
        mae([], [])


def test_alert_metrics() -> None:
    true = [10, 25, 40, 90]  # two engines truly need an alert (<= 30)
    pred = [12, 50, 28, 95]  # flags engines 1 and 3; misses engine 2
    result = alert_metrics(true, pred, threshold=30)
    assert result["alert_precision"] == pytest.approx(0.5)
    assert result["alert_recall"] == pytest.approx(0.5)
    assert result["alerts_raised"] == 2
    assert result["engines_needing_alert"] == 2


def test_alert_metrics_without_alerts_is_nan() -> None:
    result = alert_metrics([100, 120], [110, 115], threshold=30)
    assert math.isnan(result["alert_precision"])
    assert math.isnan(result["alert_recall"])
