"""Tests for the shared forecast-accuracy metrics."""

import math

import pytest

from src.metrics import all_metrics, mae, mape, rmse


def test_metrics_match_hand_calculation():
    actual, predicted = [100, 200, 300], [110, 190, 330]
    # errors: -10, 10, -30
    assert mae(actual, predicted) == pytest.approx(50 / 3)
    assert rmse(actual, predicted) == pytest.approx(math.sqrt((100 + 100 + 900) / 3))
    assert mape(actual, predicted) == pytest.approx((10 + 5 + 10) / 3)


def test_perfect_forecast_scores_zero():
    assert all_metrics([1, 2, 3], [1, 2, 3]) == {"MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0}


@pytest.mark.parametrize("actual, predicted", [([], []), ([1, 2], [1])])
def test_empty_or_mismatched_input_rejected(actual, predicted):
    with pytest.raises(ValueError):
        mae(actual, predicted)


def test_mape_undefined_for_zero_actual():
    with pytest.raises(ValueError, match="zero"):
        mape([0, 1], [1, 1])
