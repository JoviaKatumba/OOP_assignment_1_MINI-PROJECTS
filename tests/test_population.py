"""Tests for Mini-Project 1 (src/population.py)."""

import statistics

import numpy as np
import pytest

from src.population import (
    CAGRForecaster,
    ClassroomPlanner,
    DistrictPopulation,
    FibonacciRatioForecaster,
    LinearTrendForecaster,
    ModelComparison,
    fibonacci_ratios,
)

YEARS = np.arange(2015, 2025)
KAMPALA = [1200, 1250, 1300, 1350, 1420, 1500, 1580, 1650, 1720, 1800]


@pytest.fixture
def kampala() -> DistrictPopulation:
    return DistrictPopulation("Kampala", YEARS, KAMPALA)


# ----- DistrictPopulation -----------------------------------------------------
def test_repr_and_len(kampala):
    assert len(kampala) == 10
    assert "Kampala" in repr(kampala) and "2015-2024" in repr(kampala)


@pytest.mark.parametrize(
    "years, pops",
    [
        (YEARS, KAMPALA[:-1]),  # unequal lengths
        (YEARS, [-5] + KAMPALA[1:]),  # negative value
        ([], []),  # empty input
        (YEARS[::-1], KAMPALA),  # years not increasing
    ],
)
def test_invalid_input_rejected(years, pops):
    with pytest.raises(ValueError):
        DistrictPopulation("Bad", years, pops)


def test_statistics_vs_numpy_variance(kampala):
    s = kampala.summary_statistics()
    assert s["variance"] == pytest.approx(statistics.variance(KAMPALA))
    # ddof=1 reproduces statistics.variance; ddof=0 is smaller by n/(n-1)
    assert kampala.numpy_statistics(ddof=1)["variance"] == pytest.approx(s["variance"])
    assert s["variance"] / kampala.numpy_statistics()["variance"] == pytest.approx(10 / 9)


def test_growth_rates_and_cagr(kampala):
    assert kampala.growth_rates()[0] == pytest.approx(50 / 1200)
    assert kampala.cagr() == pytest.approx((1800 / 1200) ** (1 / 9) - 1)


def test_cagr_undefined_from_zero():
    with pytest.raises(ValueError):
        DistrictPopulation("Zero", [2020, 2021], [0, 10]).cagr()


def test_split_rejects_empty_side(kampala):
    train, test = kampala.split(2021)
    assert len(train) == 7 and len(test) == 3
    with pytest.raises(ValueError):
        kampala.split(2030)


# ----- Forecasters --------------------------------------------------------------
def test_linear_trend_recovers_exact_line():
    values = 10 * np.arange(5) + 100.0
    model = LinearTrendForecaster().fit(np.arange(2000, 2005), values)
    np.testing.assert_allclose(model.predict(2), [150, 160])
    np.testing.assert_allclose(model.forecast_years(2), [2005, 2006])


def test_cagr_forecaster_on_exact_exponential():
    values = 100 * 1.05 ** np.arange(6)
    model = CAGRForecaster().fit(np.arange(6), values)
    assert model.rate == pytest.approx(0.05)
    np.testing.assert_allclose(model.predict(1), [values[-1] * 1.05])
    np.testing.assert_allclose(model.fitted_values(), values)


def test_fibonacci_ratios_and_forecast():
    np.testing.assert_allclose(fibonacci_ratios(5), [1, 2, 1.5, 5 / 3, 1.6])
    model = FibonacciRatioForecaster().fit([0, 1], [10, 10])
    np.testing.assert_allclose(model.predict(3), [10, 20, 30])


def test_predict_requires_fit_and_positive_horizon():
    with pytest.raises(RuntimeError):
        LinearTrendForecaster().predict(3)
    model = LinearTrendForecaster().fit([0, 1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        model.predict(0)


def test_bootstrap_interval_brackets_forecast(kampala):
    model = LinearTrendForecaster().fit(kampala.years, kampala.populations)
    lower, upper = model.bootstrap_interval(5, n_boot=300, rng=np.random.default_rng(1))
    point = model.predict(5)
    assert np.all(lower <= point) and np.all(point <= upper)
    assert (upper - lower)[-1] >= (upper - lower)[0]  # uncertainty grows with horizon


def test_model_comparison_selects_cagr_for_exponential_series():
    years = np.arange(2015, 2025)
    district = DistrictPopulation("Expo", years, 500 * 1.08 ** np.arange(10))
    models = [LinearTrendForecaster(), CAGRForecaster(), FibonacciRatioForecaster()]
    table = ModelComparison(district, models, 2021).run()
    assert table.index[0] == "Exponential (CAGR)"
    assert set(table.columns) == {"MAE", "RMSE", "MAPE"}


# ----- ClassroomPlanner ---------------------------------------------------------
def test_classroom_planner_hand_calculation():
    planner = ClassroomPlanner()
    # 100k extra people * 0.18 = 18,000 pupils / 53 = 339.6 -> 340
    assert planner.additional_classrooms(1000, 1100) == 340


def test_classroom_planner_edge_cases():
    planner = ClassroomPlanner()
    assert planner.additional_classrooms(500, 500) == 0  # no growth
    assert planner.additional_classrooms(500, 400) == 0  # decline never negative
    with pytest.raises(ValueError):
        ClassroomPlanner(pupils_per_classroom=0)
