"""Tests for Mini-Project 5 (src/taxi.py)."""

import numpy as np
import pytest

from src.taxi import (
    Backtester,
    ExponentialSmoothingForecaster,
    FleetPlanner,
    LinearTrendForecaster,
    MarketModel,
    MovingAverageForecaster,
    Route,
    SeasonalNaiveForecaster,
    simulate_weekly_demand,
)

NTINDA = [35, 40, 42, 50, 55, 60, 48, 52, 47, 45]


def test_route_revenue_and_statistics():
    r = Route("Kampala-Ntinda", NTINDA, fare=2000)
    assert r.daily_revenue()[0] == 70_000
    assert r.total_revenue() == sum(NTINDA) * 2000
    assert r.describe()["mean"] == pytest.approx(47.4)
    assert len(r) == 10


@pytest.mark.parametrize("pax, fare", [([], 2000), ([10, -1], 2000), ([10, 20], 0)])
def test_route_rejects_invalid_input(pax, fare):
    with pytest.raises(ValueError):
        Route("Bad", pax, fare)


def test_equilibrium_matches_hand_calculation():
    q, p = MarketModel().equilibrium()
    assert p == pytest.approx(2200)  # 110 / 0.05
    assert q == pytest.approx(76)
    assert MarketModel().excess_demand(2000) == pytest.approx(10)  # 80 demanded - 70 supplied


def test_parallel_market_rejected():
    with pytest.raises(ValueError):
        MarketModel(demand_slope=0, supply_slope=0)


def test_moving_average_and_ses_hand_values():
    assert MovingAverageForecaster(3).fit([1, 2, 3, 4]).predict(1)[0] == pytest.approx(3)
    # level: 10 -> 0.5*20 + 0.5*10 = 15 -> 0.5*30 + 0.5*15 = 22.5
    assert ExponentialSmoothingForecaster(0.5).fit([10, 20, 30]).predict(2).tolist() == [22.5, 22.5]
    # alpha = 1 is the naive forecast
    assert ExponentialSmoothingForecaster(1.0).fit(NTINDA).predict(1)[0] == NTINDA[-1]


def test_linear_trend_and_seasonal_naive():
    np.testing.assert_allclose(LinearTrendForecaster().fit([2, 4, 6]).predict(2), [8, 10])
    week = [10, 20, 30, 40, 50, 60, 70]
    np.testing.assert_allclose(SeasonalNaiveForecaster(7).fit(week * 2).predict(8), week + [10])


def test_forecaster_edge_cases():
    with pytest.raises(ValueError):
        MovingAverageForecaster(3).fit([1, 2])  # too little history
    with pytest.raises(RuntimeError):
        LinearTrendForecaster().predict(1)  # not fitted
    with pytest.raises(ValueError):
        ExponentialSmoothingForecaster(0)  # alpha out of range


def test_walk_forward_targets_days_4_to_10_without_lookahead():
    res = Backtester(start_index=3).run(NTINDA, MovingAverageForecaster(3))
    assert (res.target_index + 1).tolist() == [4, 5, 6, 7, 8, 9, 10]
    assert res.predicted[0] == pytest.approx(np.mean(NTINDA[:3]))  # uses days 1-3 only
    assert res.mae == pytest.approx(np.mean(np.abs(res.actual - res.predicted)))


def test_tune_alpha_returns_grid_minimum():
    best, scores = Backtester(3).tune_alpha(NTINDA, [0.2, 0.5, 1.0])
    assert best == min(scores, key=scores.get) and set(scores) == {0.2, 0.5, 1.0}


def test_fleet_planner():
    fp = FleetPlanner()
    assert fp.daily_capacity == 112
    assert fp.vehicles_needed(45) == 1
    assert fp.vehicles_needed(200) == 3  # 230 / 112 = 2.05 -> 3
    assert fp.vehicles_needed(0) == 0  # zero demand edge case
    with pytest.raises(ValueError):
        fp.vehicles_needed(-1)


def test_seasonal_naive_beats_moving_average_on_weekly_data():
    sim = simulate_weekly_demand(60, base=50, rng=np.random.default_rng(0))
    bt = Backtester(start_index=14)
    assert bt.run(sim, SeasonalNaiveForecaster(7)).mae < bt.run(sim, MovingAverageForecaster(3)).mae
