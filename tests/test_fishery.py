"""Tests for Mini-Project 3 (src/fishery.py)."""

import numpy as np
import pytest

from src.fishery import FishStock, PriceModel, RiskAssessor, fibonacci, weekly_revenue


def test_fibonacci_first_terms_and_empty():
    assert fibonacci(7) == [1, 1, 2, 3, 5, 8, 13]
    assert fibonacci(0) == []


def test_first_logistic_step_matches_hand_calculation():
    traj = FishStock(r=0.4, K=10_000, N0=4_000, h=0.1).simulate(1)
    # 4000 + 0.4*4000*0.6 - 0.1*4000 = 4560
    assert traj.stock[1] == pytest.approx(4560)
    assert traj.harvest[0] == pytest.approx(400)


@pytest.mark.parametrize("h", [0.05, 0.1, 0.2, 0.3])
def test_stock_converges_to_equilibrium(h):
    fs = FishStock(h=h)
    assert fs.simulate(500).final_stock == pytest.approx(fs.equilibrium_stock, rel=1e-6)


def test_msy_theory():
    fs = FishStock(r=0.4, K=10_000, h=0.2)
    assert fs.msy == 1000 and fs.msy_rate == 0.2
    assert fs.equilibrium_yield == pytest.approx(fs.msy)


def test_zero_stock_stays_zero_and_no_harvest_in_closed_weeks():
    assert FishStock(N0=0).simulate(10).final_stock == 0
    traj = FishStock(h=0.2, closed_weeks=[0, 1]).simulate(3)
    assert traj.harvest[0] == 0 and traj.harvest[1] == 0 and traj.harvest[2] > 0


@pytest.mark.parametrize("kwargs", [{"h": 1.0}, {"h": -0.1}, {"r": 0}, {"closed_weeks": [60]}])
def test_invalid_stock_parameters(kwargs):
    with pytest.raises(ValueError):
        FishStock(**kwargs)


def test_price_model_bounded_and_seeded():
    pm = PriceModel(step_sd=2000)  # large steps so the bounds are hit
    a = pm.simulate_paths(50, 52, np.random.default_rng(1))
    b = pm.simulate_paths(50, 52, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    assert a.min() >= 9000 and a.max() <= 16000
    assert np.all(a[:, 0] == 12000)


def test_weekly_revenue_units():
    # 2 t at UGX 10,000/kg = 2000 kg * 10,000 = 20,000,000
    np.testing.assert_allclose(weekly_revenue([2.0], [10_000]), [2e7])


def test_cv_is_scale_invariant_but_variance_is_not():
    rev = np.array([100.0, 120.0, 90.0, 110.0])
    a = RiskAssessor.describe(rev)
    b = RiskAssessor.describe(rev * 1e6)
    assert a["cv"] == pytest.approx(b["cv"])
    assert b["variance"] == pytest.approx(a["variance"] * 1e12)


def test_risk_classes():
    ra = RiskAssessor(low_threshold=0.1, high_threshold=0.25)
    assert ra.classify_cv(0.05) == "Low"
    assert ra.classify_cv(0.1) == "Moderate"
    assert ra.classify_cv(0.3) == "High"
    assert ra.classify([100, 100, 100]) == "Low"  # zero variability edge case


def test_describe_rejects_zero_mean_and_short_input():
    with pytest.raises(ValueError):
        RiskAssessor.describe([0, 0, 0])
    with pytest.raises(ValueError):
        RiskAssessor.describe([5])


def test_value_at_risk():
    outcomes = np.arange(1, 101, dtype=float)  # 1..100
    var = RiskAssessor(var_level=0.05).value_at_risk(outcomes)
    assert var["quantile"] == pytest.approx(np.quantile(outcomes, 0.05))
    assert var["var_shortfall"] == pytest.approx(50.5 - var["quantile"])
