"""Tests for Mini-Project 2 (src/microgrid.py)."""

import numpy as np
import pytest

from src.microgrid import (
    HybridMicroGrid,
    MicroGrid,
    SingularSystemError,
    generate_demand_csv,
    load_demand_csv,
    read_non_negative_float,
)


@pytest.fixture
def grid() -> MicroGrid:
    return MicroGrid()


def test_determinant_and_condition_number(grid):
    assert grid.determinant == pytest.approx(-5.0)
    assert grid.condition_number == pytest.approx(3 + 2 * np.sqrt(2))  # exact value for this A
    assert grid.is_well_conditioned()


def test_solve_day_matches_cramers_rule(grid):
    x, y = grid.solve_day(120, 140)
    assert x == pytest.approx((2 * 140 - 120) / 5)
    assert y == pytest.approx((4 * 120 - 3 * 140) / 5)


def test_loop_and_vectorised_agree(grid):
    demands = np.random.default_rng(0).uniform(80, 150, size=(30, 2))
    np.testing.assert_allclose(grid.solve_loop(demands), grid.solve_vectorised(demands))


def test_zero_demand_gives_zero_dispatch(grid):
    np.testing.assert_allclose(grid.solve_day(0, 0), [0, 0])


def test_infeasible_day_flagged_and_repaired(grid):
    # D2/D1 = 1.5 > 4/3 -> exact battery output is negative
    result = grid.dispatch([[90, 135], [120, 140]])
    assert result.infeasible.tolist() == [True, False]
    assert np.all(result.dispatch >= 0)
    np.testing.assert_allclose(result.unmet[1], [0, 0], atol=1e-9)  # feasible day fully met
    assert result.daily_cost[1] == pytest.approx(150 * 32 + 450 * 12)


def test_invalid_demands_rejected(grid):
    with pytest.raises(ValueError):
        grid.solve_day(100)  # wrong count
    with pytest.raises(ValueError):
        grid.solve_day(-1, 50)  # negative


def test_singular_matrix_rejected():
    with pytest.raises(SingularSystemError):
        MicroGrid(coefficients=[[1, 2], [2, 4]])


def test_hybrid_solves_3x3_and_inherits(grid):
    hybrid = HybridMicroGrid()
    assert isinstance(hybrid, MicroGrid) and hybrid.size == 3
    x = hybrid.solve_day(120, 140, 50)
    np.testing.assert_allclose(hybrid.A @ x, [120, 140, 50])
    dependent = np.vstack([hybrid.A[:2], hybrid.A[0] + hybrid.A[1]])
    with pytest.raises(SingularSystemError):
        HybridMicroGrid(coefficients=dependent)


def test_read_non_negative_float_reprompts():
    answers = iter(["", "abc", "-4", "inf", "12.5"])
    messages: list[str] = []
    value = read_non_negative_float("D1: ", input_fn=lambda _: next(answers), output_fn=messages.append)
    assert value == 12.5
    assert len(messages) == 4  # one complaint per bad entry


def test_read_non_negative_float_gives_up():
    with pytest.raises(ValueError):
        read_non_negative_float("D1: ", input_fn=lambda _: "x", output_fn=lambda _: None, max_attempts=3)


def test_csv_round_trip_is_reproducible(tmp_path):
    p1 = generate_demand_csv(tmp_path / "a.csv", days=30, seed=7)
    p2 = generate_demand_csv(tmp_path / "b.csv", days=30, seed=7)
    d1, dates = load_demand_csv(p1)
    d2, _ = load_demand_csv(p2)
    assert d1.shape == (30, 2) and len(dates) == 30
    np.testing.assert_array_equal(d1, d2)


def test_empty_csv_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("day,date,weekday,D1_kWh,D2_kWh\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no data"):
        load_demand_csv(path)


def test_sensitivity_never_exceeds_condition_number(grid):
    sens = grid.sensitivity((120, 140), n_draws=500, rng=np.random.default_rng(3))
    assert sens["solutions"].shape == (500, 2)
    assert sens["amplification"].max() <= sens["condition_number"] + 1e-9
