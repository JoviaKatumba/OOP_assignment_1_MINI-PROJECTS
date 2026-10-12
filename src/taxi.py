"""Mini-Project 5: Taxi Route Revenue, Pricing & Fleet Planner (Kampala matatus).

Classes
-------
Route
    Daily passenger counts and the fare for one route.
MarketModel
    Linear supply and demand, solved for equilibrium with ``scipy.linalg.solve``.
SeriesForecaster
    Abstract base for one-step-ahead daily forecasters, with subclasses
    MovingAverageForecaster, ExponentialSmoothingForecaster,
    LinearTrendForecaster and SeasonalNaiveForecaster.
Backtester
    Rolling-origin (walk-forward) evaluation and alpha grid search.
FleetPlanner
    Converts a passenger forecast into a number of vehicles.

Functions
---------
simulate_weekly_demand
"""

from __future__ import annotations

import math
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike

from src.metrics import mae


# =================================================================================
# Route
# =================================================================================
class Route:
    """One taxi route.

    Parameters
    ----------
    name:
        Route name, e.g. ``"Kampala-Ntinda"``.
    passengers:
        Daily passenger counts (non-negative).
    fare:
        Fare per passenger in UGX (positive).
    """

    def __init__(self, name: str, passengers: ArrayLike, fare: float) -> None:
        counts = np.asarray(passengers, dtype=float)
        if not name:
            raise ValueError("name must be non-empty")
        if counts.ndim != 1 or counts.size == 0:
            raise ValueError("passengers must be a non-empty 1-D sequence")
        if not np.all(np.isfinite(counts)) or np.any(counts < 0):
            raise ValueError("passenger counts must be finite and non-negative")
        if fare <= 0:
            raise ValueError("fare must be positive")
        self.name = name
        self.passengers = counts
        self.fare = float(fare)

    def __repr__(self) -> str:
        return f"Route({self.name!r}, days={len(self)}, fare=UGX {self.fare:,.0f})"

    def __len__(self) -> int:
        return int(self.passengers.size)

    def daily_revenue(self) -> np.ndarray:
        """Revenue for each day (UGX)."""
        return self.passengers * self.fare

    def total_revenue(self) -> float:
        return float(self.daily_revenue().sum())

    def describe(self) -> dict[str, float]:
        """Mean, sample variance and standard deviation of daily passengers (``statistics``)."""
        data = self.passengers.tolist()
        if len(data) < 2:
            raise ValueError("need at least two days for variance")
        return {"mean": statistics.mean(data), "variance": statistics.variance(data), "std": statistics.stdev(data)}


# =================================================================================
# Supply and demand
# =================================================================================
@dataclass(frozen=True)
class MarketModel:
    """Linear demand ``Qd = a - b P`` and supply ``Qs = c + d P``.

    Q is passengers per trip-hour and P is the fare in UGX.
    """

    demand_intercept: float = 120.0
    demand_slope: float = 0.02
    supply_intercept: float = 10.0
    supply_slope: float = 0.03

    def __post_init__(self) -> None:
        if self.demand_slope < 0 or self.supply_slope < 0:
            raise ValueError("slopes are given as magnitudes and must be non-negative")
        if self.demand_slope + self.supply_slope == 0:
            raise ValueError("demand and supply are parallel: no unique equilibrium")

    def as_linear_system(self) -> tuple[np.ndarray, np.ndarray]:
        """Unknowns [Q, P]:  Q + bP = a  and  Q - dP = c."""
        A = np.array([[1.0, self.demand_slope], [1.0, -self.supply_slope]])
        rhs = np.array([self.demand_intercept, self.supply_intercept])
        return A, rhs

    def equilibrium(self) -> tuple[float, float]:
        """Return (quantity, price) where demand equals supply."""
        A, rhs = self.as_linear_system()
        q, p = scipy.linalg.solve(A, rhs)
        return float(q), float(p)

    def quantity_demanded(self, price: float) -> float:
        return self.demand_intercept - self.demand_slope * price

    def quantity_supplied(self, price: float) -> float:
        return self.supply_intercept + self.supply_slope * price

    def excess_demand(self, price: float) -> float:
        """Positive = shortage of seats, negative = empty seats."""
        return self.quantity_demanded(price) - self.quantity_supplied(price)


# =================================================================================
# Forecasters
# =================================================================================
class SeriesForecaster(ABC):
    """One-step-ahead forecaster for a daily series.

    ``fit(series)`` learns from the history and ``predict(horizon)`` forecasts
    the next ``horizon`` days. ``min_history`` is the shortest series the model
    can be fitted on.
    """

    name: str = "Forecaster"
    min_history: int = 1

    def __init__(self) -> None:
        self._history: np.ndarray | None = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def fit(self, series: ArrayLike) -> "SeriesForecaster":
        y = np.asarray(series, dtype=float)
        if y.ndim != 1 or y.size < self.min_history:
            raise ValueError(f"{self.name} needs at least {self.min_history} observations")
        self._history = y
        self._fit(y)
        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        if self._history is None:
            raise RuntimeError(f"{type(self).__name__} must be fitted first")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        return self._predict(horizon)

    def _fit(self, y: np.ndarray) -> None:  # most models need no fitting step
        pass

    @abstractmethod
    def _predict(self, horizon: int) -> np.ndarray: ...


class MovingAverageForecaster(SeriesForecaster):
    """Mean of the last ``window`` days (the original assignment's method)."""

    def __init__(self, window: int = 3) -> None:
        super().__init__()
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.min_history = window
        self.name = f"{window}-day moving average"

    def __repr__(self) -> str:
        return f"MovingAverageForecaster(window={self.window})"

    def _predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._history[-self.window:].mean())


class ExponentialSmoothingForecaster(SeriesForecaster):
    """Simple exponential smoothing: ``level = alpha * y + (1 - alpha) * level``.

    The level starts at the first observation. Forecasts are flat at the final level.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.name = f"SES (alpha={alpha:.2f})"

    def __repr__(self) -> str:
        return f"ExponentialSmoothingForecaster(alpha={self.alpha})"

    def _fit(self, y: np.ndarray) -> None:
        level = y[0]
        for value in y[1:]:
            level = self.alpha * value + (1 - self.alpha) * level
        self.level = level

    def _predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.level)


class LinearTrendForecaster(SeriesForecaster):
    """Least-squares straight line through (day index, value), extrapolated."""

    name = "Linear trend"
    min_history = 2

    def _fit(self, y: np.ndarray) -> None:
        self.slope, self.intercept = np.polyfit(np.arange(y.size), y, deg=1)

    def _predict(self, horizon: int) -> np.ndarray:
        future = np.arange(self._history.size, self._history.size + horizon)
        return self.slope * future + self.intercept


class SeasonalNaiveForecaster(SeriesForecaster):
    """Forecast each day with the value observed one season (``period`` days) earlier."""

    def __init__(self, period: int = 7) -> None:
        super().__init__()
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self.min_history = period
        self.name = f"Seasonal naive (period={period})"

    def __repr__(self) -> str:
        return f"SeasonalNaiveForecaster(period={self.period})"

    def _predict(self, horizon: int) -> np.ndarray:
        last_season = self._history[-self.period:]
        return np.array([last_season[h % self.period] for h in range(horizon)])


# =================================================================================
# Evaluation
# =================================================================================
@dataclass
class BacktestResult:
    """One-step-ahead forecasts from a walk-forward backtest."""

    model_name: str
    target_index: np.ndarray
    actual: np.ndarray
    predicted: np.ndarray

    @property
    def mae(self) -> float:
        return mae(self.actual, self.predicted)


class Backtester:
    """Rolling-origin evaluation.

    For each target index ``t`` from ``start_index`` to the end of the series,
    the model is fitted on ``series[:t]`` and forecasts ``series[t]``.
    With ``start_index=3`` (0-based) the targets are days 4-10 of a 10-day series.
    """

    def __init__(self, start_index: int = 3) -> None:
        if start_index < 1:
            raise ValueError("start_index must be >= 1")
        self.start_index = start_index

    def run(self, series: ArrayLike, model: SeriesForecaster) -> BacktestResult:
        y = np.asarray(series, dtype=float)
        if self.start_index >= y.size:
            raise ValueError("series too short for the chosen start index")
        if self.start_index < model.min_history:
            raise ValueError(f"{model.name} needs {model.min_history} days before the first forecast")
        targets = np.arange(self.start_index, y.size)
        preds = np.array([model.fit(y[:t]).predict(1)[0] for t in targets])
        return BacktestResult(model.name, targets, y[targets], preds)

    def tune_alpha(self, series: ArrayLike, alphas: ArrayLike) -> tuple[float, dict[float, float]]:
        """Grid-search the SES smoothing constant. Returns (best alpha, {alpha: MAE})."""
        scores = {
            float(a): self.run(series, ExponentialSmoothingForecaster(float(a))).mae for a in np.asarray(alphas, dtype=float)
        }
        if not scores:
            raise ValueError("alphas must not be empty")
        best = min(scores, key=scores.get)
        return best, scores


# =================================================================================
# Fleet planning and simulation
# =================================================================================
@dataclass(frozen=True)
class FleetPlanner:
    """Vehicles needed = ceil(passengers x (1 + buffer) / (trips x seats)).

    At least one vehicle is deployed whenever any demand is expected.
    """

    trips_per_vehicle: int = 8
    seats: int = 14
    buffer: float = 0.15

    def __post_init__(self) -> None:
        if self.trips_per_vehicle <= 0 or self.seats <= 0 or self.buffer < 0:
            raise ValueError("trips and seats must be positive and buffer non-negative")

    @property
    def daily_capacity(self) -> int:
        """Passengers one vehicle can carry per day."""
        return self.trips_per_vehicle * self.seats

    def vehicles_needed(self, passengers: float) -> int:
        if passengers < 0:
            raise ValueError("passengers cannot be negative")
        if passengers == 0:
            return 0
        return max(1, math.ceil(passengers * (1 + self.buffer) / self.daily_capacity - 1e-9))

    def load_factor(self, passengers: float) -> float:
        """Share of deployed seats that would be filled (without the buffer)."""
        n = self.vehicles_needed(passengers)
        return passengers / (n * self.daily_capacity) if n else 0.0


WEEKDAY_FACTORS = {"Mon": 1.00, "Tue": 0.95, "Wed": 0.95, "Thu": 1.00, "Fri": 1.35, "Sat": 1.10, "Sun": 0.60}


def simulate_weekly_demand(
    days: int,
    base: float,
    rng: np.random.Generator,
    noise_sd: float = 3.0,
    factors: dict[str, float] | None = None,
) -> np.ndarray:
    """Daily passengers with a weekly pattern, starting on a Monday.

    ``passengers = round(base x factor[weekday] + N(0, noise_sd))``, floored at 0.
    The default factors make Fridays busiest and Sundays quietest.
    """
    if days < 1 or base < 0 or noise_sd < 0:
        raise ValueError("days must be positive; base and noise_sd non-negative")
    pattern = np.array(list((factors or WEEKDAY_FACTORS).values()))
    weekly = pattern[np.arange(days) % pattern.size]
    return np.maximum(0, np.round(base * weekly + rng.normal(0, noise_sd, size=days)))
