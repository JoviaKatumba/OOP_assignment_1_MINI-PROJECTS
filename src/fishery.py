"""Mini-Project 3: Lake Victoria Fish Stock & Export Risk Model (Jinja cooperative).

Classes
-------
FishStock
    Discrete logistic growth with proportional harvesting and optional closed season.
StockTrajectory
    Result of a stock simulation (stock, harvest, and helpers).
PriceModel
    Seeded, bounded random walk for the weekly fish price (UGX/kg).
RiskAssessor
    Descriptive revenue statistics, CV-based risk classes and Value-at-Risk.

Functions
---------
fibonacci, weekly_revenue
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike

WEEKS_PER_YEAR = 52
KG_PER_TONNE = 1000


def fibonacci(n: int) -> list[int]:
    """The first ``n`` Fibonacci numbers, starting 1, 1, 2, 3, ..."""
    if n < 0:
        raise ValueError("n must be non-negative")
    seq: list[int] = []
    a, b = 1, 1
    for _ in range(n):
        seq.append(a)
        a, b = b, a + b
    return seq


@dataclass
class StockTrajectory:
    """Output of :meth:`FishStock.simulate`.

    Attributes
    ----------
    stock:
        Biomass at the start of each week, length ``weeks + 1`` (tonnes).
    harvest:
        Catch taken during each week, length ``weeks`` (tonnes).
    h:
        Harvest rate used.
    """

    stock: np.ndarray
    harvest: np.ndarray
    h: float

    @property
    def final_stock(self) -> float:
        return float(self.stock[-1])

    @property
    def total_harvest(self) -> float:
        return float(self.harvest.sum())

    def __len__(self) -> int:
        return int(self.harvest.size)


class FishStock:
    """Logistic fish stock with proportional harvesting.

    ``N(t+1) = N(t) + r N(t) (1 - N(t)/K) - h N(t)``, with ``h = 0`` during
    closed-season weeks and the stock floored at zero.

    Parameters
    ----------
    r:
        Intrinsic growth rate per week.
    K:
        Carrying capacity (tonnes).
    N0:
        Initial stock (tonnes).
    h:
        Fraction of the stock harvested each open week (0 <= h < 1).
    closed_weeks:
        Week-of-year numbers (0-51) with no harvesting. They repeat every year.
    """

    def __init__(
        self,
        r: float = 0.4,
        K: float = 10_000,
        N0: float = 4_000,
        h: float = 0.1,
        closed_weeks: Iterable[int] = (),
    ) -> None:
        if r <= 0 or K <= 0:
            raise ValueError("r and K must be positive")
        if not 0 <= N0 <= K * 1.5:
            raise ValueError("N0 must be non-negative and not wildly above K")
        if not 0 <= h < 1:
            raise ValueError("h must be in [0, 1)")
        closed = frozenset(int(w) for w in closed_weeks)
        if any(not 0 <= w < WEEKS_PER_YEAR for w in closed):
            raise ValueError("closed weeks must be in 0..51")
        self.r, self.K, self.N0, self.h = float(r), float(K), float(N0), float(h)
        self.closed_weeks = closed

    def __repr__(self) -> str:
        closed = f", closed_weeks={len(self.closed_weeks)}" if self.closed_weeks else ""
        return f"FishStock(r={self.r}, K={self.K:g}, N0={self.N0:g}, h={self.h}{closed})"

    # ----- theory -----------------------------------------------------------------
    @property
    def msy(self) -> float:
        """Maximum sustainable yield rK/4 (tonnes per week), reached at h = r/2."""
        return self.r * self.K / 4

    @property
    def msy_rate(self) -> float:
        """Harvest rate that achieves MSY in the long run."""
        return self.r / 2

    @property
    def equilibrium_stock(self) -> float:
        """Long-run stock K(1 - h/r) under constant harvesting (0 if h >= r)."""
        return max(0.0, self.K * (1 - self.h / self.r))

    @property
    def equilibrium_yield(self) -> float:
        """Long-run weekly catch h * equilibrium_stock (tonnes per week)."""
        return self.h * self.equilibrium_stock

    # ----- simulation -----------------------------------------------------------------
    def harvest_rate(self, week: int) -> float:
        """Harvest rate in force during ``week`` (counted from 0)."""
        return 0.0 if week % WEEKS_PER_YEAR in self.closed_weeks else self.h

    def simulate(self, weeks: int = WEEKS_PER_YEAR) -> StockTrajectory:
        """Run the model forward for ``weeks`` weeks."""
        if weeks < 1:
            raise ValueError("weeks must be positive")
        stock = np.empty(weeks + 1)
        harvest = np.empty(weeks)
        stock[0] = self.N0
        for t in range(weeks):
            n = stock[t]
            catch = self.harvest_rate(t) * n
            growth = self.r * n * (1 - n / self.K)
            stock[t + 1] = max(0.0, n + growth - catch)
            harvest[t] = catch
        return StockTrajectory(stock=stock, harvest=harvest, h=self.h)


class PriceModel:
    """Weekly fish price (UGX/kg) as a seeded random walk clipped to [lower, upper].

    ``P(t+1) = clip(P(t) + drift + eps,  lower, upper)`` with ``eps ~ N(0, step_sd)``.
    Clipping stands in for a market floor (export contracts, processing-plant
    minimum price) and a ceiling (buyers switch to substitutes such as tilapia
    from farms).
    """

    def __init__(
        self,
        start: float = 12_000,
        step_sd: float = 400,
        lower: float = 9_000,
        upper: float = 16_000,
        drift: float = 0.0,
    ) -> None:
        if not lower < upper:
            raise ValueError("lower bound must be below upper bound")
        if not lower <= start <= upper:
            raise ValueError("start price must lie within the bounds")
        if step_sd < 0:
            raise ValueError("step_sd cannot be negative")
        self.start, self.step_sd, self.lower, self.upper, self.drift = start, step_sd, lower, upper, drift

    def __repr__(self) -> str:
        return f"PriceModel(start={self.start:g}, sd={self.step_sd:g}, bounds=[{self.lower:g}, {self.upper:g}])"

    def simulate_paths(self, n_paths: int, weeks: int, rng: np.random.Generator) -> np.ndarray:
        """Simulate ``n_paths`` price paths. Returns (n_paths, weeks); week 0 = start price."""
        if n_paths < 1 or weeks < 1:
            raise ValueError("n_paths and weeks must be positive")
        shocks = rng.normal(self.drift, self.step_sd, size=(n_paths, weeks - 1))
        prices = np.empty((n_paths, weeks))
        prices[:, 0] = self.start
        for t in range(1, weeks):
            prices[:, t] = np.clip(prices[:, t - 1] + shocks[:, t - 1], self.lower, self.upper)
        return prices

    def simulate(self, weeks: int, rng: np.random.Generator) -> np.ndarray:
        """One price path of length ``weeks``."""
        return self.simulate_paths(1, weeks, rng)[0]


def weekly_revenue(harvest_tonnes: ArrayLike, price_ugx_per_kg: ArrayLike) -> np.ndarray:
    """Revenue (UGX) = harvest (kg) x price (UGX/kg). Broadcasts over price paths."""
    harvest = np.asarray(harvest_tonnes, dtype=float)
    price = np.asarray(price_ugx_per_kg, dtype=float)
    if harvest.shape[-1] != price.shape[-1]:
        raise ValueError("harvest and price must cover the same number of weeks")
    if np.any(harvest < 0) or np.any(price < 0):
        raise ValueError("harvest and price must be non-negative")
    return harvest * KG_PER_TONNE * price


class RiskAssessor:
    """Classify revenue risk by the coefficient of variation and compute VaR.

    The CV (standard deviation / mean) is dimensionless, so the same thresholds
    work whatever the currency or the scale of the cooperative. That is not true
    of a raw variance threshold.

    Parameters
    ----------
    low_threshold, high_threshold:
        CV < low is "Low" risk, CV >= high is "High", anything else "Moderate".
    var_level:
        Tail probability for Value-at-Risk (0.05 = 5% VaR).
    """

    def __init__(self, low_threshold: float = 0.10, high_threshold: float = 0.25, var_level: float = 0.05) -> None:
        if not 0 < low_threshold < high_threshold:
            raise ValueError("need 0 < low_threshold < high_threshold")
        if not 0 < var_level < 1:
            raise ValueError("var_level must be in (0, 1)")
        self.low_threshold, self.high_threshold, self.var_level = low_threshold, high_threshold, var_level

    def __repr__(self) -> str:
        return f"RiskAssessor(low<{self.low_threshold}, high>={self.high_threshold}, VaR level={self.var_level})"

    @staticmethod
    def describe(values: Sequence[float]) -> dict[str, float]:
        """Mean, median, sample variance, standard deviation and CV via ``statistics``."""
        data = [float(v) for v in values]
        if len(data) < 2:
            raise ValueError("need at least two values")
        mean = statistics.mean(data)
        if mean == 0:
            raise ValueError("CV is undefined when the mean is zero")
        std = statistics.stdev(data)
        return {
            "mean": mean,
            "median": statistics.median(data),
            "variance": statistics.variance(data),
            "std": std,
            "cv": std / mean,
        }

    def classify_cv(self, cv: float) -> str:
        if cv < 0:
            raise ValueError("CV cannot be negative")
        if cv < self.low_threshold:
            return "Low"
        if cv < self.high_threshold:
            return "Moderate"
        return "High"

    def classify(self, values: Sequence[float]) -> str:
        """Risk class of a revenue series."""
        return self.classify_cv(self.describe(values)["cv"])

    def value_at_risk(self, outcomes: ArrayLike) -> dict[str, float]:
        """Value-at-Risk of simulated outcomes (e.g. annual revenue).

        Returns the ``var_level`` quantile (the revenue exceeded in 95% of
        scenarios) and the VaR expressed as a shortfall from the mean.
        """
        x = np.asarray(outcomes, dtype=float)
        if x.size < 2:
            raise ValueError("need at least two simulated outcomes")
        q = float(np.quantile(x, self.var_level))
        return {"quantile": q, "mean": float(x.mean()), "var_shortfall": float(x.mean() - q)}
