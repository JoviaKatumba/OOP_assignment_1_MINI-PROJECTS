"""Mini-Project 1: UBOS District Population Forecaster.

Classes
-------
DistrictPopulation
    A validated time series of a district's population (in thousands).
Forecaster
    Abstract base class exposing ``fit(years, values)`` and ``predict(horizon)``.
LinearTrendForecaster, CAGRForecaster, FibonacciRatioForecaster
    The three concrete forecasting models.
ModelComparison
    Train/test validation of several forecasters on one district.
ClassroomPlanner
    Turns a population change into a count of additional classrooms.
"""

from __future__ import annotations

import copy
import math
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from src.metrics import all_metrics


class DistrictPopulation:
    """Population estimates for one district.

    Parameters
    ----------
    name:
        District name, e.g. ``"Kampala"``.
    years:
        Strictly increasing calendar years.
    populations:
        Population for each year (thousands of people). Must be non-negative.

    Raises
    ------
    ValueError
        If the arrays are empty, have different lengths, contain negative or
        non-finite values, or the years are not strictly increasing.
    """

    def __init__(self, name: str, years: ArrayLike, populations: ArrayLike) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        years_arr = np.asarray(years, dtype=int)
        pop_arr = np.asarray(populations, dtype=float)
        if years_arr.ndim != 1 or pop_arr.ndim != 1:
            raise ValueError("years and populations must be one-dimensional")
        if years_arr.size == 0:
            raise ValueError("at least one observation is required")
        if years_arr.size != pop_arr.size:
            raise ValueError(f"years ({years_arr.size}) and populations ({pop_arr.size}) differ in length")
        if not np.all(np.isfinite(pop_arr)):
            raise ValueError("populations must be finite numbers")
        if np.any(pop_arr < 0):
            raise ValueError("populations cannot be negative")
        if np.any(np.diff(years_arr) <= 0):
            raise ValueError("years must be strictly increasing")
        self.name = name
        self.years = years_arr
        self.populations = pop_arr

    def __repr__(self) -> str:
        return (
            f"DistrictPopulation(name={self.name!r}, years={self.years[0]}-{self.years[-1]}, "
            f"n={len(self)}, last={self.populations[-1]:g}k)"
        )

    def __len__(self) -> int:
        return int(self.years.size)

    # ----- descriptive statistics -------------------------------------------------
    def summary_statistics(self) -> dict[str, float]:
        """Mean, median, *sample* variance and standard deviation via ``statistics``.

        ``statistics.variance`` divides by n - 1 (an unbiased estimate of the
        population variance from a sample).
        """
        values = self.populations.tolist()
        if len(values) < 2:
            raise ValueError("variance needs at least two observations")
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "variance": statistics.variance(values),
            "std": statistics.stdev(values),
        }

    def numpy_statistics(self, ddof: int = 0) -> dict[str, float]:
        """The same statistics via NumPy.

        ``np.var`` defaults to ``ddof=0`` (divide by n). ``ddof=1`` reproduces
        ``statistics.variance``.
        """
        if len(self) - ddof <= 0:
            raise ValueError("ddof must be smaller than the number of observations")
        return {
            "mean": float(np.mean(self.populations)),
            "median": float(np.median(self.populations)),
            "variance": float(np.var(self.populations, ddof=ddof)),
            "std": float(np.std(self.populations, ddof=ddof)),
        }

    # ----- growth -----------------------------------------------------------------
    def growth_rates(self) -> np.ndarray:
        """Year-on-year growth rates as fractions (0.05 means 5 %), length n - 1."""
        if np.any(self.populations[:-1] == 0):
            raise ValueError("growth rate is undefined after a zero population")
        return np.diff(self.populations) / self.populations[:-1]

    def cagr(self) -> float:
        """Compound annual growth rate between the first and last observation."""
        if len(self) < 2:
            raise ValueError("CAGR needs at least two observations")
        if self.populations[0] == 0:
            raise ValueError("CAGR is undefined when the first value is zero")
        periods = self.years[-1] - self.years[0]
        return float((self.populations[-1] / self.populations[0]) ** (1.0 / periods) - 1.0)

    def split(self, last_train_year: int) -> tuple["DistrictPopulation", "DistrictPopulation"]:
        """Split into a training series (<= last_train_year) and a test series (after it)."""
        mask = self.years <= last_train_year
        if mask.all() or not mask.any():
            raise ValueError(f"split year {last_train_year} leaves an empty train or test set")
        train = DistrictPopulation(self.name, self.years[mask], self.populations[mask])
        test = DistrictPopulation(self.name, self.years[~mask], self.populations[~mask])
        return train, test


# =================================================================================
# Forecasting models
# =================================================================================
class Forecaster(ABC):
    """Abstract base class for annual population forecasters.

    Subclasses implement :meth:`_fit` and :meth:`_predict`. The base class
    handles input validation, remembers the training data and offers residual
    bootstrap prediction intervals for every model.
    """

    name: str = "Forecaster"

    def __init__(self) -> None:
        self._years: np.ndarray | None = None
        self._values: np.ndarray | None = None

    def __repr__(self) -> str:
        state = "fitted" if self.is_fitted else "unfitted"
        return f"{type(self).__name__}({state})"

    @property
    def is_fitted(self) -> bool:
        return self._values is not None

    def fit(self, years: ArrayLike, values: ArrayLike) -> "Forecaster":
        """Fit the model to a series and return ``self`` (so calls can be chained)."""
        y = np.asarray(years, dtype=float)
        v = np.asarray(values, dtype=float)
        if y.size < 2 or y.size != v.size:
            raise ValueError("need at least two (year, value) pairs of equal length")
        self._years, self._values = y, v
        self._fit(y, v)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Forecast the next ``horizon`` years after the last training year."""
        self._require_fitted()
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        return self._predict(int(horizon))

    def forecast_years(self, horizon: int) -> np.ndarray:
        """Calendar years that :meth:`predict` refers to."""
        self._require_fitted()
        return self._years[-1] + np.arange(1, horizon + 1)

    @abstractmethod
    def fitted_values(self) -> np.ndarray:
        """In-sample values implied by the model for the training years."""

    @abstractmethod
    def _fit(self, years: np.ndarray, values: np.ndarray) -> None: ...

    @abstractmethod
    def _predict(self, horizon: int) -> np.ndarray: ...

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before use")

    def bootstrap_interval(
        self,
        horizon: int,
        n_boot: int = 1000,
        level: float = 0.95,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Residual-bootstrap prediction interval for :meth:`predict`.

        Each resample builds a synthetic history ``fitted + resampled residuals``,
        refits a copy of the model on it, forecasts, and adds a resampled residual
        to each forecast step (so both parameter and observation noise are
        included). Residuals are centred so they average to zero.

        Returns
        -------
        (lower, upper)
            Arrays of length ``horizon`` holding the interval bounds.
        """
        self._require_fitted()
        if n_boot < 1:
            raise ValueError("n_boot must be positive")
        if not 0 < level < 1:
            raise ValueError("level must be between 0 and 1")
        rng = rng if rng is not None else np.random.default_rng(0)
        fitted = self.fitted_values()
        residuals = self._values - fitted
        residuals = residuals - residuals.mean()
        sims = np.empty((n_boot, horizon))
        for b in range(n_boot):
            synthetic = fitted + rng.choice(residuals, size=residuals.size, replace=True)
            model = copy.deepcopy(self).fit(self._years, synthetic)
            sims[b] = model.predict(horizon) + rng.choice(residuals, size=horizon, replace=True)
        tail = (1 - level) / 2 * 100
        lower, upper = np.percentile(sims, [tail, 100 - tail], axis=0)
        return lower, upper


class LinearTrendForecaster(Forecaster):
    """Straight-line trend ``value = slope * year + intercept`` fitted with ``np.polyfit``."""

    name = "Linear trend"

    def _fit(self, years: np.ndarray, values: np.ndarray) -> None:
        self.slope, self.intercept = np.polyfit(years, values, deg=1)

    def fitted_values(self) -> np.ndarray:
        self._require_fitted()
        return self.slope * self._years + self.intercept

    def _predict(self, horizon: int) -> np.ndarray:
        return self.slope * self.forecast_years(horizon) + self.intercept


class CAGRForecaster(Forecaster):
    """Exponential growth at the training period's compound annual growth rate.

    The growth rate is ``(last / first) ** (1 / years) - 1`` and forecasts
    compound it from the last observed value.
    """

    name = "Exponential (CAGR)"

    def _fit(self, years: np.ndarray, values: np.ndarray) -> None:
        if values[0] <= 0:
            raise ValueError("CAGR model needs a positive first value")
        periods = years[-1] - years[0]
        self.rate = (values[-1] / values[0]) ** (1.0 / periods) - 1.0

    def fitted_values(self) -> np.ndarray:
        self._require_fitted()
        return self._values[0] * (1 + self.rate) ** (self._years - self._years[0])

    def _predict(self, horizon: int) -> np.ndarray:
        return self._values[-1] * (1 + self.rate) ** np.arange(1, horizon + 1)


def fibonacci_ratios(count: int, start: int = 1) -> np.ndarray:
    """Successive ratios F(k+1)/F(k) for k = start, ..., start + count - 1.

    With F(1) = F(2) = 1 the sequence begins 1, 2, 1.5, 1.667, 1.6, ... and
    converges to the golden ratio (about 1.618).
    """
    if count < 0 or start < 1:
        raise ValueError("count must be >= 0 and start >= 1")
    fib = [1, 1]
    while len(fib) < start + count + 1:
        fib.append(fib[-1] + fib[-2])
    # fib[i] holds F(i + 1)
    return np.array([fib[k] / fib[k - 1] for k in range(start, start + count)], dtype=float)


class FibonacciRatioForecaster(Forecaster):
    """The previous cohort's model: scale the last value by successive Fibonacci ratios.

    Step k of the forecast multiplies the previous step by F(start+k)/F(start+k-1).
    There is nothing to estimate from the data, so ``fit`` only stores the series.
    In-sample "fitted" values apply the same rule from the first observation.
    """

    name = "Fibonacci ratio"

    def __init__(self, start: int = 1) -> None:
        super().__init__()
        self.start = start

    def _fit(self, years: np.ndarray, values: np.ndarray) -> None:
        pass  # no parameters to estimate

    def fitted_values(self) -> np.ndarray:
        self._require_fitted()
        ratios = fibonacci_ratios(self._values.size - 1, self.start)
        return self._values[0] * np.concatenate([[1.0], np.cumprod(ratios)])

    def _predict(self, horizon: int) -> np.ndarray:
        return self._values[-1] * np.cumprod(fibonacci_ratios(horizon, self.start))


# =================================================================================
# Validation and planning
# =================================================================================
class ModelComparison:
    """Train each model on years <= ``last_train_year`` and score it on the rest.

    Parameters
    ----------
    district:
        The full series to split.
    models:
        Unfitted forecaster instances (they are fitted in place).
    last_train_year:
        Final year included in the training set.
    """

    def __init__(self, district: DistrictPopulation, models: list[Forecaster], last_train_year: int) -> None:
        if not models:
            raise ValueError("at least one model is required")
        self.district = district
        self.models = models
        self.train, self.test = district.split(last_train_year)
        self.predictions: dict[str, np.ndarray] = {}

    def run(self) -> pd.DataFrame:
        """Return a table of MAE, RMSE and MAPE (rows = models), sorted best first by RMSE."""
        rows = {}
        for model in self.models:
            model.fit(self.train.years, self.train.populations)
            pred = model.predict(len(self.test))
            self.predictions[model.name] = pred
            rows[model.name] = all_metrics(self.test.populations, pred)
        return pd.DataFrame(rows).T.sort_values("RMSE")

    def best_model(self, metric: str = "RMSE") -> Forecaster:
        """The model with the lowest value of ``metric`` on the test years."""
        table = self.run()
        best_name = table[metric].idxmin()
        return next(m for m in self.models if m.name == best_name)


@dataclass(frozen=True)
class ClassroomPlanner:
    """Convert population growth into additional primary classrooms.

    Attributes
    ----------
    school_age_share:
        Fraction of the population that is of primary-school age (0-1].
    pupils_per_classroom:
        Classroom capacity.
    """

    school_age_share: float = 0.18
    pupils_per_classroom: int = 53

    def __post_init__(self) -> None:
        if not 0 < self.school_age_share <= 1:
            raise ValueError("school_age_share must be in (0, 1]")
        if self.pupils_per_classroom <= 0:
            raise ValueError("pupils_per_classroom must be positive")

    def classrooms_for(self, population_thousands: float) -> float:
        """Classrooms needed (not rounded) for a population given in thousands."""
        if population_thousands < 0:
            raise ValueError("population cannot be negative")
        return population_thousands * 1000 * self.school_age_share / self.pupils_per_classroom

    def additional_classrooms(self, current_thousands: float, future_thousands: float) -> int:
        """Extra classrooms needed to keep pace with growth, rounded up (never below 0)."""
        extra = self.classrooms_for(future_thousands) - self.classrooms_for(current_thousands)
        return max(0, math.ceil(extra - 1e-9))
