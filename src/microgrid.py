"""Mini-Project 2: Solar Micro-Grid Dispatch Planner (Kasese health centre).

The daily energy drawn from each source must satisfy a square linear system
``A @ x = d``. For the base grid (solar ``x``, battery ``y``)::

    3x + 2y = D1   (daytime load, kWh)
    4x +  y = D2   (critical-equipment load, kWh)

Classes
-------
MicroGrid
    Holds the coefficient matrix and unit costs. Checks the system is well-posed,
    solves one day or many days, flags infeasible days and repairs them with NNLS.
HybridMicroGrid
    Subclass adding a diesel generator and a third constraint (3x3 system).
DispatchResult
    Container for a multi-day solution with feasibility flags and costs.

Functions
---------
read_non_negative_float, prompt_demands
    Robust interactive input (the input function is injectable for testing).
generate_demand_csv, load_demand_csv
    Create / read a seeded 30-day demand file.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike
from scipy.optimize import nnls


class SingularSystemError(ValueError):
    """Raised when the coefficient matrix has no unique solution."""


@dataclass
class DispatchResult:
    """Solution of the dispatch problem over several days.

    Attributes
    ----------
    demands:
        (days, n_constraints) array of right-hand sides.
    exact:
        (days, n_sources) exact solution of ``A x = d`` (may contain negatives).
    dispatch:
        (days, n_sources) physically usable dispatch (non-negative).
    infeasible:
        Boolean mask of days whose exact solution had a negative component.
    unmet:
        (days, n_constraints) residual ``d - A @ dispatch`` (kWh); zero on feasible days.
    daily_cost:
        Cost in UGX of ``dispatch`` for each day.
    """

    demands: np.ndarray
    exact: np.ndarray
    dispatch: np.ndarray
    infeasible: np.ndarray
    unmet: np.ndarray
    daily_cost: np.ndarray
    sources: tuple[str, ...] = field(default=("solar", "battery"))

    @property
    def monthly_cost(self) -> float:
        """Total cost over all days in the result (UGX)."""
        return float(self.daily_cost.sum())

    @property
    def n_infeasible(self) -> int:
        return int(self.infeasible.sum())


class MicroGrid:
    """A micro-grid whose daily dispatch solves a square linear system.

    Parameters
    ----------
    coefficients:
        Square matrix ``A``; row i gives each source's contribution to constraint i.
    unit_costs:
        Cost of each source in UGX/kWh (same order as the columns of ``A``).
    sources:
        Names of the sources (columns).
    cond_warning:
        Condition number above which :meth:`is_well_conditioned` returns False.

    Raises
    ------
    SingularSystemError
        If ``A`` is singular (rank-deficient), so a unique dispatch does not exist.
    """

    DEFAULT_A = np.array([[3.0, 2.0], [4.0, 1.0]])
    DEFAULT_COSTS = (150.0, 450.0)  # UGX/kWh, illustrative
    DEFAULT_SOURCES = ("solar", "battery")

    def __init__(
        self,
        coefficients: ArrayLike | None = None,
        unit_costs: Sequence[float] | None = None,
        sources: Sequence[str] | None = None,
        cond_warning: float = 1e3,
    ) -> None:
        A = np.array(self.DEFAULT_A if coefficients is None else coefficients, dtype=float)
        costs = np.array(self.DEFAULT_COSTS if unit_costs is None else unit_costs, dtype=float)
        names = tuple(self.DEFAULT_SOURCES if sources is None else sources)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"coefficient matrix must be square, got shape {A.shape}")
        if costs.shape != (A.shape[1],) or len(names) != A.shape[1]:
            raise ValueError("need one unit cost and one source name per column of A")
        if np.any(costs < 0):
            raise ValueError("unit costs cannot be negative")
        if np.linalg.matrix_rank(A) < A.shape[0]:
            raise SingularSystemError(
                "coefficient matrix is singular: at least one constraint is a linear "
                "combination of the others, so there is no unique dispatch"
            )
        self.A = A
        self.unit_costs = costs
        self.sources = names
        self.cond_warning = cond_warning

    def __repr__(self) -> str:
        return f"{type(self).__name__}(sources={self.sources}, det={self.determinant:.3g}, cond={self.condition_number:.3g})"

    @property
    def size(self) -> int:
        """Number of sources (= number of constraints)."""
        return self.A.shape[0]

    # ----- well-posedness ------------------------------------------------------
    @property
    def determinant(self) -> float:
        return float(np.linalg.det(self.A))

    @property
    def condition_number(self) -> float:
        """2-norm condition number: worst-case amplification of relative input error."""
        return float(np.linalg.cond(self.A))

    def is_well_conditioned(self) -> bool:
        return self.condition_number < self.cond_warning

    # ----- solving ---------------------------------------------------------------
    def _validate_demands(self, demands: ArrayLike) -> np.ndarray:
        d = np.asarray(demands, dtype=float)
        if d.ndim == 1:
            d = d[np.newaxis, :]
        if d.ndim != 2 or d.shape[1] != self.size or d.shape[0] == 0:
            raise ValueError(f"demands must have shape (days, {self.size})")
        if not np.all(np.isfinite(d)) or np.any(d < 0):
            raise ValueError("demands must be finite and non-negative")
        return d

    def solve_day(self, *demands: float) -> np.ndarray:
        """Exact solution for one day, e.g. ``grid.solve_day(d1, d2)``.

        The result may contain negative values; see :meth:`dispatch`.
        """
        d = self._validate_demands(demands)[0]
        return scipy.linalg.solve(self.A, d)

    def solve_loop(self, demands: ArrayLike) -> np.ndarray:
        """Solve each day separately in a Python loop. Returns (days, n_sources)."""
        d = self._validate_demands(demands)
        return np.array([scipy.linalg.solve(self.A, row) for row in d])

    def solve_vectorised(self, demands: ArrayLike) -> np.ndarray:
        """Solve all days in one call with an (n, days) right-hand side."""
        d = self._validate_demands(demands)
        return scipy.linalg.solve(self.A, d.T).T

    def solve_nonnegative(self, *demands: float) -> tuple[np.ndarray, float]:
        """Closest non-negative dispatch via ``scipy.optimize.nnls``.

        Returns the dispatch and the residual norm ``||A x - d||`` (kWh).
        """
        d = self._validate_demands(demands)[0]
        x, residual = nnls(self.A, d)
        return x, float(residual)

    def dispatch(self, demands: ArrayLike, tol: float = 1e-9) -> DispatchResult:
        """Solve many days, flag infeasible ones and repair them with NNLS.

        Strategy: a day is infeasible if the exact solution needs a negative
        amount from any source (physically, the battery would have to *absorb*
        energy while still meeting the load). For those days NNLS finds the
        non-negative dispatch that is closest to meeting both constraints, and
        the remaining shortfall is reported in ``unmet`` so an operator can see
        how much load would have to be shed or covered from elsewhere.
        """
        d = self._validate_demands(demands)
        exact = self.solve_vectorised(d)
        infeasible = np.any(exact < -tol, axis=1)
        usable = exact.copy()
        for i in np.flatnonzero(infeasible):
            usable[i] = nnls(self.A, d[i])[0]
        usable = np.clip(usable, 0, None)  # tidy tiny negative round-off on feasible days
        unmet = d - usable @ self.A.T
        return DispatchResult(
            demands=d,
            exact=exact,
            dispatch=usable,
            infeasible=infeasible,
            unmet=unmet,
            daily_cost=self.daily_cost(usable),
            sources=self.sources,
        )

    # ----- costs and sensitivity ---------------------------------------------------
    def daily_cost(self, dispatch: ArrayLike) -> np.ndarray:
        """Cost in UGX for each row of a (days, n_sources) dispatch array."""
        x = np.atleast_2d(np.asarray(dispatch, dtype=float))
        if x.shape[1] != self.size:
            raise ValueError(f"dispatch must have {self.size} columns")
        return x @ self.unit_costs

    def sensitivity(
        self,
        demands: Sequence[float],
        n_draws: int = 1000,
        pct: float = 0.05,
        rng: np.random.Generator | None = None,
    ) -> dict[str, np.ndarray | float]:
        """Monte Carlo sensitivity of the exact solution to +-``pct`` demand noise.

        Each demand is multiplied independently by ``U(1 - pct, 1 + pct)``.

        Returns
        -------
        dict with ``solutions`` (n_draws, n), ``baseline`` (n,),
        ``amplification`` (relative solution change / relative demand change for
        each draw) and ``condition_number`` (the theoretical upper bound).
        """
        if n_draws < 1 or not 0 < pct < 1:
            raise ValueError("n_draws must be >= 1 and pct in (0, 1)")
        rng = rng if rng is not None else np.random.default_rng(0)
        base_d = self._validate_demands(demands)[0]
        base_x = scipy.linalg.solve(self.A, base_d)
        factors = rng.uniform(1 - pct, 1 + pct, size=(n_draws, self.size))
        perturbed = base_d * factors
        sols = scipy.linalg.solve(self.A, perturbed.T).T
        rel_in = np.linalg.norm(perturbed - base_d, axis=1) / np.linalg.norm(base_d)
        rel_out = np.linalg.norm(sols - base_x, axis=1) / np.linalg.norm(base_x)
        return {
            "solutions": sols,
            "baseline": base_x,
            "amplification": rel_out / rel_in,
            "condition_number": self.condition_number,
        }


class HybridMicroGrid(MicroGrid):
    """Micro-grid with a diesel generator ``z`` and a third (evening) constraint.

    Default system (illustrative)::

        3x + 2y +  z = D1   daytime load
        4x +  y + 2z = D2   critical-equipment load
         x +  y +  z = D3   total energy drawn over the day

    Diesel is assumed to cost UGX 1,200/kWh (fuel plus maintenance, illustrative).
    """

    DEFAULT_A = np.array([[3.0, 2.0, 1.0], [4.0, 1.0, 2.0], [1.0, 1.0, 1.0]])
    DEFAULT_COSTS = (150.0, 450.0, 1200.0)
    DEFAULT_SOURCES = ("solar", "battery", "diesel")


# =================================================================================
# Input modes
# =================================================================================
def read_non_negative_float(
    prompt: str,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    max_attempts: int | None = None,
) -> float:
    """Ask until the user types a finite, non-negative number.

    Empty strings, non-numeric text, negative numbers, ``nan`` and ``inf`` are
    rejected with a message and the user is asked again.

    Parameters
    ----------
    input_fn, output_fn:
        Injectable for testing; default to the built-in ``input`` and ``print``.
    max_attempts:
        Optional cap; ``ValueError`` is raised once it is exceeded.
    """
    attempts = 0
    while max_attempts is None or attempts < max_attempts:
        attempts += 1
        raw = input_fn(prompt).strip()
        if not raw:
            output_fn("  Please enter a value (input was empty).")
            continue
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            output_fn(f"  '{raw}' is not a number. Try again.")
            continue
        if not math.isfinite(value):
            output_fn("  Value must be a finite number.")
            continue
        if value < 0:
            output_fn("  Demand cannot be negative.")
            continue
        return value
    raise ValueError(f"no valid input after {max_attempts} attempts")


def prompt_demands(
    grid: MicroGrid,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> np.ndarray:
    """Interactively read one day's demands (D1, D2, ...) for ``grid``."""
    return np.array(
        [read_non_negative_float(f"Enter D{i + 1} (kWh): ", input_fn, output_fn) for i in range(grid.size)]
    )


# =================================================================================
# CSV mode
# =================================================================================
CSV_FIELDS = ("day", "date", "weekday", "D1_kWh", "D2_kWh")


def generate_demand_csv(
    path: str | Path,
    days: int = 30,
    seed: int = 42,
    start: dt.date = dt.date(2026, 6, 1),
) -> Path:
    """Write ``days`` of synthetic demand with a weekly pattern and noise.

    Assumptions (illustrative, for a rural health centre):

    * Daytime load D1 is about 120 kWh on weekdays (outpatient clinics, lab)
      and about 95 kWh at weekends.
    * Critical load D2 (vaccine fridges, theatre, oxygen) is about 140 kWh on
      weekdays and about 120 kWh at weekends.
    * Gaussian noise with SD 6 kWh (D1) and 8 kWh (D2) is added.
    """
    if days < 1:
        raise ValueError("days must be positive")
    rng = np.random.default_rng(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_FIELDS)
        for i in range(days):
            date = start + dt.timedelta(days=i)
            weekend = date.weekday() >= 5
            d1 = (95 if weekend else 120) + rng.normal(0, 6)
            d2 = (120 if weekend else 140) + rng.normal(0, 8)
            writer.writerow([i + 1, date.isoformat(), date.strftime("%a"), f"{max(d1, 0):.2f}", f"{max(d2, 0):.2f}"])
    return path


def load_demand_csv(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Read a demand CSV. Returns a (days, 2) array and the list of dates.

    Raises
    ------
    ValueError
        If the file is empty, a column is missing, or a value is non-numeric or negative.
    """
    demands, dates = [], []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = {"D1_kWh", "D2_kWh"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")
        for line_no, row in enumerate(reader, start=2):
            try:
                d1, d2 = float(row["D1_kWh"]), float(row["D2_kWh"])
            except (TypeError, ValueError) as err:
                raise ValueError(f"line {line_no}: non-numeric demand") from err
            if d1 < 0 or d2 < 0:
                raise ValueError(f"line {line_no}: negative demand")
            demands.append((d1, d2))
            dates.append(row.get("date", str(line_no - 1)))
    if not demands:
        raise ValueError("CSV contains no data rows")
    return np.array(demands), dates
