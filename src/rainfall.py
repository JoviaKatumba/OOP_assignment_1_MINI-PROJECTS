"""Mini-Project 4: Rainfall Pattern & Crop Suitability Analyser.

Classes
-------
Region
    Twelve months of rainfall for one place, with summary statistics and
    automatic rainy-season detection (``scipy.signal.find_peaks``).
CropRule
    A crop's suitable monthly rainfall range and a month classifier.
RegionComparator
    Pairwise similarity / distance matrices between regions.

Functions
---------
cosine_similarity, pearson_correlation, euclidean_distance, load_nasa_power_csv
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.signal import find_peaks

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


# =================================================================================
# Similarity measures
# =================================================================================
def _as_vector_pair(a: ArrayLike, b: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    u, v = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    if u.size == 0 or u.shape != v.shape:
        raise ValueError("vectors must be non-empty and the same length")
    return u, v


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """Cosine of the angle between two vectors: (a . b) / (|a| |b|).

    This replaces last year's ``math.cos(...)``, which returns the cosine of a
    given angle and cannot measure how similar two vectors are.

    Raises
    ------
    ValueError
        If either vector has zero length, since the angle is then undefined.
    """
    u, v = _as_vector_pair(a, b)
    norms = np.linalg.norm(u) * np.linalg.norm(v)
    if norms == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float(np.dot(u, v) / norms)


def pearson_correlation(a: ArrayLike, b: ArrayLike) -> float:
    """Pearson correlation: cosine similarity of the mean-centred vectors."""
    u, v = _as_vector_pair(a, b)
    return cosine_similarity(u - u.mean(), v - v.mean())


def euclidean_distance(a: ArrayLike, b: ArrayLike) -> float:
    """Straight-line distance between two vectors (same units as the data, mm)."""
    u, v = _as_vector_pair(a, b)
    return float(np.linalg.norm(u - v))


# =================================================================================
# Region
# =================================================================================
class Region:
    """Monthly rainfall climatology for one region.

    Parameters
    ----------
    name:
        Region name.
    rainfall:
        Twelve non-negative monthly totals (mm), January to December.
    """

    def __init__(self, name: str, rainfall: ArrayLike) -> None:
        values = np.asarray(rainfall, dtype=float)
        if not name:
            raise ValueError("name must be non-empty")
        if values.shape != (12,):
            raise ValueError(f"rainfall must have 12 monthly values, got shape {values.shape}")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("rainfall must be finite and non-negative")
        self.name = name
        self.rainfall = values

    def __repr__(self) -> str:
        return f"Region({self.name!r}, annual={self.annual_total():.0f} mm, {self.modality()})"

    def __len__(self) -> int:
        return 12

    # ----- summary statistics ---------------------------------------------------
    def annual_total(self) -> float:
        return float(self.rainfall.sum())

    def mean(self) -> float:
        return float(self.rainfall.mean())

    def wettest_month(self) -> tuple[str, float]:
        i = int(np.argmax(self.rainfall))
        return MONTHS[i], float(self.rainfall[i])

    def driest_month(self) -> tuple[str, float]:
        i = int(np.argmin(self.rainfall))
        return MONTHS[i], float(self.rainfall[i])

    def coefficient_of_variation(self) -> float:
        """Sample SD / mean of the monthly totals (how uneven the year is)."""
        mean = statistics.mean(self.rainfall.tolist())
        if mean == 0:
            raise ValueError("CV is undefined for a region with no rain")
        return statistics.stdev(self.rainfall.tolist()) / mean

    # ----- rainy seasons -----------------------------------------------------------
    def rainy_season_peaks(self, min_prominence: float = 5.0, dry_threshold: float | None = None) -> list[int]:
        """Month indices (0 = Jan) at the centre of each distinct rainy season.

        Method:

        1. The year is circular (December is followed by January), so the series
           is padded with six months on each side before calling
           ``scipy.signal.find_peaks``. Peaks are then mapped back to 0-11.
           This lets a season peak in December or January be detected.
        2. ``find_peaks`` with a small ``min_prominence`` finds *candidate* peaks.
        3. Two neighbouring peaks count as separate seasons only if the rainfall
           between them drops below ``dry_threshold`` (default: the monthly mean),
           i.e. there is a genuine drier spell. Otherwise they are merged into one
           long season and the higher peak is kept.
        """
        x = self.rainfall
        threshold = self.mean() if dry_threshold is None else dry_threshold
        pad = 6
        padded = np.concatenate([x[-pad:], x, x[:pad]])
        idx, _ = find_peaks(padded, prominence=min_prominence)
        peaks = sorted({int(p - pad) % 12 for p in idx if pad <= p < pad + 12})

        def trough(a: int, b: int) -> float:
            """Lowest rainfall strictly between month a and month b going forward."""
            months = [(a + k) % 12 for k in range(1, (b - a) % 12 or 12)]
            return float(min(x[m] for m in months)) if months else float(x[a])

        merged = True
        while merged and len(peaks) > 1:
            merged = False
            for i in range(len(peaks)):
                a, b = peaks[i], peaks[(i + 1) % len(peaks)]
                if trough(a, b) >= threshold:
                    keep = a if x[a] >= x[b] else b
                    peaks = sorted(set(peaks) - {a, b} | {keep})
                    merged = True
                    break
        return peaks

    def modality(self, **kwargs: float) -> str:
        """``"unimodal"``, ``"bimodal"`` or ``"multimodal"`` from :meth:`rainy_season_peaks`."""
        n = len(self.rainy_season_peaks(**kwargs))
        return {0: "no clear season", 1: "unimodal", 2: "bimodal"}.get(n, "multimodal")


# =================================================================================
# Crops
# =================================================================================
@dataclass(frozen=True)
class CropRule:
    """Suitable monthly rainfall window for a crop.

    Attributes
    ----------
    name:
        Crop name, e.g. ``"maize"``.
    min_mm, max_mm:
        Monthly rainfall range (mm) considered suitable during the growing season.
    source:
        Citation for the thresholds.
    """

    name: str
    min_mm: float
    max_mm: float
    source: str = ""

    def __post_init__(self) -> None:
        if not 0 <= self.min_mm < self.max_mm:
            raise ValueError("need 0 <= min_mm < max_mm")

    def code(self, rainfall_mm: float) -> int:
        """-1 = too dry, 0 = suitable, +1 = too wet."""
        if rainfall_mm < 0:
            raise ValueError("rainfall cannot be negative")
        if rainfall_mm < self.min_mm:
            return -1
        if rainfall_mm > self.max_mm:
            return 1
        return 0

    def classify(self, rainfall_mm: float) -> str:
        return {-1: "Drought risk", 0: f"Good for {self.name}", 1: "Waterlogging risk"}[self.code(rainfall_mm)]

    def classify_region(self, region: Region) -> list[str]:
        return [self.classify(mm) for mm in region.rainfall]


# =================================================================================
# Region comparison
# =================================================================================
class RegionComparator:
    """Pairwise similarity and distance matrices for a list of regions."""

    MEASURES: dict[str, Callable[[ArrayLike, ArrayLike], float]] = {
        "cosine similarity": cosine_similarity,
        "Pearson correlation": pearson_correlation,
        "Euclidean distance (mm)": euclidean_distance,
    }

    def __init__(self, regions: Sequence[Region]) -> None:
        if len(regions) < 2:
            raise ValueError("need at least two regions to compare")
        self.regions = list(regions)

    def matrix(self, measure: str) -> pd.DataFrame:
        """Symmetric DataFrame of ``measure`` for every pair of regions."""
        fn = self.MEASURES[measure]
        names = [r.name for r in self.regions]
        values = [[fn(a.rainfall, b.rainfall) for b in self.regions] for a in self.regions]
        return pd.DataFrame(values, index=names, columns=names)

    def all_matrices(self) -> dict[str, pd.DataFrame]:
        return {name: self.matrix(name) for name in self.MEASURES}


# =================================================================================
# Real data
# =================================================================================
def load_nasa_power_csv(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load the CSV written by ``scripts/fetch_nasa_power.py``.

    Returns a mapping region -> DataFrame (index = year, columns = Jan..Dec, mm).
    """
    rows: list[dict[str, float | str | int]] = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({"region": row["region"], "year": int(row["year"]),
                         "month": int(row["month"]), "mm": float(row["rainfall_mm"])})
    if not rows:
        raise ValueError("no rainfall rows found")
    df = pd.DataFrame(rows)
    out = {}
    for region, grp in df.groupby("region"):
        table = grp.pivot(index="year", columns="month", values="mm").sort_index()
        table.columns = [MONTHS[m - 1] for m in table.columns]
        out[str(region)] = table
    return out
