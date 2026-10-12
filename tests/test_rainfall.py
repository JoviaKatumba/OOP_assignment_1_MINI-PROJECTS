"""Tests for Mini-Project 4 (src/rainfall.py)."""

import numpy as np
import pytest
from scipy.spatial.distance import cosine as scipy_cosine

from src.rainfall import (
    CropRule,
    Region,
    RegionComparator,
    cosine_similarity,
    euclidean_distance,
    load_nasa_power_csv,
    pearson_correlation,
)

KAMPALA = [120, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130]
GULU = [8, 25, 75, 160, 190, 145, 170, 215, 175, 150, 60, 15]
MBARARA = [70, 85, 120, 140, 90, 25, 20, 55, 100, 125, 120, 90]


def test_region_summary_statistics():
    r = Region("Kampala", KAMPALA)
    assert r.annual_total() == 1600
    assert r.mean() == pytest.approx(1600 / 12)
    assert r.wettest_month() == ("May", 220)
    assert r.driest_month() == ("Sep", 60)
    assert r.coefficient_of_variation() == pytest.approx(np.std(KAMPALA, ddof=1) / np.mean(KAMPALA))


@pytest.mark.parametrize("bad", [KAMPALA[:11], [-1] + KAMPALA[1:], []])
def test_region_rejects_invalid_rainfall(bad):
    with pytest.raises(ValueError):
        Region("Bad", bad)


def test_cv_undefined_for_zero_rain():
    with pytest.raises(ValueError):
        Region("Desert", [0] * 12).coefficient_of_variation()


def test_cosine_similarity_matches_scipy_and_hand_value():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)  # scale-invariant
    assert cosine_similarity(KAMPALA, GULU) == pytest.approx(1 - scipy_cosine(KAMPALA, GULU))


def test_cosine_similarity_zero_vector_raises():
    with pytest.raises(ValueError):
        cosine_similarity([0, 0, 0], [1, 2, 3])


def test_pearson_and_euclidean():
    assert pearson_correlation(KAMPALA, GULU) == pytest.approx(np.corrcoef(KAMPALA, GULU)[0, 1])
    assert euclidean_distance([0, 0], [3, 4]) == pytest.approx(5.0)


def test_comparator_matrices_are_symmetric():
    comp = RegionComparator([Region("K", KAMPALA), Region("G", GULU), Region("M", MBARARA)])
    for m in comp.all_matrices().values():
        np.testing.assert_allclose(m.to_numpy(), m.to_numpy().T)
    with pytest.raises(ValueError):
        RegionComparator([Region("K", KAMPALA)])


@pytest.mark.parametrize(
    "rain, expected",
    [(KAMPALA, "bimodal"), (GULU, "unimodal"), (MBARARA, "bimodal")],
)
def test_modality_matches_known_climate_zones(rain, expected):
    assert Region("x", rain).modality() == expected


def test_december_peak_detected_on_circular_year():
    # Single season peaking in December: needs wrap-around to be found
    rain = [150, 100, 50, 30, 20, 20, 20, 20, 30, 60, 120, 200]
    assert Region("x", rain).rainy_season_peaks() == [11]


def test_crop_rule_classification_and_edges():
    maize = CropRule("maize", 100, 200)
    assert maize.classify(50) == "Drought risk"
    assert maize.classify(100) == "Good for maize"  # boundary is inclusive
    assert maize.classify(250) == "Waterlogging risk"
    with pytest.raises(ValueError):
        CropRule("bad", 200, 100)
    with pytest.raises(ValueError):
        maize.classify(-5)


def test_load_nasa_power_csv(tmp_path):
    path = tmp_path / "rain.csv"
    lines = ["region,lat,lon,year,month,rainfall_mm"]
    lines += [f"Test,0,0,2020,{m},{m * 10}" for m in range(1, 13)]
    path.write_text("\n".join(lines), encoding="utf-8")
    table = load_nasa_power_csv(path)["Test"]
    assert table.shape == (1, 12)
    assert table.loc[2020, "Dec"] == 120
