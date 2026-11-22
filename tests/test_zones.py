import numpy as np
import pytest

from cropmon.analysis.zones import health_class, health_score, summarise_map, zone_map


def test_zone_map_is_regular_tiling():
    zones = zone_map((32, 32), 4, 4)
    assert zones.shape == (32, 32)
    assert zones.min() == 0
    assert zones.max() == 15
    assert len(np.unique(zones)) == 16
    assert np.all(zones[:8, :8] == 0)


def test_zone_map_rejects_bad_grid():
    with pytest.raises(ValueError):
        zone_map((8, 8), 0, 4)


def test_summarise_map_matches_manual_mean():
    values = np.array([[0.2, 0.4], [0.6, 0.8]])
    zones = np.array([[0, 0], [1, 1]])
    summary = summarise_map(values, zones)
    assert summary.loc[0, "mean"] == pytest.approx(0.3)
    assert summary.loc[1, "mean"] == pytest.approx(0.7)


def test_summarise_map_shape_mismatch():
    with pytest.raises(ValueError):
        summarise_map(np.zeros((2, 2)), np.zeros((3, 3)))


def test_health_score_bounds_and_classes():
    scores = health_score(np.array([-0.2, 0.2, 0.5, 0.9]))
    assert np.all(scores >= 0) and np.all(scores <= 100)
    assert health_class(20) == "poor"
    assert health_class(45) == "fair"
    assert health_class(65) == "good"
    assert health_class(95) == "excellent"


def test_health_score_penalises_low_water_index():
    dry = health_score(np.array([0.6]), np.array([0.0]))
    wet = health_score(np.array([0.6]), np.array([0.3]))
    assert wet > dry
