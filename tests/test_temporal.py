import numpy as np
import pytest

from cropmon.features.temporal import (
    detect_anomalies,
    linear_slope,
    robust_zscore,
    summarise_series,
    trend_label,
)


def test_linear_slope_direction():
    assert linear_slope([1, 2, 3, 4]) == pytest.approx(1.0)
    assert linear_slope([4, 3, 2, 1]) == pytest.approx(-1.0)
    assert linear_slope([2.0]) == 0.0
    assert linear_slope([]) == 0.0


def test_trend_label():
    assert trend_label([0.5, 0.55, 0.60]) == "improving"
    assert trend_label([0.6, 0.55, 0.50]) == "declining"
    assert trend_label([0.5, 0.5, 0.5]) == "stable"


def test_robust_zscore_constant_series_is_zero():
    assert np.allclose(robust_zscore(np.full(6, 0.5)), 0.0)


def test_detect_anomalies_finds_spike():
    values = np.array([0.60, 0.61, 0.59, 0.60, 0.20, 0.60, 0.61])
    mask, scores = detect_anomalies(values, z_threshold=2.5)
    assert mask[4]
    assert not mask[0]
    assert abs(scores[4]) > 2.5


def test_detect_anomalies_trailing_window():
    values = np.array([0.60, 0.61, 0.59, 0.60, 0.20, 0.60, 0.61])
    mask, _ = detect_anomalies(values, z_threshold=2.5, window=4)
    assert mask[4]


def test_summarise_series_handles_nan():
    stats = summarise_series([0.5, np.nan, 0.7])
    assert stats["mean"] == pytest.approx(0.6)
    assert stats["change"] == pytest.approx(0.2)
