import numpy as np
import pytest

from cropmon.analysis.classification import (
    CLASS_LABELS,
    NODATA,
    class_counts,
    class_fractions,
    classify_canopy,
)


def _indices(ndvi, ndre):
    return {"ndvi": np.asarray(ndvi, dtype=float), "ndre": np.asarray(ndre, dtype=float)}


def test_thresholds_partition_the_field():
    classes = classify_canopy(_indices([[0.2, 0.45], [0.7, 0.55]], np.full((2, 2), 0.4)))
    assert classes.tolist() == [[0, 1], [2, 2]]


def test_low_red_edge_downgrades_class():
    classes = classify_canopy(_indices([[0.7]], [[0.10]]))
    assert classes[0, 0] == 1


def test_nodata_is_masked():
    classes = classify_canopy(_indices([[np.nan, 0.6]], [[0.3, 0.3]]))
    assert classes[0, 0] == NODATA
    assert classes[0, 1] == 2


def test_counts_ignore_nodata_and_fractions_sum_to_one():
    classes = np.array([[0, 1, 2, NODATA]])
    assert class_counts(classes) == {"Stressed": 1, "Moderate": 1, "Healthy": 1}
    fractions = class_fractions(classes)
    assert sum(fractions.values()) == pytest.approx(1.0, abs=1e-3)
    assert set(fractions) == set(CLASS_LABELS.values())


def test_class_is_monotonic_in_ndvi():
    ndvi = np.linspace(0.0, 1.0, 11)
    classes = classify_canopy(_indices(ndvi, np.full(11, 0.4)))
    assert np.all(np.diff(classes) >= 0)


def test_missing_red_edge_falls_back_to_ndvi():
    classes = classify_canopy({"ndvi": np.array([[0.7]])})
    assert classes[0, 0] == 2
