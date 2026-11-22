import numpy as np
import pytest

from cropmon.data.synthetic import simulate_field, simulate_history, simulate_sensors
from cropmon.features.indices import compute_indices


def test_field_is_deterministic():
    first = simulate_field(24, 24, seed=3)
    second = simulate_field(24, 24, seed=3)
    assert np.allclose(first.cube, second.cube)
    assert first.shape == (24, 24)


def test_field_reflectance_is_physical():
    field = simulate_field(24, 24, seed=3)
    assert field.cube.min() > 0
    assert field.cube.max() <= 0.95
    assert field.stress.min() >= 0.0 and field.stress.max() <= 1.0
    ndvi = compute_indices(field.cube, field.wavelengths)["ndvi"]
    assert np.nanmin(ndvi) > -0.2
    assert np.nanmax(ndvi) < 0.95


def test_sensors_schema_and_ranges():
    sensors = simulate_sensors(days=3, seed=1)
    assert len(sensors) == 72
    expected = {
        "timestamp",
        "air_temp_c",
        "humidity_pct",
        "soil_moisture",
        "leaf_wetness",
        "rainfall_mm",
        "vpd_kpa",
        "dew_point_c",
    }
    assert expected.issubset(sensors.columns)
    assert sensors["humidity_pct"].between(0, 100).all()
    assert sensors["soil_moisture"].between(0, 0.5).all()
    assert sensors["leaf_wetness"].isin([0.0, 1.0]).all()
    assert sensors["rainfall_mm"].ge(0).all()
    assert sensors["vpd_kpa"].ge(0).all()


def test_sensors_deterministic():
    assert simulate_sensors(days=2, seed=5).equals(simulate_sensors(days=2, seed=5))


def test_history_has_one_row_per_zone_week():
    base = np.linspace(0.4, 0.8, 16)
    history = simulate_history(base, weeks=6, seed=2, declining_zones=2)
    assert len(history) == 16 * 6
    assert history["zone_id"].nunique() == 16
    assert history["ndvi"].between(0, 1).all()
    slopes = history.groupby("zone_id")["ndvi"].apply(lambda s: np.polyfit(range(len(s)), s, 1)[0])
    assert (slopes < 0).any(), "at least one zone should be declining"


def test_field_rejects_invalid_size():
    with pytest.raises(ValueError):
        simulate_field(0, 10, seed=1)
