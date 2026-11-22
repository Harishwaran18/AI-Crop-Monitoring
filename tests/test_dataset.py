
from cropmon.analysis.zones import summarise_map, zone_map
from cropmon.data.dataset import FEATURE_COLUMNS, build_zone_dataset
from cropmon.data.synthetic import simulate_field, simulate_sensors
from cropmon.features.indices import compute_indices


def _dataset(seed: int = 11):
    field = simulate_field(32, 32, seed=seed)
    sensors = simulate_sensors(days=6, seed=seed)
    zones = zone_map(field.shape, 8, 8)
    base = summarise_map(compute_indices(field.cube, field.wavelengths)["ndvi"], zones)["mean"].to_numpy()
    from cropmon.data.synthetic import simulate_history

    history = simulate_history(base, weeks=8, seed=seed)
    return build_zone_dataset(field, sensors, history, 8, 8, seed=seed)


def test_dataset_schema_and_values():
    frame = _dataset()
    assert len(frame) == 64
    assert set(FEATURE_COLUMNS).issubset(frame.columns)
    assert frame[FEATURE_COLUMNS].isna().sum().sum() == 0
    assert set(frame["risk_class"].unique()).issubset({0, 1, 2})
    assert frame["risk_score"].between(0, 1).all()


def test_dataset_is_deterministic():
    assert _dataset().equals(_dataset())


def test_dataset_covers_multiple_classes():
    assert _dataset()["risk_class"].nunique() >= 2
