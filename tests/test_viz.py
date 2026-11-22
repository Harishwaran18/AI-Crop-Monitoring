import numpy as np
import pytest

from cropmon.viz import PANEL_INDICES, index_panel


def test_index_panel_writes_every_index_and_class_map(tmp_path):
    rng = np.random.default_rng(3)
    maps = {name: rng.random((16, 16)) for name in PANEL_INDICES}
    classes = rng.integers(0, 3, size=(16, 16)).astype(np.int8)
    classes[0, 0] = -1

    path = index_panel(maps, classes, tmp_path / "panel.png")
    assert path.exists()
    assert path.stat().st_size > 10_000


def test_index_panel_requires_all_indices(tmp_path):
    with pytest.raises(KeyError):
        index_panel({"ndvi": np.zeros((8, 8))}, np.zeros((8, 8), dtype=np.int8), tmp_path / "p.png")
