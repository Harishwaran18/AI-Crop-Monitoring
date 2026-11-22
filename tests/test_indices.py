import numpy as np
import pytest

from cropmon.features.indices import (
    DEFAULT_WAVELENGTHS,
    band_index,
    cire,
    compute_indices,
    evi,
    gndvi,
    ndre,
    ndvi,
    psri,
    savi,
)


def test_ndvi_known_value():
    assert ndvi(np.array([0.5]), np.array([0.1]))[0] == pytest.approx(0.6667, abs=1e-4)


def test_ndvi_zero_denominator_returns_nan():
    assert np.isnan(ndvi(np.array([0.0]), np.array([0.0]))[0])


def test_index_identities():
    nir, red, green, blue, red_edge = 0.5, 0.1, 0.2, 0.05, 0.25
    assert gndvi(np.array([nir]), np.array([green]))[0] == pytest.approx((nir - green) / (nir + green))
    assert ndre(np.array([nir]), np.array([red_edge]))[0] == pytest.approx((nir - red_edge) / (nir + red_edge))
    assert savi(np.array([nir]), np.array([red]))[0] == pytest.approx(1.5 * (nir - red) / (nir + red + 0.5))
    assert evi(np.array([nir]), np.array([red]), np.array([blue]))[0] == pytest.approx(
        2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
    )
    assert psri(np.array([red]), np.array([blue]), np.array([nir]))[0] == pytest.approx((red - blue) / nir)
    assert cire(np.array([nir]), np.array([red_edge]))[0] == pytest.approx(nir / red_edge - 1)


def test_band_index_picks_nearest():
    assert DEFAULT_WAVELENGTHS[band_index(DEFAULT_WAVELENGTHS, 705)] == 710
    with pytest.raises(ValueError):
        band_index(np.array([]), 700)


def test_compute_indices_shape_and_keys():
    cube = np.full((5, 6, len(DEFAULT_WAVELENGTHS)), 0.2)
    result = compute_indices(cube)
    assert set(result) == {"ndvi", "gndvi", "ndre", "savi", "evi", "ndwi", "psri", "cire"}
    assert all(values.shape == (5, 6) for values in result.values())


def test_compute_indices_rejects_bad_shape():
    with pytest.raises(ValueError):
        compute_indices(np.zeros((5, 5)))
    with pytest.raises(ValueError):
        compute_indices(np.zeros((5, 5, 3)))


def test_vegetation_beats_stressed_leaf():
    healthy = np.full((1, 1, len(DEFAULT_WAVELENGTHS)), 0.05)
    healthy[..., band_index(DEFAULT_WAVELENGTHS, 800)] = 0.46
    healthy[..., band_index(DEFAULT_WAVELENGTHS, 670)] = 0.04
    stressed = healthy.copy()
    stressed[..., band_index(DEFAULT_WAVELENGTHS, 800)] = 0.26
    stressed[..., band_index(DEFAULT_WAVELENGTHS, 670)] = 0.135
    assert compute_indices(healthy)["ndvi"].mean() > compute_indices(stressed)["ndvi"].mean()
