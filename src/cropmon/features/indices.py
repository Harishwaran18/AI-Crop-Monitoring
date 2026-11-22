"""Spectral vegetation and soil indices.

Every index is computed from named bands so that real sensors can be plugged in
by supplying their own ``wavelengths`` array.
"""

from __future__ import annotations

import numpy as np

# 12-band multispectral stack: blue -> NIR (nanometres).
DEFAULT_WAVELENGTHS = np.array(
    [450, 490, 530, 570, 610, 660, 680, 710, 740, 780, 850, 900], dtype=float
)

INDEX_DESCRIPTIONS = {
    "ndvi": "Normalised Difference Vegetation Index (NIR, red)",
    "gndvi": "Green NDVI (NIR, green): sensitive to chlorophyll concentration",
    "ndre": "Normalised Difference Red Edge (NIR, red edge): early stress detection",
    "savi": "Soil Adjusted Vegetation Index (NIR, red): corrects for soil background",
    "evi": "Enhanced Vegetation Index (NIR, red, blue): resists atmosphere and canopy saturation",
    "ndwi": "Normalised Difference Water Index (NIR, green): canopy water content",
    "psri": "Plant Senescence Reflectance Index: detects canopy senescence",
    "cire": "Chlorophyll Index Red Edge (NIR/red edge - 1)",
}


def band_index(wavelengths: np.ndarray, target_nm: float) -> int:
    """Return the index of the band closest to ``target_nm``."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    if wavelengths.ndim != 1 or wavelengths.size == 0:
        raise ValueError("wavelengths must be a non-empty 1-D array")
    return int(np.abs(wavelengths - target_nm).argmin())


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    denominator = np.asarray(denominator, dtype=float)
    safe = np.where(np.abs(denominator) < 1e-6, np.nan, denominator)
    return np.asarray(numerator, dtype=float) / safe


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    return _safe_ratio(nir - red, nir + red)


def gndvi(nir: np.ndarray, green: np.ndarray) -> np.ndarray:
    return _safe_ratio(nir - green, nir + green)


def ndre(nir: np.ndarray, red_edge: np.ndarray) -> np.ndarray:
    return _safe_ratio(nir - red_edge, nir + red_edge)


def savi(nir: np.ndarray, red: np.ndarray, soil_factor: float = 0.5) -> np.ndarray:
    return _safe_ratio((nir - red) * (1.0 + soil_factor), nir + red + soil_factor)


def evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    denominator = nir + 6.0 * red - 7.5 * blue + 1.0
    return _safe_ratio(2.5 * (nir - red), denominator)


def ndwi(nir: np.ndarray, green: np.ndarray) -> np.ndarray:
    return _safe_ratio(nir - green, nir + green)


def psri(red: np.ndarray, blue: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return _safe_ratio(red - blue, nir)


def cire(nir: np.ndarray, red_edge: np.ndarray) -> np.ndarray:
    return _safe_ratio(nir, red_edge) - 1.0


def compute_indices(
    cube: np.ndarray, wavelengths: np.ndarray = DEFAULT_WAVELENGTHS
) -> dict[str, np.ndarray]:
    """Compute every supported index for a ``(rows, cols, bands)`` cube."""
    cube = np.asarray(cube, dtype=float)
    if cube.ndim != 3:
        raise ValueError("cube must have shape (rows, cols, bands)")
    if cube.shape[-1] != len(np.asarray(wavelengths)):
        raise ValueError("cube band count does not match wavelengths")

    blue = cube[..., band_index(wavelengths, 480)]
    green = cube[..., band_index(wavelengths, 550)]
    red = cube[..., band_index(wavelengths, 670)]
    red_edge = cube[..., band_index(wavelengths, 705)]
    nir = cube[..., band_index(wavelengths, 800)]

    indices = {
        "ndvi": ndvi(nir, red),
        "gndvi": gndvi(nir, green),
        "ndre": ndre(nir, red_edge),
        "savi": savi(nir, red),
        "evi": evi(nir, red, blue),
        "ndwi": ndwi(nir, green),
        "psri": psri(red, blue, nir),
        "cire": cire(nir, red_edge),
    }
    return {name: np.clip(values, -1.5, 1.5) for name, values in indices.items()}
