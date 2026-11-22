"""Zone aggregation and canopy health scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

HEALTH_BANDS = [(0.0, 0.35, "poor"), (0.35, 0.55, "fair"), (0.55, 0.75, "good"), (0.75, 1.01, "excellent")]


def zone_map(shape: tuple[int, int], zone_rows: int = 16, zone_cols: int = 16) -> np.ndarray:
    """Assign a zone id to every pixel using a regular tiling."""
    rows, cols = shape
    if zone_rows <= 0 or zone_cols <= 0:
        raise ValueError("zone grid must be positive")
    row_idx = np.minimum((np.arange(rows) * zone_rows // rows), zone_rows - 1)
    col_idx = np.minimum((np.arange(cols) * zone_cols // cols), zone_cols - 1)
    return (row_idx[:, None] * zone_cols + col_idx[None, :]).astype(int)


def summarise_map(values: np.ndarray, zones: np.ndarray) -> pd.DataFrame:
    """Aggregate a pixel map into per-zone statistics."""
    values = np.asarray(values, dtype=float)
    zones = np.asarray(zones)
    if values.shape != zones.shape:
        raise ValueError("values and zones must share a shape")
    flat = pd.DataFrame({"zone_id": zones.ravel(), "value": values.ravel()})
    grouped = flat.groupby("zone_id")["value"]
    return pd.DataFrame(
        {
            "mean": grouped.mean(),
            "median": grouped.median(),
            "std": grouped.std().fillna(0.0),
            "min": grouped.min(),
            "p10": grouped.quantile(0.10),
            "p90": grouped.quantile(0.90),
        }
    )


def health_score(ndvi: np.ndarray | pd.Series, ndwi: np.ndarray | pd.Series | None = None) -> np.ndarray:
    """Map NDVI (and optional canopy water index) onto a 0-100 health score."""
    ndvi_arr = np.asarray(ndvi, dtype=float)
    score = np.clip((ndvi_arr - 0.15) / (0.85 - 0.15), 0.0, 1.0) * 100.0
    if ndwi is not None:
        water = np.clip((np.asarray(ndwi, dtype=float) - 0.05) / 0.35, 0.0, 1.0) * 100.0
        score = 0.8 * score + 0.2 * water
    return np.round(score, 1)


def health_class(score: float) -> str:
    normalised = float(np.clip(score, 0.0, 100.0)) / 100.0
    for low, high, label in HEALTH_BANDS:
        if low <= normalised < high:
            return label
    return "excellent"
