"""Temporal feature extraction and change detection."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_mean(series: pd.Series, window: int = 3, min_periods: int = 1) -> pd.Series:
    return series.rolling(window=window, min_periods=min_periods).mean()


def linear_slope(values: np.ndarray | pd.Series) -> float:
    """Least-squares slope per step. Returns 0.0 for fewer than two points."""
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 2:
        return 0.0
    x = np.arange(y.size, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def robust_zscore(values: np.ndarray | pd.Series) -> np.ndarray:
    """Median absolute deviation z-score; resistant to a single bad observation."""
    y = np.asarray(values, dtype=float)
    median = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - median))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-9:
        return np.zeros_like(y)
    return (y - median) / scale


def ewma(series: pd.Series, halflife: float = 3.0) -> pd.Series:
    return series.ewm(halflife=halflife, adjust=False).mean()


def detect_anomalies(
    values: np.ndarray | pd.Series, z_threshold: float = 2.5, window: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Flag points whose robust z-score exceeds ``z_threshold``.

    Returns ``(mask, scores)``. When ``window`` is given the score is computed
    against a trailing window, which is what an operational system needs.
    """
    y = np.asarray(values, dtype=float)
    scores = np.zeros_like(y)
    if window is None:
        scores = robust_zscore(y)
    else:
        for i in range(y.size):
            start = max(0, i - window + 1)
            scores[i] = robust_zscore(y[start : i + 1])[-1]
    mask = np.abs(scores) >= z_threshold
    return mask, scores


def trend_label(values: np.ndarray | pd.Series, tolerance: float = 0.005) -> str:
    """Classify a series as improving, stable, or declining."""
    slope = linear_slope(values)
    if slope > tolerance:
        return "improving"
    if slope < -tolerance:
        return "declining"
    return "stable"


def summarise_series(values: np.ndarray | pd.Series, window: int = 3) -> dict[str, float]:
    y = np.asarray(values, dtype=float)
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "slope": 0.0, "change": 0.0}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "slope": linear_slope(finite),
        "change": float(finite[-1] - finite[0]),
        "recent_mean": float(np.mean(finite[-window:])),
    }
