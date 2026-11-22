"""Deterministic simulation of a field, its sensors, and its seasonal history.

The simulator exists so the whole platform can be exercised without private
imagery or hardware. It is physics-inspired, not a substitute for real data:
reflectance follows a leaf/soil mixing model, and the weather generator uses a
diurnal cycle plus a soil-moisture bucket with rainfall and evapotranspiration.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from cropmon.features.indices import DEFAULT_WAVELENGTHS

# Per-band reflectance of a fully healthy leaf, a stressed leaf, and bare soil.
LEAF_HEALTHY = np.array(
    [0.045, 0.075, 0.055, 0.060, 0.045, 0.040, 0.038, 0.20, 0.36, 0.46, 0.50, 0.42]
)
LEAF_STRESSED = np.array(
    [0.070, 0.100, 0.100, 0.110, 0.120, 0.130, 0.135, 0.170, 0.220, 0.260, 0.300, 0.270]
)
SOIL = np.array(
    [0.120, 0.140, 0.160, 0.180, 0.200, 0.220, 0.230, 0.240, 0.250, 0.260, 0.280, 0.290]
)


@dataclass
class Field:
    """A simulated reflectance cube plus the latent maps behind it."""

    cube: np.ndarray
    wavelengths: np.ndarray
    stress: np.ndarray
    disease: np.ndarray
    vegetation_fraction: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.cube.shape[0], self.cube.shape[1]


def _gaussian_blob(
    rows: int, cols: int, rng: np.random.Generator, count: int, sigma_range: tuple[float, float]
) -> np.ndarray:
    yy, xx = np.mgrid[0:rows, 0:cols]
    field = np.zeros((rows, cols), dtype=float)
    for _ in range(count):
        cy, cx = rng.uniform(0, rows), rng.uniform(0, cols)
        sigma = rng.uniform(*sigma_range)
        amplitude = rng.uniform(0.4, 1.0)
        field += amplitude * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)))
    return field


def simulate_field(
    rows: int = 96, cols: int = 96, seed: int = 42
) -> Field:
    """Simulate a multispectral reflectance cube with spatial stress patterns."""
    if rows < 2 or cols < 2:
        raise ValueError("field must be at least 2x2 pixels")
    rng = np.random.default_rng(seed)
    yy, _ = np.mgrid[0:rows, 0:cols]

    stress = (
        0.60 * _gaussian_blob(rows, cols, rng, count=3, sigma_range=(7, 16))
        + 0.22 * (yy / max(rows - 1, 1))
        + 0.08 * rng.random((rows, cols))
    )
    stress = np.clip(stress / max(np.percentile(stress, 99), 1e-6) * 0.75, 0.0, 1.0)

    disease = _gaussian_blob(rows, cols, rng, count=3, sigma_range=(3, 7))
    disease = np.clip(disease / max(disease.max(), 1e-6), 0.0, 1.0)

    vegetation_fraction = np.clip(0.95 - 0.55 * stress + 0.05 * rng.standard_normal((rows, cols)), 0.05, 1.0)

    leaf = LEAF_HEALTHY[None, None, :] * (1.0 - stress[..., None]) + LEAF_STRESSED[None, None, :] * stress[..., None]
    cube = vegetation_fraction[..., None] * leaf + (1.0 - vegetation_fraction[..., None]) * SOIL[None, None, :]

    # Diseased tissue loses NIR reflectance and gains red reflectance.
    cube[..., 5:7] += 0.06 * disease[..., None]
    cube[..., 8:12] -= 0.08 * disease[..., None]
    cube *= 1.0 + 0.03 * rng.standard_normal(cube.shape)

    return Field(
        cube=np.clip(cube, 0.01, 0.95),
        wavelengths=DEFAULT_WAVELENGTHS.copy(),
        stress=stress,
        disease=disease,
        vegetation_fraction=vegetation_fraction,
    )


def simulate_sensors(days: int = 14, seed: int = 42, start: str = "2025-06-01") -> pd.DataFrame:
    """Hourly micro-climate and soil-moisture series with rainfall events."""
    rng = np.random.default_rng(seed + 1)
    hours = days * 24
    t = np.arange(hours)
    hour_of_day = t % 24
    start_dt = datetime.fromisoformat(start)

    seasonal = 1.5 * np.sin(2 * np.pi * t / (24 * 7))
    air_temp = 25.0 + 7.0 * np.sin(2 * np.pi * (hour_of_day - 9) / 24) + seasonal
    air_temp += rng.normal(0, 0.6, hours)

    humidity = np.clip(96.0 - 2.1 * (air_temp - 18.0) + rng.normal(0, 2.5, hours), 25.0, 99.0)

    rain = np.where(rng.random(hours) < 0.020, rng.exponential(4.0, hours), 0.0)

    soil = np.zeros(hours)
    soil[0] = 0.32
    for i in range(1, hours):
        # Root-zone water balance: ~5 mm/day evapotranspiration over 300 mm depth.
        et = 0.00040 + 0.000045 * max(air_temp[i] - 20.0, 0.0)
        soil[i] = soil[i - 1] + rain[i] / 300.0 - et
        soil[i] = min(max(soil[i], 0.10), 0.42)

    wet = ((humidity > 87) | (rain > 0)).astype(float)
    for i in range(1, hours):
        if wet[i - 1] == 1.0 and humidity[i] > 80:
            wet[i] = 1.0

    saturation_vp = 0.6108 * np.exp(17.27 * air_temp / (air_temp + 237.3))
    vpd = saturation_vp * (1.0 - humidity / 100.0)
    dew_point = air_temp - (100.0 - humidity) / 5.0

    timestamps = [start_dt + timedelta(hours=int(i)) for i in t]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "air_temp_c": np.round(air_temp, 2),
            "humidity_pct": np.round(humidity, 1),
            "soil_moisture": np.round(soil, 4),
            "leaf_wetness": np.round(wet, 2),
            "rainfall_mm": np.round(rain, 2),
            "vpd_kpa": np.round(vpd, 3),
            "dew_point_c": np.round(dew_point, 2),
        }
    )


def simulate_history(
    base_ndvi: np.ndarray, weeks: int = 12, seed: int = 42, declining_zones: int = 5
) -> pd.DataFrame:
    """Weekly NDVI per zone with a few injected declining and anomalous zones."""
    rng = np.random.default_rng(seed + 2)
    n_zones = int(base_ndvi.size)
    decline = np.zeros(n_zones)
    declining = rng.choice(n_zones, size=min(declining_zones, n_zones), replace=False)
    decline[declining] = rng.uniform(0.012, 0.030, size=declining.size)

    records = []
    for week in range(weeks):
        noise = rng.normal(0, 0.008, n_zones)
        ndvi = base_ndvi - decline * week + noise
        if week == weeks - 3:
            ndvi[declining[:2]] -= 0.10  # abrupt stress event, for anomaly detection
        records.append(pd.DataFrame({"week": week, "zone_id": np.arange(n_zones), "ndvi": ndvi}))
    history = pd.concat(records, ignore_index=True)
    return history.sort_values(["zone_id", "week"], ignore_index=True)
