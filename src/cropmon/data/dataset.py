"""Build the supervised zone-level dataset from simulated inputs.

Fusing spectral indices with micro-climate is the core idea of the platform:
the same NDVI drop means different things at 90% humidity and 35 C than it does
in dry, cool conditions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cropmon.analysis.risk import disease_pressure, water_stress
from cropmon.analysis.zones import summarise_map, zone_map
from cropmon.data.synthetic import Field
from cropmon.features.indices import compute_indices
from cropmon.features.temporal import robust_zscore, summarise_series

FEATURE_COLUMNS = [
    "ndvi",
    "gndvi",
    "ndre",
    "savi",
    "ndwi",
    "psri",
    "cire",
    "ndvi_slope",
    "ndvi_change",
    "ndvi_z",
    "air_temp_mean",
    "air_temp_max",
    "humidity_mean",
    "leaf_wetness_frac",
    "rainfall_sum",
    "soil_moisture",
    "vpd_mean",
]

TARGET_COLUMN = "risk_class"
RISK_CLASS_NAMES = {0: "healthy", 1: "moderate", 2: "high"}


def _sensor_aggregates(sensors: pd.DataFrame, days: int = 7) -> dict[str, float]:
    cutoff = sensors["timestamp"].max() - pd.Timedelta(days=days)
    window = sensors[sensors["timestamp"] >= cutoff]
    return {
        "air_temp_mean": float(window["air_temp_c"].mean()),
        "air_temp_max": float(window["air_temp_c"].max()),
        "humidity_mean": float(window["humidity_pct"].mean()),
        "leaf_wetness_frac": float(window["leaf_wetness"].mean()),
        "rainfall_sum": float(window["rainfall_mm"].sum()),
        "soil_moisture": float(window["soil_moisture"].iloc[-1]),
        "vpd_mean": float(window["vpd_kpa"].mean()),
    }


def _history_features(history: pd.DataFrame, n_zones: int) -> pd.DataFrame:
    rows = []
    for zone_id, group in history.groupby("zone_id", sort=True):
        series = group.sort_values("week")["ndvi"].to_numpy()
        stats = summarise_series(series)
        scores = robust_zscore(series)
        rows.append(
            {
                "zone_id": int(zone_id),
                "ndvi_slope": stats["slope"],
                "ndvi_change": stats["change"],
                "ndvi_z": float(scores[-1]),
            }
        )
    frame = pd.DataFrame(rows).set_index("zone_id").reindex(range(n_zones))
    return frame.fillna({"ndvi_slope": 0.0, "ndvi_change": 0.0, "ndvi_z": 0.0})


def build_zone_dataset(
    field: Field,
    sensors: pd.DataFrame,
    history: pd.DataFrame,
    zone_rows: int = 16,
    zone_cols: int = 16,
    label_noise: float = 0.08,
    seed: int = 42,
) -> pd.DataFrame:
    """Return one row per zone with fused features and a synthetic risk label."""
    zones = zone_map(field.shape, zone_rows, zone_cols)
    n_zones = zone_rows * zone_cols
    indices = compute_indices(field.cube, field.wavelengths)

    frame = pd.DataFrame(index=pd.RangeIndex(n_zones, name="zone_id"))
    for name, values in indices.items():
        frame[name] = summarise_map(values, zones)["mean"]

    sensor = _sensor_aggregates(sensors)

    # Zones sit in different microclimates; give each one a small offset so
    # climate features carry real information instead of a field-wide constant.
    rng = np.random.default_rng(seed + 3)
    frame["air_temp_mean"] = sensor["air_temp_mean"] + rng.normal(0, 1.1, n_zones)
    frame["air_temp_max"] = sensor["air_temp_max"] + rng.normal(0, 1.3, n_zones)
    frame["humidity_mean"] = np.clip(sensor["humidity_mean"] + rng.normal(0, 4.0, n_zones), 30, 99)
    frame["leaf_wetness_frac"] = np.clip(sensor["leaf_wetness_frac"] + rng.normal(0, 0.08, n_zones), 0, 1)
    frame["rainfall_sum"] = np.clip(sensor["rainfall_sum"] * rng.uniform(0.6, 1.4, n_zones), 0, None)
    frame["soil_moisture"] = np.clip(sensor["soil_moisture"] + rng.normal(0, 0.02, n_zones), 0.05, 0.45)
    frame["vpd_mean"] = np.clip(sensor["vpd_mean"] + rng.normal(0, 0.15, n_zones), 0.05, None)

    history_features = _history_features(history, n_zones)
    frame = frame.join(history_features)
    frame["ndvi_z"] = frame["ndvi_z"].fillna(0.0)

    latent_stress = summarise_map(field.stress, zones)["mean"].to_numpy()
    latent_disease = summarise_map(field.disease, zones)["mean"].to_numpy()
    weather = np.array(
        [
            disease_pressure(t, h, w, r)
            for t, h, w, r in zip(
                frame["air_temp_mean"],
                frame["humidity_mean"],
                frame["leaf_wetness_frac"],
                frame["rainfall_sum"],
                strict=True,
            )
        ]
    )
    deficit = np.array([water_stress(v) for v in frame["soil_moisture"]])

    canopy_deficit = np.clip((0.78 - frame["ndvi"].to_numpy()) / 0.45, 0.0, 1.0)
    risk = np.array(
        [
            0.35 * canopy + 0.30 * weather_i + 0.20 * deficit_i + 0.15 * disease
            for canopy, weather_i, deficit_i, disease in zip(
                canopy_deficit, weather, deficit, latent_disease, strict=True
            )
        ]
    )
    frame["latent_stress"] = latent_stress
    frame["weather_pressure"] = weather
    frame["risk_score"] = risk
    labels = np.digitize(risk, bins=[0.42, 0.58]).astype(int)

    # Real scouting labels are noisy; inject a little to keep metrics honest.
    rng = np.random.default_rng(seed + 7)
    flips = rng.random(labels.size) < label_noise
    labels[flips] = np.clip(labels[flips] + rng.choice([-1, 1], size=int(flips.sum())), 0, 2)
    frame["risk_class"] = labels

    return frame.reset_index()
