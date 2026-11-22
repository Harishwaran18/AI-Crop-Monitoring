"""Agronomic risk primitives: weather-driven disease pressure and stress indices."""

from __future__ import annotations

import numpy as np

RISK_LEVELS = ("low", "moderate", "high")


def disease_pressure(
    air_temp_c: float, humidity_pct: float, leaf_wetness: float, rainfall_mm: float
) -> float:
    """Fungal/bacterial pressure in ``[0, 1]``.

    Leaf wetness duration is the dominant driver of infection, humidity and
    rainfall sustain it, and the temperature response peaks around 23 C.
    """
    temp_factor = float(np.exp(-(((air_temp_c - 23.0) / 6.0) ** 2)))
    humidity_factor = float(np.clip((humidity_pct - 70.0) / 30.0, 0.0, 1.0))
    wet_factor = float(np.clip(leaf_wetness, 0.0, 1.0))
    rain_factor = float(np.clip(rainfall_mm / 10.0, 0.0, 1.0))
    pressure = 0.45 * wet_factor + 0.25 * humidity_factor + 0.20 * temp_factor + 0.10 * rain_factor
    return float(np.clip(pressure, 0.0, 1.0))


def heat_stress(air_temp_c: float) -> float:
    """0 below 30 C, rising to 1 at 42 C."""
    return float(np.clip((air_temp_c - 30.0) / 12.0, 0.0, 1.0))


def water_stress(soil_moisture: float, optimum: float = 0.28) -> float:
    """Deficit below the optimum volumetric water content, in ``[0, 1]``."""
    return float(np.clip((optimum - soil_moisture) / 0.20, 0.0, 1.0))


def risk_level(probability: float) -> str:
    if probability >= 0.60:
        return "high"
    if probability >= 0.35:
        return "moderate"
    return "low"


def composite_risk(
    latent_stress: float, latent_disease: float, weather: float, soil_deficit: float
) -> float:
    return float(
        np.clip(0.40 * latent_stress + 0.20 * latent_disease + 0.25 * weather + 0.15 * soil_deficit, 0.0, 1.0)
    )
