"""Pixel-level canopy classification.

The zone model answers "where should a scout go?". This module answers the
companion question "how much of the field looks like each condition?", which is
what the MATLAB prototypes plotted as a single-colour ``ClassifiedMap``.
"""

from __future__ import annotations

import numpy as np

CLASS_LABELS = {0: "Stressed", 1: "Moderate", 2: "Healthy"}
CLASS_COLORS = {0: "#c62828", 1: "#f9a825", 2: "#2e7d32"}
CANOPY_THRESHOLDS = (0.35, 0.55)
RED_EDGE_FLOOR = 0.18
NODATA = -1


def classify_canopy(
    indices: dict[str, np.ndarray],
    thresholds: tuple[float, float] = CANOPY_THRESHOLDS,
    red_edge_floor: float = RED_EDGE_FLOOR,
) -> np.ndarray:
    """Assign each pixel to Stressed / Moderate / Healthy.

    NDVI sets the base class. A low NDRE downgrades it by one step because the
    red edge responds to chlorophyll loss before visible NDVI does, which is the
    whole point of carrying more than one index.
    """
    ndvi_map = np.asarray(indices["ndvi"], dtype=float)
    red_edge = np.asarray(indices.get("ndre", ndvi_map), dtype=float)
    classes = np.digitize(ndvi_map, thresholds).astype(np.int8)
    classes = np.where(red_edge < red_edge_floor, np.clip(classes - 1, 0, 2), classes)
    return np.where(np.isfinite(ndvi_map), classes, NODATA).astype(np.int8)


def class_counts(classes: np.ndarray) -> dict[str, int]:
    """Count valid pixels per class; nodata pixels are ignored."""
    classes = np.asarray(classes)
    valid = classes[classes >= 0]
    return {label: int((valid == value).sum()) for value, label in CLASS_LABELS.items()}


def class_fractions(classes: np.ndarray) -> dict[str, float]:
    counts = class_counts(classes)
    total = sum(counts.values()) or 1
    return {label: round(count / total, 4) for label, count in counts.items()}
