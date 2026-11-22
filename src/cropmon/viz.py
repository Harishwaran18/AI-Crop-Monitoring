"""Multi-panel figures shared by the pipeline and documentation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from cropmon.analysis.classification import CLASS_COLORS, CLASS_LABELS  # noqa: E402

PANEL_INDICES = ("ndvi", "gndvi", "savi", "evi", "msavi", "ndre")

# Fixed limits keep panels comparable across runs; auto-scaling would make a
# uniformly stressed field look as healthy as a mixed one.
INDEX_LIMITS = {
    "ndvi": (0.0, 0.9),
    "gndvi": (0.0, 0.9),
    "savi": (0.0, 1.1),
    "msavi": (0.0, 1.1),
    "evi": (0.0, 1.1),
    "ndre": (0.0, 0.8),
}


def index_panel(
    maps: dict[str, np.ndarray],
    classes: np.ndarray,
    path: Path,
    suptitle: str = "Spectral index stack and canopy classification",
) -> Path:
    """Render NDVI, GNDVI, SAVI, EVI, MSAVI, NDRE and the class map in one grid."""
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 6.8))
    flat = axes.ravel()

    for ax, name in zip(flat, PANEL_INDICES, strict=False):
        values = np.asarray(maps[name], dtype=float)
        low, high = INDEX_LIMITS.get(name, (float(np.nanmin(values)), float(np.nanmax(values))))
        image = ax.imshow(values, cmap="RdYlGn", vmin=low, vmax=high)
        ax.set_title(name.upper())
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)

    ax = flat[len(PANEL_INDICES)]
    cmap = ListedColormap([CLASS_COLORS[key] for key in sorted(CLASS_COLORS)])
    ax.imshow(np.where(classes < 0, np.nan, classes), cmap=cmap, vmin=-0.5, vmax=2.5)
    ax.set_title("CLASSIFIED MAP")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        handles=[Patch(color=CLASS_COLORS[key], label=CLASS_LABELS[key]) for key in sorted(CLASS_LABELS)],
        loc="lower right",
        fontsize=7,
        framealpha=0.9,
    )

    for ax in flat[len(PANEL_INDICES) + 1 :]:
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
