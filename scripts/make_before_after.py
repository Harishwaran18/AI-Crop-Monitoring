"""Compose the pre-fix and current index panels into one before/after figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BEFORE = ROOT / "legacy" / "figures" / "legacy_01_full_stack_7panel.png"
AFTER = ROOT / "docs" / "figures" / "index_panel.png"
OUT = ROOT / "docs" / "figures" / "before_after.png"

PANELS = [
    (
        BEFORE,
        "Before - original MATLAB export (pre-fix)",
        "EVI colourbar -5000..1000  |  ClassifiedMap one class  |  NDVI mostly below 0, per-pixel noise",
    ),
    (
        AFTER,
        "After - cropmon pipeline",
        "EVI 0..1.1  |  three classes (5517 healthy / 3035 moderate / 664 stressed)  |  contiguous stress patches",
    ),
]


def main() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 13.4))
    for ax, (path, title, subtitle) in zip(axes, PANELS, strict=True):
        ax.imshow(plt.imread(path))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=15, fontweight="bold", loc="left", pad=10)
        ax.text(0.0, -0.015, subtitle, transform=ax.transAxes, fontsize=10, va="top", color="#444444")

    fig.tight_layout(h_pad=2.4)
    fig.savefig(OUT, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
