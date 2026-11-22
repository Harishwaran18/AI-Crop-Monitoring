"""Central configuration for the crop monitoring platform.

All secrets and deployment-specific values are read from the environment so the
repository stays free of credentials and personal data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings.

    Defaults produce a self-contained demo that writes into ``data/`` and
    ``artifacts/`` inside the repository.
    """

    seed: int = 42
    grid_rows: int = 96
    grid_cols: int = 96
    zone_rows: int = 16
    zone_cols: int = 16
    sensor_days: int = 14
    history_weeks: int = 12
    alert_cooldown_hours: float = 6.0
    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "data")
    artifacts_dir: Path = field(default_factory=lambda: REPO_ROOT / "artifacts")
    figures_dir: Path = field(default_factory=lambda: REPO_ROOT / "docs" / "figures")

    @classmethod
    def from_env(cls) -> Settings:
        defaults = cls()
        return replace(
            defaults,
            seed=int(os.environ.get("CROPMON_SEED", defaults.seed)),
            grid_rows=int(os.environ.get("CROPMON_GRID_ROWS", defaults.grid_rows)),
            grid_cols=int(os.environ.get("CROPMON_GRID_COLS", defaults.grid_cols)),
            zone_rows=int(os.environ.get("CROPMON_ZONE_ROWS", defaults.zone_rows)),
            zone_cols=int(os.environ.get("CROPMON_ZONE_COLS", defaults.zone_cols)),
            sensor_days=int(os.environ.get("CROPMON_SENSOR_DAYS", defaults.sensor_days)),
            history_weeks=int(os.environ.get("CROPMON_HISTORY_WEEKS", defaults.history_weeks)),
            data_dir=_path_from_env("CROPMON_DATA_DIR", defaults.data_dir),
            artifacts_dir=_path_from_env("CROPMON_ARTIFACTS_DIR", defaults.artifacts_dir),
            figures_dir=_path_from_env("CROPMON_FIGURES_DIR", defaults.figures_dir),
        )

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.artifacts_dir, self.figures_dir):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / "risk_model.joblib"

    @property
    def metrics_path(self) -> Path:
        return self.artifacts_dir / "metrics.json"

    @property
    def zones_path(self) -> Path:
        return self.artifacts_dir / "zones.csv"

    @property
    def alerts_path(self) -> Path:
        return self.artifacts_dir / "alerts.json"

    @property
    def summary_path(self) -> Path:
        return self.artifacts_dir / "summary.json"
