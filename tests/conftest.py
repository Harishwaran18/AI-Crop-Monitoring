"""Shared fixtures: one small pipeline run reused across the suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from cropmon.config import Settings
from cropmon.pipeline import PipelineResult, run_pipeline


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    root = tmp_path_factory.mktemp("cropmon")
    return Settings(
        seed=7,
        grid_rows=40,
        grid_cols=40,
        zone_rows=8,
        zone_cols=8,
        sensor_days=6,
        history_weeks=9,
        data_dir=root / "data",
        artifacts_dir=root / "artifacts",
        figures_dir=root / "figures",
    )


@pytest.fixture(scope="session")
def pipeline_result(settings: Settings) -> PipelineResult:
    return run_pipeline(settings, make_figures=False)


@pytest.fixture(scope="session")
def artifacts(pipeline_result: PipelineResult) -> dict:
    return {
        "zones": pipeline_result.zones,
        "sensors": pipeline_result.sensors,
        "history": pipeline_result.history,
        "alerts": [alert.to_dict() for alert in pipeline_result.alerts],
        "metrics": pipeline_result.metrics,
        "summary": pipeline_result.summary,
        "model": pipeline_result.model,
        "settings": pipeline_result.settings,
    }


@pytest.fixture(scope="session")
def artifacts_dir(pipeline_result: PipelineResult) -> Path:
    return pipeline_result.settings.artifacts_dir
