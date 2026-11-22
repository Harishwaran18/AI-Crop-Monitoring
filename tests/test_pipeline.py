import json

import numpy as np

from cropmon.pipeline import PipelineResult, load_artifacts, load_summary


def test_artifacts_written(pipeline_result: PipelineResult):
    settings = pipeline_result.settings
    for path in [
        settings.model_path,
        settings.metrics_path,
        settings.zones_path,
        settings.alerts_path,
        settings.summary_path,
        settings.data_dir / "sensors.csv",
        settings.data_dir / "zone_history.csv",
        settings.data_dir / "spectral_maps.npz",
    ]:
        assert path.exists(), f"missing artifact {path}"


def test_summary_contents(pipeline_result: PipelineResult):
    summary = pipeline_result.summary
    assert summary["n_zones"] == pipeline_result.settings.zone_rows * pipeline_result.settings.zone_cols
    assert summary["alert_count"] == len(pipeline_result.alerts)
    assert 0 <= summary["mean_health_score"] <= 100
    assert set(summary["risk_level_counts"]) <= {"low", "moderate", "high"}
    assert "synthetic" in summary["data_source"]
    assert "ndvi" in summary["index_names"]


def test_zones_are_scored(pipeline_result: PipelineResult):
    zones = pipeline_result.zones
    assert zones["risk_probability"].between(0, 1).all()
    assert zones["risk_level"].isin(["low", "moderate", "high"]).all()
    assert zones["health_class"].isin(["poor", "fair", "good", "excellent"]).all()
    assert zones["predicted_class"].isin(["healthy", "moderate", "high"]).all()


def test_alerts_are_serialisable(pipeline_result: PipelineResult):
    frame = pipeline_result.alerts_frame
    assert list(frame.columns)[:5] == ["alert_id", "zone_id", "code", "severity", "message"]
    assert frame["severity"].isin(["info", "warning", "critical"]).all()


def test_maps_contain_every_index(pipeline_result: PipelineResult):
    for name in ("ndvi", "ndre", "savi", "msavi", "ndwi", "psri", "cire", "risk", "zones", "classes"):
        assert name in pipeline_result.maps
    assert pipeline_result.maps["ndvi"].shape == tuple(pipeline_result.summary["field_shape"])


def test_load_artifacts_roundtrip(pipeline_result: PipelineResult):
    art = load_artifacts(pipeline_result.settings)
    assert len(art["zones"]) == pipeline_result.summary["n_zones"]
    assert art["model"].metadata()["feature_names"]
    assert art["summary"] == json.loads(json.dumps(art["summary"]))
    assert load_summary(pipeline_result.settings)["artifact_version"] == "1.0.0"


def test_load_artifacts_missing_raises(tmp_path):
    from cropmon.config import Settings

    empty = Settings(data_dir=tmp_path / "d", artifacts_dir=tmp_path / "a", figures_dir=tmp_path / "f")
    try:
        load_artifacts(empty)
    except FileNotFoundError as exc:
        assert "run_pipeline" in str(exc) or "run `python" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_ndvi_map_matches_zones_mean(pipeline_result: PipelineResult):
    zones = pipeline_result.zones.set_index("zone_id")
    zone_ids = pipeline_result.maps["zones"]
    ndvi = pipeline_result.maps["ndvi"]
    zone = int(zone_ids[0, 0])
    assert np.isclose(np.nanmean(ndvi[zone_ids == zone]), zones.loc[zone, "ndvi"], atol=1e-9)


def test_pixel_classification_exported(pipeline_result: PipelineResult):
    classes = pipeline_result.maps["classes"]
    counts = pipeline_result.summary["class_counts"]
    assert classes.shape == tuple(pipeline_result.summary["field_shape"])
    assert set(counts) == {"Stressed", "Moderate", "Healthy"}
    assert sum(counts.values()) <= classes.size
    summary_csv = pipeline_result.settings.data_dir / "class_summary.csv"
    assert summary_csv.exists()
    assert "PixelCount" in summary_csv.read_text().splitlines()[0]
