"""End-to-end pipeline: simulate -> features -> model -> zones -> alerts -> artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402 - must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cropmon.alerts.rules import Alert, AlertEngine, evaluate_zones
from cropmon.analysis.classification import class_counts, classify_canopy
from cropmon.analysis.risk import risk_level
from cropmon.analysis.zones import health_class, health_score, summarise_map, zone_map
from cropmon.config import Settings
from cropmon.data.dataset import RISK_CLASS_NAMES, build_zone_dataset
from cropmon.data.synthetic import simulate_field, simulate_history, simulate_sensors
from cropmon.features.indices import DEFAULT_WAVELENGTHS, INDEX_DESCRIPTIONS, compute_indices
from cropmon.models.fusion import RiskModel, train_risk_model
from cropmon.viz import index_panel

ARTIFACT_VERSION = "1.0.0"


@dataclass
class PipelineResult:
    settings: Settings
    zones: pd.DataFrame
    sensors: pd.DataFrame
    history: pd.DataFrame
    alerts: list[Alert]
    metrics: dict
    summary: dict
    model: RiskModel
    figures: dict[str, Path] = field(default_factory=dict)
    maps: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def alerts_frame(self) -> pd.DataFrame:
        if not self.alerts:
            return pd.DataFrame(columns=["alert_id", "zone_id", "code", "severity", "message", "created_at"])
        return pd.DataFrame([alert.to_dict() for alert in self.alerts])


def _score_zones(zones: pd.DataFrame, model: RiskModel) -> pd.DataFrame:
    scored = zones.copy()
    scored["risk_probability"] = model.expected_risk(scored)
    scored["risk_level"] = [risk_level(p) for p in scored["risk_probability"]]
    # Keep the class label consistent with the expected-risk level used by alerts.
    scored["predicted_class"] = scored["risk_level"].map(
        {"low": RISK_CLASS_NAMES[0], "moderate": RISK_CLASS_NAMES[1], "high": RISK_CLASS_NAMES[2]}
    )
    scored["health_score"] = health_score(scored["ndvi"].to_numpy(), scored["ndwi"].to_numpy())
    scored["health_class"] = [health_class(score) for score in scored["health_score"]]
    return scored


def _plot_figures(result: PipelineResult) -> dict[str, Path]:
    out = result.settings.figures_dir
    out.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Path] = {}
    ndvi = result.maps["ndvi"]
    risk = result.maps["risk"]
    zones = result.zones
    history = result.history
    sensors = result.sensors

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    image = ax.imshow(ndvi, cmap="RdYlGn", vmin=0.1, vmax=0.9)
    ax.set_title("NDVI health map (simulated field)")
    ax.set_xlabel("pixel column")
    ax.set_ylabel("pixel row")
    fig.colorbar(image, ax=ax, label="NDVI")
    figures["ndvi_map"] = _save(fig, out / "ndvi_map.png")

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    image = ax.imshow(risk, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_title("Predicted disease risk by zone")
    ax.set_xlabel("pixel column")
    ax.set_ylabel("pixel row")
    fig.colorbar(image, ax=ax, label="risk probability")
    figures["risk_map"] = _save(fig, out / "risk_map.png")

    figures["index_panel"] = index_panel(result.maps, result.maps["classes"], out / "index_panel.png")

    daily = (
        sensors.set_index("timestamp")
        .resample("1D")
        .agg(
            air_temp_mean=("air_temp_c", "mean"),
            air_temp_max=("air_temp_c", "max"),
            humidity=("humidity_pct", "mean"),
            soil_moisture=("soil_moisture", "mean"),
            rainfall=("rainfall_mm", "sum"),
        )
    )
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True)
    axes[0].plot(daily.index, daily["air_temp_mean"], color="tab:red", label="mean air temp (C)")
    axes[0].plot(daily.index, daily["air_temp_max"], color="tab:red", linestyle="--", alpha=0.6, label="max air temp (C)")
    axes[0].plot(daily.index, daily["humidity"], color="tab:blue", label="humidity (%)")
    axes[0].set_ylabel("value")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_title("Micro-climate and soil moisture (simulated sensors)")
    axes[1].bar(daily.index, daily["rainfall"], color="tab:cyan", label="rainfall (mm)")
    axes[1].plot(daily.index, daily["soil_moisture"], color="tab:green", marker="o", label="soil moisture (fraction)")
    axes[1].set_ylabel("value")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    figures["sensor_trends"] = _save(fig, out / "sensor_trends.png")

    pivot = history.pivot(index="week", columns="zone_id", values="ndvi")
    worst = zones.sort_values("risk_probability", ascending=False)["zone_id"].head(3).to_numpy()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(pivot.index, pivot.mean(axis=1), color="tab:green", linewidth=2, label="field mean NDVI")
    for zone_id in worst:
        ax.plot(pivot.index, pivot[zone_id], marker="o", linewidth=1.4, label=f"zone {zone_id} (highest risk)")
    ax.set_xlabel("week")
    ax.set_ylabel("NDVI")
    ax.set_title("Weekly NDVI trend and early-warning zones")
    ax.legend(fontsize=8)
    figures["ndvi_timeseries"] = _save(fig, out / "ndvi_timeseries.png")

    importance = pd.DataFrame(result.metrics["feature_importance"]).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.barh(importance["feature"], importance["importance"], color="tab:olive")
    ax.set_xlabel("permutation importance (accuracy drop)")
    ax.set_title("Fused model: what drives the risk prediction")
    figures["feature_importance"] = _save(fig, out / "feature_importance.png")

    matrix = np.asarray(result.metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    image = ax.imshow(matrix, cmap="Blues")
    labels = [RISK_CLASS_NAMES[i] for i in range(matrix.shape[0])]
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Holdout confusion matrix (synthetic labels)")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax)
    figures["confusion_matrix"] = _save(fig, out / "confusion_matrix.png")
    return figures


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run_pipeline(settings: Settings | None = None, make_figures: bool = True) -> PipelineResult:
    settings = settings or Settings.from_env()
    settings.ensure_dirs()

    field = simulate_field(settings.grid_rows, settings.grid_cols, settings.seed)
    sensors = simulate_sensors(settings.sensor_days, settings.seed)
    indices = compute_indices(field.cube, field.wavelengths)
    zones_index = zone_map(field.shape, settings.zone_rows, settings.zone_cols)

    base_ndvi = summarise_map(indices["ndvi"], zones_index)["mean"].to_numpy()
    history = simulate_history(base_ndvi, settings.history_weeks, settings.seed)

    dataset = build_zone_dataset(field, sensors, history, settings.zone_rows, settings.zone_cols)
    model, metrics, _, _ = train_risk_model(dataset, seed=settings.seed)
    zones = _score_zones(dataset, model)

    engine = AlertEngine(cooldown_hours=settings.alert_cooldown_hours)
    alerts = engine.filter(evaluate_zones(zones))

    classes = classify_canopy(indices)
    risk_map = zones.set_index("zone_id")["risk_probability"].to_numpy()[zones_index]
    result = PipelineResult(
        settings=settings,
        zones=zones,
        sensors=sensors,
        history=history,
        alerts=alerts,
        metrics=metrics,
        summary={},
        model=model,
        maps={**indices, "risk": risk_map, "zones": zones_index, "classes": classes},
    )

    result.summary = _build_summary(result, field.shape)
    if make_figures:
        result.figures = _plot_figures(result)
    _write_artifacts(result)
    return result


def _build_summary(result: PipelineResult, field_shape: tuple[int, int]) -> dict:
    zones = result.zones
    counts = zones["risk_level"].value_counts().to_dict()
    severity_counts = pd.Series([a.severity for a in result.alerts]).value_counts().to_dict()
    return {
        "artifact_version": ARTIFACT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "field_shape": list(field_shape),
        "n_zones": int(len(zones)),
        "mean_health_score": float(zones["health_score"].mean()),
        "mean_ndvi": float(zones["ndvi"].mean()),
        "mean_risk_probability": float(zones["risk_probability"].mean()),
        "risk_level_counts": {k: int(v) for k, v in counts.items()},
        "alert_severity_counts": {k: int(v) for k, v in severity_counts.items()},
        "alert_count": int(len(result.alerts)),
        "model": {
            "accuracy": result.metrics["accuracy"],
            "macro_f1": result.metrics["macro_f1"],
            "roc_auc_ovr": result.metrics["roc_auc_ovr"],
            "n_train": result.metrics["n_train"],
            "n_test": result.metrics["n_test"],
        },
        "top_features": [row["feature"] for row in result.metrics["feature_importance"][:5]],
        "index_names": [name for name in result.maps if name not in {"risk", "zones", "classes"}],
        "class_counts": class_counts(result.maps["classes"]),
        "data_source": result.metrics["data_source"],
    }


def _write_artifacts(result: PipelineResult) -> None:
    settings = result.settings
    result.sensors.to_csv(settings.data_dir / "sensors.csv", index=False)
    result.history.to_csv(settings.data_dir / "zone_history.csv", index=False)
    result.zones.to_csv(settings.zones_path, index=False)
    pd.DataFrame(
        [(label, count) for label, count in result.summary["class_counts"].items()],
        columns=["Class", "PixelCount"],
    ).to_csv(settings.data_dir / "class_summary.csv", index=False)

    np.savez_compressed(
        settings.data_dir / "spectral_maps.npz",
        **result.maps,
        wavelengths=DEFAULT_WAVELENGTHS,
    )
    result.model.save(settings.model_path)

    settings.metrics_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    settings.summary_path.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    settings.alerts_path.write_text(
        json.dumps([alert.to_dict() for alert in result.alerts], indent=2), encoding="utf-8"
    )


def load_summary(settings: Settings | None = None) -> dict:
    settings = settings or Settings.from_env()
    if not settings.summary_path.exists():
        raise FileNotFoundError(
            f"{settings.summary_path} is missing; run `python scripts/run_pipeline.py` first"
        )
    return json.loads(settings.summary_path.read_text(encoding="utf-8"))


def load_artifacts(settings: Settings | None = None) -> dict:
    """Load persisted artifacts for the dashboard and API."""
    settings = settings or Settings.from_env()
    if not settings.zones_path.exists():
        raise FileNotFoundError(
            f"{settings.zones_path} is missing; run `python scripts/run_pipeline.py` first"
        )
    return {
        "settings": settings,
        "zones": pd.read_csv(settings.zones_path),
        "sensors": pd.read_csv(settings.data_dir / "sensors.csv", parse_dates=["timestamp"]),
        "history": pd.read_csv(settings.data_dir / "zone_history.csv"),
        "alerts": json.loads(settings.alerts_path.read_text(encoding="utf-8")),
        "metrics": json.loads(settings.metrics_path.read_text(encoding="utf-8")),
        "summary": json.loads(settings.summary_path.read_text(encoding="utf-8")),
        "model": RiskModel.load(settings.model_path),
        "index_descriptions": INDEX_DESCRIPTIONS,
    }
