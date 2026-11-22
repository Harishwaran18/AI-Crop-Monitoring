"""FastAPI service exposing the fused risk model and snapshot artifacts.

Run with:  uvicorn cropmon.api.main:app --reload
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from cropmon.config import Settings
from cropmon.pipeline import load_artifacts


class ZoneFeatures(BaseModel):
    """Any subset of model features; missing values fall back to training medians."""

    ndvi: float | None = Field(None, ge=-1.5, le=1.5)
    gndvi: float | None = None
    ndre: float | None = None
    savi: float | None = None
    ndwi: float | None = None
    psri: float | None = None
    cire: float | None = None
    ndvi_slope: float | None = None
    ndvi_change: float | None = None
    ndvi_z: float | None = None
    air_temp_mean: float | None = None
    air_temp_max: float | None = None
    humidity_mean: float | None = None
    leaf_wetness_frac: float | None = None
    rainfall_sum: float | None = None
    soil_moisture: float | None = None
    vpd_mean: float | None = None


class PredictRequest(BaseModel):
    zones: list[ZoneFeatures] = Field(..., min_length=1, max_length=1000)


class ZonePrediction(BaseModel):
    risk_probability: float
    risk_level: str
    probabilities: dict[str, float]
    predicted_class: str


class PredictResponse(BaseModel):
    predictions: list[ZonePrediction]


@lru_cache(maxsize=1)
def _artifacts() -> dict[str, Any]:
    return load_artifacts(Settings.from_env())


def create_app() -> FastAPI:
    app = FastAPI(
        title="Crop Monitor API",
        version="1.0.0",
        description="Spectral + micro-climate fusion for zone-level crop disease risk (SIH25099).",
    )

    @app.get("/health")
    def health() -> dict:
        try:
            summary = _artifacts()["summary"]
        except FileNotFoundError:
            return {"status": "degraded", "detail": "artifacts missing; run scripts/run_pipeline.py"}
        return {"status": "ok", "generated_at": summary["generated_at"], "model_accuracy": summary["model"]["accuracy"]}

    @app.get("/metadata")
    def metadata() -> dict:
        art = _artifacts()
        return {
            "model": art["model"].metadata(),
            "metrics": {k: art["metrics"][k] for k in ("accuracy", "macro_f1", "roc_auc_ovr")},
            "summary": art["summary"],
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        art = _artifacts()
        model = art["model"]
        records = [zone.model_dump(exclude_none=True) for zone in request.zones]
        probabilities = model.predict_proba(records)
        expected = model.expected_risk(records)
        labels = model.metadata()["classes"]
        predictions = []
        for index in range(len(records)):
            predictions.append(
                ZonePrediction(
                    risk_probability=round(float(expected[index]), 4),
                    risk_level=_level(float(expected[index])),
                    probabilities={labels[str(c)]: round(float(p), 4) for c, p in enumerate(probabilities[index])},
                    predicted_class=_class_label(_level(float(expected[index]))),
                )
            )
        return PredictResponse(predictions=predictions)

    @app.get("/zones")
    def zones(
        risk_level: str | None = Query(None, pattern="^(low|moderate|high)$"),
        limit: int = Query(50, ge=1, le=500),
    ) -> dict:
        frame = _artifacts()["zones"]
        if risk_level:
            frame = frame[frame["risk_level"] == risk_level]
        frame = frame.sort_values("risk_probability", ascending=False).head(limit)
        columns = [
            "zone_id",
            "risk_probability",
            "risk_level",
            "health_score",
            "health_class",
            "ndvi",
            "ndre",
            "soil_moisture",
        ]
        return {"count": int(len(frame)), "zones": json.loads(frame[columns].round(4).to_json(orient="records"))}

    @app.get("/alerts")
    def alerts(severity: str | None = Query(None, pattern="^(info|warning|critical)$")) -> dict:
        items = _artifacts()["alerts"]
        if severity:
            items = [alert for alert in items if alert["severity"] == severity]
        return {"count": len(items), "alerts": items}

    @app.get("/summary")
    def summary() -> dict:
        return _artifacts()["summary"]

    return app


def _class_label(level: str) -> str:
    return {"low": "healthy", "moderate": "moderate", "high": "high"}[level]


def _level(probability: float) -> str:
    if probability >= 0.60:
        return "high"
    if probability >= 0.35:
        return "moderate"
    return "low"


app = create_app()
