import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(settings, pipeline_result, monkeypatch):
    monkeypatch.setenv("CROPMON_DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("CROPMON_ARTIFACTS_DIR", str(settings.artifacts_dir))
    monkeypatch.setenv("CROPMON_FIGURES_DIR", str(settings.figures_dir))

    from cropmon.api.main import _artifacts, create_app

    _artifacts.cache_clear()
    with TestClient(create_app()) as test_client:
        yield test_client
    _artifacts.cache_clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert 0 <= body["model_accuracy"] <= 1


def test_metadata(client):
    body = client.get("/metadata").json()
    assert body["model"]["algorithm"] == "HistGradientBoostingClassifier"
    assert "accuracy" in body["metrics"]
    assert body["summary"]["n_zones"] > 0


def test_summary(client):
    body = client.get("/summary").json()
    assert body["artifact_version"] == "1.0.0"
    assert body["alert_count"] >= 0


def test_predict_single_zone(client):
    response = client.post("/predict", json={"zones": [{"ndvi": 0.32, "soil_moisture": 0.16}]})
    assert response.status_code == 200
    prediction = response.json()["predictions"][0]
    assert 0 <= prediction["risk_probability"] <= 1
    assert prediction["risk_level"] in {"low", "moderate", "high"}
    assert pytest.approx(sum(prediction["probabilities"].values()), abs=1e-3) == 1.0


def test_predict_stressed_zone_scores_higher(client):
    healthy = client.post("/predict", json={"zones": [{"ndvi": 0.85, "psri": 0.0, "soil_moisture": 0.32}]}).json()
    stressed = client.post("/predict", json={"zones": [{"ndvi": 0.30, "psri": 0.25, "soil_moisture": 0.12}]}).json()
    assert stressed["predictions"][0]["risk_probability"] > healthy["predictions"][0]["risk_probability"]


def test_predict_validates_payload(client):
    assert client.post("/predict", json={"zones": []}).status_code == 422
    assert client.post("/predict", json={"zones": [{"ndvi": 9.0}]}).status_code == 422


def test_zones_filter_and_limit(client):
    body = client.get("/zones", params={"risk_level": "high", "limit": 5}).json()
    assert body["count"] <= 5
    assert all(zone["risk_level"] == "high" for zone in body["zones"])
    assert client.get("/zones", params={"risk_level": "nonsense"}).status_code == 422


def test_alerts_filter(client):
    body = client.get("/alerts", params={"severity": "critical"}).json()
    assert all(alert["severity"] == "critical" for alert in body["alerts"])
    assert body["count"] == len(body["alerts"])
