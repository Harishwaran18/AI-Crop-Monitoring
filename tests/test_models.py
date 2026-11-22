import numpy as np
import pandas as pd
import pytest

from cropmon.data.dataset import FEATURE_COLUMNS, RISK_CLASS_NAMES
from cropmon.models.fusion import RiskModel, evaluate_model, train_risk_model


def test_training_produces_usable_metrics(pipeline_result):
    metrics = pipeline_result.metrics
    assert 0.5 <= metrics["accuracy"] <= 1.0
    assert 0.4 <= metrics["macro_f1"] <= 1.0
    assert metrics["n_train"] > 0 and metrics["n_test"] > 0
    assert len(metrics["confusion_matrix"]) >= 2
    assert "synthetic" in metrics["data_source"]
    assert metrics["feature_importance"][0]["importance"] >= 0


def test_probabilities_are_normalised(pipeline_result):
    probabilities = pipeline_result.model.predict_proba(pipeline_result.zones)
    assert probabilities.shape[1] == len(RISK_CLASS_NAMES)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_expected_risk_is_bounded(pipeline_result):
    risk = pipeline_result.model.expected_risk(pipeline_result.zones)
    assert np.all(risk >= 0) and np.all(risk <= 1)


def test_missing_features_fall_back_to_defaults(pipeline_result):
    model = pipeline_result.model
    frame = model.prepare([{"ndvi": 0.4}])
    assert list(frame.columns) == FEATURE_COLUMNS
    assert not frame.isna().any().any()


def test_save_and_load_roundtrip(pipeline_result, tmp_path):
    model = pipeline_result.model
    path = model.save(tmp_path / "model.joblib")
    reloaded = RiskModel.load(path)
    sample = pipeline_result.zones.head(5)
    assert np.allclose(model.predict_proba(sample), reloaded.predict_proba(sample))
    assert reloaded.metadata()["version"] == model.version


def test_unfitted_model_raises():
    with pytest.raises(RuntimeError):
        RiskModel().predict_proba([{"ndvi": 0.5}])


def test_training_is_deterministic(pipeline_result):
    frame = pipeline_result.zones
    first, _, _, _ = train_risk_model(frame, seed=3)
    second, _, _, _ = train_risk_model(frame, seed=3)
    assert np.allclose(first.predict_proba(frame), second.predict_proba(frame))


def test_evaluate_model_returns_expected_keys(pipeline_result):
    frame = pipeline_result.zones
    metrics = evaluate_model(pipeline_result.model, frame[FEATURE_COLUMNS], frame["risk_class"])
    assert {"accuracy", "macro_f1", "confusion_matrix", "feature_importance"} <= set(metrics)


def test_load_rejects_wrong_type(tmp_path):
    import joblib

    path = tmp_path / "not_a_model.joblib"
    joblib.dump({"a": 1}, path)
    with pytest.raises(TypeError):
        RiskModel.load(path)


def test_predict_accepts_dataframe_and_records(pipeline_result):
    row = pipeline_result.zones.iloc[[0]][FEATURE_COLUMNS]
    from_frame = pipeline_result.model.predict(row)
    from_record = pipeline_result.model.predict([row.iloc[0].to_dict()])
    assert from_frame[0] == from_record[0]


def test_pandas_frame_keeps_dtypes(pipeline_result):
    frame = pd.DataFrame([{"ndvi": 0.55, "soil_moisture": 0.24}])
    assert isinstance(pipeline_result.model.predict(frame), np.ndarray)
