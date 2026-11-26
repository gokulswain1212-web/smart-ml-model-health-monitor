from __future__ import annotations

import json
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import backend.main as main_module
import ml.model2_development_prediction as development_prediction
import ml.model2_prediction as production_prediction
from backend.main import app
from ml.model2_config import MODEL2_FEATURE_NAMES


@pytest.fixture(scope="module")
def client():
    test_client = TestClient(app)
    response = test_client.post(
        "/api/v1/auth/login",
        json={
            "username": "tests@example.com",
            "password": "test-only-password-123",
        },
    )
    assert response.status_code == 200
    return test_client


@pytest.fixture(scope="module")
def development_artifacts(tmp_path_factory):
    artifact_dir = tmp_path_factory.mktemp("model2-development")
    metadata_path = (
        development_prediction.DEVELOPMENT_DIR
        / "model2-development-random-forest-seed-42.json"
    )
    model_path = metadata_path.with_suffix(".joblib")
    shutil.copyfile(metadata_path, artifact_dir / metadata_path.name)
    shutil.copyfile(model_path, artifact_dir / model_path.name)
    shutil.copyfile(
        metadata_path.with_name(f"{metadata_path.stem}-metrics.json"),
        artifact_dir / f"{metadata_path.stem}-metrics.json",
    )
    target_definition_path = artifact_dir / "target_definition.json"
    shutil.copyfile(development_prediction.TARGET_DEFINITION_PATH, target_definition_path)
    return artifact_dir, target_definition_path


@pytest.fixture
def configured_development_artifacts(monkeypatch, development_artifacts):
    artifact_dir, target_definition_path = development_artifacts
    monkeypatch.setattr(development_prediction, "DEVELOPMENT_DIR", artifact_dir)
    monkeypatch.setattr(
        development_prediction, "TARGET_DEFINITION_PATH", target_definition_path
    )
    return artifact_dir, target_definition_path


def valid_features() -> dict[str, float]:
    values = {
        "dataset_drift_score": 0.1,
        "max_feature_drift_score": 0.2,
        "number_drifted_features": 1,
        "missing_value_percentage": 2.0,
        "invalid_value_percentage": 1.0,
        "data_quality_score": 97.0,
        "duplicate_rate": 0.02,
        "prediction_positive_rate": 0.4,
        "prediction_positive_rate_change": 0.05,
        "prediction_mean_probability": 0.5,
        "historical_accuracy": 0.8,
        "historical_precision": 0.7,
        "historical_recall": 0.7,
        "historical_f1_score": 0.7,
        "previous_f1_trend": -0.05,
        "historical_metrics_available": 1,
        "historical_performance_observations": 8,
    }
    return {name: values[name] for name in MODEL2_FEATURE_NAMES}


def predict(client: TestClient, features: dict[str, object]):
    return client.post(
        "/api/v1/model2/development/predict",
        content=json.dumps(
            {"monitoring_features": features},
            allow_nan=True,
        ),
        headers={"Content-Type": "application/json"},
    )


def predict_batch(client: TestClient, batch_id: str):
    return client.post(
        "/api/v1/model2/development/predict",
        json={"batch_id": batch_id},
    )


def test_development_prediction_response_and_provenance(
    client, configured_development_artifacts, monkeypatch
):
    production_calls = []
    monkeypatch.setattr(
        main_module,
        "predict_degradation",
        lambda _: production_calls.append(True),
    )
    response = predict(client, valid_features())

    assert response.status_code == 200
    result = response.json()
    assert result["api_status"] == "DEVELOPMENT_PREDICTION"
    assert result["model_name"] == "Model 2 Synthetic Development Experiment"
    assert result["model_version"] == "model2-development-random-forest-seed-42"
    assert result["model_provenance"] == "SYNTHETIC_DEVELOPMENT_ONLY"
    assert result["prediction"] in (0, 1)
    assert result["predicted_class_label"] in ("stable", "degraded")
    assert set(result["class_probabilities"]) == {"stable", "degraded"}
    assert sum(result["class_probabilities"].values()) == pytest.approx(1.0)
    assert result["risk_assessment"]["risk_level"] is None
    assert "synthetic development-only" in result["explanation"]
    assert result["production_validated"] is False
    assert "not production validated" in result["explanation"]
    assert result["unavailable_features"] == []
    assert result["prediction_timestamp"]
    assert production_calls == []


def test_monitoring_batch_uses_existing_feature_builder_in_model_feature_order(
    client, configured_development_artifacts, monkeypatch
):
    features = valid_features()
    calls = []

    def build_dataset(**_kwargs):
        return pd.DataFrame(
            [
                {
                    "batch_id": "monitored-batch",
                    "source": "synthetic_test",
                    "data_source_type": "synthetic_test",
                    **features,
                }
            ]
        )

    monkeypatch.setattr(
        production_prediction, "build_historical_dataset", build_dataset
    )
    original_feature_builder = main_module.features_for_batch

    def capture_features(batch_id, **kwargs):
        result = original_feature_builder(batch_id, **kwargs)
        calls.append((batch_id, kwargs, list(result)))
        return result

    monkeypatch.setattr(main_module, "features_for_batch", capture_features)
    monkeypatch.setattr(
        main_module,
        "predict_degradation",
        lambda *_args, **_kwargs: pytest.fail(
            "Development batch prediction called the production Model 2 predictor."
        ),
    )

    response = predict_batch(client, "monitored-batch")

    assert response.status_code == 200
    result = response.json()
    assert result["batch_id"] == "monitored-batch"
    assert result["model_provenance"] == "SYNTHETIC_DEVELOPMENT_ONLY"
    assert result["production_validated"] is False
    assert calls == [
        (
            "monitored-batch",
            {"expected_source": "synthetic_test"},
            MODEL2_FEATURE_NAMES,
        )
    ]


def test_development_batch_prediction_rejects_production_observations(
    client, configured_development_artifacts, monkeypatch
):
    features = valid_features()
    monkeypatch.setattr(
        production_prediction,
        "build_historical_dataset",
        lambda **_kwargs: pd.DataFrame(
            [
                {
                    "batch_id": "genuine-production-batch",
                    "source": "production",
                    "data_source_type": "real",
                    **features,
                }
            ]
        ),
    )

    response = predict_batch(client, "genuine-production-batch")

    assert response.status_code == 404
    assert "required 'synthetic_test' source" in response.json()["detail"]


def test_monitoring_batch_preserves_legitimately_unavailable_features(
    client, configured_development_artifacts, monkeypatch
):
    features = valid_features()
    unavailable = [
        "dataset_drift_score",
        "prediction_positive_rate",
        "historical_accuracy",
        "historical_precision",
        "historical_recall",
        "historical_f1_score",
        "previous_f1_trend",
    ]
    for name in unavailable:
        features[name] = None
    features["historical_metrics_available"] = 0
    features["historical_performance_observations"] = 0
    monkeypatch.setattr(
        main_module,
        "features_for_batch",
        lambda _batch_id, **_kwargs: {
            name: features[name] for name in MODEL2_FEATURE_NAMES
        },
    )

    response = predict_batch(client, "batch-with-unavailable-metrics")

    assert response.status_code == 200
    result = response.json()
    assert result["unavailable_features"] == unavailable
    assert "fitted median imputer" in result["explanation"]


def test_monitoring_batch_missing_from_feature_dataset_returns_404(
    client, configured_development_artifacts, monkeypatch
):
    monkeypatch.setattr(
        production_prediction,
        "build_historical_dataset",
        lambda **_kwargs: pd.DataFrame(columns=["batch_id", *MODEL2_FEATURE_NAMES]),
    )

    response = predict_batch(client, "missing-monitoring-batch")

    assert response.status_code == 404
    assert "No monitoring report" in response.json()["detail"]


def test_incomplete_monitoring_features_fail_validation(
    client, configured_development_artifacts, monkeypatch
):
    features = valid_features()
    features.pop("historical_accuracy")
    monkeypatch.setattr(
        main_module, "features_for_batch", lambda _batch_id, **_kwargs: features
    )

    response = predict_batch(client, "incomplete-monitoring-batch")

    assert response.status_code == 422
    assert "Missing required Model 2 features" in response.json()["detail"]


def test_development_batch_prediction_does_not_modify_production_artifacts(
    client, configured_development_artifacts, monkeypatch
):
    features = valid_features()
    monkeypatch.setattr(
        main_module,
        "features_for_batch",
        lambda _batch_id, **_kwargs: {
            name: features[name] for name in MODEL2_FEATURE_NAMES
        },
    )
    production_writes = []
    for name in (
        "log_batch_predictions",
        "record_production_outcome",
        "evaluate_batch_performance",
    ):
        monkeypatch.setattr(
            main_module,
            name,
            lambda *args, _name=name, **kwargs: production_writes.append(_name),
        )
    artifact_dir, _ = configured_development_artifacts
    artifact_bytes = {
        path.name: path.read_bytes()
        for path in artifact_dir.iterdir()
        if path.is_file()
    }

    response = predict_batch(client, "isolated-development-prediction")

    assert response.status_code == 200
    assert production_writes == []
    assert artifact_bytes == {
        path.name: path.read_bytes()
        for path in artifact_dir.iterdir()
        if path.is_file()
    }


def test_missing_features_are_rejected(client, configured_development_artifacts):
    features = valid_features()
    features.pop(MODEL2_FEATURE_NAMES[0])

    response = predict(client, features)

    assert response.status_code == 422
    assert "Missing required" in response.json()["detail"]


def test_extra_features_are_rejected(client, configured_development_artifacts):
    features = valid_features()
    features["unexpected_feature"] = 1

    response = predict(client, features)

    assert response.status_code == 422
    assert "Unexpected" in response.json()["detail"]


@pytest.mark.parametrize(
    ("feature", "value", "message"),
    [
        ("dataset_drift_score", "0.5", "must be a number"),
        ("dataset_drift_score", float("nan"), "must be finite"),
        ("dataset_drift_score", float("inf"), "must be finite"),
        ("historical_metrics_available", None, "cannot be unavailable"),
        ("duplicate_rate", 1.1, "at most 1"),
        ("historical_metrics_available", 0.5, "whole number"),
        ("historical_performance_observations", -1, "at least 0"),
    ],
)
def test_invalid_feature_values_are_rejected(
    client, configured_development_artifacts, feature, value, message
):
    features = valid_features()
    features[feature] = value

    response = predict(client, features)

    assert response.status_code == 422
    assert message in response.json()["detail"]


def test_non_finite_positive_infinity_is_rejected(
    client, configured_development_artifacts
):
    features = valid_features()
    features["prediction_mean_probability"] = float("-inf")

    response = predict(client, features)

    assert response.status_code == 422
    assert "must be finite" in response.json()["detail"]


def test_feature_order_is_validated(client, configured_development_artifacts):
    features = dict(reversed(list(valid_features().items())))

    response = predict(client, features)

    assert response.status_code == 422
    assert "exact MODEL2_FEATURE_NAMES order" in response.json()["detail"]


def test_one_unavailable_feature_uses_the_fitted_development_imputer(
    client, configured_development_artifacts
):
    artifact_dir, _ = configured_development_artifacts
    model_path = artifact_dir / "model2-development-random-forest-seed-42.joblib"
    saved_model = joblib.load(model_path)
    imputer = saved_model.named_steps["imputer"]
    statistics_before = imputer.statistics_.copy()
    features = valid_features()
    features["historical_accuracy"] = None
    feature_row = development_prediction.validate_development_features(features)
    expected_probabilities = saved_model.predict_proba(feature_row)[0]

    response = predict(client, features)

    assert response.status_code == 200
    result = response.json()
    assert result["unavailable_features"] == ["historical_accuracy"]
    assert "fitted median imputer" in result["explanation"]
    assert result["class_probabilities"]["stable"] == pytest.approx(
        expected_probabilities[0]
    )
    assert result["class_probabilities"]["degraded"] == pytest.approx(
        expected_probabilities[1]
    )
    np.testing.assert_array_equal(imputer.statistics_, statistics_before)


def test_multiple_unavailable_features_are_reported_and_imputed(
    client, configured_development_artifacts
):
    features = valid_features()
    unavailable = [
        "dataset_drift_score",
        "max_feature_drift_score",
        "number_drifted_features",
        "prediction_positive_rate",
    ]
    for name in unavailable:
        features[name] = None

    response = predict(client, features)

    assert response.status_code == 200
    assert response.json()["unavailable_features"] == unavailable


def test_all_historical_performance_metrics_can_be_unavailable(
    client, configured_development_artifacts
):
    features = valid_features()
    unavailable = [
        "historical_accuracy",
        "historical_precision",
        "historical_recall",
        "historical_f1_score",
        "previous_f1_trend",
    ]
    for name in unavailable:
        features[name] = None
    features["historical_metrics_available"] = 0
    features["historical_performance_observations"] = 0

    response = predict(client, features)

    assert response.status_code == 200
    assert response.json()["unavailable_features"] == unavailable
    assert response.json()["model_provenance"] == "SYNTHETIC_DEVELOPMENT_ONLY"


def test_missing_development_artifact_returns_503(
    client, configured_development_artifacts
):
    artifact_dir, _ = configured_development_artifacts
    model_path = artifact_dir / "model2-development-random-forest-seed-42.joblib"
    original_bytes = model_path.read_bytes()
    try:
        model_path.unlink()
        response = predict(client, valid_features())
    finally:
        model_path.write_bytes(original_bytes)

    assert response.status_code == 503
    assert "artifact is missing" in response.json()["detail"]


def test_missing_development_metadata_returns_503(client, monkeypatch, tmp_path):
    monkeypatch.setattr(development_prediction, "DEVELOPMENT_DIR", tmp_path)
    monkeypatch.setattr(
        development_prediction,
        "TARGET_DEFINITION_PATH",
        development_prediction.TARGET_DEFINITION_PATH,
    )

    response = predict(client, valid_features())

    assert response.status_code == 503
    assert "No Model 2 development metadata" in response.json()["detail"]


@pytest.mark.parametrize(
    ("metadata_change", "message"),
    [
        ({"data_provenance": "PRODUCTION"}, "provenance"),
        ({"feature_names": ["wrong_feature"]}, "feature names"),
        ({"model_version": "wrong-version"}, "model version"),
        ({"target_name": "wrong_target"}, "target"),
    ],
)
def test_invalid_development_metadata_returns_503(
    client, configured_development_artifacts, metadata_change, message
):
    artifact_dir, _ = configured_development_artifacts
    metadata_path = artifact_dir / "model2-development-random-forest-seed-42.json"
    original_text = metadata_path.read_text(encoding="utf-8")
    try:
        metadata = json.loads(original_text)
        metadata.update(metadata_change)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        response = predict(client, valid_features())
    finally:
        metadata_path.write_text(original_text, encoding="utf-8")

    assert response.status_code == 503
    assert message in response.json()["detail"]


def test_corrupt_development_model_returns_503(
    client, configured_development_artifacts
):
    artifact_dir, _ = configured_development_artifacts
    model_path = artifact_dir / "model2-development-random-forest-seed-42.joblib"
    original_bytes = model_path.read_bytes()
    try:
        model_path.write_bytes(b"not a valid joblib artifact")
        response = predict(client, valid_features())
    finally:
        model_path.write_bytes(original_bytes)

    assert response.status_code == 503
    assert "Unable to load development model artifact" in response.json()["detail"]


def test_development_prediction_failure_returns_503(client, monkeypatch):
    def fail_prediction(_features):
        raise development_prediction.DevelopmentModelError(
            "Development model prediction failed."
        )

    monkeypatch.setattr(
        main_module, "predict_development_degradation", fail_prediction
    )
    response = predict(client, valid_features())

    assert response.status_code == 503
    assert response.json()["detail"] == "Development model prediction failed."


def test_production_model2_endpoint_remains_separate_from_development(
    client, configured_development_artifacts, monkeypatch
):
    monkeypatch.setattr(
        main_module,
        "predict_degradation",
        lambda _: {
            "model2_status": "NOT_TRAINED",
            "explanation": "No genuine labeled production observations.",
        },
    )

    development_response = predict(client, valid_features())
    production_response = client.post(
        "/api/v1/predictions/performance-degradation",
        json={"monitoring_features": valid_features()},
    )

    assert development_response.status_code == 200
    assert production_response.status_code == 503
    assert "genuine labeled production observations" in production_response.json()["detail"]


def test_model1_prediction_route_remains_unchanged(client, monkeypatch):
    expected = {
        "batch_id": "model1-isolation-test",
        "prediction_timestamp": "2026-01-01T00:00:00+00:00",
        "prediction_count": 1,
        "predictions": [{"prediction": 0}],
        "model_version": "model1-test",
        "model_artifact_sha256": "test-hash",
        "quality_status": "HEALTHY",
    }
    monkeypatch.setattr(main_module, "log_batch_predictions", lambda _: expected)
    response = client.post(
        "/api/v1/predictions/cancellation",
        json={
            "batch_id": "model1-isolation-test",
            "records": [{"hotel": "City Hotel"}],
        },
    )

    assert response.status_code == 200
    assert response.json() == expected
