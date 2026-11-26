from fastapi.testclient import TestClient
import pytest

import backend.main as main_module
from backend.main import app
from ml.model2_config import MODEL2_FEATURE_NAMES
from ml.model2_development_training import run_development_training
from ml.model2_prediction import Model2ArtifactError
from ml.model2_prediction import predict_degradation

client = TestClient(app)
assert client.post(
    "/api/v1/auth/login",
    json={
        "username": "tests@example.com",
        "password": "test-only-password-123",
    },
).status_code == 200


@pytest.fixture(autouse=True)
def allow_prediction_tests_to_reach_the_production_predictor(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "_model2_summary",
        lambda: {"status": "TRAINED", "reason": None},
    )


def test_model2_api_blocks_predictions_until_production_training_is_ready(
    monkeypatch,
):
    monkeypatch.setattr(
        main_module,
        "_model2_summary",
        lambda: {
            "status": "NOT_TRAINED",
            "reason": "Need at least 20 genuine labeled production observations; found 0.",
        },
    )
    monkeypatch.setattr(
        main_module,
        "predict_degradation",
        lambda _: pytest.fail("Production prediction ran before training eligibility."),
    )

    response = client.post(
        "/api/v1/predictions/performance-degradation",
        json={"monitoring_features": {feature: None for feature in MODEL2_FEATURE_NAMES}},
    )

    assert response.status_code == 503
    assert "Need at least 20 genuine labeled production observations" in response.json()["detail"]


def test_model2_api_returns_service_unavailable_for_untrained_model(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "list_production_observations",
        lambda **_: [{"batch_id": "normal-20261030T131227352136Z-seed-42"}],
    )
    monkeypatch.setattr(
        main_module,
        "predict_batch_degradation",
        lambda batch_id: {
            "batch_id": batch_id,
            "model2_status": "NOT_TRAINED",
            "explanation": "Model 2 needs sufficient genuine labeled historical observations.",
        },
    )
    response = client.post(
        "/api/v1/predictions/performance-degradation",
        json={"batch_id": "normal-20261030T131227352136Z-seed-42"},
    )

    assert response.status_code == 503
    result = response.json()
    assert "sufficient genuine labeled historical observations" in result["detail"]


def test_model2_production_prediction_requires_and_records_trusted_batch(
    monkeypatch,
):
    saved = []
    monkeypatch.setattr(
        main_module,
        "list_production_observations",
        lambda **_: [{"batch_id": "trusted-production-batch"}],
    )
    monkeypatch.setattr(
        main_module,
        "predict_batch_degradation",
        lambda batch_id: {
            "batch_id": batch_id,
            "model2_status": "TRAINED",
            "predicted_degradation_risk": 1,
            "degradation_probability": 0.9,
            "risk_category": "HIGH",
            "health_status": "CRITICAL",
            "prediction_timestamp": "2026-04-04T10:01:00Z",
            "model_version": "model2-production-v1",
        },
    )
    monkeypatch.setattr(
        main_module,
        "record_model2_production_prediction",
        lambda prediction: saved.append(prediction)
        or {"prediction_id": "prediction-1"},
    )

    response = client.post(
        "/api/v1/predictions/performance-degradation",
        json={"batch_id": "trusted-production-batch"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["source"] == "production"
    assert result["data_source_type"] == "real"
    assert result["prediction_id"] == "prediction-1"
    assert result["health_status"] == "CRITICAL"
    assert saved[0]["batch_id"] == "trusted-production-batch"


def test_model2_production_prediction_rejects_untrusted_batch(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "list_production_observations",
        lambda **_: [],
    )
    monkeypatch.setattr(
        main_module,
        "predict_batch_degradation",
        lambda _: pytest.fail("Untrusted batch reached the production predictor."),
    )

    response = client.post(
        "/api/v1/predictions/performance-degradation",
        json={"batch_id": "operator-declared-batch"},
    )

    assert response.status_code == 404
    assert "trusted production observations" in response.json()["detail"]


def test_model2_api_returns_service_unavailable_for_features_when_not_trained(
    monkeypatch,
):
    monkeypatch.setattr(
        main_module,
        "predict_degradation",
        lambda _: {
            "model2_status": "NOT_TRAINED",
            "explanation": "Model 2 needs sufficient genuine labeled historical observations.",
        },
    )
    response = client.post(
        "/api/v1/predictions/performance-degradation",
        json={
            "monitoring_features": {
                feature: None for feature in MODEL2_FEATURE_NAMES
            }
        },
    )

    assert response.status_code == 503
    assert "sufficient genuine labeled historical observations" in response.json()["detail"]


def test_model2_api_reports_corrupted_artifact_as_service_unavailable(monkeypatch):
    def raise_artifact_error(_):
        raise Model2ArtifactError("Model 2 artifact metadata is corrupted.")

    monkeypatch.setattr(main_module, "predict_degradation", raise_artifact_error)
    response = client.post(
        "/api/v1/predictions/performance-degradation",
        json={"monitoring_features": {feature: None for feature in MODEL2_FEATURE_NAMES}},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Model 2 artifact metadata is corrupted."


def test_model2_api_does_not_load_nested_development_artifacts(
    monkeypatch, tmp_path
):
    run_development_training(
        sample_count=400,
        random_state=42,
        output_dir=tmp_path / "development",
    )
    monkeypatch.setattr(
        main_module,
        "predict_degradation",
        lambda features: predict_degradation(features, model_dir=tmp_path),
    )

    response = client.post(
        "/api/v1/predictions/performance-degradation",
        json={
            "monitoring_features": {
                feature: 0.5 for feature in MODEL2_FEATURE_NAMES
            }
        },
    )

    assert response.status_code == 503
    assert "genuine labeled" in response.json()["detail"]


def test_model2_api_validates_payload_and_batch():
    empty = client.post(
        "/api/v1/predictions/performance-degradation",
        json={},
    )
    missing_batch = client.post(
        "/api/v1/predictions/performance-degradation",
        json={"batch_id": "does-not-exist"},
    )
    invalid_features = client.post(
        "/api/v1/predictions/performance-degradation",
        json={"monitoring_features": {"unexpected": 3}},
    )

    assert empty.status_code == 422
    assert missing_batch.status_code == 404
    assert invalid_features.status_code == 422
