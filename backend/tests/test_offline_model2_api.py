from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)
assert client.post(
    "/api/v1/auth/login",
    json={
        "username": "tests@example.com",
        "password": "test-only-password-123",
    },
).status_code == 200


def test_offline_model2_status_is_authenticated_and_scope_labeled(monkeypatch):
    import backend.main as main

    monkeypatch.setattr(
        main,
        "offline_model2_status",
        lambda: {
            "status": "TRAINED",
            "model_scope": "HISTORICAL_OFFLINE",
            "provenance": "HISTORICAL_OFFLINE",
            "model1_replay_version": "historical-refit-test",
            "eligible_observation_count": 41,
        },
    )
    response = client.get("/api/v1/model2/offline/status")

    assert response.status_code == 200
    assert response.json()["model_scope"] == "HISTORICAL_OFFLINE"
    assert response.json()["provenance"] == "HISTORICAL_OFFLINE"
    assert response.json()["model1_replay_version"] == "historical-refit-test"

    unauthenticated = TestClient(app).get("/api/v1/model2/offline/status")
    assert unauthenticated.status_code == 401


def test_offline_prediction_requires_trained_offline_model_and_returns_scope(monkeypatch):
    import backend.main as main

    monkeypatch.setattr(
        main,
        "offline_model2_status",
        lambda: {
            "status": "TRAINED",
            "model_scope": "HISTORICAL_OFFLINE",
            "provenance": "HISTORICAL_OFFLINE",
            "reason": None,
        },
    )
    monkeypatch.setattr(
        main,
        "predict_offline_degradation",
        lambda features: {
            "model2_status": "TRAINED",
            "model_scope": "HISTORICAL_OFFLINE",
            "provenance": "HISTORICAL_OFFLINE",
            "model1_replay_version": "historical-refit-test",
            "prediction_timestamp": "2026-11-20T12:00:00+00:00",
            "predicted_degradation_risk": 0,
            "degradation_probability": 0.2,
            "model_version": "offline-test",
        },
    )

    response = client.post(
        "/api/v1/model2/offline/predict",
        json={"monitoring_features": {"dataset_drift_score": 0.1}},
    )

    assert response.status_code == 200
    assert response.json()["model_scope"] == "HISTORICAL_OFFLINE"
    assert response.json()["predicted_degradation_risk"] == 0
    assert response.json()["model1_replay_version"] == "historical-refit-test"
    assert response.json()["prediction_timestamp"] == "2026-11-20T12:00:00+00:00"


def test_offline_prediction_rejects_untrained_model_and_invalid_inputs(monkeypatch):
    import backend.main as main

    monkeypatch.setattr(
        main,
        "offline_model2_status",
        lambda: {
            "status": "NOT_TRAINED",
            "reason": "Need 20 eligible offline observations.",
        },
    )
    response = client.post(
        "/api/v1/model2/offline/predict",
        json={"monitoring_features": {}},
    )
    assert response.status_code == 503
    assert "Need 20" in response.json()["detail"]

    monkeypatch.setattr(
        main,
        "offline_model2_status",
        lambda: {"status": "TRAINED", "reason": None},
    )
    invalid = client.post(
        "/api/v1/model2/offline/predict",
        json={"monitoring_features": {"dataset_drift_score": "not-a-number"}},
    )
    assert invalid.status_code == 422


def test_production_model2_status_remains_a_distinct_route(monkeypatch):
    import backend.main as main

    monkeypatch.setattr(
        main,
        "_model2_summary",
        lambda: {
            "status": "NOT_TRAINED",
            "training_eligibility": "NOT_TRAINED",
            "reason": "No genuine production outcomes.",
        },
    )

    production = client.get("/api/v1/model2/training-status")
    offline = client.get("/api/v1/model2/offline/status")

    assert production.status_code == 200
    assert production.json()["reason"] == "No genuine production outcomes."
    assert offline.status_code == 200
    assert offline.json()["model_scope"] == "HISTORICAL_OFFLINE"


def test_saved_offline_model_is_served_only_from_offline_routes():
    from ml.model2_config import MODEL2_FEATURE_NAMES

    status = client.get("/api/v1/model2/offline/status")
    assert status.status_code == 200
    status_data = status.json()
    assert status_data["model_scope"] == "HISTORICAL_OFFLINE"
    assert status_data["provenance"] == "HISTORICAL_OFFLINE"

    request = {
        "monitoring_features": {name: 0.1 for name in MODEL2_FEATURE_NAMES}
    }
    response = client.post("/api/v1/model2/offline/predict", json=request)
    if status_data["status"] == "TRAINED":
        assert response.status_code == 200
        assert response.json()["model_scope"] == "HISTORICAL_OFFLINE"
        assert response.json()["provenance"] == "HISTORICAL_OFFLINE"
        assert 0 <= response.json()["degradation_probability"] <= 1
    else:
        assert response.status_code == 503
