import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import backend.main as main
from ml.performance_tracking import (
    PoorQualityBatchError,
    evaluate_batch_performance as evaluate_batch_performance_with_paths,
    log_batch_predictions as log_batch_predictions_with_paths,
    record_production_outcome as record_production_outcome_with_paths,
)
from ml.model2_config import DEGRADATION_TARGET
from ml.model2_dataset import build_historical_dataset


client = TestClient(main.app)
assert client.post(
    "/api/v1/auth/login",
    json={
        "username": "tests@example.com",
        "password": "test-only-password-123",
    },
).status_code == 200
ROOT = Path(__file__).resolve().parents[2]


def test_cancellation_prediction_calls_logged_batch_service(monkeypatch):
    expected_result = {
        "batch_id": "api-batch",
        "prediction_timestamp": "2026-11-06T00:00:00+00:00",
        "prediction_count": 1,
        "predictions": [
            {
                "row_id": "api-batch:000000",
                "predicted_class": 1,
                "probability_positive": 0.75,
            }
        ],
        "model_version": "Model 1@2026-10-30",
        "model_artifact_sha256": "a" * 64,
        "quality_status": "HEALTHY",
        "prediction_file": "internal/path.jsonl",
    }
    received = {}

    def fake_log_batch_predictions(batch):
        received["batch"] = batch
        return expected_result

    monkeypatch.setattr(main, "log_batch_predictions", fake_log_batch_predictions)

    response = client.post(
        "/api/v1/predictions/cancellation",
        json={"batch_id": "api-batch", "records": [{"hotel": "City Hotel"}]},
    )

    assert response.status_code == 200
    assert response.json() == {
        key: value
        for key, value in expected_result.items()
        if key != "prediction_file"
    }
    assert isinstance(received["batch"], pd.DataFrame)
    assert received["batch"].to_dict(orient="records") == [
        {"batch_id": "api-batch", "hotel": "City Hotel"}
    ]


@pytest.mark.parametrize(
    ("path", "body"),
    [
        (
            "/api/v1/monitoring/observations",
            {"records": [{"hotel": "City Hotel"}]},
        ),
        (
            "/api/v1/monitoring/outcomes",
            {"observation_id": "observation-uuid", "is_canceled": 1},
        ),
    ],
)
def test_production_ingestion_routes_require_trusted_token(
    monkeypatch, path, body
):
    monkeypatch.delenv("PRODUCTION_INGESTION_TOKEN", raising=False)
    unconfigured = client.post(path, json=body)
    assert unconfigured.status_code == 503

    monkeypatch.setenv("PRODUCTION_INGESTION_TOKEN", "p" * 40)
    unauthorized = client.post(path, json=body)
    invalid_token = client.post(
        path,
        json=body,
        headers={"X-Production-Ingestion-Token": "x" * 40},
    )
    assert unauthorized.status_code == 401
    assert invalid_token.status_code == 401


def test_production_observation_api_rejects_synthetic_provenance(monkeypatch):
    monkeypatch.setenv("PRODUCTION_INGESTION_TOKEN", "p" * 40)

    response = client.post(
        "/api/v1/monitoring/observations",
        json={
            "batch_id": "development-simulation-batch-001",
            "data_source_type": "synthetic",
            "records": [
                {
                    "booking_id": "synthetic-booking-001",
                    "data_source_type": "synthetic",
                    "source": "synthetic_simulator",
                    "environment": "development",
                    "is_synthetic": True,
                }
            ],
        },
        headers={"X-Production-Ingestion-Token": "p" * 40},
    )

    assert response.status_code == 422


def test_ingestion_status_records_sanitized_auth_and_validation_events(monkeypatch):
    main.PRODUCTION_INGESTION_EVENT_HISTORY.clear()
    monkeypatch.setenv("PRODUCTION_INGESTION_TOKEN", "p" * 40)
    secret = "x" * 40

    rejected_auth = client.post(
        "/api/v1/monitoring/observations",
        json={"records": [{"hotel": "City Hotel"}]},
        headers={"X-Production-Ingestion-Token": secret},
    )
    rejected_validation = client.post(
        "/api/v1/monitoring/observations",
        json={"unexpected": "field"},
        headers={"X-Production-Ingestion-Token": "p" * 40},
    )
    status = client.get("/api/v1/monitoring/ingestion-status")

    assert rejected_auth.status_code == 401
    assert rejected_validation.status_code == 422
    assert status.status_code == 200
    result = status.json()
    assert result["configured"] is True
    assert result["retention_limit"] == 50
    assert [event["category"] for event in result["events"]] == [
        "validation_rejected",
        "authentication_rejected",
    ]
    assert all(
        {"event_id", "timestamp", "endpoint", "status_code", "category"}
        == set(event)
        for event in result["events"]
    )
    assert secret not in status.text
    assert "City Hotel" not in status.text


def test_production_observation_api_validates_features_and_runs_pipeline(
    monkeypatch, tmp_path
):
    main.PRODUCTION_INGESTION_EVENT_HISTORY.clear()
    monkeypatch.setenv("PRODUCTION_INGESTION_TOKEN", "p" * 40)
    monkeypatch.setattr(main, "PREDICTIONS_DIR", tmp_path / "predictions")
    feature_names = json.loads(main.FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
    record = {"booking_id": "hotel-booking-1001"}
    record.update({feature: 0 for feature in feature_names})
    record["hotel"] = "City Hotel"
    headers = {"X-Production-Ingestion-Token": "p" * 40}
    captured = {}

    def fake_log(batch, *, source):
        captured["batch"] = batch
        captured["source"] = source
        return {
            "prediction_timestamp": "2026-11-06T00:00:00+00:00",
            "prediction_count": 1,
            "observation_ids": ["observation-uuid"],
            "predictions": [
                {
                    "row_id": "observation-uuid",
                    "observation_id": "observation-uuid",
                    "predicted_class": 1,
                    "probability_positive": 0.75,
                }
            ],
            "model_version": "Model 1@test",
            "model_artifact_sha256": "a" * 64,
            "quality_status": "HEALTHY",
        }

    def fake_monitor(batch, *, data_source_type, source):
        captured["monitor_source"] = source
        captured["data_source_type"] = data_source_type
        return (
            {
                "data_source_type": data_source_type,
                "monitoring_timestamp": "2026-11-05T00:00:00+00:00",
            },
            tmp_path / "production-report.json",
        )

    monkeypatch.setattr(main, "log_batch_predictions", fake_log)
    monkeypatch.setattr(main, "monitor_batch_data", fake_monitor)
    monkeypatch.setattr(
        main,
        "evaluate_batch_performance",
        lambda batch_id: {"number_of_pending_outcomes": 1},
    )

    invalid = client.post(
        "/api/v1/monitoring/observations",
        json={
            "batch_id": "production-api-invalid",
            "data_source_type": "real",
            "records": [{"booking_id": "booking-1", "hotel": "City Hotel"}],
        },
        headers=headers,
    )
    assert invalid.status_code == 422

    kaggle_record = {**record, "is_canceled": 0}
    historical = client.post(
        "/api/v1/monitoring/observations",
        json={
            "batch_id": "production-api-historical-data",
            "data_source_type": "real",
            "records": [kaggle_record],
        },
        headers=headers,
    )
    duplicate_booking = client.post(
        "/api/v1/monitoring/observations",
        json={
            "batch_id": "production-api-duplicate-booking",
            "data_source_type": "real",
            "records": [record, record],
        },
        headers=headers,
    )
    assert historical.status_code == 422
    assert duplicate_booking.status_code == 422

    missing_provenance = client.post(
        "/api/v1/monitoring/observations",
        json={"batch_id": "production-api-no-provenance", "records": [record]},
        headers=headers,
    )
    assert missing_provenance.status_code == 422

    invalid_provenance = client.post(
        "/api/v1/monitoring/observations",
        json={
            "batch_id": "production-api-synthetic",
            "data_source_type": "synthetic_test",
            "records": [record],
        },
        headers=headers,
    )
    assert invalid_provenance.status_code == 422

    response = client.post(
        "/api/v1/monitoring/observations",
        json={
            "batch_id": "production-api-valid",
            "data_source_type": "real",
            "records": [record],
        },
        headers=headers,
    )

    assert response.status_code == 200
    result = response.json()
    assert result["source"] == "production"
    assert result["data_source_type"] == "real"
    assert result["prediction_timestamp"] == "2026-11-06T00:00:00+00:00"
    assert result["observation_ids"] == ["observation-uuid"]
    assert result["number_of_pending_outcomes"] == 1
    assert captured["source"] == captured["monitor_source"] == "production"
    assert captured["data_source_type"] == "real"
    assert captured["batch"]["batch_id"].tolist() == ["production-api-valid"]
    assert captured["batch"]["row_id"].str.fullmatch(r"[0-9a-f]{32}").all()
    assert captured["batch"]["source_booking_id_hmac"].str.fullmatch(
        r"[0-9a-f]{64}"
    ).all()
    assert "hotel-booking-1001" not in response.text
    ingestion_status = client.get("/api/v1/monitoring/ingestion-status")
    assert ingestion_status.status_code == 200
    assert ingestion_status.json()["events"][0]["category"] == "accepted"
    assert ingestion_status.json()["events"][0]["endpoint"] == "observations"

    synthetic_id = client.post(
        "/api/v1/monitoring/observations",
        json={
            "batch_id": "normal-2026-seed-42",
            "data_source_type": "synthetic_test",
            "records": [record],
        },
        headers=headers,
    )
    assert synthetic_id.status_code == 422

    prediction_path = tmp_path / "predictions" / "production-api-valid.jsonl"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text("{}\n", encoding="utf-8")
    duplicate_batch = client.post(
        "/api/v1/monitoring/observations",
        json={
            "batch_id": "production-api-valid",
            "data_source_type": "real",
            "records": [record],
        },
        headers=headers,
    )
    assert duplicate_batch.status_code == 409


def test_failed_production_monitoring_removes_new_prediction_file(
    monkeypatch, tmp_path
):
    token = "p" * 40
    monkeypatch.setenv("PRODUCTION_INGESTION_TOKEN", token)
    monkeypatch.setattr(main, "PREDICTIONS_DIR", tmp_path / "predictions")
    feature_names = json.loads(main.FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
    record = {"booking_id": "isolated-test-booking"}
    record.update({feature: 0 for feature in feature_names})
    record["hotel"] = "City Hotel"

    def write_predictions(batch, *, source):
        assert source == "production"
        prediction_path = (
            main.PREDICTIONS_DIR / f"{batch['batch_id'].iloc[0]}.jsonl"
        )
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text('{"test_only": true}\n', encoding="utf-8")
        return {"prediction_timestamp": "2026-11-06T00:00:00+00:00"}

    def fail_monitoring(*_args, **_kwargs):
        raise FileNotFoundError("isolated monitoring failure")

    monkeypatch.setattr(main, "log_batch_predictions", write_predictions)
    monkeypatch.setattr(main, "monitor_batch_data", fail_monitoring)
    monkeypatch.setattr(
        main,
        "evaluate_batch_performance",
        lambda _batch_id: pytest.fail("Performance evaluation must not run."),
    )

    response = client.post(
        "/api/v1/monitoring/observations",
        json={
            "batch_id": "isolated-monitoring-failure",
            "data_source_type": "real",
            "records": [record],
        },
        headers={"X-Production-Ingestion-Token": token},
    )

    assert response.status_code == 503
    assert not (
        main.PREDICTIONS_DIR / "isolated-monitoring-failure.jsonl"
    ).exists()


def test_production_outcome_api_validates_and_reports_duplicate(monkeypatch):
    from ml.performance_tracking import DuplicateOutcomeError

    monkeypatch.setenv("PRODUCTION_INGESTION_TOKEN", "p" * 40)
    monkeypatch.setattr(
        main,
        "record_production_outcome",
        lambda observation_id, is_canceled: {
            "observation_id": observation_id,
            "batch_id": "production-batch",
            "outcome": {"outcome_received_at": "2026-11-05T00:00:00+00:00"},
            "performance": {"f1_score": 0.8},
        },
    )
    training_calls = []

    def fake_training():
        training_calls.append(True)
        return {
            "model2_status": "NOT_TRAINED",
            "training_eligibility": "NOT_TRAINED",
            "reason": "Need at least 20 labeled production observations; found 0.",
        }
    monkeypatch.setattr(main, "run_training_if_eligible", fake_training)
    headers = {"X-Production-Ingestion-Token": "p" * 40}
    response = client.post(
        "/api/v1/monitoring/outcomes",
        json={"observation_id": "observation-uuid", "is_canceled": 1},
        headers=headers,
    )
    invalid = client.post(
        "/api/v1/monitoring/outcomes",
        json={"observation_id": "observation-uuid", "is_canceled": 2},
        headers=headers,
    )
    assert response.status_code == 200
    assert "p" * 40 not in response.text
    assert response.json()["performance"]["f1_score"] == 0.8
    assert response.json()["model2_training"]["model2_status"] == "NOT_TRAINED"
    assert len(training_calls) == 1
    assert invalid.status_code == 422

    def duplicate_outcome(observation_id, _is_canceled):
        raise DuplicateOutcomeError(
            f"Outcome for observation {observation_id} was already recorded."
        )

    monkeypatch.setattr(main, "record_production_outcome", duplicate_outcome)
    duplicate = client.post(
        "/api/v1/monitoring/outcomes",
        json={"observation_id": "observation-uuid", "is_canceled": 1},
        headers=headers,
    )
    assert duplicate.status_code == 409
    assert len(training_calls) == 1

    def unmatched_observation(observation_id, _is_canceled):
        raise FileNotFoundError(
            f"No production prediction matches observation {observation_id}."
        )

    monkeypatch.setattr(main, "record_production_outcome", unmatched_observation)
    unmatched = client.post(
        "/api/v1/monitoring/outcomes",
        json={"observation_id": "unmatched-observation", "is_canceled": 1},
        headers=headers,
    )
    assert unmatched.status_code == 404
    assert len(training_calls) == 1


def test_production_collection_to_complete_future_f1_label(
    monkeypatch, tmp_path
):
    token = "p" * 40
    headers = {"X-Production-Ingestion-Token": token}
    directories = {
        name: tmp_path / name
        for name in ("reports", "predictions", "outcomes", "performance")
    }
    for directory in directories.values():
        directory.mkdir()
    monkeypatch.setenv("PRODUCTION_INGESTION_TOKEN", token)
    monkeypatch.setattr(main, "PREDICTIONS_DIR", directories["predictions"])

    model_version = "integration-test-model1"
    baseline_time = datetime.now(timezone.utc) - timedelta(days=2)

    def write_report(
        batch_id: str,
        timestamp: datetime,
        *,
        data_source_type: str = "real",
        source: str = "production",
    ):
        report = {
            "batch_id": batch_id,
            "monitoring_timestamp": timestamp.isoformat(),
            "data_source_type": data_source_type,
            "source": source,
            "quality": {},
            "drift": {},
        }
        (directories["reports"] / f"{batch_id}.json").write_text(
            json.dumps(report), encoding="utf-8"
        )

    def write_performance(
        batch_id: str,
        timestamp: datetime,
        *,
        source: str,
        data_source_type: str,
        f1: float,
    ):
        prediction_path = directories["predictions"] / f"{batch_id}.jsonl"
        observation_ids = (
            [
                json.loads(line)["observation_id"]
                for line in prediction_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if prediction_path.is_file()
            else []
        )
        record = {
            "batch_id": batch_id,
            "evaluation_timestamp": timestamp.isoformat(),
            "model_version": model_version,
            "number_of_predictions": 2,
            "number_of_actual_outcomes_received": 2,
            "number_of_evaluated_records": 2,
            "number_of_pending_outcomes": 0,
            "metrics_available": True,
            "actual_outcome_class_counts": {"0": 1, "1": 1},
            "data_source_type": data_source_type,
            "source": source,
            "outcome_provenance": "verified_production_api",
            "contributing_observation_ids": observation_ids,
            "latest_outcome_received_at": timestamp.isoformat(),
            "accuracy": f1,
            "precision": f1,
            "recall": f1,
            "f1_score": f1,
        }
        (directories["performance"] / f"{batch_id}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )

    baseline_batch = "test-only-baseline"
    write_report(baseline_batch, baseline_time)
    (directories["predictions"] / f"{baseline_batch}.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "batch_id": baseline_batch,
                    "row_id": f"baseline-row-{index}",
                    "observation_id": f"baseline-observation-{index}",
                    "source": "production",
                    "predicted_at": baseline_time.isoformat(),
                    "predicted_class": index,
                    "probability_positive": float(index),
                    "model_version": model_version,
                }
            )
            + "\n"
            for index in range(2)
        ),
        encoding="utf-8",
    )
    write_performance(
        baseline_batch,
        baseline_time + timedelta(hours=1),
        source="production",
        data_source_type="real",
        f1=0.9,
    )

    feature_names = json.loads(main.FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
    records = [
        {"booking_id": f"hotel-booking-{index}"}
        for index in range(2)
    ]
    for record in records:
        record.update({name: 0 for name in feature_names})
        record["hotel"] = "City Hotel"

    def fake_log_predictions(batch, *, source):
        assert source == "production"
        batch_id = batch["batch_id"].iloc[0]
        predicted_at = datetime.now(timezone.utc).isoformat()
        predictions = [
            {
                "batch_id": batch_id,
                "row_id": f"{batch_id}:{index}",
                "observation_id": f"{batch_id}:{index}",
                "source": "production",
                "predicted_at": predicted_at,
                "predicted_class": 1,
                "probability_positive": 0.9,
                "model_version": model_version,
            }
            for index in range(len(batch))
        ]
        prediction_path = directories["predictions"] / f"{batch_id}.jsonl"
        prediction_path.write_text(
            "".join(json.dumps(item) + "\n" for item in predictions),
            encoding="utf-8",
        )
        return {
            "prediction_timestamp": predicted_at,
            "prediction_count": len(predictions),
            "observation_ids": [
                item["observation_id"] for item in predictions
            ],
            "predictions": [
                {
                    "row_id": item["row_id"],
                    "observation_id": item["observation_id"],
                    "predicted_class": item["predicted_class"],
                    "probability_positive": item["probability_positive"],
                }
                for item in predictions
            ],
            "model_version": model_version,
            "model_artifact_sha256": "test-only",
            "quality_status": "HEALTHY",
        }

    def fake_monitor(batch, *, data_source_type, source):
        batch_id = batch["batch_id"].iloc[0]
        timestamp = datetime.now(timezone.utc)
        report = {
            "batch_id": batch_id,
            "monitoring_timestamp": timestamp.isoformat(),
            "data_source_type": data_source_type,
            "source": source,
            "quality": {},
            "drift": {},
        }
        path = directories["reports"] / f"{batch_id}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return report, path

    monkeypatch.setattr(main, "log_batch_predictions", fake_log_predictions)
    monkeypatch.setattr(main, "monitor_batch_data", fake_monitor)
    monkeypatch.setattr(
        main,
        "evaluate_batch_performance",
        lambda batch_id: evaluate_batch_performance_with_paths(
            batch_id,
            predictions_dir=directories["predictions"],
            outcomes_dir=directories["outcomes"],
            performance_dir=directories["performance"],
            reports_dir=directories["reports"],
        ),
    )
    monkeypatch.setattr(
        main,
        "record_production_outcome",
        lambda observation_id, is_canceled: record_production_outcome_with_paths(
            observation_id,
            is_canceled,
            predictions_dir=directories["predictions"],
            outcomes_dir=directories["outcomes"],
            performance_dir=directories["performance"],
            reports_dir=directories["reports"],
        ),
    )
    monkeypatch.setattr(
        main,
        "run_training_if_eligible",
        lambda: {"model2_status": "NOT_TRAINED", "training_performed": False},
    )

    batch_id = "integration-test-production-batch"
    response = client.post(
        "/api/v1/monitoring/observations",
        json={
            "batch_id": batch_id,
            "data_source_type": "real",
            "records": records,
        },
        headers=headers,
    )
    assert response.status_code == 200
    observation_ids = response.json()["observation_ids"]
    assert len(observation_ids) == 2
    assert response.json()["number_of_pending_outcomes"] == 2

    def current_dataset():
        return build_historical_dataset(
            reports_dir=directories["reports"],
            predictions_dir=directories["predictions"],
            performance_dir=directories["performance"],
        ).set_index("batch_id")

    assert pd.isna(current_dataset().loc[batch_id, DEGRADATION_TARGET])

    partial = client.post(
        "/api/v1/monitoring/outcomes",
        json={"observation_id": observation_ids[0], "is_canceled": 0},
        headers=headers,
    )
    assert partial.status_code == 200
    assert partial.json()["performance"]["number_of_actual_outcomes_received"] == 1
    assert partial.json()["performance"]["number_of_pending_outcomes"] == 1
    assert pd.isna(current_dataset().loc[batch_id, DEGRADATION_TARGET])

    complete = client.post(
        "/api/v1/monitoring/outcomes",
        json={"observation_id": observation_ids[1], "is_canceled": 1},
        headers=headers,
    )
    assert complete.status_code == 200
    performance = complete.json()["performance"]
    assert performance["number_of_predictions"] == 2
    assert performance["number_of_actual_outcomes_received"] == 2
    assert performance["number_of_evaluated_records"] == 2
    assert performance["number_of_pending_outcomes"] == 0

    dataset = current_dataset()
    assert dataset.loc[batch_id, DEGRADATION_TARGET] == 1
    assert dataset.loc[batch_id, "target_baseline_f1"] == 0.9
    assert dataset.loc[batch_id, "target_future_f1"] == pytest.approx(0.6666666666666666)

    synthetic_batch = "normal-20261117T000000Z-seed-42"
    unknown_batch = "test-only-unknown-source"
    for source_batch, data_source_type, source in (
        (synthetic_batch, "synthetic_test", "synthetic_test"),
        (unknown_batch, "unknown", "unknown"),
    ):
        timestamp = baseline_time + timedelta(minutes=1)
        write_report(
            source_batch,
            timestamp,
            data_source_type=data_source_type,
            source=source,
        )
        write_performance(
            source_batch,
            timestamp + timedelta(minutes=1),
            source=source,
            data_source_type=data_source_type,
            f1=0.0,
        )

    dataset = current_dataset()
    assert dataset.loc[synthetic_batch, "source"] == "synthetic_test"
    assert pd.isna(dataset.loc[synthetic_batch, DEGRADATION_TARGET])
    assert dataset.loc[unknown_batch, "source"] == "unknown"
    assert pd.isna(dataset.loc[unknown_batch, DEGRADATION_TARGET])


def test_production_observation_performance_and_model2_status_apis(monkeypatch):
    observation = {
        "observation_id": "production-observation-1",
        "batch_id": "production-batch",
        "timestamp": "2026-11-05T00:00:00+00:00",
        "source": "production",
        "model_version": "Model 1@test",
        "predicted_class": 1,
        "probability_positive": 0.75,
        "actual_outcome": None,
        "outcome_received_at": None,
    }
    performance = {
        "batch_id": "production-batch",
        "evaluation_timestamp": "2026-11-05T00:00:00+00:00",
        "number_of_actual_outcomes_received": 1,
        "number_of_pending_outcomes": 0,
        "accuracy": 1.0,
    }
    monkeypatch.setattr(
        main,
        "list_production_observations",
        lambda **kwargs: [observation],
    )
    monkeypatch.setattr(
        main,
        "list_production_performance",
        lambda **kwargs: [performance],
    )
    monkeypatch.setattr(
        main,
        "_model2_summary",
        lambda: {
            "status": "NOT_TRAINED",
            "training_eligibility": "NOT_TRAINED",
            "production_batch_count": 1,
            "labeled_batch_count": 0,
            "unlabeled_batch_count": 1,
            "labeled_observation_count": 0,
            "minimum_labeled_observations": 20,
            "training_progress_percent": 0,
            "latest_training_result": None,
        },
    )

    observations_response = client.get("/api/v1/monitoring/observations")
    performance_response = client.get("/api/v1/monitoring/performance")
    status_response = client.get("/api/v1/model2/training-status")

    assert observations_response.status_code == 200
    assert observations_response.json() == {
        "count": 1,
        "observations": [observation],
    }
    assert performance_response.status_code == 200
    assert performance_response.json() == {
        "count": 1,
        "performance": [performance],
    }
    assert status_response.status_code == 200
    assert status_response.json()["minimum_labeled_observations"] == 20
    assert status_response.json()["training_eligibility"] == "NOT_TRAINED"
    assert status_response.json()["production_batch_count"] == 1
    assert status_response.json()["labeled_batch_count"] == 0
    assert status_response.json()["unlabeled_batch_count"] == 1


def test_cancellation_prediction_runs_model_and_persists_results(
    monkeypatch, tmp_path
):
    source_path = next((ROOT / "data" / "monitoring").glob("normal_*.csv"))
    source_record = pd.read_csv(source_path, nrows=1).drop(
        columns=["batch_id", "generated_at", "row_id"], errors="ignore"
    ).iloc[0]
    record = json.loads(source_record.to_json())
    prediction_dir = tmp_path / "predictions"

    def log_to_temp_dir(batch):
        return log_batch_predictions_with_paths(
            batch,
            predictions_dir=prediction_dir,
            reports_dir=tmp_path / "reports",
        )

    monkeypatch.setattr(main, "log_batch_predictions", log_to_temp_dir)

    response = client.post(
        "/api/v1/predictions/cancellation",
        json={"batch_id": "api-integration-batch", "records": [record]},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["batch_id"] == "api-integration-batch"
    assert result["prediction_count"] == 1
    assert len(result["predictions"]) == 1
    assert result["predictions"][0]["predicted_class"] in (0, 1)
    assert 0 <= result["predictions"][0]["probability_positive"] <= 1
    assert list(prediction_dir.glob("*.jsonl"))


def test_cancellation_prediction_rejects_empty_or_extra_request_fields():
    empty_records = client.post(
        "/api/v1/predictions/cancellation",
        json={"batch_id": "api-batch", "records": []},
    )
    extra_field = client.post(
        "/api/v1/predictions/cancellation",
        json={
            "batch_id": "api-batch",
            "records": [{"hotel": "City Hotel"}],
            "feature": "not-supported",
        },
    )
    record_batch_id = client.post(
        "/api/v1/predictions/cancellation",
        json={
            "batch_id": "api-batch",
            "records": [{"batch_id": "different-batch", "hotel": "City Hotel"}],
        },
    )

    assert empty_records.status_code == 422
    assert extra_field.status_code == 422
    assert record_batch_id.status_code == 422


def test_cancellation_prediction_csv_runs_and_persists_model_predictions(
    monkeypatch, tmp_path
):
    source_path = next((ROOT / "data" / "monitoring").glob("normal_*.csv"))
    batch = pd.read_csv(source_path, nrows=2).drop(
        columns=["batch_id", "generated_at", "row_id"], errors="ignore"
    )
    csv_content = batch.to_csv(index=False).encode("utf-8")
    prediction_dir = tmp_path / "predictions"

    def log_to_temp_dir(incoming):
        return log_batch_predictions_with_paths(
            incoming,
            predictions_dir=prediction_dir,
            reports_dir=tmp_path / "reports",
        )

    monkeypatch.setattr(main, "log_batch_predictions", log_to_temp_dir)
    response = client.post(
        "/api/v1/predictions/cancellation/csv?batch_id=csv-api-integration",
        content=csv_content,
        headers={"Content-Type": "text/csv"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["batch_id"] == "csv-api-integration"
    assert result["prediction_count"] == 2
    assert len(result["predictions"]) == 2
    assert list(prediction_dir.glob("*.jsonl"))


def test_batch_id_validation_rejects_traversal_and_excessive_length():
    traversal = client.post(
        "/api/v1/predictions/cancellation/csv?batch_id=..%2F..%2Foutside",
        content=b"hotel\nCity Hotel\n",
        headers={"Content-Type": "text/csv"},
    )
    overlong = client.post(
        f"/api/v1/predictions/cancellation/csv?batch_id={'a' * 129}",
        content=b"hotel\nCity Hotel\n",
        headers={"Content-Type": "text/csv"},
    )

    assert traversal.status_code == 422
    assert overlong.status_code == 422


def test_csv_upload_size_is_enforced_while_streaming(monkeypatch):
    monkeypatch.setattr(main, "MAX_OUTCOME_CSV_BYTES", 16)

    response = client.post(
        "/api/v1/outcomes/any-batch/csv",
        content=b"row_id,is_canceled\n" + b"x" * 50,
        headers={"Content-Type": "text/csv"},
    )

    assert response.status_code == 413


def test_actual_outcome_csv_import_matches_predictions_and_rejects_duplicates(
    monkeypatch, tmp_path
):
    source_path = next((ROOT / "data" / "monitoring").glob("normal_*.csv"))
    batch = pd.read_csv(source_path, nrows=2).drop(
        columns=["batch_id", "generated_at", "row_id"], errors="ignore"
    )
    batch["row_id"] = ["booking-1001", "booking-1002"]
    batch["batch_id"] = "api-real-outcome-batch"
    paths = {
        "predictions_dir": tmp_path / "predictions",
        "outcomes_dir": tmp_path / "outcomes",
        "performance_dir": tmp_path / "performance",
        "reports_dir": tmp_path / "reports",
    }
    prediction_result = log_batch_predictions_with_paths(
        batch,
        predictions_dir=paths["predictions_dir"],
        reports_dir=paths["reports_dir"],
    )
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    (paths["reports_dir"] / "real-monitoring.json").write_text(
        json.dumps(
            {
                "batch_id": prediction_result["batch_id"],
                "data_source_type": "real",
                "monitoring_timestamp": "2026-11-04T10:00:00+00:00",
                "overall_status": "HEALTHY",
                "quality": {"quality_status": "HEALTHY"},
                "drift": {"numerical": {}, "categorical": {}},
            }
        ),
        encoding="utf-8",
    )

    def import_to_temp_paths(batch_id, csv_content):
        from ml.performance_tracking import import_actual_outcomes_csv

        return import_actual_outcomes_csv(batch_id, csv_content, **paths)

    monkeypatch.setattr(main, "import_actual_outcomes_csv", import_to_temp_paths)
    csv_content = b"row_id,is_canceled\nbooking-1001,0\nbooking-1002,1\n"

    response = client.post(
        "/api/v1/outcomes/api-real-outcome-batch/csv",
        content=csv_content,
        headers={"Content-Type": "text/csv"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["batch_id"] == "api-real-outcome-batch"
    assert result["outcome_count"] == 2
    assert result["number_of_evaluated_records"] == 2
    assert result["number_of_pending_outcomes"] == 0
    assert result["metrics_available"] is True
    assert result["accuracy"] is not None
    assert result["precision"] is not None
    assert result["recall"] is not None
    assert result["f1_score"] is not None
    assert result["confusion_matrix"] is not None
    assert result["data_source_type"] == "real"

    duplicate = client.post(
        "/api/v1/outcomes/api-real-outcome-batch/csv",
        content=csv_content,
        headers={"Content-Type": "text/csv"},
    )
    invalid = client.post(
        "/api/v1/outcomes/api-real-outcome-batch/csv",
        content=b"row_id,is_canceled\n,1\n",
        headers={"Content-Type": "text/csv"},
    )
    wrong_content_type = client.post(
        "/api/v1/outcomes/api-real-outcome-batch/csv",
        content=csv_content,
        headers={"Content-Type": "application/json"},
    )

    assert duplicate.status_code == 409
    assert invalid.status_code == 422
    assert wrong_content_type.status_code == 415


def test_actual_outcome_import_requires_real_monitoring_source(monkeypatch):
    def reject_unverified_batch(_batch_id, _csv_content):
        raise ValueError(
            "Actual outcomes require a monitoring report explicitly marked as real."
        )

    monkeypatch.setattr(main, "import_actual_outcomes_csv", reject_unverified_batch)
    response = client.post(
        "/api/v1/outcomes/simulated-batch/csv",
        content=b"row_id,is_canceled\nrow-1,1\n",
        headers={"Content-Type": "text/csv"},
    )

    assert response.status_code == 422
    assert "explicitly marked as real" in response.json()["detail"]


def test_cancellation_prediction_maps_quality_and_duplicate_batch_errors(
    monkeypatch,
):
    def reject_quality(_batch):
        raise PoorQualityBatchError(
            {
                "batch_id": "api-batch",
                "quality_status": "CRITICAL",
                "critical_reasons": ["Too many invalid values."],
            }
        )

    monkeypatch.setattr(main, "log_batch_predictions", reject_quality)
    payload = {"batch_id": "api-batch", "records": [{"hotel": "City Hotel"}]}

    poor_quality = client.post("/api/v1/predictions/cancellation", json=payload)

    assert poor_quality.status_code == 422
    assert poor_quality.json()["detail"]["code"] == "poor_quality_batch"
    assert poor_quality.json()["detail"]["quality_report"]["quality_status"] == "CRITICAL"

    def reject_duplicate(_batch):
        raise FileExistsError("Predictions already exist for batch 'api-batch'.")

    monkeypatch.setattr(main, "log_batch_predictions", reject_duplicate)
    duplicate = client.post("/api/v1/predictions/cancellation", json=payload)

    assert duplicate.status_code == 409
    assert "already exist" in duplicate.json()["detail"]


@pytest.mark.parametrize(
    ("exception", "status_code"),
    [
        (ValueError("Invalid feature schema."), 422),
        (FileNotFoundError("missing model artifact"), 503),
    ],
)
def test_cancellation_prediction_surfaces_service_validation_errors(
    monkeypatch, exception, status_code
):
    def fail(_batch):
        raise exception

    monkeypatch.setattr(main, "log_batch_predictions", fail)

    response = client.post(
        "/api/v1/predictions/cancellation",
        json={"batch_id": "api-batch", "records": [{"hotel": "City Hotel"}]},
    )

    assert response.status_code == status_code
