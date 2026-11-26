import json
from pathlib import Path

import pandas as pd

from backend.main import app
from fastapi.testclient import TestClient
from ml.model2_config import MODEL2_FEATURE_NAMES

client = TestClient(app)
assert client.post(
    "/api/v1/auth/login",
    json={
        "username": "tests@example.com",
        "password": "test-only-password-123",
    },
).status_code == 200


def test_health_endpoint_reports_truthful_readiness():
    response = client.get("/health")

    assert response.status_code == 200
    result = response.json()
    assert result["application_status"] == "OK"
    assert result["model1_readiness"] == "READY"
    assert result["model2_readiness"] == "NOT_TRAINED"
    assert result["schema_readiness"] == "READY"
    assert result["application_status"] != "UNHEALTHY"
    assert isinstance(result["production_ingestion_configured"], bool)
    assert "PRODUCTION_INGESTION_TOKEN" not in response.text


def test_health_endpoint_reports_only_production_token_configuration(monkeypatch):
    import backend.main as main

    monkeypatch.delenv("PRODUCTION_INGESTION_TOKEN", raising=False)
    assert main.get_health()["production_ingestion_configured"] is False
    monkeypatch.setenv("PRODUCTION_INGESTION_TOKEN", "s" * 31)
    assert main.get_health()["production_ingestion_configured"] is False
    monkeypatch.setenv("PRODUCTION_INGESTION_TOKEN", "s" * 32)
    assert main.get_health()["production_ingestion_configured"] is True
    assert "s" * 32 not in str(main.get_health())


def test_dashboard_returns_model_evaluation_and_monitoring_data(
    tmp_path, monkeypatch
):
    import backend.dashboard as dashboard

    reports_dir = tmp_path / "reports"
    predictions_dir = tmp_path / "predictions"
    performance_dir = tmp_path / "performance"
    reports_dir.mkdir()
    predictions_dir.mkdir()
    performance_dir.mkdir()
    monkeypatch.setattr(dashboard, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(dashboard, "PREDICTIONS_DIR", predictions_dir)
    monkeypatch.setattr(dashboard, "PERFORMANCE_DIR", performance_dir)
    monkeypatch.setattr(
        dashboard,
        "DEVELOPMENT_SIMULATION_DIR",
        tmp_path / "development-simulation-runs",
    )
    monkeypatch.setattr(
        dashboard,
        "HISTORICAL_REPLAY_DIR",
        tmp_path / "historical-replay-runs",
    )
    monkeypatch.setattr(dashboard, "ALERTS_PATH", tmp_path / "alerts.jsonl")
    monkeypatch.setattr(
        dashboard,
        "MODEL2_DEVELOPMENT_IMPORTANCE_PATH",
        tmp_path / "missing-feature-importance.json",
    )
    offline_status = {
        "status": "NOT_TRAINED",
        "model_scope": "HISTORICAL_OFFLINE",
        "provenance": "HISTORICAL_OFFLINE",
        "historical_observation_count": 0,
        "eligible_observation_count": 0,
        "target_class_counts": {},
        "training_size": 0,
        "testing_size": 0,
        "evaluation_metrics": None,
        "confusion_matrix": None,
        "evaluation_predictions": [],
        "observation_history": [],
        "missing_requirements": ["Offline artifact is not present."],
        "reason": "Offline artifact is not present.",
        "model_version": None,
        "training_timestamp": None,
        "target_definition": None,
    }
    monkeypatch.setattr(
        dashboard,
        "offline_model2_status",
        lambda: offline_status,
    )
    monkeypatch.setattr(dashboard, "list_production_observations", lambda **kwargs: [])

    response = client.get("/api/v1/dashboard")

    assert response.status_code == 200
    result = response.json()
    assert result["model"]["status"] == "READY"
    assert result["system"]["application_status"] == "OK"
    assert result["system"]["model1_readiness"] == "READY"
    assert result["system"]["schema_readiness"] == "READY"
    assert isinstance(result["system"]["production_ingestion_configured"], bool)
    assert result["model"]["dataset_row_count"] == 119390
    assert result["model"]["dataset_period"] == "Jul 2015–Aug 2017"
    assert result["metrics"]["sample_count"] == 23878
    assert result["metrics"]["accuracy"] > 0.8
    assert result["metrics"]["f1_score"] > 0.8
    assert result["metrics"]["roc_auc"] > 0.94
    assert len(result["metrics"]["roc_curve"]) == 51
    assert result["metrics"]["roc_curve"][0] == {
        "false_positive_rate": 0.0,
        "true_positive_rate": 0.0,
    }
    assert result["metrics"]["roc_curve"][-1] == {
        "false_positive_rate": 1.0,
        "true_positive_rate": 1.0,
    }
    assert len(result["model"]["feature_importance"]) == 5
    assert result["model"]["feature_importance"][0]["name"] == "lead_time"
    history = result["metrics"]["performance_history"]
    assert len(history) >= 20
    assert history[0]["period"] == "Jul 2015"
    assert history[-1]["period"] == "Aug 2017"
    assert all(item["sample_count"] > 0 for item in history)
    assert result["predictions"]["total"] == 0
    assert result["predictions"]["distribution_sample_count"] == 0
    assert result["latest_monitoring"] is None
    assert result["drift"]["features"] == []
    assert result["monitoring"]["report_count"] == 0
    assert result["monitoring"]["simulation_report_count"] == 0
    assert result["system"]["live_outcomes"] == "AWAITING"
    assert result["model2"]["status"] == "NOT_TRAINED"
    assert result["model2"]["labeled_observation_count"] == 0
    assert result["model2"]["minimum_labeled_observations"] == 20
    assert result["model2"]["training_eligibility"] == "NOT_TRAINED"
    assert result["model2"]["reason"]
    assert result["development_simulation"]["status"] == "NOT_RUN"
    assert result["development_simulation"]["is_synthetic"] is True
    assert result["development_simulation"]["mode"] == "DEVELOPMENT SIMULATION"
    assert result["historical_replay"]["status"] == "NOT_RUN"
    assert result["historical_replay"]["provenance"] == "HISTORICAL_OFFLINE"
    assert result["model2_offline"] == offline_status
    development = result["model2_development"]
    development_metrics_path = (
        dashboard.MODEL2_DEVELOPMENT_DIR
        / "model2-development-random-forest-seed-42-metrics.json"
    )
    saved_development_metrics = json.loads(
        development_metrics_path.read_text(encoding="utf-8")
    )
    assert development["status"] == "AVAILABLE"
    assert development["training_status"] == "DEVELOPMENT_TRAINED"
    assert development["provenance"] == "SYNTHETIC_DEVELOPMENT_ONLY"
    assert development["evaluation_metrics"]["accuracy"] == (
        saved_development_metrics["evaluation_metrics"]["accuracy"]
    )
    assert development["evaluation_metrics"]["f1_score"] == (
        saved_development_metrics["evaluation_metrics"]["f1_score"]
    )
    assert result["model2_development_explainability"]["status"] == "UNAVAILABLE"
    assert result["model2_development_explainability"]["features"] == []


def test_dashboard_only_displays_completed_tagged_simulation_runs(tmp_path, monkeypatch):
    import backend.dashboard as dashboard

    simulation_dir = tmp_path / "simulation-runs"
    run_dir = simulation_dir / "simulation-test"
    run_dir.mkdir(parents=True)
    summary = {
        "data_source_type": "synthetic",
        "source": "synthetic_simulator",
        "environment": "development",
        "is_synthetic": True,
        "mode": "DEVELOPMENT SIMULATION",
        "run_status": "COMPLETE",
        "run_id": "simulation-test",
        "batch_count": 24,
        "booking_count": 3840,
        "seed": 42,
        "baseline_f1_score": 0.9,
        "minimum_batch_f1_score": 0.6,
        "drift_detected_batch_count": 24,
        "monitoring_status_counts": {"HEALTHY": 0, "WARNING": 0, "CRITICAL": 24},
        "model2_training": {
            "status": "DEVELOPMENT_SIMULATION_TRAINED",
            "model_version": "model2-development-simulation-test",
            "training_size": 18,
            "testing_size": 6,
            "split_method": "chronological holdout",
            "training_class_counts": {"0": 11, "1": 7},
            "testing_class_counts": {"0": 3, "1": 3},
            "evaluation_metrics": {
                "accuracy": 0.8333333333333334,
                "precision": 0.75,
                "recall": 1.0,
                "f1_score": 0.8571428571428571,
            },
        },
    }
    summary_path = run_dir / "simulation_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(
        dashboard, "DEVELOPMENT_SIMULATION_DIR", simulation_dir
    )

    result = dashboard._development_simulation_summary()

    assert result["status"] == "AVAILABLE"
    assert result["run_id"] == "simulation-test"
    assert result["is_synthetic"] is True
    assert result["model2_training"]["split_method"] == "chronological holdout"

    response = client.get("/api/v1/dashboard")
    assert response.status_code == 200
    api_simulation = response.json()["development_simulation"]
    assert api_simulation["status"] == "AVAILABLE"
    assert api_simulation["batch_count"] == 24
    assert api_simulation["booking_count"] == 3840
    assert api_simulation["model2_training"]["training_size"] == 18
    assert api_simulation["model2_training"]["testing_size"] == 6
    assert api_simulation["source"] == "synthetic_simulator"

    summary["source"] = "production"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    rejected = dashboard._development_simulation_summary()
    assert rejected["status"] == "UNAVAILABLE"
    assert rejected["run_id"] is None


def test_dashboard_loads_only_valid_saved_historical_replay_summaries(
    tmp_path, monkeypatch
):
    import backend.dashboard as dashboard

    replay_dir = tmp_path / "historical-replay-runs"
    run_dir = replay_dir / "replay-test"
    run_dir.mkdir(parents=True)
    summary = {
        "status": "COMPLETE",
        "model_scope": "HISTORICAL_OFFLINE",
        "provenance": "HISTORICAL_OFFLINE",
        "data_source_type": "historical_offline",
        "source": "historical_hotel_booking_dataset",
        "environment": "offline",
        "is_synthetic": False,
        "run_id": "replay-test",
        "historical_record_count": 119390,
        "arrival_date_range": ["2015-07-01", "2017-08-31"],
        "booking_created_timestamp_available": False,
        "prediction_scope": "retrospective_historical_feature_replay_not_booking_time",
        "model1_metrics": {
            "accuracy": 0.974,
            "precision": 0.974,
            "recall": 0.955,
            "f1_score": 0.965,
        },
        "model1_evaluation_window_count": 114,
        "model2_eligibility": {
            "status": "NOT_ELIGIBLE",
            "eligible_observation_count": 109,
            "target_class_counts": {"0.0": 106, "1.0": 3},
            "training_size_after_purge": 86,
            "testing_size": 22,
            "training_class_counts": {"0": 83, "1": 3},
            "testing_class_counts": {"0": 22},
            "missing_requirements": [
                "Chronological Model 2 holdout windows lack both target classes."
            ],
            "reason": "Chronological Model 2 holdout windows lack both target classes.",
        },
        "model2_training": {"status": "NOT_ELIGIBLE"},
        "limitations": [
            "No booking-created timestamp is available.",
            "Historical offline evidence cannot satisfy production eligibility.",
        ],
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(dashboard, "HISTORICAL_REPLAY_DIR", replay_dir)

    result = dashboard._historical_replay_summary()

    assert result["status"] == "AVAILABLE"
    assert result["run_id"] == "replay-test"
    assert result["historical_record_count"] == 119390
    assert result["arrival_window_count"] == 114
    assert result["model1_metrics"]["accuracy"] == 0.974
    assert result["model2_eligibility"]["eligible_observation_count"] == 109
    assert result["model2_eligibility"]["testing_class_counts"] == {"0": 22}
    assert result["model2_eligibility"]["status"] == "NOT_ELIGIBLE"
    assert result["is_synthetic"] is False

    summary["is_synthetic"] = True
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    rejected = dashboard._historical_replay_summary()
    assert rejected["status"] == "UNAVAILABLE"
    assert rejected["model1_metrics"] is None


def test_dashboard_only_includes_explicitly_real_monitoring_and_prediction_data(
    tmp_path, monkeypatch
):
    import backend.dashboard as dashboard

    reports_dir = tmp_path / "reports"
    predictions_dir = tmp_path / "predictions"
    reports_dir.mkdir()
    predictions_dir.mkdir()
    timestamp = "2026-11-04T10:00:00+00:00"
    (reports_dir / "verified.json").write_text(
        json.dumps(
            {
                "batch_id": "verified-original-batch",
                "data_source_type": "real",
                "monitoring_timestamp": timestamp,
                "overall_status": "HEALTHY",
                "quality": {"quality_status": "HEALTHY", "total_rows": 2},
                "drift": {"numerical": {}, "categorical": {}},
            }
        ),
        encoding="utf-8",
    )
    (reports_dir / "unverified.json").write_text(
        json.dumps(
            {
                "batch_id": "unverified-batch",
                "monitoring_timestamp": timestamp,
                "overall_status": "CRITICAL",
                "quality": {"quality_status": "CRITICAL", "total_rows": 2},
                "drift": {"numerical": {}, "categorical": {}},
            }
        ),
        encoding="utf-8",
    )
    (predictions_dir / "batches.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "batch_id": batch_id,
                    "predicted_at": timestamp,
                    "predicted_class": predicted_class,
                }
            )
            for batch_id, predicted_class in (
                ("verified-original-batch", 1),
                ("unverified-batch", 0),
                ("normal-2026-seed-42", 1),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(dashboard, "PREDICTIONS_DIR", predictions_dir)
    monkeypatch.setattr(
        dashboard,
        "list_production_observations",
        lambda **kwargs: [
            {
                "observation_id": "labeled-observation",
                "batch_id": "verified-original-batch",
                "timestamp": timestamp,
                "model_version": "model1",
                "predicted_class": 1,
                "probability_positive": 0.8,
                "actual_outcome": 1,
                "outcome_received_at": timestamp,
            },
            {
                "observation_id": "pending-observation",
                "batch_id": "verified-original-batch",
                "timestamp": timestamp,
                "model_version": "model1",
                "predicted_class": 0,
                "probability_positive": 0.2,
                "actual_outcome": None,
                "outcome_received_at": None,
            },
        ],
    )

    reports = dashboard._load_reports(30)
    summary = dashboard._load_prediction_summary(
        {str(report["batch_id"]) for report in reports}
    )

    assert [report["batch_id"] for report in reports] == [
        "verified-original-batch"
    ]
    assert summary["total"] == 1
    assert summary["class_counts"] == {"not_cancelled": 0, "cancelled": 1}
    dashboard_data = dashboard.build_dashboard_data()
    assert dashboard_data["production_observations"]["total_observations"] == 2
    assert dashboard_data["production_observations"]["labeled_observations"] == 1
    assert dashboard_data["production_observations"]["pending_outcomes"] == 1


def test_dashboard_separates_real_simulated_and_unverified_outcomes(
    tmp_path, monkeypatch
):
    import backend.dashboard as dashboard
    from datetime import datetime, timezone

    reports_dir = tmp_path / "reports"
    performance_dir = tmp_path / "performance"
    predictions_dir = tmp_path / "predictions"
    reports_dir.mkdir()
    performance_dir.mkdir()
    predictions_dir.mkdir()
    timestamp = datetime.now(timezone.utc).isoformat()
    reports = [
        ("real-batch", "real"),
        ("normal-2026-seed-42", "synthetic_test"),
        ("unknown-batch", "unknown"),
    ]
    for batch_id, source in reports:
        (reports_dir / f"{batch_id}.json").write_text(
            json.dumps(
                {
                    "batch_id": batch_id,
                    "data_source_type": source,
                    "monitoring_timestamp": timestamp,
                    "overall_status": "HEALTHY",
                    "quality": {"quality_status": "HEALTHY", "total_rows": 2},
                    "drift": {"numerical": {}, "categorical": {}},
                }
            ),
            encoding="utf-8",
        )
    for batch_id, source in reports[:2]:
        (performance_dir / f"{batch_id}.json").write_text(
            json.dumps(
                {
                    "batch_id": batch_id,
                    "evaluation_timestamp": timestamp,
                    "model_version": "model-1",
                    "number_of_evaluated_records": 2,
                    "number_of_pending_outcomes": 0,
                    "metrics_available": True,
                    "data_source_type": source,
                    "accuracy": 0.5,
                    "precision": 0.5,
                    "recall": 0.5,
                    "f1_score": 0.5,
                    "confusion_matrix": [[1, 0], [1, 0]],
                    "source": "production" if batch_id == "real-batch" else source,
                    "outcome_provenance": (
                        "verified_production_api"
                        if batch_id == "real-batch"
                        else "csv_import"
                    ),
                    "monitoring_report": {
                        "data_source_type": source,
                        "source": "production" if batch_id == "real-batch" else source,
                    },
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(dashboard, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(dashboard, "PERFORMANCE_DIR", performance_dir)
    monkeypatch.setattr(dashboard, "PREDICTIONS_DIR", predictions_dir)

    result = dashboard.build_dashboard_data(30)

    assert result["monitoring"]["report_count"] == 1
    assert result["monitoring"]["source_counts"] == {
        "real": 1,
        "unknown": 1,
    }
    assert result["operational_performance"]["evaluated_batch_count"] == 1
    assert result["operational_performance"]["evaluated_record_count"] == 2
    assert result["operational_performance"]["history"][0]["batch_id"] == "real-batch"
    assert result["operational_performance"]["history"][0]["data_source_type"] == "real"
    assert result["system"]["live_outcomes"] == "AVAILABLE"
    assert result["drift"]["insights_by_source"]["real"]["data_source"] == "REAL"
    assert result["drift"]["insights_by_source"]["synthetic_test"] is None
    assert result["drift"]["insights_by_source"]["unknown"]["data_source"] == "UNKNOWN"


def test_alert_history_api_filters_and_returns_persisted_alerts(monkeypatch):
    from datetime import datetime, timezone

    import backend.main as main

    sample_alerts = [
        {
            "alert_id": "real-warning",
            "batch_id": "batch-real",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": "WARNING",
            "affected_feature": "lead_time",
            "detection_reason": "PSI warning threshold exceeded.",
            "data_source": "REAL",
            "recommended_action": "Inspect the feature pipeline.",
            "evidence": {"psi": 0.12},
        },
        {
            "alert_id": "sim-critical",
            "batch_id": "batch-sim",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": "CRITICAL",
            "affected_feature": "adr",
            "detection_reason": "Invalid values detected.",
            "data_source": "SIMULATED",
            "recommended_action": "Validate the input.",
            "evidence": {"invalid_count": 1},
        },
    ]
    monkeypatch.setattr(
        main,
        "load_alert_history",
        lambda *args, **filters: [
            alert
            for alert in sample_alerts
            if not filters["severity"] or alert["severity"] == filters["severity"]
            if not filters["batch_id"] or alert["batch_id"] == filters["batch_id"]
            if not filters["source"] or alert["data_source"] == filters["source"]
        ],
    )
    monkeypatch.setattr(main, "backfill_alert_history", lambda **kwargs: None)

    response = client.get(
        "/api/v1/alerts?severity=critical&source=simulated&batch_id=batch-sim"
    )

    assert response.status_code == 200
    result = response.json()
    assert result["count"] == 1
    assert result["alerts"][0]["alert_id"] == "sim-critical"
    all_alerts = client.get("/api/v1/alerts")
    assert [alert["alert_id"] for alert in all_alerts.json()["alerts"]] == [
        "real-warning",
        "sim-critical",
    ]
    assert client.get("/api/v1/alerts?source=other").status_code == 422
    assert client.get("/api/v1/alerts?severity=info").status_code == 422


def test_monitoring_csv_saves_report_and_report_can_be_downloaded(
    tmp_path, monkeypatch
):
    import backend.main as main
    from ml.monitoring import monitor_batch_data as monitor_data_with_paths

    source_path = next(
        (Path(__file__).resolve().parents[2] / "data" / "monitoring").glob(
            "normal_*.csv"
        )
    )
    batch = pd.read_csv(source_path, nrows=12).drop(
        columns=["batch_id", "generated_at", "row_id"], errors="ignore"
    )
    reports_dir = tmp_path / "reports"

    def monitor_to_temp(batch_data, *, data_source_type):
        return monitor_data_with_paths(
            batch_data,
            reports_dir=reports_dir,
            data_source_type=data_source_type,
        )

    monkeypatch.setattr(main, "monitor_batch_data", monitor_to_temp)
    monkeypatch.setattr(main, "REPORTS_DIR", reports_dir)

    response = client.post(
        "/api/v1/monitoring/batches/csv?batch_id=normal-api-seed-42&data_source_type=synthetic_test",
        content=batch.to_csv(index=False).encode("utf-8"),
        headers={"Content-Type": "text/csv"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["batch_id"] == "normal-api-seed-42"
    assert result["data_source_type"] == "synthetic_test"
    reports = client.get("/api/v1/monitoring/reports")
    assert reports.status_code == 200
    matching = [
        report
        for report in reports.json()["reports"]
        if report["batch_id"] == "normal-api-seed-42"
    ]
    assert len(matching) == 1
    assert matching[0]["source"] == "SIMULATED"
    assert matching[0]["quality_summary"]["total_rows"] == 12
    assert set(matching[0]["quality_summary"]) == {
        "missing_percentage",
        "duplicate_percentage",
        "invalid_percentage",
        "total_rows",
    }
    assert isinstance(matching[0]["quality_warnings"], list)
    drift_summary = matching[0]["drift_summary"]
    assert drift_summary["average_psi"] is not None
    assert drift_summary["feature_results"]
    assert {"feature", "feature_type", "psi", "ks_statistic", "ks_p_value"} <= set(
        drift_summary["feature_results"][0]
    )
    simulated_reports = client.get(
        "/api/v1/monitoring/reports?source=SIMULATED"
    )
    assert simulated_reports.status_code == 200
    assert any(
        report["batch_id"] == "normal-api-seed-42"
        for report in simulated_reports.json()["reports"]
    )
    download = client.get(
        "/api/v1/monitoring/reports/normal-api-seed-42"
    )
    assert download.status_code == 200


def test_dashboard_surfaces_persisted_dataset_simulation_results(
    tmp_path, monkeypatch
):
    import backend.dashboard as dashboard
    from ml.alert_engine import persist_alerts

    reports_dir = tmp_path / "reports"
    predictions_dir = tmp_path / "predictions"
    reports_dir.mkdir()
    predictions_dir.mkdir()
    timestamp = pd.Timestamp.now(tz="UTC").isoformat()
    report = {
        "batch_id": "normal-dataset-source-seed-5",
        "data_source_type": "synthetic_test",
        "batch_metadata": {
            "source": "original_hotel_booking_dataset_simulation",
            "record_count": 2,
            "timestamp": timestamp,
        },
        "monitoring_timestamp": timestamp,
        "overall_status": "WARNING",
        "reasons": ["lead_time drift"],
        "thresholds": {"psi_warning": 0.1, "psi_critical": 0.2},
        "quality": {
            "quality_status": "HEALTHY",
            "total_rows": 2,
            "duplicate_rate": 0,
            "missing_value_counts": {"lead_time": 0},
            "invalid_numerical_value_counts": {},
            "infinite_value_counts": {},
            "negative_value_counts": {},
            "unexpected_category_counts": {},
            "feature_schema": {"expected_features": ["lead_time"]},
        },
        "drift": {
            "numerical": {
                "lead_time": {
                    "usable": True,
                    "psi": 0.15,
                    "psi_level": "moderate",
                    "ks_meaningful_effect": False,
                }
            },
            "categorical": {},
        },
    }
    report_path = reports_dir / "simulated.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    predictions_dir.joinpath(f"{report['batch_id']}.jsonl").write_text(
        json.dumps(
            {
                "batch_id": report["batch_id"],
                "predicted_at": timestamp,
                "predicted_class": 1,
                "probability_positive": 0.8,
                "model_version": "Model 1@test",
                "source": "synthetic_test",
            }
        )
        + "\n"
        + json.dumps(
            {
                "batch_id": report["batch_id"],
                "predicted_at": timestamp,
                "predicted_class": 0,
                "probability_positive": 0.2,
                "model_version": "Model 1@test",
                "source": "synthetic_test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    alerts_path = tmp_path / "alerts.jsonl"
    persist_alerts(report, alerts_path)
    monkeypatch.setattr(dashboard, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(dashboard, "PREDICTIONS_DIR", predictions_dir)
    monkeypatch.setattr(dashboard, "ALERTS_PATH", alerts_path)
    monkeypatch.setattr(dashboard, "list_production_observations", lambda **kwargs: [])

    result = dashboard.build_dashboard_data(30)

    assert result["latest_monitoring"]["source"] == "synthetic_test"
    assert result["drift"]["features"] == [{"name": "lead_time", "score": 0.15}]
    assert result["predictions"]["source"] == "simulated"
    assert result["predictions"]["distribution_sample_count"] == 2
    assert len(result["predictions"]["prediction_history"]) == 2
    assert {
        item["source"] for item in result["predictions"]["prediction_history"]
    } == {"synthetic_test"}
    assert {
        item["row_id"] for item in result["predictions"]["prediction_history"]
    } == {"row-1", "row-2"}
    assert result["monitoring"]["simulation_report_count"] == 1
    assert result["monitoring"]["simulation_record_count"] == 2
    assert result["alerts"][0]["source"] == "simulated"
    assert result["alerts"][0]["alert_id"]
    assert len({alert["alert_id"] for alert in result["alerts"]}) == len(
        result["alerts"]
    )


def test_dashboard_rejects_invalid_period():
    response = client.get("/api/v1/dashboard?period_days=0")

    assert response.status_code == 422


def test_dashboard_allows_local_frontend_origin():
    response = client.get(
        "/api/v1/dashboard",
        headers={"Origin": "http://localhost:5173"},
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_model2_dashboard_status_requires_a_trained_artifact(tmp_path, monkeypatch):
    import backend.dashboard as dashboard

    dataset = pd.DataFrame(
        [
            {
                "batch_id": f"production-batch-{index}",
                "monitoring_timestamp": (
                    pd.Timestamp("2026-01-01T00:00:00Z")
                    + pd.Timedelta(days=index)
                ).isoformat(),
                "target_future_timestamp": (
                    pd.Timestamp("2026-01-02T00:00:00Z")
                    + pd.Timedelta(days=index)
                ).isoformat(),
                "data_source_type": "real",
                "source": "production",
                "label_provenance": "verified_production_api",
                "performance_degraded": index % 2,
            }
            for index in range(21)
        ]
    )
    monkeypatch.setattr(dashboard, "build_historical_dataset", lambda: dataset)
    metadata_path = tmp_path / "model2-random-forest-v1.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model2_status": "TRAINED",
                "model_version": metadata_path.stem,
                "feature_names": MODEL2_FEATURE_NAMES,
                "training_source": "verified_genuine_production_monitoring",
                "feature_importance": [
                    {"feature": MODEL2_FEATURE_NAMES[0], "importance": 1.0},
                ],
                "historical_dataset": {
                    "genuine_labeled_observation_count": 21,
                },
                "training_timestamp": "2026-11-06T10:00:00+00:00",
                "evaluation_metrics": {
                    "random_forest_primary_candidate": {
                        "accuracy": 0.8,
                        "recall": 0.75,
                        "f1_score": 0.77,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "MODEL2_DIR", tmp_path)

    missing_artifact = dashboard._model2_summary()
    assert missing_artifact["status"] == "NOT_TRAINED"
    assert "is missing" in missing_artifact["reason"]

    metadata_path.with_suffix(".joblib").touch()
    result = dashboard._model2_summary()

    assert result["status"] == "TRAINED"
    assert result["labeled_observation_count"] == 21
    assert result["production_batch_count"] == 21
    assert result["labeled_batch_count"] == 21
    assert result["unlabeled_batch_count"] == 0
    assert result["training_progress_percent"] == 100
    assert result["model_version"] == metadata_path.stem
    assert result["training_timestamp"] == "2026-11-06T10:00:00+00:00"
    assert result["evaluation_metrics"]["accuracy"] == 0.8
    assert result["latest_training_result"]["model_version"] == metadata_path.stem
    assert result["latest_training_result"]["training_performed"] is True


def test_model2_dashboard_counts_only_genuine_production_batches(
    tmp_path, monkeypatch
):
    import backend.dashboard as dashboard

    dataset = pd.DataFrame(
        [
            {
                "batch_id": f"synthetic-batch-{index}",
                "monitoring_timestamp": "2026-01-01T00:00:00Z",
                "target_future_timestamp": "2026-01-02T00:00:00Z",
                "data_source_type": "synthetic_test",
                "source": "synthetic_test",
                "performance_degraded": index % 2,
            }
            for index in range(30)
        ]
        + [
            {
                "batch_id": f"production-batch-{index}",
                "monitoring_timestamp": (
                    pd.Timestamp("2026-01-01T00:00:00Z")
                    + pd.Timedelta(days=index)
                ).isoformat(),
                "target_future_timestamp": (
                    pd.Timestamp("2026-01-02T00:00:00Z")
                    + pd.Timedelta(days=index)
                ).isoformat(),
                "data_source_type": "real",
                "source": "production",
                "label_provenance": (
                    "verified_production_api" if index < 2 else None
                ),
                "performance_degraded": index % 2 if index < 2 else None,
            }
            for index in range(4)
        ]
    )
    monkeypatch.setattr(dashboard, "MODEL2_DIR", tmp_path)
    monkeypatch.setattr(dashboard, "build_historical_dataset", lambda: dataset)
    production_metadata_path = tmp_path / "model2-random-forest-trained.json"
    production_metadata_path.write_text(
        json.dumps(
            {
                "model2_status": "TRAINED",
                "model_version": production_metadata_path.stem,
                "feature_names": MODEL2_FEATURE_NAMES,
                "historical_dataset": {
                    "genuine_labeled_observation_count": 21,
                },
                "evaluation_metrics": {},
            }
        ),
        encoding="utf-8",
    )
    production_metadata_path.with_suffix(".joblib").touch()
    (tmp_path / "training_status_20261111T100000000000Z.json").write_text(
        json.dumps(
            {
                "model2_status": "NOT_TRAINED",
                "training_attempt_timestamp": "2026-11-11T10:00:00+00:00",
                "training_performed": False,
                "reason": "Need at least 20 labeled production observations; found 2.",
            }
        ),
        encoding="utf-8",
    )

    result = dashboard._model2_summary()

    assert result["status"] == "NOT_TRAINED"
    assert result["labeled_observation_count"] == 2
    assert result["minimum_labeled_observations"] == 20
    assert result["production_batch_count"] == 4
    assert result["labeled_batch_count"] == 2
    assert result["unlabeled_batch_count"] == 2
    assert result["training_progress_percent"] == 10
    assert result["latest_training_result"]["timestamp"] == (
        "2026-11-11T10:00:00+00:00"
    )
    assert result["latest_training_result"]["reason"].endswith("found 2.")


def test_model2_development_summary_requires_usable_artifacts(tmp_path, monkeypatch):
    import backend.dashboard as dashboard

    monkeypatch.setattr(dashboard, "MODEL2_DEVELOPMENT_DIR", tmp_path)

    result = dashboard._model2_development_summary()

    assert result["status"] == "UNAVAILABLE"
    assert result["evaluation_metrics"] is None
    assert result["reason"]


def test_model2_dashboard_reports_corrupt_artifact_metadata(tmp_path, monkeypatch):
    import backend.dashboard as dashboard

    metadata_path = tmp_path / "model2-random-forest-v1.json"
    metadata_path.write_text("{", encoding="utf-8")
    monkeypatch.setattr(dashboard, "MODEL2_DIR", tmp_path)

    result = dashboard._model2_summary()

    assert result["status"] == "NOT_TRAINED"
    assert "Unable to read Model 2 metadata" in result["reason"]


def test_model2_development_explainability_is_provenance_checked(tmp_path, monkeypatch):
    import backend.dashboard as dashboard

    report_path = tmp_path / "feature_importance.json"
    monkeypatch.setattr(dashboard, "MODEL2_DEVELOPMENT_IMPORTANCE_PATH", report_path)
    unavailable = dashboard._model2_development_explainability()
    assert unavailable["status"] == "UNAVAILABLE"
    assert unavailable["features"] == []

    features = [
        {
            "feature": feature_name,
            "importance": 1 / len(MODEL2_FEATURE_NAMES),
            "rank": rank,
            "measures": "test",
            "model_rationale": "test",
            "risk_relation": "test",
        }
        for rank, feature_name in enumerate(MODEL2_FEATURE_NAMES, start=1)
    ]
    report_path.write_text(
        json.dumps(
            {
                "provenance": "SYNTHETIC_DEVELOPMENT_ONLY",
                "model_version": "development-test",
                "feature_names": MODEL2_FEATURE_NAMES,
                "features": features,
                "top_features": features[:10],
                "generation_timestamp": "2026-11-06T00:00:00+00:00",
                "importance_method": "Random Forest mean decrease in impurity",
                "causality_warning": "Importance is not causation.",
            }
        ),
        encoding="utf-8",
    )
    available = dashboard._model2_development_explainability()
    assert available["status"] == "AVAILABLE"
    assert available["provenance"] == "SYNTHETIC_DEVELOPMENT_ONLY"
    assert len(available["features"]) == 10

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["provenance"] = "PRODUCTION"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    invalid = dashboard._model2_development_explainability()
    assert invalid["status"] == "UNAVAILABLE"
    assert "invalid provenance" in invalid["reason"]

    report["provenance"] = "SYNTHETIC_DEVELOPMENT_ONLY"
    report["features"][0]["importance"] = float("nan")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    invalid_importance = dashboard._model2_development_explainability()
    assert invalid_importance["status"] == "UNAVAILABLE"
    assert "invalid ranked features" in invalid_importance["reason"]
