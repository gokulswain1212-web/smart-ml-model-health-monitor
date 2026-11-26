import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, BrokenBarrierError

import pandas as pd
import pytest

import ml.performance_tracking as performance_tracking
from ml.performance_tracking import (
    DuplicateOutcomeError,
    PoorQualityBatchError,
    build_monitoring_history,
    evaluate_batch_performance,
    import_actual_outcomes_csv,
    list_production_model2_predictions,
    list_production_observations,
    log_batch_predictions,
    record_model2_production_prediction,
    record_production_outcome,
    record_actual_outcomes,
)


ROOT = Path(__file__).resolve().parents[2]
FEATURE_NAMES_PATH = ROOT / "ml" / "models" / "feature_names.json"
MODEL_METADATA_PATH = ROOT / "ml" / "models" / "model_metadata.json"
PIPELINE_PATH = ROOT / "ml" / "models" / "hotel_cancellation_pipeline.pkl"
REFERENCE_PATH = ROOT / "data" / "raw" / "hotel_bookings.csv"
NORMAL_BATCH_PATH = next((ROOT / "data" / "monitoring").glob("normal_*.csv"))
POOR_BATCH_PATH = next((ROOT / "data" / "monitoring").glob("poor_quality_*.csv"))


@pytest.fixture
def tracking_paths(tmp_path):
    return {
        "predictions_dir": tmp_path / "predictions",
        "outcomes_dir": tmp_path / "outcomes",
        "performance_dir": tmp_path / "performance",
        "reports_dir": tmp_path / "reports",
    }


def load_normal_batch():
    return pd.read_csv(NORMAL_BATCH_PATH, nrows=12)


def log_test_batch(batch, tracking_paths, batch_id="test-batch", source="unknown"):
    batch = batch.copy()
    batch["batch_id"] = batch_id
    batch["generated_at"] = "2026-01-01T00:00:00+00:00"
    return log_batch_predictions(
        batch,
        pipeline_path=PIPELINE_PATH,
        feature_names_path=FEATURE_NAMES_PATH,
        metadata_path=MODEL_METADATA_PATH,
        reference_path=REFERENCE_PATH,
        source=source,
        predictions_dir=tracking_paths["predictions_dir"],
        reports_dir=tracking_paths["reports_dir"],
    )


def mark_batch_real(batch_id, tracking_paths, source=None):
    reports_dir = tracking_paths["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{batch_id}-real.json").write_text(
        json.dumps(
            {
                "batch_id": batch_id,
                "data_source_type": "real",
                **({"source": source} if source else {}),
                "monitoring_timestamp": "2026-01-01T00:00:00+00:00",
                "overall_status": "HEALTHY",
                "quality": {"quality_status": "HEALTHY"},
                "drift": {"numerical": {}, "categorical": {}},
            }
        ),
        encoding="utf-8",
    )


def test_prediction_logging_has_unique_ids_classes_probabilities_and_version(
    tracking_paths,
):
    result = log_test_batch(load_normal_batch(), tracking_paths)
    records = [
        json.loads(line)
        for line in Path(result["prediction_file"]).read_text(encoding="utf-8").splitlines()
    ]

    assert len(records) == 12
    assert len({record["row_id"] for record in records}) == 12
    assert all(record["row_id"].startswith("test-batch:") for record in records)
    assert all(record["predicted_class"] in (0, 1) for record in records)
    assert all(0 <= record["probability_positive"] <= 1 for record in records)
    assert all(record["model_version"] for record in records)
    assert all(len(record["model_artifact_sha256"]) == 64 for record in records)
    assert result["prediction_count"] == len(result["predictions"]) == len(records)
    assert result["predictions"] == [
        {
            "row_id": record["row_id"],
            "predicted_class": record["predicted_class"],
            "probability_positive": record["probability_positive"],
        }
        for record in records
    ]


def test_feature_schema_validation_and_target_leakage_are_rejected(tracking_paths):
    batch = load_normal_batch()
    batch = batch.drop(columns=["hotel"])
    batch["batch_id"] = "invalid-schema"
    batch["generated_at"] = "2026-01-01"
    with pytest.raises(ValueError, match="Invalid feature schema"):
        log_test_batch(batch, tracking_paths, "invalid-schema")

    batch = load_normal_batch()
    batch["is_canceled"] = 0
    with pytest.raises(ValueError, match="must not include target"):
        log_test_batch(batch, tracking_paths, "target-leak")


def test_poor_quality_batch_is_not_predicted(tracking_paths, monkeypatch):
    batch = pd.read_csv(POOR_BATCH_PATH, nrows=20)

    def fail_if_pipeline_loads(*args, **kwargs):
        pytest.fail("Model 1 must not be loaded for a critical-quality batch.")

    monkeypatch.setattr(
        "ml.performance_tracking.joblib.load",
        fail_if_pipeline_loads,
    )
    with pytest.raises(PoorQualityBatchError):
        log_test_batch(batch, tracking_paths, "poor-quality-test")

    assert not list(tracking_paths["predictions_dir"].glob("*.jsonl"))


def test_supplied_row_ids_are_used_and_duplicates_rejected(tracking_paths):
    batch = load_normal_batch()
    batch["row_id"] = [f"source-row-{index}" for index in range(len(batch))]
    result = log_test_batch(batch, tracking_paths)
    records = [
        json.loads(line)
        for line in Path(result["prediction_file"]).read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["row_id"] == "source-row-0"

    duplicate_batch = load_normal_batch()
    duplicate_batch["row_id"] = ["same-id"] * len(duplicate_batch)
    with pytest.raises(ValueError, match="unique within the batch"):
        log_test_batch(duplicate_batch, tracking_paths, "duplicate-ids")


def test_delayed_outcomes_are_pending_then_evaluated_and_appended(
    tracking_paths,
):
    result = log_test_batch(load_normal_batch(), tracking_paths)
    pending = evaluate_batch_performance(
        result["batch_id"], **{key: tracking_paths[key] for key in ("predictions_dir", "outcomes_dir", "performance_dir", "reports_dir")}
    )

    assert pending["number_of_predictions"] == 12
    assert pending["number_of_actual_outcomes_received"] == 0
    assert pending["number_of_pending_outcomes"] == 12
    assert pending["accuracy"] is None
    assert pending["metrics_available"] is False

    prediction_records = [
        json.loads(line)
        for line in Path(result["prediction_file"]).read_text(encoding="utf-8").splitlines()
    ]
    outcomes = pd.DataFrame(
        {
            "row_id": [record["row_id"] for record in prediction_records[:4]],
            "is_canceled": [0, 1, 0, 1],
        }
    )
    mark_batch_real(result["batch_id"], tracking_paths)
    outcome_result = record_actual_outcomes(
        result["batch_id"], outcomes, **{
            key: tracking_paths[key]
            for key in ("predictions_dir", "outcomes_dir", "reports_dir")
        }
    )
    assert outcome_result["outcome_count"] == 4

    evaluated = evaluate_batch_performance(
        result["batch_id"],
        **{
            key: tracking_paths[key]
            for key in ("predictions_dir", "outcomes_dir", "performance_dir", "reports_dir")
        },
    )
    assert evaluated["number_of_actual_outcomes_received"] == 4
    assert evaluated["number_of_pending_outcomes"] == 8
    assert evaluated["metrics_available"] is True
    assert evaluated["data_source_type"] == "real"
    assert evaluated["accuracy"] is not None
    assert evaluated["precision"] is not None
    assert evaluated["recall"] is not None
    assert evaluated["f1_score"] is not None
    assert evaluated["roc_auc"] is not None
    expected_predictions = [
        record["predicted_class"] for record in prediction_records[:4]
    ]
    true_positives = sum(
        actual == 1 and predicted == 1
        for actual, predicted in zip([0, 1, 0, 1], expected_predictions)
    )
    false_positives = sum(
        actual == 0 and predicted == 1
        for actual, predicted in zip([0, 1, 0, 1], expected_predictions)
    )
    false_negatives = sum(
        actual == 1 and predicted == 0
        for actual, predicted in zip([0, 1, 0, 1], expected_predictions)
    )
    true_negatives = sum(
        actual == 0 and predicted == 0
        for actual, predicted in zip([0, 1, 0, 1], expected_predictions)
    )
    assert evaluated["confusion_matrix"] == [
        [true_negatives, false_positives],
        [false_negatives, true_positives],
    ]
    assert evaluated["accuracy"] == pytest.approx(
        (true_positives + true_negatives) / 4
    )
    expected_precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0
    )
    expected_recall = true_positives / (true_positives + false_negatives)
    expected_f1 = (
        2 * expected_precision * expected_recall / (expected_precision + expected_recall)
        if expected_precision + expected_recall
        else 0
    )
    assert evaluated["precision"] == pytest.approx(expected_precision)
    assert evaluated["recall"] == pytest.approx(expected_recall)
    assert evaluated["f1_score"] == pytest.approx(expected_f1)
    assert len(list(tracking_paths["performance_dir"].glob("*.json"))) == 2


def test_outcome_matching_rejects_unknown_duplicate_and_invalid_labels(
    tracking_paths,
):
    result = log_test_batch(load_normal_batch(), tracking_paths)
    mark_batch_real(result["batch_id"], tracking_paths)
    prediction_records = [
        json.loads(line)
        for line in Path(result["prediction_file"]).read_text(encoding="utf-8").splitlines()
    ]
    row_id = prediction_records[0]["row_id"]

    with pytest.raises(ValueError, match="must be 0 or 1"):
        record_actual_outcomes(
            result["batch_id"],
            pd.DataFrame({"row_id": [row_id], "is_canceled": [2]}),
            **{
                key: tracking_paths[key]
                for key in ("predictions_dir", "outcomes_dir", "reports_dir")
            },
        )
    with pytest.raises(ValueError, match="without predictions"):
        record_actual_outcomes(
            result["batch_id"],
            pd.DataFrame({"row_id": ["unknown-row"], "is_canceled": [1]}),
            **{
                key: tracking_paths[key]
                for key in ("predictions_dir", "outcomes_dir", "reports_dir")
            },
        )

    outcomes = pd.DataFrame({"row_id": [row_id], "is_canceled": [1]})
    record_actual_outcomes(
        result["batch_id"],
        outcomes,
        **{
            key: tracking_paths[key]
            for key in ("predictions_dir", "outcomes_dir", "reports_dir")
        },
    )
    with pytest.raises(ValueError, match="already recorded"):
        record_actual_outcomes(
            result["batch_id"],
            outcomes,
            **{
                key: tracking_paths[key]
                for key in ("predictions_dir", "outcomes_dir", "reports_dir")
            },
        )


def test_concurrent_duplicate_outcomes_are_recorded_once(
    tracking_paths, monkeypatch
):
    result = log_test_batch(
        load_normal_batch().iloc[:2],
        tracking_paths,
        batch_id="concurrent-production-outcome",
        source="production",
    )
    mark_batch_real(result["batch_id"], tracking_paths, source="production")
    predictions = [
        json.loads(line)
        for line in Path(result["prediction_file"]).read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    row_id = predictions[0]["row_id"]
    outcomes_path = (
        tracking_paths["outcomes_dir"] / f"{result['batch_id']}.jsonl"
    )
    original_append = performance_tracking._append_jsonl
    append_barrier = Barrier(2, timeout=1)

    def synchronize_duplicate_writes(path, records):
        if path == outcomes_path:
            try:
                append_barrier.wait()
            except BrokenBarrierError:
                pass
        original_append(path, records)

    monkeypatch.setattr(
        performance_tracking, "_append_jsonl", synchronize_duplicate_writes
    )
    request_barrier = Barrier(2)

    def submit_outcome():
        request_barrier.wait()
        try:
            record_actual_outcomes(
                result["batch_id"],
                pd.DataFrame({"row_id": [row_id], "is_canceled": [1]}),
                source="production",
                **{
                    key: tracking_paths[key]
                    for key in ("predictions_dir", "outcomes_dir", "reports_dir")
                },
            )
        except DuplicateOutcomeError:
            return "duplicate"
        return "accepted"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: submit_outcome(), range(2)))

    assert sorted(results) == ["accepted", "duplicate"]
    assert len(outcomes_path.read_text(encoding="utf-8").splitlines()) == 1


def test_roc_auc_is_unavailable_without_both_actual_classes(tracking_paths):
    result = log_test_batch(load_normal_batch(), tracking_paths)
    mark_batch_real(result["batch_id"], tracking_paths)
    prediction_records = [
        json.loads(line)
        for line in Path(result["prediction_file"]).read_text(encoding="utf-8").splitlines()
    ]
    record_actual_outcomes(
        result["batch_id"],
        pd.DataFrame(
            {
                "row_id": [record["row_id"] for record in prediction_records[:3]],
                "is_canceled": [1, 1, 1],
            }
        ),
        **{
            key: tracking_paths[key]
            for key in ("predictions_dir", "outcomes_dir", "reports_dir")
        },
    )
    performance = evaluate_batch_performance(
        result["batch_id"],
        **{
            key: tracking_paths[key]
            for key in ("predictions_dir", "outcomes_dir", "performance_dir", "reports_dir")
        },
    )

    assert performance["accuracy"] is not None
    assert performance["roc_auc"] is None


def test_csv_outcome_import_matches_real_predictions_and_persists_metrics(
    tracking_paths,
):
    result = log_test_batch(load_normal_batch(), tracking_paths)
    mark_batch_real(result["batch_id"], tracking_paths)
    prediction_records = [
        json.loads(line)
        for line in Path(result["prediction_file"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    csv_content = (
        "row_id,is_canceled\n"
        + "\n".join(
            f"{record['row_id']},{index % 2}"
            for index, record in enumerate(prediction_records[:4])
        )
        + "\n"
    ).encode()

    imported = import_actual_outcomes_csv(
        result["batch_id"],
        csv_content,
        **tracking_paths,
    )

    assert imported["outcome"]["outcome_count"] == 4
    performance = imported["performance"]
    assert performance["number_of_evaluated_records"] == 4
    assert performance["data_source_type"] == "real"
    assert performance["metrics_available"] is True
    assert performance["accuracy"] is not None
    assert performance["precision"] is not None
    assert performance["recall"] is not None
    assert performance["f1_score"] is not None
    assert performance["confusion_matrix"] is not None
    stored_outcomes = [
        json.loads(line)
        for line in Path(imported["outcome"]["outcome_file"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(item["data_source_type"] == "real" for item in stored_outcomes)
    assert all(item["outcome_source"] == "csv_import" for item in stored_outcomes)
    assert all(item["outcome_received_at"] for item in stored_outcomes)
    assert performance["outcome_provenance"] == "csv_import"


@pytest.mark.parametrize(
    ("csv_content", "error"),
    [
        (b"row_id,is_canceled\n,1\n", "row_id values cannot be empty"),
        (b"row_id,is_canceled\nr1,1\nr1,0\n", "Duplicate row_id"),
        (b"row_id,is_canceled\nunknown-row,1\n", "without predictions"),
        (b"row_id,is_canceled\nr1,2\n", "must be 0 or 1"),
        (b"row_id,is_canceled,batch_id\nr1,1,wrong-batch\n", "do not match"),
        (b"row_id,is_canceled,extra\nr1,1,x\n", "must contain row_id"),
    ],
)
def test_csv_outcome_import_rejects_invalid_rows_without_writing(
    tracking_paths, csv_content, error
):
    result = log_test_batch(load_normal_batch(), tracking_paths)
    mark_batch_real(result["batch_id"], tracking_paths)
    prediction_records = [
        json.loads(line)
        for line in Path(result["prediction_file"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    csv_content = csv_content.replace(
        b"r1", prediction_records[0]["row_id"].encode("utf-8")
    )

    with pytest.raises(ValueError, match=error):
        import_actual_outcomes_csv(
            result["batch_id"],
            csv_content,
            **tracking_paths,
        )

    assert not list(tracking_paths["outcomes_dir"].glob("*.jsonl"))


def test_outcomes_require_a_real_monitoring_report(tracking_paths):
    result = log_test_batch(load_normal_batch(), tracking_paths)

    with pytest.raises(ValueError, match="explicitly marked as real"):
        record_actual_outcomes(
            result["batch_id"],
            pd.DataFrame({"row_id": ["source-row"], "is_canceled": [1]}),
            **{
                key: tracking_paths[key]
                for key in ("predictions_dir", "outcomes_dir", "reports_dir")
            },
        )

    assert not list(tracking_paths["outcomes_dir"].glob("*.jsonl"))


def test_monitoring_report_link_uses_batch_id(tracking_paths):
    result = log_test_batch(load_normal_batch(), tracking_paths)
    tracking_paths["reports_dir"].mkdir(parents=True)
    report_path = tracking_paths["reports_dir"] / "monitoring.json"
    report_path.write_text(
        json.dumps(
            {
                "batch_id": result["batch_id"],
                "monitoring_timestamp": "2026-01-01T00:00:00+00:00",
                "overall_status": "WARNING",
                "quality": {"quality_status": "HEALTHY"},
                "drift": {"numerical": {}, "categorical": {}},
            }
        ),
        encoding="utf-8",
    )
    performance = evaluate_batch_performance(
        result["batch_id"],
        **{
            key: tracking_paths[key]
            for key in ("predictions_dir", "outcomes_dir", "performance_dir", "reports_dir")
        },
    )
    history = build_monitoring_history(
        reports_dir=tracking_paths["reports_dir"],
        performance_dir=tracking_paths["performance_dir"],
    )

    assert performance["monitoring_report"]["overall_status"] == "WARNING"
    assert history[0]["batch_id"] == result["batch_id"]
    assert history[0]["monitoring_report"]["quality_status"] == "HEALTHY"
    assert len(history[0]["performance_records"]) == 1


def test_duplicate_batch_prediction_log_is_not_overwritten(tracking_paths):
    batch = load_normal_batch()
    result = log_test_batch(batch, tracking_paths)
    with pytest.raises(FileExistsError, match="already exist"):
        log_test_batch(batch, tracking_paths)
    assert Path(result["prediction_file"]).is_file()


def test_production_predictions_get_unique_observation_ids_and_remain_pending(
    tracking_paths,
):
    batch = load_normal_batch().drop(columns=["row_id", "generated_at"], errors="ignore")
    batch["batch_id"] = "production-observation-test"
    batch["source_booking_id_hmac"] = ["a" * 64] * len(batch)
    result = log_batch_predictions(
        batch,
        source="production",
        pipeline_path=PIPELINE_PATH,
        feature_names_path=FEATURE_NAMES_PATH,
        metadata_path=MODEL_METADATA_PATH,
        reference_path=REFERENCE_PATH,
        predictions_dir=tracking_paths["predictions_dir"],
        reports_dir=tracking_paths["reports_dir"],
    )
    records = [
        json.loads(line)
        for line in Path(result["prediction_file"]).read_text(encoding="utf-8").splitlines()
    ]

    assert len(result["observation_ids"]) == len(records) == 12
    assert len(set(result["observation_ids"])) == 12
    assert all(record["source"] == "production" for record in records)
    assert all(record["observation_id"] == record["row_id"] for record in records)
    assert all(record["source_booking_id_hmac"] == "a" * 64 for record in records)
    assert "source_booking_id_hmac" not in result


def test_production_predictions_preserve_supplied_booking_ids(tracking_paths):
    batch = load_normal_batch().iloc[:3].drop(
        columns=["row_id", "generated_at"], errors="ignore"
    )
    batch["batch_id"] = "production-booking-id-test"
    batch["row_id"] = ["booking-1001", "booking-1002", "booking-1003"]

    result = log_batch_predictions(
        batch,
        source="production",
        pipeline_path=PIPELINE_PATH,
        feature_names_path=FEATURE_NAMES_PATH,
        metadata_path=MODEL_METADATA_PATH,
        reference_path=REFERENCE_PATH,
        predictions_dir=tracking_paths["predictions_dir"],
        reports_dir=tracking_paths["reports_dir"],
    )
    records = [
        json.loads(line)
        for line in Path(result["prediction_file"]).read_text(encoding="utf-8").splitlines()
    ]

    assert [record["row_id"] for record in records] == [
        "booking-1001",
        "booking-1002",
        "booking-1003",
    ]
    assert all(record["observation_id"] == record["row_id"] for record in records)


def test_production_outcome_matches_observation_once_and_updates_metrics(
    tracking_paths,
):
    batch = load_normal_batch().iloc[:4].drop(
        columns=["row_id", "generated_at"], errors="ignore"
    )
    batch["batch_id"] = "production-outcome-test"
    result = log_batch_predictions(
        batch,
        source="production",
        pipeline_path=PIPELINE_PATH,
        feature_names_path=FEATURE_NAMES_PATH,
        metadata_path=MODEL_METADATA_PATH,
        reference_path=REFERENCE_PATH,
        predictions_dir=tracking_paths["predictions_dir"],
        reports_dir=tracking_paths["reports_dir"],
    )
    reports_dir = tracking_paths["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "production.json").write_text(
        json.dumps(
            {
                "batch_id": result["batch_id"],
                "data_source_type": "real",
                "source": "production",
                "monitoring_timestamp": "2026-01-01T00:00:00+00:00",
                "quality": {"quality_status": "HEALTHY"},
                "drift": {"numerical": {}, "categorical": {}},
            }
        ),
        encoding="utf-8",
    )

    pending = list_production_observations(
        predictions_dir=tracking_paths["predictions_dir"],
        outcomes_dir=tracking_paths["outcomes_dir"],
        reports_dir=reports_dir,
    )
    assert len(pending) == 4
    assert all(item["actual_outcome"] is None for item in pending)

    observation_id = result["observation_ids"][0]
    saved = record_production_outcome(
        observation_id,
        1,
        **tracking_paths,
    )
    assert saved["performance"]["number_of_actual_outcomes_received"] == 1
    assert saved["performance"]["number_of_pending_outcomes"] == 3
    assert saved["performance"]["accuracy"] is not None
    assert saved["performance"]["source"] == "production"
    with pytest.raises(ValueError, match="already recorded"):
        record_production_outcome(observation_id, 0, **tracking_paths)

    observations = list_production_observations(
        predictions_dir=tracking_paths["predictions_dir"],
        outcomes_dir=tracking_paths["outcomes_dir"],
        reports_dir=reports_dir,
    )
    labeled = [item for item in observations if item["actual_outcome"] is not None]
    assert len(labeled) == 1
    assert labeled[0]["observation_id"] == observation_id
    assert labeled[0]["actual_outcome"] == 1
    assert labeled[0]["outcome_received_at"]


def test_csv_import_cannot_label_a_production_batch(tracking_paths):
    result = log_test_batch(
        load_normal_batch().iloc[:2],
        tracking_paths,
        batch_id="production-csv-outcomes",
        source="production",
    )
    mark_batch_real(result["batch_id"], tracking_paths, source="production")
    prediction_records = [
        json.loads(line)
        for line in Path(result["prediction_file"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    csv_content = (
        "row_id,is_canceled\n"
        + "\n".join(
            f"{record['row_id']},{index % 2}"
            for index, record in enumerate(prediction_records)
        )
        + "\n"
    ).encode()

    with pytest.raises(ValueError, match="token-authenticated production outcome API"):
        import_actual_outcomes_csv(
            result["batch_id"],
            csv_content,
            **tracking_paths,
        )

    assert not list(tracking_paths["outcomes_dir"].glob("*.jsonl"))
    assert not list(tracking_paths["performance_dir"].glob("*.json"))


def test_model2_production_prediction_history_preserves_source_and_batch(
    tracking_paths,
):
    timestamp = pd.Timestamp.now(tz="UTC").isoformat()
    saved = record_model2_production_prediction(
        {
            "batch_id": "trusted-production-batch",
            "model2_status": "TRAINED",
            "prediction_timestamp": timestamp,
            "predicted_degradation_risk": 1,
            "degradation_probability": 0.9,
            "risk_category": "HIGH",
            "model_version": "model2-production-v1",
        },
        predictions_dir=tracking_paths["predictions_dir"],
    )

    history = list_production_model2_predictions(
        predictions_dir=tracking_paths["predictions_dir"],
    )

    assert len(history) == 1
    assert history[0]["prediction_id"] == saved["prediction_id"]
    assert history[0]["batch_id"] == "trusted-production-batch"
    assert history[0]["source"] == "production"
    assert history[0]["health_status"] == "CRITICAL"
