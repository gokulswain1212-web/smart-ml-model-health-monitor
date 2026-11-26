import json
from pathlib import Path

import pandas as pd
import pytest

from ml.model2_config import DEGRADATION_TARGET, MODEL2_FEATURE_NAMES
from ml.model2_dataset import (
    _timestamp,
    build_historical_dataset,
    dataset_summary,
    save_historical_dataset,
)
from ml.model2_prediction import features_for_batch, predict_degradation
from ml.model2_training import train_model2, training_eligibility


def _write_report(
    reports_dir: Path,
    batch_id: str,
    timestamp: str,
    *,
    source: str = "real",
    origin: str = "production",
    fixture_provenance: str | None = None,
) -> None:
    report = {
        "batch_id": batch_id,
        "monitoring_timestamp": timestamp,
        "data_source_type": source,
        "source": origin if source == "real" else source,
        **(
            {"fixture_provenance": fixture_provenance}
            if fixture_provenance
            else {}
        ),
        "quality": {
            "total_rows": 100,
            "duplicate_rate": 0.01,
            "feature_schema": {"expected_features": ["lead_time", "hotel"]},
            "missing_value_counts": {"lead_time": 10, "hotel": 0},
            "invalid_numerical_value_counts": {"lead_time": 1},
            "infinite_value_counts": {"lead_time": 0},
            "negative_value_counts": {"lead_time": 0},
            "unexpected_category_counts": {"hotel": 2},
        },
        "drift": {
            "numerical": {
                "lead_time": {"usable": True, "psi": 0.25},
            },
            "categorical": {
                "hotel": {"usable": True, "drift_score": 0.05},
            },
        },
    }
    (reports_dir / f"{batch_id}.json").write_text(
        json.dumps(report), encoding="utf-8"
    )


def _write_performance(
    performance_dir: Path,
    batch_id: str,
    timestamp: str,
    f1_score: float,
    *,
    number_of_predictions: int = 2,
    number_of_actual_outcomes_received: int = 2,
    number_of_evaluated_records: int = 2,
    number_of_pending_outcomes: int = 0,
    record_name: str | None = None,
    fixture_provenance: str | None = None,
) -> None:
    record = {
        "batch_id": batch_id,
        "evaluation_timestamp": timestamp,
        "metrics_available": True,
        "number_of_predictions": number_of_predictions,
        "number_of_actual_outcomes_received": number_of_actual_outcomes_received,
        "number_of_evaluated_records": number_of_evaluated_records,
        "number_of_pending_outcomes": number_of_pending_outcomes,
        "actual_outcome_class_counts": (
            {"0": 1, "1": 1}
            if number_of_actual_outcomes_received == 2
            else {"0": number_of_actual_outcomes_received}
        ),
        "data_source_type": "real",
        "source": "production",
        "outcome_provenance": "verified_production_api",
        "contributing_observation_ids": [f"{batch_id}-{index}" for index in range(
            number_of_actual_outcomes_received
        )],
        "latest_outcome_received_at": timestamp,
        "model_version": "model1-version-1",
        **(
            {"fixture_provenance": fixture_provenance}
            if fixture_provenance
            else {}
        ),
        "accuracy": f1_score,
        "precision": f1_score,
        "recall": f1_score,
        "f1_score": f1_score,
    }
    (performance_dir / f"{record_name or batch_id}.json").write_text(
        json.dumps(record), encoding="utf-8"
    )


def test_timestamp_parsing_normalizes_iso_values_and_rejects_invalid_values():
    assert _timestamp("2026-01-01T05:30:00+05:30", field="timestamp") == pd.Timestamp(
        "2026-01-01T00:00:00Z"
    )
    assert _timestamp("2026-01-01T00:00:00Z", field="timestamp") == pd.Timestamp(
        "2026-01-01T00:00:00Z"
    )
    with pytest.raises(ValueError, match="Invalid or missing timestamp"):
        _timestamp("not-a-timestamp", field="timestamp")


def test_target_uses_next_future_genuine_f1_and_only_prior_metrics_are_features(
    tmp_path,
):
    reports = tmp_path / "reports"
    predictions = tmp_path / "predictions"
    performance = tmp_path / "performance"
    for folder in (reports, predictions, performance):
        folder.mkdir()

    _write_report(reports, "baseline", "2026-01-01T00:00:00Z")
    _write_report(reports, "sample-degraded", "2026-01-03T00:00:00Z")
    _write_report(reports, "sample-stable", "2026-01-05T00:00:00Z")
    _write_report(reports, "future-degraded", "2026-01-04T00:00:00Z")
    _write_report(reports, "future-stable", "2026-01-06T00:00:00Z")
    for batch_id, timestamp in (
        ("baseline", "2026-01-01T00:00:00Z"),
        ("sample-degraded", "2026-01-03T00:00:00Z"),
        ("sample-stable", "2026-01-05T00:00:00Z"),
        ("future-degraded", "2026-01-04T00:00:00Z"),
        ("future-stable", "2026-01-06T00:00:00Z"),
    ):
        _write_prediction_records(
            predictions,
            batch_id,
            timestamp,
            [0, 1],
            source="production",
            model_version="model1-version-1",
        )
    _write_performance(performance, "baseline", "2026-01-02T00:00:00Z", 0.90)
    _write_performance(performance, "future-degraded", "2026-01-04T12:00:00Z", 0.84)
    _write_performance(performance, "future-stable", "2026-01-06T12:00:00Z", 0.80)

    dataset = build_historical_dataset(
        reports_dir=reports,
        predictions_dir=predictions,
        performance_dir=performance,
    ).set_index("batch_id")

    degraded = dataset.loc["sample-degraded"]
    stable = dataset.loc["sample-stable"]
    assert degraded[DEGRADATION_TARGET] == 1
    assert degraded["target_baseline_f1"] == 0.90
    assert degraded["target_future_f1"] == 0.84
    assert degraded["historical_f1_score"] == 0.90
    assert stable[DEGRADATION_TARGET] == 0
    assert stable["target_baseline_f1"] == 0.84
    assert stable["target_future_f1"] == 0.80
    assert degraded["data_source_type"] == "real"
    assert degraded["source"] == "production"

    assert list(MODEL2_FEATURE_NAMES) == [
        name for name in MODEL2_FEATURE_NAMES if name not in {
            DEGRADATION_TARGET,
            "target_future_f1",
            "target_f1_drop",
        }
    ]
    assert "target_future_f1" not in MODEL2_FEATURE_NAMES
    assert "target_baseline_f1" not in MODEL2_FEATURE_NAMES


def test_partial_future_evaluation_does_not_label_until_batch_is_complete(tmp_path):
    reports = tmp_path / "reports"
    predictions = tmp_path / "predictions"
    performance = tmp_path / "performance"
    for folder in (reports, predictions, performance):
        folder.mkdir()

    for batch_id, timestamp in (
        ("baseline", "2026-01-01T00:00:00Z"),
        ("observation", "2026-01-03T00:00:00Z"),
        ("future", "2026-01-04T00:00:00Z"),
    ):
        _write_report(reports, batch_id, timestamp)
        (predictions / f"{batch_id}.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "batch_id": batch_id,
                        "row_id": f"{batch_id}-{index}",
                        "observation_id": f"{batch_id}-{index}",
                        "source": "production",
                        "predicted_at": timestamp,
                        "predicted_class": index,
                        "probability_positive": float(index),
                        "model_version": "model1-version-1",
                    }
                )
                + "\n"
                for index in range(2)
            ),
            encoding="utf-8",
        )

    _write_performance(
        performance, "baseline", "2026-01-02T00:00:00Z", 0.90
    )
    _write_performance(
        performance,
        "future",
        "2026-01-04T12:00:00Z",
        0.70,
        number_of_predictions=2,
        number_of_actual_outcomes_received=0,
        number_of_evaluated_records=0,
        number_of_pending_outcomes=2,
        record_name="future-partial",
    )

    dataset = build_historical_dataset(
        reports_dir=reports,
        predictions_dir=predictions,
        performance_dir=performance,
    ).set_index("batch_id")
    assert pd.isna(dataset.loc["observation", DEGRADATION_TARGET])

    _write_performance(
        performance,
        "future",
        "2026-01-05T12:00:00Z",
        0.88,
        number_of_predictions=2,
        number_of_actual_outcomes_received=2,
        number_of_evaluated_records=2,
        number_of_pending_outcomes=0,
        record_name="future-complete",
    )
    dataset = build_historical_dataset(
        reports_dir=reports,
        predictions_dir=predictions,
        performance_dir=performance,
    ).set_index("batch_id")

    observation = dataset.loc["observation"]
    assert observation[DEGRADATION_TARGET] == 0
    assert observation["target_baseline_f1"] == 0.90
    assert observation["target_future_f1"] == 0.88
    assert observation["target_future_timestamp"] == "2026-01-05T12:00:00+00:00"


def test_synthetic_and_unverified_records_never_receive_training_labels(tmp_path):
    reports = tmp_path / "reports"
    predictions = tmp_path / "predictions"
    performance = tmp_path / "performance"
    for folder in (reports, predictions, performance):
        folder.mkdir()

    _write_report(
        reports,
        "normal-20260101T000000Z-seed-42",
        "2026-01-01T00:00:00Z",
        source="real",
    )
    (performance / "seeded.json").write_text(
        json.dumps(
            {
                "batch_id": "normal-20260101T000000Z-seed-42",
                "evaluation_timestamp": "2026-01-02T00:00:00Z",
                "metrics_available": True,
                "number_of_actual_outcomes_received": 20,
                "f1_score": 0.2,
            }
        ),
        encoding="utf-8",
    )
    _write_report(reports, "unverified-batch", "2026-01-03T00:00:00Z", source="unknown")

    dataset = build_historical_dataset(
        reports_dir=reports,
        predictions_dir=predictions,
        performance_dir=performance,
    ).set_index("batch_id")

    assert dataset.loc["normal-20260101T000000Z-seed-42", "data_source_type"] == (
        "synthetic_test"
    )
    assert pd.isna(
        dataset.loc["normal-20260101T000000Z-seed-42", DEGRADATION_TARGET]
    )
    assert dataset.loc[
        "normal-20260101T000000Z-seed-42", "historical_metrics_available"
    ] == 0
    assert dataset.loc[
        "normal-20260101T000000Z-seed-42", "historical_performance_observations"
    ] == 0
    assert dataset.loc["unverified-batch", "data_source_type"] == "unknown"
    assert pd.isna(dataset.loc["unverified-batch", DEGRADATION_TARGET])


def test_future_performance_from_a_different_model_version_does_not_label_row(
    tmp_path,
):
    reports = tmp_path / "reports"
    predictions = tmp_path / "predictions"
    performance = tmp_path / "performance"
    for folder in (reports, predictions, performance):
        folder.mkdir()
    _write_report(reports, "baseline", "2026-01-01T00:00:00Z")
    _write_report(reports, "sample", "2026-01-03T00:00:00Z")
    _write_report(reports, "future", "2026-01-04T00:00:00Z")
    for batch_id, timestamp in (
        ("baseline", "2026-01-01T00:00:00Z"),
        ("sample", "2026-01-03T00:00:00Z"),
        ("future", "2026-01-04T00:00:00Z"),
    ):
        _write_prediction_records(
            predictions,
            batch_id,
            timestamp,
            [0],
            source="production",
            model_version="model1-version-1",
        )
    _write_performance(performance, "baseline", "2026-01-02T00:00:00Z", 0.90)
    future = {
        "batch_id": "future",
        "evaluation_timestamp": "2026-01-04T12:00:00Z",
        "metrics_available": True,
        "number_of_predictions": 20,
        "number_of_actual_outcomes_received": 20,
        "number_of_evaluated_records": 20,
        "number_of_pending_outcomes": 0,
        "data_source_type": "real",
        "source": "production",
        "outcome_provenance": "verified_production_api",
        "contributing_observation_ids": ["future-0"],
        "latest_outcome_received_at": "2026-01-04T12:00:00Z",
        "model_version": "model1-version-2",
        "f1_score": 0.70,
    }
    (performance / "future.json").write_text(json.dumps(future), encoding="utf-8")

    dataset = build_historical_dataset(
        reports_dir=reports,
        predictions_dir=predictions,
        performance_dir=performance,
    ).set_index("batch_id")

    assert pd.isna(dataset.loc["sample", DEGRADATION_TARGET])


def test_operator_declared_real_reports_are_not_production_training_rows(tmp_path):
    reports = tmp_path / "reports"
    predictions = tmp_path / "predictions"
    performance = tmp_path / "performance"
    for folder in (reports, predictions, performance):
        folder.mkdir()
    _write_report(
        reports,
        "operator-declared-real",
        "2026-01-01T00:00:00Z",
        source="real",
        origin="real",
    )

    row = build_historical_dataset(
        reports_dir=reports,
        predictions_dir=predictions,
        performance_dir=performance,
    ).iloc[0]

    assert row["data_source_type"] == "real"
    assert row["source"] == "real"
    assert pd.isna(row[DEGRADATION_TARGET])
    assert dataset_summary(pd.DataFrame([row]))[
        "genuine_labeled_observation_count"
    ] == 0


def test_isolated_fixture_dataset_trains_and_predicts_without_production_writes(
    tmp_path,
):
    reports = tmp_path / "reports"
    predictions = tmp_path / "predictions"
    performance = tmp_path / "performance"
    model_dir = tmp_path / "model2"
    for folder in (reports, predictions, performance):
        folder.mkdir()

    fixture_provenance = "SYNTHETIC_TEST_FIXTURE_ONLY"
    model_version = "model1-version-1"
    first_observation = pd.Timestamp("2025-01-01T00:00:00Z")
    baseline_batch = "isolated-synthetic-test-fixture-baseline"
    baseline_time = first_observation - pd.Timedelta(days=2)
    _write_report(
        reports,
        baseline_batch,
        baseline_time.isoformat(),
        fixture_provenance=fixture_provenance,
    )
    _write_prediction_records(
        predictions,
        baseline_batch,
        baseline_time.isoformat(),
        [0, 1],
        source="production",
        model_version=model_version,
        fixture_provenance=fixture_provenance,
    )
    _write_performance(
        performance,
        baseline_batch,
        (baseline_time + pd.Timedelta(hours=1)).isoformat(),
        0.9,
        fixture_provenance=fixture_provenance,
    )

    fixture_batches = []
    for index in range(20):
        batch_id = f"isolated-synthetic-test-fixture-production-{index:02d}"
        observation_time = first_observation + pd.Timedelta(days=32 * index)
        outcome_f1 = 0.8 if index % 2 == 0 else 0.9
        fixture_batches.append(batch_id)
        _write_report(
            reports,
            batch_id,
            observation_time.isoformat(),
            fixture_provenance=fixture_provenance,
        )
        _write_prediction_records(
            predictions,
            batch_id,
            observation_time.isoformat(),
            [0, 1],
            source="production",
            model_version=model_version,
            fixture_provenance=fixture_provenance,
        )
        _write_performance(
            performance,
            batch_id,
            (observation_time + pd.Timedelta(hours=1)).isoformat(),
            outcome_f1,
            fixture_provenance=fixture_provenance,
        )

    dataset = build_historical_dataset(
        reports_dir=reports,
        predictions_dir=predictions,
        performance_dir=performance,
    )

    assert set(dataset["batch_id"]) >= set(fixture_batches)
    labeled = dataset.loc[dataset["batch_id"].isin(fixture_batches)]
    assert labeled[DEGRADATION_TARGET].notna().all()
    assert labeled[DEGRADATION_TARGET].astype(int).value_counts().to_dict() == {
        0: 10,
        1: 10,
    }
    assert set(labeled["target_baseline_f1"].dropna()).issubset({0.8, 0.9})
    assert set(labeled["target_future_f1"].dropna()).issubset({0.8, 0.9})
    assert "target_future_f1" not in MODEL2_FEATURE_NAMES
    assert training_eligibility(dataset)["status"] == "READY"

    result = train_model2(dataset, model_dir=model_dir)

    assert result["model2_status"] == "TRAINED"
    assert result["historical_dataset"]["genuine_labeled_observation_count"] == 20
    assert result["training_size"] == 16
    assert result["testing_size"] == 4
    metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["temporal_split"]["training_target_end"] < (
        metadata["temporal_split"]["testing_start"]
    )
    feature_row = labeled.iloc[0]
    features = {
        name: None if pd.isna(feature_row[name]) else float(feature_row[name])
        for name in MODEL2_FEATURE_NAMES
    }
    prediction = predict_degradation(features, model_dir=model_dir)
    assert prediction["model2_status"] == "TRAINED"
    assert 0 <= prediction["degradation_probability"] <= 1
    assert prediction["predicted_degradation_risk"] in (0, 1)
    assert model_dir.parent == tmp_path
    assert not (tmp_path / "ml" / "models").exists()


def test_duplicate_production_observation_ids_are_rejected(tmp_path):
    reports = tmp_path / "reports"
    predictions = tmp_path / "predictions"
    performance = tmp_path / "performance"
    for folder in (reports, predictions, performance):
        folder.mkdir()
    for batch_id in ("production-a", "production-b"):
        _write_report(reports, batch_id, "2026-01-01T00:00:00Z")
        (predictions / f"{batch_id}.jsonl").write_text(
            json.dumps(
                {
                    "batch_id": batch_id,
                    "row_id": f"{batch_id}:1",
                    "observation_id": "duplicate-observation-id",
                    "source": "production",
                    "predicted_at": "2026-01-01T00:00:00Z",
                    "predicted_class": 0,
                    "probability_positive": 0.1,
                    "model_version": "model1-version-1",
                }
            )
            + "\n",
            encoding="utf-8",
        )

    import pytest

    with pytest.raises(ValueError, match="Duplicate production observation ID"):
        build_historical_dataset(
            reports_dir=reports,
            predictions_dir=predictions,
            performance_dir=performance,
        )


def test_feature_values_and_missing_values_are_prepared_and_saved(tmp_path):
    reports = tmp_path / "reports"
    predictions = tmp_path / "predictions"
    performance = tmp_path / "performance"
    for folder in (reports, predictions, performance):
        folder.mkdir()
    _write_report(reports, "batch-1", "2026-01-01T00:00:00Z")

    dataset = build_historical_dataset(
        reports_dir=reports,
        predictions_dir=predictions,
        performance_dir=performance,
    )
    row = dataset.iloc[0]
    assert row["dataset_drift_score"] == 0.15
    assert row["max_feature_drift_score"] == 0.25
    assert row["number_drifted_features"] == 1
    assert row["missing_value_percentage"] == 5.0
    assert row["invalid_value_percentage"] == 1.5
    assert row["data_quality_score"] == 93.5
    assert pd.isna(row["historical_accuracy"])
    assert row["historical_metrics_available"] == 0
    assert row["historical_performance_observations"] == 0
    assert pd.isna(row["prediction_positive_rate"])
    assert pd.isna(row[DEGRADATION_TARGET])

    output = save_historical_dataset(dataset, output_dir=tmp_path / "output")
    loaded = pd.read_csv(output)
    assert output.is_file()
    assert len(loaded) == 1
    assert loaded.loc[0, "data_source_type"] == "real"
    summary = dataset_summary(dataset)
    assert summary["observation_count"] == 1
    assert summary["genuine_labeled_observation_count"] == 0


def test_prediction_distribution_is_available_only_at_or_after_prediction_time(
    tmp_path,
):
    reports = tmp_path / "reports"
    predictions = tmp_path / "predictions"
    performance = tmp_path / "performance"
    for folder in (reports, predictions, performance):
        folder.mkdir()
    _write_report(reports, "batch-1", "2026-01-01T00:00:00Z")
    (predictions / "batch-1.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "batch_id": "batch-1",
                    "row_id": f"row-{index}",
                    "observation_id": f"batch-1-observation-{index}",
                    "source": "production",
                    "predicted_at": "2026-01-02T00:00:00Z",
                    "predicted_class": predicted_class,
                    "probability_positive": float(predicted_class),
                    "model_version": "model1-version-1",
                }
            )
            for index, predicted_class in enumerate([0, 1])
        )
        + "\n",
        encoding="utf-8",
    )

    row = build_historical_dataset(
        reports_dir=reports,
        predictions_dir=predictions,
        performance_dir=performance,
    ).iloc[0]

    assert row["monitoring_report_timestamp"] == "2026-01-01T00:00:00+00:00"
    assert row["monitoring_timestamp"] == "2026-01-02T00:00:00+00:00"
    assert row["prediction_positive_rate"] == 0.5
    assert row["prediction_mean_probability"] == 0.5


def _build_report_row(
    tmp_path,
    *,
    expected_features,
    total_rows,
    missing_value_counts,
    missing_features=None,
    numerical_drift=None,
    categorical_drift=None,
):
    reports = tmp_path / "reports"
    predictions = tmp_path / "predictions"
    performance = tmp_path / "performance"
    for folder in (reports, predictions, performance):
        folder.mkdir()

    batch_id = "feature-calculation"
    _write_report(reports, batch_id, "2026-01-01T00:00:00Z")
    report_path = reports / f"{batch_id}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    quality = report["quality"]
    quality["total_rows"] = total_rows
    quality["feature_schema"] = {
        "expected_features": expected_features,
        "missing_features": missing_features or [],
    }
    quality["missing_value_counts"] = missing_value_counts
    report["drift"] = {
        "numerical": numerical_drift or {},
        "categorical": categorical_drift or {},
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    return build_historical_dataset(
        reports_dir=reports,
        predictions_dir=predictions,
        performance_dir=performance,
    ).iloc[0]


@pytest.mark.parametrize(
    ("expected_features", "missing_value_counts", "missing_features", "expected"),
    [
        (["a", "b"], {"a": 1, "b": 0}, [], 12.5),
        (["a", "b"], {"b": 0}, ["a"], 50.0),
        (["a", "b", "c"], {"c": 0}, ["a", "b"], 200.0 / 3.0),
    ],
)
def test_missing_value_percentage_counts_missing_cells_and_absent_columns(
    tmp_path,
    expected_features,
    missing_value_counts,
    missing_features,
    expected,
):
    row = _build_report_row(
        tmp_path,
        expected_features=expected_features,
        total_rows=4,
        missing_value_counts=missing_value_counts,
        missing_features=missing_features,
    )

    assert row["missing_value_percentage"] == pytest.approx(expected)


def test_empty_batch_has_unavailable_quality_percentages(tmp_path):
    row = _build_report_row(
        tmp_path,
        expected_features=["a", "b"],
        total_rows=0,
        missing_value_counts={},
        missing_features=["a", "b"],
    )

    assert pd.isna(row["missing_value_percentage"])
    assert pd.isna(row["invalid_value_percentage"])
    assert pd.isna(row["data_quality_score"])


@pytest.mark.parametrize(
    (
        "numerical_drift",
        "categorical_drift",
        "expected_average",
        "expected_maximum",
        "expected_count",
    ),
    [
        (
            {"a": {"usable": True, "psi": 0.05}},
            {"b": {"usable": True, "drift_score": 0.15}},
            0.10,
            0.15,
            0.0,
        ),
        (
            {"a": {"usable": True, "psi": 0.21}},
            {"b": {"usable": True, "drift_score": 0.31}},
            0.26,
            0.31,
            2.0,
        ),
    ],
)
def test_usable_drift_scores_are_aggregated_with_existing_critical_threshold(
    tmp_path,
    numerical_drift,
    categorical_drift,
    expected_average,
    expected_maximum,
    expected_count,
):
    row = _build_report_row(
        tmp_path,
        expected_features=["a", "b"],
        total_rows=4,
        missing_value_counts={"a": 0, "b": 0},
        numerical_drift=numerical_drift,
        categorical_drift=categorical_drift,
    )

    assert row["dataset_drift_score"] == pytest.approx(expected_average)
    assert row["max_feature_drift_score"] == pytest.approx(expected_maximum)
    assert row["number_drifted_features"] == expected_count


@pytest.mark.parametrize(
    "numerical_drift,categorical_drift",
    [
        (
            {"a": {"usable": False, "reason": "not available"}},
            {"b": {"usable": False, "reason": "not available"}},
        ),
        (
            {
                "a": {"usable": True, "psi": 0.1},
                "b": {"usable": False, "psi": 0.9},
                "c": {"usable": True, "psi": float("nan")},
            },
            {},
        ),
    ],
)
def test_unusable_drift_scores_are_unavailable_and_excluded(
    tmp_path, numerical_drift, categorical_drift
):
    row = _build_report_row(
        tmp_path,
        expected_features=["a", "b"],
        total_rows=4,
        missing_value_counts={"a": 0, "b": 0},
        numerical_drift=numerical_drift,
        categorical_drift=categorical_drift,
    )

    if not any(
        metrics.get("usable") and pd.notna(metrics.get("psi"))
        for metrics in numerical_drift.values()
    ):
        assert pd.isna(row["dataset_drift_score"])
        assert pd.isna(row["max_feature_drift_score"])
        assert pd.isna(row["number_drifted_features"])
    else:
        assert row["dataset_drift_score"] == pytest.approx(0.1)
        assert row["max_feature_drift_score"] == pytest.approx(0.1)
        assert row["number_drifted_features"] == 0.0


def _write_prediction_records(
    predictions_dir,
    batch_id,
    timestamp,
    predicted_classes,
    *,
    source,
    model_version,
    fixture_provenance=None,
):
    records = [
        {
            "batch_id": batch_id,
            "row_id": f"{batch_id}-{index}",
            **(
                {"observation_id": f"{batch_id}-{index}"}
                if source == "production"
                else {}
            ),
            "source": source,
            "predicted_at": timestamp,
            "predicted_class": predicted_class,
            "probability_positive": float(predicted_class),
            "model_version": model_version,
            **(
                {"fixture_provenance": fixture_provenance}
                if fixture_provenance
                else {}
            ),
        }
        for index, predicted_class in enumerate(predicted_classes)
    ]
    (predictions_dir / f"{batch_id}.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_prediction_change_uses_only_trusted_prior_predictions_of_same_model(
    tmp_path,
):
    reports = tmp_path / "reports"
    predictions = tmp_path / "predictions"
    performance = tmp_path / "performance"
    for folder in (reports, predictions, performance):
        folder.mkdir()

    _write_report(reports, "production-v1", "2026-01-01T00:00:00Z")
    _write_report(reports, "production-v2", "2026-01-02T00:00:00Z")
    _write_report(
        reports,
        "synthetic-batch",
        "2026-01-03T00:00:00Z",
        source="synthetic_test",
    )
    _write_report(
        reports,
        "unverified-batch",
        "2026-01-04T00:00:00Z",
        source="unknown",
    )
    _write_report(reports, "production-current", "2026-01-05T00:00:00Z")
    _write_prediction_records(
        predictions,
        "production-v1",
        "2026-01-01T00:00:00Z",
        [0, 0, 0, 1],
        source="production",
        model_version="model-v1",
    )
    _write_prediction_records(
        predictions,
        "production-v2",
        "2026-01-02T00:00:00Z",
        [1, 1],
        source="production",
        model_version="model-v2",
    )
    _write_prediction_records(
        predictions,
        "synthetic-batch",
        "2026-01-03T00:00:00Z",
        [1, 1],
        source="synthetic_test",
        model_version="model-v1",
    )
    _write_prediction_records(
        predictions,
        "unverified-batch",
        "2026-01-04T00:00:00Z",
        [1, 1],
        source="development",
        model_version="model-v1",
    )
    _write_prediction_records(
        predictions,
        "production-current",
        "2026-01-05T00:00:00Z",
        [0, 1],
        source="production",
        model_version="model-v1",
    )

    dataset = build_historical_dataset(
        reports_dir=reports,
        predictions_dir=predictions,
        performance_dir=performance,
    ).set_index("batch_id")
    current = dataset.loc["production-current"]

    assert current["prediction_positive_rate"] == 0.5
    assert current["prediction_mean_probability"] == 0.5
    assert current["prediction_positive_rate_change"] == pytest.approx(0.25)
    assert list(dataset.loc["production-current", MODEL2_FEATURE_NAMES].index) == (
        MODEL2_FEATURE_NAMES
    )
    feature_map = features_for_batch(
        "production-current",
        reports_dir=reports,
        predictions_dir=predictions,
        performance_dir=performance,
    )
    assert list(feature_map) == MODEL2_FEATURE_NAMES
    with pytest.raises(KeyError, match="required 'synthetic_test' source"):
        features_for_batch(
            "production-current",
            reports_dir=reports,
            predictions_dir=predictions,
            performance_dir=performance,
            expected_source="synthetic_test",
        )


def test_historical_performance_features_require_trusted_available_metrics(tmp_path):
    reports = tmp_path / "reports"
    predictions = tmp_path / "predictions"
    performance = tmp_path / "performance"
    for folder in (reports, predictions, performance):
        folder.mkdir()
    _write_report(reports, "performance-base", "2026-01-01T00:00:00Z")
    _write_report(reports, "performance-current", "2026-01-03T00:00:00Z")
    _write_prediction_records(
        predictions,
        "performance-base",
        "2026-01-01T00:00:00Z",
        [0, 1],
        source="production",
        model_version="model1-version-1",
    )
    _write_prediction_records(
        predictions,
        "performance-current",
        "2026-01-03T00:00:00Z",
        [0, 1],
        source="production",
        model_version="model1-version-1",
    )
    _write_performance(performance, "performance-base", "2026-01-02T00:00:00Z", 0.8)
    untrusted_metrics = {
        "batch_id": "performance-base",
        "evaluation_timestamp": "2026-01-02T12:00:00Z",
        "metrics_available": True,
        "number_of_actual_outcomes_received": 20,
        "data_source_type": "synthetic_test",
        "source": "synthetic_test",
        "model_version": "model-v1",
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1_score": 1.0,
    }
    (performance / "synthetic-metrics.json").write_text(
        json.dumps(untrusted_metrics), encoding="utf-8"
    )

    dataset = build_historical_dataset(
        reports_dir=reports,
        predictions_dir=predictions,
        performance_dir=performance,
    ).set_index("batch_id")
    current = dataset.loc["performance-current"]

    assert current["historical_accuracy"] == 0.8
    assert current["historical_precision"] == 0.8
    assert current["historical_recall"] == 0.8
    assert current["historical_f1_score"] == 0.8
    assert current["historical_metrics_available"] == 1.0
    assert current["historical_performance_observations"] == 1.0
