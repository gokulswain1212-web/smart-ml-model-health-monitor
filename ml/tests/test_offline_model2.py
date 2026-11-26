import json
from pathlib import Path

import pytest
import pandas as pd

import ml.offline_model2 as offline_model2_module
from ml.model2_config import DEGRADATION_TARGET, MODEL2_FEATURE_NAMES
from ml.model2_training import training_eligibility
from ml.offline_model2 import (
    PROVENANCE,
    _build_offline_observations,
    _model1_replay_version,
    _write_training_status,
    offline_training_eligibility,
    offline_model2_status,
    predict_offline_degradation,
    run_offline_model2_training,
)


def test_model1_replay_version_is_deterministic_and_tracks_input_changes(tmp_path):
    replay_input = tmp_path / "model1-input.json"
    replay_input.write_text('{"version": 1}', encoding="utf-8")

    version = _model1_replay_version(replay_input)

    assert version == _model1_replay_version(replay_input)
    replay_input.write_text('{"version": 2}', encoding="utf-8")
    assert version != _model1_replay_version(replay_input)


def _eligibility_dataset(batch_count=25):
    rows = []
    f1_values = [
        0.90 if index % 4 in (0, 1) else 0.70
        for index in range(batch_count)
    ]
    timestamps = [
        pd.Timestamp("2020-01-05T23:59:59Z") + pd.Timedelta(days=7 * index)
        for index in range(batch_count)
    ]
    for index in range(batch_count):
        baseline = f1_values[index - 1] if index > 0 else None
        future = f1_values[index + 1] if index + 1 < batch_count else None
        target = (
            int(baseline - future >= 0.05)
            if baseline is not None and future is not None
            else None
        )
        row = {
            "batch_id": f"isolated-offline-test-fixture-{index:03d}",
            "fixture_provenance": "SYNTHETIC_OFFLINE_TEST_FIXTURE_ONLY",
            "monitoring_timestamp": timestamps[index].isoformat(),
            "model1_replay_version": "offline-test-model1-v1",
            "target_future_timestamp": (
                timestamps[index + 1].isoformat()
                if target is not None
                else None
            ),
            "data_source_type": PROVENANCE,
            "source": "historical_kaggle_offline",
            "complete_verified_outcomes": True,
            "both_actual_classes": True,
            "prediction_count": 20,
            "actual_outcome_count": 20,
            "sample_count": 20,
            "pending_outcome_count": 0,
            "actual_class_0_count": 10,
            "actual_class_1_count": 10,
            "performance_eligible": True,
            "f1_score": f1_values[index],
            "accuracy": 0.8,
            "precision": 0.75,
            "recall": 0.7,
            "dataset_drift_score": 0.0,
            "average_ks_statistic": 0.0,
            "target_baseline_f1": baseline,
            "target_future_f1": future,
            "target_f1_drop": baseline - future if target is not None else None,
            DEGRADATION_TARGET: target,
        }
        row.update({name: float(index) for name in MODEL2_FEATURE_NAMES})
        rows.append(row)
    return pd.DataFrame(rows)


def test_offline_observations_keep_actual_outcomes_aligned_and_targets_future_only():
    reference = pd.DataFrame({"lead_time": [1.0, 4.0, 8.0]})
    outcome_dates = pd.date_range("2020-01-06", periods=6, freq="7D").repeat(2)
    future = pd.DataFrame(
        {
            "lead_time": [1.0, 4.0, 2.0, 5.0, 3.0, 8.0] * 2,
            "reservation_status_date": outcome_dates,
            "is_canceled": [0, 1] * 6,
        }
    )
    labels = future["is_canceled"].to_numpy()
    dataset, history = _build_offline_observations(
        reference,
        future,
        labels,
        [0.1, 0.9] * 6,
        feature_names=["lead_time"],
        numerical_features=["lead_time"],
        categorical_features=[],
        model1_replay_version="offline-test-model1-v1",
    )

    assert len(history) == 6
    assert dataset["batch_id"].is_unique
    assert dataset["monitoring_timestamp"].is_monotonic_increasing
    assert dataset["model1_replay_version"].eq("offline-test-model1-v1").all()
    assert "is_canceled" not in dataset.columns
    assert (dataset["prediction_count"] == dataset["actual_outcome_count"]).all()
    assert (dataset["actual_class_0_count"] == 1).all()
    assert (dataset["actual_class_1_count"] == 1).all()
    assert dataset[DEGRADATION_TARGET].notna().sum() == 4
    assert dataset.loc[dataset[DEGRADATION_TARGET].notna(), "target_future_timestamp"].notna().all()
    assert dataset.loc[1, "historical_metrics_available"] == 1


def test_offline_eligibility_requires_20_distinct_verified_batches_and_purges_overlap():
    dataset = _eligibility_dataset()
    result = offline_training_eligibility(dataset)

    assert result["status"] == "READY"
    assert result["eligible_observation_count"] == 23
    assert result["target_class_counts"] == {"0": 11, "1": 12}
    assert result["training_size_after_purge"] < 18
    assert result["training_class_counts"].keys() == {"0", "1"}
    assert result["testing_size"] == 5

    too_small = offline_training_eligibility(_eligibility_dataset(batch_count=20))
    assert too_small["status"] == "NOT_TRAINED"
    assert too_small["eligible_observation_count"] == 18
    assert any("at least 20" in item for item in too_small["missing_requirements"])


def test_offline_eligibility_rejects_duplicate_batches_without_counting_them_twice():
    dataset = _eligibility_dataset()
    dataset.loc[2, "batch_id"] = dataset.loc[1, "batch_id"]

    result = offline_training_eligibility(dataset)

    assert result["status"] == "NOT_TRAINED"
    assert result["eligible_observation_count"] == 22
    assert any("distinct batch IDs" in item for item in result["missing_requirements"])


def test_offline_eligibility_rejects_mixed_model1_replay_versions():
    dataset = _eligibility_dataset()
    dataset.loc[4, "model1_replay_version"] = "different-model1-replay"

    result = offline_training_eligibility(dataset)

    assert result["status"] == "NOT_TRAINED"
    assert any(
        "same identified Model 1 replay version" in item
        for item in result["missing_requirements"]
    )


def test_offline_eligibility_rejects_unknown_provenance_and_invalid_target_labels():
    dataset = _eligibility_dataset()
    dataset.loc[2, "data_source_type"] = "unknown"
    result = offline_training_eligibility(dataset)
    assert result["eligible_observation_count"] == 22
    assert result["status"] == "NOT_TRAINED"

    corrupted = _eligibility_dataset()
    label_row = corrupted.index[corrupted[DEGRADATION_TARGET].notna()][0]
    corrupted.loc[label_row, DEGRADATION_TARGET] = 1 - int(
        corrupted.loc[label_row, DEGRADATION_TARGET]
    )
    invalid_result = offline_training_eligibility(corrupted)
    assert invalid_result["status"] == "NOT_TRAINED"
    assert any(
        "target labels do not match" in item
        for item in invalid_result["missing_requirements"]
    )


def test_offline_eligibility_rejects_unordered_observations_and_stays_out_of_production():
    dataset = _eligibility_dataset()
    unordered = dataset.iloc[::-1].reset_index(drop=True)
    result = offline_training_eligibility(unordered)
    assert result["status"] == "NOT_TRAINED"
    assert any("uniquely chronological" in item for item in result["missing_requirements"])

    production_result = training_eligibility(dataset)
    assert production_result["eligible_labeled_observations"] == 0
    assert production_result["status"] == "NOT_TRAINED"


def test_offline_training_status_is_versioned_and_reports_not_trained_requirements(tmp_path):
    payload = {
        "model2_status": "NOT_TRAINED",
        "reason": "Need 20 eligible offline observations.",
        "eligible_observation_count": 0,
        "missing_requirements": ["Need 20 eligible offline observations."],
        "source_summary": {"evaluation_batch_count": 12, "performance_history": []},
    }

    _write_training_status(tmp_path, payload)
    _write_training_status(tmp_path, payload)
    status = offline_model2_status(tmp_path)

    assert len(list(tmp_path.glob("offline_model2_training_status_*.json"))) == 2
    assert status["status"] == "NOT_TRAINED"
    assert status["historical_observation_count"] == 12
    assert status["reason"] == payload["reason"]


def test_offline_training_saves_separate_artifact_and_predicts_from_fixture(
    tmp_path, monkeypatch
):
    dataset = _eligibility_dataset()
    source_summary = {
        "source": PROVENANCE,
        "model_scope": PROVENANCE,
        "model1_replay_version": "offline-test-model1-v1",
        "evaluation_batch_count": len(dataset),
        "performance_history": [],
    }
    monkeypatch.setattr(
        offline_model2_module,
        "build_offline_monitoring_dataset",
        lambda **_: (dataset, source_summary),
    )

    result = run_offline_model2_training(output_dir=tmp_path)

    assert result["model2_status"] == "TRAINED"
    assert result["model_scope"] == PROVENANCE
    assert result["provenance"] == PROVENANCE
    assert result["eligible_observation_count"] == 23
    assert result["target_class_counts"] == {"0": 11, "1": 12}
    assert result["training_size"] < 18
    assert result["testing_size"] == 5
    for artifact_key in ("model_path", "metadata_path", "evaluation_report_path"):
        assert pd.notna(result[artifact_key])
        assert (tmp_path / Path(result[artifact_key]).name).is_file()

    metadata = json.loads(
        Path(result["metadata_path"]).read_text(encoding="utf-8")
    )
    assert metadata["model_scope"] == PROVENANCE
    assert metadata["provenance"] == PROVENANCE
    assert metadata["model1_replay_version"] == "offline-test-model1-v1"
    assert metadata["training_source"] == "historical_kaggle_offline"
    assert metadata["temporal_split"]["training_target_end"] < (
        metadata["temporal_split"]["testing_start"]
    )

    status = offline_model2_status(tmp_path)
    assert status["status"] == "TRAINED"
    assert status["model_version"] == result["model_version"]
    assert status["model1_replay_version"] == "offline-test-model1-v1"
    assert len(status["evaluation_predictions"]) == 5

    features = {name: 0.5 for name in MODEL2_FEATURE_NAMES}
    prediction = predict_offline_degradation(features, model_dir=tmp_path)
    assert prediction["model2_status"] == "TRAINED"
    assert prediction["model_scope"] == PROVENANCE
    assert prediction["provenance"] == PROVENANCE
    assert prediction["model1_replay_version"] == "offline-test-model1-v1"
    assert pd.Timestamp(prediction["prediction_timestamp"]).tzinfo is not None
    assert 0 <= prediction["degradation_probability"] <= 1
    assert prediction["predicted_degradation_risk"] in (0, 1)


def test_offline_prediction_rejects_non_finite_features_before_loading_an_artifact(tmp_path):
    with pytest.raises(ValueError, match="must be finite"):
        predict_offline_degradation(
            {"dataset_drift_score": float("nan")},
            model_dir=tmp_path,
        )
