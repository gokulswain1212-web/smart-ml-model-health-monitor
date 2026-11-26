import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from ml.model2_config import DEGRADATION_TARGET
from ml.model2_training import training_eligibility
from ml.synthetic_hotel_simulator import (
    PROVENANCE,
    generate_synthetic_booking_batch,
    run_synthetic_hotel_simulation,
)


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_synthetic_booking_generation_is_reproducible_and_tagged():
    timestamp = datetime(2026, 1, 5, tzinfo=timezone.utc)
    first = generate_synthetic_booking_batch(
        batch_id="reproducible-batch",
        batch_index=0,
        batch_size=40,
        seed=19,
        timestamp=timestamp,
    )
    second = generate_synthetic_booking_batch(
        batch_id="reproducible-batch",
        batch_index=0,
        batch_size=40,
        seed=19,
        timestamp=timestamp,
    )

    pd.testing.assert_frame_equal(first, second)
    assert first["booking_id"].is_unique
    assert first["batch_id"].eq("reproducible-batch").all()
    assert first["is_canceled"].nunique() == 2
    for field, value in PROVENANCE.items():
        assert first[field].eq(value).all()


def test_simulation_is_isolated_chronological_and_excluded_from_production_gate(
    tmp_path,
):
    summary = run_synthetic_hotel_simulation(
        batch_count=24,
        batch_size=50,
        seed=42,
        output_root=tmp_path / "development-simulation",
    )
    run_dir = tmp_path / "development-simulation" / summary["run_id"]

    assert summary["run_status"] == "COMPLETE"
    assert summary["mode"] == "DEVELOPMENT SIMULATION"
    assert summary["data_source_type"] == "synthetic"
    assert summary["source"] == "synthetic_simulator"
    assert summary["environment"] == "development"
    assert summary["is_synthetic"] is True
    assert summary["batch_count"] >= 20
    assert summary["booking_count"] == 24 * 50
    assert summary["minimum_batch_f1_score"] < summary["baseline_f1_score"]
    assert summary["drift_detected_batch_count"] > 0

    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    outcomes = _read_jsonl(run_dir / "outcomes.jsonl")
    assert len(predictions) == len(outcomes) == summary["booking_count"]
    outcome_by_observation = {
        row["observation_id"]: row for row in outcomes
    }
    assert len(outcome_by_observation) == len(outcomes)
    for prediction in predictions:
        outcome = outcome_by_observation[prediction["observation_id"]]
        assert prediction["batch_id"] == outcome["batch_id"]
        assert prediction["booking_id"] == outcome["booking_id"]
        assert prediction["timestamp"] == outcome["timestamp"]
        for field, value in PROVENANCE.items():
            assert prediction[field] == outcome[field] == value

    batch_files = sorted((run_dir / "batches").glob("*.csv"))
    assert len(batch_files) == summary["batch_count"]
    batch_timestamps = []
    for batch_file in batch_files:
        batch = pd.read_csv(batch_file)
        assert batch["booking_id"].is_unique
        assert batch["observation_id"].is_unique
        assert batch["batch_id"].nunique() == 1
        assert batch["is_canceled"].nunique() == 2
        batch_timestamps.append(pd.Timestamp(batch["timestamp"].iloc[0]))
        for field, value in PROVENANCE.items():
            assert batch[field].eq(value).all()
    assert batch_timestamps == sorted(batch_timestamps)

    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_dir / "reports").glob("*.json"))
    ]
    assert len(reports) == summary["batch_count"]
    assert all(report["data_source_type"] == "synthetic" for report in reports)
    assert all(report["source"] == "synthetic_simulator" for report in reports)
    assert all(report["environment"] == "development" for report in reports)
    assert all(report["is_synthetic"] is True for report in reports)
    assert any(
        result.get("ks_statistic") is not None
        for report in reports
        for result in report["drift"]["numerical"].values()
    )

    model2_dataset = pd.read_csv(run_dir / "model2" / "synthetic_model2_dataset.csv")
    assert model2_dataset[DEGRADATION_TARGET].nunique() == 2
    assert model2_dataset["monitoring_timestamp"].is_monotonic_increasing
    for field, value in PROVENANCE.items():
        assert model2_dataset[field].eq(value).all()
    eligibility = training_eligibility(model2_dataset)
    assert eligibility["status"] == "NOT_TRAINED"
    assert eligibility["eligible_labeled_observations"] == 0
    assert summary["model2_training"]["split_method"] == "chronological holdout"
    assert set(summary["model2_training"]["training_class_counts"]) == {"0", "1"}
    assert set(summary["model2_training"]["testing_class_counts"]) == {"0", "1"}
    holdout = pd.read_csv(
        run_dir / "model2" / "chronological_holdout_predictions.csv"
    )
    assert holdout["monitoring_timestamp"].min() > model2_dataset.iloc[
        summary["model2_training"]["training_size"] - 1
    ]["monitoring_timestamp"]


def test_simulator_rejects_fewer_than_twenty_batches(tmp_path):
    with pytest.raises(ValueError, match="at least 20"):
        run_synthetic_hotel_simulation(
            batch_count=19,
            output_root=tmp_path / "simulation",
        )
