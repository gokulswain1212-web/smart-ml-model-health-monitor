import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.simulate_data import generate_batches, save_batches, simulate_data


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = ROOT / "data" / "raw" / "hotel_bookings.csv"
FEATURE_NAMES_PATH = ROOT / "ml" / "models" / "feature_names.json"
PIPELINE_PATH = ROOT / "ml" / "models" / "hotel_cancellation_pipeline.pkl"


@pytest.fixture(scope="module")
def simulation_inputs():
    feature_names = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
    reference_data = pd.read_csv(
        REFERENCE_PATH,
        usecols=lambda column: column in set(feature_names),
    ).loc[:, feature_names]
    pipeline = joblib.load(PIPELINE_PATH)
    return feature_names, reference_data, pipeline


def test_simulation_is_reproducible(simulation_inputs):
    feature_names, reference_data, _ = simulation_inputs
    first = generate_batches(reference_data, feature_names, batch_size=40, seed=867)
    second = generate_batches(reference_data, feature_names, batch_size=40, seed=867)

    for batch_type in ("normal", "drifted", "poor_quality"):
        pd.testing.assert_frame_equal(first[batch_type], second[batch_type])


def test_batches_have_expected_features_and_shape(simulation_inputs):
    feature_names, reference_data, _ = simulation_inputs
    batches = generate_batches(reference_data, feature_names, batch_size=40, seed=12)

    assert list(batches) == ["normal", "drifted", "poor_quality"]
    for batch in batches.values():
        assert batch.shape == (40, 28)
        assert batch.columns.tolist() == feature_names
        assert "is_canceled" not in batch.columns


def test_drifted_batch_changes_numeric_and_categorical_distributions(simulation_inputs):
    feature_names, reference_data, _ = simulation_inputs
    batches = generate_batches(reference_data, feature_names, batch_size=500, seed=7)
    normal = batches["normal"]
    drifted = batches["drifted"]

    assert drifted["lead_time"].mean() != pytest.approx(normal["lead_time"].mean())
    assert drifted["adr"].mean() != pytest.approx(normal["adr"].mean())
    assert drifted["market_segment"].value_counts(normalize=True).get("Direct", 0) > (
        normal["market_segment"].value_counts(normalize=True).get("Direct", 0)
    )
    assert drifted["hotel"].value_counts(normalize=True).get("Resort Hotel", 0) > (
        normal["hotel"].value_counts(normalize=True).get("Resort Hotel", 0)
    )


def test_poor_quality_batch_contains_missing_invalid_and_unseen_values(simulation_inputs):
    feature_names, reference_data, _ = simulation_inputs
    poor_quality = generate_batches(
        reference_data, feature_names, batch_size=50, seed=31
    )["poor_quality"]

    assert poor_quality.isna().any().any()
    assert poor_quality.loc[0, "lead_time"] == -999
    assert poor_quality.loc[1, "market_segment"] == "UNEXPECTED_MARKET_SEGMENT"
    assert poor_quality.loc[2, "adr"] == "not-a-number"


def test_saved_pipeline_processes_normal_and_drifted_batches(simulation_inputs):
    feature_names, reference_data, pipeline = simulation_inputs
    batches = generate_batches(reference_data, feature_names, batch_size=32, seed=20)

    for batch_type in ("normal", "drifted"):
        predictions = pipeline.predict(batches[batch_type].loc[:, feature_names])
        assert predictions.shape == (32,)
        assert np.isin(predictions, pipeline.classes_).all()


def test_saved_batches_include_ids_timestamps_and_features(
    simulation_inputs, tmp_path
):
    feature_names, reference_data, _ = simulation_inputs
    batches = generate_batches(reference_data, feature_names, batch_size=12, seed=4)
    paths = save_batches(batches, tmp_path, seed=4)

    assert set(paths) == {"normal", "drifted", "poor_quality"}
    for batch_type, path in paths.items():
        saved = pd.read_csv(path)
        assert saved["batch_id"].nunique() == 1
        assert saved["batch_id"].iloc[0].startswith(f"{batch_type}-")
        assert saved["generated_at"].nunique() == 1
        assert saved["timestamp"].nunique() == 1
        assert saved["source"].eq("original_hotel_booking_dataset_simulation").all()
        assert saved["data_source_type"].eq("synthetic_test").all()
        assert saved["record_count"].eq(len(saved)).all()
        assert saved[feature_names].columns.tolist() == feature_names


def test_simulation_preserves_source_labels_and_expected_feature_order(
    simulation_inputs,
):
    feature_names, _, _ = simulation_inputs
    reference = pd.read_csv(
        REFERENCE_PATH,
        usecols=lambda column: column in {*feature_names, "is_canceled"},
    ).loc[:, [*feature_names, "is_canceled"]]
    batches = generate_batches(
        reference,
        feature_names,
        batch_size=40,
        seed=867,
        target_column="is_canceled",
    )
    first_rng = np.random.default_rng(np.random.SeedSequence(867).spawn(3)[0])
    positions = first_rng.choice(len(reference), size=40, replace=True)

    assert batches["normal"].columns.tolist() == [*feature_names, "is_canceled"]
    pd.testing.assert_series_equal(
        batches["normal"]["is_canceled"],
        reference.iloc[positions]["is_canceled"].reset_index(drop=True),
        check_names=False,
    )
    for batch in batches.values():
        assert batch["is_canceled"].dropna().isin([0, 1]).all()


def test_simulate_data_persists_quality_drift_and_model_predictions(
    tmp_path, simulation_inputs
):
    result = simulate_data(
        batch_size=24,
        seed=713,
        output_dir=tmp_path,
    )

    assert len(result["files"]) == 3
    assert set(result["record_counts"].values()) == {24}
    assert result["pipeline_prediction_counts"] == {"normal": 24, "drifted": 24}
    assert set(result["monitoring"]) == {"normal", "drifted", "poor_quality"}
    assert result["monitoring"]["poor_quality"]["quality_status"] == "CRITICAL"
    assert result["monitoring"]["poor_quality"]["model_prediction"]["performed"] is False
    assert result["monitoring"]["normal"]["model_prediction"]["performed"] is True
    assert result["monitoring"]["drifted"]["model_prediction"]["performed"] is True

    report_files = list((tmp_path / "reports").glob("*.json"))
    assert len(report_files) == 3
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_files]
    assert {
        report["batch_metadata"]["source"] for report in reports
    } == {"original_hotel_booking_dataset_simulation"}
    assert all(report["quality"]["total_rows"] == 24 for report in reports)
    assert all(report["batch_metadata"]["record_count"] == 24 for report in reports)

    prediction_files = list((tmp_path / "predictions").glob("*.jsonl"))
    assert len(prediction_files) == 2
    predictions = [
        json.loads(line)
        for path in prediction_files
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(predictions) == 48
    assert all(record["source"] == "synthetic_test" for record in predictions)
    assert all(record["probability_positive"] is not None for record in predictions)
    assert all(record["model_version"] for record in predictions)
    assert all(record["predicted_at"] for record in predictions)
