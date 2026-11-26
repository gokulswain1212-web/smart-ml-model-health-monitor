import json

import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from ml.model2_config import DEGRADATION_TARGET, MODEL2_FEATURE_NAMES
from ml.model2_development_training import (
    DEVELOPMENT_LABEL,
    DEVELOPMENT_PROVENANCE_COLUMN,
    evaluate_chronological_development_model,
    evaluate_synthetic_development_model,
    generate_synthetic_development_dataset,
    run_chronological_development_training,
    run_development_training,
)
from ml.model2_prediction import _latest_trained_artifact, predict_degradation
from ml.model2_training import train_model2


def test_development_training_saves_separate_tagged_artifacts(tmp_path):
    result = run_development_training(
        sample_count=400,
        random_state=17,
        output_dir=tmp_path / "development",
    )

    assert result["model2_status"] == "DEVELOPMENT_TRAINED"
    assert result["data_provenance"] == DEVELOPMENT_LABEL
    assert result["feature_names"] == MODEL2_FEATURE_NAMES
    assert (tmp_path / "development" / "synthetic_development_dataset_seed-17.csv").is_file()
    assert (tmp_path / "development" / f"{result['model_version']}.joblib").is_file()
    assert json.loads((tmp_path / "development" / f"{result['model_version']}.json").read_text())[
        "data_provenance"
    ] == DEVELOPMENT_LABEL


def test_development_dataset_and_metrics_are_reproducible():
    first = generate_synthetic_development_dataset(sample_count=500, random_state=91)
    second = generate_synthetic_development_dataset(sample_count=500, random_state=91)
    pd.testing.assert_frame_equal(first, second)

    _, first_metrics = evaluate_synthetic_development_model(first, random_state=91)
    _, second_metrics = evaluate_synthetic_development_model(second, random_state=91)
    assert first_metrics == second_metrics


def test_development_evaluation_metrics_match_holdout_calculations():
    dataset = generate_synthetic_development_dataset(
        sample_count=500,
        random_state=7,
    )
    _, metrics = evaluate_synthetic_development_model(dataset, random_state=7)
    features = dataset.loc[:, MODEL2_FEATURE_NAMES]
    labels = dataset[DEGRADATION_TARGET]
    _, test_features, _, test_labels = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=7,
        stratify=labels,
    )
    model, _ = evaluate_synthetic_development_model(dataset, random_state=7)
    predictions = model.predict(test_features)

    assert metrics["accuracy"] == accuracy_score(test_labels, predictions)
    assert metrics["precision"] == precision_score(
        test_labels, predictions, zero_division=0
    )
    assert metrics["recall"] == recall_score(test_labels, predictions, zero_division=0)
    assert metrics["f1_score"] == f1_score(test_labels, predictions, zero_division=0)
    assert metrics["confusion_matrix"] == confusion_matrix(
        test_labels, predictions, labels=[0, 1]
    ).tolist()
    assert metrics["classification_report"] == classification_report(
        test_labels,
        predictions,
        labels=[0, 1],
        target_names=["stable", "degraded"],
        output_dict=True,
        zero_division=0,
    )


def test_development_data_cannot_be_passed_to_production_training(tmp_path):
    dataset = generate_synthetic_development_dataset(
        sample_count=400,
        random_state=42,
    )
    production_shaped = dataset.drop(columns=[DEVELOPMENT_PROVENANCE_COLUMN]).copy()
    production_shaped["data_source_type"] = "synthetic_test"
    production_shaped["source"] = "production"
    production_shaped["batch_id"] = [
        f"development-{index}" for index in range(len(production_shaped))
    ]
    production_shaped["monitoring_timestamp"] = pd.date_range(
        "2026-01-01", periods=len(production_shaped), freq="D", tz="UTC"
    ).astype(str)
    production_shaped["target_future_timestamp"] = pd.date_range(
        "2026-01-02", periods=len(production_shaped), freq="D", tz="UTC"
    ).astype(str)

    result = train_model2(production_shaped, model_dir=tmp_path)

    assert result["model2_status"] == "NOT_TRAINED"
    assert result["historical_dataset"]["genuine_labeled_observation_count"] == 0
    assert not list(tmp_path.glob("*.joblib"))


def test_production_loader_and_prediction_ignore_nested_development_artifacts(
    tmp_path,
):
    development_dir = tmp_path / "development"
    result = run_development_training(
        sample_count=400,
        random_state=42,
        output_dir=development_dir,
    )
    assert _latest_trained_artifact(tmp_path) is None

    prediction = predict_degradation(
        {name: 0.5 for name in MODEL2_FEATURE_NAMES},
        model_dir=tmp_path,
    )
    assert prediction["model2_status"] == "NOT_TRAINED"
    assert prediction["model_version"] is None
    assert result["model2_status"] == "DEVELOPMENT_TRAINED"


def test_development_generator_rejects_invalid_size_and_no_class_dataset():
    with pytest.raises(ValueError, match="at least 20"):
        generate_synthetic_development_dataset(sample_count=19)
    with pytest.raises(ValueError, match="non-negative"):
        generate_synthetic_development_dataset(random_state=-1)


def test_chronological_development_training_requires_synthetic_tags_and_both_classes(
    tmp_path,
):
    timestamps = pd.date_range("2026-01-01", periods=24, freq="7D", tz="UTC")
    dataset = pd.DataFrame(
        {
            **{
                feature: [float(index % 5) for index in range(24)]
                for feature in MODEL2_FEATURE_NAMES
            },
            DEGRADATION_TARGET: [index % 2 for index in range(24)],
            DEVELOPMENT_PROVENANCE_COLUMN: DEVELOPMENT_LABEL,
            "batch_id": [f"synthetic-{index:02d}" for index in range(24)],
            "monitoring_timestamp": timestamps.astype(str),
            "data_source_type": "synthetic",
            "source": "synthetic_simulator",
            "environment": "development",
            "is_synthetic": True,
        }
    )

    _, metrics, holdout = evaluate_chronological_development_model(dataset)

    assert metrics["holdout_method"] == "chronological holdout"
    assert metrics["train_end_timestamp"] < metrics["test_start_timestamp"]
    assert set(metrics["training_class_counts"]) == {"0", "1"}
    assert set(metrics["testing_class_counts"]) == {"0", "1"}
    assert holdout["batch_id"].tolist() == dataset["batch_id"].iloc[18:].tolist()
    result = run_chronological_development_training(
        dataset,
        random_state=11,
        run_id="simulator-test",
        output_dir=tmp_path / "model2",
    )
    assert result["model2_status"] == "DEVELOPMENT_SIMULATION_TRAINED"
    assert result["source"] == "synthetic_simulator"
    assert (
        tmp_path
        / "model2"
        / f"{result['model_version']}.joblib"
    ).is_file()
    with pytest.raises(ValueError, match="at least 20"):
        evaluate_chronological_development_model(dataset.iloc[:19])

    untagged = dataset.copy()
    untagged["source"] = "production"
    with pytest.raises(ValueError, match="synthetic simulator"):
        evaluate_chronological_development_model(untagged)
