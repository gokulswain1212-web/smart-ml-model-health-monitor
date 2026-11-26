import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ml.model2_prediction as prediction_module
import ml.model2_training as training_module
from ml.model2_config import DEGRADATION_TARGET, MODEL2_FEATURE_NAMES
from ml.model2_dataset import build_historical_dataset, dataset_summary
from ml.model2_prediction import predict_degradation
from ml.model2_training import (
    run_training_if_eligible,
    train_model2,
    training_eligibility,
)

VERIFIED_BATCH_EVIDENCE = {
    "batch_prediction_count": 2,
    "batch_actual_outcome_count": 2,
    "batch_pending_outcome_count": 0,
    "batch_actual_class_0_count": 1,
    "batch_actual_class_1_count": 1,
    "batch_outcome_provenance": "verified_production_api",
}


def _trained_metadata(model_version, *, feature_importance=None):
    return {
        "model2_status": "TRAINED",
        "model_version": model_version,
        "training_source": "verified_genuine_production_monitoring",
        "feature_names": MODEL2_FEATURE_NAMES,
        "historical_dataset": {"genuine_labeled_observation_count": 20},
        "feature_importance": feature_importance
        or [{"feature": MODEL2_FEATURE_NAMES[0], "importance": 1.0}],
    }


def test_training_is_gated_without_genuine_labeled_history(tmp_path):
    dataset = build_historical_dataset()
    result = train_model2(dataset, model_dir=tmp_path)

    assert result["model2_status"] == "NOT_TRAINED"
    assert result["historical_dataset"]["genuine_labeled_observation_count"] == 0
    assert result["training_size"] == 0
    assert result["testing_size"] == 0
    assert not list(tmp_path.glob("*.joblib"))
    assert list(tmp_path.glob("training_status_*.json"))
    assert json.loads((tmp_path / "feature_names.json").read_text())["feature_names"] == (
        MODEL2_FEATURE_NAMES
    )
    assert (tmp_path / "target_definition.json").is_file()


def test_training_rejects_synthetic_target_rows(tmp_path):
    rows = []
    for index in range(24):
        row = {name: float(index % 5) for name in MODEL2_FEATURE_NAMES}
        row.update(
            {
                "batch_id": f"synthetic-{index}",
                "monitoring_timestamp": f"2026-01-{index + 1:02d}T00:00:00Z",
                "target_future_timestamp": (
                    f"2026-02-{index + 1:02d}T00:00:00Z"
                ),
                "data_source_type": "synthetic_test",
                "source": "production",
                "label_provenance": None,
                DEGRADATION_TARGET: index % 2,
            }
        )
        rows.append(row)
    dataset = pd.DataFrame(rows)

    result = train_model2(
        dataset,
        model_dir=tmp_path,
        minimum_labeled_observations=20,
    )

    assert result["model2_status"] == "NOT_TRAINED"
    assert "Need at least" in result["reason"]
    assert not list(tmp_path.glob("*.joblib"))


def test_training_minimum_cannot_be_lowered(tmp_path):
    dataset = build_historical_dataset()

    with pytest.raises(ValueError, match="cannot be lower than 20"):
        train_model2(dataset, model_dir=tmp_path, minimum_labeled_observations=4)

    with pytest.raises(ValueError, match="cannot be lower than 20"):
        training_eligibility(dataset, minimum_labeled_observations=4)


@pytest.mark.parametrize(
    (
        "real_labeled_count",
        "synthetic_labeled_count",
        "real_unlabeled_count",
        "expected_labeled_count",
        "expected_status",
    ),
    [
        (0, 0, 0, 0, "NOT_TRAINED"),
        (10, 0, 0, 10, "NOT_TRAINED"),
        (19, 0, 0, 19, "NOT_TRAINED"),
        (20, 0, 0, 20, "READY"),
        (19, 1, 0, 19, "NOT_TRAINED"),
        (19, 0, 1, 19, "NOT_TRAINED"),
    ],
)
def test_training_eligibility_counts_only_labeled_real_production_batches(
    real_labeled_count,
    synthetic_labeled_count,
    real_unlabeled_count,
    expected_labeled_count,
    expected_status,
):
    rows = []
    row_groups = (
        ("real", "production", real_labeled_count, True),
        ("synthetic_test", "synthetic_test", synthetic_labeled_count, True),
        ("real", "production", real_unlabeled_count, False),
    )
    index = 0
    for data_source_type, source, count, is_labeled in row_groups:
        for _ in range(count):
            monitoring_time = (
                pd.Timestamp("2026-01-01T00:00:00Z")
                + pd.Timedelta(days=index)
            )
            row = {name: float(index % 5) for name in MODEL2_FEATURE_NAMES}
            row.update(
                {
                    "batch_id": f"eligibility-fixture-{index}",
                    "monitoring_timestamp": monitoring_time.isoformat(),
                    "target_future_timestamp": (
                        monitoring_time + pd.Timedelta(days=1)
                    ).isoformat(),
                    "data_source_type": data_source_type,
                    "source": source,
                    "label_provenance": (
                        "verified_production_api"
                        if data_source_type == "real" and is_labeled
                        else None
                    ),
                    DEGRADATION_TARGET: index % 2 if is_labeled else None,
                    **VERIFIED_BATCH_EVIDENCE,
                }
            )
            rows.append(row)
            index += 1

    dataset = pd.DataFrame(rows)
    result = training_eligibility(dataset)

    assert result["eligible_labeled_observations"] == expected_labeled_count
    assert result["status"] == expected_status
    if expected_status == "NOT_TRAINED":
        assert f"found {expected_labeled_count}" in result["reason"]


def test_training_eligibility_rejects_incomplete_or_single_class_batches():
    rows = []
    for index in range(20):
        monitoring_time = pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(
            days=index
        )
        evidence = VERIFIED_BATCH_EVIDENCE.copy()
        if index == 0:
            evidence["batch_actual_class_1_count"] = 0
        row = {name: float(index) for name in MODEL2_FEATURE_NAMES}
        row.update(
            {
                "batch_id": f"eligibility-evidence-{index}",
                "monitoring_timestamp": monitoring_time.isoformat(),
                "target_future_timestamp": (
                    monitoring_time + pd.Timedelta(days=1)
                ).isoformat(),
                "data_source_type": "real",
                "source": "production",
                "label_provenance": "verified_production_api",
                DEGRADATION_TARGET: index % 2,
                **evidence,
            }
        )
        rows.append(row)

    result = training_eligibility(pd.DataFrame(rows))

    assert result["status"] == "NOT_TRAINED"
    assert result["eligible_labeled_observations"] == 19
    assert "Need at least 20" in result["reason"]


def test_training_persists_rf_and_uses_chronological_holdout_for_unit_fixture(tmp_path):
    rows = []
    for index in range(40):
        target = index % 2
        row = {
            name: (
                None
                if feature_index == 1
                else float(target if feature_index == 0 else index % 7)
            )
            for feature_index, name in enumerate(MODEL2_FEATURE_NAMES)
        }
        row.update(
            {
                "batch_id": f"isolated-unit-fixture-{index}",
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
                DEGRADATION_TARGET: target,
                **VERIFIED_BATCH_EVIDENCE,
            }
        )
        rows.append(row)
    dataset = pd.DataFrame(rows)

    result = train_model2(
        dataset,
        model_dir=tmp_path,
        minimum_labeled_observations=20,
        test_fraction=0.25,
    )

    assert result["model2_status"] == "TRAINED"
    assert result["training_size"] == 29
    assert result["testing_size"] == 10
    metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["temporal_split"]["training_target_end"] < (
        metadata["temporal_split"]["testing_start"]
    )
    assert metadata["temporal_split"]["training_end"] < (
        metadata["temporal_split"]["testing_start"]
    )
    assert metadata["evaluation_metrics"]["random_forest_primary_candidate"][
        "confusion_matrix"
    ]
    assert metadata["evaluation_metrics"]["random_forest_primary_candidate"][
        "roc_auc"
    ] is not None
    assert Path(result["model_path"]).is_file()
    prediction = predict_degradation(
        {name: 0.5 for name in MODEL2_FEATURE_NAMES},
        model_dir=tmp_path,
    )
    assert prediction["model2_status"] == "TRAINED"
    assert prediction["model_version"] == result["model_version"]

    leaked_dataset = dataset.copy()
    leaked_dataset.loc[0, "target_future_timestamp"] = leaked_dataset.loc[
        0, "monitoring_timestamp"
    ]
    rejected = train_model2(
        leaked_dataset,
        model_dir=tmp_path / "leaked",
        minimum_labeled_observations=20,
        test_fraction=0.25,
    )
    assert rejected["model2_status"] == "NOT_TRAINED"
    assert "must occur after" in rejected["reason"]
    assert not list((tmp_path / "leaked").glob("*.joblib"))


def test_automatic_training_waits_for_eligible_real_production_rows(
    tmp_path, monkeypatch
):
    rows = []
    for index in range(45):
        row = {name: float(index % 5) for name in MODEL2_FEATURE_NAMES}
        row.update(
            {
                "batch_id": f"synthetic-or-unlabeled-{index}",
                "monitoring_timestamp": (
                    pd.Timestamp("2026-01-01T00:00:00Z")
                    + pd.Timedelta(days=index)
                ).isoformat(),
                "target_future_timestamp": (
                    pd.Timestamp("2026-01-02T00:00:00Z")
                    + pd.Timedelta(days=index)
                ).isoformat(),
                "data_source_type": "synthetic_test" if index < 21 else "real",
                "source": "production",
                "label_provenance": (
                    "verified_production_api"
                    if index >= 21 and index < 40
                    else None
                ),
                DEGRADATION_TARGET: index % 2 if index < 40 else None,
                **VERIFIED_BATCH_EVIDENCE,
            }
        )
        rows.append(row)
    dataset = pd.DataFrame(rows)
    monkeypatch.setattr(
        training_module, "build_historical_dataset", lambda: dataset
    )
    monkeypatch.setattr(training_module, "MODEL2_DIR", tmp_path / "models")

    result = run_training_if_eligible()

    assert result["model2_status"] == "NOT_TRAINED"
    assert result["historical_dataset"]["genuine_labeled_observation_count"] == 19
    assert "Need at least 20" in result["reason"]
    assert not list((tmp_path / "models").glob("*.joblib"))
    status_path = next((tmp_path / "models").glob("training_status_*.json"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["model2_status"] == "NOT_TRAINED"
    assert status["training_performed"] is False
    assert status["training_attempt_timestamp"]


def test_automatic_training_saves_one_production_model_when_eligible(
    tmp_path, monkeypatch
):
    rows = []
    for index in range(20):
        target = index % 2
        row = {
            name: (
                None
                if feature_index == 1
                else float(target if feature_index == 0 else index % 7)
            )
            for feature_index, name in enumerate(MODEL2_FEATURE_NAMES)
        }
        row.update(
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
                DEGRADATION_TARGET: target,
                **VERIFIED_BATCH_EVIDENCE,
            }
        )
        rows.append(row)
    dataset = pd.DataFrame(rows)
    model_dir = tmp_path / "models"
    monkeypatch.setattr(
        training_module, "build_historical_dataset", lambda: dataset
    )
    monkeypatch.setattr(training_module, "MODEL2_DIR", model_dir)
    monkeypatch.setattr(
        training_module, "MODEL2_DATA_DIR", tmp_path / "datasets"
    )

    result = run_training_if_eligible()
    repeated = run_training_if_eligible()

    assert result["model2_status"] == "TRAINED"
    assert result["training_eligibility"] == "TRAINED"
    assert result["historical_dataset"]["genuine_labeled_observation_count"] == 20
    metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["model2_status"] == "TRAINED"
    assert metadata["historical_dataset"]["genuine_labeled_observation_count"] == 20
    assert Path(result["model_path"]).is_file()
    assert repeated["model_version"] == result["model_version"]
    assert repeated["training_timestamp"] == result["training_timestamp"]
    assert result["training_performed"] is True
    assert repeated["training_performed"] is False
    assert "no retraining was needed" in repeated["reason"]
    assert len(list(model_dir.glob("model2-random-forest-*.joblib"))) == 1
    status_files = sorted(model_dir.glob("training_status_*.json"))
    assert len(status_files) == 2
    assert json.loads(status_files[-1].read_text(encoding="utf-8"))[
        "training_performed"
    ] is False


def test_prediction_returns_not_trained_without_artifact(tmp_path):
    features = {name: None for name in MODEL2_FEATURE_NAMES}
    result = predict_degradation(features, model_dir=tmp_path)

    assert result["model2_status"] == "NOT_TRAINED"
    assert result["predicted_degradation_risk"] is None
    assert result["degradation_probability"] is None
    assert result["risk_category"] is None
    assert result["important_contributing_signals"] == []


def test_prediction_rejects_corrupted_or_missing_trained_artifact(tmp_path):
    features = {name: None for name in MODEL2_FEATURE_NAMES}
    metadata_path = tmp_path / "model2-random-forest-v1.json"
    metadata_path.write_text("{", encoding="utf-8")

    with pytest.raises(prediction_module.Model2ArtifactError, match="Unable to read"):
        predict_degradation(features, model_dir=tmp_path)

    metadata_path.write_text(
        json.dumps(
            _trained_metadata(metadata_path.stem)
        ),
        encoding="utf-8",
    )
    with pytest.raises(prediction_module.Model2ArtifactError, match="is missing"):
        predict_degradation(features, model_dir=tmp_path)

    metadata_path.with_suffix(".joblib").write_bytes(b"not a valid model artifact")
    with pytest.raises(prediction_module.Model2ArtifactError, match="Unable to load"):
        predict_degradation(features, model_dir=tmp_path)


def test_prediction_rejects_artifact_without_minimum_genuine_training_evidence(
    tmp_path, monkeypatch
):
    metadata_path = tmp_path / "model2-random-forest-v1.json"
    metadata = _trained_metadata(metadata_path.stem)
    metadata["historical_dataset"]["genuine_labeled_observation_count"] = 19
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    metadata_path.with_suffix(".joblib").touch()
    monkeypatch.setattr(
        prediction_module.joblib,
        "load",
        lambda _: pytest.fail("An ineligible artifact must not be loaded."),
    )

    with pytest.raises(
        prediction_module.Model2ArtifactError,
        match="minimum 20 genuine labeled production observations",
    ):
        predict_degradation(
            {name: 0.5 for name in MODEL2_FEATURE_NAMES},
            model_dir=tmp_path,
        )


@pytest.mark.parametrize(
    ("classes", "feature_names", "message"),
    [
        ([0, 2], MODEL2_FEATURE_NAMES, "stable/degraded mapping"),
        ([0, 1], MODEL2_FEATURE_NAMES[::-1], "feature order"),
    ],
)
def test_prediction_rejects_incompatible_fitted_estimator(
    tmp_path, monkeypatch, classes, feature_names, message
):
    class FakeImputer:
        strategy = "median"
        statistics_ = np.zeros(len(MODEL2_FEATURE_NAMES))

    class FakeEstimator:
        classes_ = np.asarray(classes)
        feature_names_in_ = np.asarray(feature_names)
        named_steps = {"imputer": FakeImputer()}

        def predict_proba(self, frame):
            return np.tile([[0.5, 0.5]], (len(frame), 1))

    metadata_path = tmp_path / "model2-random-forest-invalid.json"
    metadata_path.write_text(
        json.dumps(_trained_metadata(metadata_path.stem)),
        encoding="utf-8",
    )
    metadata_path.with_suffix(".joblib").touch()
    monkeypatch.setattr(
        prediction_module.joblib, "load", lambda _: FakeEstimator()
    )

    with pytest.raises(prediction_module.Model2ArtifactError, match=message):
        predict_degradation(
            {name: 0.5 for name in MODEL2_FEATURE_NAMES},
            model_dir=tmp_path,
        )


@pytest.mark.parametrize(
    ("probability", "category"),
    [(0.1, "LOW"), (0.5, "MEDIUM"), (0.9, "HIGH")],
)
def test_prediction_probability_risk_categories_and_explanations(
    tmp_path, monkeypatch, probability, category
):
    class FakeClassifier:
        classes_ = np.array([0, 1])
        feature_names_in_ = np.array(MODEL2_FEATURE_NAMES)
        named_steps = {
            "imputer": type(
                "FakeImputer",
                (),
                {
                    "strategy": "median",
                    "statistics_": np.zeros(len(MODEL2_FEATURE_NAMES)),
                },
            )()
        }

        def predict_proba(self, frame):
            return np.tile([[1 - probability, probability]], (len(frame), 1))

    model_path = tmp_path / "model2-random-forest-test.joblib"
    metadata_path = tmp_path / "model2-random-forest-test.json"
    model_path.touch()
    monkeypatch.setattr(
        prediction_module.joblib, "load", lambda _: FakeClassifier()
    )
    metadata_path.write_text(
        json.dumps(_trained_metadata(metadata_path.stem)),
        encoding="utf-8",
    )
    features = {name: 0.5 for name in MODEL2_FEATURE_NAMES}

    result = predict_degradation(features, model_dir=tmp_path)

    assert result["model2_status"] == "TRAINED"
    assert result["degradation_probability"] == pytest.approx(probability)
    assert result["risk_category"] == category
    assert result["predicted_degradation_risk"] == int(probability >= 0.5)
    assert result["important_contributing_signals"][0]["feature"] == (
        MODEL2_FEATURE_NAMES[0]
    )
    assert "does not establish causality" in result["explanation"]


def test_prediction_rejects_unexpected_or_non_numeric_features(tmp_path):
    features = {name: None for name in MODEL2_FEATURE_NAMES}
    features["unexpected"] = 1
    with pytest.raises(ValueError, match="Unexpected Model 2 features"):
        predict_degradation(features, model_dir=tmp_path)

    features = {name: None for name in MODEL2_FEATURE_NAMES}
    features[MODEL2_FEATURE_NAMES[0]] = "not-numeric"
    with pytest.raises(ValueError, match="must be numeric"):
        predict_degradation(features, model_dir=tmp_path)


def test_dataset_summary_counts_only_non_null_targets():
    dataset = pd.DataFrame(
        {
            **{name: [1.0, 2.0] for name in MODEL2_FEATURE_NAMES},
            DEGRADATION_TARGET: pd.array([1, pd.NA], dtype="Int64"),
            "data_source_type": ["real", "synthetic_test"],
            "source": ["production", "production"],
            "label_provenance": ["verified_production_api", None],
        }
    )
    summary = dataset_summary(dataset)
    assert summary["genuine_labeled_observation_count"] == 1
    assert summary["unlabeled_observation_count"] == 1
    assert summary["target_class_counts"] == {"1": 1}
