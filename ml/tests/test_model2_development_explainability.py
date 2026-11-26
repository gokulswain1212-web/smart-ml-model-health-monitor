import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import ml.model2_development_explainability as explainability
from ml.model2_config import DEGRADATION_TARGET, MODEL2_FEATURE_NAMES
from ml.model2_development_training import (
    DEVELOPMENT_LABEL,
    generate_synthetic_development_dataset,
)
from ml.model2_prediction import _latest_trained_artifact


@pytest.fixture
def development_artifact(tmp_path):
    dataset = generate_synthetic_development_dataset(
        sample_count=160,
        random_state=42,
    )
    features = dataset.loc[:, MODEL2_FEATURE_NAMES]
    labels = dataset[DEGRADATION_TARGET]
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "classifier",
                RandomForestClassifier(n_estimators=16, random_state=42),
            ),
        ]
    )
    model.fit(features, labels)
    version = "model2-development-random-forest-test"
    model_path = tmp_path / f"{version}.joblib"
    joblib.dump(model, model_path)
    metadata = {
        "model2_status": "DEVELOPMENT_TRAINED",
        "data_provenance": DEVELOPMENT_LABEL,
        "model_version": version,
        "feature_names": MODEL2_FEATURE_NAMES,
    }
    (tmp_path / f"{version}.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    return tmp_path, model


def test_extracts_actual_aligned_numeric_feature_importance(development_artifact):
    development_dir, model = development_artifact

    report = explainability.extract_development_feature_importance(
        development_dir=development_dir
    )
    actual = model.named_steps["classifier"].feature_importances_

    assert report["provenance"] == DEVELOPMENT_LABEL
    assert report["model_version"] == "model2-development-random-forest-test"
    assert report["feature_names"] == MODEL2_FEATURE_NAMES
    assert [feature["feature"] for feature in report["features"]] == [
        row[0]
        for row in sorted(
            zip(MODEL2_FEATURE_NAMES, actual, strict=True),
            key=lambda item: (-float(item[1]), item[0]),
        )
    ]
    assert np.isclose(report["importance_sum"], 1.0)
    assert len(report["top_features"]) == 10
    assert [feature["rank"] for feature in report["top_features"]] == list(range(1, 11))


def test_rejects_misaligned_feature_schema(development_artifact):
    development_dir, _ = development_artifact
    metadata_path = next(development_dir.glob("model2-development-random-forest-*.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["feature_names"] = list(reversed(MODEL2_FEATURE_NAMES))
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(
        explainability.DevelopmentExplainabilityError,
        match="feature order",
    ):
        explainability.extract_development_feature_importance(
            development_dir=development_dir
        )


def test_rejects_fitted_estimator_feature_order_mismatch(monkeypatch, tmp_path):
    class FakeImputer:
        feature_names_in_ = np.asarray(list(reversed(MODEL2_FEATURE_NAMES)))

    class FakeClassifier:
        feature_importances_ = np.asarray([1.0] + [0.0] * 16)

    metadata = {
        "model2_status": "DEVELOPMENT_TRAINED",
        "data_provenance": DEVELOPMENT_LABEL,
        "model_version": "test-version",
        "feature_names": MODEL2_FEATURE_NAMES,
    }
    fake_model = Pipeline(
        [
            ("imputer", FakeImputer()),
            ("classifier", FakeClassifier()),
        ]
    )
    monkeypatch.setattr(
        explainability,
        "_read_development_model",
        lambda _: (tmp_path / "model.joblib", metadata, fake_model),
    )

    with pytest.raises(
        explainability.DevelopmentExplainabilityError,
        match="Fitted estimator feature order",
    ):
        explainability.extract_development_feature_importance(
            development_dir=tmp_path
        )


@pytest.mark.parametrize(
    ("importance_values", "message"),
    [
        ([float("nan")] + [0.0] * 16, "finite and non-negative"),
        ([-0.1] + [0.0] * 16, "finite and non-negative"),
        ([0.1] * 17, "sum approximately to 1"),
        ([0.5] * 16, "length does not match"),
    ],
)
def test_rejects_invalid_importance_values(
    monkeypatch, tmp_path, importance_values, message
):
    class FakeClassifier:
        feature_importances_ = np.asarray(importance_values, dtype=float)

    class FakeImputer:
        feature_names_in_ = np.asarray(MODEL2_FEATURE_NAMES)

    fake_model = Pipeline(
        [
            ("imputer", FakeImputer()),
            ("classifier", FakeClassifier()),
        ]
    )
    metadata = {
        "model2_status": "DEVELOPMENT_TRAINED",
        "data_provenance": DEVELOPMENT_LABEL,
        "model_version": "test-version",
        "feature_names": MODEL2_FEATURE_NAMES,
    }
    monkeypatch.setattr(
        explainability,
        "_read_development_model",
        lambda _: (tmp_path / "model.joblib", metadata, fake_model),
    )

    with pytest.raises(explainability.DevelopmentExplainabilityError, match=message):
        explainability.extract_development_feature_importance(
            development_dir=tmp_path
        )


def test_missing_and_invalid_model_artifacts_are_reported(development_artifact):
    development_dir, _ = development_artifact
    model_path = next(development_dir.glob("*.joblib"))
    model_path.unlink()
    with pytest.raises(FileNotFoundError, match="artifact is missing"):
        explainability.extract_development_feature_importance(
            development_dir=development_dir
        )

    model_path.write_bytes(b"not a valid joblib artifact")
    with pytest.raises(
        explainability.DevelopmentExplainabilityError,
        match="Unable to load development model artifact",
    ):
        explainability.extract_development_feature_importance(
            development_dir=development_dir
        )


def test_writes_tagged_json_chart_and_explanation_report(development_artifact, tmp_path):
    development_dir, _ = development_artifact
    report_dir = tmp_path / "reports"

    result = explainability.generate_development_explainability(
        development_dir=development_dir,
        report_dir=report_dir,
    )

    saved = json.loads(
        (development_dir / "feature_importance.json").read_text(encoding="utf-8")
    )
    assert saved["provenance"] == DEVELOPMENT_LABEL
    assert saved["feature_names"] == MODEL2_FEATURE_NAMES
    assert saved["model_version"] == result["model_version"]
    assert saved["generation_timestamp"]
    assert (report_dir / "feature_importance_top10.png").is_file()
    report = (report_dir / "feature_importance_explanations.md").read_text(
        encoding="utf-8"
    )
    assert "does not show that a feature causes" in report
    assert DEVELOPMENT_LABEL in report


def test_production_loader_does_not_discover_development_model(development_artifact):
    development_dir, _ = development_artifact

    assert _latest_trained_artifact(development_dir.parent) is None
