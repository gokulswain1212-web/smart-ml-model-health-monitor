import json
from pathlib import Path

import joblib
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "ml" / "models"
DATASET = ROOT / "data" / "processed" / "cleane_hotel.csv"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
MODEL_PATH = MODEL_DIR / "hotel_cancellation_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
PIPELINE_PATH = MODEL_DIR / "hotel_cancellation_pipeline.pkl"


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    feature_names = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
    return model, preprocessor, pipeline, feature_names


def test_model_artifacts_exist():
    for path in (
        MODEL_PATH,
        PREPROCESSOR_PATH,
        PIPELINE_PATH,
        FEATURE_NAMES_PATH,
        METADATA_PATH,
    ):
        assert path.is_file(), f"Missing model artifact: {path}"


def test_metadata_records_training_and_evaluation_details():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    assert metadata["target_column"] == "is_canceled"
    assert metadata["algorithm"] == "RandomForestClassifier"
    assert metadata["feature_names"] == json.loads(
        FEATURE_NAMES_PATH.read_text(encoding="utf-8")
    )
    assert 0.0 <= metadata["evaluation_metrics"]["accuracy"] <= 1.0
    assert metadata["environment"]["python_version"]
    assert metadata["environment"]["scikit_learn_version"]


def test_saved_model_and_preprocessing_are_loadable():
    model, preprocessor, pipeline, feature_names = load_artifacts()

    assert hasattr(model, "predict")
    assert hasattr(preprocessor, "transform")
    assert hasattr(pipeline, "predict")
    assert feature_names == preprocessor.feature_names
    assert feature_names == list(model.feature_names_in_)


def test_preprocessing_and_prediction_preserve_feature_order():
    model, preprocessor, pipeline, feature_names = load_artifacts()
    raw = pd.read_csv(DATASET, nrows=8)
    raw_features = raw.loc[:, feature_names].iloc[::-1]

    transformed = preprocessor.transform(raw_features)
    predictions = pipeline.predict(raw_features)
    expected_predictions = model.predict(transformed)

    assert list(transformed.columns) == feature_names
    assert transformed.shape == (len(raw_features), len(feature_names))
    assert predictions.shape == (len(raw_features),)
    assert set(predictions).issubset(set(model.classes_))
    assert (predictions == expected_predictions).all()


def test_preprocessing_fills_missing_values():
    _, preprocessor, _, feature_names = load_artifacts()
    raw = pd.read_csv(DATASET)
    sample_indices = sorted(
        {
            raw.index[raw["children"].isna()][0],
            raw.index[raw["agent"].isna()][0],
        }
    )

    transformed = preprocessor.transform(raw.loc[sample_indices, feature_names])

    missing_children_index = raw.index[raw["children"].isna()][0]
    missing_agent_index = raw.index[raw["agent"].isna()][0]
    assert transformed.loc[missing_children_index, "children"] == preprocessor.children_median
    assert transformed.loc[missing_agent_index, "agent"] == 0


def test_missing_required_feature_is_rejected():
    _, preprocessor, _, feature_names = load_artifacts()
    raw = pd.read_csv(DATASET, nrows=1).loc[:, feature_names]

    with pytest.raises(ValueError, match="Missing required feature columns"):
        preprocessor.transform(raw.drop(columns=feature_names[0]))
