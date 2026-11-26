from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from ml.model2_config import DEGRADATION_TARGET, MODEL2_FEATURE_NAMES
from ml.model2_development_training import DEVELOPMENT_DIR, DEVELOPMENT_LABEL

ROOT = Path(__file__).resolve().parents[1]
TARGET_DEFINITION_PATH = ROOT / "ml" / "models" / "model2" / "target_definition.json"

FEATURE_RANGES: dict[str, tuple[float | None, float | None, bool]] = {
    "dataset_drift_score": (0.0, None, False),
    "max_feature_drift_score": (0.0, None, False),
    "number_drifted_features": (0.0, None, True),
    "missing_value_percentage": (0.0, 100.0, False),
    "invalid_value_percentage": (0.0, 100.0, False),
    "data_quality_score": (0.0, 100.0, False),
    "duplicate_rate": (0.0, 1.0, False),
    "prediction_positive_rate": (0.0, 1.0, False),
    "prediction_positive_rate_change": (-1.0, 1.0, False),
    "prediction_mean_probability": (0.0, 1.0, False),
    "historical_accuracy": (0.0, 1.0, False),
    "historical_precision": (0.0, 1.0, False),
    "historical_recall": (0.0, 1.0, False),
    "historical_f1_score": (0.0, 1.0, False),
    "previous_f1_trend": (-1.0, 1.0, False),
    "historical_metrics_available": (0.0, 1.0, True),
    "historical_performance_observations": (0.0, None, True),
}

UNAVAILABLE_FEATURE_NAMES = frozenset(
    {
        "dataset_drift_score",
        "max_feature_drift_score",
        "number_drifted_features",
        "missing_value_percentage",
        "invalid_value_percentage",
        "data_quality_score",
        "prediction_positive_rate",
        "prediction_positive_rate_change",
        "prediction_mean_probability",
        "historical_accuracy",
        "historical_precision",
        "historical_recall",
        "historical_f1_score",
        "previous_f1_trend",
    }
)


class DevelopmentModelError(RuntimeError):
    """The isolated synthetic development model is unavailable or invalid."""


class DevelopmentFeatureError(ValueError):
    """A supplied monitoring feature does not match the Model 2 contract."""


def validate_development_features(features: Mapping[str, Any]) -> pd.DataFrame:
    if not isinstance(features, Mapping):
        raise DevelopmentFeatureError("monitoring_features must be a JSON object.")

    provided_names = list(features.keys())
    missing = [name for name in MODEL2_FEATURE_NAMES if name not in features]
    extra = [name for name in provided_names if name not in MODEL2_FEATURE_NAMES]
    if missing:
        raise DevelopmentFeatureError(
            f"Missing required Model 2 features: {missing}."
        )
    if extra:
        raise DevelopmentFeatureError(f"Unexpected Model 2 features: {extra}.")
    if provided_names != MODEL2_FEATURE_NAMES:
        raise DevelopmentFeatureError(
            "Model 2 features must be supplied in the exact MODEL2_FEATURE_NAMES order."
        )

    values: dict[str, float] = {}
    for name in MODEL2_FEATURE_NAMES:
        value = features[name]
        if value is None:
            if name not in UNAVAILABLE_FEATURE_NAMES:
                raise DevelopmentFeatureError(
                    f"Feature {name!r} cannot be unavailable."
                )
            values[name] = np.nan
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DevelopmentFeatureError(f"Feature {name!r} must be a number.")
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            raise DevelopmentFeatureError(f"Feature {name!r} must be finite.")
        minimum, maximum, integer_only = FEATURE_RANGES[name]
        if minimum is not None and numeric_value < minimum:
            raise DevelopmentFeatureError(
                f"Feature {name!r} must be at least {minimum:g}."
            )
        if maximum is not None and numeric_value > maximum:
            raise DevelopmentFeatureError(
                f"Feature {name!r} must be at most {maximum:g}."
            )
        if integer_only and not numeric_value.is_integer():
            raise DevelopmentFeatureError(
                f"Feature {name!r} must be a whole number."
            )
        if name == "historical_metrics_available" and numeric_value not in (0, 1):
            raise DevelopmentFeatureError(
                "Feature 'historical_metrics_available' must be 0 or 1."
            )
        values[name] = numeric_value

    return pd.DataFrame([values], columns=MODEL2_FEATURE_NAMES)


def _read_development_artifact(
    development_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    metadata_paths = sorted(
        (
            path
            for path in development_dir.glob(
                "model2-development-random-forest-*.json"
            )
            if not path.stem.endswith("-metrics")
        ),
        reverse=True,
    )
    if not metadata_paths:
        raise FileNotFoundError(
            f"No Model 2 development metadata found under {development_dir}."
        )

    metadata_path = metadata_paths[0]
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DevelopmentModelError(
            f"Unable to read development model metadata {metadata_path.name}: {error}"
        ) from error
    if not isinstance(metadata, dict):
        raise DevelopmentModelError(
            f"Development model metadata {metadata_path.name} must be a JSON object."
        )
    if metadata.get("data_provenance") != DEVELOPMENT_LABEL:
        raise DevelopmentModelError(
            "Development model metadata lacks SYNTHETIC_DEVELOPMENT_ONLY provenance."
        )
    if metadata.get("model2_status") != "DEVELOPMENT_TRAINED":
        raise DevelopmentModelError(
            "Development model metadata does not identify a trained development model."
        )
    if metadata.get("algorithm") != "RandomForestClassifier":
        raise DevelopmentModelError(
            "Development model metadata does not identify a Random Forest classifier."
        )
    if metadata.get("feature_names") != MODEL2_FEATURE_NAMES:
        raise DevelopmentModelError(
            "Development model feature names do not match MODEL2_FEATURE_NAMES order."
        )
    if metadata.get("target_name") != DEGRADATION_TARGET:
        raise DevelopmentModelError(
            "Development model target does not match the configured Model 2 target."
        )
    if not isinstance(metadata.get("model_name"), str) or not metadata["model_name"]:
        raise DevelopmentModelError("Development model metadata has no model name.")
    model_version = metadata.get("model_version")
    if not isinstance(model_version, str) or model_version != metadata_path.stem:
        raise DevelopmentModelError(
            "Development model version does not match its metadata filename."
        )

    model_path = development_dir / f"{model_version}.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Development model artifact is missing: {model_path.name}."
        )
    try:
        model = joblib.load(model_path)
    except (
        AttributeError,
        EOFError,
        ImportError,
        KeyError,
        OSError,
        pickle.UnpicklingError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise DevelopmentModelError(
            f"Unable to load development model artifact {model_path.name}: {error}"
        ) from error

    estimator_features = getattr(model, "feature_names_in_", None)
    if estimator_features is None or list(estimator_features) != MODEL2_FEATURE_NAMES:
        raise DevelopmentModelError(
            "Development model estimator feature order does not match MODEL2_FEATURE_NAMES."
        )
    named_steps = getattr(model, "named_steps", {})
    imputer = named_steps.get("imputer") if hasattr(named_steps, "get") else None
    imputer_statistics = getattr(imputer, "statistics_", None)
    try:
        numeric_statistics = np.asarray(imputer_statistics, dtype=float)
        fitted_imputer_is_compatible = (
            imputer is not None
            and getattr(imputer, "strategy", None) == "median"
            and numeric_statistics.shape == (len(MODEL2_FEATURE_NAMES),)
            and np.isfinite(numeric_statistics).all()
        )
    except (TypeError, ValueError):
        fitted_imputer_is_compatible = False
    if not fitted_imputer_is_compatible:
        raise DevelopmentModelError(
            "Development model does not contain a compatible fitted median imputer."
        )
    if not callable(getattr(model, "predict_proba", None)):
        raise DevelopmentModelError(
            "Development model does not support class probability predictions."
        )
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise DevelopmentModelError(
            "Development model classes do not match the target's stable/degraded mapping."
        )
    try:
        class_values = np.asarray(classes).tolist()
        if set(class_values) != {0, 1}:
            raise DevelopmentModelError(
                "Development model classes do not match the target's stable/degraded mapping."
            )
    except (TypeError, ValueError) as error:
        raise DevelopmentModelError(
            "Development model has an invalid class mapping."
        ) from error

    return model, metadata


def _read_target_definition(target_definition_path: Path) -> dict[int, str]:
    try:
        target_definition = json.loads(
            target_definition_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DevelopmentModelError(
            f"Unable to read Model 2 target definition: {error}"
        ) from error
    if not isinstance(target_definition, dict):
        raise DevelopmentModelError("Model 2 target definition must be a JSON object.")
    if target_definition.get("target") != DEGRADATION_TARGET:
        raise DevelopmentModelError(
            "Model 2 target definition does not match the configured target."
        )
    stable_value = target_definition.get("stable_value")
    degraded_value = target_definition.get("degraded_value")
    if (
        isinstance(stable_value, bool)
        or isinstance(degraded_value, bool)
        or stable_value != 0
        or degraded_value != 1
    ):
        raise DevelopmentModelError(
            "Model 2 target mapping must define stable=0 and degraded=1."
        )
    return {stable_value: "stable", degraded_value: "degraded"}


def predict_development_degradation(
    features: Mapping[str, Any],
    *,
    development_dir: Path | None = None,
    target_definition_path: Path | None = None,
) -> dict[str, Any]:
    development_dir = development_dir or DEVELOPMENT_DIR
    target_definition_path = target_definition_path or TARGET_DEFINITION_PATH
    feature_row = validate_development_features(features)
    model, metadata = _read_development_artifact(development_dir)
    class_labels = _read_target_definition(target_definition_path)
    try:
        probabilities = np.asarray(model.predict_proba(feature_row), dtype=float)
        classes = list(np.asarray(model.classes_).tolist())
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as error:
        raise DevelopmentModelError(
            f"Development model prediction failed: {error}"
        ) from error
    if (
        probabilities.ndim != 2
        or probabilities.shape != (1, len(classes))
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.isclose(probabilities.sum(axis=1)[0], 1.0, atol=1e-6)
    ):
        raise DevelopmentModelError(
            "Development model returned invalid class probabilities."
        )

    class_probabilities = {
        class_labels[int(class_value)]: float(probabilities[0, index])
        for index, class_value in enumerate(classes)
    }
    predicted_class = int(classes[int(np.argmax(probabilities[0]))])
    predicted_label = class_labels[predicted_class]
    unavailable_features = [
        name for name in MODEL2_FEATURE_NAMES if pd.isna(feature_row.iloc[0][name])
    ]
    return {
        "api_status": "DEVELOPMENT_PREDICTION",
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "model_provenance": DEVELOPMENT_LABEL,
        "target_name": DEGRADATION_TARGET,
        "prediction": predicted_class,
        "predicted_class_label": predicted_label,
        "class_probabilities": class_probabilities,
        "unavailable_features": unavailable_features,
        "production_validated": False,
        "risk_assessment": {
            "predicted_class": predicted_label,
            "degradation_probability": class_probabilities["degraded"],
            "risk_level": None,
        },
        "explanation": (
            "Unavailable monitoring features were passed to the development "
            "model's fitted median imputer. "
            if unavailable_features
            else ""
        )
        + (
            "This prediction is produced by a synthetic development-only model, "
            "is not production validated, and is not production health status or "
            "production risk evidence."
        ),
        "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
    }
