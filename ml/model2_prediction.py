from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

import joblib
import numpy as np
import pandas as pd

from ml.model2_config import (
    DEGRADATION_CLASSIFICATION_THRESHOLD,
    HIGH_RISK_MIN_PROBABILITY,
    LOW_RISK_MAX_PROBABILITY,
    MIN_LABELED_OBSERVATIONS,
    MODEL2_FEATURE_NAMES,
)
from ml.model2_dataset import (
    PERFORMANCE_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    build_historical_dataset,
)
from ml.model2_training import MODEL2_DIR


class Model2ArtifactError(RuntimeError):
    """A saved Model 2 artifact is missing or cannot be used safely."""


def _latest_trained_artifact(
    model_dir: Path,
) -> tuple[Path, dict[str, Any]] | None:
    for metadata_path in sorted(model_dir.glob("model2-random-forest-*.json"), reverse=True):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise Model2ArtifactError(
                f"Unable to read Model 2 metadata {metadata_path.name}: {error}"
            ) from error
        if not isinstance(metadata, dict):
            raise Model2ArtifactError(
                f"Model 2 metadata {metadata_path.name} must contain a JSON object."
            )
        if metadata.get("model2_status") != "TRAINED":
            continue
        model_path = metadata_path.with_suffix(".joblib")
        if metadata.get("model_version") != metadata_path.stem:
            raise Model2ArtifactError(
                f"Model 2 metadata {metadata_path.name} has an inconsistent model version."
            )
        if metadata.get("feature_names") != MODEL2_FEATURE_NAMES:
            raise Model2ArtifactError(
                f"Model 2 metadata {metadata_path.name} has an incompatible feature schema."
            )
        historical_dataset = metadata.get("historical_dataset")
        trained_observation_count = (
            historical_dataset.get("genuine_labeled_observation_count")
            if isinstance(historical_dataset, dict)
            else None
        )
        if (
            metadata.get("training_source")
            != "verified_genuine_production_monitoring"
            or not isinstance(trained_observation_count, int)
            or isinstance(trained_observation_count, bool)
            or trained_observation_count < MIN_LABELED_OBSERVATIONS
        ):
            raise Model2ArtifactError(
                f"Model 2 metadata {metadata_path.name} does not verify the minimum "
                f"{MIN_LABELED_OBSERVATIONS} genuine labeled production observations."
            )
        importances = metadata.get("feature_importance")
        if not isinstance(importances, list) or any(
            not isinstance(item, dict)
            or item.get("feature") not in MODEL2_FEATURE_NAMES
            or isinstance(item.get("importance"), bool)
            or not isinstance(item.get("importance"), (int, float))
            or not np.isfinite(item["importance"])
            or item["importance"] < 0
            for item in importances
        ):
            raise Model2ArtifactError(
                f"Model 2 metadata {metadata_path.name} has invalid feature importances."
            )
        if not model_path.is_file():
            raise Model2ArtifactError(
                f"Trained Model 2 artifact {model_path.name} is missing."
            )
        return model_path, metadata
    return None


def _validate_features(features: Mapping[str, Any]) -> pd.DataFrame:
    if not isinstance(features, Mapping):
        raise TypeError("Model 2 input features must be a mapping.")
    unexpected = set(features) - set(MODEL2_FEATURE_NAMES)
    if unexpected:
        raise ValueError(f"Unexpected Model 2 features: {sorted(unexpected)}.")

    values: dict[str, float] = {}
    for name in MODEL2_FEATURE_NAMES:
        raw_value = features.get(name)
        if raw_value is None:
            values[name] = np.nan
            continue
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Model 2 feature {name!r} must be numeric or null.") from error
        if not np.isfinite(numeric_value):
            raise ValueError(f"Model 2 feature {name!r} must be finite or null.")
        values[name] = numeric_value
    return pd.DataFrame([values], columns=MODEL2_FEATURE_NAMES)


def _top_signals(
    metadata: dict[str, Any],
    feature_row: pd.DataFrame,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    importances = metadata.get("feature_importance", [])
    return [
        {
            "feature": item["feature"],
            "importance": float(item["importance"]),
            "observed_value": (
                None
                if pd.isna(feature_row.iloc[0][item["feature"]])
                else float(feature_row.iloc[0][item["feature"]])
            ),
        }
        for item in importances[:limit]
    ]


def predict_degradation(
    features: Mapping[str, Any],
    *,
    model_dir: Path = MODEL2_DIR,
    classification_threshold: float | None = None,
    low_risk_max_probability: float = LOW_RISK_MAX_PROBABILITY,
    high_risk_min_probability: float = HIGH_RISK_MIN_PROBABILITY,
) -> dict[str, Any]:
    if not 0 <= low_risk_max_probability < high_risk_min_probability <= 1:
        raise ValueError("Risk thresholds must satisfy 0 <= low < high <= 1.")
    feature_row = _validate_features(features)
    artifact = _latest_trained_artifact(model_dir)
    if artifact is None:
        return {
            "model2_status": "NOT_TRAINED",
            "predicted_degradation_risk": None,
            "degradation_probability": None,
            "risk_category": None,
            "important_contributing_signals": [],
            "explanation": (
                "Model 2 has not been trained because sufficient genuine labeled "
                "historical observations are unavailable."
            ),
            "model_version": None,
        }

    model_path, metadata = artifact
    try:
        model = joblib.load(model_path)
        classes = list(model.classes_)
        estimator_features = getattr(model, "feature_names_in_", None)
        named_steps = getattr(model, "named_steps", {})
        imputer = named_steps.get("imputer") if hasattr(named_steps, "get") else None
        imputer_statistics = np.asarray(
            getattr(imputer, "statistics_", None), dtype=float
        )
        if estimator_features is None or list(estimator_features) != MODEL2_FEATURE_NAMES:
            raise Model2ArtifactError(
                "Saved Model 2 estimator feature order does not match MODEL2_FEATURE_NAMES."
            )
        if (
            imputer is None
            or getattr(imputer, "strategy", None) != "median"
            or imputer_statistics.shape != (len(MODEL2_FEATURE_NAMES),)
            or not np.isfinite(imputer_statistics).all()
        ):
            raise Model2ArtifactError(
                "Saved Model 2 estimator does not contain a compatible fitted median imputer."
            )
        if set(classes) != {0, 1}:
            raise Model2ArtifactError(
                "Saved Model 2 estimator classes do not match the stable/degraded mapping."
            )
        probabilities = np.asarray(model.predict_proba(feature_row), dtype=float)
    except Model2ArtifactError:
        raise
    except (
        AttributeError,
        EOFError,
        ImportError,
        IndexError,
        KeyError,
        OSError,
        pickle.UnpicklingError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise Model2ArtifactError(
            f"Unable to load or use Model 2 artifact {model_path.name}: {error}"
        ) from error
    if (
        probabilities.ndim != 2
        or probabilities.shape != (1, len(classes))
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise Model2ArtifactError(
            "Saved Model 2 estimator returned invalid class probabilities."
        )
    probability = float(probabilities[0, classes.index(1)])
    decision_threshold = (
        float(metadata.get("training_configuration", {}).get(
            "degradation_classification_threshold",
            DEGRADATION_CLASSIFICATION_THRESHOLD,
        ))
        if classification_threshold is None
        else classification_threshold
    )
    if not 0 <= decision_threshold <= 1:
        raise ValueError("classification_threshold must be between 0 and 1.")
    prediction = int(probability >= decision_threshold)
    if probability < low_risk_max_probability:
        risk_category = "LOW"
    elif probability >= high_risk_min_probability:
        risk_category = "HIGH"
    else:
        risk_category = "MEDIUM"
    health_status = {
        "LOW": "HEALTHY",
        "MEDIUM": "WARNING",
        "HIGH": "CRITICAL",
    }[risk_category]

    return {
        "model2_status": "TRAINED",
        "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
        "predicted_degradation_risk": prediction,
        "degradation_probability": probability,
        "risk_category": risk_category,
        "health_status": health_status,
        "important_contributing_signals": _top_signals(metadata, feature_row),
        "explanation": (
            "Signals are ranked by global Random Forest feature importance; this "
            "describes model behavior and does not establish causality."
        ),
        "model_version": metadata["model_version"],
    }


def features_for_batch(
    batch_id: str,
    *,
    reports_dir: Path = REPORTS_DIR,
    predictions_dir: Path = PREDICTIONS_DIR,
    performance_dir: Path = PERFORMANCE_DIR,
    expected_source: Literal["production", "synthetic_test"] = "production",
) -> dict[str, float | None]:
    dataset = build_historical_dataset(
        reports_dir=reports_dir,
        predictions_dir=predictions_dir,
        performance_dir=performance_dir,
    )
    matching = dataset.loc[dataset["batch_id"] == batch_id]
    if matching.empty or not {"source", "data_source_type"}.issubset(dataset.columns):
        raise KeyError(
            f"No monitoring report for batch {batch_id!r} has the required "
            f"{expected_source!r} source."
        )
    required_source_type = (
        "real" if expected_source == "production" else "synthetic_test"
    )
    matching = matching.loc[
        matching["source"].eq(expected_source)
        & matching["data_source_type"].eq(required_source_type)
    ]
    if matching.empty:
        raise KeyError(
            f"No monitoring report for batch {batch_id!r} has the required "
            f"{expected_source!r} source."
        )
    row = matching.iloc[-1]
    return {
        name: (None if pd.isna(row[name]) else float(row[name]))
        for name in MODEL2_FEATURE_NAMES
    }


def predict_batch_degradation(
    batch_id: str,
    *,
    model_dir: Path = MODEL2_DIR,
    reports_dir: Path = REPORTS_DIR,
    predictions_dir: Path = PREDICTIONS_DIR,
    performance_dir: Path = PERFORMANCE_DIR,
) -> dict[str, Any]:
    features = features_for_batch(
        batch_id,
        reports_dir=reports_dir,
        predictions_dir=predictions_dir,
        performance_dir=performance_dir,
    )
    result = predict_degradation(features, model_dir=model_dir)
    return {"batch_id": batch_id, **result}
