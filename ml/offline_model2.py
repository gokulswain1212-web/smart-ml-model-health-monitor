from __future__ import annotations

import hashlib
import json
import os
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from ml.hotel_preprocessing import HotelCancellationPreprocessor
from ml.model2_config import (
    DEGRADATION_CLASSIFICATION_THRESHOLD,
    DEGRADATION_TARGET,
    F1_DEGRADATION_THRESHOLD,
    FUTURE_PERFORMANCE_WINDOW_DAYS,
    MIN_LABELED_OBSERVATIONS,
    MODEL2_FEATURE_NAMES,
    TEST_FRACTION,
)
from ml.monitoring_config import DEFAULT_THRESHOLDS, NON_NEGATIVE_FEATURES
from ml.offline_drift_analysis import compare_offline_features

ROOT = Path(__file__).resolve().parents[1]
MODEL1_METADATA_PATH = ROOT / "ml" / "models" / "model_metadata.json"
MODEL1_PIPELINE_PATH = ROOT / "ml" / "models" / "hotel_cancellation_pipeline.pkl"
HISTORICAL_DATA_PATH = ROOT / "data" / "processed" / "hotel_bookings_historical.csv"
FUTURE_DATA_PATH = ROOT / "data" / "processed" / "hotel_bookings_future_evaluation.csv"
OFFLINE_MODEL2_DIR = ROOT / "ml" / "models" / "offline_model2"
OFFLINE_DATASET_NAME = "historical_offline_monitoring_dataset.csv"
PROVENANCE = "HISTORICAL_OFFLINE"
MODEL_SCOPE = "HISTORICAL_OFFLINE"
STATUS_GLOB = "offline_model2_training_status_*.json"
MODEL1_REPLAY_RECIPE_VERSION = "hotel-booking-historical-refit-v1"

_MAPPINGS: dict[str, dict[str, int]] = {
    "hotel": {"Resort Hotel": 1, "City Hotel": 0},
    "arrival_date_month": {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    },
    "meal": {"BB": 1, "FB": 2, "HB": 3, "SC": 4, "Undefined": 0},
}


class OfflineModel2ArtifactError(RuntimeError):
    """A historical offline Model 2 artifact is missing or invalid."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OfflineModel2ArtifactError(
            f"Unable to read offline Model 2 metadata {path.name}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise OfflineModel2ArtifactError(
            f"Offline Model 2 metadata {path.name} must contain a JSON object."
        )
    return value


def _model1_replay_version(*paths: Path) -> str:
    digest = hashlib.sha256()
    digest.update(MODEL1_REPLAY_RECIPE_VERSION.encode("utf-8"))
    digest.update(sklearn.__version__.encode("utf-8"))
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
    return f"historical-refit-{digest.hexdigest()[:16]}"


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("x", encoding="utf-8", newline="\n") as file:
            json.dump(_json_safe(payload), file, indent=2, allow_nan=False)
            file.write("\n")
        os.link(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_training_status(output_dir: Path, payload: dict[str, Any]) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    _write_json_new(
        output_dir / f"offline_model2_training_status_{timestamp}.json",
        payload,
    )


def _save_csv_new(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary_path, index=False)
        if path.exists():
            if temporary_path.read_bytes() != path.read_bytes():
                raise FileExistsError(
                    f"Refusing to overwrite different offline Model 2 data at {path}."
                )
            return
        os.link(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _read_model1_metadata(path: Path) -> dict[str, Any]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read Model 1 metadata {path}: {error}") from error
    if not isinstance(metadata, dict):
        raise ValueError("Model 1 metadata must contain a JSON object.")
    return metadata


def _fit_historical_model1(
    historical: pd.DataFrame,
    future: pd.DataFrame,
    *,
    metadata_path: Path = MODEL1_METADATA_PATH,
    pipeline_path: Path = MODEL1_PIPELINE_PATH,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    metadata = _read_model1_metadata(metadata_path)
    feature_names = metadata.get("feature_names")
    numerical_features = metadata.get("numerical_features")
    categorical_features = metadata.get("categorical_features")
    target_column = metadata.get("target_column")
    if (
        not isinstance(feature_names, list)
        or not feature_names
        or not isinstance(numerical_features, list)
        or not isinstance(categorical_features, list)
        or target_column != "is_canceled"
    ):
        raise ValueError("Model 1 metadata has an invalid feature schema.")
    if set(numerical_features) | set(categorical_features) != set(feature_names):
        raise ValueError("Model 1 feature types do not match its feature schema.")
    if set(numerical_features) & set(categorical_features):
        raise ValueError("Model 1 feature types overlap.")
    forbidden = {"is_canceled", "reservation_status", "reservation_status_date"}
    if forbidden.intersection(feature_names):
        raise ValueError("Model 1 feature schema contains a target or post-outcome field.")
    for frame_name, frame in (("historical", historical), ("future", future)):
        missing = [name for name in feature_names if name not in frame]
        if missing:
            raise ValueError(f"{frame_name} dataset is missing Model 1 features: {missing}")

    training = historical.loc[:, feature_names].copy()
    evaluation = future.loc[:, feature_names].copy()
    fill_values = {
        name: str(training[name].mode(dropna=True).iloc[0])
        for name in categorical_features
        if training[name].isna().any()
    }
    children_median = float(
        pd.to_numeric(training["children"], errors="coerce").median()
    )
    if not np.isfinite(children_median):
        raise ValueError("Historical Model 1 training data has no valid children values.")

    clipping_bounds: dict[str, tuple[float, float]] = {}
    for feature in numerical_features:
        values = pd.to_numeric(training[feature], errors="coerce")
        lower_quartile = values.quantile(0.25)
        upper_quartile = values.quantile(0.75)
        if pd.isna(lower_quartile) or pd.isna(upper_quartile):
            continue
        spread = float(upper_quartile - lower_quartile)
        clipping_bounds[feature] = (
            float(lower_quartile - 1.5 * spread),
            float(upper_quartile + 1.5 * spread),
        )

    def clip_using_historical_bounds(frame: pd.DataFrame) -> pd.DataFrame:
        clipped = frame.copy()
        for feature, (lower, upper) in clipping_bounds.items():
            clipped[feature] = pd.to_numeric(
                clipped[feature], errors="coerce"
            ).clip(lower=lower, upper=upper)
        return clipped

    training = clip_using_historical_bounds(training)
    evaluation = clip_using_historical_bounds(evaluation)

    encoder_input = training.copy()
    for name, value in fill_values.items():
        encoder_input[name] = encoder_input[name].fillna(value)
    encoder_input["children"] = encoder_input["children"].fillna(children_median)
    encoder_input["agent"] = encoder_input["agent"].fillna(0)
    for name, mapping in _MAPPINGS.items():
        encoded = encoder_input[name].map(mapping)
        invalid = encoded.isna() & encoder_input[name].notna()
        if invalid.any():
            raise ValueError(
                f"Historical data has unsupported values in {name!r}: "
                f"{encoder_input.loc[invalid, name].drop_duplicates().tolist()}"
            )
        encoder_input[name] = encoded

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.float64,
    )
    encoder.fit(encoder_input.loc[:, categorical_features].astype(object))
    preprocessor = HotelCancellationPreprocessor(
        encoder,
        feature_names,
        categorical_features,
        children_median,
        fill_values,
    )
    try:
        saved_pipeline = joblib.load(pipeline_path)
    except (OSError, EOFError, ImportError, pickle.UnpicklingError) as error:
        raise ValueError(f"Unable to load Model 1 pipeline: {error}") from error
    if not isinstance(saved_pipeline, Pipeline) or "classifier" not in saved_pipeline.named_steps:
        raise ValueError("Saved Model 1 pipeline does not contain its classifier.")
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", clone(saved_pipeline.named_steps["classifier"])),
        ]
    )
    target = pd.to_numeric(historical[target_column], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("Historical Model 1 labels must be complete binary outcomes.")
    pipeline.fit(training, target.astype(int))
    return (
        pipeline,
        training,
        evaluation,
        feature_names,
        numerical_features,
        categorical_features,
    )


def _quality_features(
    batch: pd.DataFrame,
    reference: pd.DataFrame,
    feature_names: list[str],
    numerical_features: list[str],
    categorical_features: list[str],
) -> tuple[float, float, float]:
    expected_cells = len(batch) * len(feature_names)
    missing_count = int(batch.loc[:, feature_names].isna().sum().sum())
    missing_percentage = 100.0 * missing_count / expected_cells if expected_cells else 0.0
    invalid_count = 0
    for feature in numerical_features:
        values = batch[feature]
        numeric = pd.to_numeric(values, errors="coerce")
        invalid_count += int((values.notna() & numeric.isna()).sum())
        numeric = numeric.astype(float)
        invalid_count += int((numeric.notna() & ~np.isfinite(numeric)).sum())
        if feature in NON_NEGATIVE_FEATURES:
            invalid_count += int((numeric.notna() & numeric.lt(0)).sum())
    for feature in categorical_features:
        supported = set(reference[feature].dropna().astype(str))
        invalid_count += int(
            (
                batch[feature].notna()
                & ~batch[feature].astype(str).isin(supported)
            ).sum()
        )
    invalid_percentage = (
        min(100.0, 100.0 * invalid_count / expected_cells)
        if expected_cells
        else 0.0
    )
    duplicate_rate = (
        float(batch.loc[:, feature_names].duplicated().mean()) if len(batch) else 0.0
    )
    return missing_percentage, invalid_percentage, duplicate_rate


def _performance_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1_score": float(f1_score(actual, predicted, zero_division=0)),
    }


def _finite_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _json_number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _build_offline_observations(
    historical: pd.DataFrame,
    future: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    *,
    feature_names: list[str],
    numerical_features: list[str],
    categorical_features: list[str],
    model1_replay_version: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if not model1_replay_version.strip():
        raise ValueError("Offline observations require a Model 1 replay version.")
    if len(future) != len(predictions) or len(future) != len(probabilities):
        raise ValueError("Every offline booking must have one aligned prediction and probability.")
    if future["is_canceled"].isna().any() or not future["is_canceled"].isin([0, 1]).all():
        raise ValueError("Future evaluation outcomes must be complete binary labels.")
    if "reservation_status_date" not in future:
        raise ValueError("Future evaluation data is missing actual outcome dates.")
    outcome_dates = pd.to_datetime(
        future["reservation_status_date"], errors="coerce"
    )
    if outcome_dates.isna().any():
        raise ValueError("Future evaluation data contains invalid outcome dates.")
    reference = historical.loc[:, feature_names]
    eval_frame = future.loc[:, feature_names].copy()
    eval_frame["_outcome_date"] = outcome_dates.to_numpy()
    eval_frame["_actual"] = future["is_canceled"].astype(int).to_numpy()
    eval_frame["_prediction"] = np.asarray(predictions, dtype=int)
    eval_frame["_probability"] = np.asarray(probabilities, dtype=float)
    eval_frame["_week"] = (
        eval_frame["_outcome_date"].dt.to_period("W-SUN").dt.start_time
    )

    batches: list[dict[str, Any]] = []
    for week, batch in eval_frame.groupby("_week", sort=True):
        actual = batch["_actual"].to_numpy(dtype=int)
        predicted = batch["_prediction"].to_numpy(dtype=int)
        feature_batch = batch.loc[:, feature_names]
        drift = compare_offline_features(
            reference,
            feature_batch,
            feature_names,
            numerical_features,
            categorical_features,
        )
        psi_values = [float(item["psi"]) for item in drift.values()]
        ks_values = [
            float(item["ks"]["statistic"])
            for item in drift.values()
            if item.get("ks") and item["ks"].get("available")
        ]
        missing_percentage, invalid_percentage, duplicate_rate = _quality_features(
            feature_batch,
            reference,
            feature_names,
            numerical_features,
            categorical_features,
        )
        classes = {
            str(label): int(count)
            for label, count in pd.Series(actual).value_counts().sort_index().items()
        }
        batch_timestamp = week + pd.Timedelta(days=6, hours=23, minutes=59, seconds=59)
        batches.append(
            {
                "batch_id": f"historical-offline-outcomes-{week:%Y%m%d}",
                "monitoring_timestamp": batch_timestamp.isoformat(),
                "model1_replay_version": model1_replay_version,
                "data_source_type": PROVENANCE,
                "source": "historical_kaggle_offline",
                "sample_count": int(len(batch)),
                "prediction_count": int(len(batch)),
                "actual_outcome_count": int(len(batch)),
                "pending_outcome_count": 0,
                "actual_class_counts": classes,
                "complete_verified_outcomes": True,
                "both_actual_classes": set(classes) == {"0", "1"},
                "data_quality_score": max(
                    0.0, 100.0 - missing_percentage - invalid_percentage
                ),
                "missing_value_percentage": missing_percentage,
                "invalid_value_percentage": invalid_percentage,
                "duplicate_rate": duplicate_rate,
                "dataset_drift_score": float(np.mean(psi_values)),
                "max_feature_drift_score": float(max(psi_values)),
                "number_drifted_features": float(
                    sum(value >= DEFAULT_THRESHOLDS.psi_critical for value in psi_values)
                ),
                "average_ks_statistic": _finite_mean(ks_values),
                "prediction_positive_rate": float(np.mean(predicted)),
                "prediction_mean_probability": float(batch["_probability"].mean()),
                "model1_metrics": (
                    _performance_metrics(actual, predicted)
                    if set(classes) == {"0", "1"}
                    else None
                ),
            }
        )

    batches.sort(key=lambda item: (item["monitoring_timestamp"], item["batch_id"]))
    previous_performance: list[dict[str, Any]] = []
    previous_prediction_rate: float | None = None
    for item in batches:
        item["prediction_positive_rate_change"] = (
            item["prediction_positive_rate"] - previous_prediction_rate
            if previous_prediction_rate is not None
            else np.nan
        )
        item["historical_accuracy"] = np.nan
        item["historical_precision"] = np.nan
        item["historical_recall"] = np.nan
        item["historical_f1_score"] = np.nan
        item["previous_f1_trend"] = np.nan
        item["historical_metrics_available"] = 0.0
        item["historical_performance_observations"] = float(
            len(previous_performance)
        )
        if previous_performance:
            latest = previous_performance[-1]
            item["historical_accuracy"] = latest["accuracy"]
            item["historical_precision"] = latest["precision"]
            item["historical_recall"] = latest["recall"]
            item["historical_f1_score"] = latest["f1_score"]
            item["historical_metrics_available"] = 1.0
            if len(previous_performance) >= 2:
                item["previous_f1_trend"] = (
                    latest["f1_score"] - previous_performance[-2]["f1_score"]
                )
        if item["model1_metrics"] is not None:
            previous_performance.append(item["model1_metrics"])
        previous_prediction_rate = item["prediction_positive_rate"]

    eligible = [
        item
        for item in batches
        if item["complete_verified_outcomes"] and item["both_actual_classes"]
    ]
    for item in eligible:
        now = pd.Timestamp(item["monitoring_timestamp"])
        prior = [
            candidate
            for candidate in eligible
            if candidate["model1_replay_version"] == item["model1_replay_version"]
            and pd.Timestamp(candidate["monitoring_timestamp"]) < now
        ]
        future_limit = now + timedelta(days=FUTURE_PERFORMANCE_WINDOW_DAYS)
        future_items = [
            candidate
            for candidate in eligible
            if candidate["model1_replay_version"] == item["model1_replay_version"]
            and now < pd.Timestamp(candidate["monitoring_timestamp"]) <= future_limit
        ]
        if not prior or not future_items:
            item[DEGRADATION_TARGET] = None
            item["target_baseline_f1"] = None
            item["target_future_f1"] = None
            item["target_f1_drop"] = None
            item["target_future_timestamp"] = None
            continue
        baseline = float(prior[-1]["model1_metrics"]["f1_score"])
        future_item = future_items[0]
        future_f1 = float(future_item["model1_metrics"]["f1_score"])
        drop = baseline - future_f1
        item[DEGRADATION_TARGET] = int(drop >= F1_DEGRADATION_THRESHOLD)
        item["target_baseline_f1"] = baseline
        item["target_future_f1"] = future_f1
        item["target_f1_drop"] = drop
        item["target_future_timestamp"] = future_item["monitoring_timestamp"]

    output_rows: list[dict[str, Any]] = []
    for item in batches:
        row = {
            key: value
            for key, value in item.items()
            if key not in {"model1_metrics", "actual_class_counts"}
        }
        row["actual_class_0_count"] = item["actual_class_counts"].get("0", 0)
        row["actual_class_1_count"] = item["actual_class_counts"].get("1", 0)
        metrics = item["model1_metrics"] or {}
        row.update(
            {
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1_score": metrics.get("f1_score"),
                "performance_eligible": item["model1_metrics"] is not None,
            }
        )
        output_rows.append(row)
    dataset = pd.DataFrame(output_rows)
    if dataset["batch_id"].duplicated().any():
        raise ValueError("Offline monitoring generation produced duplicate batches.")
    timestamps = pd.to_datetime(dataset["monitoring_timestamp"], utc=True)
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValueError("Offline monitoring batches are not uniquely chronological.")
    return dataset, batches


def build_offline_monitoring_dataset(
    *,
    historical_path: Path = HISTORICAL_DATA_PATH,
    future_path: Path = FUTURE_DATA_PATH,
    metadata_path: Path = MODEL1_METADATA_PATH,
    pipeline_path: Path = MODEL1_PIPELINE_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    historical = pd.read_csv(historical_path)
    future = pd.read_csv(future_path)
    required_helpers = {"offline_row_id", "estimated_booking_date", "reservation_status_date"}
    for frame_name, frame in (("historical", historical), ("future", future)):
        missing = required_helpers - set(frame.columns)
        if missing:
            raise ValueError(f"{frame_name} dataset is missing chronology fields: {sorted(missing)}")
        if frame["offline_row_id"].isna().any() or frame["offline_row_id"].duplicated().any():
            raise ValueError(f"{frame_name} dataset contains missing or duplicate offline row IDs.")
    if set(historical["offline_row_id"]).intersection(future["offline_row_id"]):
        raise ValueError("Historical and future offline row IDs overlap.")
    historical_dates = pd.to_datetime(historical["estimated_booking_date"], errors="coerce")
    future_dates = pd.to_datetime(future["estimated_booking_date"], errors="coerce")
    if historical_dates.isna().any() or future_dates.isna().any():
        raise ValueError("Offline evaluation partitions contain invalid estimated booking dates.")
    if historical_dates.max() >= future_dates.min():
        raise ValueError("Historical and future partitions are not strictly chronological.")
    if not historical_dates.is_monotonic_increasing or not future_dates.is_monotonic_increasing:
        raise ValueError("Offline evaluation partitions must be sorted chronologically.")

    (
        pipeline,
        historical_features,
        future_features,
        feature_names,
        numerical_features,
        categorical_features,
    ) = _fit_historical_model1(
        historical,
        future,
        metadata_path=metadata_path,
        pipeline_path=pipeline_path,
    )
    model1_replay_version = _model1_replay_version(
        historical_path, metadata_path, pipeline_path
    )
    predictions = np.asarray(pipeline.predict(future_features), dtype=int)
    classes = list(pipeline.classes_)
    if 1 not in classes:
        raise ValueError("Chronological Model 1 backtest has no cancellation class.")
    probabilities = np.asarray(
        pipeline.predict_proba(future_features), dtype=float
    )[:, classes.index(1)]
    if (
        predictions.shape != (len(future),)
        or probabilities.shape != (len(future),)
        or not np.isfinite(probabilities).all()
    ):
        raise ValueError("Chronological Model 1 backtest returned invalid predictions.")
    observation_dataset, performance_history = _build_offline_observations(
        historical,
        future,
        predictions,
        probabilities,
        feature_names=feature_names,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        model1_replay_version=model1_replay_version,
    )
    target = future["is_canceled"].astype(int).to_numpy()
    model1_metrics = _performance_metrics(target, predictions)
    return observation_dataset, {
        "source": PROVENANCE,
        "model_scope": MODEL_SCOPE,
        "model1_replay_version": model1_replay_version,
        "historical_row_count": int(len(historical)),
        "future_evaluation_row_count": int(len(future)),
        "historical_estimated_booking_date_range": [
            historical_dates.min().date().isoformat(),
            historical_dates.max().date().isoformat(),
        ],
        "future_estimated_booking_date_range": [
            future_dates.min().date().isoformat(),
            future_dates.max().date().isoformat(),
        ],
        "model1_backtest_metrics": model1_metrics,
        "model1_backtest_confusion_matrix": confusion_matrix(
            target, predictions, labels=[0, 1]
        ).tolist(),
        "feature_names": feature_names,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "evaluation_batch_count": int(len(performance_history)),
        "two_class_batch_count": int(
            sum(item["both_actual_classes"] for item in performance_history)
        ),
        "performance_history": performance_history,
        "model1_training_method": (
            "The saved Model 1 pipeline recipe and classifier configuration were "
            "refit on the historical partition only. The existing fitted Model 1 "
            "artifact was not used because its random split included future-period "
            "rows in training."
        ),
    }


def offline_training_eligibility(
    dataset: pd.DataFrame,
    *,
    minimum_observations: int = MIN_LABELED_OBSERVATIONS,
    test_fraction: float = TEST_FRACTION,
) -> dict[str, Any]:
    if minimum_observations < MIN_LABELED_OBSERVATIONS:
        raise ValueError(
            f"Offline minimum cannot be lower than {MIN_LABELED_OBSERVATIONS}."
        )
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    required = {
        "batch_id",
        "monitoring_timestamp",
        "target_future_timestamp",
        "model1_replay_version",
        "data_source_type",
        "complete_verified_outcomes",
        "both_actual_classes",
        "prediction_count",
        "actual_outcome_count",
        "pending_outcome_count",
        "actual_class_0_count",
        "actual_class_1_count",
        "performance_eligible",
        "f1_score",
        "target_baseline_f1",
        "target_future_f1",
        "target_f1_drop",
        DEGRADATION_TARGET,
        *MODEL2_FEATURE_NAMES,
    }
    missing_columns = required - set(dataset.columns)
    missing_requirements = (
        [f"Missing offline eligibility fields: {sorted(missing_columns)}."]
        if missing_columns
        else []
    )
    eligible = dataset.copy()
    if not missing_columns:
        eligible = eligible.loc[
            eligible["data_source_type"].eq(PROVENANCE)
            & eligible["complete_verified_outcomes"].eq(True)
            & eligible["both_actual_classes"].eq(True)
            & eligible[DEGRADATION_TARGET].notna()
        ].copy()
    replay_versions = (
        dataset["model1_replay_version"].astype("string").str.strip()
        if "model1_replay_version" in dataset
        else pd.Series(dtype="string")
    )
    if (
        not missing_columns
        and (
            replay_versions.isna().any()
            or replay_versions.eq("").any()
            or replay_versions.nunique() != 1
        )
    ):
        missing_requirements.append(
            "Every offline observation must use the same identified Model 1 replay version."
        )
    if eligible.empty and not missing_columns:
        missing_requirements.append(
            "No complete two-class historical offline observations have valid targets."
        )
    if not missing_columns:
        if dataset["batch_id"].isna().any() or dataset["batch_id"].duplicated().any():
            missing_requirements.append("Offline target observations require distinct batch IDs.")
        if not eligible[DEGRADATION_TARGET].isin([0, 1]).all():
            missing_requirements.append("Offline degradation targets must be binary 0/1.")
        if not (eligible["prediction_count"] == eligible["actual_outcome_count"]).all():
            missing_requirements.append("Offline observations must have complete aligned outcomes.")
        if not eligible["pending_outcome_count"].eq(0).all():
            missing_requirements.append("Offline observations cannot have pending outcomes.")
        if not eligible["performance_eligible"].eq(True).all():
            missing_requirements.append(
                "Offline observations require measured two-class Model 1 performance."
            )
        class_evidence_valid = True
        for _, row in eligible.iterrows():
            try:
                class_0_count = int(row["actual_class_0_count"])
                class_1_count = int(row["actual_class_1_count"])
                actual_count = int(row["actual_outcome_count"])
            except (TypeError, ValueError, OverflowError):
                class_evidence_valid = False
                break
            if (
                class_0_count <= 0
                or class_1_count <= 0
                or class_0_count + class_1_count != actual_count
            ):
                class_evidence_valid = False
                break
        if not class_evidence_valid:
            missing_requirements.append(
                "Every eligible offline batch must have complete verified outcomes from both classes."
            )
        timestamps = pd.to_datetime(
            eligible["monitoring_timestamp"], utc=True, errors="coerce"
        )
        target_timestamps = pd.to_datetime(
            eligible["target_future_timestamp"], utc=True, errors="coerce"
        )
        if timestamps.isna().any() or target_timestamps.isna().any():
            missing_requirements.append("Offline observations require valid monitoring and target timestamps.")
        elif (
            timestamps.duplicated().any()
            or not timestamps.is_monotonic_increasing
            or (target_timestamps <= timestamps).any()
        ):
            missing_requirements.append(
                "Offline observations and future target timestamps must be uniquely chronological."
            )
        else:
            observed_batches = dataset.loc[
                dataset["data_source_type"].eq(PROVENANCE)
                & dataset["complete_verified_outcomes"].eq(True)
                & dataset["both_actual_classes"].eq(True)
                & dataset["performance_eligible"].eq(True)
            ].copy()
            observed_batches = observed_batches.loc[
                observed_batches["model1_replay_version"].eq(
                    eligible["model1_replay_version"].iloc[0]
                )
            ]
            observed_batches["_parsed_timestamp"] = pd.to_datetime(
                observed_batches["monitoring_timestamp"], utc=True, errors="coerce"
            )
            target_evidence_valid = True
            for _, row in eligible.iterrows():
                now = pd.to_datetime(row["monitoring_timestamp"], utc=True)
                prior = observed_batches.loc[
                    observed_batches["_parsed_timestamp"] < now
                ].sort_values("_parsed_timestamp")
                future = observed_batches.loc[
                    (observed_batches["_parsed_timestamp"] > now)
                    & (
                        observed_batches["_parsed_timestamp"]
                        <= now + pd.Timedelta(days=FUTURE_PERFORMANCE_WINDOW_DAYS)
                    )
                ].sort_values("_parsed_timestamp")
                if prior.empty or future.empty:
                    target_evidence_valid = False
                    break
                baseline = float(prior.iloc[-1]["f1_score"])
                future_row = future.iloc[0]
                future_f1 = float(future_row["f1_score"])
                drop = baseline - future_f1
                expected_label = int(drop >= F1_DEGRADATION_THRESHOLD)
                future_timestamp = pd.to_datetime(
                    row["target_future_timestamp"], utc=True
                )
                if (
                    not np.isclose(float(row["target_baseline_f1"]), baseline)
                    or not np.isclose(float(row["target_future_f1"]), future_f1)
                    or not np.isclose(float(row["target_f1_drop"]), drop)
                    or int(row[DEGRADATION_TARGET]) != expected_label
                    or future_timestamp
                    != pd.to_datetime(future_row["monitoring_timestamp"], utc=True)
                ):
                    target_evidence_valid = False
                    break
            if not target_evidence_valid:
                missing_requirements.append(
                    "Offline target labels do not match the existing F1-drop and 30-day target definition."
                )
    count = (
        int(eligible["batch_id"].nunique(dropna=True))
        if not missing_columns
        else 0
    )
    if count < minimum_observations:
        missing_requirements.append(
            f"Need at least {minimum_observations} distinct eligible historical offline "
            f"observations; found {count}."
        )
    train_rows = eligible.iloc[0:0]
    test_rows = eligible.iloc[0:0]
    if count >= 2 and not missing_columns:
        split_index = int(np.floor(count * (1 - test_fraction)))
        if split_index < 2 or split_index >= count:
            missing_requirements.append(
                "Chronological offline split leaves too few training or testing observations."
            )
        else:
            test_rows = eligible.iloc[split_index:].copy()
            test_start = pd.to_datetime(test_rows["monitoring_timestamp"], utc=True).iloc[0]
            train_rows = eligible.iloc[:split_index].copy()
            train_rows = train_rows.loc[
                pd.to_datetime(train_rows["target_future_timestamp"], utc=True)
                < test_start
            ]
            if len(train_rows) < 2:
                missing_requirements.append(
                    "Purging overlapping future-label windows leaves too few offline training observations."
                )
            elif train_rows[DEGRADATION_TARGET].nunique() < 2:
                missing_requirements.append(
                    "Purged chronological offline training partition does not contain both target classes."
                )
    classes = {
        str(int(label)): int(value)
        for label, value in eligible[DEGRADATION_TARGET]
        .astype(int)
        .value_counts()
        .sort_index()
        .items()
    } if not eligible.empty else {}
    return {
        "status": "READY" if not missing_requirements else "NOT_TRAINED",
        "eligible_observation_count": count,
        "target_class_counts": classes,
        "training_size_after_purge": int(len(train_rows)),
        "testing_size": int(len(test_rows)),
        "training_class_counts": {
            str(int(label)): int(value)
            for label, value in train_rows[DEGRADATION_TARGET]
            .astype(int)
            .value_counts()
            .sort_index()
            .items()
        } if not train_rows.empty else {},
        "testing_class_counts": {
            str(int(label)): int(value)
            for label, value in test_rows[DEGRADATION_TARGET]
            .astype(int)
            .value_counts()
            .sort_index()
            .items()
        } if not test_rows.empty else {},
        "missing_requirements": missing_requirements,
        "reason": "; ".join(missing_requirements) if missing_requirements else None,
    }


def _eligible_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = dataset.loc[
        dataset["data_source_type"].eq(PROVENANCE)
        & dataset["complete_verified_outcomes"].eq(True)
        & dataset["both_actual_classes"].eq(True)
        & dataset[DEGRADATION_TARGET].notna()
    ].sort_values("monitoring_timestamp", kind="stable")
    split_index = int(np.floor(len(eligible) * (1 - TEST_FRACTION)))
    test_rows = eligible.iloc[split_index:].copy()
    test_start = pd.to_datetime(test_rows["monitoring_timestamp"], utc=True).iloc[0]
    train_rows = eligible.iloc[:split_index].copy()
    train_rows = train_rows.loc[
        pd.to_datetime(train_rows["target_future_timestamp"], utc=True) < test_start
    ]
    return train_rows, test_rows


def _evaluate_offline_model(
    model: Pipeline,
    test_rows: pd.DataFrame,
) -> dict[str, Any]:
    probabilities = model.predict_proba(test_rows.loc[:, MODEL2_FEATURE_NAMES])[:, 1]
    predictions = (
        probabilities >= DEGRADATION_CLASSIFICATION_THRESHOLD
    ).astype(int)
    actual = test_rows[DEGRADATION_TARGET].astype(int).to_numpy()
    both_classes = set(np.unique(actual)) == {0, 1}
    return {
        "accuracy": float(accuracy_score(actual, predictions)),
        "precision": float(precision_score(actual, predictions, zero_division=0)),
        "recall": float(recall_score(actual, predictions, zero_division=0)),
        "f1_score": float(f1_score(actual, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(actual, predictions, labels=[0, 1]).tolist(),
        "roc_auc": float(roc_auc_score(actual, probabilities)) if both_classes else None,
        "test_class_counts": {
            str(label): int(count)
            for label, count in pd.Series(actual).value_counts().sort_index().items()
        },
    }


def run_offline_model2_training(
    *,
    output_dir: Path = OFFLINE_MODEL2_DIR,
    historical_path: Path = HISTORICAL_DATA_PATH,
    future_path: Path = FUTURE_DATA_PATH,
    metadata_path: Path = MODEL1_METADATA_PATH,
    pipeline_path: Path = MODEL1_PIPELINE_PATH,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset, source_summary = build_offline_monitoring_dataset(
        historical_path=historical_path,
        future_path=future_path,
        metadata_path=metadata_path,
        pipeline_path=pipeline_path,
    )
    replay_version = source_summary.get("model1_replay_version")
    if not isinstance(replay_version, str) or not replay_version.strip():
        raise ValueError("Offline monitoring data has no Model 1 replay version.")
    dataset_path = output_dir / (
        f"{Path(OFFLINE_DATASET_NAME).stem}_{replay_version}.csv"
    )
    _save_csv_new(dataset_path, dataset)
    eligibility = offline_training_eligibility(dataset)
    if eligibility["status"] != "READY":
        result = {
            "model2_status": "NOT_TRAINED",
            "model_scope": MODEL_SCOPE,
            "reason": eligibility["reason"],
            "eligible_observation_count": eligibility["eligible_observation_count"],
            "target_class_counts": eligibility["target_class_counts"],
            "missing_requirements": eligibility["missing_requirements"],
            "dataset_path": str(dataset_path),
            "source_summary": source_summary,
        }
        _write_training_status(output_dir, result)
        return result

    train_rows, test_rows = _eligible_split(dataset)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(
        train_rows.loc[:, MODEL2_FEATURE_NAMES],
        train_rows[DEGRADATION_TARGET].astype(int),
    )
    evaluation_metrics = _evaluate_offline_model(model, test_rows)
    test_probabilities = model.predict_proba(
        test_rows.loc[:, MODEL2_FEATURE_NAMES]
    )[:, 1]
    test_predictions = (
        test_probabilities >= DEGRADATION_CLASSIFICATION_THRESHOLD
    ).astype(int)
    evaluation_predictions = [
        {
            "batch_id": str(row["batch_id"]),
            "timestamp": str(row["monitoring_timestamp"]),
            "actual_degradation_risk": int(row[DEGRADATION_TARGET]),
            "predicted_degradation_risk": int(predicted),
            "degradation_probability": float(probability),
        }
        for (_, row), predicted, probability in zip(
            test_rows.iterrows(),
            test_predictions,
            test_probabilities,
            strict=True,
        )
    ]
    classifier = model.named_steps["classifier"]
    training_time = datetime.now(timezone.utc).isoformat()
    version = "model2-offline-random-forest-" + datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")
    model_path = output_dir / f"{version}.joblib"
    metadata_path_out = output_dir / f"{version}.json"
    evaluation_path = output_dir / f"evaluation_report_{version}.json"
    if any(path.exists() for path in (model_path, metadata_path_out, evaluation_path)):
        raise FileExistsError(f"Refusing to overwrite offline Model 2 version {version}.")
    history = [
        {
            "batch_id": str(row["batch_id"]),
            "timestamp": str(row["monitoring_timestamp"]),
            "sample_count": int(row["sample_count"]),
            "prediction_count": int(row["prediction_count"]),
            "actual_outcome_count": int(row["actual_outcome_count"]),
            "psi_score": float(row["dataset_drift_score"]),
            "ks_statistic": _json_number(row["average_ks_statistic"]),
            "f1_score": _json_number(row["f1_score"]),
            "accuracy": _json_number(row["accuracy"]),
            "precision": _json_number(row["precision"]),
            "recall": _json_number(row["recall"]),
            "performance_eligible": bool(row["performance_eligible"]),
            "target": None if pd.isna(row[DEGRADATION_TARGET]) else int(row[DEGRADATION_TARGET]),
        }
        for _, row in dataset.sort_values("monitoring_timestamp").iterrows()
    ]
    metadata = {
        "model2_status": "TRAINED",
        "model_scope": MODEL_SCOPE,
        "provenance": PROVENANCE,
        "model1_replay_version": source_summary.get("model1_replay_version"),
        "model_name": "Historical Offline Model 2 Performance Degradation Predictor",
        "model_version": version,
        "algorithm": "RandomForestClassifier",
        "training_timestamp": training_time,
        "training_source": "historical_kaggle_offline",
        "feature_names": MODEL2_FEATURE_NAMES,
        "target_name": DEGRADATION_TARGET,
        "target_definition": {
            "rule": (
                "Label 1 when the latest eligible historical F1 before an offline "
                "outcome-week observation exceeds the first eligible F1 within "
                "the next 30 days by at least 0.05; otherwise label 0."
            ),
            "f1_absolute_drop_threshold": F1_DEGRADATION_THRESHOLD,
            "future_window_days": FUTURE_PERFORMANCE_WINDOW_DAYS,
            "target_source": PROVENANCE,
        },
        "training_configuration": {
            "minimum_labeled_observations": MIN_LABELED_OBSERVATIONS,
            "test_fraction": TEST_FRACTION,
            "split_method": "chronological",
            "overlapping_future_label_windows_purged": True,
            "classification_threshold": DEGRADATION_CLASSIFICATION_THRESHOLD,
            "training_class_counts": eligibility["training_class_counts"],
            "testing_class_counts": eligibility["testing_class_counts"],
        },
        "temporal_split": {
            "training_start": str(train_rows["monitoring_timestamp"].iloc[0]),
            "training_end": str(train_rows["monitoring_timestamp"].iloc[-1]),
            "training_target_end": str(train_rows["target_future_timestamp"].max()),
            "testing_start": str(test_rows["monitoring_timestamp"].iloc[0]),
            "testing_end": str(test_rows["monitoring_timestamp"].iloc[-1]),
        },
        "eligible_observation_count": eligibility["eligible_observation_count"],
        "total_monitoring_observation_count": int(len(dataset)),
        "target_class_counts": eligibility["target_class_counts"],
        "training_size": int(len(train_rows)),
        "testing_size": int(len(test_rows)),
        "evaluation_metrics": evaluation_metrics,
        "confusion_matrix": evaluation_metrics["confusion_matrix"],
        "evaluation_predictions": evaluation_predictions,
        "feature_importance": [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in sorted(
                zip(MODEL2_FEATURE_NAMES, classifier.feature_importances_, strict=True),
                key=lambda item: item[1],
                reverse=True,
            )
        ],
        "source_summary": source_summary,
        "dataset_path": str(dataset_path),
        "evaluation_report_path": str(evaluation_path),
        "observation_history": history,
        "limitations": [
            "This is a historical offline experiment using Kaggle data, not live production evidence.",
            "The saved Model 1 artifact was not used because it was trained on a random split containing future-period rows; its pipeline recipe and classifier configuration were refit on the historical partition.",
            "The derived booking chronology is based on estimated_booking_date, not a recorded booking creation timestamp.",
            "Outcome weeks are grouped by reservation_status_date; this historical date is not a verified production outcome-receipt timestamp.",
            "The number of historical monitoring batches is small; holdout metrics are exploratory and do not establish live performance.",
        ],
    }
    evaluation_report = {
        "model_scope": MODEL_SCOPE,
        "provenance": PROVENANCE,
        "model1_replay_version": source_summary.get("model1_replay_version"),
        "model_version": version,
        "evaluation_metrics": evaluation_metrics,
        "training_size": len(train_rows),
        "testing_size": len(test_rows),
        "temporal_split": metadata["temporal_split"],
        "target_class_counts": eligibility["target_class_counts"],
    }
    staged_model = model_path.with_name(f".{model_path.name}.{uuid4().hex}.tmp")
    try:
        joblib.dump(model, staged_model)
        os.link(staged_model, model_path)
        _write_json_new(evaluation_path, evaluation_report)
        _write_json_new(metadata_path_out, metadata)
    finally:
        if staged_model.exists():
            staged_model.unlink()
    result = {
        "model2_status": "TRAINED",
        "model_scope": MODEL_SCOPE,
        "provenance": PROVENANCE,
        "model1_replay_version": source_summary.get("model1_replay_version"),
        "reason": None,
        "eligible_observation_count": eligibility["eligible_observation_count"],
        "target_class_counts": eligibility["target_class_counts"],
        "training_size": len(train_rows),
        "testing_size": len(test_rows),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path_out),
        "dataset_path": str(dataset_path),
        "evaluation_report_path": str(evaluation_path),
        "evaluation_metrics": evaluation_metrics,
        "model_version": version,
        "training_timestamp": training_time,
    }
    _write_training_status(output_dir, result)
    return result


def _latest_offline_artifact(
    model_dir: Path = OFFLINE_MODEL2_DIR,
) -> tuple[Path, dict[str, Any]] | None:
    for metadata_path in sorted(
        model_dir.glob("model2-offline-random-forest-*.json"), reverse=True
    ):
        metadata = _read_json(metadata_path)
        if metadata.get("model_scope") != MODEL_SCOPE:
            continue
        if (
            metadata.get("provenance") != PROVENANCE
            or metadata.get("training_source") != "historical_kaggle_offline"
        ):
            raise OfflineModel2ArtifactError(
                f"Offline Model 2 metadata {metadata_path.name} has invalid provenance."
            )
        if metadata.get("model2_status") != "TRAINED":
            continue
        if metadata.get("model_version") != metadata_path.stem:
            raise OfflineModel2ArtifactError(
                f"Offline Model 2 metadata {metadata_path.name} has an inconsistent version."
            )
        if metadata.get("feature_names") != MODEL2_FEATURE_NAMES:
            raise OfflineModel2ArtifactError(
                f"Offline Model 2 metadata {metadata_path.name} has an incompatible feature schema."
            )
        replay_version = metadata.get("model1_replay_version")
        if not isinstance(replay_version, str) or not replay_version.strip():
            raise OfflineModel2ArtifactError(
                f"Offline Model 2 metadata {metadata_path.name} has no Model 1 replay version."
            )
        try:
            eligible_count = int(metadata.get("eligible_observation_count", 0))
        except (TypeError, ValueError, OverflowError) as error:
            raise OfflineModel2ArtifactError(
                f"Offline Model 2 metadata {metadata_path.name} has an invalid observation count."
            ) from error
        if eligible_count < MIN_LABELED_OBSERVATIONS:
            raise OfflineModel2ArtifactError(
                f"Offline Model 2 metadata {metadata_path.name} does not meet the minimum observation requirement."
            )
        model_path = metadata_path.with_suffix(".joblib")
        if not model_path.is_file():
            raise OfflineModel2ArtifactError(
                f"Offline Model 2 artifact {model_path.name} is missing."
            )
        return model_path, metadata
    return None


def _dashboard_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in history:
        metrics = item.get("model1_metrics") or {}
        target = item.get(DEGRADATION_TARGET)
        result.append(
            {
                "batch_id": str(item["batch_id"]),
                "timestamp": str(item["monitoring_timestamp"]),
                "sample_count": int(item["sample_count"]),
                "prediction_count": int(item["prediction_count"]),
                "actual_outcome_count": int(item["actual_outcome_count"]),
                "psi_score": float(item["dataset_drift_score"]),
                "ks_statistic": _json_number(item.get("average_ks_statistic")),
                "f1_score": _json_number(metrics.get("f1_score")),
                "accuracy": _json_number(metrics.get("accuracy")),
                "precision": _json_number(metrics.get("precision")),
                "recall": _json_number(metrics.get("recall")),
                "performance_eligible": metrics is not None and bool(metrics),
                "target": None if target is None or pd.isna(target) else int(target),
            }
        )
    return result


def offline_model2_status(
    model_dir: Path = OFFLINE_MODEL2_DIR,
) -> dict[str, Any]:
    artifact = _latest_offline_artifact(model_dir)
    if artifact is None:
        status_files = sorted(model_dir.glob(STATUS_GLOB), reverse=True)
        status_path = status_files[0] if status_files else None
        status = _read_json(status_path) if status_path is not None else {}
        return {
            "status": "NOT_TRAINED",
            "model_scope": MODEL_SCOPE,
            "provenance": PROVENANCE,
            "historical_observation_count": int(
                status.get("source_summary", {}).get("evaluation_batch_count", 0)
            ),
            "eligible_observation_count": int(status.get("eligible_observation_count", 0)),
            "target_class_counts": status.get("target_class_counts", {}),
            "training_size": int(status.get("training_size", 0)),
            "testing_size": int(status.get("testing_size", 0)),
            "evaluation_metrics": status.get("evaluation_metrics"),
            "confusion_matrix": (
                status.get("evaluation_metrics", {}).get("confusion_matrix")
                if isinstance(status.get("evaluation_metrics"), dict)
                else None
            ),
            "evaluation_predictions": status.get("evaluation_predictions", []),
            "observation_history": _dashboard_history(
                status.get("source_summary", {}).get("performance_history", [])
            ),
            "missing_requirements": status.get(
                "missing_requirements",
                [f"Offline Model 2 has not been trained; at least {MIN_LABELED_OBSERVATIONS} valid labeled observations are required."],
            ),
            "reason": status.get(
                "reason",
                "Historical offline Model 2 training has not been run.",
            ),
            "model_version": None,
            "model1_replay_version": status.get("model1_replay_version"),
            "training_timestamp": None,
            "target_definition": None,
        }
    _, metadata = artifact
    return {
        "status": "TRAINED",
        "model_scope": MODEL_SCOPE,
        "provenance": PROVENANCE,
        "historical_observation_count": int(
            metadata.get("total_monitoring_observation_count", 0)
        ),
        "eligible_observation_count": int(
            metadata.get("eligible_observation_count", 0)
        ),
        "target_class_counts": metadata.get("target_class_counts", {}),
        "training_size": int(metadata.get("training_size", 0)),
        "testing_size": int(metadata.get("testing_size", 0)),
        "evaluation_metrics": metadata.get("evaluation_metrics"),
        "confusion_matrix": metadata.get("confusion_matrix"),
        "evaluation_predictions": metadata.get("evaluation_predictions", []),
        "observation_history": metadata.get("observation_history", []),
        "missing_requirements": [],
        "reason": None,
        "model_version": metadata.get("model_version"),
        "model1_replay_version": metadata.get("model1_replay_version"),
        "training_timestamp": metadata.get("training_timestamp"),
        "target_definition": metadata.get("target_definition"),
        "limitations": metadata.get("limitations", []),
    }


def predict_offline_degradation(
    features: Mapping[str, Any],
    *,
    model_dir: Path = OFFLINE_MODEL2_DIR,
) -> dict[str, Any]:
    if not isinstance(features, Mapping):
        raise TypeError("Offline Model 2 input features must be a mapping.")
    unexpected = set(features) - set(MODEL2_FEATURE_NAMES)
    if unexpected:
        raise ValueError(f"Unexpected offline Model 2 features: {sorted(unexpected)}.")
    values: dict[str, float] = {}
    for name in MODEL2_FEATURE_NAMES:
        raw = features.get(name)
        if raw is None:
            values[name] = np.nan
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Offline Model 2 feature {name!r} must be numeric or null.") from error
        if not np.isfinite(value):
            raise ValueError(f"Offline Model 2 feature {name!r} must be finite or null.")
        values[name] = value
    artifact = _latest_offline_artifact(model_dir)
    if artifact is None:
        raise OfflineModel2ArtifactError(
            "Historical offline Model 2 has not been trained and cannot make predictions."
        )
    model_path, metadata = artifact
    try:
        model = joblib.load(model_path)
        row = pd.DataFrame([values], columns=MODEL2_FEATURE_NAMES)
        probabilities = np.asarray(model.predict_proba(row), dtype=float)
        classes = list(model.classes_)
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
        raise OfflineModel2ArtifactError(
            f"Unable to use historical offline Model 2 artifact {model_path.name}: {error}"
        ) from error
    if (
        set(classes) != {0, 1}
        or probabilities.shape != (1, len(classes))
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise OfflineModel2ArtifactError(
            "Offline Model 2 artifact returned invalid binary class probabilities."
        )
    probability = float(probabilities[0, classes.index(1)])
    prediction = int(
        probability
        >= float(
            metadata["training_configuration"]["classification_threshold"]
        )
    )
    return {
        "model2_status": "TRAINED",
        "model_scope": MODEL_SCOPE,
        "provenance": PROVENANCE,
        "model1_replay_version": metadata["model1_replay_version"],
        "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
        "predicted_degradation_risk": prediction,
        "degradation_probability": probability,
        "model_version": metadata["model_version"],
        "explanation": (
            "Historical offline estimate only; not validated on live production observations."
        ),
    }


def main() -> None:
    result = run_offline_model2_training()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
