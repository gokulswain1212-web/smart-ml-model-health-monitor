from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from ml.model2_config import (
    DEGRADATION_CLASSIFICATION_THRESHOLD,
    DEGRADATION_TARGET,
    MIN_LABELED_OBSERVATIONS,
    MODEL2_FEATURE_NAMES,
    TEST_FRACTION,
)
from ml.monitoring_config import DEFAULT_THRESHOLDS, NON_NEGATIVE_FEATURES
from ml.offline_drift_analysis import compare_offline_features

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "raw" / "hotel_bookings.csv"
FEATURE_NAMES_PATH = ROOT / "ml" / "models" / "feature_names.json"
MODEL1_METADATA_PATH = ROOT / "ml" / "models" / "model_metadata.json"
MODEL1_PIPELINE_PATH = ROOT / "ml" / "models" / "hotel_cancellation_pipeline.pkl"
TARGET_DEFINITION_PATH = ROOT / "ml" / "models" / "model2" / "target_definition.json"
OUTPUT_ROOT = ROOT / "data" / "offline_replay" / "runs"

PROVENANCE = "HISTORICAL_OFFLINE"
DATA_SOURCE_TYPE = "historical_offline"
SOURCE = "historical_hotel_booking_dataset"
ENVIRONMENT = "offline"
REFERENCE_WINDOW_COUNT = 4
FORBIDDEN_MODEL1_FEATURES = {
    "is_canceled",
    "reservation_status",
    "reservation_status_date",
}
AVAILABILITY_REVIEW_FEATURES = {
    "assigned_room_type",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "required_car_parking_spaces",
    "total_of_special_requests",
}
MONTH_NUMBERS = {
    month: index
    for index, month in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        start=1,
    )
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read offline replay input {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Offline replay input {path.name} must be a JSON object.")
    return value


def _read_feature_names(path: Path) -> list[str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read Model 1 feature list {path.name}: {error}") from error
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(name, str) or not name for name in value)
        or len(value) != len(set(value))
    ):
        raise ValueError("Model 1 feature_names.json must be a unique, non-empty JSON string list.")
    forbidden = FORBIDDEN_MODEL1_FEATURES.intersection(value)
    if forbidden:
        raise ValueError(f"Model 1 feature list contains forbidden fields: {sorted(forbidden)}.")
    return value


def _load_model1_artifacts(
    *,
    feature_names_path: Path,
    metadata_path: Path,
    pipeline_path: Path,
) -> tuple[Any, dict[str, Any], list[str], list[str], list[str]]:
    feature_names = _read_feature_names(feature_names_path)
    metadata = _read_json(metadata_path)
    if metadata.get("feature_names") != feature_names:
        raise ValueError("Model 1 metadata feature order does not match feature_names.json.")
    if metadata.get("target_column") != "is_canceled":
        raise ValueError("Model 1 metadata must identify is_canceled as its target.")
    numerical = metadata.get("numerical_features")
    categorical = metadata.get("categorical_features")
    if (
        not isinstance(numerical, list)
        or not isinstance(categorical, list)
        or set(numerical) | set(categorical) != set(feature_names)
        or set(numerical) & set(categorical)
    ):
        raise ValueError("Model 1 numerical/categorical metadata does not partition its features.")
    try:
        pipeline = joblib.load(pipeline_path)
    except (OSError, EOFError, ImportError, ValueError) as error:
        raise ValueError(f"Unable to load saved Model 1 pipeline: {error}") from error
    named_steps = getattr(pipeline, "named_steps", None)
    if not isinstance(named_steps, dict) or not {"preprocessor", "classifier"}.issubset(named_steps):
        raise ValueError("Saved Model 1 artifact is missing its preprocessor or classifier.")
    preprocessor = named_steps["preprocessor"]
    classifier = named_steps["classifier"]
    if list(getattr(preprocessor, "feature_names", [])) != feature_names:
        raise ValueError("Saved Model 1 preprocessor order does not match feature_names.json.")
    classifier_features = list(getattr(classifier, "feature_names_in_", []))
    if classifier_features != feature_names:
        raise ValueError("Saved Model 1 classifier order does not match feature_names.json.")
    if set(getattr(classifier, "classes_", [])) != {0, 1}:
        raise ValueError("Saved Model 1 pipeline must support both cancellation classes.")
    return pipeline, metadata, feature_names, numerical, categorical


def _load_target_definition(path: Path) -> tuple[str, int, int, float, int]:
    definition = _read_json(path)
    target_name = definition.get("target")
    stable_value = definition.get("stable_value")
    degraded_value = definition.get("degraded_value")
    threshold = definition.get("f1_absolute_drop_threshold")
    future_days = definition.get("future_window_days")
    if (
        target_name != DEGRADATION_TARGET
        or isinstance(stable_value, bool)
        or not isinstance(stable_value, int)
        or isinstance(degraded_value, bool)
        or not isinstance(degraded_value, int)
        or {stable_value, degraded_value} != {0, 1}
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0 < float(threshold) <= 1
        or isinstance(future_days, bool)
        or not isinstance(future_days, int)
        or future_days < 1
    ):
        raise ValueError("Model 2 target_definition.json has invalid target or temporal thresholds.")
    return target_name, stable_value, degraded_value, float(threshold), future_days


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _arrival_dates(frame: pd.DataFrame) -> pd.Series:
    month_number = frame["arrival_date_month"].map(MONTH_NUMBERS)
    if month_number.isna().any():
        invalid = frame.loc[month_number.isna(), "arrival_date_month"].drop_duplicates().tolist()
        raise ValueError(f"Dataset has unsupported arrival month values: {invalid}.")
    dates = pd.to_datetime(
        {
            "year": pd.to_numeric(frame["arrival_date_year"], errors="coerce"),
            "month": month_number,
            "day": pd.to_numeric(frame["arrival_date_day_of_month"], errors="coerce"),
        },
        errors="coerce",
    )
    if dates.isna().any():
        raise ValueError("Dataset contains invalid arrival dates; chronological replay stopped.")
    return dates


def _performance_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1_score": float(f1_score(actual, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
    }


def _quality_metrics(
    batch: pd.DataFrame,
    reference: pd.DataFrame,
    feature_names: list[str],
    numerical_features: list[str],
    categorical_features: list[str],
) -> tuple[float, float, float]:
    feature_batch = batch.loc[:, feature_names]
    cell_count = len(feature_batch) * len(feature_names)
    missing_count = int(feature_batch.isna().sum().sum())
    invalid_count = 0
    for name in numerical_features:
        original = feature_batch[name]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid_count += int((original.notna() & numeric.isna()).sum())
        numeric = numeric.astype(float)
        invalid_count += int((numeric.notna() & ~np.isfinite(numeric)).sum())
        if name in NON_NEGATIVE_FEATURES:
            invalid_count += int((numeric.notna() & numeric.lt(0)).sum())
    for name in categorical_features:
        supported = set(reference[name].dropna().astype(str))
        invalid_count += int(
            (feature_batch[name].notna() & ~feature_batch[name].astype(str).isin(supported)).sum()
        )
    missing_percentage = 100.0 * missing_count / cell_count if cell_count else 0.0
    invalid_percentage = min(100.0, 100.0 * invalid_count / cell_count) if cell_count else 0.0
    duplicate_rate = float(feature_batch.duplicated().mean()) if len(feature_batch) else 0.0
    return missing_percentage, invalid_percentage, duplicate_rate


def _summarize_duplicates(
    frame: pd.DataFrame,
    feature_names: list[str],
    arrival_weeks: pd.Series,
) -> tuple[dict[str, int], pd.Series]:
    exact_extra = int(frame.duplicated(keep="first").sum())
    exact_group_rows = int(frame.duplicated(keep=False).sum())
    feature_groups = frame.groupby(feature_names, dropna=False)["is_canceled"].agg(
        row_count="size",
        label_count="nunique",
    )
    conflicting = feature_groups.loc[feature_groups["label_count"] > 1]
    signatures = pd.util.hash_pandas_object(
        frame.loc[:, feature_names], index=False
    ).astype("uint64").astype(str)
    window_counts = pd.DataFrame(
        {"signature": signatures.to_numpy(), "arrival_week": arrival_weeks.to_numpy()}
    ).groupby("signature", sort=False)["arrival_week"].nunique()
    if window_counts.gt(1).any():
        raise ValueError("Identical Model 1 feature rows span multiple arrival-date windows.")
    return (
        {
            "exact_duplicate_rows_beyond_first": exact_extra,
            "rows_in_exact_duplicate_groups": exact_group_rows,
            "repeated_model1_feature_rows": int(frame.duplicated(feature_names, keep=False).sum()),
            "model1_feature_groups_with_conflicting_cancellation_labels": int(len(conflicting)),
            "rows_in_conflicting_model1_feature_groups": int(conflicting["row_count"].sum()),
            "feature_groups_crossing_arrival_windows": 0,
        },
        signatures,
    )


def _build_windows(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    *,
    feature_names: list[str],
    numerical_features: list[str],
    categorical_features: list[str],
    target_name: str,
    stable_value: int,
    degraded_value: int,
    f1_drop_threshold: float,
    future_window_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    arrival_dates = _arrival_dates(frame)
    arrival_weeks = arrival_dates.dt.to_period("W-SUN")
    groups: list[dict[str, Any]] = []
    feature_frame = frame.loc[:, feature_names]
    labels = frame["is_canceled"].astype(int).to_numpy()
    for week in sorted(arrival_weeks.unique()):
        mask = arrival_weeks.eq(week).to_numpy()
        row_indexes = np.flatnonzero(mask)
        actual = labels[row_indexes]
        predicted = predictions[row_indexes]
        classes = {
            str(label): int(count)
            for label, count in pd.Series(actual).value_counts().sort_index().items()
        }
        metrics = _performance_metrics(actual, predicted)
        week_start = pd.Timestamp(week.start_time).tz_localize("UTC")
        week_end = week_start + pd.Timedelta(days=7) - pd.Timedelta(nanoseconds=1)
        groups.append(
            {
                "week_start": week_start,
                "week_end": week_end,
                "batch_id": f"historical-offline-arrival-week-{week_start:%Y%m%d}",
                "record_count": int(len(row_indexes)),
                "actual_class_counts": classes,
                "both_actual_classes": set(classes) == {"0", "1"},
                "metrics": metrics,
                "prediction_positive_rate": float(np.mean(predicted)),
                "prediction_mean_probability": float(probabilities[row_indexes].mean()),
                "feature_rows": feature_frame.iloc[row_indexes],
                "row_indexes": row_indexes,
            }
        )

    reference_groups = groups[:REFERENCE_WINDOW_COUNT]
    reference = (
        pd.concat([item["feature_rows"] for item in reference_groups], ignore_index=True)
        if reference_groups
        else frame.loc[:, feature_names].iloc[0:0]
    )
    performance_history: list[dict[str, Any]] = []
    previous_f1: float | None = None
    previous_prediction_rate: float | None = None
    for index, item in enumerate(groups):
        if index < REFERENCE_WINDOW_COUNT:
            continue
        feature_rows = item["feature_rows"]
        drift = compare_offline_features(
            reference,
            feature_rows,
            feature_names,
            numerical_features,
            categorical_features,
        )
        psi_values = [float(value["psi"]) for value in drift.values()]
        ks_values = [
            float(value["ks"]["statistic"])
            for value in drift.values()
            if isinstance(value.get("ks"), dict)
            and value["ks"].get("available")
            and value["ks"].get("statistic") is not None
        ]
        missing_pct, invalid_pct, duplicate_rate = _quality_metrics(
            frame.iloc[item["row_indexes"]],
            reference,
            feature_names,
            numerical_features,
            categorical_features,
        )
        metrics = item["metrics"]
        item["monitoring_features"] = {
            "dataset_drift_score": float(np.mean(psi_values)),
            "max_feature_drift_score": float(max(psi_values, default=0.0)),
            "number_drifted_features": float(
                sum(value >= DEFAULT_THRESHOLDS.psi_critical for value in psi_values)
            ),
            "missing_value_percentage": missing_pct,
            "invalid_value_percentage": invalid_pct,
            "data_quality_score": max(0.0, 100.0 - missing_pct - invalid_pct),
            "duplicate_rate": duplicate_rate,
            "prediction_positive_rate": item["prediction_positive_rate"],
            "prediction_positive_rate_change": (
                item["prediction_positive_rate"] - previous_prediction_rate
                if previous_prediction_rate is not None
                else np.nan
            ),
            "prediction_mean_probability": item["prediction_mean_probability"],
            "historical_accuracy": metrics["accuracy"],
            "historical_precision": metrics["precision"],
            "historical_recall": metrics["recall"],
            "historical_f1_score": metrics["f1_score"],
            "previous_f1_trend": (
                metrics["f1_score"] - previous_f1
                if previous_f1 is not None
                else np.nan
            ),
            "historical_metrics_available": 1.0,
            "historical_performance_observations": float(len(performance_history) + 1),
        }
        item["average_ks_statistic"] = (
            float(np.mean(ks_values)) if ks_values else None
        )
        item["drifted_features"] = int(
            sum(value >= DEFAULT_THRESHOLDS.psi_critical for value in psi_values)
        )
        item["performance_eligible"] = item["both_actual_classes"]
        performance_history.append(item)
        previous_f1 = metrics["f1_score"]
        previous_prediction_rate = item["prediction_positive_rate"]

    target_rows: list[dict[str, Any]] = []
    eligible_groups = [
        item
        for item in performance_history
        if item["performance_eligible"]
    ]
    for item in performance_history:
        if not item["performance_eligible"]:
            continue
        observation_time = item["week_end"]
        future_limit = observation_time + pd.Timedelta(days=future_window_days)
        future_groups = [
            candidate
            for candidate in eligible_groups
            if observation_time < candidate["week_end"] <= future_limit
        ]
        target_value: int | None = None
        baseline_f1: float | None = None
        future_f1: float | None = None
        f1_drop: float | None = None
        future_timestamp: str | None = None
        if future_groups:
            baseline_f1 = float(item["metrics"]["f1_score"])
            future_item = future_groups[0]
            future_f1 = float(future_item["metrics"]["f1_score"])
            f1_drop = baseline_f1 - future_f1
            target_value = (
                degraded_value if f1_drop >= f1_drop_threshold else stable_value
            )
            future_timestamp = future_item["week_end"].isoformat()
        row = {
            "batch_id": item["batch_id"],
            "monitoring_timestamp": item["week_end"].isoformat(),
            "target_future_timestamp": future_timestamp,
            "target_baseline_f1": baseline_f1,
            "target_future_f1": future_f1,
            "target_f1_drop": f1_drop,
            target_name: target_value,
            "data_source_type": DATA_SOURCE_TYPE,
            "source": SOURCE,
            "environment": ENVIRONMENT,
            "is_synthetic": False,
            "complete_actual_outcomes": True,
            "both_actual_classes": True,
            "prediction_count": item["record_count"],
            "actual_outcome_count": item["record_count"],
            "pending_outcome_count": 0,
            "actual_class_0_count": item["actual_class_counts"].get("0", 0),
            "actual_class_1_count": item["actual_class_counts"].get("1", 0),
            "accuracy": item["metrics"]["accuracy"],
            "precision": item["metrics"]["precision"],
            "recall": item["metrics"]["recall"],
            "f1_score": item["metrics"]["f1_score"],
            "average_ks_statistic": item["average_ks_statistic"],
            "number_drifted_features": item["drifted_features"],
            **item["monitoring_features"],
        }
        target_rows.append(row)

    evaluation_rows = [
        {
            "batch_id": item["batch_id"],
            "arrival_window_start": item["week_start"].isoformat(),
            "arrival_window_end": item["week_end"].isoformat(),
            "record_count": item["record_count"],
            "actual_class_0_count": item["actual_class_counts"].get("0", 0),
            "actual_class_1_count": item["actual_class_counts"].get("1", 0),
            "accuracy": item["metrics"]["accuracy"],
            "precision": item["metrics"]["precision"],
            "recall": item["metrics"]["recall"],
            "f1_score": item["metrics"]["f1_score"],
            "average_ks_statistic": item.get("average_ks_statistic"),
            "number_drifted_features": item.get("drifted_features"),
            "data_source_type": DATA_SOURCE_TYPE,
            "source": SOURCE,
            "environment": ENVIRONMENT,
            "is_synthetic": False,
        }
        for item in groups
    ]
    model2_columns = [
        "batch_id",
        "monitoring_timestamp",
        "target_future_timestamp",
        "target_baseline_f1",
        "target_future_f1",
        "target_f1_drop",
        target_name,
        "data_source_type",
        "source",
        "environment",
        "is_synthetic",
        "complete_actual_outcomes",
        "both_actual_classes",
        "prediction_count",
        "actual_outcome_count",
        "pending_outcome_count",
        "actual_class_0_count",
        "actual_class_1_count",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "average_ks_statistic",
        "number_drifted_features",
        *MODEL2_FEATURE_NAMES,
    ]
    return pd.DataFrame(evaluation_rows), pd.DataFrame(
        target_rows, columns=list(dict.fromkeys(model2_columns))
    )


def _offline_model2_eligibility(
    dataset: pd.DataFrame,
    *,
    target_name: str,
    minimum_observations: int = MIN_LABELED_OBSERVATIONS,
    test_fraction: float = TEST_FRACTION,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if minimum_observations < MIN_LABELED_OBSERVATIONS:
        raise ValueError(
            f"Offline minimum cannot be lower than {MIN_LABELED_OBSERVATIONS}."
        )
    required = {
        "batch_id",
        "monitoring_timestamp",
        "target_future_timestamp",
        "data_source_type",
        "source",
        "environment",
        "is_synthetic",
        "complete_actual_outcomes",
        "both_actual_classes",
        target_name,
        *MODEL2_FEATURE_NAMES,
    }
    missing_columns = sorted(required - set(dataset.columns))
    reasons = (
        [f"Missing required offline Model 2 fields: {missing_columns}."]
        if missing_columns
        else []
    )
    eligible = dataset.iloc[0:0].copy()
    if not missing_columns:
        eligible = dataset.loc[
            dataset["data_source_type"].eq(DATA_SOURCE_TYPE)
            & dataset["source"].eq(SOURCE)
            & dataset["environment"].eq(ENVIRONMENT)
            & dataset["is_synthetic"].eq(False)
            & dataset["complete_actual_outcomes"].eq(True)
            & dataset["both_actual_classes"].eq(True)
            & dataset[target_name].notna()
        ].copy()
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if not missing_columns:
        if dataset["batch_id"].isna().any() or dataset["batch_id"].duplicated().any():
            reasons.append("Offline Model 2 observations require distinct batch IDs.")
        if eligible[target_name].empty:
            reasons.append("No measured historical observations have valid future F1 labels.")
        elif not eligible[target_name].isin([0, 1]).all():
            reasons.append("Historical Model 2 target labels must be binary 0/1.")
        if eligible[target_name].nunique() < 2:
            reasons.append("Eligible historical target labels do not contain both classes.")
    if not missing_columns and not eligible.empty:
        timestamps = pd.to_datetime(
            eligible["monitoring_timestamp"], utc=True, errors="coerce"
        )
        future_timestamps = pd.to_datetime(
            eligible["target_future_timestamp"], utc=True, errors="coerce"
        )
        if (
            timestamps.isna().any()
            or future_timestamps.isna().any()
            or timestamps.duplicated().any()
            or not timestamps.is_monotonic_increasing
            or (future_timestamps <= timestamps).any()
        ):
            reasons.append(
                "Eligible Model 2 observations and targets must be uniquely chronological."
            )

    train_rows = eligible.iloc[0:0].copy()
    test_rows = eligible.iloc[0:0].copy()
    eligible_count = (
        int(eligible["batch_id"].nunique(dropna=True))
        if "batch_id" in eligible
        else 0
    )
    if eligible_count < minimum_observations:
        reasons.append(
            f"Need at least {minimum_observations} distinct labeled historical windows; "
            f"found {eligible_count}."
        )
    if eligible_count >= 2 and not reasons:
        split_index = int(np.floor(len(eligible) * (1 - test_fraction)))
        test_rows = eligible.iloc[split_index:].copy()
        if test_rows.empty:
            reasons.append("Chronological holdout has no historical Model 2 test windows.")
        else:
            test_start = pd.to_datetime(
                test_rows["monitoring_timestamp"], utc=True
            ).iloc[0]
            train_rows = eligible.iloc[:split_index].copy()
            train_rows = train_rows.loc[
                pd.to_datetime(train_rows["target_future_timestamp"], utc=True)
                < test_start
            ].copy()
            if train_rows.empty:
                reasons.append("Purging future-label overlap leaves no Model 2 training windows.")
            elif train_rows[target_name].nunique() < 2:
                reasons.append("Purged chronological Model 2 training windows lack both target classes.")
            if not test_rows.empty and test_rows[target_name].nunique() < 2:
                reasons.append("Chronological Model 2 holdout windows lack both target classes.")
            if not train_rows.empty:
                train_end_target = pd.to_datetime(
                    train_rows["target_future_timestamp"], utc=True
                ).max()
                if train_end_target >= test_start:
                    reasons.append("Training target windows overlap the chronological holdout.")
    valid_labels = (
        eligible[target_name].loc[eligible[target_name].isin([0, 1])]
        if target_name in eligible
        else pd.Series(dtype=int)
    )
    counts = (
        {
            str(label): int(count)
            for label, count in valid_labels.value_counts().sort_index().items()
        }
        if not valid_labels.empty
        else {}
    )
    result = {
        "status": "READY" if not reasons else "NOT_ELIGIBLE",
        "eligible_observation_count": eligible_count,
        "target_class_counts": counts,
        "training_size_after_purge": int(len(train_rows)),
        "testing_size": int(len(test_rows)),
        "training_class_counts": (
            {
                str(int(label)): int(count)
                for label, count in train_rows[target_name].astype(int).value_counts().sort_index().items()
            }
            if not train_rows.empty
            else {}
        ),
        "testing_class_counts": (
            {
                str(int(label)): int(count)
                for label, count in test_rows[target_name].astype(int).value_counts().sort_index().items()
            }
            if not test_rows.empty
            else {}
        ),
        "missing_requirements": reasons,
        "reason": "; ".join(reasons) if reasons else None,
    }
    return result, train_rows, test_rows


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, indent=2, allow_nan=False, default=_json_default)
        file.write("\n")


def _fit_and_save_model2(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    *,
    target_name: str,
    run_dir: Path,
    eligibility: dict[str, Any],
    target_definition: dict[str, Any],
) -> dict[str, Any]:
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
        train_rows[target_name].astype(int),
    )
    classes = list(model.classes_)
    probabilities = model.predict_proba(test_rows.loc[:, MODEL2_FEATURE_NAMES])
    if set(classes) != {0, 1} or probabilities.shape != (len(test_rows), len(classes)):
        raise ValueError("Offline Model 2 produced an invalid binary holdout prediction shape.")
    positive = probabilities[:, classes.index(1)]
    predictions = (positive >= DEGRADATION_CLASSIFICATION_THRESHOLD).astype(int)
    actual = test_rows[target_name].astype(int).to_numpy()
    metrics = {
        "accuracy": float(accuracy_score(actual, predictions)),
        "precision": float(precision_score(actual, predictions, zero_division=0)),
        "recall": float(recall_score(actual, predictions, zero_division=0)),
        "f1_score": float(f1_score(actual, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(actual, predictions, labels=[0, 1]).tolist(),
        "classification_threshold": DEGRADATION_CLASSIFICATION_THRESHOLD,
    }
    version = f"model2-historical-offline-{uuid4().hex}"
    model_path = run_dir / f"{version}.joblib"
    metadata_path = run_dir / f"{version}.json"
    evaluation_path = run_dir / f"{version}-evaluation.json"
    joblib.dump(model, model_path)
    metadata = {
        "model_scope": PROVENANCE,
        "provenance": PROVENANCE,
        "data_source_type": DATA_SOURCE_TYPE,
        "source": SOURCE,
        "environment": ENVIRONMENT,
        "is_synthetic": False,
        "model_version": version,
        "target_name": target_name,
        "target_definition": target_definition,
        "feature_names": MODEL2_FEATURE_NAMES,
        "minimum_labeled_observations": MIN_LABELED_OBSERVATIONS,
        "split_method": "chronological with future-target overlap purge",
        "training_size": int(len(train_rows)),
        "testing_size": int(len(test_rows)),
        "training_class_counts": eligibility["training_class_counts"],
        "testing_class_counts": eligibility["testing_class_counts"],
        "training_start": str(train_rows["monitoring_timestamp"].iloc[0]),
        "training_end": str(train_rows["monitoring_timestamp"].iloc[-1]),
        "training_target_end": str(train_rows["target_future_timestamp"].max()),
        "testing_start": str(test_rows["monitoring_timestamp"].iloc[0]),
        "testing_end": str(test_rows["monitoring_timestamp"].iloc[-1]),
        "evaluation_metrics": metrics,
    }
    report = {
        "model_scope": PROVENANCE,
        "provenance": PROVENANCE,
        "model_version": version,
        "evaluation_metrics": metrics,
        "training_size": len(train_rows),
        "testing_size": len(test_rows),
        "training_class_counts": eligibility["training_class_counts"],
        "testing_class_counts": eligibility["testing_class_counts"],
        "temporal_split": {
            key: metadata[key]
            for key in (
                "training_start",
                "training_end",
                "training_target_end",
                "testing_start",
                "testing_end",
            )
        },
        "predictions": [
            {
                "batch_id": str(row["batch_id"]),
                "monitoring_timestamp": str(row["monitoring_timestamp"]),
                "actual_target": int(actual[index]),
                "predicted_target": int(predictions[index]),
                "degradation_probability": float(positive[index]),
                "provenance": PROVENANCE,
            }
            for index, (_, row) in enumerate(test_rows.iterrows())
        ],
    }
    _write_json_new(metadata_path, metadata)
    _write_json_new(evaluation_path, report)
    return {
        "status": "TRAINED",
        "model_scope": PROVENANCE,
        "model_version": version,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "evaluation_path": str(evaluation_path),
        "training_size": int(len(train_rows)),
        "testing_size": int(len(test_rows)),
        "evaluation_metrics": metrics,
        "temporal_split": report["temporal_split"],
    }


def run_historical_replay(
    *,
    dataset_path: Path = DATASET_PATH,
    feature_names_path: Path = FEATURE_NAMES_PATH,
    model1_metadata_path: Path = MODEL1_METADATA_PATH,
    model1_pipeline_path: Path = MODEL1_PIPELINE_PATH,
    target_definition_path: Path = TARGET_DEFINITION_PATH,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    pipeline, model1_metadata, feature_names, numerical, categorical = _load_model1_artifacts(
        feature_names_path=feature_names_path,
        metadata_path=model1_metadata_path,
        pipeline_path=model1_pipeline_path,
    )
    (
        target_name,
        stable_value,
        degraded_value,
        f1_threshold,
        future_days,
    ) = _load_target_definition(target_definition_path)
    try:
        source = pd.read_csv(dataset_path)
    except (OSError, UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise ValueError(f"Unable to read historical booking dataset: {error}") from error
    required = {*feature_names, "is_canceled", "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Historical booking dataset is missing required columns: {missing}.")
    if source.empty:
        raise ValueError("Historical booking dataset contains no rows.")
    labels = pd.to_numeric(source["is_canceled"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Historical is_canceled labels must be complete binary values.")
    source = source.copy()
    source["is_canceled"] = labels.astype(int)
    arrival_dates = _arrival_dates(source)
    arrival_weeks = arrival_dates.dt.to_period("W-SUN")
    duplicate_diagnostics, signatures = _summarize_duplicates(
        source, feature_names, arrival_weeks
    )

    model1_features = source.loc[:, feature_names]
    predictions = np.asarray(pipeline.predict(model1_features), dtype=int)
    class_labels = list(pipeline.classes_)
    if set(class_labels) != {0, 1}:
        raise ValueError("Saved Model 1 pipeline must predict both cancellation classes.")
    probabilities = np.asarray(pipeline.predict_proba(model1_features), dtype=float)
    if (
        predictions.shape != (len(source),)
        or probabilities.shape != (len(source), len(class_labels))
        or not np.isfinite(probabilities).all()
    ):
        raise ValueError("Saved Model 1 pipeline returned invalid historical predictions.")
    positive_probabilities = probabilities[:, class_labels.index(1)]
    model1_metrics = _performance_metrics(
        source["is_canceled"].to_numpy(dtype=int), predictions
    )
    windows, model2_dataset = _build_windows(
        source,
        predictions,
        positive_probabilities,
        feature_names=feature_names,
        numerical_features=numerical,
        categorical_features=categorical,
        target_name=target_name,
        stable_value=stable_value,
        degraded_value=degraded_value,
        f1_drop_threshold=f1_threshold,
        future_window_days=future_days,
    )
    for name in MODEL2_FEATURE_NAMES:
        if name not in model2_dataset:
            raise ValueError(f"Offline Model 2 feature generation omitted {name!r}.")
    if set(MODEL2_FEATURE_NAMES) & {
        target_name,
        "target_future_f1",
        "target_f1_drop",
        "target_future_timestamp",
    }:
        raise ValueError("Future Model 2 target evidence must never be an input feature.")
    if not model2_dataset["is_synthetic"].eq(False).all():
        raise ValueError("Historical replay data must not be tagged as synthetic.")
    if not model2_dataset["data_source_type"].eq(DATA_SOURCE_TYPE).all():
        raise ValueError("Historical replay provenance was not preserved.")

    target_definition = _read_json(target_definition_path)
    eligibility, train_rows, test_rows = _offline_model2_eligibility(
        model2_dataset,
        target_name=target_name,
    )
    if len(windows) <= REFERENCE_WINDOW_COUNT:
        eligibility["missing_requirements"].append(
            f"Need at least {REFERENCE_WINDOW_COUNT + 1} chronological arrival windows "
            "to establish the drift reference and a later evaluation window; "
            f"found {len(windows)}."
        )
        eligibility["reason"] = "; ".join(eligibility["missing_requirements"])
        eligibility["status"] = "NOT_ELIGIBLE"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    model2_training: dict[str, Any]
    if eligibility["status"] == "READY":
        model2_training = _fit_and_save_model2(
            train_rows,
            test_rows,
            target_name=target_name,
            run_dir=run_dir,
            eligibility=eligibility,
            target_definition=target_definition,
        )
    else:
        model2_training = {
            "status": "NOT_ELIGIBLE",
            "model_scope": PROVENANCE,
            "reason": eligibility["reason"],
            "missing_requirements": eligibility["missing_requirements"],
        }

    row_ids = [f"historical-row-{index:06d}" for index in range(len(source))]
    predictions_frame = pd.DataFrame(
        {
            "offline_record_id": row_ids,
            "source_row_number": np.arange(1, len(source) + 1),
            "arrival_date": arrival_dates.dt.date.astype(str),
            "arrival_week": arrival_weeks.astype(str),
            "predicted_is_canceled": predictions,
            "cancellation_probability": positive_probabilities,
            "data_source_type": DATA_SOURCE_TYPE,
            "source": SOURCE,
            "environment": ENVIRONMENT,
            "is_synthetic": False,
        }
    )
    outcomes_frame = pd.DataFrame(
        {
            "offline_record_id": row_ids,
            "source_row_number": np.arange(1, len(source) + 1),
            "is_canceled": source["is_canceled"].to_numpy(dtype=int),
            "data_source_type": DATA_SOURCE_TYPE,
            "source": SOURCE,
            "environment": ENVIRONMENT,
            "is_synthetic": False,
        }
    )
    predictions_frame.to_csv(run_dir / "model1_predictions.csv", index=False)
    outcomes_frame.to_csv(run_dir / "historical_outcomes.csv", index=False)
    windows.to_csv(run_dir / "model1_evaluation_windows.csv", index=False)
    model2_dataset.to_csv(run_dir / "model2_historical_dataset.csv", index=False)
    with (run_dir / "model1_feature_signatures.csv").open(
        "x", encoding="utf-8", newline="\n"
    ) as file:
        pd.DataFrame(
            {
                "offline_record_id": row_ids,
                "feature_signature": signatures.to_numpy(),
                "arrival_week": arrival_weeks.astype(str).to_numpy(),
            }
        ).to_csv(file, index=False)

    trained_rows = int(model1_metadata.get("training_configuration", {}).get(
        "train_test_split", {}
    ).get("train_rows", 0))
    evaluation_rows = int(model1_metadata.get("training_configuration", {}).get(
        "train_test_split", {}
    ).get("test_rows", 0))
    summary = {
        "run_id": run_id,
        "status": "COMPLETE",
        "model_scope": PROVENANCE,
        "provenance": PROVENANCE,
        "data_source_type": DATA_SOURCE_TYPE,
        "source": SOURCE,
        "environment": ENVIRONMENT,
        "is_synthetic": False,
        "prediction_scope": "retrospective_historical_feature_replay_not_booking_time",
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "model1_pipeline_path": str(model1_pipeline_path),
        "model1_pipeline_sha256": _sha256_file(model1_pipeline_path),
        "model1_replay_method": "fixed existing saved Model 1 pipeline; no Model 1 refit",
        "model1_artifact_training_rows": trained_rows,
        "model1_artifact_random_test_rows": evaluation_rows,
        "historical_record_count": int(len(source)),
        "arrival_date_range": [
            arrival_dates.min().date().isoformat(),
            arrival_dates.max().date().isoformat(),
        ],
        "booking_created_timestamp_available": False,
        "chronology_method": "weekly arrival-date cohorts ending Sunday UTC",
        "feature_names": feature_names,
        "model1_forbidden_features_excluded": sorted(FORBIDDEN_MODEL1_FEATURES),
        "feature_availability_review_required": sorted(
            AVAILABILITY_REVIEW_FEATURES.intersection(feature_names)
        ),
        "duplicate_diagnostics": duplicate_diagnostics,
        "model1_metrics": model1_metrics,
        "model1_evaluation_window_count": int(len(windows)),
        "model1_evaluation_windows": "model1_evaluation_windows.csv",
        "model1_predictions": "model1_predictions.csv",
        "historical_outcomes": "historical_outcomes.csv",
        "model2_dataset": "model2_historical_dataset.csv",
        "model2_target_definition": {
            "target": target_name,
            "f1_absolute_drop_threshold": f1_threshold,
            "future_window_days": future_days,
            "baseline": "current measured evaluation-window F1 available at observation time",
            "future_metric": "first later measured evaluation-window F1 within the configured future window",
        },
        "model2_eligibility": eligibility,
        "model2_training": model2_training,
        "limitations": [
            "The dataset has no recorded booking-created timestamp. Arrival-date cohorts are retrospective ordering proxies, not verified booking chronology.",
            "Feature lifecycle availability cannot be established; this is not claimed as a booking-time replay.",
            "The supplied Model 1 artifact was fitted on a random split covering this dataset's full historical period. Its chronological-window metrics are exploratory and not independent out-of-time validation.",
            "Exact repeated rows are retained because there is no booking identifier to prove they are duplicate bookings; their counts and within-window grouping are reported.",
            "Identical Model 1 feature groups with conflicting is_canceled labels are retained as distinct historical records and kept in the same arrival-date window.",
            "Historical offline evidence is not genuine production evidence and cannot satisfy production Model 2 eligibility.",
        ],
        "output_directory": str(run_dir),
    }
    _write_json_new(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an isolated historical hotel booking replay (offline only)."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Historical CSV path (defaults to data/raw/hotel_bookings.csv).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Run directory root (defaults to data/offline_replay/runs).",
    )
    arguments = parser.parse_args()
    result = run_historical_replay(
        dataset_path=arguments.dataset,
        output_root=arguments.output_root,
    )
    print(json.dumps(result, indent=2, allow_nan=False, default=_json_default))


if __name__ == "__main__":
    main()
