from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.data_quality import assess_data_quality

ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES_PATH = ROOT / "ml" / "models" / "feature_names.json"
MODEL_METADATA_PATH = ROOT / "ml" / "models" / "model_metadata.json"
PIPELINE_PATH = ROOT / "ml" / "models" / "hotel_cancellation_pipeline.pkl"
REFERENCE_DATA_PATH = ROOT / "data" / "raw" / "hotel_bookings.csv"
MONITORING_DIR = ROOT / "data" / "monitoring"
PREDICTIONS_DIR = MONITORING_DIR / "predictions"
OUTCOMES_DIR = MONITORING_DIR / "outcomes"
PERFORMANCE_DIR = MONITORING_DIR / "performance"
REPORTS_DIR = MONITORING_DIR / "reports"
MODEL2_PREDICTIONS_DIR = MONITORING_DIR / "model2_predictions"
METADATA_COLUMNS = {
    "batch_id",
    "generated_at",
    "row_id",
    "source_booking_id_hmac",
}
TARGET_COLUMN = "is_canceled"
SEEDED_BATCH = re.compile(
    r"(?:normal|drifted|poor_quality)-.+-seed-\d+",
    flags=re.IGNORECASE,
)


class DuplicateOutcomeError(ValueError):
    pass


class PoorQualityBatchError(ValueError):
    def __init__(self, report: dict[str, Any]) -> None:
        self.quality_report = report
        super().__init__(
            f"Batch {report['batch_id']!r} has CRITICAL data quality and was not predicted: "
            + "; ".join(report["critical_reasons"])
        )


_OUTCOME_WRITE_LOCK = Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_batch_id(batch_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", batch_id):
        raise ValueError(
            "batch_id must start with a letter or digit and contain at most "
            "128 letters, digits, dots, underscores, or hyphens."
        )
    return batch_id


def _batch_id_from_frame(batch: pd.DataFrame) -> str:
    if "batch_id" not in batch or batch.empty:
        raise ValueError("A non-empty batch with a batch_id column is required.")
    batch_ids = batch["batch_id"].dropna().astype(str).unique().tolist()
    if len(batch_ids) != 1 or not batch_ids[0]:
        raise ValueError("Every row in a batch must have the same non-empty batch_id.")
    return _safe_batch_id(batch_ids[0])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Expected JSON objects in {path}.")
    return records


def _write_jsonl_new(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, allow_nan=False) + "\n")


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, allow_nan=False) + "\n")


def _latest_monitoring_report(batch_id: str, reports_dir: Path) -> dict[str, Any] | None:
    matches = sorted(reports_dir.glob("*.json"))
    for path in reversed(matches):
        with path.open(encoding="utf-8") as file:
            report = json.load(file)
        if report.get("batch_id") == batch_id:
            drift = report.get("drift", {})
            significant_features = [
                feature
                for section in ("numerical", "categorical")
                for feature, metrics in drift.get(section, {}).items()
                if metrics.get("usable")
                and metrics.get("psi", metrics.get("drift_score", 0)) >= 0.20
            ]
            return {
                "batch_id": batch_id,
                "report_file": path.name,
                "data_source_type": report.get(
                    "data_source_type",
                    (
                        "synthetic_test"
                        if SEEDED_BATCH.fullmatch(str(batch_id))
                        else "unknown"
                    ),
                ),
                "source": report.get("source"),
                "monitoring_timestamp": report.get("monitoring_timestamp"),
                "overall_status": report.get("overall_status"),
                "quality_status": report.get("quality", {}).get("quality_status"),
                "significant_drift_features": significant_features,
            }
    return None


def log_batch_predictions(
    batch: pd.DataFrame,
    *,
    source: str = "unknown",
    pipeline_path: Path = PIPELINE_PATH,
    feature_names_path: Path = FEATURE_NAMES_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
    reference_path: Path = REFERENCE_DATA_PATH,
    predictions_dir: Path = PREDICTIONS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    if not isinstance(batch, pd.DataFrame):
        raise TypeError("Incoming batch must be a pandas DataFrame.")
    if source not in {"production", "unknown", "synthetic_test"}:
        raise ValueError("source must be production, unknown, or synthetic_test.")
    batch_id = _batch_id_from_frame(batch)
    feature_names = json.loads(feature_names_path.read_text(encoding="utf-8"))
    model_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if model_metadata.get("feature_names") != feature_names:
        raise ValueError("Model metadata feature order does not match feature_names.json.")
    if TARGET_COLUMN in batch.columns:
        raise ValueError(
            f"Incoming prediction batches must not include target column {TARGET_COLUMN!r}."
        )

    missing_features = [name for name in feature_names if name not in batch.columns]
    unexpected_columns = [
        name for name in batch.columns
        if name not in feature_names and name not in METADATA_COLUMNS
    ]
    if missing_features or unexpected_columns:
        raise ValueError(
            f"Invalid feature schema. Missing: {missing_features}; "
            f"unexpected columns: {unexpected_columns}."
        )
    if batch["batch_id"].isna().any():
        raise ValueError("batch_id cannot be missing.")

    supplied_row_ids = "row_id" in batch.columns
    if supplied_row_ids:
        if batch["row_id"].isna().any():
            raise ValueError("row_id cannot be missing.")
        row_ids = batch["row_id"].astype(str).str.strip().tolist()
        if any(not row_id for row_id in row_ids) or len(set(row_ids)) != len(row_ids):
            raise ValueError("row_id values must be non-empty and unique within the batch.")
    elif source == "production":
        row_ids = [str(uuid4()) for _ in range(len(batch))]
    else:
        row_ids = [f"{batch_id}:{index:06d}" for index in range(len(batch))]
    observation_ids: list[str | None] = (
        row_ids.copy()
        if source == "production"
        else [None] * len(batch)
    )
    if source == "production" and predictions_dir.exists():
        existing_ids = {
            record.get("observation_id")
            for path in predictions_dir.glob("*.jsonl")
            for record in _read_jsonl(path)
            if record.get("observation_id")
        }
        if existing_ids.intersection(observation_ids):
            raise ValueError("Production observation IDs must be globally unique.")

    batch_features = batch.loc[:, feature_names].copy()
    reference_data = pd.read_csv(
        reference_path,
        usecols=lambda column: column in set(feature_names),
    ).loc[:, feature_names]
    quality_report = assess_data_quality(
        batch.drop(
            columns=["row_id", "source_booking_id_hmac", "generated_at"],
            errors="ignore",
        ),
        reference_data,
        feature_names,
        model_metadata["numerical_features"],
        model_metadata["categorical_features"],
    )
    if quality_report["quality_status"] == "CRITICAL":
        raise PoorQualityBatchError(quality_report)

    pipeline = joblib.load(pipeline_path)
    if getattr(pipeline, "feature_names_in_", None) is not None:
        if list(pipeline.feature_names_in_) != feature_names:
            raise ValueError("Saved pipeline feature order does not match feature_names.json.")
    predictions = pipeline.predict(batch_features)
    if len(predictions) != len(batch):
        raise RuntimeError("Model returned a different number of predictions than input rows.")

    probabilities: list[float | None] = [None] * len(batch)
    if hasattr(pipeline, "predict_proba"):
        probability_matrix = np.asarray(pipeline.predict_proba(batch_features), dtype=float)
        classes = list(pipeline.classes_)
        if 1 in classes:
            positive_index = classes.index(1)
            probabilities = probability_matrix[:, positive_index].tolist()
        elif probability_matrix.ndim == 2 and probability_matrix.shape[1] == 1:
            probabilities = [None] * len(batch)

    model_version = (
        f"{model_metadata.get('model_name', 'Model 1')}@"
        f"{model_metadata.get('training_date', 'unknown')}"
    )
    artifact_sha256 = hashlib.sha256(pipeline_path.read_bytes()).hexdigest()
    predicted_at = _utc_now()
    source_booking_id_hmacs = (
        batch["source_booking_id_hmac"].astype(str).tolist()
        if "source_booking_id_hmac" in batch
        else [None] * len(batch)
    )
    records = [
        {
            "batch_id": batch_id,
            "row_id": row_id,
            "source": source,
            **(
                {"source_booking_id_hmac": str(booking_id_hmac)}
                if booking_id_hmac is not None
                else {}
            ),
            **(
                {"observation_id": observation_id}
                if observation_id is not None
                else {}
            ),
            "predicted_at": predicted_at,
            "predicted_class": int(predicted_class),
            "probability_positive": (
                float(probability) if probability is not None else None
            ),
            "model_version": model_version,
            "model_artifact_sha256": artifact_sha256,
        }
        for row_id, observation_id, predicted_class, probability, booking_id_hmac in zip(
            row_ids,
            observation_ids,
            predictions,
            probabilities,
            source_booking_id_hmacs,
            strict=True,
        )
    ]

    predictions_path = predictions_dir / f"{_safe_batch_id(batch_id)}.jsonl"
    if predictions_path.exists():
        raise FileExistsError(
            f"Predictions already exist for batch {batch_id!r}: {predictions_path}"
        )
    _write_jsonl_new(predictions_path, records)
    return {
        "batch_id": batch_id,
        "prediction_count": len(records),
        "prediction_file": str(predictions_path),
        "prediction_timestamp": predicted_at,
        "row_ids": row_ids,
        "observation_ids": [
            observation_id for observation_id in observation_ids
            if observation_id is not None
        ],
        "predictions": [
            {
                "row_id": record["row_id"],
                **(
                    {"observation_id": record["observation_id"]}
                    if "observation_id" in record
                    else {}
                ),
                "predicted_class": record["predicted_class"],
                "probability_positive": record["probability_positive"],
            }
            for record in records
        ],
        "model_version": model_version,
        "model_artifact_sha256": artifact_sha256,
        "quality_status": quality_report["quality_status"],
    }


def record_actual_outcomes(
    batch_id: str,
    outcomes: pd.DataFrame,
    *,
    source: str = "real",
    predictions_dir: Path = PREDICTIONS_DIR,
    outcomes_dir: Path = OUTCOMES_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    if source not in {"real", "production"}:
        raise ValueError("Outcome source must be real or production.")
    if not batch_id:
        raise ValueError("batch_id must not be empty.")
    if not isinstance(outcomes, pd.DataFrame):
        raise TypeError("Actual outcomes must be supplied as a pandas DataFrame.")
    required_columns = {"row_id", TARGET_COLUMN}
    missing_columns = required_columns - set(outcomes.columns)
    unexpected_columns = set(outcomes.columns) - required_columns - {"batch_id"}
    if missing_columns or unexpected_columns:
        raise ValueError(
            f"Invalid actual outcome schema. Missing: {sorted(missing_columns)}; "
            f"unexpected: {sorted(unexpected_columns)}."
        )
    if SEEDED_BATCH.fullmatch(batch_id):
        raise ValueError("Actual outcomes cannot be recorded for seeded simulator batches.")
    monitoring_report = _latest_monitoring_report(batch_id, reports_dir)
    if monitoring_report is None or monitoring_report["data_source_type"] != "real":
        raise ValueError(
            "Actual outcomes require a monitoring report explicitly marked as real."
        )
    report_is_production = monitoring_report.get("source") == "production"
    if source == "production" and monitoring_report.get("source") != "production":
        raise ValueError(
            "Production outcomes require a report created by trusted production ingestion."
        )
    if report_is_production and source != "production":
        raise ValueError(
            "Production outcomes must use the token-authenticated production outcome API; "
            "CSV or session-authenticated outcomes cannot label production batches."
        )
    prediction_path = predictions_dir / f"{_safe_batch_id(batch_id)}.jsonl"
    predictions = _read_jsonl(prediction_path)
    if not predictions:
        raise ValueError(f"No logged predictions exist for batch {batch_id!r}.")
    prediction_by_row: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        row_id = prediction.get("row_id")
        if not isinstance(row_id, str) or not row_id or row_id in prediction_by_row:
            raise ValueError("Prediction log contains missing or duplicate row IDs.")
        if prediction.get("batch_id") != batch_id:
            raise ValueError("Prediction log contains records for a different batch ID.")
        prediction_by_row[row_id] = prediction
    prediction_rows = set(prediction_by_row)
    if source == "production" and any(
        prediction.get("source") != "production"
        or not isinstance(prediction.get("observation_id"), str)
        or not prediction.get("observation_id")
        for row_id, prediction in prediction_by_row.items()
    ):
        raise ValueError(
            "Production outcomes require predictions with verified production observation IDs."
        )

    if "batch_id" in outcomes:
        supplied_batch_ids = outcomes["batch_id"].dropna().astype(str).unique().tolist()
        if supplied_batch_ids != [batch_id] or outcomes["batch_id"].isna().any():
            raise ValueError("Actual outcome batch_id values do not match the requested batch.")
    if outcomes["row_id"].isna().any():
        raise ValueError("Actual outcome row_id values cannot be missing.")
    row_ids = outcomes["row_id"].astype(str).str.strip().tolist()
    if any(not row_id for row_id in row_ids):
        raise ValueError("Actual outcome row_id values cannot be empty.")
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("Duplicate row_id values were supplied in the outcomes.")

    unknown_rows = sorted(set(row_ids) - prediction_rows)
    if unknown_rows:
        raise ValueError(f"Outcomes reference rows without predictions: {unknown_rows[:10]}")

    labels = pd.to_numeric(outcomes[TARGET_COLUMN], errors="coerce")
    invalid_labels = labels.isna() | ~labels.isin([0, 1])
    if invalid_labels.any():
        invalid_values = outcomes.loc[invalid_labels, TARGET_COLUMN].astype(str).tolist()
        raise ValueError(f"{TARGET_COLUMN} outcomes must be 0 or 1: {invalid_values[:10]}")
    if not np.equal(labels.to_numpy(dtype=float), labels.to_numpy(dtype=int)).all():
        raise ValueError(f"{TARGET_COLUMN} outcomes must be integer values 0 or 1.")

    outcomes_path = outcomes_dir / f"{_safe_batch_id(batch_id)}.jsonl"
    with _OUTCOME_WRITE_LOCK:
        existing = _read_jsonl(outcomes_path)
        existing_row_ids = {record["row_id"] for record in existing}
        duplicate_rows = sorted(set(row_ids) & existing_row_ids)
        if duplicate_rows:
            raise DuplicateOutcomeError(
                f"Actual outcomes were already recorded for rows: {duplicate_rows[:10]}"
            )

        received_at = _utc_now()
        records = [
            {
                "batch_id": batch_id,
                "row_id": row_id,
                **(
                    {"observation_id": prediction_by_row[row_id]["observation_id"]}
                    if source == "production"
                    else {}
                ),
                TARGET_COLUMN: int(label),
                "data_source_type": "real",
                "source": source,
                "outcome_source": (
                    "production_api" if source == "production" else "csv_import"
                ),
                "outcome_received_at": received_at,
            }
            for row_id, label in zip(row_ids, labels, strict=True)
        ]
        _append_jsonl(outcomes_path, records)
    return {
        "batch_id": batch_id,
        "outcome_count": len(records),
        "outcome_file": str(outcomes_path),
        "outcome_received_at": received_at,
    }


def import_actual_outcomes_csv(
    batch_id: str,
    csv_content: bytes,
    *,
    predictions_dir: Path = PREDICTIONS_DIR,
    outcomes_dir: Path = OUTCOMES_DIR,
    performance_dir: Path = PERFORMANCE_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    try:
        content = csv_content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Outcome CSV must use UTF-8 encoding.") from error

    reader = csv.reader(io.StringIO(content, newline=""), strict=True)
    try:
        raw_headers = next(reader)
    except StopIteration as error:
        raise ValueError("Outcome CSV is empty.") from error
    except csv.Error as error:
        raise ValueError(f"Invalid outcome CSV: {error}") from error

    headers = [header.strip() for header in raw_headers]
    if len(set(headers)) != len(headers):
        raise ValueError("Outcome CSV contains duplicate column names.")
    required_headers = {"row_id", TARGET_COLUMN}
    if not required_headers.issubset(headers) or set(headers) - required_headers - {"batch_id"}:
        raise ValueError(
            "Outcome CSV must contain row_id and is_canceled columns, "
            "with only an optional batch_id column."
        )

    rows = []
    try:
        for line_number, values in enumerate(reader, start=2):
            if not values:
                continue
            if len(values) != len(headers):
                raise ValueError(
                    f"Outcome CSV row {line_number} has {len(values)} values; "
                    f"expected {len(headers)}."
                )
            rows.append(dict(zip(headers, values, strict=True)))
    except csv.Error as error:
        raise ValueError(f"Invalid outcome CSV: {error}") from error
    if not rows:
        raise ValueError("Outcome CSV contains no outcome records.")

    outcomes = pd.DataFrame(rows, columns=headers)
    outcome_result = record_actual_outcomes(
        batch_id,
        outcomes,
        predictions_dir=predictions_dir,
        outcomes_dir=outcomes_dir,
        reports_dir=reports_dir,
    )
    performance_result = evaluate_batch_performance(
        batch_id,
        predictions_dir=predictions_dir,
        outcomes_dir=outcomes_dir,
        performance_dir=performance_dir,
        reports_dir=reports_dir,
    )
    return {
        "outcome": outcome_result,
        "performance": performance_result,
    }


def evaluate_batch_performance(
    batch_id: str,
    *,
    predictions_dir: Path = PREDICTIONS_DIR,
    outcomes_dir: Path = OUTCOMES_DIR,
    performance_dir: Path = PERFORMANCE_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    predictions_path = predictions_dir / f"{_safe_batch_id(batch_id)}.jsonl"
    predictions = _read_jsonl(predictions_path)
    if not predictions:
        raise ValueError(f"No prediction records exist for batch {batch_id!r}.")
    if any(record.get("batch_id") != batch_id for record in predictions):
        raise ValueError("Prediction log contains records for a different batch ID.")
    prediction_frame = pd.DataFrame(predictions)
    if prediction_frame["row_id"].duplicated().any():
        raise ValueError("Prediction log contains duplicate row IDs.")

    outcomes_path = outcomes_dir / f"{_safe_batch_id(batch_id)}.jsonl"
    outcomes = _read_jsonl(outcomes_path)
    if any(record.get("batch_id") != batch_id for record in outcomes):
        raise ValueError("Outcome log contains records for a different batch ID.")
    if outcomes and pd.DataFrame(outcomes)["row_id"].duplicated().any():
        raise ValueError("Outcome log contains duplicate row IDs.")

    if outcomes:
        outcome_frame = pd.DataFrame(outcomes)
        unknown_rows = set(outcome_frame["row_id"]) - set(prediction_frame["row_id"])
        if unknown_rows:
            raise ValueError(f"Outcome log references unknown row IDs: {sorted(unknown_rows)[:10]}")
        evaluated = prediction_frame.merge(
            outcome_frame[["row_id", TARGET_COLUMN]],
            on="row_id",
            how="inner",
            validate="one_to_one",
        )
    else:
        evaluated = prediction_frame.iloc[0:0].copy()
        evaluated[TARGET_COLUMN] = pd.Series(dtype=int)

    n_predictions = len(prediction_frame)
    n_actual = len(evaluated)
    pending = n_predictions - n_actual
    actual_outcome_class_counts: dict[str, int] = {}
    if n_actual:
        actual_outcome_class_counts = {
            str(label): int(count)
            for label, count in
            pd.Series(evaluated[TARGET_COLUMN].astype(int).to_numpy())
            .value_counts()
            .sort_index()
            .items()
        }
    metrics: dict[str, Any] = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "roc_auc": None,
        "confusion_matrix": None,
    }
    y_true = (
        evaluated[TARGET_COLUMN].astype(int).to_numpy()
        if n_actual
        else np.array([], dtype=int)
    )
    verified_two_class = n_actual > 0 and set(np.unique(y_true)) == {0, 1}
    if n_actual:
        y_pred = evaluated["predicted_class"].astype(int).to_numpy()
        metrics.update(
            {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(
                    precision_score(y_true, y_pred, pos_label=1, zero_division=0)
                ),
                "recall": float(
                    recall_score(y_true, y_pred, pos_label=1, zero_division=0)
                ),
                "f1_score": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
                "confusion_matrix": confusion_matrix(
                    y_true, y_pred, labels=[0, 1]
                ).tolist(),
            }
        )
        probabilities = pd.to_numeric(
            evaluated["probability_positive"], errors="coerce"
        )
        if set(np.unique(y_true)) == {0, 1} and probabilities.notna().all():
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, probabilities.to_numpy(dtype=float))
            )

    model_versions = prediction_frame["model_version"].dropna().unique().tolist()
    if len(model_versions) != 1:
        raise ValueError(f"Batch predictions have inconsistent model versions: {model_versions}")
    monitoring_report = _latest_monitoring_report(batch_id, reports_dir)
    report_is_production = (
        monitoring_report is not None
        and monitoring_report.get("data_source_type") == "real"
        and monitoring_report.get("source") == "production"
    )
    predictions_by_row = {
        str(record["row_id"]): record for record in predictions
    }
    if len(predictions_by_row) != len(predictions):
        raise ValueError("Prediction log contains duplicate row IDs.")
    if report_is_production and any(
        prediction.get("source") != "production"
        or not isinstance(prediction.get("observation_id"), str)
        or not prediction.get("observation_id")
        for row_id, prediction in predictions_by_row.items()
    ):
        raise ValueError(
            "Production performance requires predictions with verified production observation IDs."
        )
    contributing_observation_ids: list[str] = []
    outcome_provenance = "pending"
    latest_outcome_received_at = None
    if outcomes:
        if report_is_production:
            received_times = []
            for outcome in outcomes:
                row_id = str(outcome.get("row_id", ""))
                prediction = predictions_by_row.get(row_id)
                observation_id = outcome.get("observation_id")
                if (
                    outcome.get("source") != "production"
                    or outcome.get("data_source_type") != "real"
                    or outcome.get("outcome_source") != "production_api"
                    or prediction is None
                    or observation_id != prediction.get("observation_id")
                ):
                    raise ValueError(
                        "Production performance requires each outcome to be linked to "
                        "a token-authenticated production observation."
                    )
                predicted_at = pd.to_datetime(
                    prediction.get("predicted_at"), utc=True, errors="coerce"
                )
                received_at = pd.to_datetime(
                    outcome.get("outcome_received_at"), utc=True, errors="coerce"
                )
                if (
                    pd.isna(predicted_at)
                    or pd.isna(received_at)
                    or received_at < predicted_at
                ):
                    raise ValueError(
                        "Production outcomes require valid timestamps at or after their predictions."
                    )
                received_times.append(pd.Timestamp(received_at))
                contributing_observation_ids.append(str(observation_id))
            if len(set(contributing_observation_ids)) != len(
                contributing_observation_ids
            ):
                raise ValueError("Production outcome observation IDs must be unique.")
            outcome_provenance = "verified_production_api"
            latest_outcome_received_at = max(received_times).isoformat()
        else:
            if any(outcome.get("source") == "production" for outcome in outcomes):
                raise ValueError(
                    "Production outcomes cannot be evaluated against a non-production report."
                )
            outcome_sources = {
                str(outcome.get("outcome_source", "unknown"))
                for outcome in outcomes
            }
            outcome_provenance = (
                "csv_import"
                if outcome_sources == {"csv_import"}
                else "unverified"
            )
    record = {
        "batch_id": batch_id,
        "evaluation_timestamp": _utc_now(),
        "model_version": model_versions[0],
        "number_of_predictions": n_predictions,
        "number_of_actual_outcomes_received": n_actual,
        "number_of_evaluated_records": n_actual,
        "number_of_pending_outcomes": pending,
        "metrics_available": verified_two_class,
        "data_source_type": (
            monitoring_report["data_source_type"] if monitoring_report else None
        ),
        "source": monitoring_report.get("source") if monitoring_report else None,
        "outcome_provenance": outcome_provenance,
        "contributing_observation_ids": contributing_observation_ids,
        "latest_outcome_received_at": latest_outcome_received_at,
        "actual_outcome_class_counts": actual_outcome_class_counts,
        **metrics,
        "monitoring_report": monitoring_report,
    }
    performance_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = performance_dir / f"{_safe_batch_id(batch_id)}_{timestamp}.json"
    with path.open("x", encoding="utf-8") as file:
        json.dump(record, file, indent=2, allow_nan=False)
        file.write("\n")
    record["performance_record_file"] = str(path)
    return record


def list_production_observations(
    *,
    period_days: int = 3650,
    limit: int | None = 1000,
    predictions_dir: Path = PREDICTIONS_DIR,
    outcomes_dir: Path = OUTCOMES_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)
    production_batches: set[str] = set()
    for path in reports_dir.glob("*.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report.get("data_source_type") == "real"
            and report.get("source") == "production"
        ):
            production_batches.add(str(report.get("batch_id", "")))

    outcomes_by_row: dict[tuple[str, str], dict[str, Any]] = {}
    for path in outcomes_dir.glob("*.jsonl"):
        for outcome in _read_jsonl(path):
            if (
                outcome.get("source") == "production"
                and outcome.get("outcome_source") == "production_api"
            ):
                key = (str(outcome.get("batch_id")), str(outcome.get("row_id")))
                if key in outcomes_by_row:
                    raise ValueError(
                        f"Duplicate production outcome for batch/row {key!r}."
                    )
                outcomes_by_row[key] = outcome

    observations: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in predictions_dir.glob("*.jsonl"):
        for prediction in _read_jsonl(path):
            observation_id = prediction.get("observation_id")
            batch_id = str(prediction.get("batch_id", ""))
            if (
                prediction.get("source") != "production"
                or not isinstance(observation_id, str)
                or batch_id not in production_batches
            ):
                continue
            if observation_id in seen_ids:
                raise ValueError(f"Duplicate production observation ID: {observation_id}")
            seen_ids.add(observation_id)
            predicted_at = datetime.fromisoformat(
                str(prediction["predicted_at"]).replace("Z", "+00:00")
            )
            if predicted_at.tzinfo is None:
                predicted_at = predicted_at.replace(tzinfo=timezone.utc)
            if predicted_at < cutoff:
                continue
            outcome = outcomes_by_row.get((batch_id, str(prediction["row_id"])))
            if (
                outcome is not None
                and outcome.get("observation_id") != observation_id
            ):
                raise ValueError(
                    f"Production outcome does not match observation {observation_id!r}."
                )
            observations.append(
                {
                    "observation_id": observation_id,
                    "batch_id": batch_id,
                    "timestamp": predicted_at.astimezone(timezone.utc).isoformat(),
                    "source": "production",
                    "model_version": prediction.get("model_version"),
                    "predicted_class": prediction.get("predicted_class"),
                    "probability_positive": prediction.get("probability_positive"),
                    "actual_outcome": (
                        outcome.get("is_canceled") if outcome is not None else None
                    ),
                    "outcome_received_at": (
                        outcome.get("outcome_received_at")
                        if outcome is not None
                        else None
                    ),
                }
            )
    observations.sort(key=lambda item: (item["timestamp"], item["observation_id"]), reverse=True)
    return observations if limit is None else observations[:limit]


def record_production_outcome(
    observation_id: str,
    is_canceled: int,
    *,
    predictions_dir: Path = PREDICTIONS_DIR,
    outcomes_dir: Path = OUTCOMES_DIR,
    performance_dir: Path = PERFORMANCE_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    if not isinstance(is_canceled, int) or isinstance(is_canceled, bool) or is_canceled not in (0, 1):
        raise ValueError("is_canceled must be the integer 0 or 1.")
    if not observation_id:
        raise ValueError("observation_id must not be empty.")

    matches = [
        (str(record.get("batch_id")), str(record.get("row_id")))
        for path in predictions_dir.glob("*.jsonl")
        for record in _read_jsonl(path)
        if record.get("observation_id") == observation_id
        and record.get("source") == "production"
    ]
    if not matches:
        raise FileNotFoundError(
            f"No production prediction exists for observation {observation_id!r}."
        )
    if len(matches) != 1:
        raise ValueError(f"Observation ID is not unique: {observation_id!r}.")
    batch_id, row_id = matches[0]
    outcome = record_actual_outcomes(
        batch_id,
        pd.DataFrame([{"row_id": row_id, TARGET_COLUMN: is_canceled}]),
        source="production",
        predictions_dir=predictions_dir,
        outcomes_dir=outcomes_dir,
        reports_dir=reports_dir,
    )
    performance = evaluate_batch_performance(
        batch_id,
        predictions_dir=predictions_dir,
        outcomes_dir=outcomes_dir,
        performance_dir=performance_dir,
        reports_dir=reports_dir,
    )
    return {
        "observation_id": observation_id,
        "batch_id": batch_id,
        "outcome": outcome,
        "performance": performance,
    }


def list_production_performance(
    *,
    period_days: int = 3650,
    limit: int = 1000,
    performance_dir: Path = PERFORMANCE_DIR,
) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)
    records = []
    for path in performance_dir.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        report = record.get("monitoring_report") or {}
        if (
            record.get("source") != "production"
            or report.get("source") != "production"
            or record.get("outcome_provenance") != "verified_production_api"
        ):
            continue
        timestamp = datetime.fromisoformat(
            str(record["evaluation_timestamp"]).replace("Z", "+00:00")
        )
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        if timestamp < cutoff:
            continue
        records.append(
            {
                key: record.get(key)
                for key in (
                    "batch_id",
                    "evaluation_timestamp",
                    "model_version",
                    "number_of_predictions",
                    "number_of_actual_outcomes_received",
                    "number_of_pending_outcomes",
                    "metrics_available",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "roc_auc",
                    "confusion_matrix",
                )
            }
        )
    records.sort(key=lambda item: str(item["evaluation_timestamp"]), reverse=True)
    return records[:limit]


def record_model2_production_prediction(
    prediction: dict[str, Any],
    *,
    predictions_dir: Path = MODEL2_PREDICTIONS_DIR,
) -> dict[str, Any]:
    batch_id = prediction.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("A production Model 2 prediction requires a batch ID.")
    prediction_timestamp = pd.to_datetime(
        prediction.get("prediction_timestamp"), utc=True, errors="coerce"
    )
    probability = prediction.get("degradation_probability")
    if (
        pd.isna(prediction_timestamp)
        or not isinstance(probability, (int, float))
        or isinstance(probability, bool)
        or not np.isfinite(probability)
        or not 0 <= probability <= 1
        or prediction.get("model2_status") != "TRAINED"
        or prediction.get("risk_category") not in {"LOW", "MEDIUM", "HIGH"}
    ):
        raise ValueError("Model 2 returned an invalid production prediction.")
    health_status = {
        "LOW": "HEALTHY",
        "MEDIUM": "WARNING",
        "HIGH": "CRITICAL",
    }[prediction["risk_category"]]
    record = {
        "prediction_id": uuid4().hex,
        "batch_id": _safe_batch_id(batch_id),
        "prediction_timestamp": pd.Timestamp(prediction_timestamp).isoformat(),
        "source": "production",
        "data_source_type": "real",
        "model_version": prediction.get("model_version"),
        "predicted_degradation_risk": prediction.get(
            "predicted_degradation_risk"
        ),
        "degradation_probability": float(probability),
        "risk_category": prediction["risk_category"],
        "health_status": health_status,
    }
    if (
        record["predicted_degradation_risk"] not in (0, 1)
        or not isinstance(record["model_version"], str)
        or not record["model_version"]
    ):
        raise ValueError("Model 2 returned incomplete production prediction metadata.")
    _append_jsonl(
        predictions_dir / f"{record['batch_id']}.jsonl",
        [record],
    )
    return record


def list_production_model2_predictions(
    *,
    period_days: int = 3650,
    limit: int = 20,
    predictions_dir: Path = MODEL2_PREDICTIONS_DIR,
) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)
    records: list[dict[str, Any]] = []
    if not predictions_dir.exists():
        return records
    for path in predictions_dir.glob("*.jsonl"):
        for record in _read_jsonl(path):
            if (
                record.get("source") != "production"
                or record.get("data_source_type") != "real"
            ):
                continue
            timestamp = pd.to_datetime(
                record.get("prediction_timestamp"), utc=True, errors="coerce"
            )
            if pd.isna(timestamp) or timestamp < cutoff:
                continue
            records.append(
                {
                    key: record.get(key)
                    for key in (
                        "prediction_id",
                        "batch_id",
                        "prediction_timestamp",
                        "source",
                        "data_source_type",
                        "model_version",
                        "predicted_degradation_risk",
                        "degradation_probability",
                        "risk_category",
                        "health_status",
                    )
                }
            )
    records.sort(
        key=lambda item: (
            str(item["prediction_timestamp"]),
            str(item["prediction_id"]),
        ),
        reverse=True,
    )
    return records[:limit]


def build_monitoring_history(
    *,
    reports_dir: Path = REPORTS_DIR,
    performance_dir: Path = PERFORMANCE_DIR,
) -> list[dict[str, Any]]:
    reports_by_batch: dict[str, dict[str, Any]] = {}
    for path in sorted(reports_dir.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        batch_id = report.get("batch_id")
        if batch_id:
            reports_by_batch[batch_id] = {
                "batch_id": batch_id,
                "report_file": path.name,
                "monitoring_timestamp": report.get("monitoring_timestamp"),
                "overall_status": report.get("overall_status"),
                "quality_status": report.get("quality", {}).get("quality_status"),
                "drift": report.get("drift", {}),
            }

    performance_by_batch: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(performance_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        batch_id = record.get("batch_id")
        if batch_id:
            performance_by_batch.setdefault(batch_id, []).append(record)

    batch_ids = sorted(set(reports_by_batch) | set(performance_by_batch))
    return [
        {
            "batch_id": batch_id,
            "monitoring_report": reports_by_batch.get(batch_id),
            "performance_records": performance_by_batch.get(batch_id, []),
        }
        for batch_id in batch_ids
    ]


def track_batch_file(batch_path: Path) -> dict[str, Any]:
    batch = pd.read_csv(batch_path)
    prediction_result = log_batch_predictions(batch)
    performance_result = evaluate_batch_performance(prediction_result["batch_id"])
    return {
        "prediction": prediction_result,
        "performance_snapshot": performance_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log valid Model 1 predictions and pending-outcome performance snapshots."
    )
    parser.add_argument("batches", nargs="*", type=Path)
    args = parser.parse_args()
    batch_paths = args.batches or sorted(MONITORING_DIR.glob("*.csv"))
    if not batch_paths:
        parser.error("No monitoring batch CSV files were found.")

    results = []
    for batch_path in batch_paths:
        try:
            results.append({"batch_path": str(batch_path), **track_batch_file(batch_path)})
        except PoorQualityBatchError as error:
            results.append(
                {
                    "batch_path": str(batch_path),
                    "status": "BLOCKED_POOR_QUALITY",
                    "batch_id": error.quality_report["batch_id"],
                    "reasons": error.quality_report["critical_reasons"],
                    "predictions_written": False,
                }
            )
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
