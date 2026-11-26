from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.model2_config import (
    DEGRADATION_TARGET,
    FUTURE_PERFORMANCE_WINDOW_DAYS,
    F1_DEGRADATION_THRESHOLD,
    MODEL2_FEATURE_NAMES,
    PSI_CRITICAL_THRESHOLD,
)

ROOT = Path(__file__).resolve().parents[1]
MONITORING_DIR = ROOT / "data" / "monitoring"
REPORTS_DIR = MONITORING_DIR / "reports"
PREDICTIONS_DIR = MONITORING_DIR / "predictions"
PERFORMANCE_DIR = MONITORING_DIR / "performance"
MODEL2_DATA_DIR = MONITORING_DIR / "model2"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}.")
            records.append(record)
    return records


def _timestamp(value: object, *, field: str) -> pd.Timestamp:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"Invalid or missing {field}: {value!r}.")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return pd.Timestamp(parsed.astimezone(timezone.utc))


def _source_type(report: dict[str, Any]) -> str:
    declared = report.get("data_source_type")
    batch_id = str(report.get("batch_id", ""))
    if re.fullmatch(
        r"(?:normal|drifted|poor_quality)-.+-seed-\d+",
        batch_id,
        flags=re.IGNORECASE,
    ):
        return "synthetic_test"
    if declared in {"real", "synthetic_test"}:
        return str(declared)

    return "unknown"


def _load_reports(reports_dir: Path) -> list[dict[str, Any]]:
    latest_by_batch: dict[str, dict[str, Any]] = {}
    for path in sorted(reports_dir.glob("*.json")):
        report = _read_json(path)
        batch_id = report.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            continue
        report_time = _timestamp(
            report.get("monitoring_timestamp"), field="monitoring_timestamp"
        )
        current = latest_by_batch.get(batch_id)
        if current is None or report_time > current["observation_time"]:
            latest_by_batch[batch_id] = {
                "report": report,
                "report_path": path,
                "observation_time": report_time,
                "data_source_type": _source_type(report),
                "source": (
                    "production"
                    if report.get("data_source_type") == "real"
                    and report.get("source") == "production"
                    and not re.fullmatch(
                        r"(?:normal|drifted|poor_quality)-.+-seed-\d+",
                        batch_id,
                        flags=re.IGNORECASE,
                    )
                    else _source_type(report)
                ),
            }
    return sorted(
        latest_by_batch.values(),
        key=lambda item: (item["observation_time"], item["report"]["batch_id"]),
    )


def _load_predictions(predictions_dir: Path) -> dict[str, list[dict[str, Any]]]:
    by_batch: dict[str, list[dict[str, Any]]] = {}
    production_observation_ids: set[str] = set()
    for path in sorted(predictions_dir.glob("*.jsonl")):
        for record in _read_jsonl(path):
            batch_id = record.get("batch_id")
            predicted_at = record.get("predicted_at")
            if not isinstance(batch_id, str) or not batch_id or not predicted_at:
                continue
            observation_id = record.get("observation_id")
            if record.get("source") == "production":
                if not isinstance(observation_id, str) or not observation_id:
                    raise ValueError(
                        f"Production prediction in {path} has no observation_id."
                    )
                if observation_id in production_observation_ids:
                    raise ValueError(
                        f"Duplicate production observation ID: {observation_id}."
                    )
                production_observation_ids.add(observation_id)
            record["_event_time"] = _timestamp(predicted_at, field="predicted_at")
            by_batch.setdefault(batch_id, []).append(record)
    return by_batch


def _load_performance(
    performance_dir: Path,
    source_by_batch: dict[str, str],
    predictions_by_batch: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in sorted(performance_dir.glob("*.json")):
        record = _read_json(path)
        batch_id = record.get("batch_id")
        if (
            not isinstance(batch_id, str)
            or source_by_batch.get(batch_id) != "production"
            or record.get("data_source_type") != "real"
            or record.get("source") != "production"
            or record.get("metrics_available") is not True
            or record.get("outcome_provenance") != "verified_production_api"
        ):
            continue

        number_of_predictions = record.get("number_of_predictions")
        number_of_actual_outcomes = record.get(
            "number_of_actual_outcomes_received"
        )
        number_of_evaluated_records = record.get("number_of_evaluated_records")
        number_of_pending_outcomes = record.get("number_of_pending_outcomes")
        counts = (
            number_of_predictions,
            number_of_actual_outcomes,
            number_of_evaluated_records,
            number_of_pending_outcomes,
        )
        if any(
            not isinstance(count, int) or isinstance(count, bool)
            for count in counts
        ):
            continue
        if (
            number_of_predictions <= 0
            or number_of_actual_outcomes != number_of_predictions
            or number_of_evaluated_records != number_of_predictions
            or number_of_pending_outcomes != 0
        ):
            continue
        class_counts = record.get("actual_outcome_class_counts")
        if (
            not isinstance(class_counts, dict)
            or set(class_counts) != {"0", "1"}
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in class_counts.values()
            )
            or int(class_counts.get("0", 0)) <= 0
            or int(class_counts.get("1", 0)) <= 0
            or sum(int(value) for value in class_counts.values())
            != number_of_actual_outcomes
        ):
            continue
        predictions = predictions_by_batch.get(batch_id, [])
        observation_ids = record.get("contributing_observation_ids")
        prediction_observation_ids = [
            prediction.get("observation_id")
            for prediction in predictions
            if prediction.get("source") == "production"
        ]
        if (
            len(predictions) != number_of_predictions
            or len(prediction_observation_ids) != number_of_predictions
            or not isinstance(observation_ids, list)
            or len(observation_ids) != number_of_actual_outcomes
            or len(set(observation_ids)) != len(observation_ids)
            or set(observation_ids) != set(prediction_observation_ids)
        ):
            continue

        f1 = pd.to_numeric(pd.Series([record.get("f1_score")]), errors="coerce").iloc[0]
        if pd.isna(f1) or not np.isfinite(f1) or not 0 <= f1 <= 1:
            continue
        evaluation_time = _timestamp(
            record.get("evaluation_timestamp"),
            field="evaluation_timestamp",
        )
        latest_outcome_time = _timestamp(
            record.get("latest_outcome_received_at"),
            field="latest_outcome_received_at",
        )
        model_versions = {
            prediction.get("model_version") for prediction in predictions
        }
        if (
            latest_outcome_time > evaluation_time
            or record.get("model_version") not in model_versions
            or len(model_versions) != 1
        ):
            continue
        events.append(
            {
                **record,
                "batch_id": batch_id,
                "_event_time": evaluation_time,
                "_f1": float(f1),
            }
        )
    return sorted(
        events,
        key=lambda event: (event["_event_time"], event["batch_id"]),
    )


def _prediction_summary(
    batch_id: str,
    observation_time: pd.Timestamp,
    predictions: dict[str, list[dict[str, Any]]],
    *,
    model_version: str | None = None,
) -> dict[str, float] | None:
    eligible = [
        record
        for record in predictions.get(batch_id, [])
        if record["_event_time"] <= observation_time
        and record.get("source") == "production"
        and (
            model_version is None
            or record.get("model_version") == model_version
        )
    ]
    if not eligible:
        return None

    classes = pd.to_numeric(
        pd.Series([record.get("predicted_class") for record in eligible]),
        errors="coerce",
    )
    valid_classes = classes[classes.isin([0, 1])]
    if valid_classes.empty:
        positive_rate = np.nan
    else:
        positive_rate = float(valid_classes.mean())

    probabilities = pd.to_numeric(
        pd.Series([record.get("probability_positive") for record in eligible]),
        errors="coerce",
    )
    probabilities = probabilities[np.isfinite(probabilities)]
    return {
        "positive_rate": positive_rate,
        "mean_probability": (
            float(probabilities.mean()) if not probabilities.empty else np.nan
        ),
    }


def _report_features(
    report: dict[str, Any],
    predictions: list[dict[str, Any]] | None,
    previous_prediction_history: list[tuple[str, list[dict[str, Any]]]],
    performance_events: list[dict[str, Any]],
    observation_time: pd.Timestamp,
) -> dict[str, float]:
    quality = report.get("quality", {})
    drift = report.get("drift", {})
    all_drift = [
        metrics
        for section in ("numerical", "categorical")
        for metrics in drift.get(section, {}).values()
        if metrics.get("usable")
    ]
    drift_scores: list[float] = []
    for metrics in all_drift:
        raw_score = metrics.get("psi", metrics.get("drift_score"))
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if np.isfinite(score):
            drift_scores.append(score)

    feature_schema = quality.get("feature_schema", {})
    expected_features = feature_schema.get("expected_features", [])
    missing_features = set(feature_schema.get("missing_features", []))
    feature_count = len(expected_features)
    row_count = max(0, int(quality.get("total_rows", 0)))
    denominator = row_count * feature_count
    missing_counts = quality.get("missing_value_counts", {})
    missing_total = sum(
        row_count
        if feature in missing_features or feature not in missing_counts
        else int(missing_counts[feature])
        for feature in expected_features
    )
    missing_percentage = (
        100.0 * missing_total / denominator if denominator else np.nan
    )

    invalid_counts = quality.get("invalid_numerical_value_counts", {})
    infinite_counts = quality.get("infinite_value_counts", {})
    negative_counts = quality.get("negative_value_counts", {})
    category_counts = quality.get("unexpected_category_counts", {})
    invalid_total = (
        sum(int(value) for value in invalid_counts.values())
        + sum(int(value) for value in infinite_counts.values())
        + sum(int(value) for value in negative_counts.values())
        + sum(int(value) for value in category_counts.values())
    )
    invalid_percentage = (
        min(100.0, 100.0 * invalid_total / denominator)
        if denominator
        else np.nan
    )
    duplicate_rate = float(quality.get("duplicate_rate", 0.0))

    prediction_summary: dict[str, float] | None = None
    trusted_production_report = (
        report.get("data_source_type") == "real"
        and report.get("source") == "production"
    )
    production_predictions = [
        record
        for record in predictions or []
        if record.get("source") == "production"
        and record["_event_time"] <= observation_time
    ]
    current_versions = {
        record.get("model_version") for record in production_predictions
    }
    current_model_version = (
        next(iter(current_versions))
        if trusted_production_report
        and len(current_versions) == 1
        and None not in current_versions
        else None
    )
    if current_model_version is not None:
        prediction_summary = _prediction_summary(
            str(report.get("batch_id", "")),
            observation_time,
            {str(report.get("batch_id", "")): production_predictions},
            model_version=current_model_version,
        )

    previous_prediction_summaries = [
        _prediction_summary(
            candidate_batch,
            observation_time,
            {candidate_batch: records},
            model_version=current_model_version,
        )
        for candidate_batch, records in previous_prediction_history
        if candidate_batch != report.get("batch_id") and current_model_version
    ]
    previous_prediction_summaries = [
        summary
        for summary in previous_prediction_summaries
        if summary is not None and np.isfinite(summary["positive_rate"])
    ]
    prediction_rate_change = (
        prediction_summary["positive_rate"]
        - previous_prediction_summaries[-1]["positive_rate"]
        if prediction_summary is not None
        and np.isfinite(prediction_summary["positive_rate"])
        and previous_prediction_summaries
        else np.nan
    )

    historical = [
        event
        for event in performance_events
        if event["_event_time"] <= observation_time
    ]
    current_model_version = (
        historical[-1].get("model_version") if historical else None
    )
    if current_model_version is not None:
        historical = [
            event
            for event in historical
            if event.get("model_version") == current_model_version
        ]
    current_performance = historical[-1] if historical else None
    previous_performance = historical[-2] if len(historical) >= 2 else None
    historical_metrics_available = current_performance is not None

    return {
        "dataset_drift_score": (
            float(np.mean(drift_scores)) if drift_scores else np.nan
        ),
        "max_feature_drift_score": max(drift_scores) if drift_scores else np.nan,
        "number_drifted_features": (
            float(sum(score >= PSI_CRITICAL_THRESHOLD for score in drift_scores))
            if drift_scores
            else np.nan
        ),
        "missing_value_percentage": missing_percentage,
        "invalid_value_percentage": invalid_percentage,
        "data_quality_score": (
            max(0.0, 100.0 - missing_percentage - invalid_percentage)
            if np.isfinite(missing_percentage) and np.isfinite(invalid_percentage)
            else np.nan
        ),
        "duplicate_rate": duplicate_rate,
        "prediction_positive_rate": (
            prediction_summary["positive_rate"]
            if prediction_summary is not None
            else np.nan
        ),
        "prediction_positive_rate_change": prediction_rate_change,
        "prediction_mean_probability": (
            prediction_summary["mean_probability"]
            if prediction_summary is not None
            else np.nan
        ),
        "historical_accuracy": (
            float(current_performance["accuracy"])
            if current_performance
            and current_performance.get("accuracy") is not None
            else np.nan
        ),
        "historical_precision": (
            float(current_performance["precision"])
            if current_performance
            and current_performance.get("precision") is not None
            else np.nan
        ),
        "historical_recall": (
            float(current_performance["recall"])
            if current_performance
            and current_performance.get("recall") is not None
            else np.nan
        ),
        "historical_f1_score": (
            float(current_performance["_f1"]) if current_performance else np.nan
        ),
        "previous_f1_trend": (
            float(current_performance["_f1"] - previous_performance["_f1"])
            if current_performance and previous_performance
            else np.nan
        ),
        "historical_metrics_available": float(historical_metrics_available),
        "historical_performance_observations": float(len(historical)),
    }


def _performance_evidence(
    batch_id: str,
    performance_events: list[dict[str, Any]],
) -> dict[str, Any]:
    matches = [
        event for event in performance_events if event["batch_id"] == batch_id
    ]
    if not matches:
        return {
            "batch_prediction_count": 0,
            "batch_actual_outcome_count": 0,
            "batch_pending_outcome_count": 0,
            "batch_actual_class_0_count": 0,
            "batch_actual_class_1_count": 0,
            "batch_outcome_provenance": None,
        }
    event = matches[-1]
    class_counts = event["actual_outcome_class_counts"]
    return {
        "batch_prediction_count": int(event["number_of_predictions"]),
        "batch_actual_outcome_count": int(
            event["number_of_actual_outcomes_received"]
        ),
        "batch_pending_outcome_count": int(event["number_of_pending_outcomes"]),
        "batch_actual_class_0_count": int(class_counts["0"]),
        "batch_actual_class_1_count": int(class_counts["1"]),
        "batch_outcome_provenance": event["outcome_provenance"],
    }


def build_historical_dataset(
    *,
    reports_dir: Path = REPORTS_DIR,
    predictions_dir: Path = PREDICTIONS_DIR,
    performance_dir: Path = PERFORMANCE_DIR,
    f1_degradation_threshold: float = F1_DEGRADATION_THRESHOLD,
    future_window_days: int = FUTURE_PERFORMANCE_WINDOW_DAYS,
) -> pd.DataFrame:
    if not 0 < f1_degradation_threshold < 1:
        raise ValueError("f1_degradation_threshold must be between 0 and 1.")
    if future_window_days <= 0:
        raise ValueError("future_window_days must be positive.")

    reports = _load_reports(reports_dir)
    if not reports:
        return pd.DataFrame(
            columns=[
                "batch_id",
                "monitoring_timestamp",
                "monitoring_report_timestamp",
                "data_source_type",
                "source",
                "label_provenance",
                *MODEL2_FEATURE_NAMES,
                DEGRADATION_TARGET,
                "target_future_timestamp",
            ]
        )
    predictions = _load_predictions(predictions_dir)
    for item in reports:
        batch_id = item["report"]["batch_id"]
        batch_predictions = predictions.get(batch_id, [])
        if item["source"] == "production" and (
            not batch_predictions
            or any(record.get("source") != "production" for record in batch_predictions)
        ):
            item["source"] = item["data_source_type"]
    source_by_batch = {item["report"]["batch_id"]: item["source"] for item in reports}
    performance_events = _load_performance(
        performance_dir,
        source_by_batch,
        predictions,
    )

    observations: list[dict[str, Any]] = []
    for item in reports:
        batch_id = item["report"]["batch_id"]
        current_predictions = (
            predictions.get(batch_id, [])
            if item["source"] == "production"
            else []
        )
        prediction_times = [
            record["_event_time"] for record in current_predictions
        ]
        decision_time = max([item["observation_time"], *prediction_times])
        observations.append(
            {
                **item,
                "decision_time": decision_time,
                "predictions": current_predictions,
            }
        )
    observations.sort(
        key=lambda item: (item["decision_time"], item["report"]["batch_id"])
    )

    rows: list[dict[str, Any]] = []
    earlier_prediction_history: list[tuple[str, list[dict[str, Any]]]] = []
    for item in observations:
        report = item["report"]
        batch_id = report["batch_id"]
        observation_time = item["decision_time"]
        current_predictions = item["predictions"]
        features = _report_features(
            report,
            current_predictions,
            earlier_prediction_history,
            performance_events,
            observation_time,
        )
        source_type = item["data_source_type"]

        target: int | None = None
        baseline_f1: float | None = None
        future_f1: float | None = None
        f1_drop: float | None = None
        future_timestamp: str | None = None
        source = item["source"]
        batch_evidence = _performance_evidence(batch_id, performance_events)
        if source == "production":
            prior = [
                event
                for event in performance_events
                if event["_event_time"] <= observation_time
            ]
            baseline_version = prior[-1].get("model_version") if prior else None
            future_limit = observation_time + timedelta(days=future_window_days)
            future = [
                event
                for event in performance_events
                if observation_time < event["_event_time"] <= future_limit
                and baseline_version is not None
                and event.get("model_version") == baseline_version
            ]
            if prior and future:
                baseline_f1 = float(prior[-1]["_f1"])
                future_f1 = float(future[0]["_f1"])
                f1_drop = baseline_f1 - future_f1
                future_timestamp = future[0]["_event_time"].isoformat()
                target = int(f1_drop >= f1_degradation_threshold)

        rows.append(
            {
                "batch_id": batch_id,
                "monitoring_timestamp": observation_time.isoformat(),
                "monitoring_report_timestamp": item["observation_time"].isoformat(),
                "data_source_type": source_type,
                "source": source,
                "source_report": item["report_path"].name,
                "label_provenance": (
                    "verified_production_api"
                    if (
                        source == "production"
                        and target is not None
                        and batch_evidence["batch_outcome_provenance"]
                        == "verified_production_api"
                    )
                    else None
                ),
                **batch_evidence,
                **features,
                DEGRADATION_TARGET: target,
                "target_baseline_f1": baseline_f1,
                "target_future_f1": future_f1,
                "target_f1_drop": f1_drop,
                "target_future_timestamp": future_timestamp,
            }
        )
        if item["source"] == "production" and current_predictions:
            earlier_prediction_history.append((batch_id, current_predictions))

    dataset = pd.DataFrame(rows)
    dataset[DEGRADATION_TARGET] = pd.array(
        dataset[DEGRADATION_TARGET], dtype="Int64"
    )
    return dataset


def save_historical_dataset(
    dataset: pd.DataFrame,
    *,
    output_dir: Path = MODEL2_DATA_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = output_dir / f"historical_monitoring_dataset_{timestamp}.csv"
    dataset.to_csv(path, index=False)
    return path


def dataset_summary(dataset: pd.DataFrame) -> dict[str, Any]:
    target = dataset.get(DEGRADATION_TARGET, pd.Series(dtype="Int64"))
    production_observations = (
        dataset["source"].eq("production")
        & dataset["data_source_type"].eq("real")
        if {"source", "data_source_type"}.issubset(dataset.columns)
        else pd.Series(False, index=dataset.index)
    )
    verified_labels = (
        dataset["label_provenance"].eq("verified_production_api")
        if "label_provenance" in dataset
        else pd.Series(False, index=dataset.index)
    )
    labeled = target.notna() & production_observations & verified_labels
    counts = (
        dataset.loc[labeled, DEGRADATION_TARGET]
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
        if labeled.any()
        else {}
    )
    source_counts = (
        dataset["source"].value_counts().to_dict()
        if "source" in dataset
        else {}
    )
    return {
        "observation_count": int(len(dataset)),
        "production_observation_count": int(production_observations.sum()),
        "genuine_labeled_observation_count": int(labeled.sum()),
        "unlabeled_observation_count": int((~labeled).sum()),
        "target_class_counts": {str(key): int(value) for key, value in counts.items()},
        "data_source_counts": {
            str(key): int(value) for key, value in source_counts.items()
        },
        "feature_names": MODEL2_FEATURE_NAMES.copy(),
        "target_name": DEGRADATION_TARGET,
    }
