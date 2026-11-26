from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ml.alert_engine import (
    ALERTS_PATH,
    explain_monitoring_report,
    load_alert_history,
)
from ml.monitoring_config import DEFAULT_THRESHOLDS, MonitoringThresholds
from ml.performance_tracking import (
    MODEL_METADATA_PATH,
    PERFORMANCE_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    list_production_model2_predictions,
    list_production_observations,
)
from ml.model2_config import MIN_LABELED_OBSERVATIONS, MODEL2_FEATURE_NAMES
from ml.model2_dataset import build_historical_dataset, dataset_summary
from ml.model2_development_explainability import (
    DEVELOPMENT_LABEL,
    DevelopmentExplainabilityError,
)
from ml.model2_development_prediction import (
    DevelopmentModelError,
    _read_development_artifact,
)
from ml.model2_prediction import Model2ArtifactError, _latest_trained_artifact
from ml.model2_training import training_eligibility
from ml.model2_training import MODEL2_DIR as PRODUCTION_MODEL2_DIR
from ml.model2_development_training import DEVELOPMENT_DIR
from ml.offline_model2 import offline_model2_status

ROOT = Path(__file__).resolve().parents[1]
MODEL_PIPELINE_PATH = ROOT / "ml" / "models" / "hotel_cancellation_pipeline.pkl"
MODEL_EVALUATION_PATH = ROOT / "ml" / "models" / "model1_evaluation_visualizations.json"
MODEL2_DIR = PRODUCTION_MODEL2_DIR
MODEL2_DEVELOPMENT_DIR = DEVELOPMENT_DIR
DEVELOPMENT_SIMULATION_DIR = ROOT / "data" / "development_simulation" / "runs"
HISTORICAL_REPLAY_DIR = ROOT / "data" / "offline_replay" / "runs"
MODEL2_DEVELOPMENT_IMPORTANCE_PATH = MODEL2_DIR / "development" / "feature_importance.json"
MODEL2_DEVELOPMENT_PROVENANCE = "SYNTHETIC_DEVELOPMENT_ONLY"
HISTORICAL_REPLAY_PROVENANCE = "HISTORICAL_OFFLINE"
HISTORICAL_REPLAY_SOURCE = "historical_hotel_booking_dataset"
SYNTHETIC_SIMULATION_PROVENANCE = {
    "data_source_type": "synthetic",
    "source": "synthetic_simulator",
    "environment": "development",
    "is_synthetic": True,
}
SEEDED_BATCH = re.compile(
    r"(?:normal|drifted|poor_quality)-.+-seed-\d+",
    flags=re.IGNORECASE,
)
DATASET_SIMULATION_SOURCE = "original_hotel_booking_dataset_simulation"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _report_source(report: dict[str, Any]) -> str:
    source = report.get("data_source_type")
    if source in {"real", "synthetic_test", "unknown"}:
        return source
    batch_id = str(report.get("batch_id", ""))
    return "synthetic_test" if SEEDED_BATCH.fullmatch(batch_id) else "unknown"


def _is_dataset_simulation(report: dict[str, Any]) -> bool:
    return (
        _report_source(report) == "synthetic_test"
        and report.get("batch_metadata", {}).get("source")
        == DATASET_SIMULATION_SOURCE
    )


def _load_reports(period_days: int) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)
    reports = []
    if not REPORTS_DIR.exists():
        return reports
    for path in REPORTS_DIR.glob("*.json"):
        report = _read_json(path)
        report_time = _timestamp(report.get("monitoring_timestamp"))
        if report_time is None or report_time < cutoff:
            continue
        report_source = _report_source(report)
        if report_source != "real":
            continue
        report["_dashboard_timestamp"] = report_time
        report["_dashboard_source"] = report_source
        reports.append(report)
    return sorted(
        reports,
        key=lambda item: item["_dashboard_timestamp"],
        reverse=True,
    )


def _load_simulation_reports(period_days: int) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)
    reports = []
    if not REPORTS_DIR.exists():
        return reports
    for path in REPORTS_DIR.glob("*.json"):
        report = _read_json(path)
        report_time = _timestamp(report.get("monitoring_timestamp"))
        if (
            report_time is None
            or report_time < cutoff
            or not _is_dataset_simulation(report)
        ):
            continue
        report["_dashboard_timestamp"] = report_time
        report["_dashboard_source"] = "synthetic_test"
        reports.append(report)
    return sorted(
        reports,
        key=lambda item: item["_dashboard_timestamp"],
        reverse=True,
    )


def _load_all_reports(period_days: int) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)
    reports = []
    if not REPORTS_DIR.exists():
        return reports
    for path in REPORTS_DIR.glob("*.json"):
        report = _read_json(path)
        report_time = _timestamp(report.get("monitoring_timestamp"))
        if report_time is None or report_time < cutoff:
            continue
        report_source = _report_source(report)
        if report_source == "synthetic_test" and not _is_dataset_simulation(report):
            continue
        report["_dashboard_timestamp"] = report_time
        report["_dashboard_source"] = report_source
        reports.append(report)
    return sorted(
        reports,
        key=lambda item: item["_dashboard_timestamp"],
        reverse=True,
    )


def _drift_insights_by_source(
    reports: list[dict[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    latest_by_source: dict[str, dict[str, Any]] = {}
    for report in reports:
        latest_by_source.setdefault(report["_dashboard_source"], report)
    result: dict[str, dict[str, Any] | None] = {
        "real": None,
        "synthetic_test": None,
        "unknown": None,
    }
    for source, report in latest_by_source.items():
        threshold_values = report.get("thresholds", {})
        thresholds = MonitoringThresholds(
            **{
                name: threshold_values[name]
                for name in DEFAULT_THRESHOLDS.__dataclass_fields__
                if name in threshold_values
            }
        )
        expected_features = report.get("quality", {}).get(
            "feature_schema", {}
        ).get("expected_features", [])
        feature_count = len(
            expected_features
            or set(report.get("drift", {}).get("numerical", {}))
            | set(report.get("drift", {}).get("categorical", {}))
        )
        result[source] = {
            **explain_monitoring_report(report, thresholds=thresholds),
            "quality_summary": _quality_summary(report, feature_count),
        }
    return result


def _monitoring_source_counts(period_days: int) -> dict[str, int]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)
    counts = {"real": 0, "unknown": 0}
    if not REPORTS_DIR.exists():
        return counts
    for path in REPORTS_DIR.glob("*.json"):
        report = _read_json(path)
        report_time = _timestamp(report.get("monitoring_timestamp"))
        if report_time is None or report_time < cutoff:
            continue
        source = _report_source(report)
        if source != "synthetic_test":
            counts[source] += 1
    return counts


def _load_real_performance(
    period_days: int,
    real_batch_ids: set[str],
) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)
    latest_by_batch: dict[str, dict[str, Any]] = {}
    if not PERFORMANCE_DIR.exists():
        return []
    for path in PERFORMANCE_DIR.glob("*.json"):
        record = _read_json(path)
        batch_id = str(record.get("batch_id", ""))
        timestamp = _timestamp(record.get("evaluation_timestamp"))
        monitoring_report = record.get("monitoring_report") or {}
        if (
            batch_id not in real_batch_ids
            or timestamp is None
            or timestamp < cutoff
            or record.get("source") != "production"
            or monitoring_report.get("source") != "production"
            or record.get("outcome_provenance") != "verified_production_api"
            or record.get("data_source_type") != "real"
            or monitoring_report.get("data_source_type") != "real"
            or not record.get("number_of_evaluated_records")
        ):
            continue
        record["_dashboard_timestamp"] = timestamp
        existing = latest_by_batch.get(batch_id)
        if existing is None or timestamp > existing["_dashboard_timestamp"]:
            latest_by_batch[batch_id] = record
    return sorted(
        latest_by_batch.values(),
        key=lambda item: item["_dashboard_timestamp"],
    )


def _load_prediction_summary(
    real_batch_ids: set[str],
) -> dict[str, Any]:
    total = 0
    classes = {"not_cancelled": 0, "cancelled": 0}
    batches: dict[str, int] = {}
    predictions_today = 0
    today = datetime.now(timezone.utc).date()
    if not PREDICTIONS_DIR.exists():
        return {
            "total": total,
            "class_counts": classes,
            "batches": batches,
            "today": predictions_today,
        }
    for path in PREDICTIONS_DIR.glob("*.jsonl"):
        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                batch_id = str(record["batch_id"])
                if batch_id not in real_batch_ids:
                    continue
                total += 1
                batches[batch_id] = batches.get(batch_id, 0) + 1
                predicted_class = record.get("predicted_class")
                if predicted_class in (1, True):
                    classes["cancelled"] += 1
                elif predicted_class in (0, False):
                    classes["not_cancelled"] += 1
                predicted_at = _timestamp(record.get("predicted_at"))
                if predicted_at is not None and predicted_at.date() == today:
                    predictions_today += 1
    return {
        "total": total,
        "class_counts": classes,
        "batches": batches,
        "today": predictions_today,
    }


def _load_prediction_history(
    batch_sources: dict[str, str],
) -> list[dict[str, Any]]:
    history = []
    if not PREDICTIONS_DIR.exists():
        return history
    for path in PREDICTIONS_DIR.glob("*.jsonl"):
        with path.open(encoding="utf-8") as file:
            for index, line in enumerate(file):
                if not line.strip():
                    continue
                record = json.loads(line)
                batch_id = str(record.get("batch_id", ""))
                source = batch_sources.get(batch_id)
                timestamp = _timestamp(record.get("predicted_at"))
                if source is None or timestamp is None:
                    continue
                history.append(
                    {
                        "batch_id": batch_id,
                        "row_id": str(record.get("row_id", f"row-{index + 1}")),
                        "observation_id": record.get("observation_id"),
                        "timestamp": timestamp.isoformat(),
                        "source": source,
                        "model_version": record.get("model_version"),
                        "predicted_class": record.get("predicted_class"),
                        "probability_positive": record.get(
                            "probability_positive"
                        ),
                    }
                )
    return sorted(
        history,
        key=lambda item: (item["timestamp"], item["batch_id"], item["row_id"]),
        reverse=True,
    )[:20]


def _drift_features(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if report is None:
        return []
    features = []
    drift = report.get("drift", {})
    for group in ("numerical", "categorical"):
        for name, result in drift.get(group, {}).items():
            if not result.get("usable"):
                continue
            score = result.get("psi", result.get("drift_score"))
            if isinstance(score, (int, float)):
                features.append({"name": name, "score": float(score)})
    return sorted(features, key=lambda item: item["score"], reverse=True)


def _quality_summary(
    report: dict[str, Any] | None, feature_count: int
) -> dict[str, float | int | None]:
    if report is None:
        return {
            "missing_percentage": None,
            "duplicate_percentage": None,
            "invalid_percentage": None,
            "total_rows": 0,
        }
    quality = report.get("quality", {})
    row_count = int(quality.get("total_rows", 0))
    denominator = row_count * feature_count
    missing_counts = quality.get("missing_value_counts", {})
    missing_total = sum(
        int(missing_counts.get(name, 0))
        for name in quality.get("feature_schema", {}).get(
            "expected_features", []
        )
    )
    if not missing_total:
        missing_total = sum(
            int(value)
            for name, value in missing_counts.items()
            if name not in {"batch_id", "generated_at", "row_id"}
        )
    invalid_counts: list[dict[str, Any]] = [
        quality.get("invalid_numerical_value_counts", {}),
        quality.get("infinite_value_counts", {}),
        quality.get("negative_value_counts", {}),
        quality.get("unexpected_category_counts", {}),
    ]
    invalid_total = sum(
        int(value)
        for group in invalid_counts
        for value in group.values()
        if isinstance(value, (int, float))
    )
    return {
        "missing_percentage": (
            100 * missing_total / denominator if denominator else 0.0
        ),
        "duplicate_percentage": 100 * float(quality.get("duplicate_rate", 0)),
        "invalid_percentage": (
            100 * invalid_total / denominator if denominator else 0.0
        ),
        "total_rows": row_count,
    }


def monitoring_report_trend_summary(report: dict[str, Any]) -> dict[str, Any]:
    quality = report.get("quality", {})
    feature_names = quality.get("feature_schema", {}).get("expected_features", [])
    quality_summary = _quality_summary(
        report,
        len(feature_names) if isinstance(feature_names, list) else 0,
    )
    thresholds = report.get("thresholds", {})
    warning_threshold = float(
        thresholds.get("psi_warning", DEFAULT_THRESHOLDS.psi_warning)
    )
    critical_threshold = float(
        thresholds.get("psi_critical", DEFAULT_THRESHOLDS.psi_critical)
    )
    drift_features = []
    drift = report.get("drift", {})
    for feature_type in ("numerical", "categorical"):
        for feature, result in drift.get(feature_type, {}).items():
            if not isinstance(result, dict) or not result.get("usable"):
                continue
            psi = result.get("psi", result.get("drift_score"))
            if not isinstance(psi, (int, float)) or not math.isfinite(psi):
                continue
            drift_features.append(
                {
                    "feature": feature,
                    "feature_type": feature_type,
                    "psi": float(psi),
                    "ks_statistic": result.get("ks_statistic"),
                    "ks_p_value": result.get("ks_p_value"),
                    "ks_meaningful_effect": bool(
                        result.get("ks_meaningful_effect", False)
                    ),
                    "severity": (
                        "CRITICAL"
                        if psi >= critical_threshold
                        else "WARNING"
                        if psi >= warning_threshold
                        else "HEALTHY"
                    ),
                }
            )
    drift_features.sort(key=lambda item: item["psi"], reverse=True)
    psi_values = [feature["psi"] for feature in drift_features]
    return {
        "quality_summary": quality_summary,
        "quality_warnings": [
            item
            for item in quality.get("warning_reasons", [])
            if isinstance(item, str)
        ],
        "quality_critical_reasons": [
            item
            for item in quality.get("critical_reasons", [])
            if isinstance(item, str)
        ],
        "drift_summary": {
            "average_psi": (
                sum(psi_values) / len(psi_values) if psi_values else None
            ),
            "drifted_features": sum(
                psi >= warning_threshold for psi in psi_values
            ),
            "critical_features": sum(
                psi >= critical_threshold for psi in psi_values
            ),
            "feature_results": drift_features,
        },
    }


def _model2_summary() -> dict[str, Any]:
    artifact_error = None
    try:
        artifact = _latest_trained_artifact(MODEL2_DIR)
    except Model2ArtifactError as error:
        artifact = None
        artifact_error = str(error)
    dataset = build_historical_dataset()
    dataset_stats = dataset_summary(dataset)
    labeled_count = int(dataset_stats["genuine_labeled_observation_count"])
    production_batch_count = int(dataset_stats["production_observation_count"])
    unlabeled_count = max(0, production_batch_count - labeled_count)
    eligibility = training_eligibility(dataset)
    common = {
        "production_batch_count": production_batch_count,
        "genuine_production_observation_count": production_batch_count,
        "labeled_batch_count": labeled_count,
        "eligible_labeled_observation_count": labeled_count,
        "unlabeled_batch_count": unlabeled_count,
        "labeled_observation_count": labeled_count,
        "training_dataset_size": labeled_count,
        "target_class_counts": dataset_stats["target_class_counts"],
        "minimum_labeled_observations": MIN_LABELED_OBSERVATIONS,
        "missing_requirements": eligibility.get("missing_requirements", []),
        "training_progress_percent": min(
            100.0,
            100.0 * labeled_count / MIN_LABELED_OBSERVATIONS,
        ),
        "prediction_history": list_production_model2_predictions(limit=20),
    }
    status_files = sorted(
        MODEL2_DIR.glob("training_status_*.json"), reverse=True
    )
    latest_training_result = None
    if status_files:
        latest_status = _read_json(status_files[0])
        attempt_timestamp = latest_status.get("training_attempt_timestamp")
        if attempt_timestamp is None:
            try:
                attempt_timestamp = datetime.strptime(
                    status_files[0].stem.removeprefix("training_status_"),
                    "%Y%m%dT%H%M%S%fZ",
                ).replace(tzinfo=timezone.utc).isoformat()
            except ValueError:
                attempt_timestamp = None
        latest_training_result = {
            "status": latest_status.get("model2_status"),
            "timestamp": attempt_timestamp,
            "reason": latest_status.get("reason"),
            "training_performed": latest_status.get("training_performed"),
            "model_version": latest_status.get("model_version"),
            "training_size": latest_status.get("training_size"),
            "testing_size": latest_status.get("testing_size"),
        }
    if artifact is not None:
        _, metadata = artifact
        historical = metadata.get("historical_dataset", {})
        evaluation_metrics = metadata.get("evaluation_metrics", {})
        trained_observations = int(
            historical.get("genuine_labeled_observation_count", 0)
        )
        if (
            labeled_count < MIN_LABELED_OBSERVATIONS
            or trained_observations < MIN_LABELED_OBSERVATIONS
        ):
            artifact = None
    if artifact is not None:
        _, metadata = artifact
        historical = metadata.get("historical_dataset", {})
        evaluation_metrics = metadata.get("evaluation_metrics", {})
        if latest_training_result is None:
            latest_training_result = {
                "status": "TRAINED",
                "timestamp": metadata.get("training_timestamp"),
                "reason": None,
                "training_performed": True,
                "model_version": metadata.get("model_version"),
                "training_size": historical.get("training_size"),
                "testing_size": historical.get("testing_size"),
            }
        trained_observations = int(
            historical.get("genuine_labeled_observation_count", 0)
        )
        trained_at = _timestamp(metadata.get("training_timestamp"))
        last_attempt_at = _timestamp(latest_training_result.get("timestamp"))
        if (
            trained_at is not None
            and (last_attempt_at is None or last_attempt_at < trained_at)
        ):
            latest_training_result = {
                "status": "TRAINED",
                "timestamp": metadata.get("training_timestamp"),
                "reason": None,
                "training_performed": True,
                "model_version": metadata.get("model_version"),
                "training_size": historical.get("training_size"),
                "testing_size": historical.get("testing_size"),
            }
        return {
            "status": "TRAINED",
            "training_eligibility": (
                "READY"
                if eligibility["status"] == "READY"
                and labeled_count > trained_observations
                else "TRAINED"
            ),
            "reason": (
                eligibility["reason"]
                if eligibility["status"] != "READY"
                else None
            ),
            **common,
            "training_size": historical.get("training_size", 0),
            "testing_size": historical.get("testing_size", 0),
            "evaluation_dataset_size": historical.get("testing_size", 0),
            "training_class_counts": historical.get("training_class_counts", {}),
            "testing_class_counts": historical.get("testing_class_counts", {}),
            "model_version": metadata.get("model_version"),
            "training_timestamp": metadata.get("training_timestamp"),
            "latest_training_result": latest_training_result,
            "evaluation_metrics": (
                evaluation_metrics.get("random_forest_primary_candidate")
                if isinstance(evaluation_metrics, dict)
                else None
            ),
        }

    latest_training_result = latest_training_result or None
    if artifact_error:
        return {
            "status": "NOT_TRAINED",
            "training_eligibility": eligibility["status"],
            "reason": artifact_error,
            **common,
            "training_size": 0,
            "testing_size": 0,
            "evaluation_dataset_size": 0,
            "training_class_counts": {},
            "testing_class_counts": {},
            "model_version": None,
            "training_timestamp": None,
            "latest_training_result": latest_training_result,
            "evaluation_metrics": None,
        }
    if eligibility["status"] == "READY":
        return {
            "status": "NOT_TRAINED",
            "training_eligibility": "READY",
            "reason": (
                "Production training requirements are satisfied, but no trained "
                "production Model 2 artifact is available."
            ),
            **common,
            "training_size": 0,
            "testing_size": 0,
            "evaluation_dataset_size": 0,
            "training_class_counts": {},
            "testing_class_counts": {},
            "model_version": None,
            "training_timestamp": None,
            "latest_training_result": latest_training_result,
            "evaluation_metrics": None,
        }
    return {
        "status": "NOT_TRAINED",
        "training_eligibility": "NOT_TRAINED",
        "reason": eligibility["reason"] or "Model 2 has not been trained yet.",
        **common,
        "training_size": 0,
        "testing_size": 0,
        "evaluation_dataset_size": 0,
        "training_class_counts": {},
        "testing_class_counts": {},
        "model_version": None,
        "training_timestamp": None,
        "latest_training_result": latest_training_result,
        "evaluation_metrics": None,
    }


def _model2_development_summary() -> dict[str, Any]:
    try:
        _, metadata = _read_development_artifact(MODEL2_DEVELOPMENT_DIR)
        metrics_path = (
            MODEL2_DEVELOPMENT_DIR
            / f"{metadata['model_version']}-metrics.json"
        )
        metrics_artifact = _read_json(metrics_path)
        if (
            metadata.get("data_provenance") != MODEL2_DEVELOPMENT_PROVENANCE
            or metrics_artifact.get("data_provenance")
            != MODEL2_DEVELOPMENT_PROVENANCE
        ):
            raise DevelopmentModelError(
                "Development metrics lack synthetic-only provenance."
            )
        evaluation = metrics_artifact.get("evaluation_metrics")
        if (
            not isinstance(evaluation, dict)
            or evaluation.get("data_provenance")
            != MODEL2_DEVELOPMENT_PROVENANCE
        ):
            raise DevelopmentModelError(
                "Development evaluation metrics are missing or have invalid provenance."
            )
        metric_names = ("accuracy", "precision", "recall", "f1_score")
        metrics: dict[str, float] = {}
        for name in metric_names:
            value = evaluation.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise DevelopmentModelError(
                    f"Development evaluation metric {name!r} is invalid."
                )
            metrics[name] = float(value)
        train_size = evaluation.get("train_size")
        test_size = evaluation.get("test_size")
        sample_count = metadata.get("sample_count")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (train_size, test_size, sample_count)
        ):
            raise DevelopmentModelError(
                "Development training and evaluation sample counts are invalid."
            )
        return {
            "status": "AVAILABLE",
            "training_status": "DEVELOPMENT_TRAINED",
            "provenance": MODEL2_DEVELOPMENT_PROVENANCE,
            "model_name": metadata["model_name"],
            "model_version": metadata["model_version"],
            "algorithm": metadata["algorithm"],
            "sample_count": sample_count,
            "training_size": train_size,
            "testing_size": test_size,
            "evaluation_metrics": metrics,
            "reason": None,
        }
    except (
        DevelopmentModelError,
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
        KeyError,
    ) as error:
        return {
            "status": "UNAVAILABLE",
            "training_status": "UNAVAILABLE",
            "provenance": MODEL2_DEVELOPMENT_PROVENANCE,
            "model_name": None,
            "model_version": None,
            "algorithm": None,
            "sample_count": 0,
            "training_size": 0,
            "testing_size": 0,
            "evaluation_metrics": None,
            "reason": str(error),
        }


def _development_simulation_summary() -> dict[str, Any]:
    empty_summary = {
        **SYNTHETIC_SIMULATION_PROVENANCE,
        "mode": "DEVELOPMENT SIMULATION",
        "status": "NOT_RUN",
        "run_id": None,
        "batch_count": 0,
        "booking_count": 0,
        "seed": None,
        "baseline_f1_score": None,
        "minimum_batch_f1_score": None,
        "drift_detected_batch_count": 0,
        "monitoring_status_counts": {},
        "model2_training": None,
        "reason": "Run the synthetic hotel simulator to create a development-only result.",
    }
    if not DEVELOPMENT_SIMULATION_DIR.exists():
        return empty_summary
    summaries = list(DEVELOPMENT_SIMULATION_DIR.glob("*/simulation_summary.json"))
    if not summaries:
        return empty_summary
    summary_path = max(summaries, key=lambda path: path.stat().st_mtime)
    try:
        summary = _read_json(summary_path)
        if (
            any(summary.get(name) != value for name, value in SYNTHETIC_SIMULATION_PROVENANCE.items())
            or summary.get("mode") != "DEVELOPMENT SIMULATION"
            or summary.get("run_status") != "COMPLETE"
        ):
            raise ValueError("Latest simulation summary has invalid synthetic provenance.")
        batch_count = summary.get("batch_count")
        booking_count = summary.get("booking_count")
        training = summary.get("model2_training")
        training_classes = (
            training.get("training_class_counts")
            if isinstance(training, dict)
            else None
        )
        testing_classes = (
            training.get("testing_class_counts")
            if isinstance(training, dict)
            else None
        )
        evaluation = (
            training.get("evaluation_metrics")
            if isinstance(training, dict)
            else None
        )
        if (
            not isinstance(batch_count, int)
            or batch_count < 20
            or not isinstance(booking_count, int)
            or booking_count <= 0
            or not isinstance(training, dict)
            or training.get("status") != "DEVELOPMENT_SIMULATION_TRAINED"
            or training.get("split_method") != "chronological holdout"
            or not isinstance(training_classes, dict)
            or not isinstance(testing_classes, dict)
            or set(training_classes) != {"0", "1"}
            or set(testing_classes) != {"0", "1"}
            or not all(
                isinstance(count, int) and count > 0
                for count in (*training_classes.values(), *testing_classes.values())
            )
            or not isinstance(evaluation, dict)
            or any(
                isinstance(evaluation.get(name), bool)
                or not isinstance(evaluation.get(name), (int, float))
                or not math.isfinite(evaluation[name])
                or not 0 <= evaluation[name] <= 1
                for name in ("accuracy", "precision", "recall", "f1_score")
            )
        ):
            raise ValueError("Latest simulation summary is incomplete.")
        return {
            **SYNTHETIC_SIMULATION_PROVENANCE,
            "mode": "DEVELOPMENT SIMULATION",
            "status": "AVAILABLE",
            "run_id": str(summary["run_id"]),
            "batch_count": batch_count,
            "booking_count": booking_count,
            "seed": int(summary["seed"]),
            "baseline_f1_score": float(summary["baseline_f1_score"]),
            "minimum_batch_f1_score": float(summary["minimum_batch_f1_score"]),
            "drift_detected_batch_count": int(
                summary["drift_detected_batch_count"]
            ),
            "monitoring_status_counts": summary["monitoring_status_counts"],
            "model2_training": training,
            "reason": None,
        }
    except (OSError, TypeError, ValueError, KeyError) as error:
        return {
            **empty_summary,
            "status": "UNAVAILABLE",
            "reason": str(error),
        }


def _historical_replay_summary() -> dict[str, Any]:
    empty_summary = {
        "status": "NOT_RUN",
        "model_scope": HISTORICAL_REPLAY_PROVENANCE,
        "provenance": HISTORICAL_REPLAY_PROVENANCE,
        "data_source_type": "historical_offline",
        "source": HISTORICAL_REPLAY_SOURCE,
        "environment": "offline",
        "is_synthetic": False,
        "run_id": None,
        "historical_record_count": 0,
        "arrival_window_count": 0,
        "arrival_date_range": [],
        "model1_metrics": None,
        "model2_training_status": None,
        "model2_eligibility": None,
        "prediction_scope": None,
        "booking_created_timestamp_available": None,
        "limitations": [],
        "reason": "Run the offline historical replay to create a saved result.",
    }
    if not HISTORICAL_REPLAY_DIR.exists():
        return empty_summary
    summaries = list(HISTORICAL_REPLAY_DIR.glob("*/summary.json"))
    if not summaries:
        return empty_summary

    try:
        summary_path = max(summaries, key=lambda path: path.stat().st_mtime)
        summary = _read_json(summary_path)
        if (
            summary.get("status") != "COMPLETE"
            or summary.get("model_scope") != HISTORICAL_REPLAY_PROVENANCE
            or summary.get("provenance") != HISTORICAL_REPLAY_PROVENANCE
            or summary.get("data_source_type") != "historical_offline"
            or summary.get("source") != HISTORICAL_REPLAY_SOURCE
            or summary.get("environment") != "offline"
            or summary.get("is_synthetic") is not False
            or summary.get("run_id") != summary_path.parent.name
            or summary.get("prediction_scope")
            != "retrospective_historical_feature_replay_not_booking_time"
        ):
            raise ValueError(
                "Latest historical replay summary has invalid provenance or scope."
            )

        model1_metrics = summary.get("model1_metrics")
        metric_names = ("accuracy", "precision", "recall", "f1_score")
        if not isinstance(model1_metrics, dict) or any(
            isinstance(model1_metrics.get(name), bool)
            or not isinstance(model1_metrics.get(name), (int, float))
            or not math.isfinite(model1_metrics[name])
            or not 0 <= model1_metrics[name] <= 1
            for name in metric_names
        ):
            raise ValueError("Historical replay Model 1 metrics are invalid.")

        eligibility = summary.get("model2_eligibility")
        if not isinstance(eligibility, dict) or eligibility.get("status") not in {
            "READY",
            "NOT_ELIGIBLE",
        }:
            raise ValueError("Historical replay Model 2 eligibility is invalid.")
        training = summary.get("model2_training")
        if not isinstance(training, dict) or training.get("status") not in {
            "TRAINED",
            "NOT_ELIGIBLE",
        }:
            raise ValueError("Historical replay Model 2 training status is invalid.")

        def nonnegative_integer(value: Any) -> bool:
            return (
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
            )

        def count_record(value: Any) -> bool:
            return isinstance(value, dict) and all(
                isinstance(key, str) and nonnegative_integer(count)
                for key, count in value.items()
            )

        eligibility_count_fields = (
            "eligible_observation_count",
            "training_size_after_purge",
            "testing_size",
        )
        if any(
            not nonnegative_integer(eligibility.get(name))
            for name in eligibility_count_fields
        ) or any(
            not count_record(eligibility.get(name))
            for name in (
                "target_class_counts",
                "training_class_counts",
                "testing_class_counts",
            )
        ):
            raise ValueError("Historical replay Model 2 counts are invalid.")
        missing_requirements = eligibility.get("missing_requirements")
        limitations = summary.get("limitations")
        date_range = summary.get("arrival_date_range")
        record_count = summary.get("historical_record_count")
        window_count = summary.get("model1_evaluation_window_count")
        if (
            not isinstance(missing_requirements, list)
            or not all(isinstance(item, str) for item in missing_requirements)
            or not isinstance(limitations, list)
            or not all(isinstance(item, str) for item in limitations)
            or not isinstance(date_range, list)
            or len(date_range) != 2
            or not all(isinstance(item, str) for item in date_range)
            or not nonnegative_integer(record_count)
            or not nonnegative_integer(window_count)
            or summary.get("booking_created_timestamp_available") is not False
        ):
            raise ValueError("Historical replay summary fields are invalid.")

        reason = eligibility.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("Historical replay eligibility reason is invalid.")
        return {
            **empty_summary,
            "status": "AVAILABLE",
            "run_id": summary["run_id"],
            "historical_record_count": record_count,
            "arrival_window_count": window_count,
            "arrival_date_range": date_range,
            "model1_metrics": {
                name: float(model1_metrics[name]) for name in metric_names
            },
            "model2_training_status": training["status"],
            "model2_eligibility": {
                "status": eligibility["status"],
                "eligible_observation_count": eligibility[
                    "eligible_observation_count"
                ],
                "target_class_counts": eligibility["target_class_counts"],
                "training_size_after_purge": eligibility[
                    "training_size_after_purge"
                ],
                "testing_size": eligibility["testing_size"],
                "training_class_counts": eligibility["training_class_counts"],
                "testing_class_counts": eligibility["testing_class_counts"],
                "missing_requirements": missing_requirements,
                "reason": reason,
            },
            "prediction_scope": summary["prediction_scope"],
            "booking_created_timestamp_available": False,
            "limitations": limitations,
            "reason": None,
        }
    except (OSError, TypeError, ValueError, KeyError) as error:
        return {
            **empty_summary,
            "status": "UNAVAILABLE",
            "reason": f"Latest historical replay summary is unavailable or invalid: {error}",
        }


def _model2_development_explainability() -> dict[str, Any]:
    try:
        report = _read_json(MODEL2_DEVELOPMENT_IMPORTANCE_PATH)
        if report.get("provenance") != DEVELOPMENT_LABEL:
            raise DevelopmentExplainabilityError(
                "Development explainability report has invalid provenance."
            )
        if report.get("feature_names") != MODEL2_FEATURE_NAMES:
            raise DevelopmentExplainabilityError(
                "Development explainability report has an incompatible feature schema."
            )
        all_features = report.get("features")
        features = report.get("top_features")
        if (
            not isinstance(all_features, list)
            or len(all_features) != len(MODEL2_FEATURE_NAMES)
            or not isinstance(features, list)
            or not features
            or len(features) > 10
        ):
            raise DevelopmentExplainabilityError(
                "Development explainability report has an invalid ranked feature list."
            )
        seen_features: set[str] = set()
        for expected_rank, feature in enumerate(all_features, start=1):
            if (
                not isinstance(feature, dict)
                or feature.get("rank") != expected_rank
                or feature.get("feature") not in MODEL2_FEATURE_NAMES
                or feature["feature"] in seen_features
                or not isinstance(feature.get("importance"), (int, float))
                or not math.isfinite(feature["importance"])
                or feature["importance"] < 0
                or feature["importance"] > 1
                or (expected_rank > 1 and all_features[expected_rank - 2]["importance"] < feature["importance"])
            ):
                raise DevelopmentExplainabilityError(
                    "Development explainability report contains invalid ranked features."
                )
            seen_features.add(feature["feature"])
        if seen_features != set(MODEL2_FEATURE_NAMES):
            raise DevelopmentExplainabilityError(
                "Development explainability report does not include the full feature schema."
            )
        if not math.isclose(
            sum(feature["importance"] for feature in all_features),
            1.0,
            rel_tol=1e-5,
            abs_tol=1e-8,
        ):
            raise DevelopmentExplainabilityError(
                "Development explainability importances do not sum approximately to 1."
            )
        for expected_rank, feature in enumerate(features, start=1):
            if (
                not isinstance(feature, dict)
                or feature.get("rank") != expected_rank
                or feature != all_features[expected_rank - 1]
                or not isinstance(feature.get("importance"), (int, float))
            ):
                raise DevelopmentExplainabilityError(
                    "Development explainability report contains invalid ranked features."
                )
        return {
            "status": "AVAILABLE",
            "provenance": DEVELOPMENT_LABEL,
            "model_version": report.get("model_version"),
            "importance_method": report.get("importance_method"),
            "generation_timestamp": report.get("generation_timestamp"),
            "causality_warning": report.get("causality_warning"),
            "features": features,
            "feature_names": MODEL2_FEATURE_NAMES,
            "chart_path": report.get("chart_path"),
        }
    except (OSError, TypeError, ValueError, KeyError) as error:
        return {
            "status": "UNAVAILABLE",
            "provenance": DEVELOPMENT_LABEL,
            "model_version": None,
            "importance_method": None,
            "generation_timestamp": None,
            "causality_warning": (
                "Development-only explainability is unavailable; no production "
                "feature-importance values are substituted."
            ),
            "features": [],
            "feature_names": MODEL2_FEATURE_NAMES,
            "chart_path": None,
            "reason": str(error),
        }


def _production_observation_summary() -> dict[str, Any]:
    observations = list_production_observations(
        period_days=3650,
        limit=None,
    )
    labeled_count = sum(
        observation["actual_outcome"] is not None
        for observation in observations
    )
    return {
        "total_observations": len(observations),
        "labeled_observations": labeled_count,
        "pending_outcomes": len(observations) - labeled_count,
        "prediction_history": observations[:20],
    }


def _model_evaluation_visualizations() -> dict[str, Any]:
    return _read_json(MODEL_EVALUATION_PATH)


def build_dashboard_data(period_days: int = 30) -> dict[str, Any]:
    metadata = _read_json(MODEL_METADATA_PATH)
    evaluation = metadata["evaluation_metrics"]
    positive_class = evaluation["classification_report"]["1"]
    split = metadata["training_configuration"]["train_test_split"]
    feature_names = metadata["feature_names"]
    evaluation_visualizations = _model_evaluation_visualizations()

    reports = _load_reports(period_days)
    simulation_reports = _load_simulation_reports(period_days)
    drift_insights_by_source = _drift_insights_by_source(
        _load_all_reports(period_days)
    )
    display_reports = reports or simulation_reports
    display_batch_sources = {
        str(report["batch_id"]): str(report["_dashboard_source"])
        for report in display_reports
    }
    latest_by_batch: dict[str, dict[str, Any]] = {}
    for report in display_reports:
        latest_by_batch.setdefault(str(report["batch_id"]), report)
    latest_reports = sorted(
        latest_by_batch.values(),
        key=lambda item: item["_dashboard_timestamp"],
        reverse=True,
    )
    latest_report = latest_reports[0] if latest_reports else None
    drift_features = _drift_features(latest_report)
    warning_threshold = float(
        (latest_report or {}).get("thresholds", {}).get(
            "psi_warning", DEFAULT_THRESHOLDS.psi_warning
        )
    )
    critical_threshold = float(
        (latest_report or {}).get("thresholds", {}).get(
            "psi_critical", DEFAULT_THRESHOLDS.psi_critical
        )
    )
    usable_drift_scores = [item["score"] for item in drift_features]
    display_batch_ids = {str(report["batch_id"]) for report in display_reports}
    prediction_summary = _load_prediction_summary(display_batch_ids)
    real_prediction_summary = (
        prediction_summary
        if reports
        else _load_prediction_summary(
            {str(report["batch_id"]) for report in reports}
        )
    )
    operational_records = _load_real_performance(
        period_days,
        {str(report["batch_id"]) for report in reports},
    )
    operational_history = [
        {
            "batch_id": record["batch_id"],
            "timestamp": record["_dashboard_timestamp"].isoformat(),
            "model_version": record["model_version"],
            "sample_count": int(record["number_of_evaluated_records"]),
            "accuracy": record["accuracy"],
            "precision": record["precision"],
            "recall": record["recall"],
            "f1_score": record["f1_score"],
            "confusion_matrix": record["confusion_matrix"],
            "data_source_type": "real",
        }
        for record in operational_records
    ]
    evaluated_by_batch = {
        str(record["batch_id"]): int(record["number_of_evaluated_records"])
        for record in operational_records
    }
    pending_outcomes = sum(
        max(0, count - evaluated_by_batch.get(batch_id, 0))
        for batch_id, count in real_prediction_summary["batches"].items()
    )
    source_counts = _monitoring_source_counts(period_days)
    latest_timestamp = (
        latest_report["_dashboard_timestamp"].isoformat()
        if latest_report
        else None
    )
    latest_status = latest_report.get("overall_status", "UNKNOWN") if latest_report else "UNKNOWN"

    alert_source = "REAL" if reports else "SIMULATED"
    selected_batch_ids = {str(report["batch_id"]) for report in latest_reports}
    alerts = [
        {
            "alert_id": str(alert["alert_id"]),
            "title": (
                f"{alert['severity'].title()} · {alert['affected_feature']} · "
                f"{alert['batch_id']}"
            ),
            "level": str(alert["severity"]).lower(),
            "timestamp": str(alert["timestamp"]),
            "source": str(alert.get("data_source", "UNKNOWN")).lower(),
            "batch_id": str(alert["batch_id"]),
        }
        for alert in load_alert_history(ALERTS_PATH, source=alert_source)
        if alert.get("severity") != "HEALTHY"
        and str(alert.get("batch_id")) in selected_batch_ids
    ]

    training_date = _timestamp(metadata.get("training_date"))
    prediction_total = int(prediction_summary["total"])
    distribution_total = sum(prediction_summary["class_counts"].values())
    simulation_batch_ids = {
        str(report["batch_id"]) for report in simulation_reports
    }
    simulation_prediction_summary = _load_prediction_summary(simulation_batch_ids)
    production_observations = _production_observation_summary()
    return {
        "model": {
            "name": metadata["model_name"],
            "algorithm": metadata["algorithm"],
            "training_date": (
                training_date.isoformat() if training_date is not None else None
            ),
            "version": "1.0.0",
            "status": "READY" if MODEL_PIPELINE_PATH.is_file() else "UNAVAILABLE",
            "validation_scope": "Held-out test set",
            "dataset_row_count": evaluation_visualizations["dataset_row_count"],
            "dataset_period": evaluation_visualizations["dataset_period"],
            "feature_importance": evaluation_visualizations["feature_importance"],
        },
        "model2": _model2_summary(),
        "model2_offline": offline_model2_status(),
        "historical_replay": _historical_replay_summary(),
        "model2_development": _model2_development_summary(),
        "model2_development_explainability": (
            _model2_development_explainability()
        ),
        "development_simulation": _development_simulation_summary(),
        "production_observations": production_observations,
        "metrics": {
            "accuracy": float(evaluation["accuracy"]),
            "precision": float(positive_class["precision"]),
            "recall": float(positive_class["recall"]),
            "f1_score": float(positive_class["f1-score"]),
            "sample_count": int(split["test_rows"]),
            "confusion_matrix": evaluation["confusion_matrix"],
            "roc_auc": evaluation_visualizations["roc_auc"],
            "roc_curve": evaluation_visualizations["roc_curve"],
            "performance_history": evaluation_visualizations[
                "performance_history"
            ],
        },
        "latest_monitoring": (
            {
                "batch_id": latest_report["batch_id"],
                "timestamp": latest_timestamp,
                "status": latest_status,
                "quality_status": latest_report.get("quality", {}).get(
                    "quality_status", "UNKNOWN"
                ),
                "source": latest_report["_dashboard_source"],
                "rows": int(
                    latest_report.get("quality", {}).get("total_rows", 0)
                ),
                "reasons": latest_report.get("reasons", []),
            }
            if latest_report
            else None
        ),
        "drift": {
            "features": drift_features,
            "average_score": (
                sum(usable_drift_scores) / len(usable_drift_scores)
                if usable_drift_scores
                else None
            ),
            "drifted_features": sum(
                score >= warning_threshold for score in usable_drift_scores
            ),
            "critical_features": sum(
                score >= critical_threshold for score in usable_drift_scores
            ),
            "threshold": critical_threshold,
            "warning_threshold": warning_threshold,
            "insights_by_source": drift_insights_by_source,
        },
        "quality": _quality_summary(latest_report, len(feature_names)),
        "predictions": {
            "total": prediction_total,
            "today": prediction_summary["today"],
            "class_counts": prediction_summary["class_counts"],
            "distribution_sample_count": distribution_total,
            "batch_count": len(prediction_summary["batches"]),
            "batches": prediction_summary["batches"],
            "prediction_history": _load_prediction_history(
                display_batch_sources
            ),
            "source": (
                "real"
                if reports
                else "simulated"
                if simulation_prediction_summary["total"]
                else "none"
            ),
        },
        "alerts": alerts[:5],
        "monitoring": {
            "report_count": len(reports),
            "batch_count": len(latest_reports),
            "simulation_report_count": len(simulation_reports),
            "simulation_batch_count": len(simulation_batch_ids),
            "simulation_record_count": sum(
                int(report.get("quality", {}).get("total_rows", 0))
                for report in simulation_reports
            ),
            "period_days": period_days,
            "source_counts": source_counts,
        },
        "operational_performance": {
            "evaluated_batch_count": len(operational_history),
            "evaluated_record_count": sum(
                item["sample_count"] for item in operational_history
            ),
            "pending_outcome_count": pending_outcomes,
            "history": operational_history,
        },
        "system": {
            "api": "ONLINE",
            "model_artifact": (
                "AVAILABLE" if MODEL_PIPELINE_PATH.is_file() else "MISSING"
            ),
            "monitoring_reports": (
                "AVAILABLE" if reports else "AWAITING"
            ),
            "live_outcomes": (
                "AVAILABLE"
                if operational_history
                else "AWAITING"
            ),
        },
    }
