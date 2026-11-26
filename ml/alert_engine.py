from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml.monitoring_config import DEFAULT_THRESHOLDS, MonitoringThresholds

ROOT = Path(__file__).resolve().parents[1]
ALERTS_PATH = ROOT / "data" / "monitoring" / "alerts" / "alerts.jsonl"
REPORTS_PATH = ROOT / "data" / "monitoring" / "reports"
SEEDED_BATCH = re.compile(
    r"(?:normal|drifted|poor_quality)-.+-seed-\d+",
    flags=re.IGNORECASE,
)


def _source(report: dict[str, Any]) -> str:
    source = report.get("data_source_type")
    if source not in {"real", "synthetic_test", "unknown"}:
        source = report.get("_dashboard_source", "unknown")
    return {
        "real": "REAL",
        "synthetic_test": "SIMULATED",
        "unknown": "UNKNOWN",
    }.get(str(source), "UNKNOWN")


def _severity_for_feature(
    psi: float | None,
    quality_critical: bool,
    meaningful_ks: bool,
    thresholds: MonitoringThresholds,
) -> str | None:
    if quality_critical or (psi is not None and psi >= thresholds.psi_critical):
        return "CRITICAL"
    if (
        (psi is not None and psi >= thresholds.psi_warning)
        or meaningful_ks
    ):
        return "WARNING"
    return None


def _recommendations(causes: list[dict[str, Any]], severity: str) -> list[str]:
    recommendations: list[str] = []
    kinds = {cause["type"] for cause in causes}
    if "missing_values" in kinds:
        recommendations.append(
            "Validate the incoming schema and investigate why this feature is missing more often."
        )
    if "unexpected_categories" in kinds:
        recommendations.append(
            "Check category mapping and upstream feature-pipeline changes before accepting new values."
        )
    if "invalid_values" in kinds:
        recommendations.append(
            "Validate feature types, ranges, and unit conversions at ingestion."
        )
    if "numerical_distribution_change" in kinds or "categorical_frequency_change" in kinds:
        recommendations.append(
            "Investigate the affected feature's source, collection window, and transformation pipeline."
        )
    if severity != "HEALTHY":
        recommendations.append(
            "Do not retrain from drift alone; consider retraining only after a persistent shift is confirmed and genuine labeled outcomes are evaluated."
        )
    if not recommendations:
        recommendations.extend(
            [
                "Continue routine monitoring and validate the next incoming batch.",
                "Request genuine actual labels when evaluating prediction performance.",
            ]
        )
    elif "Request genuine actual labels when evaluating prediction performance." not in recommendations:
        recommendations.append(
            "Request genuine actual labels before making claims about model performance."
        )
    return recommendations


def explain_monitoring_report(
    report: dict[str, Any],
    *,
    thresholds: MonitoringThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    drift = report.get("drift", {})
    quality = report.get("quality", {})
    numerical = drift.get("numerical", {})
    categorical = drift.get("categorical", {})
    drifted_features: list[dict[str, Any]] = []
    root_causes: list[dict[str, Any]] = []

    feature_names = sorted(set(numerical) | set(categorical))
    for feature in feature_names:
        numeric_result = numerical.get(feature, {})
        categorical_result = categorical.get(feature, {})
        result = numeric_result if numeric_result else categorical_result
        if not result.get("usable"):
            continue

        psi = result.get("psi", result.get("drift_score"))
        psi = float(psi) if isinstance(psi, (int, float)) else None
        ks_statistic = numeric_result.get("ks_statistic")
        ks_p_value = numeric_result.get("ks_p_value")
        meaningful_ks = bool(numeric_result.get("ks_meaningful_effect"))
        invalid_count = int(
            quality.get("invalid_numerical_value_counts", {}).get(feature, 0)
        )
        infinite_count = int(
            quality.get("infinite_value_counts", {}).get(feature, 0)
        )
        negative_count = int(
            quality.get("negative_value_counts", {}).get(feature, 0)
        )
        incorrect_type = quality.get("incorrect_data_types", {}).get(feature)
        unexpected_count = int(
            quality.get("unexpected_category_counts", {}).get(feature, 0)
        )
        missing_count = int(quality.get("missing_value_counts", {}).get(feature, 0))
        missing_delta = quality.get("missing_excess_percentage_points", {}).get(
            feature
        )
        quality_critical = bool(
            invalid_count
            or infinite_count
            or negative_count
            or unexpected_count
            or incorrect_type
        )
        severity = _severity_for_feature(
            psi, quality_critical, meaningful_ks, thresholds
        )
        causes: list[dict[str, Any]] = []

        if missing_count or isinstance(missing_delta, (int, float)) and missing_delta > 0:
            causes.append(
                {
                    "type": "missing_values",
                    "confidence": "evidence",
                    "description": "Missing values increased or are present in this feature.",
                    "evidence": {
                        "missing_count": missing_count,
                        "incoming_missing_percent": quality.get(
                            "missing_value_percentages", {}
                        ).get(feature),
                        "reference_missing_percent": quality.get(
                            "reference_missing_percentages", {}
                        ).get(feature),
                        "increase_percentage_points": missing_delta,
                    },
                }
            )
        if unexpected_count:
            causes.append(
                {
                    "type": "unexpected_categories",
                    "confidence": "evidence",
                    "description": "Values not present in the reference categories were observed.",
                    "evidence": {
                        "count": unexpected_count,
                        "values": quality.get("unexpected_category_values", {}).get(
                            feature, []
                        ),
                        "reference_categories": categorical_result.get(
                            "reference_categories", []
                        ),
                    },
                }
            )
        if invalid_count or infinite_count or negative_count or incorrect_type:
            causes.append(
                {
                    "type": "invalid_values",
                    "confidence": "evidence",
                    "description": "Invalid, infinite, or out-of-range values were found.",
                    "evidence": {
                        "invalid_numerical_count": invalid_count,
                        "infinite_count": infinite_count,
                        "negative_count": negative_count,
                        "type_evidence": incorrect_type,
                    },
                }
            )

        if psi is not None and psi >= thresholds.psi_warning:
            distribution = (
                numeric_result.get("distribution")
                if numeric_result
                else categorical_result.get("category_distribution_differences")
            )
            causes.append(
                {
                    "type": (
                        "numerical_distribution_change"
                        if numeric_result
                        else "categorical_frequency_change"
                    ),
                    "confidence": "suggested",
                    "description": (
                        "Observed distributions differ from the reference; this is a possible signal, not a confirmed root cause."
                    ),
                    "evidence": {
                        "psi": psi,
                        "psi_level": result.get(
                            "psi_level", result.get("drift_level")
                        ),
                        "ks_statistic": ks_statistic,
                        "ks_p_value": ks_p_value,
                        "ks_meaningful_effect": meaningful_ks,
                        "reference_valid_count": result.get("reference_valid_count"),
                        "incoming_valid_count": result.get("incoming_valid_count"),
                        "distribution": distribution,
                        "new_categories": categorical_result.get(
                            "new_categories", []
                        ),
                    },
                }
            )
        elif meaningful_ks:
            causes.append(
                {
                    "type": "numerical_distribution_change",
                    "confidence": "suggested",
                    "description": "KS detects a meaningful distribution difference even though PSI is below its warning threshold.",
                    "evidence": {
                        "psi": psi,
                        "ks_statistic": ks_statistic,
                        "ks_p_value": ks_p_value,
                        "reference_valid_count": numeric_result.get(
                            "reference_valid_count"
                        ),
                        "incoming_valid_count": numeric_result.get(
                            "incoming_valid_count"
                        ),
                    },
                }
            )

        frequency_changes = categorical_result.get(
            "category_distribution_differences", {}
        )
        largest_changes = sorted(
            (
                {
                    "category": category,
                    "reference_proportion": values["reference_proportion"],
                    "incoming_proportion": values["incoming_proportion"],
                    "change_percentage_points": values["difference"] * 100,
                }
                for category, values in frequency_changes.items()
                if abs(values.get("difference", 0)) >= 0.05
            ),
            key=lambda item: abs(item["change_percentage_points"]),
            reverse=True,
        )[:5]
        if largest_changes and not any(
            cause["type"] == "categorical_frequency_change" for cause in causes
        ):
            causes.append(
                {
                    "type": "categorical_frequency_change",
                    "confidence": "suggested",
                    "description": "Several category proportions changed; review upstream sources and mappings.",
                    "evidence": {
                        "psi": psi,
                        "largest_category_changes": largest_changes,
                    },
                }
            )

        explanation = (
            f"PSI {psi:.3f}"
            if psi is not None
            else "PSI unavailable"
        )
        if numeric_result:
            explanation += (
                f"; KS statistic {ks_statistic:.3f}"
                if isinstance(ks_statistic, (int, float))
                else "; KS unavailable"
            )
            if isinstance(ks_p_value, (int, float)):
                explanation += f" (p={ks_p_value:.3g})"
        explanation += (
            f"; {len(causes)} evidence item(s)"
            if causes
            else "; no configured drift threshold exceeded"
        )
        if numeric_result:
            reference_distribution = numeric_result.get("distribution", {}).get(
                "reference"
            )
            current_distribution = numeric_result.get("distribution", {}).get(
                "incoming"
            )
        else:
            reference_distribution = [
                {"category": category, "proportion": values["reference_proportion"]}
                for category, values in frequency_changes.items()
            ]
            current_distribution = [
                {"category": category, "proportion": values["incoming_proportion"]}
                for category, values in frequency_changes.items()
            ]
        drifted_features.append(
            {
                "feature": feature,
                "feature_type": "numerical" if numeric_result else "categorical",
                "psi": psi,
                "psi_level": result.get("psi_level", result.get("drift_level")),
                "ks_statistic": ks_statistic,
                "ks_p_value": ks_p_value,
                "ks_meaningful_effect": meaningful_ks,
                "severity": severity or "HEALTHY",
                "change_summary": explanation,
                "reference_distribution": reference_distribution,
                "current_distribution": current_distribution,
                "root_causes": causes,
            }
        )
        root_causes.extend(
            {"feature": feature, **cause}
            for cause in causes
        )

    for feature, excess in quality.get(
        "missing_excess_percentage_points", {}
    ).items():
        if feature in feature_names or not isinstance(excess, (int, float)) or excess <= 0:
            continue
        cause = {
            "feature": feature,
            "type": "missing_values",
            "confidence": "evidence",
            "description": "Missing values increased or are present in this feature.",
            "evidence": {
                "missing_count": quality.get("missing_value_counts", {}).get(
                    feature, 0
                ),
                "increase_percentage_points": excess,
            },
        }
        root_causes.append(cause)

    drifted_features.sort(
        key=lambda item: (
            {"CRITICAL": 2, "WARNING": 1, "HEALTHY": 0}[item["severity"]],
            item["psi"] or 0,
            item["ks_statistic"] or 0,
        ),
        reverse=True,
    )
    status = str(report.get("overall_status", "UNKNOWN")).upper()
    recommendations = _recommendations(root_causes, status)
    return {
        "batch_id": report.get("batch_id", "UNKNOWN"),
        "timestamp": report.get("monitoring_timestamp"),
        "data_source": _source(report),
        "status": status,
        "thresholds": {
            "psi_warning": thresholds.psi_warning,
            "psi_critical": thresholds.psi_critical,
            "ks_significance_level": thresholds.ks_significance_level,
            "ks_min_effect_size": thresholds.ks_min_effect_size,
        },
        "drifted_features": drifted_features,
        "root_causes": root_causes,
        "recommendations": recommendations,
    }


def _condition_key(alert: dict[str, Any]) -> str:
    return str(alert["condition"])


def generate_alerts(
    report: dict[str, Any],
    *,
    thresholds: MonitoringThresholds = DEFAULT_THRESHOLDS,
) -> list[dict[str, Any]]:
    explanation = explain_monitoring_report(report, thresholds=thresholds)
    batch_id = str(report.get("batch_id", "UNKNOWN"))
    timestamp = str(report.get("monitoring_timestamp") or datetime.now(timezone.utc).isoformat())
    source = _source(report)
    alerts: list[dict[str, Any]] = []
    features_by_name = {
        item["feature"]: item for item in explanation["drifted_features"]
    }

    for feature in explanation["drifted_features"]:
        severity = feature["severity"]
        if severity == "HEALTHY":
            continue
        alert = {
            "batch_id": batch_id,
            "timestamp": timestamp,
            "severity": severity,
            "affected_feature": feature["feature"],
            "detection_reason": feature["change_summary"],
            "data_source": source,
            "recommended_action": " ".join(
                _recommendations(feature["root_causes"], severity)
            ),
            "evidence": {
                "psi": feature["psi"],
                "ks_statistic": feature["ks_statistic"],
                "ks_p_value": feature["ks_p_value"],
                "root_causes": feature["root_causes"],
            },
            "condition": f"feature:{feature['feature']}:{severity}",
        }
        alerts.append(alert)

    quality = report.get("quality", {})
    for feature, count in quality.get(
        "unexpected_category_counts", {}
    ).items():
        if not count:
            continue
        alerts.append(
            {
                "batch_id": batch_id,
                "timestamp": timestamp,
                "severity": "CRITICAL",
                "affected_feature": feature,
                "detection_reason": (
                    f"{count} unexpected categorical value(s): "
                    f"{quality.get('unexpected_category_values', {}).get(feature, [])}"
                ),
                "data_source": source,
                "recommended_action": "Check category mapping and upstream feature pipelines before accepting unseen values. Request genuine labels for later performance evaluation.",
                "evidence": {
                    "unexpected_category_count": count,
                    "values": quality.get("unexpected_category_values", {}).get(
                        feature, []
                    ),
                },
                "condition": f"quality:unexpected_category:{feature}",
            }
        )
    for field in (
        "invalid_numerical_value_counts",
        "infinite_value_counts",
        "negative_value_counts",
    ):
        for feature, count in quality.get(field, {}).items():
            if not count:
                continue
            alerts.append(
                {
                    "batch_id": batch_id,
                    "timestamp": timestamp,
                    "severity": "CRITICAL",
                    "affected_feature": feature,
                    "detection_reason": f"{count} invalid value(s) detected ({field.replace('_', ' ')}).",
                    "data_source": source,
                    "recommended_action": "Validate incoming types, ranges, and units and inspect the feature-generation pipeline.",
                    "evidence": {field: count},
                    "condition": f"quality:{field}:{feature}",
                }
            )
    for feature, type_evidence in quality.get("incorrect_data_types", {}).items():
        alerts.append(
            {
                "batch_id": batch_id,
                "timestamp": timestamp,
                "severity": "CRITICAL",
                "affected_feature": feature,
                "detection_reason": (
                    "Feature data type did not match the expected schema."
                ),
                "data_source": source,
                "recommended_action": "Validate the incoming schema and correct the upstream feature type before using this batch.",
                "evidence": type_evidence,
                "condition": f"quality:incorrect_type:{feature}",
            }
        )
    for feature, excess in quality.get(
        "missing_excess_percentage_points", {}
    ).items():
        if not isinstance(excess, (int, float)) or excess < thresholds.missing_rate_warning_delta * 100:
            continue
        severity = (
            "CRITICAL"
            if excess >= thresholds.missing_rate_critical_delta * 100
            else "WARNING"
        )
        alerts.append(
            {
                "batch_id": batch_id,
                "timestamp": timestamp,
                "severity": severity,
                "affected_feature": feature,
                "detection_reason": f"Missing rate increased by {excess:.2f} percentage points versus reference.",
                "data_source": source,
                "recommended_action": "Validate incoming data and investigate the source mapping and feature pipeline for missing values.",
                "evidence": {
                    "missing_count": quality.get("missing_value_counts", {}).get(
                        feature, 0
                    ),
                    "incoming_missing_percent": quality.get(
                        "missing_value_percentages", {}
                    ).get(feature),
                    "reference_missing_percent": quality.get(
                        "reference_missing_percentages", {}
                    ).get(feature),
                    "increase_percentage_points": excess,
                },
                "condition": f"quality:missing:{feature}:{severity}",
            }
        )

    status = explanation["status"]
    if status == "HEALTHY" and not alerts:
        alerts.append(
            {
                "batch_id": batch_id,
                "timestamp": timestamp,
                "severity": "HEALTHY",
                "affected_feature": "overall",
                "detection_reason": "No configured data-quality or drift thresholds were exceeded.",
                "data_source": source,
                "recommended_action": "Continue routine monitoring and request genuine actual labels before evaluating prediction performance.",
                "evidence": {
                    "quality_status": quality.get("quality_status"),
                    "drifted_feature_count": 0,
                },
                "condition": "overall:HEALTHY",
            }
        )
    elif status in {"WARNING", "CRITICAL"}:
        alerts.append(
            {
                "batch_id": batch_id,
                "timestamp": timestamp,
                "severity": status,
                "affected_feature": "batch",
                "detection_reason": "; ".join(report.get("reasons", []))
                or f"Monitoring status is {status}.",
                "data_source": source,
                "recommended_action": " ".join(explanation["recommendations"]),
                "evidence": {
                    "quality_status": quality.get("quality_status"),
                    "feature_count": len(features_by_name),
                    "reasons": report.get("reasons", []),
                },
                "condition": f"overall:{status}",
            }
        )

    for alert in alerts:
        identity = "\x1f".join(
            (batch_id, source, _condition_key(alert))
        )
        alert["alert_id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        alert["status"] = status
    return alerts


def persist_alerts(
    report: dict[str, Any],
    alerts_path: Path = ALERTS_PATH,
    *,
    thresholds: MonitoringThresholds = DEFAULT_THRESHOLDS,
) -> list[dict[str, Any]]:
    generated = generate_alerts(report, thresholds=thresholds)
    alerts_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids: set[str] = set()
    if alerts_path.exists():
        with alerts_path.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    existing_ids.add(str(json.loads(line)["alert_id"]))
    new_alerts = [
        alert for alert in generated if alert["alert_id"] not in existing_ids
    ]
    if new_alerts:
        with alerts_path.open("a", encoding="utf-8", newline="\n") as file:
            for alert in new_alerts:
                file.write(json.dumps(alert, allow_nan=False) + "\n")
    return generated


def backfill_alert_history(
    reports_dir: Path = REPORTS_PATH,
    alerts_path: Path = ALERTS_PATH,
) -> None:
    if alerts_path.exists() or not reports_dir.exists():
        return
    for report_path in sorted(reports_dir.glob("*.json")):
        with report_path.open(encoding="utf-8") as file:
            report = json.load(file)
        if report.get("data_source_type") not in {
            "real",
            "synthetic_test",
            "unknown",
        }:
            batch_id = str(report.get("batch_id", ""))
            report["_dashboard_source"] = (
                "synthetic_test"
                if SEEDED_BATCH.fullmatch(batch_id)
                else "unknown"
            )
        report_thresholds = report.get("thresholds", {})
        thresholds = MonitoringThresholds(
            **{
                name: report_thresholds[name]
                for name in DEFAULT_THRESHOLDS.__dataclass_fields__
                if name in report_thresholds
            }
        )
        persist_alerts(report, alerts_path, thresholds=thresholds)


def load_alert_history(
    alerts_path: Path = ALERTS_PATH,
    *,
    severity: str | None = None,
    batch_id: str | None = None,
    source: str | None = None,
) -> list[dict[str, Any]]:
    if not alerts_path.exists():
        return []
    alerts = []
    with alerts_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            alert = json.loads(line)
            if severity and alert.get("severity") != severity:
                continue
            if batch_id and alert.get("batch_id") != batch_id:
                continue
            if source and alert.get("data_source") != source:
                continue
            alerts.append(alert)
    return sorted(
        alerts,
        key=lambda item: (item.get("timestamp", ""), item.get("alert_id", "")),
        reverse=True,
    )
