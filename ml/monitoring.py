from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ml.alert_engine import explain_monitoring_report, persist_alerts
from ml.data_quality import assess_data_quality
from ml.drift_detection import calculate_categorical_drift, calculate_numerical_drift
from ml.monitoring_config import DEFAULT_THRESHOLDS, MonitoringThresholds

ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES_PATH = ROOT / "ml" / "models" / "feature_names.json"
MODEL_METADATA_PATH = ROOT / "ml" / "models" / "model_metadata.json"
REFERENCE_DATA_PATH = ROOT / "data" / "raw" / "hotel_bookings.csv"
MONITORING_DIR = ROOT / "data" / "monitoring"
REPORTS_DIR = MONITORING_DIR / "reports"


def _classify(
    quality_report: dict[str, Any],
    numerical_drift: dict[str, Any],
    categorical_drift: dict[str, Any],
    thresholds: MonitoringThresholds,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if quality_report["quality_status"] == "CRITICAL":
        reasons.extend(quality_report["critical_reasons"])
        reasons.extend(quality_report["warning_reasons"])
    elif quality_report["quality_status"] == "WARNING":
        reasons.extend(quality_report["warning_reasons"])

    numerical_psi = {
        feature: result["psi"]
        for feature, result in numerical_drift.items()
        if result.get("usable")
    }
    categorical_psi = {
        feature: result["drift_score"]
        for feature, result in categorical_drift.items()
        if result.get("usable")
    }
    all_scores = {**numerical_psi, **categorical_psi}
    significant_features = [
        feature for feature, score in all_scores.items()
        if score >= thresholds.psi_critical
    ]
    moderate_features = [
        feature for feature, score in all_scores.items()
        if thresholds.psi_warning <= score < thresholds.psi_critical
    ]
    meaningful_ks_features = [
        feature
        for feature, result in numerical_drift.items()
        if result.get("usable") and result.get("ks_meaningful_effect")
    ]
    new_category_features = [
        feature
        for feature, result in categorical_drift.items()
        if result.get("usable") and result.get("has_new_categories")
    ]

    if significant_features:
        reasons.append(
            "Significant PSI drift (>= "
            f"{thresholds.psi_critical:.2f}) in: {significant_features}"
        )
    if moderate_features:
        reasons.append(
            "Moderate PSI drift (>= "
            f"{thresholds.psi_warning:.2f}) in: {moderate_features}"
        )
    if meaningful_ks_features:
        reasons.append(
            "KS tests show statistically significant, meaningful distribution changes in: "
            f"{meaningful_ks_features}"
        )
    if new_category_features:
        reasons.append(f"Unseen categories in: {new_category_features}")

    if quality_report["quality_status"] == "CRITICAL":
        return "CRITICAL", reasons
    if len(significant_features) >= thresholds.critical_drifted_features:
        reasons.append(
            f"{len(significant_features)} features exceed the significant PSI threshold."
        )
        return "CRITICAL", reasons
    if significant_features or moderate_features or meaningful_ks_features:
        return "WARNING", reasons
    if quality_report["quality_status"] == "WARNING":
        return "WARNING", reasons
    return "HEALTHY", reasons or ["No configured quality or drift thresholds were exceeded."]


def build_monitoring_report(
    batch: pd.DataFrame,
    reference_data: pd.DataFrame,
    feature_names: list[str],
    numerical_features: list[str],
    categorical_features: list[str],
    *,
    thresholds: MonitoringThresholds = DEFAULT_THRESHOLDS,
    data_source_type: str = "unknown",
    source: str | None = None,
) -> dict[str, Any]:
    if data_source_type not in {"real", "synthetic_test", "unknown"}:
        raise ValueError(
            "data_source_type must be 'real', 'synthetic_test', or 'unknown'."
        )
    if source not in {None, "production"}:
        raise ValueError("source must be omitted or 'production'.")
    if source == "production" and data_source_type != "real":
        raise ValueError("Production monitoring reports must use real data source type.")
    quality_report = assess_data_quality(
        batch,
        reference_data,
        feature_names,
        numerical_features,
        categorical_features,
        thresholds=thresholds,
    )
    numeric_drift = calculate_numerical_drift(
        reference_data, batch, numerical_features, thresholds=thresholds
    )
    categorical_drift = calculate_categorical_drift(
        reference_data, batch, categorical_features, thresholds=thresholds
    )
    overall_status, reasons = _classify(
        quality_report, numeric_drift, categorical_drift, thresholds
    )
    batch_id = quality_report["batch_id"]
    is_seeded_simulation = re.fullmatch(
        r"(?:normal|drifted|poor_quality)-.+-seed-\d+",
        batch_id,
        flags=re.IGNORECASE,
    ) is not None
    if is_seeded_simulation and data_source_type == "real":
        raise ValueError("Seeded simulator batches cannot be declared as real.")
    if is_seeded_simulation:
        data_source_type = "synthetic_test"
        source = None
    report = {
        "batch_id": batch_id,
        "data_source_type": data_source_type,
        **({"source": source} if source is not None else {}),
        "batch_metadata": {
            column: (
                batch[column].dropna().iloc[0].item()
                if hasattr(batch[column].dropna().iloc[0], "item")
                else batch[column].dropna().iloc[0]
            )
            for column in (
                "timestamp",
                "source",
                "data_source_type",
                "record_count",
                "monitoring_scenario",
                "label_provenance",
            )
            if column in batch.columns and batch[column].notna().any()
        },
        "monitoring_timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall_status,
        "reasons": reasons,
        "quality": quality_report,
        "drift": {
            "numerical": numeric_drift,
            "categorical": categorical_drift,
        },
        "model_prediction": {
            "performed": False,
            "reason": (
                "This monitoring run only evaluates input quality and drift; "
                "it does not invoke Model 1."
            ),
        },
        "thresholds": {
            name: getattr(thresholds, name)
            for name in thresholds.__dataclass_fields__
        },
    }
    report["drift_explanations"] = explain_monitoring_report(
        report, thresholds=thresholds
    )
    return report


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable.")


def save_monitoring_report(
    report: dict[str, Any],
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    batch_id = str(report.get("batch_id") or "UNKNOWN")
    safe_batch_id = re.sub(r"[^A-Za-z0-9._-]+", "_", batch_id).strip("._-")
    if not safe_batch_id:
        safe_batch_id = "UNKNOWN"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = reports_dir / f"{safe_batch_id}_{timestamp}.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite monitoring report: {report_path}")
    report_path.write_text(
        json.dumps(report, indent=2, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    alerts_path = (
        MONITORING_DIR / "alerts" / "alerts.jsonl"
        if reports_dir == REPORTS_DIR
        else reports_dir / "alerts.jsonl"
    )
    persist_alerts(
        report,
        alerts_path,
        thresholds=MonitoringThresholds(**report["thresholds"]),
    )
    return report_path


def monitor_batch_file(
    batch_path: Path,
    *,
    reference_path: Path = REFERENCE_DATA_PATH,
    feature_names_path: Path = FEATURE_NAMES_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
    reports_dir: Path = REPORTS_DIR,
    thresholds: MonitoringThresholds = DEFAULT_THRESHOLDS,
    data_source_type: str = "unknown",
    source: str | None = None,
) -> tuple[dict[str, Any], Path]:
    batch = pd.read_csv(batch_path)
    return monitor_batch_data(
        batch,
        reference_path=reference_path,
        feature_names_path=feature_names_path,
        metadata_path=metadata_path,
        reports_dir=reports_dir,
        thresholds=thresholds,
        data_source_type=data_source_type,
        source=source,
    )


def monitor_batch_data(
    batch: pd.DataFrame,
    *,
    reference_path: Path = REFERENCE_DATA_PATH,
    feature_names_path: Path = FEATURE_NAMES_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
    reports_dir: Path = REPORTS_DIR,
    thresholds: MonitoringThresholds = DEFAULT_THRESHOLDS,
    data_source_type: str = "unknown",
    source: str | None = None,
) -> tuple[dict[str, Any], Path]:
    feature_names = json.loads(feature_names_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("feature_names") != feature_names:
        raise ValueError("model_metadata.json feature order does not match feature_names.json.")
    numerical_features = metadata["numerical_features"]
    categorical_features = metadata["categorical_features"]
    if set(numerical_features) | set(categorical_features) != set(feature_names):
        raise ValueError("Metadata feature types do not partition the model input features.")

    reference_data = pd.read_csv(
        reference_path,
        usecols=lambda column: column in set(feature_names),
    ).loc[:, feature_names]
    report = build_monitoring_report(
        batch,
        reference_data,
        feature_names,
        numerical_features,
        categorical_features,
        thresholds=thresholds,
        data_source_type=data_source_type,
        source=source,
    )
    report_path = save_monitoring_report(report, reports_dir)
    return report, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor hotel booking data batches.")
    parser.add_argument("batches", nargs="*", type=Path)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument(
        "--data-source-type",
        choices=("real", "synthetic_test", "unknown"),
        default="unknown",
        help="Declare the batch provenance; unknown is not eligible for Model 2 training.",
    )
    args = parser.parse_args()
    batch_paths = args.batches or sorted(
        path
        for path in MONITORING_DIR.glob("*.csv")
        if not path.name.startswith("report")
    )
    if not batch_paths:
        parser.error("No monitoring batch CSV files were found.")

    results = []
    for batch_path in batch_paths:
        report, report_path = monitor_batch_file(
            batch_path,
            reports_dir=args.reports_dir,
            data_source_type=args.data_source_type,
        )
        results.append(
            {
                "batch_id": report["batch_id"],
                "overall_status": report["overall_status"],
                "report_path": str(report_path),
                "reasons": report["reasons"],
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
