from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ml.monitoring_config import (
    DEFAULT_THRESHOLDS,
    NON_NEGATIVE_FEATURES,
    MonitoringThresholds,
)

MONITORING_METADATA_COLUMNS = (
    "batch_id",
    "generated_at",
    "timestamp",
    "source",
    "data_source_type",
    "record_count",
    "monitoring_scenario",
    "label_provenance",
    "row_id",
    "source_booking_id_hmac",
)
REQUIRED_MONITORING_METADATA_COLUMNS = ("batch_id", "generated_at")


def _batch_identifier(batch: pd.DataFrame) -> str:
    if "batch_id" not in batch.columns or batch.empty:
        return "UNKNOWN"
    identifiers = batch["batch_id"].dropna().astype(str).unique().tolist()
    if len(identifiers) == 1:
        return identifiers[0]
    return "UNKNOWN" if not identifiers else ",".join(identifiers)


def _quality_finding(
    feature_name: str,
    check_name: str,
    affected_row_count: int,
    severity: str,
    message: str,
) -> dict[str, Any]:
    return {
        "feature_name": feature_name,
        "check_name": check_name,
        "affected_row_count": int(affected_row_count),
        "severity": severity,
        "message": message,
    }


def assess_data_quality(
    batch: pd.DataFrame,
    reference_data: pd.DataFrame,
    feature_names: Sequence[str],
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
    *,
    thresholds: MonitoringThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    if not isinstance(batch, pd.DataFrame) or not isinstance(reference_data, pd.DataFrame):
        raise TypeError("Batch and reference data must be pandas DataFrames.")

    expected_features = list(feature_names)
    batch_features = [name for name in batch.columns if name in expected_features]
    missing_features = [name for name in expected_features if name not in batch.columns]
    unexpected_features = [
        name
        for name in batch.columns
        if name not in expected_features
        and name not in MONITORING_METADATA_COLUMNS
        and name != "is_canceled"
    ]
    feature_order_matches = batch_features == expected_features
    row_count = len(batch)
    missing_counts = batch.isna().sum().astype(int).to_dict()
    missing_percentages = (
        (batch.isna().mean() * 100).round(6).to_dict() if row_count else {}
    )
    reference_missing_percentages = (
        (reference_data.reindex(columns=expected_features).isna().mean()).to_dict()
    )

    missing_excess: dict[str, float] = {}
    for column in expected_features:
        incoming_rate = float(batch[column].isna().mean()) if column in batch else 1.0
        baseline_rate = float(reference_missing_percentages.get(column, 0.0))
        excess = max(0.0, incoming_rate - baseline_rate)
        if excess > 0:
            missing_excess[column] = round(excess * 100, 6)

    duplicate_count = (
        int(batch.loc[:, batch_features].duplicated().sum()) if batch_features else 0
    )
    duplicate_rate = duplicate_count / row_count if row_count else 0.0
    reference_duplicate_rate = (
        float(reference_data.loc[:, expected_features].duplicated().mean())
        if expected_features and all(name in reference_data for name in expected_features)
        else 0.0
    )
    duplicate_rate_excess = max(0.0, duplicate_rate - reference_duplicate_rate)

    invalid_numeric_counts: dict[str, int] = {}
    incorrect_data_types: dict[str, dict[str, Any]] = {}
    infinite_counts: dict[str, int] = {}
    negative_counts: dict[str, int] = {}
    for column in numerical_features:
        if column not in batch.columns:
            continue
        source = batch[column]
        converted = pd.to_numeric(source, errors="coerce")
        invalid_mask = source.notna() & converted.isna()
        invalid_numeric_counts[column] = int(invalid_mask.sum())
        if invalid_mask.any():
            incorrect_data_types[column] = {
                "expected": "numeric",
                "actual": str(source.dtype),
                "invalid_value_count": int(invalid_mask.sum()),
            }
        elif not pd.api.types.is_numeric_dtype(source.dtype) and source.notna().any():
            incorrect_data_types[column] = {
                "expected": "numeric",
                "actual": str(source.dtype),
                "invalid_value_count": 0,
            }

        finite_mask = np.isfinite(converted.astype(float))
        infinite_counts[column] = int((converted.notna() & ~finite_mask).sum())
        if column in NON_NEGATIVE_FEATURES:
            negative_counts[column] = int((converted.notna() & (converted < 0)).sum())

    reference_categories = {
        column: set(reference_data[column].dropna().astype(str).unique())
        for column in categorical_features
        if column in reference_data.columns
    }
    unexpected_category_counts: dict[str, int] = {}
    unexpected_category_values: dict[str, list[str]] = {}
    for column in categorical_features:
        if column not in batch.columns:
            continue
        values = batch[column]
        if values.notna().any() and not (
            pd.api.types.is_object_dtype(values.dtype)
            or pd.api.types.is_string_dtype(values.dtype)
            or isinstance(values.dtype, pd.CategoricalDtype)
        ):
            incorrect_data_types[column] = {
                "expected": "categorical/string",
                "actual": str(values.dtype),
                "invalid_value_count": int(values.notna().sum()),
            }
        unexpected = values.notna() & ~values.astype(str).isin(
            reference_categories.get(column, set())
        )
        unexpected_category_counts[column] = int(unexpected.sum())
        if unexpected.any():
            unexpected_category_values[column] = sorted(
                values.loc[unexpected].astype(str).unique().tolist()
            )

    missing_metadata = [
        column
        for column in REQUIRED_MONITORING_METADATA_COLUMNS
        if column not in batch.columns or batch[column].isna().any()
    ]
    inconsistent_metadata = {
        column: int(batch[column].nunique(dropna=True))
        for column in MONITORING_METADATA_COLUMNS
        if column in batch.columns and batch[column].nunique(dropna=True) > 1
    }

    critical_reasons: list[str] = []
    warning_reasons: list[str] = []
    if row_count == 0:
        critical_reasons.append("Batch must contain at least one row.")
    if missing_features:
        critical_reasons.append(f"Missing required model features: {missing_features}")
    if unexpected_features:
        critical_reasons.append(f"Unexpected input columns: {unexpected_features}")
    if unexpected_category_values:
        critical_reasons.append(
            f"Unexpected categorical values: {unexpected_category_values}"
        )
    invalid_numeric_total = sum(invalid_numeric_counts.values())
    if invalid_numeric_total:
        critical_reasons.append(
            f"{invalid_numeric_total} non-numeric values found in numerical features"
        )
    infinite_total = sum(infinite_counts.values())
    if infinite_total:
        critical_reasons.append(f"{infinite_total} infinite numerical values found")
    invalid_negative_total = sum(negative_counts.values())
    if invalid_negative_total:
        critical_reasons.append(
            f"{invalid_negative_total} negative values found in non-negative features"
        )
    if incorrect_data_types:
        critical_reasons.append(
            f"Incorrect feature data types: {incorrect_data_types}"
        )

    for column, excess_percent in missing_excess.items():
        if excess_percent >= thresholds.missing_rate_critical_delta * 100:
            critical_reasons.append(
                f"{column} missing rate exceeds reference by {excess_percent:.2f} percentage points"
            )
        elif excess_percent >= thresholds.missing_rate_warning_delta * 100:
            warning_reasons.append(
                f"{column} missing rate exceeds reference by {excess_percent:.2f} percentage points"
            )

    if duplicate_rate_excess >= thresholds.duplicate_critical_rate:
        critical_reasons.append(
            "Duplicate feature rate exceeds reference by "
            f"{duplicate_rate_excess:.1%} ({duplicate_count} incoming duplicates)"
        )
    elif duplicate_rate_excess >= thresholds.duplicate_warning_rate:
        warning_reasons.append(
            "Duplicate feature rate exceeds reference by "
            f"{duplicate_rate_excess:.1%} ({duplicate_count} incoming duplicates)"
        )
    if not feature_order_matches and not missing_features:
        warning_reasons.append("Input model features are not in the expected order.")
    if missing_metadata:
        warning_reasons.append(f"Missing or null monitoring metadata: {missing_metadata}")
    if inconsistent_metadata:
        warning_reasons.append(
            f"Multiple metadata values found within one batch: {inconsistent_metadata}"
        )

    findings: list[dict[str, Any]] = []
    if row_count == 0:
        findings.append(
            _quality_finding(
                "__batch__",
                "empty_batch",
                0,
                "ERROR",
                "Batch contains no records.",
            )
        )
    for feature in missing_features:
        findings.append(
            _quality_finding(
                feature,
                "missing_feature",
                row_count,
                "ERROR",
                "Required model feature is absent from the batch.",
            )
        )
    for feature in unexpected_features:
        findings.append(
            _quality_finding(
                feature,
                "unexpected_feature",
                row_count,
                "ERROR",
                "Input column is not part of the model schema or allowed metadata.",
            )
        )
    if not feature_order_matches and not missing_features:
        findings.append(
            _quality_finding(
                "__schema__",
                "feature_order",
                row_count,
                "WARNING",
                "Model features are present but not in the expected feature order.",
            )
        )

    for column in expected_features:
        if column not in batch:
            continue
        affected = int(missing_counts.get(column, 0))
        if affected == 0:
            continue
        excess_percent = missing_excess.get(column, 0.0)
        if excess_percent >= thresholds.missing_rate_critical_delta * 100:
            severity = "ERROR"
        elif excess_percent >= thresholds.missing_rate_warning_delta * 100:
            severity = "WARNING"
        else:
            severity = "INFO"
        findings.append(
            _quality_finding(
                column,
                "missing_values",
                affected,
                severity,
                f"{affected} row(s) have missing values; missingness exceeds "
                f"the reference by {excess_percent:.2f} percentage points.",
            )
        )

    for column, details in incorrect_data_types.items():
        findings.append(
            _quality_finding(
                column,
                "data_type",
                int(details["invalid_value_count"])
                or int(batch[column].notna().sum()),
                "ERROR",
                f"Expected {details['expected']} values, received dtype "
                f"{details['actual']}.",
            )
        )
    for column, count in invalid_numeric_counts.items():
        if count:
            findings.append(
                _quality_finding(
                    column,
                    "invalid_numeric_values",
                    count,
                    "ERROR",
                    "Values could not be converted to finite numeric values.",
                )
            )
    for column, count in infinite_counts.items():
        if count:
            findings.append(
                _quality_finding(
                    column,
                    "infinite_numeric_values",
                    count,
                    "ERROR",
                    "Values include positive or negative infinity.",
                )
            )
    for column, count in negative_counts.items():
        if count:
            findings.append(
                _quality_finding(
                    column,
                    "negative_values",
                    count,
                    "ERROR",
                    "Negative values violate the configured non-negative constraint.",
                )
            )
    for column, count in unexpected_category_counts.items():
        if count:
            findings.append(
                _quality_finding(
                    column,
                    "unexpected_categories",
                    count,
                    "ERROR",
                    f"{count} row(s) contain categories absent from the reference data.",
                )
            )

    if duplicate_count:
        if duplicate_rate_excess >= thresholds.duplicate_critical_rate:
            duplicate_severity = "ERROR"
        elif duplicate_rate_excess >= thresholds.duplicate_warning_rate:
            duplicate_severity = "WARNING"
        else:
            duplicate_severity = "INFO"
        findings.append(
            _quality_finding(
                "__batch__",
                "duplicate_records",
                duplicate_count,
                duplicate_severity,
                "Repeated model-feature rows were detected; records were not removed.",
            )
        )

    for column in missing_metadata:
        affected = (
            row_count
            if column not in batch
            else int(batch[column].isna().sum())
        )
        findings.append(
            _quality_finding(
                column,
                "missing_batch_metadata",
                affected,
                "WARNING",
                "Required monitoring metadata is missing or null.",
            )
        )
    for column, distinct_count in inconsistent_metadata.items():
        values = batch[column]
        mode = values.dropna().mode()
        common_value = mode.iloc[0] if not mode.empty else None
        affected = int((values.notna() & values.ne(common_value)).sum())
        findings.append(
            _quality_finding(
                column,
                "inconsistent_batch_metadata",
                affected,
                "WARNING",
                f"Batch metadata has {distinct_count} distinct non-null values.",
            )
        )

    severity_counts = {
        severity: sum(finding["severity"] == severity for finding in findings)
        for severity in ("ERROR", "WARNING", "INFO")
    }
    status = (
        "CRITICAL"
        if critical_reasons
        else "WARNING"
        if warning_reasons
        else "HEALTHY"
    )
    return {
        "batch_id": _batch_identifier(batch),
        "total_rows": row_count,
        "total_columns": len(batch.columns),
        "missing_value_counts": {name: int(count) for name, count in missing_counts.items()},
        "missing_value_percentages": {
            name: float(percent) for name, percent in missing_percentages.items()
        },
        "reference_missing_percentages": {
            name: round(float(rate) * 100, 6)
            for name, rate in reference_missing_percentages.items()
        },
        "missing_excess_percentage_points": missing_excess,
        "duplicate_count": duplicate_count,
        "duplicate_rate": duplicate_rate,
        "reference_duplicate_rate": reference_duplicate_rate,
        "duplicate_rate_excess": duplicate_rate_excess,
        "invalid_numerical_value_counts": invalid_numeric_counts,
        "unexpected_category_counts": unexpected_category_counts,
        "unexpected_category_values": unexpected_category_values,
        "incorrect_data_types": incorrect_data_types,
        "infinite_value_counts": infinite_counts,
        "negative_value_counts": negative_counts,
        "feature_schema": {
            "expected_features": expected_features,
            "missing_features": missing_features,
            "unexpected_features": unexpected_features,
            "feature_order_matches": feature_order_matches,
            "metadata_columns": [
                column for column in MONITORING_METADATA_COLUMNS if column in batch
            ],
        },
        "monitoring_metadata": {
            "missing_or_null_columns": missing_metadata,
            "inconsistent_value_counts": inconsistent_metadata,
            "generated_at": (
                str(batch["generated_at"].dropna().iloc[0])
                if "generated_at" in batch and batch["generated_at"].notna().any()
                else None
            ),
        },
        "quality_status": status,
        "critical_reasons": critical_reasons,
        "warning_reasons": warning_reasons,
        "findings": findings,
        "severity_counts": severity_counts,
    }
