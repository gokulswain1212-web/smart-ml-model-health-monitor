from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml.data_quality import assess_data_quality

ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES_PATH = ROOT / "ml" / "models" / "feature_names.json"
MODEL_METADATA_PATH = ROOT / "ml" / "models" / "model_metadata.json"
MODEL_PIPELINE_PATH = ROOT / "ml" / "models" / "hotel_cancellation_pipeline.pkl"
HISTORICAL_PATH = ROOT / "data" / "processed" / "hotel_bookings_historical.csv"
FUTURE_PATH = ROOT / "data" / "processed" / "hotel_bookings_future_evaluation.csv"
REPORT_PATH = ROOT / "reports" / "offline_data_quality_report.md"
TARGET_AND_POST_OUTCOME_COLUMNS = {
    "is_canceled",
    "reservation_status",
    "reservation_status_date",
}
OFFLINE_HELPER_COLUMNS = {"offline_row_id", "estimated_booking_date"}

# These are the explicit source-value mappings used by HotelCancellationPreprocessor.
MAPPED_CATEGORIES = {
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


def _saved_pipeline_categories(
    pipeline_path: Path,
) -> dict[str, set[str]]:
    pipeline = joblib.load(pipeline_path)
    preprocessor = dict(pipeline.steps).get("preprocessor")
    if preprocessor is None or not hasattr(preprocessor, "encoder"):
        raise ValueError("Saved Model 1 pipeline has no compatible preprocessor encoder.")

    encoder = preprocessor.encoder
    encoder_categories = {
        column: set(map(str, categories.tolist()))
        for column, categories in zip(
            preprocessor.categorical_columns,
            encoder.categories_,
            strict=True,
        )
    }
    supported: dict[str, set[str]] = {}
    for column, mapping in MAPPED_CATEGORIES.items():
        if column not in encoder_categories:
            raise ValueError(f"Saved encoder does not include mapped feature {column!r}.")
        mapped_outputs = set(map(str, mapping.values()))
        if not mapped_outputs.issubset(encoder_categories[column]):
            raise ValueError(
                f"Saved encoder categories do not match preprocessing mapping for {column!r}."
            )
        supported[column] = set(mapping)

    for column, categories in encoder_categories.items():
        if column not in supported:
            supported[column] = categories
    return supported


def _model_feature_frame(
    data: pd.DataFrame, feature_names: Sequence[str]
) -> pd.DataFrame:
    return data.loc[:, [column for column in data.columns if column in feature_names]].copy()


def _assess_dataset(
    data: pd.DataFrame,
    reference_data: pd.DataFrame,
    feature_names: list[str],
    numerical_features: list[str],
    categorical_features: list[str],
) -> dict[str, Any]:
    return assess_data_quality(
        _model_feature_frame(data, feature_names),
        _model_feature_frame(reference_data, feature_names),
        feature_names,
        numerical_features,
        categorical_features,
    )


def _category_comparison(
    historical: pd.DataFrame,
    future: pd.DataFrame,
    categorical_features: Sequence[str],
    supported_categories: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    results = []
    for feature in categorical_features:
        historical_values = (
            set(historical[feature].dropna().astype(str))
            if feature in historical
            else set()
        )
        future_values = (
            set(future[feature].dropna().astype(str))
            if feature in future
            else set()
        )
        pipeline_values = supported_categories.get(feature, set())
        results.append(
            {
                "feature": feature,
                "historical_observed_count": len(historical_values),
                "pipeline_supported_count": len(pipeline_values),
                "historical_not_pipeline_supported": sorted(
                    historical_values - pipeline_values
                ),
                "future_only_vs_historical": sorted(future_values - historical_values),
                "future_not_pipeline_supported": sorted(
                    future_values - pipeline_values
                ),
            }
        )
    return results


def _findings_table(quality: dict[str, Any]) -> list[str]:
    findings = quality["findings"]
    if not findings:
        return ["| — | — | 0 | — | No findings. |"]
    return [
        "| "
        + " | ".join(
            (
                f"`{finding['feature_name']}`",
                f"`{finding['check_name']}`",
                str(finding["affected_row_count"]),
                finding["severity"],
                finding["message"].replace("|", "\\|").replace("\n", " "),
            )
        )
        + " |"
        for finding in findings
    ]


def _format_categories(values: Sequence[str]) -> str:
    return ", ".join(f"`{value}`" for value in values) if values else "None"


def build_offline_quality_report(
    historical: pd.DataFrame,
    future: pd.DataFrame,
    feature_names: list[str],
    numerical_features: list[str],
    categorical_features: list[str],
    supported_categories: Mapping[str, set[str]],
) -> str:
    historical_quality = _assess_dataset(
        historical,
        historical,
        feature_names,
        numerical_features,
        categorical_features,
    )
    future_quality = _assess_dataset(
        future,
        historical,
        feature_names,
        numerical_features,
        categorical_features,
    )

    dataset_results = [
        ("Historical", historical, historical_quality),
        ("Future evaluation", future, future_quality),
    ]
    lines = [
        "# Offline Data Quality Report",
        "",
        "> **Offline historical experiment only. This report is not production "
        "monitoring and does not assess model health or prediction quality.**",
        "",
        "## Scope and method",
        "",
        "Both source CSVs were read without modification. The existing "
        "`ml.data_quality.assess_data_quality` checker was used for each "
        "dataset. The historical dataset is its own reference for historical "
        "checks and is the reference for the future evaluation checks. Only "
        "the 28 Model 1 input features, in `feature_names.json` order, are "
        "passed to the checker.",
        "",
        "The CSVs also contain non-model columns. `is_canceled` is an offline "
        "evaluation target and was not supplied to the checker; "
        "`reservation_status` and `reservation_status_date` are post-outcome "
        "fields; `offline_row_id` and `estimated_booking_date` are offline "
        "helpers; `company` is not in the Model 1 feature schema. They are "
        "reported as dataset columns, not treated as model input features.",
        "",
        "The checker requires monitoring metadata (`batch_id` and "
        "`generated_at`). Static offline datasets do not contain those fields, "
        "so their absence appears as WARNING findings and affects the checker "
        "status. No synthetic timestamp or production metadata was added.",
        "",
        "## Dataset summaries",
        "",
        "| Dataset | Rows | CSV columns | Model features present | Missing features | Non-model columns | Checker status | ERROR / WARNING / INFO findings |",
        "|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for label, data, quality in dataset_results:
        present_features = [
            column for column in data.columns if column in feature_names
        ]
        missing_features = quality["feature_schema"]["missing_features"]
        non_model_columns = [column for column in data.columns if column not in feature_names]
        counts = quality["severity_counts"]
        lines.append(
            f"| {label} | {len(data):,} | {len(data.columns)} | "
            f"{len(present_features)}/28 | "
            f"{', '.join(f'`{name}`' for name in missing_features) or 'None'} | "
            f"{', '.join(f'`{name}`' for name in non_model_columns) or 'None'} | "
            f"**{quality['quality_status']}** | "
            f"{counts['ERROR']} / {counts['WARNING']} / {counts['INFO']} |"
        )

    lines.extend(["", "## Feature schema and data types", ""])
    for label, data, quality in dataset_results:
        actual_order = [
            column for column in data.columns if column in feature_names
        ]
        unexpected = quality["feature_schema"]["unexpected_features"]
        lines.extend(
            [
                f"### {label}",
                "",
                f"- Model feature order matches expected order: "
                f"**{quality['feature_schema']['feature_order_matches']}**.",
                f"- Missing Model 1 features: "
                f"{quality['feature_schema']['missing_features'] or 'None'}.",
                f"- Unexpected columns supplied to the checker: "
                f"{unexpected or 'None'} (non-model CSV columns listed above "
                "were excluded intentionally).",
                f"- Model feature order in CSV: "
                f"{', '.join(f'`{name}`' for name in actual_order)}.",
                "",
                "| Feature | Expected type | CSV dtype | Missing rows | Missing % |",
                "|---|---|---|---:|---:|",
            ]
        )
        for feature in feature_names:
            if feature in data:
                missing_count = int(data[feature].isna().sum())
                missing_percent = (
                    missing_count / len(data) * 100 if len(data) else 0.0
                )
                dtype = str(data[feature].dtype)
            else:
                missing_count = len(data)
                missing_percent = 100.0 if len(data) else 0.0
                dtype = "absent"
            expected_type = (
                "numerical" if feature in numerical_features else "categorical/string"
            )
            lines.append(
                f"| `{feature}` | {expected_type} | `{dtype}` | "
                f"{missing_count:,} | {missing_percent:.3f}% |"
            )
        lines.append("")

    lines.extend(
        [
            "## Numeric, category, duplicate, and metadata checks",
            "",
        ]
    )
    for label, data, quality in dataset_results:
        lines.extend(
            [
                f"### {label}",
                "",
                f"- Invalid numeric values: "
                f"`{quality['invalid_numerical_value_counts']}`.",
                f"- Infinite numeric values: `{quality['infinite_value_counts']}`.",
                f"- Configured negative-value violations: "
                f"`{quality['negative_value_counts']}`.",
                f"- Unexpected categorical values relative to the historical "
                f"observed data: `{quality['unexpected_category_counts']}`.",
                f"- Duplicate model-feature rows: "
                f"**{quality['duplicate_count']:,}** / **{quality['total_rows']:,}** "
                f"({quality['duplicate_rate']:.3%}); historical reference rate "
                f"{quality['reference_duplicate_rate']:.3%}; excess "
                f"{quality['duplicate_rate_excess']:.3%}.",
                f"- Monitoring metadata missing/null: "
                f"`{quality['monitoring_metadata']['missing_or_null_columns']}`.",
                f"- Inconsistent metadata values: "
                f"`{quality['monitoring_metadata']['inconsistent_value_counts']}`.",
                "",
                "| Feature | Check | Affected rows | Severity | Explanation |",
                "|---|---|---:|---|---|",
                *_findings_table(quality),
                "",
                f"Severity totals: **{quality['severity_counts']}**. "
                f"Existing checker status: **{quality['quality_status']}**.",
                "",
            ]
        )

    lines.extend(
        [
            "## Historical observations versus saved Model 1 category support",
            "",
            "Historical observed categories are measured directly from the "
            "historical CSV. Saved-pipeline support is derived from the "
            "serialized Model 1 preprocessor: explicit mappings for `hotel`, "
            "`arrival_date_month`, and `meal`, and fitted encoder categories "
            "for the other categorical features. These are different concepts. "
            "An absent-from-historical category can still be supported by the "
            "saved encoder; conversely, an historically observed category may "
            "not be accepted by that artifact. Missing values are excluded "
            "from category counts.",
            "",
            "| Feature | Historical observed category count | Saved pipeline supported count | Historical values not supported | Future-only vs historical | Future values not supported |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    for row in _category_comparison(
        historical, future, categorical_features, supported_categories
    ):
        lines.append(
            f"| `{row['feature']}` | {row['historical_observed_count']} | "
            f"{row['pipeline_supported_count']} | "
            f"{_format_categories(row['historical_not_pipeline_supported'])} | "
            f"{_format_categories(row['future_only_vs_historical'])} | "
            f"{_format_categories(row['future_not_pipeline_supported'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "- The checker status is its existing data-quality status, not a "
            "Model 1 readiness or overall application health status.",
            "- Category findings compare future values with categories "
            "observed historically. The separate pipeline-support table "
            "reflects the saved encoder/mappings; it does not change checker "
            "findings or assert that all unseen-but-supported categories are "
            "operationally expected.",
            "- Duplicate detection compares exact Model 1 feature rows and "
            "does not remove records. Repeated hotel bookings may be valid "
            "observations, so duplicates require provenance review.",
            "- Numeric constraints are limited to the configured "
            "`NON_NEGATIVE_FEATURES`; no observed minimum/maximum was promoted "
            "to a validation rule.",
            "- These are historical offline partitions created using "
            "`estimated_booking_date`, not genuine production observations. "
            "The historical partition is also the future comparison's "
            "baseline, so category and duplicate-rate comparisons inherit "
            "that single reference choice.",
            "- `company` and post-outcome columns are not model inputs. "
            "Their missingness and values are not evaluated as Model 1 feature "
            "quality.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_offline_data_quality_report(
    *,
    historical_path: Path = HISTORICAL_PATH,
    future_path: Path = FUTURE_PATH,
    feature_names_path: Path = FEATURE_NAMES_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
    pipeline_path: Path = MODEL_PIPELINE_PATH,
    report_path: Path = REPORT_PATH,
) -> Path:
    feature_names = json.loads(feature_names_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("feature_names") != feature_names:
        raise ValueError("Model metadata feature order does not match feature_names.json.")
    numerical_features = metadata["numerical_features"]
    categorical_features = metadata["categorical_features"]
    if set(numerical_features) | set(categorical_features) != set(feature_names):
        raise ValueError("Metadata feature types do not partition model features.")
    if set(numerical_features) & set(categorical_features):
        raise ValueError("A model feature cannot be both numerical and categorical.")

    historical = pd.read_csv(historical_path)
    future = pd.read_csv(future_path)
    supported_categories = _saved_pipeline_categories(pipeline_path)
    report = build_offline_quality_report(
        historical,
        future,
        feature_names,
        numerical_features,
        categorical_features,
        supported_categories,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return report_path


if __name__ == "__main__":
    print(generate_offline_data_quality_report())
