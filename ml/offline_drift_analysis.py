from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ml.monitoring_config import DEFAULT_THRESHOLDS

ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES_PATH = ROOT / "ml" / "models" / "feature_names.json"
MODEL_METADATA_PATH = ROOT / "ml" / "models" / "model_metadata.json"
HISTORICAL_PATH = ROOT / "data" / "processed" / "hotel_bookings_historical.csv"
FUTURE_PATH = ROOT / "data" / "processed" / "hotel_bookings_future_evaluation.csv"
REPORT_PATH = ROOT / "reports" / "offline_psi_ks_results.md"

_MISSING = ("missing", "")
_OTHER = ("other", "")
_UNSEEN = ("unseen", "")


def _smoothed_proportions(counts: np.ndarray, epsilon: float) -> np.ndarray:
    if counts.size == 0:
        raise ValueError("PSI requires at least one bin.")
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be between 0 and 1.")
    proportions = counts.astype(float) / counts.sum()
    proportions = np.maximum(proportions, epsilon)
    return proportions / proportions.sum()


def _psi(reference_counts: np.ndarray, future_counts: np.ndarray) -> float:
    if reference_counts.shape != future_counts.shape:
        raise ValueError("Reference and future PSI counts must have matching bins.")
    if reference_counts.sum() == 0 or future_counts.sum() == 0:
        raise ValueError("PSI requires observations in both periods.")
    reference = _smoothed_proportions(
        reference_counts, DEFAULT_THRESHOLDS.psi_epsilon
    )
    future = _smoothed_proportions(future_counts, DEFAULT_THRESHOLDS.psi_epsilon)
    return float(np.sum((future - reference) * np.log(future / reference)))


def _finite_numeric(values: pd.Series) -> np.ndarray:
    converted = pd.to_numeric(values, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    return converted[np.isfinite(converted)]


def calculate_numerical_feature_drift(
    reference: pd.Series,
    future: pd.Series,
    *,
    bins: int = DEFAULT_THRESHOLDS.psi_bins,
) -> dict[str, Any]:
    if bins < 1:
        raise ValueError("bins must be at least 1.")
    reference_values = _finite_numeric(reference)
    future_values = _finite_numeric(future)
    reference_missing = len(reference) - len(reference_values)
    future_missing = len(future) - len(future_values)

    discrete_reference = (
        reference_values.size > 0
        and len(np.unique(reference_values)) <= 20
        and np.equal(np.mod(reference_values, 1), 0).all()
    )
    if discrete_reference:
        reference_categories = set(reference_values.tolist())
        reference_counts_by_bucket = {
            value: int(np.count_nonzero(reference_values == value))
            for value in reference_categories
        }
        future_counts_by_bucket: dict[float | str, int] = {}
        for value in future_values:
            bucket: float | str = (
                float(value) if value in reference_categories else "__UNSEEN__"
            )
            future_counts_by_bucket[bucket] = future_counts_by_bucket.get(bucket, 0) + 1
        bins_in_order: list[float | str] = sorted(reference_categories) + [
            "__UNSEEN__"
        ]
        reference_counts = np.array(
            [reference_counts_by_bucket.get(bucket, 0) for bucket in bins_in_order],
            dtype=int,
        )
        future_counts = np.array(
            [future_counts_by_bucket.get(bucket, 0) for bucket in bins_in_order],
            dtype=int,
        )
        binning_method = "reference_discrete_values"
    elif reference_values.size:
        edges = np.unique(
            np.quantile(reference_values, np.linspace(0, 1, bins + 1))
        )
        if edges.size < 2:
            edges = np.array([-np.inf, np.inf])
        else:
            edges[0] = -np.inf
            edges[-1] = np.inf
        reference_counts = np.histogram(reference_values, bins=edges)[0]
        future_counts = np.histogram(future_values, bins=edges)[0]
        binning_method = "reference_quantile_bins"
    else:
        reference_counts = np.array([0], dtype=int)
        future_counts = np.array([len(future_values)], dtype=int)
        binning_method = "reference_empty_valid_values"

    # Missingness is a separate PSI bin rather than being dropped from the
    # denominator, so changes in missing rates contribute to the score.
    reference_counts = np.append(reference_counts, reference_missing)
    future_counts = np.append(future_counts, future_missing)
    psi = _psi(reference_counts, future_counts)

    ks: dict[str, Any]
    if reference_values.size < 2 or future_values.size < 2:
        ks = {
            "statistic": None,
            "p_value": None,
            "available": False,
            "reason": "KS requires at least two finite observations in each period.",
        }
    else:
        result = ks_2samp(reference_values, future_values, method="auto")
        ks = {
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "available": True,
            "reason": None,
        }

    return {
        "psi": psi,
        "reference_missing_count": int(reference_missing),
        "future_missing_count": int(future_missing),
        "reference_missing_percentage": float(reference_missing / len(reference) * 100)
        if len(reference)
        else 0.0,
        "future_missing_percentage": float(future_missing / len(future) * 100)
        if len(future)
        else 0.0,
        "reference_valid_count": int(reference_values.size),
        "future_valid_count": int(future_values.size),
        "ks": ks,
        "psi_bins": int(len(reference_counts)),
        "psi_binning_method": binning_method,
        "psi_missing_bin_included": True,
    }


def _category_key(value: object) -> tuple[str, str]:
    return ("value", str(value))


def _categorical_counts(
    values: pd.Series,
    *,
    reference_categories: set[tuple[str, str]],
    rare_categories: set[tuple[str, str]],
) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for value in values:
        key = _MISSING if pd.isna(value) else _category_key(value)
        if key == _MISSING:
            bucket = _MISSING
        elif key not in reference_categories:
            bucket = _UNSEEN
        elif key in rare_categories:
            bucket = _OTHER
        else:
            bucket = key
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def calculate_categorical_feature_drift(
    reference: pd.Series,
    future: pd.Series,
    *,
    minimum_expected_count: int = DEFAULT_THRESHOLDS.categorical_min_expected_count,
) -> dict[str, Any]:
    if minimum_expected_count < 0:
        raise ValueError("minimum_expected_count must not be negative.")
    reference_values = reference.dropna().map(_category_key)
    future_values = future.dropna().map(_category_key)
    reference_categories = set(reference_values)
    future_categories = set(future_values)

    if reference_values.empty and future_values.empty:
        rare_categories: set[tuple[str, str]] = set()
    else:
        reference_counts = reference_values.value_counts()
        expected_future_counts = (
            reference_counts / len(reference_values) * len(future_values)
            if len(reference_values)
            else pd.Series(dtype=float)
        )
        rare_categories = set(
            expected_future_counts[
                expected_future_counts < minimum_expected_count
            ].index
        )

    reference_counts_by_bucket = _categorical_counts(
        reference,
        reference_categories=reference_categories,
        rare_categories=rare_categories,
    )
    future_counts_by_bucket = _categorical_counts(
        future,
        reference_categories=reference_categories,
        rare_categories=rare_categories,
    )
    buckets = sorted(
        set(reference_counts_by_bucket) | set(future_counts_by_bucket)
    )
    reference_counts_array = np.array(
        [reference_counts_by_bucket.get(bucket, 0) for bucket in buckets],
        dtype=int,
    )
    future_counts_array = np.array(
        [future_counts_by_bucket.get(bucket, 0) for bucket in buckets], dtype=int
    )
    psi = _psi(reference_counts_array, future_counts_array)
    return {
        "psi": psi,
        "reference_missing_count": int(reference.isna().sum()),
        "future_missing_count": int(future.isna().sum()),
        "reference_missing_percentage": float(reference.isna().mean() * 100)
        if len(reference)
        else 0.0,
        "future_missing_percentage": float(future.isna().mean() * 100)
        if len(future)
        else 0.0,
        "reference_valid_count": int(reference.notna().sum()),
        "future_valid_count": int(future.notna().sum()),
        "reference_category_count": int(len(reference_categories)),
        "future_category_count": int(len(future_categories)),
        "future_only_category_count": int(len(future_categories - reference_categories)),
        "rare_reference_category_count": int(len(rare_categories)),
        "psi_bucket_count": int(len(buckets)),
        "psi_missing_bucket_included": True,
    }


def compare_offline_features(
    reference_data: pd.DataFrame,
    future_data: pd.DataFrame,
    feature_names: list[str],
    numerical_features: list[str],
    categorical_features: list[str],
) -> dict[str, dict[str, Any]]:
    if set(numerical_features) | set(categorical_features) != set(feature_names):
        raise ValueError("Numerical and categorical features must partition the feature list.")
    if set(numerical_features) & set(categorical_features):
        raise ValueError("A feature cannot be both numerical and categorical.")
    for frame_name, frame in (
        ("historical", reference_data),
        ("future", future_data),
    ):
        missing = [feature for feature in feature_names if feature not in frame]
        if missing:
            raise ValueError(f"{frame_name} data is missing features: {missing}")

    reference = reference_data.loc[:, feature_names]
    future = future_data.loc[:, feature_names]
    results: dict[str, dict[str, Any]] = {}
    for feature in feature_names:
        if feature in numerical_features:
            metric = calculate_numerical_feature_drift(
                reference[feature], future[feature]
            )
            metric["type"] = "numerical"
        else:
            metric = calculate_categorical_feature_drift(
                reference[feature], future[feature]
            )
            metric["type"] = "categorical"
            metric["ks"] = None
        results[feature] = metric
    return results


def _psi_level(score: float) -> str:
    if score >= DEFAULT_THRESHOLDS.psi_critical:
        return "significant"
    if score >= DEFAULT_THRESHOLDS.psi_warning:
        return "moderate"
    return "low"


def build_markdown_report(
    results: dict[str, dict[str, Any]],
    *,
    historical_rows: int,
    future_rows: int,
    feature_names: list[str],
) -> str:
    lines = [
        "# Offline PSI and KS Comparison",
        "",
        "> **Offline experiment only. These results are not production drift monitoring.**",
        "",
        "## Method",
        "",
        "Historical data is the reference distribution; future evaluation data is "
        "the comparison distribution. Only the 28 Model 1 features, in "
        "`ml/models/feature_names.json` order, were analyzed. Historical "
        "reference quantiles define bins for continuous numerical features; "
        "integer numerical features with at most 20 distinct reference values "
        "use reference value bins. "
        "Categorical PSI uses "
        "reference categories, pools reference categories whose expected future "
        f"count is below {DEFAULT_THRESHOLDS.categorical_min_expected_count} into "
        "an `other` bucket, and groups future-only categories into an `unseen` "
        "bucket. Missing values are retained as an explicit PSI bucket in both "
        "feature types. Zero-frequency bins use epsilon-smoothed proportions "
        f"(epsilon={DEFAULT_THRESHOLDS.psi_epsilon:g}), renormalized to sum to 1.",
        "",
        "KS two-sample tests are calculated for numerical features only, using "
        "finite, non-missing values. KS is unavailable when either period has "
        "fewer than two finite observations. No target/helper columns were "
        "included; no preprocessing was fitted and no model was trained.",
        "",
        f"- Historical reference rows: **{historical_rows:,}**",
        f"- Future comparison rows: **{future_rows:,}**",
        f"- Features analyzed: **{len(feature_names)}**",
        "",
        "## Results",
        "",
        "| Feature | Type | PSI | PSI level* | Missing historical / future | Valid n historical / future | KS statistic | KS p-value | Notes |",
        "|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for feature in feature_names:
        result = results[feature]
        ks = result.get("ks")
        if result["type"] == "numerical":
            ks_stat = f"{ks['statistic']:.6g}" if ks["available"] else "—"
            ks_p = (
                ("<1e-300" if ks["p_value"] == 0 else f"{ks['p_value']:.6g}")
                if ks["available"]
                else "—"
            )
            missing = (
                f"{result['reference_missing_percentage']:.3f}% / "
                f"{result['future_missing_percentage']:.3f}%"
            )
            valid = (
                f"{result['reference_valid_count']:,} / "
                f"{result['future_valid_count']:,}"
            )
            notes = (
                ks["reason"] or "Missing values included as PSI bin."
            )
        else:
            ks_stat = "N/A"
            ks_p = "N/A"
            missing = (
                f"{result['reference_missing_percentage']:.3f}% / "
                f"{result['future_missing_percentage']:.3f}%"
            )
            valid = (
                f"{result['reference_valid_count']:,} / "
                f"{result['future_valid_count']:,}"
            )
            notes = (
                f"{result['future_only_category_count']} future-only categories; "
                f"{result['rare_reference_category_count']} rare reference "
                "categories pooled."
            )
        lines.append(
            f"| `{feature}` | {result['type']} | {result['psi']:.6f} | "
            f"{_psi_level(result['psi'])} | {missing} | {valid} | {ks_stat} | "
            f"{ks_p} | {notes} |"
        )
    lines += [
        "",
        r"\* PSI levels use the project thresholds (low < 0.10; moderate "
        r"0.10–<0.20; significant ≥ 0.20) for interpretation only. They do "
        "not imply production alerts or decisions.",
        "",
        "## Limitations",
        "",
        "- PSI values depend on reference-derived binning and the documented "
        "rare-category pooling rule; they are descriptive and not p-values.",
        "- KS tests omit missing values, test only numerical features, and can "
        "be sensitive to large sample sizes. The p-values do not adjust for "
        "multiple feature comparisons.",
        "- Missingness is included as a distinct PSI bucket and is also shown "
        "separately as percentages.",
        "- The split uses estimated booking dates derived from arrival date and "
        "`lead_time`; these are not recorded booking creation timestamps.",
        "- This historical offline experiment is not evidence of live "
        "production drift or model performance.",
        "",
    ]
    return "\n".join(lines)


def run_offline_analysis() -> Path:
    feature_names = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("feature_names") != feature_names:
        raise ValueError("Model metadata feature order does not match feature_names.json.")
    forbidden = {
        "is_canceled",
        "reservation_status",
        "reservation_status_date",
        "offline_row_id",
        "estimated_booking_date",
    }
    if forbidden.intersection(feature_names):
        raise ValueError("Feature schema contains forbidden target or helper columns.")
    numerical_features = metadata["numerical_features"]
    categorical_features = metadata["categorical_features"]
    historical = pd.read_csv(HISTORICAL_PATH, usecols=feature_names).loc[
        :, feature_names
    ]
    future = pd.read_csv(FUTURE_PATH, usecols=feature_names).loc[:, feature_names]
    results = compare_offline_features(
        historical,
        future,
        feature_names,
        numerical_features,
        categorical_features,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        build_markdown_report(
            results,
            historical_rows=len(historical),
            future_rows=len(future),
            feature_names=feature_names,
        ),
        encoding="utf-8",
    )
    return REPORT_PATH


if __name__ == "__main__":
    print(run_offline_analysis())
