from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ml.monitoring_config import DEFAULT_THRESHOLDS, MonitoringThresholds


def _finite_numeric(values: pd.Series | Sequence[object]) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    return numeric[np.isfinite(numeric)]


def _smoothed_proportions(counts: np.ndarray, epsilon: float) -> np.ndarray:
    if counts.size == 0:
        return counts.astype(float)
    probabilities = counts.astype(float) / counts.sum()
    probabilities = np.maximum(probabilities, epsilon)
    return probabilities / probabilities.sum()


def _psi_bin_counts(
    expected: np.ndarray, actual: np.ndarray, bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quantiles = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges = np.unique(quantiles)
    if edges.size < 2:
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf
    return edges, np.histogram(expected, bins=edges)[0], np.histogram(actual, bins=edges)[0]


def _distribution_summary(
    edges: np.ndarray, counts: np.ndarray, sample_size: int
) -> list[dict[str, Any]]:
    return [
        {
            "bin": (
                f"[{edges[index]:.6g}, {edges[index + 1]:.6g})"
                if np.isfinite(edges[index]) and np.isfinite(edges[index + 1])
                else f"[{edges[index] if np.isfinite(edges[index]) else '-inf'}, "
                f"{edges[index + 1] if np.isfinite(edges[index + 1]) else 'inf'})"
            ),
            "count": int(count),
            "proportion": float(count / sample_size),
        }
        for index, count in enumerate(counts)
    ]


def calculate_psi(
    reference: pd.Series | Sequence[object],
    incoming: pd.Series | Sequence[object],
    *,
    bins: int = DEFAULT_THRESHOLDS.psi_bins,
    epsilon: float = DEFAULT_THRESHOLDS.psi_epsilon,
) -> float:
    if bins < 1:
        raise ValueError("bins must be at least 1.")
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be between 0 and 1.")
    expected = _finite_numeric(reference)
    actual = _finite_numeric(incoming)
    if expected.size == 0 or actual.size == 0:
        raise ValueError("PSI requires at least one finite value in both samples.")

    _, reference_counts, incoming_counts = _psi_bin_counts(expected, actual, bins)
    reference_proportions = _smoothed_proportions(reference_counts, epsilon)
    incoming_proportions = _smoothed_proportions(incoming_counts, epsilon)
    return float(
        np.sum(
            (incoming_proportions - reference_proportions)
            * np.log(incoming_proportions / reference_proportions)
        )
    )


def _psi_level(score: float, thresholds: MonitoringThresholds) -> str:
    if score >= thresholds.psi_critical:
        return "significant"
    if score >= thresholds.psi_warning:
        return "moderate"
    return "low"


def calculate_numerical_drift(
    reference_data: pd.DataFrame,
    incoming_data: pd.DataFrame,
    numerical_features: Sequence[str],
    *,
    thresholds: MonitoringThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for feature in numerical_features:
        if feature not in reference_data or feature not in incoming_data:
            report[feature] = {
                "usable": False,
                "reason": "Feature is missing from reference or incoming data.",
            }
            continue
        reference_values = _finite_numeric(reference_data[feature])
        incoming_values = _finite_numeric(incoming_data[feature])
        if reference_values.size < 2 or incoming_values.size < 2:
            report[feature] = {
                "usable": False,
                "reason": "Fewer than two finite observations in reference or incoming data.",
                "reference_valid_count": int(reference_values.size),
                "incoming_valid_count": int(incoming_values.size),
            }
            continue

        psi = calculate_psi(
            reference_values,
            incoming_values,
            bins=thresholds.psi_bins,
            epsilon=thresholds.psi_epsilon,
        )
        edges, reference_bin_counts, incoming_bin_counts = _psi_bin_counts(
            reference_values, incoming_values, thresholds.psi_bins
        )
        ks_result = ks_2samp(reference_values, incoming_values, method="auto")
        report[feature] = {
            "usable": True,
            "psi": psi,
            "psi_level": _psi_level(psi, thresholds),
            "ks_statistic": float(ks_result.statistic),
            "ks_p_value": float(ks_result.pvalue),
            "ks_statistically_significant": bool(
                ks_result.pvalue < thresholds.ks_significance_level
            ),
            "ks_meaningful_effect": bool(
                ks_result.pvalue < thresholds.ks_significance_level
                and ks_result.statistic >= thresholds.ks_min_effect_size
            ),
            "reference_valid_count": int(reference_values.size),
            "incoming_valid_count": int(incoming_values.size),
            "incoming_excluded_count": int(len(incoming_data) - incoming_values.size),
            "distribution": {
                "method": "reference_quantile_bins",
                "reference": _distribution_summary(
                    edges, reference_bin_counts, len(reference_values)
                ),
                "incoming": _distribution_summary(
                    edges, incoming_bin_counts, len(incoming_values)
                ),
            },
        }
    return report


def _category_metric_group(
    values: pd.Series,
    reference_categories: set[str],
    rare_categories: set[str],
) -> pd.Series:
    observed = values.dropna().astype(str)
    return observed.map(
        lambda value: (
            "__UNSEEN__"
            if value not in reference_categories
            else "__OTHER__"
            if value in rare_categories
            else value
        )
    )


def calculate_categorical_drift(
    reference_data: pd.DataFrame,
    incoming_data: pd.DataFrame,
    categorical_features: Sequence[str],
    *,
    thresholds: MonitoringThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for feature in categorical_features:
        if feature not in reference_data or feature not in incoming_data:
            report[feature] = {
                "usable": False,
                "reason": "Feature is missing from reference or incoming data.",
            }
            continue

        reference_values = reference_data[feature].dropna().astype(str)
        incoming_values = incoming_data[feature].dropna().astype(str)
        if reference_values.empty or incoming_values.empty:
            report[feature] = {
                "usable": False,
                "reason": "No non-missing values in reference or incoming data.",
            }
            continue

        reference_counts = reference_values.value_counts()
        expected_incoming_counts = reference_counts / len(reference_values) * len(
            incoming_values
        )
        rare_categories = set(
            expected_incoming_counts[
                expected_incoming_counts < thresholds.categorical_min_expected_count
            ].index
        )
        reference_categories = set(reference_counts.index)
        reference_metric = _category_metric_group(
            reference_values, reference_categories, rare_categories
        )
        incoming_metric = _category_metric_group(
            incoming_values, reference_categories, rare_categories
        )
        metric_categories = sorted(set(reference_metric) | set(incoming_metric))
        reference_metric_counts = reference_metric.value_counts()
        incoming_metric_counts = incoming_metric.value_counts()
        reference_proportions = _smoothed_proportions(
            np.array([reference_metric_counts.get(cat, 0) for cat in metric_categories]),
            thresholds.psi_epsilon,
        )
        incoming_proportions = _smoothed_proportions(
            np.array([incoming_metric_counts.get(cat, 0) for cat in metric_categories]),
            thresholds.psi_epsilon,
        )
        psi = float(
            np.sum(
                (incoming_proportions - reference_proportions)
                * np.log(incoming_proportions / reference_proportions)
            )
        )

        all_categories = sorted(set(reference_values) | set(incoming_values))
        reference_distribution = reference_values.value_counts(normalize=True)
        incoming_distribution = incoming_values.value_counts(normalize=True)
        differences = {
            category: {
                "reference_proportion": float(reference_distribution.get(category, 0.0)),
                "incoming_proportion": float(incoming_distribution.get(category, 0.0)),
                "difference": float(
                    incoming_distribution.get(category, 0.0)
                    - reference_distribution.get(category, 0.0)
                ),
            }
            for category in all_categories
        }
        unseen_categories = sorted(set(incoming_values) - reference_categories)
        report[feature] = {
            "usable": True,
            "drift_score": psi,
            "drift_method": "categorical_psi_with_rare_reference_categories_pooled",
            "drift_level": _psi_level(psi, thresholds),
            "reference_categories": sorted(reference_categories),
            "incoming_categories": sorted(set(incoming_values)),
            "category_distribution_differences": differences,
            "new_categories": unseen_categories,
            "has_new_categories": bool(unseen_categories),
            "reference_valid_count": int(len(reference_values)),
            "incoming_valid_count": int(len(incoming_values)),
            "pooled_rare_reference_categories": sorted(rare_categories),
        }
    return report
