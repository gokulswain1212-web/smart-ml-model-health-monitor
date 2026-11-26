import numpy as np
import pandas as pd
import pytest

from ml.offline_drift_analysis import (
    calculate_categorical_feature_drift,
    calculate_numerical_feature_drift,
    compare_offline_features,
)


def test_numerical_feature_drift_reports_psi_and_ks_for_shifted_values():
    reference = pd.Series(np.arange(100, dtype=float))
    future = pd.Series(np.arange(100, dtype=float) + 40)

    result = calculate_numerical_feature_drift(reference, future)

    assert result["psi"] > 0
    assert result["ks"]["available"] is True
    assert result["ks"]["statistic"] > 0
    assert 0 <= result["ks"]["p_value"] <= 1
    assert result["reference_missing_count"] == 0
    assert result["future_missing_count"] == 0


def test_low_cardinality_numeric_psi_uses_reference_value_bins():
    reference = pd.Series([0] * 90 + [1] * 10)
    future = pd.Series([0] * 10 + [1] * 90)

    result = calculate_numerical_feature_drift(reference, future)

    assert result["psi_binning_method"] == "reference_discrete_values"
    assert result["psi"] > 0.5
    assert result["ks"]["available"] is True


def test_numerical_psi_safely_smooths_zero_frequency_bins():
    reference = pd.Series([0] * 100)
    future = pd.Series([1] * 100)

    result = calculate_numerical_feature_drift(reference, future)

    assert np.isfinite(result["psi"])
    assert result["psi"] > 0


def test_numerical_missing_values_are_explicit_in_psi_and_ks():
    reference = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan, np.inf])
    future = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])

    result = calculate_numerical_feature_drift(reference, future)

    assert result["psi_missing_bin_included"] is True
    assert result["reference_missing_count"] == 2
    assert result["future_missing_count"] == 1
    assert result["reference_missing_percentage"] == pytest.approx(2 / 6 * 100)
    assert result["future_missing_percentage"] == pytest.approx(1 / 6 * 100)
    assert result["reference_valid_count"] == 4
    assert result["future_valid_count"] == 5
    assert result["ks"]["available"] is True


def test_categorical_psi_handles_zero_frequency_and_missing_buckets():
    reference = pd.Series(["known"] * 100 + [None])
    future = pd.Series(["unseen"] * 100 + [None])

    result = calculate_categorical_feature_drift(
        reference, future, minimum_expected_count=0
    )

    assert np.isfinite(result["psi"])
    assert result["psi"] > 0
    assert result["psi_missing_bucket_included"] is True
    assert result["future_only_category_count"] == 1
    assert result["reference_missing_count"] == 1
    assert result["future_missing_count"] == 1


def test_categorical_psi_pools_rare_reference_categories():
    reference = pd.Series(["common"] * 100 + ["rare"] * 2)
    future = pd.Series(["common"] * 100 + ["new"] * 2)

    result = calculate_categorical_feature_drift(reference, future)

    assert result["rare_reference_category_count"] == 1
    assert result["future_only_category_count"] == 1
    assert np.isfinite(result["psi"])


def test_compare_features_preserves_requested_order_and_only_numerical_ks():
    reference = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0], "category": ["A", "A", "B"]}
    )
    future = pd.DataFrame(
        {"value": [2.0, 3.0, 4.0], "category": ["A", "B", "B"]}
    )

    results = compare_offline_features(
        reference,
        future,
        ["category", "value"],
        ["value"],
        ["category"],
    )

    assert list(results) == ["category", "value"]
    assert results["category"]["ks"] is None
    assert results["value"]["ks"]["available"] is True


def test_numerical_ks_is_unavailable_for_too_few_valid_values():
    result = calculate_numerical_feature_drift(
        pd.Series([1.0, np.nan]), pd.Series([2.0, np.nan])
    )

    assert result["ks"]["available"] is False
    assert result["ks"]["statistic"] is None
    assert result["ks"]["p_value"] is None
