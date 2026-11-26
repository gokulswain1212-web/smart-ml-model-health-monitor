from datetime import date

import numpy as np
import pandas as pd
import pytest

from ml.offline_multiwindow_drift import (
    ComparisonWindow,
    analyze_multiwindow_drift,
)


def _frames():
    historical_dates = pd.date_range("2020-01-01", periods=8, freq="D")
    future_dates = pd.date_range("2020-01-09", periods=8, freq="D")
    historical = pd.DataFrame(
        {
            "estimated_booking_date": historical_dates,
            "number": [0, 1, 2, 3, 4, 5, np.nan, 7],
            "category": ["A", "A", "A", "B", "B", "B", None, "A"],
        }
    )
    future = pd.DataFrame(
        {
            "estimated_booking_date": future_dates,
            "number": [1, 2, 3, 4, 5, 6, 7, np.nan],
            "category": ["A", "B", "B", "NEW", "NEW", "A", None, "NEW"],
        }
    )
    return historical, future


def _analyze(historical, future, windows):
    return analyze_multiwindow_drift(
        historical,
        future,
        ["number", "category"],
        ["number"],
        ["category"],
        windows=windows,
    )


def test_windows_are_reported_in_chronological_order_with_inclusive_boundaries():
    historical, future = _frames()
    windows = [
        ComparisonWindow("first", date(2020, 1, 9), date(2020, 1, 12)),
        ComparisonWindow("second", date(2020, 1, 13), date(2020, 1, 16)),
    ]

    result = _analyze(historical, future, windows)

    assert [window["name"] for window in result["windows"]] == ["first", "second"]
    assert [window["future_rows"] for window in result["windows"]] == [4, 4]
    assert result["windows"][0]["start_date"] == "2020-01-09"
    assert result["windows"][1]["end_date"] == "2020-01-16"


@pytest.mark.parametrize(
    "windows",
    [
        [ComparisonWindow("reversed", date(2020, 1, 12), date(2020, 1, 9))],
        [
            ComparisonWindow("first", date(2020, 1, 9), date(2020, 1, 12)),
            ComparisonWindow("overlap", date(2020, 1, 12), date(2020, 1, 16)),
        ],
        [
            ComparisonWindow("later", date(2020, 1, 13), date(2020, 1, 16)),
            ComparisonWindow("earlier", date(2020, 1, 9), date(2020, 1, 12)),
        ],
    ],
)
def test_invalid_window_boundaries_are_rejected(windows):
    historical, future = _frames()

    with pytest.raises(ValueError):
        _analyze(historical, future, windows)


def test_unsorted_input_dates_are_rejected():
    historical, future = _frames()
    future = future.iloc[::-1].reset_index(drop=True)

    with pytest.raises(ValueError, match="chronological order"):
        _analyze(
            historical,
            future,
            [ComparisonWindow("all", date(2020, 1, 9), date(2020, 1, 16))],
        )


def test_missing_values_are_included_and_numeric_ks_uses_valid_values():
    historical, future = _frames()
    result = _analyze(
        historical,
        future,
        [ComparisonWindow("all", date(2020, 1, 9), date(2020, 1, 16))],
    )
    metric = result["windows"][0]["features"]["number"]

    assert metric["psi_missing_bin_included"] is True
    assert metric["reference_missing_count"] == 1
    assert metric["future_missing_count"] == 1
    assert metric["ks"]["available"] is True
    assert metric["reference_valid_count"] == 7
    assert metric["future_valid_count"] == 7


def test_unseen_categories_are_counted_and_grouped():
    historical, future = _frames()
    result = _analyze(
        historical,
        future,
        [ComparisonWindow("all", date(2020, 1, 9), date(2020, 1, 16))],
    )

    metric = result["windows"][0]["features"]["category"]
    assert metric["future_only_category_count"] == 1
    assert metric["psi_missing_bucket_included"] is True
    assert np.isfinite(metric["psi"])


def test_empty_window_is_reported_without_calculating_drift():
    historical, future = _frames()
    result = _analyze(
        historical,
        future,
        [ComparisonWindow("empty", date(2020, 2, 1), date(2020, 2, 3))],
    )

    window = result["windows"][0]
    assert window["status"] == "empty"
    assert window["future_rows"] == 0
    assert window["features"] == {}
    assert window["seasonal_features"] == {}


def test_analysis_is_repeatable():
    historical, future = _frames()
    windows = [
        ComparisonWindow("first", date(2020, 1, 9), date(2020, 1, 12)),
        ComparisonWindow("second", date(2020, 1, 13), date(2020, 1, 16)),
    ]

    first = _analyze(historical, future, windows)
    second = _analyze(historical, future, windows)

    assert first == second


def test_forbidden_target_or_helper_columns_cannot_be_features():
    historical, future = _frames()
    with pytest.raises(ValueError, match="Target/helper"):
        analyze_multiwindow_drift(
            historical.assign(is_canceled=0),
            future.assign(is_canceled=1),
            ["number", "is_canceled"],
            ["number"],
            ["is_canceled"],
            windows=[ComparisonWindow("all", date(2020, 1, 9), date(2020, 1, 16))],
        )
