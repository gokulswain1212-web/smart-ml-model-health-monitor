import json

import numpy as np
import pandas as pd
import pytest

from ml.data_quality import assess_data_quality
from ml.drift_detection import (
    calculate_categorical_drift,
    calculate_numerical_drift,
    calculate_psi,
)
from ml.monitoring import _classify, build_monitoring_report, save_monitoring_report
from ml.monitoring_config import MonitoringThresholds


THRESHOLDS = MonitoringThresholds()


def test_psi_is_low_for_identical_distributions_and_high_for_shifted_values():
    rng = np.random.default_rng(42)
    reference = rng.normal(size=5000)
    identical = calculate_psi(reference, reference, bins=10)
    shifted = calculate_psi(reference, rng.normal(loc=2.0, size=5000), bins=10)

    assert identical == pytest.approx(0.0, abs=1e-10)
    assert shifted >= THRESHOLDS.psi_critical


def test_psi_ignores_missing_and_non_finite_observations():
    score = calculate_psi(
        [1.0, 2.0, 3.0, np.nan, np.inf],
        [1.0, 2.0, 3.0, None],
    )

    assert score == pytest.approx(0.0, abs=1e-10)
    with pytest.raises(ValueError, match="at least one finite"):
        calculate_psi([np.nan], [1.0])


def test_numerical_drift_reports_ks_statistic_and_significance():
    rng = np.random.default_rng(12)
    reference = pd.DataFrame({"lead_time": rng.normal(size=2000)})
    incoming = pd.DataFrame({"lead_time": rng.normal(loc=1.5, size=500)})

    result = calculate_numerical_drift(reference, incoming, ["lead_time"])
    feature = result["lead_time"]

    assert feature["usable"]
    assert feature["ks_statistically_significant"]
    assert feature["ks_statistic"] > 0
    assert 0 <= feature["ks_p_value"] <= 1
    assert feature["psi"] >= THRESHOLDS.psi_warning


def test_numerical_drift_skips_missing_or_unusable_columns():
    result = calculate_numerical_drift(
        pd.DataFrame({"lead_time": ["bad", "bad"]}),
        pd.DataFrame({"lead_time": [1, 2]}),
        ["lead_time", "missing_feature"],
    )

    assert result["lead_time"]["usable"] is False
    assert result["missing_feature"]["usable"] is False


def test_categorical_drift_reports_new_categories_and_distribution_changes():
    reference = pd.DataFrame({"market_segment": ["Direct"] * 90 + ["Online TA"] * 10})
    incoming = pd.DataFrame(
        {"market_segment": ["Direct"] * 20 + ["UNSEEN"] * 80}
    )

    result = calculate_categorical_drift(
        reference, incoming, ["market_segment"]
    )["market_segment"]

    assert result["has_new_categories"]
    assert result["new_categories"] == ["UNSEEN"]
    assert "UNSEEN" in result["category_distribution_differences"]
    assert result["drift_score"] >= THRESHOLDS.psi_critical


def test_data_quality_reports_missing_duplicates_invalid_values_and_types():
    reference = pd.DataFrame(
        {
            "lead_time": [1, 2, 3, 4],
            "adr": [-1.0, 10.0, 20.0, 30.0],
            "market_segment": ["Direct", "Direct", "Online TA", "Online TA"],
        }
    )
    batch = pd.DataFrame(
        {
            "lead_time": [1, -2, np.inf, "bad"],
            "adr": [-2.0, 10.0, 20.0, 30.0],
            "market_segment": ["Direct", "UNSEEN", "Online TA", "Online TA"],
            "batch_id": ["test-batch"] * 4,
            "generated_at": ["2026-01-01T00:00:00Z"] * 4,
        }
    )
    before = batch.copy(deep=True)

    result = assess_data_quality(
        batch,
        reference,
        ["lead_time", "adr", "market_segment"],
        ["lead_time", "adr"],
        ["market_segment"],
    )

    assert result["batch_id"] == "test-batch"
    assert result["missing_value_counts"]["lead_time"] == 0
    assert result["invalid_numerical_value_counts"]["lead_time"] == 1
    assert result["infinite_value_counts"]["lead_time"] == 1
    assert result["negative_value_counts"]["lead_time"] == 1
    assert result["unexpected_category_counts"]["market_segment"] == 1
    assert result["incorrect_data_types"]["lead_time"]["actual"] == "object"
    assert result["quality_status"] == "CRITICAL"
    assert {
        "feature_name",
        "check_name",
        "affected_row_count",
        "severity",
        "message",
    }.issubset(result["findings"][0])
    assert result["severity_counts"]["ERROR"] > 0
    assert set(result["severity_counts"]) == {"ERROR", "WARNING", "INFO"}
    pd.testing.assert_frame_equal(batch, before)


def test_feature_schema_validation_reports_missing_extra_and_wrong_order():
    reference = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    batch = pd.DataFrame(
        {
            "b": ["x"],
            "unexpected": [1],
            "batch_id": ["batch"],
            "generated_at": ["now"],
        }
    )

    result = assess_data_quality(
        batch, reference, ["a", "b"], ["a"], ["b"]
    )

    assert result["feature_schema"]["missing_features"] == ["a"]
    assert result["feature_schema"]["unexpected_features"] == ["unexpected"]
    assert result["feature_schema"]["feature_order_matches"] is False
    assert result["quality_status"] == "CRITICAL"
    finding_checks = {
        (finding["feature_name"], finding["check_name"])
        for finding in result["findings"]
    }
    assert ("a", "missing_feature") in finding_checks
    assert ("unexpected", "unexpected_feature") in finding_checks


def test_feature_order_mismatch_has_warning_finding():
    reference = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    batch = pd.DataFrame(
        {
            "b": ["x", "y"],
            "a": [1, 2],
            "batch_id": ["batch"] * 2,
            "generated_at": ["now"] * 2,
        }
    )

    result = assess_data_quality(
        batch, reference, ["a", "b"], ["a"], ["b"]
    )

    assert result["feature_schema"]["feature_order_matches"] is False
    assert result["quality_status"] == "WARNING"
    assert any(
        finding["feature_name"] == "__schema__"
        and finding["check_name"] == "feature_order"
        and finding["severity"] == "WARNING"
        for finding in result["findings"]
    )


def test_data_quality_reports_missing_invalid_infinite_and_unexpected_values():
    reference = pd.DataFrame(
        {
            "lead_time": [1, 2, 3, 4],
            "market_segment": ["Direct", "Direct", "Online TA", "Online TA"],
        }
    )
    batch = pd.DataFrame(
        {
            "lead_time": [None, "invalid", np.inf, -1],
            "market_segment": ["Direct", "UNSEEN", "Online TA", "Direct"],
            "batch_id": ["quality-check"] * 4,
            "generated_at": ["2026-01-01T00:00:00Z"] * 4,
        }
    )

    result = assess_data_quality(
        batch,
        reference,
        ["lead_time", "market_segment"],
        ["lead_time"],
        ["market_segment"],
    )
    finding_checks = {
        (finding["feature_name"], finding["check_name"], finding["severity"])
        for finding in result["findings"]
    }

    assert result["missing_value_counts"]["lead_time"] == 1
    assert result["invalid_numerical_value_counts"]["lead_time"] == 1
    assert result["infinite_value_counts"]["lead_time"] == 1
    assert result["negative_value_counts"]["lead_time"] == 1
    assert result["unexpected_category_counts"]["market_segment"] == 1
    assert ("lead_time", "missing_values", "ERROR") in finding_checks
    assert ("lead_time", "invalid_numeric_values", "ERROR") in finding_checks
    assert ("lead_time", "infinite_numeric_values", "ERROR") in finding_checks
    assert ("lead_time", "negative_values", "ERROR") in finding_checks
    assert ("market_segment", "unexpected_categories", "ERROR") in finding_checks


def test_duplicate_and_metadata_findings_have_structured_severities():
    reference = pd.DataFrame(
        {"lead_time": list(range(100)), "market_segment": ["A", "B"] * 50}
    )
    batch = reference.copy()
    batch.loc[99] = batch.loc[0]
    batch["batch_id"] = ["batch-a"] * 99 + ["batch-b"]
    batch["generated_at"] = ["now"] * len(batch)

    result = assess_data_quality(
        batch,
        reference,
        ["lead_time", "market_segment"],
        ["lead_time"],
        ["market_segment"],
    )
    finding_checks = {
        (finding["feature_name"], finding["check_name"], finding["severity"])
        for finding in result["findings"]
    }

    assert ("__batch__", "duplicate_records", "WARNING") in finding_checks
    assert ("batch_id", "inconsistent_batch_metadata", "WARNING") in finding_checks


def test_empty_batch_is_critical_with_one_error_finding():
    reference = pd.DataFrame(
        {"lead_time": [1, 2], "market_segment": ["A", "B"]}
    )
    batch = pd.DataFrame(
        columns=["lead_time", "market_segment", "batch_id", "generated_at"]
    )

    result = assess_data_quality(
        batch,
        reference,
        ["lead_time", "market_segment"],
        ["lead_time"],
        ["market_segment"],
    )

    assert result["quality_status"] == "CRITICAL"
    assert result["findings"] == [
        {
            "feature_name": "__batch__",
            "check_name": "empty_batch",
            "affected_row_count": 0,
            "severity": "ERROR",
            "message": "Batch contains no records.",
        }
    ]
    assert result["severity_counts"] == {"ERROR": 1, "WARNING": 0, "INFO": 0}


def test_existing_quality_return_fields_remain_available():
    reference = pd.DataFrame(
        {"lead_time": [1, 2], "market_segment": ["A", "B"]}
    )
    batch = pd.DataFrame(
        {
            "lead_time": [1, 2],
            "market_segment": ["A", "B"],
            "batch_id": ["compatibility-batch"] * 2,
            "generated_at": ["now"] * 2,
        }
    )

    result = assess_data_quality(
        batch,
        reference,
        ["lead_time", "market_segment"],
        ["lead_time"],
        ["market_segment"],
    )

    assert {
        "batch_id",
        "total_rows",
        "total_columns",
        "missing_value_counts",
        "missing_value_percentages",
        "reference_missing_percentages",
        "missing_excess_percentage_points",
        "duplicate_count",
        "duplicate_rate",
        "reference_duplicate_rate",
        "duplicate_rate_excess",
        "invalid_numerical_value_counts",
        "unexpected_category_counts",
        "unexpected_category_values",
        "incorrect_data_types",
        "infinite_value_counts",
        "negative_value_counts",
        "feature_schema",
        "monitoring_metadata",
        "quality_status",
        "critical_reasons",
        "warning_reasons",
    }.issubset(result)
    assert result["quality_status"] == "HEALTHY"
    assert result["severity_counts"] == {"ERROR": 0, "WARNING": 0, "INFO": 0}


def test_duplicate_rows_are_counted_without_metadata_columns():
    reference = pd.DataFrame(
        {
            "lead_time": list(range(100)),
            "market_segment": ["A", "B"] * 50,
        }
    )
    batch = reference.copy()
    batch.loc[99] = batch.loc[0]
    batch["batch_id"] = "batch"
    batch["generated_at"] = "now"

    report = assess_data_quality(
        batch, reference, ["lead_time", "market_segment"], ["lead_time"], ["market_segment"]
    )

    assert report["duplicate_count"] == 1
    assert report["quality_status"] == "WARNING"


def test_status_classification_uses_configured_thresholds_and_actual_scores():
    thresholds = MonitoringThresholds(critical_drifted_features=2)
    quality = {"quality_status": "HEALTHY", "critical_reasons": [], "warning_reasons": []}
    numerical = {
        "lead_time": {"usable": True, "psi": 0.25, "ks_meaningful_effect": False},
        "adr": {"usable": True, "psi": 0.22, "ks_meaningful_effect": False},
    }

    status, reasons = _classify(quality, numerical, {}, thresholds)

    assert status == "CRITICAL"
    assert any("2 features exceed" in reason for reason in reasons)


def test_combined_report_and_json_save_do_not_predict(tmp_path):
    reference = pd.DataFrame(
        {"lead_time": [1, 2, 3, 4], "market_segment": ["A", "A", "B", "B"]}
    )
    batch = pd.DataFrame(
        {
            "lead_time": [1, 2, 3, 4],
            "market_segment": ["A", "A", "B", "B"],
            "batch_id": ["report-test"] * 4,
            "generated_at": ["2026-01-01T00:00:00Z"] * 4,
        }
    )
    report = build_monitoring_report(
        batch, reference, ["lead_time", "market_segment"], ["lead_time"], ["market_segment"]
    )

    first_path = save_monitoring_report(report, tmp_path)
    second_path = save_monitoring_report(report, tmp_path)
    saved_report = json.loads(first_path.read_text(encoding="utf-8"))

    assert first_path.is_file()
    assert second_path.is_file()
    assert first_path != second_path
    assert saved_report["batch_id"] == "report-test"
    assert saved_report["model_prediction"]["performed"] is False
    assert saved_report["overall_status"] == "HEALTHY"


def test_poor_quality_report_is_critical_and_prediction_is_not_attempted():
    reference = pd.DataFrame(
        {"lead_time": [1, 2, 3], "market_segment": ["A", "B", "A"]}
    )
    batch = pd.DataFrame(
        {
            "lead_time": [-999, 2, 3],
            "market_segment": ["UNEXPECTED", "B", "A"],
            "batch_id": ["bad-batch"] * 3,
            "generated_at": ["2026-01-01T00:00:00Z"] * 3,
        }
    )

    report = build_monitoring_report(
        batch, reference, ["lead_time", "market_segment"], ["lead_time"], ["market_segment"]
    )

    assert report["overall_status"] == "CRITICAL"
    assert report["model_prediction"]["performed"] is False


def test_monitoring_report_records_explicit_and_legacy_simulator_provenance():
    reference = pd.DataFrame(
        {"lead_time": [1, 2, 3], "market_segment": ["A", "B", "A"]}
    )
    batch = pd.DataFrame(
        {
            "lead_time": [1, 2, 3],
            "market_segment": ["A", "B", "A"],
            "batch_id": ["normal-20260101T000000Z-seed-42"] * 3,
            "generated_at": ["2026-01-01T00:00:00Z"] * 3,
        }
    )
    args = (batch, reference, ["lead_time", "market_segment"], ["lead_time"], ["market_segment"])

    inferred = build_monitoring_report(*args)
    real_batch = batch.copy()
    real_batch["batch_id"] = "production-batch-1"
    declared_real = build_monitoring_report(
        real_batch,
        reference,
        ["lead_time", "market_segment"],
        ["lead_time"],
        ["market_segment"],
        data_source_type="real",
    )
    production = build_monitoring_report(
        real_batch,
        reference,
        ["lead_time", "market_segment"],
        ["lead_time"],
        ["market_segment"],
        data_source_type="real",
        source="production",
    )

    assert inferred["data_source_type"] == "synthetic_test"
    assert declared_real["data_source_type"] == "real"
    assert "source" not in declared_real
    assert production["source"] == "production"
    with pytest.raises(ValueError, match="must use real data source type"):
        build_monitoring_report(
            real_batch,
            reference,
            ["lead_time", "market_segment"],
            ["lead_time"],
            ["market_segment"],
            source="production",
        )
    with pytest.raises(ValueError, match="cannot be declared as real"):
        build_monitoring_report(*args, data_source_type="real")
