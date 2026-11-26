import json

import numpy as np
import pandas as pd

from ml.alert_engine import (
    backfill_alert_history,
    explain_monitoring_report,
    generate_alerts,
    load_alert_history,
    persist_alerts,
)
from ml.drift_detection import calculate_numerical_drift
from ml.monitoring import build_monitoring_report


def _report(numeric_values, *, source="real", batch_id="batch-a", missing=False):
    rng = np.random.default_rng(42)
    reference = pd.DataFrame(
        {
            "adr": rng.normal(size=1000),
            "market_segment": ["Direct", "Online TA"] * 500,
        }
    )
    incoming = pd.DataFrame(
        {
            "adr": numeric_values,
            "market_segment": (
                ["Direct", "Online TA"] * ((len(numeric_values) + 1) // 2)
            )[: len(numeric_values)],
            "batch_id": [batch_id] * len(numeric_values),
            "generated_at": ["2026-01-01T00:00:00Z"] * len(numeric_values),
        }
    )
    if missing:
        incoming.loc[incoming.index[:100], "lead_time"] = np.nan
    return build_monitoring_report(
        incoming,
        reference,
        ["adr", "market_segment"],
        ["adr"],
        ["market_segment"],
        data_source_type=source,
    )


def test_healthy_moderate_and_severe_alerts_use_drift_thresholds():
    rng = np.random.default_rng(42)
    reference_sample = rng.normal(size=1000)
    healthy = _report(reference_sample.copy(), batch_id="healthy-batch")
    moderate = _report(
        np.concatenate(
            [
                reference_sample[:600],
                np.random.default_rng(7).normal(loc=1, size=400),
            ]
        ),
        batch_id="moderate-batch",
    )
    severe = _report(
        np.random.default_rng(10).normal(loc=3.0, size=1000),
        batch_id="severe-batch",
    )

    healthy_alerts = generate_alerts(healthy)
    moderate_alerts = generate_alerts(moderate)
    severe_alerts = generate_alerts(severe)

    assert [alert["severity"] for alert in healthy_alerts] == ["HEALTHY"]
    assert any(alert["severity"] == "WARNING" for alert in moderate_alerts)
    assert any(alert["severity"] == "CRITICAL" for alert in severe_alerts)
    assert all(
        {
            "alert_id",
            "batch_id",
            "timestamp",
            "severity",
            "affected_feature",
            "detection_reason",
            "data_source",
            "recommended_action",
        } <= alert.keys()
        for alert in healthy_alerts + moderate_alerts + severe_alerts
    )


def test_explanations_include_reference_and_current_distribution_evidence():
    reference = pd.DataFrame({"value": np.arange(100)})
    incoming = pd.DataFrame({"value": np.arange(100) + 50})

    feature = calculate_numerical_drift(reference, incoming, ["value"])["value"]

    assert feature["distribution"]["method"] == "reference_quantile_bins"
    assert sum(item["proportion"] for item in feature["distribution"]["reference"]) == 1
    assert sum(item["proportion"] for item in feature["distribution"]["incoming"]) == 1


def test_missing_values_unseen_categories_and_invalid_values_have_evidence():
    rng = np.random.default_rng(42)
    reference = pd.DataFrame(
        {
            "adr": rng.normal(size=1000),
            "lead_time": rng.uniform(1, 1000, size=1000),
            "market_segment": ["Direct", "Online TA"] * 500,
        }
    )
    incoming = pd.DataFrame(
        {
            "adr": rng.normal(size=1000),
            "lead_time": [np.nan] * 100 + [-5.0] * 900,
            "market_segment": ["UNEXPECTED"] * 20 + ["Direct"] * 980,
            "batch_id": ["quality-batch"] * 1000,
            "generated_at": ["2026-01-01T00:00:00Z"] * 1000,
        }
    )
    report = build_monitoring_report(
        incoming,
        reference,
        ["adr", "lead_time", "market_segment"],
        ["adr", "lead_time"],
        ["market_segment"],
        data_source_type="real",
    )

    explanation = explain_monitoring_report(report)
    cause_types = {cause["type"] for cause in explanation["root_causes"]}
    alerts = generate_alerts(report)

    assert {"missing_values", "unexpected_categories", "invalid_values"} <= cause_types
    assert any(alert["severity"] == "CRITICAL" for alert in alerts)
    assert any("UNEXPECTED" in str(alert["evidence"]) for alert in alerts)
    assert all(cause["confidence"] != "confirmed" for cause in explanation["root_causes"])


def test_alert_persistence_is_deduplicated_and_keeps_source_provenance(tmp_path):
    alerts_path = tmp_path / "alerts.jsonl"
    real_report = _report(
        np.random.default_rng(1).normal(loc=3, size=1000),
        source="real",
        batch_id="production-batch",
    )
    simulated_report = _report(
        np.random.default_rng(1).normal(loc=3, size=1000),
        source="synthetic_test",
        batch_id="normal-demo-seed-42",
    )

    first = persist_alerts(real_report, alerts_path)
    second = persist_alerts(real_report, alerts_path)
    simulated = persist_alerts(simulated_report, alerts_path)
    history = load_alert_history(alerts_path)

    assert first == second
    assert len(history) == len(first) + len(simulated)
    assert {item["data_source"] for item in history} == {"REAL", "SIMULATED"}
    assert len({item["alert_id"] for item in history}) == len(history)
    assert len(load_alert_history(alerts_path, source="REAL")) == len(first)
    assert all(json.loads(line)["alert_id"] for line in alerts_path.read_text().splitlines())


def test_alert_history_filters_by_severity_batch_and_source(tmp_path):
    alerts_path = tmp_path / "alerts.jsonl"
    report = _report(
        np.random.default_rng(1).normal(loc=3, size=1000),
        source="synthetic_test",
        batch_id="normal-demo-seed-42",
    )
    persist_alerts(report, alerts_path)

    selected = load_alert_history(
        alerts_path,
        severity="CRITICAL",
        batch_id="normal-demo-seed-42",
        source="SIMULATED",
    )

    assert selected
    assert all(alert["severity"] == "CRITICAL" for alert in selected)
    assert all(alert["batch_id"] == "normal-demo-seed-42" for alert in selected)
    assert all(alert["data_source"] == "SIMULATED" for alert in selected)


def test_backfill_reuses_existing_reports_without_promoting_unknown_to_real(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    timestamp = "2026-01-01T00:00:00+00:00"
    reports = [
        {
            "batch_id": "normal-history-seed-42",
            "monitoring_timestamp": timestamp,
            "overall_status": "HEALTHY",
            "quality": {"quality_status": "HEALTHY"},
            "drift": {"numerical": {}, "categorical": {}},
        },
        {
            "batch_id": "legacy-history",
            "monitoring_timestamp": timestamp,
            "overall_status": "HEALTHY",
            "quality": {"quality_status": "HEALTHY"},
            "drift": {"numerical": {}, "categorical": {}},
        },
    ]
    for index, report in enumerate(reports):
        (reports_dir / f"report-{index}.json").write_text(json.dumps(report))
    alerts_path = tmp_path / "alerts.jsonl"

    backfill_alert_history(reports_dir, alerts_path)
    history = load_alert_history(alerts_path)

    assert {alert["data_source"] for alert in history} == {
        "SIMULATED",
        "UNKNOWN",
    }
