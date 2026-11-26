from pathlib import Path

import pandas as pd

from ml.offline_data_quality_report import (
    build_offline_quality_report,
    generate_offline_data_quality_report,
)


def _dataset():
    historical = pd.DataFrame(
        {
            "number": [1, 2, 2],
            "category": ["A", "B", "A"],
            "is_canceled": [0, 1, 0],
            "offline_row_id": [1, 2, 3],
        }
    )
    future = pd.DataFrame(
        {
            "number": [3, 4],
            "category": ["A", "NEW"],
            "is_canceled": [1, 0],
            "offline_row_id": [4, 5],
        }
    )
    return historical, future


def test_offline_report_uses_checker_and_distinguishes_pipeline_categories():
    historical, future = _dataset()
    report = build_offline_quality_report(
        historical,
        future,
        ["number", "category"],
        ["number"],
        ["category"],
        {"category": {"A", "B", "NEW"}},
    )

    assert "Historical observed category count | Saved pipeline supported count" in report
    assert "Future-only vs historical" in report
    assert "`NEW`" in report
    assert "is_canceled" in report
    assert "offline_row_id" in report
    assert "Unexpected categorical values relative to the historical" in report


def test_report_generation_reads_inputs_and_writes_only_report(tmp_path, monkeypatch):
    historical, future = _dataset()
    historical_path = tmp_path / "historical.csv"
    future_path = tmp_path / "future.csv"
    feature_names_path = tmp_path / "feature_names.json"
    metadata_path = tmp_path / "metadata.json"
    pipeline_path = tmp_path / "pipeline.pkl"
    report_path = tmp_path / "reports" / "offline_data_quality_report.md"
    historical.to_csv(historical_path, index=False)
    future.to_csv(future_path, index=False)
    feature_names_path.write_text('["number", "category"]', encoding="utf-8")
    metadata_path.write_text(
        '{"feature_names":["number","category"],'
        '"numerical_features":["number"],'
        '"categorical_features":["category"]}',
        encoding="utf-8",
    )
    pipeline_path.touch()
    before_historical = historical_path.read_bytes()
    before_future = future_path.read_bytes()

    def fake_supported_categories(_pipeline_path: Path):
        return {"category": {"A", "B", "NEW"}}

    import ml.offline_data_quality_report as report_module

    monkeypatch.setattr(
        report_module, "_saved_pipeline_categories", fake_supported_categories
    )
    result_path = generate_offline_data_quality_report(
        historical_path=historical_path,
        future_path=future_path,
        feature_names_path=feature_names_path,
        metadata_path=metadata_path,
        pipeline_path=pipeline_path,
        report_path=report_path,
    )

    assert result_path == report_path
    assert report_path.is_file()
    assert historical_path.read_bytes() == before_historical
    assert future_path.read_bytes() == before_future
