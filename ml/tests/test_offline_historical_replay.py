from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from ml.model2_config import DEGRADATION_TARGET, MODEL2_FEATURE_NAMES
from ml.model2_training import training_eligibility
from ml.offline_historical_replay import (
    DATA_SOURCE_TYPE,
    FORBIDDEN_MODEL1_FEATURES,
    PROVENANCE,
    SOURCE,
    _load_model1_artifacts,
    _offline_model2_eligibility,
    _summarize_duplicates,
    _build_windows,
    _fit_and_save_model2,
    run_historical_replay,
)


def _write_test_model_artifacts(tmp_path: Path):
    features = ["lead_time", "adults"]
    feature_path = tmp_path / "feature_names.json"
    feature_path.write_text(json.dumps(features), encoding="utf-8")
    metadata_path = tmp_path / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "feature_names": features,
                "target_column": "is_canceled",
                "numerical_features": features,
                "categorical_features": [],
            }
        ),
        encoding="utf-8",
    )
    pipeline = Pipeline(
        [
            ("preprocessor", _IdentityPreprocessor(features)),
            ("classifier", RandomForestClassifier(n_estimators=5, random_state=3)),
        ]
    )
    train = pd.DataFrame({"lead_time": [1, 2, 3, 4], "adults": [1, 1, 2, 2]})
    pipeline.fit(train, [0, 0, 1, 1])
    model_path = tmp_path / "pipeline.joblib"
    import joblib

    joblib.dump(pipeline, model_path)
    return feature_path, metadata_path, model_path


class _IdentityPreprocessor:
    def __init__(self, feature_names):
        self.feature_names = list(feature_names)

    def fit(self, X, y=None):
        del y
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return X.loc[:, self.feature_names]


def test_saved_model_artifact_feature_order_and_forbidden_features_are_enforced(tmp_path):
    feature_path, metadata_path, model_path = _write_test_model_artifacts(tmp_path)

    _, _, names, numeric, categorical = _load_model1_artifacts(
        feature_names_path=feature_path,
        metadata_path=metadata_path,
        pipeline_path=model_path,
    )

    assert names == ["lead_time", "adults"]
    assert numeric == ["lead_time", "adults"]
    assert categorical == []
    assert not (set(names) & FORBIDDEN_MODEL1_FEATURES)

    feature_path.write_text(
        json.dumps(["lead_time", "is_canceled"]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="forbidden fields"):
        _load_model1_artifacts(
            feature_names_path=feature_path,
            metadata_path=metadata_path,
            pipeline_path=model_path,
        )


def test_duplicate_conflicts_are_reported_and_identical_features_stay_in_one_week():
    frame = pd.DataFrame(
        {
            "lead_time": [2, 2, 2, 4],
            "adults": [1, 1, 1, 2],
            "is_canceled": [0, 0, 1, 1],
        }
    )
    weeks = pd.Series(
        pd.PeriodIndex(["2020-01-06/2020-01-12"] * 3 + ["2020-01-13/2020-01-19"], freq="W-SUN")
    )
    diagnostics, signatures = _summarize_duplicates(
        frame, ["lead_time", "adults"], weeks
    )

    assert diagnostics["exact_duplicate_rows_beyond_first"] == 1
    assert diagnostics["model1_feature_groups_with_conflicting_cancellation_labels"] == 1
    assert diagnostics["rows_in_conflicting_model1_feature_groups"] == 3
    assert diagnostics["feature_groups_crossing_arrival_windows"] == 0
    assert signatures.iloc[0] == signatures.iloc[2]


def test_duplicate_feature_group_crossing_windows_is_rejected():
    frame = pd.DataFrame(
        {
            "lead_time": [2, 2],
            "adults": [1, 1],
            "is_canceled": [0, 1],
        }
    )
    weeks = pd.Series(
        pd.PeriodIndex(["2020-01-06/2020-01-12", "2020-01-13/2020-01-19"], freq="W-SUN")
    )

    with pytest.raises(ValueError, match="span multiple arrival-date windows"):
        _summarize_duplicates(frame, ["lead_time", "adults"], weeks)


def _model2_rows(count=30):
    rows = []
    start = pd.Timestamp("2020-01-12T23:59:59Z")
    for index in range(count):
        target = index % 2
        timestamp = start + pd.Timedelta(days=7 * index)
        row = {name: float((index + offset) % 7) for offset, name in enumerate(MODEL2_FEATURE_NAMES)}
        row.update(
            {
                "batch_id": f"offline-window-{index}",
                "monitoring_timestamp": timestamp.isoformat(),
                "target_future_timestamp": (timestamp + pd.Timedelta(days=7)).isoformat(),
                "data_source_type": DATA_SOURCE_TYPE,
                "source": SOURCE,
                "environment": "offline",
                "is_synthetic": False,
                "complete_actual_outcomes": True,
                "both_actual_classes": True,
                DEGRADATION_TARGET: target,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_offline_model2_requires_count_chronology_and_both_classes_in_each_partition():
    dataset = _model2_rows()

    eligibility, train_rows, test_rows = _offline_model2_eligibility(
        dataset, target_name=DEGRADATION_TARGET
    )

    assert eligibility["status"] == "READY"
    assert len(train_rows) >= 2
    assert len(test_rows) >= 2
    assert train_rows[DEGRADATION_TARGET].nunique() == 2
    assert test_rows[DEGRADATION_TARGET].nunique() == 2
    assert pd.to_datetime(train_rows["target_future_timestamp"], utc=True).max() < pd.to_datetime(
        test_rows["monitoring_timestamp"], utc=True
    ).min()

    one_class_holdout = dataset.copy()
    one_class_holdout.loc[24:, DEGRADATION_TARGET] = 0
    ineligible, _, _ = _offline_model2_eligibility(
        one_class_holdout, target_name=DEGRADATION_TARGET
    )
    assert ineligible["status"] == "NOT_ELIGIBLE"
    assert any("holdout windows lack both target classes" in reason for reason in ineligible["missing_requirements"])

    too_small, _, _ = _offline_model2_eligibility(
        dataset.iloc[:19].copy(), target_name=DEGRADATION_TARGET
    )
    assert too_small["status"] == "NOT_ELIGIBLE"
    assert any("Need at least 20" in reason for reason in too_small["missing_requirements"])

    duplicate_batches = dataset.copy()
    duplicate_batches.loc[1, "batch_id"] = duplicate_batches.loc[0, "batch_id"]
    duplicate_eligibility, _, _ = _offline_model2_eligibility(
        duplicate_batches, target_name=DEGRADATION_TARGET
    )
    assert duplicate_eligibility["status"] == "NOT_ELIGIBLE"
    assert duplicate_eligibility["eligible_observation_count"] == 29
    assert any(
        "require distinct batch IDs" in reason
        for reason in duplicate_eligibility["missing_requirements"]
    )


def test_model2_features_never_contain_future_target_values():
    future_only_fields = {
        DEGRADATION_TARGET,
        "target_future_f1",
        "target_f1_drop",
        "target_future_timestamp",
    }
    assert not (set(MODEL2_FEATURE_NAMES) & future_only_fields)


def test_offline_model2_trains_only_from_a_ready_eligibility_result(tmp_path):
    dataset = _model2_rows(40)
    eligibility, train_rows, test_rows = _offline_model2_eligibility(
        dataset, target_name=DEGRADATION_TARGET
    )
    assert eligibility["status"] == "READY"

    target_definition = {
        "target": DEGRADATION_TARGET,
        "stable_value": 0,
        "degraded_value": 1,
        "f1_absolute_drop_threshold": 0.05,
        "future_window_days": 30,
    }
    result = _fit_and_save_model2(
        train_rows,
        test_rows,
        target_name=DEGRADATION_TARGET,
        run_dir=tmp_path,
        eligibility=eligibility,
        target_definition=target_definition,
    )

    assert result["status"] == "TRAINED"
    metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["model_scope"] == PROVENANCE
    assert metadata["is_synthetic"] is False
    assert Path(result["model_path"]).parent == tmp_path


def test_targets_use_current_measured_f1_and_the_first_later_window():
    rows = []
    start = pd.Timestamp("2015-07-01")
    predictions = []
    for week_index in range(8):
        date = start + pd.Timedelta(days=7 * week_index)
        rows.extend(
            [
                {
                    "lead_time": float(week_index + 1),
                    "adults": 1.0,
                    "arrival_date_year": date.year,
                    "arrival_date_month": date.strftime("%B"),
                    "arrival_date_day_of_month": date.day,
                    "is_canceled": 0,
                },
                {
                    "lead_time": float(week_index + 1),
                    "adults": 1.0,
                    "arrival_date_year": date.year,
                    "arrival_date_month": date.strftime("%B"),
                    "arrival_date_day_of_month": date.day,
                    "is_canceled": 1,
                },
            ]
        )
        predictions.extend([0, 0] if week_index == 5 else [0, 1])

    frame = pd.DataFrame(rows)
    _, model2_dataset = _build_windows(
        frame,
        np.asarray(predictions),
        np.asarray([0.2, 0.8] * 8),
        feature_names=["lead_time", "adults"],
        numerical_features=["lead_time", "adults"],
        categorical_features=[],
        target_name=DEGRADATION_TARGET,
        stable_value=0,
        degraded_value=1,
        f1_drop_threshold=0.05,
        future_window_days=30,
    )

    first = model2_dataset.iloc[0]
    assert first["target_baseline_f1"] == 1.0
    assert first["target_future_f1"] == 0.0
    assert first["target_f1_drop"] == 1.0
    assert first[DEGRADATION_TARGET] == 1
    assert first["historical_f1_score"] == 1.0
    assert first["target_future_timestamp"] > first["monitoring_timestamp"]
    assert first["data_source_type"] == DATA_SOURCE_TYPE
    assert first["source"] == SOURCE
    assert first["is_synthetic"] == False


def test_run_writes_only_historical_offline_outputs_and_does_not_train_below_gate(
    tmp_path,
):
    feature_path, metadata_path, model_path = _write_test_model_artifacts(tmp_path)
    target_path = tmp_path / "target_definition.json"
    target_path.write_text(
        json.dumps(
            {
                "target": DEGRADATION_TARGET,
                "stable_value": 0,
                "degraded_value": 1,
                "f1_absolute_drop_threshold": 0.05,
                "future_window_days": 30,
            }
        ),
        encoding="utf-8",
    )
    rows = []
    start = pd.Timestamp("2015-07-01")
    for week_index in range(10):
        date = start + pd.Timedelta(days=7 * week_index)
        for label in (0, 1):
            rows.append(
                {
                    "lead_time": float(week_index + 1),
                    "adults": 1.0,
                    "arrival_date_year": date.year,
                    "arrival_date_month": date.strftime("%B"),
                    "arrival_date_day_of_month": date.day,
                    "is_canceled": label,
                }
            )
    dataset_path = tmp_path / "hotel_bookings.csv"
    pd.DataFrame(rows).to_csv(dataset_path, index=False)
    output_root = tmp_path / "runs"

    summary = run_historical_replay(
        dataset_path=dataset_path,
        feature_names_path=feature_path,
        model1_metadata_path=metadata_path,
        model1_pipeline_path=model_path,
        target_definition_path=target_path,
        output_root=output_root,
    )

    run_dir = Path(summary["output_directory"])
    assert summary["model_scope"] == PROVENANCE
    assert summary["data_source_type"] == DATA_SOURCE_TYPE
    assert summary["is_synthetic"] is False
    assert summary["booking_created_timestamp_available"] is False
    assert summary["model2_training"]["status"] == "NOT_ELIGIBLE"
    assert any("Need at least 20" in item for item in summary["model2_eligibility"]["missing_requirements"])
    assert (run_dir / "model1_predictions.csv").is_file()
    assert (run_dir / "historical_outcomes.csv").is_file()
    assert (run_dir / "model1_evaluation_windows.csv").is_file()
    assert (run_dir / "model2_historical_dataset.csv").is_file()
    assert (run_dir / "summary.json").is_file()
    assert not list(run_dir.glob("*.joblib"))

    prediction_rows = pd.read_csv(run_dir / "model1_predictions.csv")
    outcome_rows = pd.read_csv(run_dir / "historical_outcomes.csv")
    assert prediction_rows["offline_record_id"].equals(outcome_rows["offline_record_id"])
    assert prediction_rows["data_source_type"].eq(DATA_SOURCE_TYPE).all()
    assert outcome_rows["source"].eq(SOURCE).all()
    assert not prediction_rows["is_synthetic"].any()
    assert "is_canceled" not in prediction_rows.columns
    assert not (set(prediction_rows.columns) & FORBIDDEN_MODEL1_FEATURES)


def test_run_reports_ineligibility_when_there_are_too_few_chronological_windows(
    tmp_path,
):
    feature_path, metadata_path, model_path = _write_test_model_artifacts(tmp_path)
    target_path = tmp_path / "target_definition.json"
    target_path.write_text(
        json.dumps(
            {
                "target": DEGRADATION_TARGET,
                "stable_value": 0,
                "degraded_value": 1,
                "f1_absolute_drop_threshold": 0.05,
                "future_window_days": 30,
            }
        ),
        encoding="utf-8",
    )
    rows = []
    start = pd.Timestamp("2015-07-01")
    for week_index in range(4):
        date = start + pd.Timedelta(days=7 * week_index)
        for label in (0, 1):
            rows.append(
                {
                    "lead_time": float(week_index + 1),
                    "adults": 1.0,
                    "arrival_date_year": date.year,
                    "arrival_date_month": date.strftime("%B"),
                    "arrival_date_day_of_month": date.day,
                    "is_canceled": label,
                }
            )
    dataset_path = tmp_path / "hotel_bookings.csv"
    pd.DataFrame(rows).to_csv(dataset_path, index=False)

    summary = run_historical_replay(
        dataset_path=dataset_path,
        feature_names_path=feature_path,
        model1_metadata_path=metadata_path,
        model1_pipeline_path=model_path,
        target_definition_path=target_path,
        output_root=tmp_path / "runs",
    )

    assert summary["model2_training"]["status"] == "NOT_ELIGIBLE"
    assert any(
        "Need at least 5 chronological arrival windows" in reason
        for reason in summary["model2_eligibility"]["missing_requirements"]
    )


def test_historical_offline_rows_do_not_satisfy_production_model2_gate():
    dataset = _model2_rows()
    dataset["batch_prediction_count"] = 20
    dataset["batch_actual_outcome_count"] = 20
    dataset["batch_pending_outcome_count"] = 0
    dataset["batch_actual_class_0_count"] = 10
    dataset["batch_actual_class_1_count"] = 10
    dataset["batch_outcome_provenance"] = "historical_offline"

    production = training_eligibility(dataset)

    assert production["status"] == "NOT_TRAINED"
    assert production["eligible_labeled_observations"] == 0

