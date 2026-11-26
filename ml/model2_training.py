from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.model2_config import (
    DEGRADATION_CLASSIFICATION_THRESHOLD,
    DEGRADATION_TARGET,
    F1_DEGRADATION_THRESHOLD,
    FUTURE_PERFORMANCE_WINDOW_DAYS,
    MIN_LABELED_OBSERVATIONS,
    MODEL2_FEATURE_NAMES,
    TEST_FRACTION,
)
from ml.model2_dataset import (
    MODEL2_DATA_DIR,
    build_historical_dataset,
    dataset_summary,
    save_historical_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL2_DIR = ROOT / "ml" / "models" / "model2"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("x", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, indent=2, allow_nan=False)
            file.write("\n")
        os.link(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _stage_json(path: Path, payload: dict[str, Any]) -> Path:
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary_path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
        file.write("\n")
    return temporary_path


def _publish_artifacts(
    forest: Pipeline,
    model_path: Path,
    metadata_path: Path,
    evaluation_report_path: Path,
    metadata: dict[str, Any],
    evaluation_report: dict[str, Any],
) -> None:
    staged_model = model_path.with_name(f".{model_path.name}.{uuid4().hex}.tmp")
    staged_metadata: Path | None = None
    staged_evaluation: Path | None = None
    published: list[Path] = []
    completed = False
    try:
        joblib.dump(forest, staged_model)
        staged_evaluation = _stage_json(evaluation_report_path, evaluation_report)
        staged_metadata = _stage_json(metadata_path, metadata)
        for staged_path, final_path in (
            (staged_model, model_path),
            (staged_evaluation, evaluation_report_path),
            (staged_metadata, metadata_path),
        ):
            os.link(staged_path, final_path)
            published.append(final_path)
        completed = True
    finally:
        for path in (staged_model, staged_metadata, staged_evaluation):
            if path is not None and path.exists():
                path.unlink()
        if not completed:
            for path in reversed(published):
                if path.exists():
                    path.unlink()


def _ensure_static_artifact(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(
                f"Existing Model 2 definition differs; refusing to overwrite {path}."
            )
        return
    _write_json_new(path, payload)


def _target_definition() -> dict[str, Any]:
    return {
        "target": DEGRADATION_TARGET,
        "stable_value": 0,
        "degraded_value": 1,
        "rule": (
            "For a trusted production monitoring observation, compare the latest genuine F1 score "
            "available at its monitoring timestamp with the first genuine F1 score "
            "recorded afterward, within the configured future window. Label 1 when "
            "the absolute F1 decline is at least the configured threshold; otherwise "
            "label 0."
        ),
        "f1_absolute_drop_threshold": F1_DEGRADATION_THRESHOLD,
        "future_window_days": FUTURE_PERFORMANCE_WINDOW_DAYS,
        "required_observations": [
            "A monitoring report from the token-authenticated production ingestion at the prediction timestamp.",
            "A genuine actual-outcome performance evaluation at or before that timestamp.",
            "A later genuine actual-outcome performance evaluation within the future window.",
        ],
        "leakage_controls": [
            "Monitoring features are computed only from records timestamped at or before the monitoring observation.",
            "Future F1 is used only to create the target and is never included in the feature list.",
            "Only token-authenticated production-source observations are eligible for training labels.",
            "Synthetic, operator-declared-only, or unverified sources are not assigned training labels.",
        ],
    }


def _not_trained_result(
    reason: str,
    dataset: pd.DataFrame,
    *,
    model_dir: Path,
    dataset_path: Path | None = None,
) -> dict[str, Any]:
    summary = dataset_summary(dataset)
    result = {
        "model2_status": "NOT_TRAINED",
        "reason": reason,
        "historical_dataset": summary,
        "training_size": 0,
        "testing_size": 0,
        "model_path": None,
        "metadata_path": None,
        "dataset_path": str(dataset_path) if dataset_path else None,
    }
    _write_json_new(
        model_dir
        / f"training_status_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json",
        result,
    )
    return result


def _make_logistic_regression() -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def _make_random_forest() -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _evaluate(
    estimator: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    classification_threshold: float,
) -> dict[str, Any]:
    probabilities = estimator.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= classification_threshold).astype(int)
    has_both_classes = set(y_test.unique()) == {0, 1}
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0, pos_label=1)
        ),
        "recall": float(recall_score(y_test, predictions, zero_division=0, pos_label=1)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0, pos_label=1)),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "roc_auc": (
            float(roc_auc_score(y_test, probabilities)) if has_both_classes else None
        ),
        "pr_auc": (
            float(average_precision_score(y_test, probabilities))
            if has_both_classes
            else None
        ),
        "test_class_counts": {
            str(label): int(count)
            for label, count in y_test.value_counts().sort_index().items()
        },
    }


def _verified_complete_batch_mask(dataset: pd.DataFrame) -> pd.Series:
    required = {
        "batch_prediction_count",
        "batch_actual_outcome_count",
        "batch_pending_outcome_count",
        "batch_actual_class_0_count",
        "batch_actual_class_1_count",
        "batch_outcome_provenance",
    }
    if not required.issubset(dataset.columns):
        return pd.Series(False, index=dataset.index)
    prediction_count = pd.to_numeric(
        dataset["batch_prediction_count"], errors="coerce"
    )
    actual_count = pd.to_numeric(
        dataset["batch_actual_outcome_count"], errors="coerce"
    )
    pending_count = pd.to_numeric(
        dataset["batch_pending_outcome_count"], errors="coerce"
    )
    class_0_count = pd.to_numeric(
        dataset["batch_actual_class_0_count"], errors="coerce"
    )
    class_1_count = pd.to_numeric(
        dataset["batch_actual_class_1_count"], errors="coerce"
    )
    return (
        prediction_count.gt(0)
        & actual_count.eq(prediction_count)
        & pending_count.eq(0)
        & class_0_count.gt(0)
        & class_1_count.gt(0)
        & class_0_count.add(class_1_count).eq(actual_count)
        & dataset["batch_outcome_provenance"].eq("verified_production_api")
    )


def training_eligibility(
    dataset: pd.DataFrame,
    *,
    minimum_labeled_observations: int = MIN_LABELED_OBSERVATIONS,
    test_fraction: float = TEST_FRACTION,
) -> dict[str, Any]:
    if minimum_labeled_observations < MIN_LABELED_OBSERVATIONS:
        raise ValueError(
            "minimum_labeled_observations cannot be lower than "
            f"{MIN_LABELED_OBSERVATIONS}."
        )
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    eligible = dataset.loc[
        dataset.get("data_source_type", pd.Series(index=dataset.index, dtype=object)).eq("real")
        & dataset.get("source", pd.Series(index=dataset.index, dtype=object)).eq("production")
        & dataset.get(DEGRADATION_TARGET, pd.Series(index=dataset.index)).notna()
        & dataset.get(
            "label_provenance", pd.Series(index=dataset.index, dtype=object)
        ).eq("verified_production_api")
        & _verified_complete_batch_mask(dataset)
    ].copy()
    if "batch_id" not in eligible:
        count = 0
    else:
        count = int(eligible["batch_id"].nunique(dropna=True))
    missing_requirements: list[str] = []
    if count < minimum_labeled_observations:
        missing_requirements.append(
            f"Need at least {minimum_labeled_observations} genuine labeled production "
            f"observations; found {count}."
        )
    evidence_columns = {
        "batch_prediction_count",
        "batch_actual_outcome_count",
        "batch_pending_outcome_count",
        "batch_actual_class_0_count",
        "batch_actual_class_1_count",
        "batch_outcome_provenance",
    }
    if not evidence_columns.issubset(dataset.columns):
        missing_requirements.append(
            "Production observations require complete verified prediction and "
            "two-class outcome evidence for every batch."
        )
    if "batch_id" not in eligible or eligible["batch_id"].isna().any():
        missing_requirements.append(
            "Production labeled observations require non-empty batch IDs."
        )
    elif eligible["batch_id"].duplicated().any():
        missing_requirements.append(
            "Production training data contains duplicate batch observations."
        )
    if DEGRADATION_TARGET not in eligible:
        missing_requirements.append(
            f"Production training data is missing target column {DEGRADATION_TARGET!r}."
        )
    elif not eligible[DEGRADATION_TARGET].isin([0, 1]).all():
        missing_requirements.append(
            "Production training targets must be binary 0/1 labels."
        )

    required_timestamps = {"monitoring_timestamp", "target_future_timestamp"}
    if not required_timestamps.issubset(eligible.columns):
        missing_requirements.append(
            "Production labeled observations require monitoring and future-target timestamps."
        )
    else:
        eligible["monitoring_timestamp"] = pd.to_datetime(
            eligible["monitoring_timestamp"], utc=True, errors="coerce"
        )
        eligible["target_future_timestamp"] = pd.to_datetime(
            eligible["target_future_timestamp"], utc=True, errors="coerce"
        )
        if eligible[
            ["monitoring_timestamp", "target_future_timestamp"]
        ].isna().any().any():
            missing_requirements.append(
                "Production labeled observations require valid timestamps."
            )
        elif (
            eligible["target_future_timestamp"]
            <= eligible["monitoring_timestamp"]
        ).any():
            missing_requirements.append(
                "Future target timestamps must occur after their monitoring observations."
            )
        elif count >= 2:
            eligible = eligible.sort_values(
                "monitoring_timestamp", kind="stable"
            )
            split_index = int(np.floor(count * (1 - test_fraction)))
            if split_index < 2 or split_index >= count:
                missing_requirements.append(
                    "Chronological split leaves too few training or testing observations."
                )
            else:
                test_start = eligible.iloc[split_index]["monitoring_timestamp"]
                train_rows = eligible.iloc[:split_index]
                train_rows = train_rows.loc[
                    train_rows["target_future_timestamp"] < test_start
                ]
                if len(train_rows) < 2:
                    missing_requirements.append(
                        "Purging overlapping future-label windows leaves too few training observations."
                    )
                elif train_rows[DEGRADATION_TARGET].nunique() < 2:
                    missing_requirements.append(
                        "Purged chronological training partition does not contain both target classes."
                    )
        elif count >= minimum_labeled_observations:
            missing_requirements.append(
                "Chronological splitting requires at least two distinct observations."
            )

    class_counts = (
        {
            str(label): int(value)
            for label, value in eligible[DEGRADATION_TARGET]
            .value_counts()
            .sort_index()
            .items()
        }
        if not eligible.empty
        else {}
    )
    reason = "; ".join(missing_requirements) if missing_requirements else None
    return {
        "status": "NOT_TRAINED" if missing_requirements else "READY",
        "eligible_labeled_observations": count,
        "reason": reason,
        "missing_requirements": missing_requirements,
        "target_class_counts": class_counts,
    }


def train_model2(
    dataset: pd.DataFrame,
    *,
    model_dir: Path = MODEL2_DIR,
    minimum_labeled_observations: int = MIN_LABELED_OBSERVATIONS,
    test_fraction: float = TEST_FRACTION,
    dataset_path: Path | None = None,
) -> dict[str, Any]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if minimum_labeled_observations < MIN_LABELED_OBSERVATIONS:
        raise ValueError(
            "minimum_labeled_observations cannot be lower than "
            f"{MIN_LABELED_OBSERVATIONS}."
        )

    required = {
        *MODEL2_FEATURE_NAMES,
        DEGRADATION_TARGET,
        "batch_id",
        "data_source_type",
        "source",
        "monitoring_timestamp",
        "target_future_timestamp",
    }
    missing_columns = required - set(dataset.columns)
    if missing_columns:
        raise ValueError(f"Historical dataset is missing columns: {sorted(missing_columns)}")

    model_dir.mkdir(parents=True, exist_ok=True)
    _ensure_static_artifact(
        model_dir / "feature_names.json",
        {"feature_names": MODEL2_FEATURE_NAMES},
    )
    _ensure_static_artifact(
        model_dir / "target_definition.json",
        _target_definition(),
    )

    labeled = dataset.loc[
        (dataset["data_source_type"] == "real")
        & (dataset["source"] == "production")
        & dataset.get(
            "label_provenance", pd.Series(index=dataset.index, dtype=object)
        ).eq("verified_production_api")
        & _verified_complete_batch_mask(dataset)
        & dataset[DEGRADATION_TARGET].notna()
    ].copy()
    summary = dataset_summary(dataset)
    eligibility = training_eligibility(
        dataset,
        minimum_labeled_observations=minimum_labeled_observations,
        test_fraction=test_fraction,
    )
    if eligibility["status"] != "READY":
        return _not_trained_result(
            str(eligibility["reason"]),
            dataset,
            model_dir=model_dir,
            dataset_path=dataset_path,
        )
    if len(labeled) < minimum_labeled_observations:
        return _not_trained_result(
            (
                f"Need at least {minimum_labeled_observations} genuine labeled production observations; "
                f"found {len(labeled)}."
            ),
            dataset,
            model_dir=model_dir,
            dataset_path=dataset_path,
        )

    labeled["monitoring_timestamp"] = pd.to_datetime(
        labeled["monitoring_timestamp"], utc=True, errors="coerce"
    )
    if labeled["monitoring_timestamp"].isna().any():
        raise ValueError("Labeled observations must have valid monitoring timestamps.")
    labeled = labeled.sort_values(
        "monitoring_timestamp", kind="stable"
    ).reset_index(drop=True)
    labeled[DEGRADATION_TARGET] = labeled[DEGRADATION_TARGET].astype(int)
    if labeled["batch_id"].duplicated().any():
        raise ValueError("Labeled production data contains duplicate batch observations.")
    class_counts = labeled[DEGRADATION_TARGET].value_counts()
    if len(class_counts) < 2:
        return _not_trained_result(
            "Labeled production observations contain only one target class.",
            dataset,
            model_dir=model_dir,
            dataset_path=dataset_path,
        )

    split_index = int(np.floor(len(labeled) * (1 - test_fraction)))
    if split_index < 2 or split_index >= len(labeled):
        return _not_trained_result(
            "Chronological split leaves too few training or testing observations.",
            dataset,
            model_dir=model_dir,
            dataset_path=dataset_path,
        )
    test_rows = labeled.iloc[split_index:]
    test_start = test_rows["monitoring_timestamp"].iloc[0]
    train_candidates = labeled.iloc[:split_index].copy()
    target_future_times = pd.to_datetime(
        train_candidates["target_future_timestamp"], utc=True, errors="coerce"
    )
    train_rows = train_candidates.loc[target_future_times < test_start].copy()
    if len(train_rows) < 2:
        return _not_trained_result(
            "Purging overlapping future-label windows leaves too few training observations.",
            dataset,
            model_dir=model_dir,
            dataset_path=dataset_path,
        )
    if train_rows[DEGRADATION_TARGET].nunique() < 2:
        return _not_trained_result(
            "Purged chronological training partition does not contain both target classes.",
            dataset,
            model_dir=model_dir,
            dataset_path=dataset_path,
        )

    x_train = train_rows.loc[:, MODEL2_FEATURE_NAMES].apply(
        pd.to_numeric, errors="coerce"
    )
    y_train = train_rows[DEGRADATION_TARGET]
    x_test = test_rows.loc[:, MODEL2_FEATURE_NAMES].apply(
        pd.to_numeric, errors="coerce"
    )
    y_test = test_rows[DEGRADATION_TARGET]
    logistic = _make_logistic_regression()
    forest = _make_random_forest()
    logistic.fit(x_train, y_train)
    forest.fit(x_train, y_train)
    metrics = {
        "logistic_regression_baseline": _evaluate(
            logistic,
            x_test,
            y_test,
            classification_threshold=DEGRADATION_CLASSIFICATION_THRESHOLD,
        ),
        "random_forest_primary_candidate": _evaluate(
            forest,
            x_test,
            y_test,
            classification_threshold=DEGRADATION_CLASSIFICATION_THRESHOLD,
        ),
    }

    trained_at = _utc_timestamp()
    version = "model2-random-forest-" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    model_path = model_dir / f"{version}.joblib"
    metadata_path = model_dir / f"{version}.json"
    evaluation_report_path = model_dir / f"evaluation_report_{version}.json"
    if (
        model_path.exists()
        or metadata_path.exists()
        or evaluation_report_path.exists()
    ):
        raise FileExistsError(f"Refusing to overwrite Model 2 version {version}.")

    classifier = forest.named_steps["classifier"]
    importances = classifier.feature_importances_
    metadata = {
        "model2_status": "TRAINED",
        "model_name": "Model 2 Performance Degradation Predictor",
        "model_version": version,
        "algorithm": "RandomForestClassifier",
        "training_timestamp": trained_at,
        "training_source": "verified_genuine_production_monitoring",
        "target_distribution": summary["target_class_counts"],
        "feature_names": MODEL2_FEATURE_NAMES,
        "target_definition": _target_definition(),
        "training_configuration": {
            "minimum_labeled_observations": minimum_labeled_observations,
            "test_fraction": test_fraction,
            "split_method": "chronological",
            "random_state": 42,
            "class_weight": "balanced",
            "degradation_classification_threshold": (
                DEGRADATION_CLASSIFICATION_THRESHOLD
            ),
            "selected_model_policy": (
                "Random Forest is the predeclared primary model; Logistic Regression "
                "is a baseline. The chronological test set is not used for model selection."
            ),
        },
        "temporal_split": {
            "training_start": train_rows["monitoring_timestamp"].iloc[0].isoformat(),
            "training_end": train_rows["monitoring_timestamp"].iloc[-1].isoformat(),
            "training_target_end": pd.to_datetime(
                train_rows["target_future_timestamp"], utc=True
            ).max().isoformat(),
            "testing_start": test_rows["monitoring_timestamp"].iloc[0].isoformat(),
            "testing_end": test_rows["monitoring_timestamp"].iloc[-1].isoformat(),
        },
        "historical_dataset": {
            **summary,
            "training_size": int(len(train_rows)),
            "testing_size": int(len(test_rows)),
            "training_class_counts": {
                str(label): int(count)
                for label, count in y_train.value_counts().sort_index().items()
            },
            "testing_class_counts": {
                str(label): int(count)
                for label, count in y_test.value_counts().sort_index().items()
            },
            "dataset_path": str(dataset_path) if dataset_path else None,
        },
        "evaluation_report_path": str(evaluation_report_path),
        "evaluation_metrics": metrics,
        "feature_importance": [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in sorted(
                zip(MODEL2_FEATURE_NAMES, importances, strict=True),
                key=lambda item: item[1],
                reverse=True,
            )
        ],
        "limitations": [
            "Feature importance is descriptive, not causal.",
            "Metrics are chronological holdout results and are not claims about future performance.",
        ],
    }
    evaluation_report = {
        "model_version": version,
        "training_timestamp": trained_at,
        "source": "verified_genuine_production_monitoring",
        "target_name": DEGRADATION_TARGET,
        "target_distribution": summary["target_class_counts"],
        "evaluation_dataset_size": int(len(y_test)),
        "training_size": int(len(y_train)),
        "evaluation_metrics": metrics,
        "temporal_split": metadata["temporal_split"],
    }
    _publish_artifacts(
        forest,
        model_path,
        metadata_path,
        evaluation_report_path,
        metadata,
        evaluation_report,
    )

    return {
        "model2_status": "TRAINED",
        "reason": None,
        "historical_dataset": summary,
        "training_size": int(len(train_rows)),
        "testing_size": int(len(test_rows)),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "dataset_path": str(dataset_path) if dataset_path else None,
        "evaluation_report_path": str(evaluation_report_path),
        "evaluation_metrics": metrics,
        "model_version": version,
        "training_timestamp": trained_at,
    }


def run_training() -> dict[str, Any]:
    return run_training_if_eligible()


def _latest_trained_model(
    model_dir: Path,
) -> tuple[Path, dict[str, Any]] | None:
    for metadata_path in sorted(
        model_dir.glob("model2-random-forest-*.json"), reverse=True
    ):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Model 2 metadata {metadata_path.name} must contain a JSON object."
            )
        if metadata.get("model2_status") != "TRAINED":
            continue
        if metadata.get("model_version") != metadata_path.stem:
            raise ValueError(
                f"Model 2 metadata {metadata_path.name} has an inconsistent model version."
            )
        if metadata.get("feature_names") != MODEL2_FEATURE_NAMES:
            raise ValueError(
                f"Model 2 metadata {metadata_path.name} has an incompatible feature schema."
            )
        model_path = metadata_path.with_suffix(".joblib")
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Trained Model 2 artifact {model_path.name} is missing."
            )
        return metadata_path, metadata
    return None


def run_training_if_eligible() -> dict[str, Any]:
    def finish(result: dict[str, Any]) -> dict[str, Any]:
        result["training_attempt_timestamp"] = _utc_timestamp()
        _write_json_new(
            MODEL2_DIR
            / (
                "training_status_"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                + ".json"
            ),
            result,
        )
        return result

    try:
        dataset = build_historical_dataset()
        summary = dataset_summary(dataset)
        eligibility = training_eligibility(dataset)
        existing_model = _latest_trained_model(MODEL2_DIR)

        if existing_model is not None:
            metadata_path, metadata = existing_model
            historical = metadata.get("historical_dataset", {})
            trained_observations = historical.get(
                "genuine_labeled_observation_count"
            )
            if not isinstance(trained_observations, int) or isinstance(
                trained_observations, bool
            ):
                raise ValueError(
                    f"Model 2 metadata {metadata_path.name} has an invalid "
                    "genuine labeled observation count."
                )
            if (
                eligibility["status"] != "READY"
                or trained_observations
                >= summary["genuine_labeled_observation_count"]
            ):
                return finish(
                    {
                        "model2_status": "TRAINED",
                        "training_eligibility": (
                            "TRAINED"
                            if eligibility["status"] == "READY"
                            else eligibility["status"]
                        ),
                        "missing_requirements": eligibility[
                            "missing_requirements"
                        ],
                        "reason": (
                            eligibility["reason"]
                            if eligibility["status"] != "READY"
                            else (
                                "Model is already trained on the current eligible "
                                "production batch set; no retraining was needed."
                            )
                        ),
                        "training_performed": False,
                        "historical_dataset": summary,
                        "training_size": int(historical.get("training_size", 0)),
                        "testing_size": int(historical.get("testing_size", 0)),
                        "model_path": str(metadata_path.with_suffix(".joblib")),
                        "metadata_path": str(metadata_path),
                        "dataset_path": historical.get("dataset_path"),
                        "evaluation_metrics": metadata.get("evaluation_metrics"),
                        "model_version": metadata["model_version"],
                        "training_timestamp": metadata.get("training_timestamp"),
                    }
                )

        if eligibility["status"] != "READY":
            return finish(
                {
                    "model2_status": "NOT_TRAINED",
                    "training_eligibility": eligibility["status"],
                    "missing_requirements": eligibility[
                        "missing_requirements"
                    ],
                    "reason": eligibility["reason"],
                    "training_performed": False,
                    "historical_dataset": summary,
                    "training_size": 0,
                    "testing_size": 0,
                    "model_path": None,
                    "metadata_path": None,
                    "dataset_path": None,
                    "evaluation_metrics": None,
                    "model_version": None,
                }
            )

        dataset_path = save_historical_dataset(dataset, output_dir=MODEL2_DATA_DIR)
        result = train_model2(dataset, model_dir=MODEL2_DIR, dataset_path=dataset_path)
        result["training_eligibility"] = (
            "TRAINED" if result["model2_status"] == "TRAINED" else "NOT_TRAINED"
        )
        result["missing_requirements"] = eligibility["missing_requirements"]
        result["training_performed"] = result["model2_status"] == "TRAINED"
        return finish(result)
    except (OSError, ValueError, TypeError, RuntimeError, EOFError) as error:
        return finish(
            {
                "model2_status": "FAILED",
                "training_eligibility": "ERROR",
                "missing_requirements": [],
                "reason": f"Model 2 training failed: {error}",
                "error_type": type(error).__name__,
                "training_performed": False,
                "historical_dataset": None,
                "training_size": 0,
                "testing_size": 0,
                "model_path": None,
                "metadata_path": None,
                "dataset_path": None,
                "evaluation_metrics": None,
                "model_version": None,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare historical monitoring data and train Model 2 when eligible."
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        dataset = build_historical_dataset()
        dataset_path = save_historical_dataset(dataset, output_dir=MODEL2_DATA_DIR)
        result = {
            "model2_status": "PREPARED",
            "historical_dataset": dataset_summary(dataset),
            "dataset_path": str(dataset_path),
        }
    else:
        result = run_training_if_eligible()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
