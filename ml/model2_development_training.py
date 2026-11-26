from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.model2_config import DEGRADATION_TARGET, MODEL2_FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_DIR = ROOT / "ml" / "models" / "model2" / "development"
DEVELOPMENT_LABEL = "SYNTHETIC_DEVELOPMENT_ONLY"
DEVELOPMENT_PROVENANCE_COLUMN = "data_provenance"
TEST_FRACTION = 0.25


def generate_synthetic_development_dataset(
    *,
    sample_count: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    if sample_count < 20:
        raise ValueError("sample_count must be at least 20 for a stratified holdout.")
    if random_state < 0:
        raise ValueError("random_state must be non-negative.")

    rng = np.random.default_rng(random_state)
    drift_score = rng.beta(1.5, 6.0, sample_count) * 0.5
    max_feature_drift = np.clip(
        drift_score + rng.normal(0.0, 0.08, sample_count), 0.0, 0.5
    )
    missing_percentage = rng.beta(1.5, 10.0, sample_count) * 30.0
    invalid_percentage = rng.beta(1.2, 14.0, sample_count) * 15.0
    historical_f1 = rng.uniform(0.62, 0.98, sample_count)
    f1_trend = np.clip(rng.normal(0.0, 0.07, sample_count), -0.25, 0.25)
    metrics_available = rng.binomial(1, 0.85, sample_count)

    features = pd.DataFrame(
        {
            "dataset_drift_score": drift_score,
            "max_feature_drift_score": max_feature_drift,
            "number_drifted_features": rng.poisson(1.5, sample_count).clip(0, 12),
            "missing_value_percentage": missing_percentage,
            "invalid_value_percentage": invalid_percentage,
            "data_quality_score": np.maximum(
                0.0, 100.0 - missing_percentage - invalid_percentage
            ),
            "duplicate_rate": rng.beta(1.0, 30.0, sample_count) * 0.2,
            "prediction_positive_rate": rng.beta(2.0, 5.0, sample_count),
            "prediction_positive_rate_change": np.clip(
                rng.normal(0.0, 0.08, sample_count), -0.3, 0.3
            ),
            "prediction_mean_probability": rng.beta(3.0, 3.0, sample_count),
            "historical_accuracy": rng.uniform(0.62, 0.99, sample_count),
            "historical_precision": rng.uniform(0.55, 0.99, sample_count),
            "historical_recall": rng.uniform(0.50, 0.99, sample_count),
            "historical_f1_score": historical_f1,
            "previous_f1_trend": f1_trend,
            "historical_metrics_available": metrics_available,
            "historical_performance_observations": rng.poisson(
                12, sample_count
            ).clip(0, 100),
        },
        columns=MODEL2_FEATURE_NAMES,
    )
    for metric_name in (
        "historical_accuracy",
        "historical_precision",
        "historical_recall",
        "historical_f1_score",
        "previous_f1_trend",
    ):
        features.loc[metrics_available == 0, metric_name] = np.nan

    degradation_logit = (
        -3.0
        + 10.0 * drift_score
        + 0.15 * missing_percentage
        + 0.20 * invalid_percentage
        + 10.0 * np.maximum(-f1_trend, 0.0)
        + 7.0 * (0.82 - historical_f1)
    )
    degradation_probability = 1.0 / (1.0 + np.exp(-degradation_logit))
    labels = rng.binomial(1, degradation_probability)

    if np.unique(labels).size != 2:
        raise ValueError(
            "Synthetic labels contain only one class; increase sample_count "
            "or use another random_state."
        )

    dataset = features.copy()
    dataset[DEGRADATION_TARGET] = labels.astype(int)
    dataset[DEVELOPMENT_PROVENANCE_COLUMN] = DEVELOPMENT_LABEL
    return dataset


def evaluate_synthetic_development_model(
    dataset: pd.DataFrame,
    *,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    expected_columns = {
        *MODEL2_FEATURE_NAMES,
        DEGRADATION_TARGET,
        DEVELOPMENT_PROVENANCE_COLUMN,
    }
    if set(dataset.columns) != expected_columns:
        raise ValueError("Development dataset columns do not match the Model 2 schema.")
    if not dataset[DEVELOPMENT_PROVENANCE_COLUMN].eq(DEVELOPMENT_LABEL).all():
        raise ValueError(
            "Development training accepts only SYNTHETIC_DEVELOPMENT_ONLY rows."
        )

    features = dataset.loc[:, MODEL2_FEATURE_NAMES].apply(
        pd.to_numeric, errors="coerce"
    )
    labels = pd.to_numeric(dataset[DEGRADATION_TARGET], errors="raise").astype(int)
    if not labels.isin([0, 1]).all() or labels.nunique() != 2:
        raise ValueError("Development labels must contain both binary classes 0 and 1.")

    class_counts = labels.value_counts()
    stratify_labels = (
        labels if len(class_counts) == 2 and int(class_counts.min()) >= 2 else None
    )
    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=TEST_FRACTION,
        random_state=random_state,
        stratify=stratify_labels,
    )
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)
    metrics = {
        "data_provenance": DEVELOPMENT_LABEL,
        "holdout_method": (
            "stratified random holdout"
            if stratify_labels is not None
            else "random holdout (stratification not possible)"
        ),
        "test_fraction": TEST_FRACTION,
        "random_state": random_state,
        "train_size": int(len(train_labels)),
        "test_size": int(len(test_labels)),
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "precision": float(
            precision_score(test_labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(test_labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(test_labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            test_labels, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            test_labels,
            predictions,
            labels=[0, 1],
            target_names=["stable", "degraded"],
            output_dict=True,
            zero_division=0,
        ),
    }
    return model, metrics


def evaluate_chronological_development_model(
    dataset: pd.DataFrame,
    *,
    random_state: int = 42,
    test_fraction: float = TEST_FRACTION,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    if len(dataset) < 20:
        raise ValueError("Chronological simulation training requires at least 20 batches.")
    required_columns = {
        *MODEL2_FEATURE_NAMES,
        DEGRADATION_TARGET,
        DEVELOPMENT_PROVENANCE_COLUMN,
        "batch_id",
        "monitoring_timestamp",
        "data_source_type",
        "source",
        "environment",
        "is_synthetic",
    }
    if not required_columns.issubset(dataset.columns):
        raise ValueError(
            "Chronological development data is missing required feature, "
            "timestamp, or provenance columns."
        )
    if not dataset[DEVELOPMENT_PROVENANCE_COLUMN].eq(DEVELOPMENT_LABEL).all():
        raise ValueError(
            "Development training accepts only SYNTHETIC_DEVELOPMENT_ONLY rows."
        )
    if (
        not dataset["data_source_type"].eq("synthetic").all()
        or not dataset["source"].eq("synthetic_simulator").all()
        or not dataset["environment"].eq("development").all()
        or not pd.api.types.is_bool_dtype(dataset["is_synthetic"])
        or not dataset["is_synthetic"].all()
    ):
        raise ValueError(
            "Chronological development training accepts only explicitly tagged "
            "synthetic simulator records."
        )
    if dataset["batch_id"].isna().any() or dataset["batch_id"].duplicated().any():
        raise ValueError("Development training requires one row per distinct batch.")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if random_state < 0:
        raise ValueError("random_state must be non-negative.")

    ordered = dataset.copy()
    ordered["monitoring_timestamp"] = pd.to_datetime(
        ordered["monitoring_timestamp"], utc=True, errors="coerce"
    )
    if ordered["monitoring_timestamp"].isna().any():
        raise ValueError("Development monitoring timestamps must be valid.")
    ordered = ordered.sort_values(
        ["monitoring_timestamp", "batch_id"], kind="stable"
    ).reset_index(drop=True)
    if ordered["monitoring_timestamp"].duplicated().any():
        raise ValueError("Development monitoring timestamps must be strictly increasing.")

    labels = pd.to_numeric(
        ordered[DEGRADATION_TARGET], errors="coerce"
    )
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Development targets must contain only binary 0/1 labels.")
    labels = labels.astype(int)
    if labels.nunique() != 2:
        raise ValueError("Chronological development data must contain both target classes.")

    split_index = int(np.floor(len(ordered) * (1.0 - test_fraction)))
    if split_index < 2 or split_index >= len(ordered):
        raise ValueError("Not enough chronological batches for train and holdout sets.")
    train_labels = labels.iloc[:split_index]
    test_labels = labels.iloc[split_index:]
    if train_labels.nunique() != 2 or test_labels.nunique() != 2:
        raise ValueError(
            "Both target classes must be represented in chronological training "
            "and holdout partitions."
        )

    features = ordered.loc[:, MODEL2_FEATURE_NAMES].apply(
        pd.to_numeric, errors="coerce"
    )
    train_features = features.iloc[:split_index]
    test_features = features.iloc[split_index:]
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)
    probabilities = model.predict_proba(test_features)
    positive_index = list(model.classes_).index(1)
    holdout = pd.DataFrame(
        {
            "batch_id": ordered.loc[split_index:, "batch_id"].to_numpy(),
            "monitoring_timestamp": ordered.loc[
                split_index:, "monitoring_timestamp"
            ].astype(str).to_numpy(),
            "actual_performance_degraded": test_labels.to_numpy(),
            "predicted_performance_degraded": predictions.astype(int),
            "degradation_probability": probabilities[:, positive_index],
            "data_source_type": "synthetic",
            "source": "synthetic_simulator",
            "environment": "development",
            "is_synthetic": True,
        }
    )
    metrics = {
        "data_provenance": DEVELOPMENT_LABEL,
        "holdout_method": "chronological holdout",
        "test_fraction": test_fraction,
        "random_state": random_state,
        "train_size": int(len(train_labels)),
        "test_size": int(len(test_labels)),
        "train_start_timestamp": ordered.loc[0, "monitoring_timestamp"].isoformat(),
        "train_end_timestamp": ordered.loc[
            split_index - 1, "monitoring_timestamp"
        ].isoformat(),
        "test_start_timestamp": ordered.loc[
            split_index, "monitoring_timestamp"
        ].isoformat(),
        "test_end_timestamp": ordered.loc[
            len(ordered) - 1, "monitoring_timestamp"
        ].isoformat(),
        "training_class_counts": {
            str(label): int(count)
            for label, count in train_labels.value_counts().sort_index().items()
        },
        "testing_class_counts": {
            str(label): int(count)
            for label, count in test_labels.value_counts().sort_index().items()
        },
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "precision": float(
            precision_score(test_labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(test_labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(test_labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            test_labels, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            test_labels,
            predictions,
            labels=[0, 1],
            target_names=["stable", "degraded"],
            output_dict=True,
            zero_division=0,
        ),
    }
    return model, metrics, holdout


def run_chronological_development_training(
    dataset: pd.DataFrame,
    *,
    random_state: int,
    run_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in run_id):
        raise ValueError("run_id must contain only letters, digits, hyphens, and underscores.")
    model, metrics, holdout = evaluate_chronological_development_model(
        dataset,
        random_state=random_state,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    version = f"model2-development-simulation-{run_id}"
    dataset_path = output_dir / "synthetic_model2_dataset.csv"
    model_path = output_dir / f"{version}.joblib"
    metrics_path = output_dir / f"{version}-metrics.json"
    metadata_path = output_dir / f"{version}.json"
    holdout_path = output_dir / "chronological_holdout_predictions.csv"

    dataset.to_csv(dataset_path, index=False)
    joblib.dump(model, model_path)
    holdout.to_csv(holdout_path, index=False)
    _write_json(
        metrics_path,
        {
            "data_provenance": DEVELOPMENT_LABEL,
            "data_source_type": "synthetic",
            "source": "synthetic_simulator",
            "environment": "development",
            "is_synthetic": True,
            "evaluation_metrics": metrics,
        },
    )
    metadata = {
        "model2_status": "DEVELOPMENT_SIMULATION_TRAINED",
        "data_provenance": DEVELOPMENT_LABEL,
        "data_source_type": "synthetic",
        "source": "synthetic_simulator",
        "environment": "development",
        "is_synthetic": True,
        "model_name": "Model 2 Synthetic Hotel Development Simulation",
        "model_version": version,
        "algorithm": "RandomForestClassifier",
        "feature_names": MODEL2_FEATURE_NAMES,
        "target_name": DEGRADATION_TARGET,
        "random_state": random_state,
        "sample_count": len(dataset),
        "training_size": metrics["train_size"],
        "testing_size": metrics["test_size"],
        "split_method": metrics["holdout_method"],
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "holdout_predictions_path": str(holdout_path),
        "evaluation_metrics": metrics,
        "promotion_policy": "Never promote synthetic development artifacts to production.",
    }
    _write_json(metadata_path, metadata)
    return {**metadata, "metadata_path": str(metadata_path)}


def run_development_training(
    *,
    sample_count: int = 1000,
    random_state: int = 42,
    output_dir: Path = DEVELOPMENT_DIR,
) -> dict[str, Any]:
    dataset = generate_synthetic_development_dataset(
        sample_count=sample_count,
        random_state=random_state,
    )
    model, metrics = evaluate_synthetic_development_model(
        dataset,
        random_state=random_state,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    version = f"model2-development-random-forest-seed-{random_state}"
    dataset_path = output_dir / f"synthetic_development_dataset_seed-{random_state}.csv"
    model_path = output_dir / f"{version}.joblib"
    metrics_path = output_dir / f"{version}-metrics.json"
    metadata_path = output_dir / f"{version}.json"

    dataset.to_csv(dataset_path, index=False)
    joblib.dump(model, model_path)
    _write_json(
        metrics_path,
        {
            "data_provenance": DEVELOPMENT_LABEL,
            "evaluation_metrics": metrics,
        },
    )
    metadata = {
        "model2_status": "DEVELOPMENT_TRAINED",
        "data_provenance": DEVELOPMENT_LABEL,
        "model_name": "Model 2 Synthetic Development Experiment",
        "model_version": version,
        "algorithm": "RandomForestClassifier",
        "feature_names": MODEL2_FEATURE_NAMES,
        "target_name": DEGRADATION_TARGET,
        "synthetic_label_rule": (
            "A reproducible Bernoulli label sampled from a logistic risk function "
            "of drift, missing/invalid percentages, negative prior F1 trend, and "
            "historical F1. This is a workflow fixture, not a production target."
        ),
        "random_state": random_state,
        "sample_count": sample_count,
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "evaluation_metrics": metrics,
        "promotion_policy": "Never automatically promote development artifacts.",
    }
    _write_json(metadata_path, metadata)
    return {
        **metadata,
        "metadata_path": str(metadata_path),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train an isolated Model 2 synthetic development experiment. "
            "Artifacts are never used by production prediction."
        )
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_development_training(
        sample_count=args.samples,
        random_state=args.seed,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
