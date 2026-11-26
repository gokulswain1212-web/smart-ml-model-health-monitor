from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = ROOT / "ml" / "models" / "model_metadata.json"
MODEL_PATH = ROOT / "ml" / "models" / "hotel_cancellation_model.pkl"
PIPELINE_PATH = ROOT / "ml" / "models" / "hotel_cancellation_pipeline.pkl"
DATASET_PATH = ROOT / "data" / "processed" / "cleane_hotel.csv"
OUTPUT_PATH = ROOT / "ml" / "models" / "model1_evaluation_visualizations.json"


def build_evaluation_visualizations() -> dict[str, Any]:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    split = metadata["training_configuration"]["train_test_split"]
    feature_names = metadata["feature_names"]
    dataset = pd.read_csv(DATASET_PATH)
    features = dataset.loc[:, feature_names]
    target = dataset[metadata["target_column"]]
    _, test_features, _, test_target = train_test_split(
        features,
        target,
        test_size=float(split["test_size"]),
        random_state=int(split["random_state"]),
        stratify=split.get("stratify"),
    )
    if len(test_target) != int(split["test_rows"]):
        raise ValueError("Held-out dataset size does not match model metadata.")

    pipeline = joblib.load(PIPELINE_PATH)
    model = joblib.load(MODEL_PATH)
    classes = list(pipeline.classes_)
    if 1 not in classes:
        raise ValueError("Saved model does not expose the cancellation class.")
    probabilities = np.asarray(pipeline.predict_proba(test_features), dtype=float)[
        :, classes.index(1)
    ]
    predictions = pipeline.predict(test_features)
    actual_confusion = confusion_matrix(test_target, predictions, labels=[0, 1])
    if actual_confusion.tolist() != metadata["evaluation_metrics"]["confusion_matrix"]:
        raise ValueError("Recomputed held-out predictions do not match model metadata.")

    month_names = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }
    arrival_dates = pd.to_datetime(
        {
            "year": pd.to_numeric(
                test_features["arrival_date_year"], errors="coerce"
            ),
            "month": test_features["arrival_date_month"].map(month_names),
            "day": pd.to_numeric(
                test_features["arrival_date_day_of_month"], errors="coerce"
            ),
        },
        errors="coerce",
    )
    if arrival_dates.isna().any():
        raise ValueError("Held-out rows contain invalid arrival dates.")

    history_frame = pd.DataFrame(
        {
            "arrival_date": arrival_dates.to_numpy(),
            "actual": test_target.to_numpy(),
            "predicted": predictions,
        }
    )
    history_frame["period"] = history_frame["arrival_date"].dt.strftime("%b %Y")
    history = []
    for period, group in history_frame.groupby("period", sort=False):
        labels = group["actual"]
        estimates = group["predicted"]
        history.append(
            {
                "period": period,
                "sample_count": int(len(group)),
                "accuracy": float(accuracy_score(labels, estimates)),
                "precision": float(
                    precision_score(labels, estimates, zero_division=0)
                ),
                "recall": float(recall_score(labels, estimates, zero_division=0)),
                "f1_score": float(f1_score(labels, estimates, zero_division=0)),
            }
        )
    history.sort(
        key=lambda item: pd.to_datetime(item["period"], format="%b %Y")
    )

    false_positive_rates, true_positive_rates, _ = roc_curve(
        test_target, probabilities
    )
    sample_indices = np.linspace(
        0, len(false_positive_rates) - 1, num=51, dtype=int
    )
    feature_importances = getattr(model, "feature_importances_", None)
    if feature_importances is None or len(feature_importances) != len(feature_names):
        raise ValueError("Saved model does not expose matching feature importances.")

    return {
        "dataset_row_count": int(len(dataset)),
        "dataset_period": (
            f"{arrival_dates.min():%b %Y}–{arrival_dates.max():%b %Y}"
        ),
        "roc_auc": float(roc_auc_score(test_target, probabilities)),
        "roc_curve": [
            {
                "false_positive_rate": float(false_positive_rates[index]),
                "true_positive_rate": float(true_positive_rates[index]),
            }
            for index in sample_indices
        ],
        "feature_importance": [
            {"name": name, "importance": float(importance)}
            for name, importance in sorted(
                zip(feature_names, feature_importances, strict=True),
                key=lambda item: item[1],
                reverse=True,
            )[:5]
        ],
        "performance_history": history,
    }


def main() -> None:
    visualizations = build_evaluation_visualizations()
    OUTPUT_PATH.write_text(
        json.dumps(visualizations, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Saved held-out ROC-AUC {visualizations['roc_auc']:.3f} "
        f"and {len(visualizations['feature_importance'])} feature importances "
        f"to {OUTPUT_PATH}."
    )


if __name__ == "__main__":
    main()
