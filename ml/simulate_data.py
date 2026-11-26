from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from ml.monitoring import build_monitoring_report, save_monitoring_report
from ml.performance_tracking import log_batch_predictions


ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES_PATH = ROOT / "ml" / "models" / "feature_names.json"
MODEL_METADATA_PATH = ROOT / "ml" / "models" / "model_metadata.json"
PIPELINE_PATH = ROOT / "ml" / "models" / "hotel_cancellation_pipeline.pkl"
REFERENCE_DATA_PATH = ROOT / "data" / "raw" / "hotel_bookings.csv"
OUTPUT_DIR = ROOT / "data" / "monitoring"
TARGET_COLUMN = "is_canceled"
SIMULATION_SOURCE = "original_hotel_booking_dataset_simulation"
LABEL_PROVENANCE = (
    "Copied from the sampled original dataset row; not a production outcome. "
    "Labels in feature-shifted scenarios remain source-row labels only."
)

BATCH_TYPES = ("normal", "drifted", "poor_quality")
DRIFTED_CATEGORICAL_VALUES = {
    "hotel": "Resort Hotel",
    "arrival_date_month": "December",
    "meal": "HB",
    "country": "GBR",
    "market_segment": "Direct",
    "distribution_channel": "Direct",
    "reserved_room_type": "D",
    "assigned_room_type": "D",
    "deposit_type": "Non Refund",
    "customer_type": "Transient-Party",
}


def _sample_reference(
    reference_data: pd.DataFrame,
    feature_names: Sequence[str],
    batch_size: int,
    rng: np.random.Generator,
    target_column: str | None = None,
) -> pd.DataFrame:
    required_columns = [*feature_names]
    if target_column is not None:
        required_columns.append(target_column)
    missing_features = [name for name in required_columns if name not in reference_data]
    if missing_features:
        raise ValueError(
            f"Reference data is missing required columns: {missing_features}"
        )
    if reference_data.empty:
        raise ValueError("Reference data must contain at least one row.")

    positions = rng.choice(len(reference_data), size=batch_size, replace=True)
    return (
        reference_data.loc[:, required_columns]
        .iloc[positions]
        .reset_index(drop=True)
        .copy()
    )


def _make_drifted_batch(
    reference_data: pd.DataFrame,
    batch: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    drifted = batch.copy()
    numeric_shifts = {
        "lead_time": (1.35, 20.0),
        "adr": (1.2, 15.0),
        "stays_in_week_nights": (1.0, 1.0),
    }
    for column, (scale, offset) in numeric_shifts.items():
        values = pd.to_numeric(drifted[column], errors="raise").astype(float)
        lower = float(reference_data[column].quantile(0.01))
        upper = float(reference_data[column].quantile(0.99))
        drifted[column] = (values * scale + offset).clip(lower=lower, upper=upper)

    for column, target_value in DRIFTED_CATEGORICAL_VALUES.items():
        available_values = reference_data[column].dropna().unique()
        if target_value not in available_values:
            raise ValueError(
                f"Drift target {target_value!r} is not present in reference feature {column!r}."
            )
        replace_rows = rng.random(len(drifted)) < 0.6
        drifted.loc[replace_rows, column] = target_value
    return drifted


def _make_poor_quality_batch(
    batch: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if len(batch) < 3:
        raise ValueError("batch_size must be at least 3 to include all poor-quality examples.")

    poor_quality = batch.copy()
    missing_count = max(1, int(round(len(poor_quality) * 0.02)))
    for column in ("children", "country", "market_segment", "adr"):
        rows = rng.choice(len(poor_quality), size=missing_count, replace=False)
        poor_quality.loc[rows, column] = pd.NA

    poor_quality.loc[0, "lead_time"] = -999
    poor_quality.loc[1, "market_segment"] = "UNEXPECTED_MARKET_SEGMENT"
    poor_quality["adr"] = poor_quality["adr"].astype(object)
    poor_quality.loc[2, "adr"] = "not-a-number"
    return poor_quality


def generate_batches(
    reference_data: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    batch_size: int = 1000,
    seed: int = 42,
    target_column: str | None = None,
) -> dict[str, pd.DataFrame]:
    if batch_size < 3:
        raise ValueError("batch_size must be at least 3.")
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names must be a non-empty sequence of unique names.")

    child_seeds = np.random.SeedSequence(seed).spawn(len(BATCH_TYPES))
    generators = [np.random.default_rng(child_seed) for child_seed in child_seeds]
    normal = _sample_reference(
        reference_data,
        feature_names,
        batch_size,
        generators[0],
        target_column,
    )
    drifted_sample = _sample_reference(
        reference_data, feature_names, batch_size, generators[1], target_column
    )
    poor_quality_sample = _sample_reference(
        reference_data, feature_names, batch_size, generators[2], target_column
    )

    return {
        "normal": normal,
        "drifted": _make_drifted_batch(reference_data, drifted_sample, generators[1]),
        "poor_quality": _make_poor_quality_batch(
            poor_quality_sample, generators[2]
        ),
    }


def save_batches(
    batches: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    seed: int,
    as_of: datetime | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc)
    timestamp_text = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    generated_at = created_at.isoformat()
    period_end = as_of or created_at
    if period_end.tzinfo is None:
        period_end = period_end.replace(tzinfo=timezone.utc)
    period_end = period_end.astimezone(timezone.utc)
    paths = {
        batch_type: output_dir / f"{batch_type}_{timestamp_text}.csv"
        for batch_type in BATCH_TYPES
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite simulation batches: {existing}")

    for batch_index, (batch_type, batch) in enumerate(batches.items()):
        saved = batch.copy()
        batch_id = f"{batch_type}-{timestamp_text}-seed-{seed}"
        saved["batch_id"] = batch_id
        saved["generated_at"] = generated_at
        saved["timestamp"] = (
            period_end - timedelta(days=14 - 7 * batch_index)
        ).isoformat()
        saved["source"] = SIMULATION_SOURCE
        saved["data_source_type"] = "synthetic_test"
        saved["record_count"] = len(batch)
        saved["monitoring_scenario"] = batch_type
        if TARGET_COLUMN in saved.columns:
            saved["label_provenance"] = LABEL_PROVENANCE
        saved.to_csv(paths[batch_type], index=False)
    return paths


def simulate_data(
    *,
    batch_size: int = 1000,
    seed: int = 42,
    reference_path: Path = REFERENCE_DATA_PATH,
    feature_names_path: Path = FEATURE_NAMES_PATH,
    pipeline_path: Path = PIPELINE_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
    output_dir: Path = OUTPUT_DIR,
    as_of: datetime | None = None,
) -> dict[str, object]:
    feature_names = json.loads(feature_names_path.read_text(encoding="utf-8"))
    pipeline = joblib.load(pipeline_path)
    preprocessor = pipeline.named_steps["preprocessor"]
    if feature_names != preprocessor.feature_names:
        raise ValueError("feature_names.json does not match the saved pipeline feature order.")
    model_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if model_metadata.get("feature_names") != feature_names:
        raise ValueError(
            "model_metadata.json feature order does not match feature_names.json."
        )

    reference_data = pd.read_csv(
        reference_path,
        usecols=lambda column: column in {*feature_names, TARGET_COLUMN},
    ).loc[:, [*feature_names, TARGET_COLUMN]]
    batches = generate_batches(
        reference_data,
        feature_names,
        batch_size=batch_size,
        seed=seed,
        target_column=TARGET_COLUMN,
    )

    paths = save_batches(batches, output_dir, seed=seed, as_of=as_of)
    reports_dir = output_dir / "reports"
    predictions_dir = output_dir / "predictions"
    numerical_features = model_metadata["numerical_features"]
    categorical_features = model_metadata["categorical_features"]
    monitoring_results: dict[str, dict[str, object]] = {}
    valid_batch_predictions: dict[str, int] = {}

    for batch_type, path in paths.items():
        batch = pd.read_csv(path)
        report = build_monitoring_report(
            batch,
            reference_data.loc[:, feature_names],
            feature_names,
            numerical_features,
            categorical_features,
            data_source_type="synthetic_test",
        )
        print(
    f"\nDEBUG {batch_type}: "
    f"quality={report['quality']['quality_status']}, "
    
)
        prediction_result: dict[str, object] | None = None
        if report["quality"]["quality_status"] != "CRITICAL":
            model_input = batch.loc[:, feature_names].copy()
            model_input.insert(0, "batch_id", report["batch_id"])
            model_input["generated_at"] = batch["generated_at"]
            prediction_result = log_batch_predictions(
                model_input,
                source="synthetic_test",
                pipeline_path=pipeline_path,
                feature_names_path=feature_names_path,
                metadata_path=metadata_path,
                reference_path=reference_path,
                predictions_dir=predictions_dir,
                reports_dir=reports_dir,
            )
            if prediction_result["prediction_count"] != batch_size:
                raise RuntimeError(
                    f"Model 1 returned {prediction_result['prediction_count']} "
                    f"predictions for {batch_size} {batch_type} rows."
                )
            valid_batch_predictions[batch_type] = int(
                prediction_result["prediction_count"]
            )
            report["model_prediction"] = {
                "performed": True,
                "prediction_timestamp": prediction_result["prediction_timestamp"],
                "prediction_count": prediction_result["prediction_count"],
                "model_version": prediction_result["model_version"],
                "prediction_file": Path(
                    prediction_result["prediction_file"]
                ).name,
            }
        else:
            report["model_prediction"] = {
                "performed": False,
                "reason": (
                    "Model 1 prediction was skipped because the batch has "
                    "critical data-quality issues."
                ),
            }

        report_path = save_monitoring_report(report, reports_dir)
        monitoring_results[batch_type] = {
            "batch_id": report["batch_id"],
            "report_path": str(report_path),
            "quality_status": report["quality"]["quality_status"],
            "overall_status": report["overall_status"],
            "model_prediction": report["model_prediction"],
            "drifted_feature_count": sum(
                result.get("psi_level") in {"moderate", "significant"}
                for result in report["drift"]["numerical"].values()
                if result.get("usable")
            )
            + sum(
                result.get("drift_level") in {"moderate", "significant"}
                for result in report["drift"]["categorical"].values()
                if result.get("usable")
            ),
        }

    return {
        "feature_names": feature_names,
        "batch_size": batch_size,
        "seed": seed,
        "files": {name: str(path) for name, path in paths.items()},
        "batch_ids": {
            name: f"{name}-{Path(path).stem[len(name) + 1:]}-seed-{seed}"
            for name, path in paths.items()
        },
        "record_counts": {name: len(batch) for name, batch in batches.items()},
        "source": SIMULATION_SOURCE,
        "monitoring": monitoring_results,
        "pipeline_prediction_counts": valid_batch_predictions,
        "poor_quality_validation": (
            "Generated with intentional data-quality issues; saved as a monitoring "
            "report and excluded from Model 1 predictions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hotel booking monitoring batches.")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--as-of",
        type=lambda value: datetime.fromisoformat(value.replace("Z", "+00:00")),
        default=None,
        help="UTC reference time for the simulated batch periods.",
    )
    args = parser.parse_args()
    result = simulate_data(
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
        as_of=args.as_of,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
