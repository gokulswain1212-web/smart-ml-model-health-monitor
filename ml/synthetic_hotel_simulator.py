from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ml.model2_config import DEGRADATION_TARGET
from ml.model2_development_training import (
    DEVELOPMENT_LABEL,
    run_chronological_development_training,
)
from ml.monitoring import build_monitoring_report

ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES_PATH = ROOT / "ml" / "models" / "feature_names.json"
MODEL_METADATA_PATH = ROOT / "ml" / "models" / "model_metadata.json"
MODEL1_PATH = ROOT / "ml" / "models" / "hotel_cancellation_pipeline.pkl"
REFERENCE_DATA_PATH = ROOT / "data" / "raw" / "hotel_bookings.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "development_simulation" / "runs"
MINIMUM_BATCH_COUNT = 20
SIMULATION_F1_DROP_THRESHOLD = 0.12
PROVENANCE = {
    "data_source_type": "synthetic",
    "source": "synthetic_simulator",
    "environment": "development",
    "is_synthetic": True,
}

_CANCELLATION_BASE_RATES = {
    "hotel": {"City Hotel": 0.42, "Resort Hotel": 0.28},
    "deposit": {"No Deposit": 0.35, "Non Refund": 0.72, "Refundable": 0.24},
    "market": {
        "Online TA": 0.43,
        "Offline TA/TO": 0.30,
        "Groups": 0.55,
        "Direct": 0.18,
        "Corporate": 0.12,
        "Complementary": 0.05,
    },
    "customer": {
        "Transient": 0.40,
        "Transient-Party": 0.31,
        "Group": 0.16,
        "Contract": 0.12,
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable.")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, allow_nan=False, default=_json_default) + "\n"
            )


def _safe_id(value: str) -> str:
    safe_value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    if not safe_value:
        raise ValueError("Generated identifiers must contain letters or digits.")
    return safe_value


def _load_model_metadata() -> dict[str, Any]:
    metadata = _read_json(MODEL_METADATA_PATH)
    feature_names = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
    if not isinstance(feature_names, list):
        raise ValueError("Model 1 feature_names.json must contain a JSON list.")
    if feature_names != metadata.get("feature_names"):
        raise ValueError("Model 1 feature metadata does not match feature_names.json.")
    if not MODEL1_PATH.is_file():
        raise FileNotFoundError(f"Model 1 artifact is missing: {MODEL1_PATH}")
    return metadata


def _load_reference_data(feature_names: list[str], seed: int) -> pd.DataFrame:
    if not REFERENCE_DATA_PATH.is_file():
        raise FileNotFoundError(f"Model 1 reference data is missing: {REFERENCE_DATA_PATH}")
    reference = pd.read_csv(REFERENCE_DATA_PATH, usecols=feature_names)
    if len(reference) > 10_000:
        reference = reference.sample(n=10_000, random_state=seed)
    return reference.reset_index(drop=True)


def generate_synthetic_booking_batch(
    *,
    batch_id: str,
    batch_index: int,
    batch_size: int,
    seed: int,
    timestamp: datetime,
) -> pd.DataFrame:
    if batch_size < 20:
        raise ValueError("batch_size must be at least 20.")
    if batch_index < 0:
        raise ValueError("batch_index must be non-negative.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware.")

    rng = np.random.default_rng(np.random.SeedSequence([seed, batch_index]))
    if batch_index < 5:
        shifted = False
    elif batch_index < 10:
        shifted = True
    elif batch_index < 15:
        shifted = False
    else:
        shifted = True
    high_lead_probability = 0.52 if shifted else 0.31
    non_refundable_probability = 0.20 if shifted else 0.10
    online_probability = 0.47 if shifted else 0.34
    hotels = rng.choice(
        ["City Hotel", "Resort Hotel"], size=batch_size, p=[0.66, 0.34]
    )
    deposit_types = rng.choice(
        ["No Deposit", "Non Refund", "Refundable"],
        size=batch_size,
        p=[
            1.0 - non_refundable_probability - 0.02,
            non_refundable_probability,
            0.02,
        ],
    )
    market_segments = rng.choice(
        [
            "Online TA",
            "Offline TA/TO",
            "Groups",
            "Direct",
            "Corporate",
            "Complementary",
        ],
        size=batch_size,
        p=[
            online_probability,
            0.20,
            0.13,
            0.12 if shifted else 0.25,
            0.07,
            0.01,
        ],
    )
    customer_types = rng.choice(
        ["Transient", "Transient-Party", "Group", "Contract"],
        size=batch_size,
        p=[0.57, 0.26, 0.10, 0.07] if not shifted else [0.72, 0.18, 0.06, 0.04],
    )
    room_types = np.array(list("ABCDEFGH"))
    room_probabilities = (
        [0.18, 0.08, 0.31, 0.30, 0.07, 0.03, 0.02, 0.01]
        if not shifted
        else [0.10, 0.05, 0.43, 0.29, 0.06, 0.03, 0.03, 0.01]
    )
    arrival = timestamp
    arrival_year = arrival.year
    arrival_month = arrival.month
    lead_time = np.clip(
        rng.gamma(shape=2.2 if shifted else 1.7, scale=65, size=batch_size),
        0,
        365,
    ).astype(int)
    long_lead_mask = rng.random(batch_size) < high_lead_probability
    lead_time[long_lead_mask] += rng.integers(
        45, 130, size=int(long_lead_mask.sum())
    )
    arrival_months = rng.choice(
        [arrival_month, arrival_month % 12 + 1, (arrival_month + 1) % 12 + 1],
        size=batch_size,
        p=[0.55, 0.25, 0.20],
    )
    arrival_days = rng.integers(1, 29, size=batch_size)
    arrival_dates = [
        datetime(arrival_year, int(month), int(day))
        for month, day in zip(arrival_months, arrival_days, strict=True)
    ]
    weekend_nights = np.clip(rng.poisson(1.0, size=batch_size), 0, 4)
    week_nights = np.clip(
        rng.poisson(2.4 if not shifted else 1.8, size=batch_size), 0, 14
    )
    adults = rng.choice([1, 2, 3, 4], size=batch_size, p=[0.18, 0.66, 0.12, 0.04])
    children = rng.choice([0, 1, 2], size=batch_size, p=[0.82, 0.13, 0.05])
    babies = rng.choice([0, 1], size=batch_size, p=[0.98, 0.02])
    meal = rng.choice(
        ["BB", "FB", "HB", "SC", "Undefined"],
        size=batch_size,
        p=[0.77, 0.01, 0.11, 0.10, 0.01],
    )
    countries = rng.choice(
        ["PRT", "GBR", "FRA", "ESP", "DEU", "IRL", "ITA", "NLD", "USA"],
        size=batch_size,
        p=[0.31, 0.13, 0.10, 0.09, 0.08, 0.06, 0.07, 0.06, 0.10],
    )
    is_repeated_guest = rng.binomial(1, 0.04, size=batch_size)
    previous_cancellations = rng.choice(
        [0, 1, 2], size=batch_size, p=[0.94, 0.05, 0.01]
    )
    previous_bookings = np.where(
        is_repeated_guest == 1, rng.integers(1, 8, size=batch_size), 0
    )
    reserved_room = rng.choice(
        room_types, size=batch_size, p=room_probabilities
    )
    assigned_room = reserved_room.copy()
    room_change = rng.random(batch_size) < (0.08 if not shifted else 0.13)
    assigned_room[room_change] = rng.choice(room_types, size=room_change.sum())
    booking_changes = rng.choice(
        [0, 1, 2, 3], size=batch_size, p=[0.77, 0.16, 0.05, 0.02]
    )
    agent = rng.choice(
        [1.0, 9.0, 240.0, 303.0, np.nan],
        size=batch_size,
        p=[0.45, 0.20, 0.15, 0.07, 0.13],
    )
    days_waiting = rng.choice(
        [0, 1, 5, 15, 30], size=batch_size, p=[0.91, 0.03, 0.025, 0.02, 0.015]
    )
    base_adr = np.where(hotels == "City Hotel", 105.0, 82.0)
    adr = np.clip(
        rng.normal(base_adr + (12 if shifted else 0), 34, size=batch_size),
        0,
        320,
    ).round(2)
    parking = rng.choice([0, 1, 2], size=batch_size, p=[0.81, 0.17, 0.02])
    special_requests = rng.choice(
        [0, 1, 2, 3, 4], size=batch_size, p=[0.48, 0.32, 0.15, 0.04, 0.01]
    )

    batch = pd.DataFrame(
        {
            "hotel": hotels,
            "lead_time": lead_time,
            "arrival_date_year": arrival_year,
            "arrival_date_month": [
                datetime(2000, int(month), 1).strftime("%B")
                for month in arrival_months
            ],
            "arrival_date_week_number": [
                int(date.strftime("%U")) for date in arrival_dates
            ],
            "arrival_date_day_of_month": arrival_days,
            "stays_in_weekend_nights": weekend_nights,
            "stays_in_week_nights": week_nights,
            "adults": adults,
            "children": children,
            "babies": babies,
            "meal": meal,
            "country": countries,
            "market_segment": market_segments,
            "distribution_channel": np.where(
                market_segments == "Online TA", "TA/TO", "Direct"
            ),
            "is_repeated_guest": is_repeated_guest,
            "previous_cancellations": previous_cancellations,
            "previous_bookings_not_canceled": previous_bookings,
            "reserved_room_type": reserved_room,
            "assigned_room_type": assigned_room,
            "booking_changes": booking_changes,
            "deposit_type": deposit_types,
            "agent": agent,
            "days_in_waiting_list": days_waiting,
            "customer_type": customer_types,
            "adr": adr,
            "required_car_parking_spaces": parking,
            "total_of_special_requests": special_requests,
        }
    )
    base_risk = np.array(
        [
            _CANCELLATION_BASE_RATES["hotel"][hotel]
            + 0.12 * (lead > 90)
            + 0.20 * (_CANCELLATION_BASE_RATES["deposit"].get(deposit, 0.35) > 0.5)
            + 0.12 * (_CANCELLATION_BASE_RATES["market"].get(segment, 0.25) > 0.4)
            + 0.08 * (_CANCELLATION_BASE_RATES["customer"].get(customer, 0.3) > 0.35)
            - 0.04 * requests
            - 0.10 * repeated
            for hotel, lead, deposit, segment, customer, requests, repeated in zip(
                hotels,
                lead_time,
                deposit_types,
                market_segments,
                customer_types,
                special_requests,
                is_repeated_guest,
                strict=True,
            )
        ],
        dtype=float,
    )
    if shifted:
        base_risk += 0.06
    base_risk = np.clip(base_risk, 0.02, 0.92)
    batch["is_canceled"] = rng.binomial(1, base_risk)
    if batch["is_canceled"].nunique() != 2:
        batch.loc[batch.index[0], "is_canceled"] = 0
        batch.loc[batch.index[1], "is_canceled"] = 1

    batch.insert(0, "booking_id", [f"{batch_id}-booking-{i + 1:05d}" for i in range(batch_size)])
    batch.insert(1, "batch_id", batch_id)
    batch["timestamp"] = timestamp.isoformat()
    batch["monitoring_timestamp"] = timestamp.isoformat()
    batch["behavior_regime"] = (
        "baseline"
        if batch_index < 5
        else "behavior_shift"
        if batch_index < 10
        else "recovery"
        if batch_index < 15
        else "recurring_degradation"
    )
    for field, value in PROVENANCE.items():
        batch[field] = value
    return batch


def _performance_regime(
    batch_index: int,
) -> tuple[str, float, float]:
    if batch_index < 5:
        return "baseline", 0.03, 0.05
    if batch_index < 10:
        return "behavior_shift", 0.30, 0.12
    if batch_index < 15:
        return "recovery", 0.04, 0.07
    return "recurring_degradation", 0.38 if batch_index % 2 == 0 else 0.05, 0.10


def _ensure_binary_labels(labels: np.ndarray) -> np.ndarray:
    result = labels.astype(int, copy=True)
    if np.unique(result).size != 2 and result.size >= 2:
        result[0] = 0
        result[1] = 1
    return result


def _monitoring_metrics(
    report: dict[str, Any],
    features: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    previous_metrics: list[dict[str, float]],
) -> dict[str, Any]:
    numerical = report.get("drift", {}).get("numerical", {})
    categorical = report.get("drift", {}).get("categorical", {})
    drift_scores = [
        float(result["psi"])
        for result in numerical.values()
        if result.get("usable") and result.get("psi") is not None
    ] + [
        float(result["drift_score"])
        for result in categorical.values()
        if result.get("usable") and result.get("drift_score") is not None
    ]
    drifted_count = sum(score >= 0.1 for score in drift_scores)
    quality = report.get("quality", {})
    total_cells = max(1, len(features) * len(features.columns))
    missing_counts = quality.get("missing_value_counts", {})
    invalid_counts = quality.get("invalid_numerical_value_counts", {})
    missing_percentage = 100.0 * sum(missing_counts.values()) / total_cells
    invalid_percentage = 100.0 * sum(invalid_counts.values()) / total_cells
    duplicate_rate = float(quality.get("duplicate_rate", 0.0))
    history = previous_metrics
    latest = history[-1] if history else {}
    prior = history[-2] if len(history) > 1 else {}
    return {
        "dataset_drift_score": float(np.mean(drift_scores)) if drift_scores else 0.0,
        "max_feature_drift_score": max(drift_scores, default=0.0),
        "number_drifted_features": int(drifted_count),
        "missing_value_percentage": float(missing_percentage),
        "invalid_value_percentage": float(invalid_percentage),
        "data_quality_score": float(
            max(0.0, 100.0 - missing_percentage - invalid_percentage)
        ),
        "duplicate_rate": duplicate_rate,
        "prediction_positive_rate": float(np.mean(predictions)),
        "prediction_positive_rate_change": (
            float(np.mean(predictions) - latest.get("prediction_positive_rate", 0.0))
            if latest
            else 0.0
        ),
        "prediction_mean_probability": float(np.mean(probabilities)),
        "historical_accuracy": latest.get("accuracy"),
        "historical_precision": latest.get("precision"),
        "historical_recall": latest.get("recall"),
        "historical_f1_score": latest.get("f1_score"),
        "previous_f1_trend": (
            float(latest["f1_score"] - prior["f1_score"])
            if latest and prior
            else None
        ),
        "historical_metrics_available": int(bool(latest)),
        "historical_performance_observations": len(history),
    }


def run_synthetic_hotel_simulation(
    *,
    batch_count: int = 24,
    batch_size: int = 160,
    seed: int = 42,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    start_at: datetime | None = None,
) -> dict[str, Any]:
    if batch_count < MINIMUM_BATCH_COUNT:
        raise ValueError(
            f"batch_count must be at least {MINIMUM_BATCH_COUNT} for simulation."
        )
    if batch_size < 20:
        raise ValueError("batch_size must be at least 20.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    start = start_at or datetime(2026, 1, 5, tzinfo=timezone.utc)
    if start.tzinfo is None:
        raise ValueError("start_at must be timezone-aware.")

    metadata = _load_model_metadata()
    feature_names = metadata.get("feature_names")
    numerical_features = metadata.get("numerical_features")
    categorical_features = metadata.get("categorical_features")
    if not all(
        isinstance(value, list) and value
        for value in (feature_names, numerical_features, categorical_features)
    ):
        raise ValueError("Model 1 feature metadata is incomplete.")
    if feature_names != json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8")):
        raise ValueError("Model 1 feature metadata is inconsistent.")
    pipeline = joblib.load(MODEL1_PATH)
    reference = _load_reference_data(feature_names, seed)

    output_root.mkdir(parents=True, exist_ok=True)
    run_id = f"simulation-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    run_dir = output_root / run_id
    run_dir.mkdir()
    batches_dir = run_dir / "batches"
    reports_dir = run_dir / "reports"
    batches_dir.mkdir()
    reports_dir.mkdir()

    rng = np.random.default_rng(seed)
    batch_results: list[dict[str, Any]] = []
    predictions_jsonl: list[dict[str, Any]] = []
    outcomes_jsonl: list[dict[str, Any]] = []
    for batch_index in range(batch_count):
        timestamp = start.astimezone(timezone.utc) + timedelta(days=7 * batch_index)
        batch_id = _safe_id(f"{run_id}-batch-{batch_index + 1:03d}")
        booking_records = generate_synthetic_booking_batch(
            batch_id=batch_id,
            batch_index=batch_index,
            batch_size=batch_size,
            seed=seed,
            timestamp=timestamp,
        )
        features = booking_records.loc[:, feature_names].copy()
        raw_predictions = pipeline.predict(features).astype(int)
        class_labels = list(pipeline.classes_)
        if 1 not in class_labels:
            raise ValueError("Model 1 must expose class 1 for cancellation probability.")
        raw_probabilities = pipeline.predict_proba(features)[:, class_labels.index(1)]
        regime, prediction_error_rate, outcome_noise_rate = _performance_regime(
            batch_index
        )
        injected_error = rng.random(batch_size) < prediction_error_rate
        simulated_predictions = raw_predictions.copy()
        simulated_predictions[injected_error] = 1 - simulated_predictions[injected_error]
        actual_outcomes = raw_predictions.copy()
        outcome_flip = rng.random(batch_size) < outcome_noise_rate
        actual_outcomes[outcome_flip] = 1 - actual_outcomes[outcome_flip]
        actual_outcomes = _ensure_binary_labels(actual_outcomes)
        row_ids = [
            f"{batch_id}-row-{index + 1:05d}" for index in range(batch_size)
        ]
        booking_records["row_id"] = row_ids
        booking_records["observation_id"] = row_ids
        booking_records["model1_raw_predicted_class"] = raw_predictions
        booking_records["predicted_class"] = simulated_predictions
        booking_records["probability_positive"] = raw_probabilities
        booking_records["prediction_error_injected"] = injected_error
        booking_records["is_canceled"] = actual_outcomes
        booking_records["label_provenance"] = "synthetic_simulator_outcome"
        booking_records["behavior_regime"] = regime
        booking_records.to_csv(batches_dir / f"{batch_id}.csv", index=False)

        report_batch = features.copy()
        report_batch["batch_id"] = batch_id
        report_batch["timestamp"] = timestamp.isoformat()
        report_batch["source"] = PROVENANCE["source"]
        report_batch["data_source_type"] = PROVENANCE["data_source_type"]
        report_batch["record_count"] = batch_size
        report_batch["monitoring_scenario"] = regime
        report = build_monitoring_report(
            report_batch,
            reference,
            feature_names,
            numerical_features,
            categorical_features,
            data_source_type="synthetic_test",
        )
        report.update(PROVENANCE)
        report["monitoring_timestamp"] = timestamp.isoformat()
        report["monitoring_scenario"] = regime
        report["model_prediction"] = {
            "performed": True,
            "model_version": metadata.get("model_version"),
            "prediction_count": batch_size,
            "simulation_error_injection_rate": prediction_error_rate,
            "simulated_prediction_error_count": int(injected_error.sum()),
            "actual_outcome_noise_rate": outcome_noise_rate,
        }
        report_path = reports_dir / f"{batch_id}.json"
        _write_json(report_path, report)

        for index, row_id in enumerate(row_ids):
            common = {
                **PROVENANCE,
                "batch_id": batch_id,
                "row_id": row_id,
                "timestamp": timestamp.isoformat(),
            }
            predictions_jsonl.append(
                {
                    **common,
                    "observation_id": row_id,
                    "booking_id": booking_records.iloc[index]["booking_id"],
                    "model_version": metadata.get("model_version"),
                    "predicted_class": int(simulated_predictions[index]),
                    "model1_raw_predicted_class": int(raw_predictions[index]),
                    "probability_positive": float(raw_probabilities[index]),
                    "prediction_error_injected": bool(injected_error[index]),
                }
            )
            outcomes_jsonl.append(
                {
                    **common,
                    "observation_id": row_id,
                    "booking_id": booking_records.iloc[index]["booking_id"],
                    "outcome_received_at": (timestamp + timedelta(days=1)).isoformat(),
                    "is_canceled": int(actual_outcomes[index]),
                    "label_provenance": "synthetic_simulator_outcome",
                }
            )

        metrics = {
            "accuracy": float(accuracy_score(actual_outcomes, simulated_predictions)),
            "precision": float(
                precision_score(
                    actual_outcomes, simulated_predictions, zero_division=0
                )
            ),
            "recall": float(
                recall_score(actual_outcomes, simulated_predictions, zero_division=0)
            ),
            "f1_score": float(
                f1_score(actual_outcomes, simulated_predictions, zero_division=0)
            ),
        }
        batch_results.append(
            {
                "batch_id": batch_id,
                "timestamp": timestamp,
                "outcome_received_at": timestamp + timedelta(days=1),
                "features": features,
                "predictions": simulated_predictions,
                "probabilities": raw_probabilities,
                "outcomes": actual_outcomes,
                "metrics": metrics,
                "report": report,
                "prediction_error_rate": prediction_error_rate,
                "regime": regime,
            }
        )

    _append_jsonl(run_dir / "predictions.jsonl", predictions_jsonl)
    _append_jsonl(run_dir / "outcomes.jsonl", outcomes_jsonl)

    baseline_f1 = float(batch_results[0]["metrics"]["f1_score"])
    model2_rows: list[dict[str, Any]] = []
    historical_metrics: list[dict[str, float]] = []
    for index, result in enumerate(batch_results):
        target = int(
            baseline_f1 - result["metrics"]["f1_score"]
            >= SIMULATION_F1_DROP_THRESHOLD
        )
        monitored = _monitoring_metrics(
            result["report"],
            result["features"],
            result["predictions"],
            result["probabilities"],
            historical_metrics,
        )
        model2_rows.append(
            {
                "batch_id": result["batch_id"],
                "monitoring_timestamp": result["timestamp"].isoformat(),
                "target_future_timestamp": result["outcome_received_at"].isoformat(),
                **PROVENANCE,
                "data_provenance": DEVELOPMENT_LABEL,
                "label_provenance": "synthetic_simulator_outcome",
                **monitored,
                DEGRADATION_TARGET: target,
            }
        )
        historical_metrics.append(result["metrics"])
    model2_dataset = pd.DataFrame(model2_rows)
    model2_result = run_chronological_development_training(
        model2_dataset,
        random_state=seed,
        run_id=run_id,
        output_dir=run_dir / "model2",
    )

    f1_values = [result["metrics"]["f1_score"] for result in batch_results]
    summary = {
        **PROVENANCE,
        "run_id": run_id,
        "run_status": "COMPLETE",
        "mode": "DEVELOPMENT SIMULATION",
        "seed": seed,
        "batch_count": batch_count,
        "batch_size": batch_size,
        "booking_count": batch_count * batch_size,
        "first_batch_timestamp": batch_results[0]["timestamp"].isoformat(),
        "last_batch_timestamp": batch_results[-1]["timestamp"].isoformat(),
        "baseline_f1_score": baseline_f1,
        "target_definition": (
            "Synthetic performance_degraded=1 when current batch F1 is at least "
            f"{SIMULATION_F1_DROP_THRESHOLD:.2f} below the first batch baseline F1."
        ),
        "minimum_batch_f1_score": min(f1_values),
        "maximum_batch_f1_score": max(f1_values),
        "degraded_batch_count": sum(
            row[DEGRADATION_TARGET] == 1 for row in model2_rows
        ),
        "stable_batch_count": sum(
            row[DEGRADATION_TARGET] == 0 for row in model2_rows
        ),
        "monitoring_status_counts": {
            status: sum(
                result["report"].get("overall_status") == status
                for result in batch_results
            )
            for status in ("HEALTHY", "WARNING", "CRITICAL")
        },
        "drift_detected_batch_count": sum(
            row["number_drifted_features"] > 0 for row in model2_rows
        ),
        "model2_training": {
            "status": model2_result["model2_status"],
            "model_version": model2_result["model_version"],
            "training_size": model2_result["training_size"],
            "testing_size": model2_result["testing_size"],
            "split_method": model2_result["split_method"],
            "training_class_counts": model2_result["evaluation_metrics"][
                "training_class_counts"
            ],
            "testing_class_counts": model2_result["evaluation_metrics"][
                "testing_class_counts"
            ],
            "evaluation_metrics": {
                name: model2_result["evaluation_metrics"][name]
                for name in ("accuracy", "precision", "recall", "f1_score")
            },
        },
        "paths": {
            "run_dir": str(run_dir),
            "batches_dir": str(batches_dir),
            "reports_dir": str(reports_dir),
            "predictions": str(run_dir / "predictions.jsonl"),
            "outcomes": str(run_dir / "outcomes.jsonl"),
            "model2_dataset": model2_result["dataset_path"],
            "model2_model": model2_result["model_path"],
            "model2_metadata": model2_result["metadata_path"],
        },
    }
    _write_json(run_dir / "simulation_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate isolated synthetic hotel booking batches and train/evaluate "
            "a development-only Model 2 experiment."
        )
    )
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = run_synthetic_hotel_simulation(
        batch_count=args.batches,
        batch_size=args.batch_size,
        seed=args.seed,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, allow_nan=False, default=_json_default))


if __name__ == "__main__":
    main()
