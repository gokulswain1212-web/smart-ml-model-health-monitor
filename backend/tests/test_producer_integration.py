from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient

import backend.main as main
from ml.performance_tracking import (
    TARGET_COLUMN,
    evaluate_batch_performance as evaluate_batch_performance_with_paths,
    record_production_outcome as record_production_outcome_with_paths,
)
from producer.contracts import WorkflowState
from producer.feature_mapping import FeatureMapper
from producer.identity import ProducerIdentity
from producer.observation_client import ObservationApiClient
from producer.outcome_client import OutcomeApiClient
from producer.schemas import CATEGORICAL_FEATURES, FEATURE_NAMES, BookingEvent, FinalOutcomeEvent
from producer.storage import SQLiteProducerStore
from producer.workflow import ProducerWorkflow

SYNTHETIC_TOKEN = "synthetic-integration-token-" + "t" * 32
SYNTHETIC_HMAC_KEY = "synthetic-integration-hmac-" + "k" * 32
BATCH_ID = "producer-integration-synthetic"


def _synthetic_features(offset: int) -> dict[str, object]:
    features: dict[str, object] = {}
    for name in FEATURE_NAMES:
        if name in CATEGORICAL_FEATURES:
            features[name] = {
                "hotel": "City Hotel",
                "arrival_date_month": "November",
                "meal": "BB",
                "country": "PRT",
                "market_segment": "Online TA",
                "distribution_channel": "TA/TO",
                "reserved_room_type": "A",
                "assigned_room_type": "A",
                "deposit_type": "No Deposit",
                "customer_type": "Transient",
            }[name]
        elif name == "adr":
            features[name] = 85.0 + offset
        elif name == "arrival_date_year":
            features[name] = 2025
        elif name == "arrival_date_week_number":
            features[name] = 45
        elif name == "arrival_date_day_of_month":
            features[name] = 15 + offset
        elif name == "lead_time":
            features[name] = 5 + offset
        elif name == "adults":
            features[name] = 2
        elif name == "agent":
            features[name] = 9
        else:
            features[name] = 0
    return features


def _synthetic_booking(index: int) -> BookingEvent:
    return BookingEvent(
        source_event_id=f"SYNTHETIC-BOOKING-EVENT-{index}",
        booking_id=f"SYNTHETIC-BOOKING-{index}",
        source_event_timestamp=datetime.now(timezone.utc) - timedelta(minutes=5),
        features=_synthetic_features(index),
    )


def test_producer_posts_synthetic_observations_and_final_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PRODUCTION_INGESTION_TOKEN", SYNTHETIC_TOKEN)
    monkeypatch.setattr(main, "PRODUCTION_INGESTION_EVENT_HISTORY", deque(maxlen=50))

    directories = {
        name: tmp_path / name
        for name in ("predictions", "outcomes", "performance", "reports")
    }
    for directory in directories.values():
        directory.mkdir()
    monkeypatch.setattr(main, "PREDICTIONS_DIR", directories["predictions"])

    feature_names = json.loads(main.FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
    assert feature_names == list(FEATURE_NAMES)
    backend_observation_ids: list[str] = []
    bridge_requests: list[str] = []

    def fake_log_batch_predictions(batch, *, source: str) -> dict[str, object]:
        assert source == "production"
        assert list(batch.columns) == [
            "batch_id",
            *feature_names,
            "row_id",
            "source_booking_id_hmac",
        ]
        assert batch["batch_id"].tolist() == [BATCH_ID, BATCH_ID]
        assert batch["row_id"].is_unique
        assert batch["source_booking_id_hmac"].str.fullmatch(r"[0-9a-f]{64}").all()

        predicted_at = datetime.now(timezone.utc).isoformat()
        predictions = []
        for row_number, row in enumerate(batch.to_dict(orient="records")):
            observation_id = f"synthetic-observation-{uuid4().hex}"
            backend_observation_ids.append(observation_id)
            predictions.append(
                {
                    "batch_id": BATCH_ID,
                    "row_id": row["row_id"],
                    "observation_id": observation_id,
                    "source": "production",
                    "predicted_at": predicted_at,
                    "predicted_class": row_number % 2,
                    "probability_positive": 0.25 + 0.5 * (row_number % 2),
                    "model_version": "synthetic-integration-model",
                    "model_artifact_sha256": "0" * 64,
                }
            )
        path = directories["predictions"] / f"{BATCH_ID}.jsonl"
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in predictions),
            encoding="utf-8",
        )
        return {
            "batch_id": BATCH_ID,
            "prediction_timestamp": predicted_at,
            "prediction_count": len(predictions),
            "observation_ids": [record["observation_id"] for record in predictions],
            "predictions": [
                {
                    "row_id": record["row_id"],
                    "observation_id": record["observation_id"],
                    "predicted_class": record["predicted_class"],
                    "probability_positive": record["probability_positive"],
                }
                for record in predictions
            ],
            "model_version": "synthetic-integration-model",
            "model_artifact_sha256": "0" * 64,
            "quality_status": "HEALTHY",
        }

    def fake_monitor_batch_data(
        batch,
        *,
        data_source_type: str,
        source: str,
    ) -> tuple[dict[str, object], Path]:
        assert source == "production"
        assert data_source_type == "real"
        batch_id = str(batch["batch_id"].iloc[0])
        report = {
            "batch_id": batch_id,
            "monitoring_timestamp": datetime.now(timezone.utc).isoformat(),
            "data_source_type": data_source_type,
            "source": source,
            "quality": {},
            "drift": {},
        }
        report_path = directories["reports"] / f"{batch_id}.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report, report_path

    monkeypatch.setattr(main, "log_batch_predictions", fake_log_batch_predictions)
    monkeypatch.setattr(main, "monitor_batch_data", fake_monitor_batch_data)
    monkeypatch.setattr(
        main,
        "evaluate_batch_performance",
        lambda batch_id: evaluate_batch_performance_with_paths(
            batch_id,
            predictions_dir=directories["predictions"],
            outcomes_dir=directories["outcomes"],
            performance_dir=directories["performance"],
            reports_dir=directories["reports"],
        ),
    )
    monkeypatch.setattr(
        main,
        "record_production_outcome",
        lambda observation_id, is_canceled: record_production_outcome_with_paths(
            observation_id,
            is_canceled,
            predictions_dir=directories["predictions"],
            outcomes_dir=directories["outcomes"],
            performance_dir=directories["performance"],
            reports_dir=directories["reports"],
        ),
    )
    training_hook_calls: list[bool] = []
    monkeypatch.setattr(
        main,
        "run_training_if_eligible",
        lambda: (
            training_hook_calls.append(True)
            or {"model2_status": "NOT_TRAINED", "training_performed": False}
        ),
    )

    with TestClient(main.app) as backend:

        def bridge(request: httpx.Request) -> httpx.Response:
            assert request.url.scheme == "https"
            assert request.url.host == "monitor.synthetic.invalid"
            assert request.headers["X-Production-Ingestion-Token"] == SYNTHETIC_TOKEN
            bridge_requests.append(request.url.path)
            backend_response = backend.post(
                request.url.path,
                content=request.content,
                headers={
                    "Content-Type": request.headers["Content-Type"],
                    "X-Production-Ingestion-Token": request.headers[
                        "X-Production-Ingestion-Token"
                    ],
                },
            )
            return httpx.Response(
                backend_response.status_code,
                headers={
                    "Content-Type": backend_response.headers.get(
                        "Content-Type", "application/json"
                    )
                },
                content=backend_response.content,
                request=request,
            )

        identity = ProducerIdentity(SYNTHETIC_HMAC_KEY)
        store = SQLiteProducerStore(tmp_path / "producer-state.sqlite3", identity)
        with (
            ObservationApiClient(
                "https://monitor.synthetic.invalid",
                SYNTHETIC_TOKEN,
                transport=httpx.MockTransport(bridge),
            ) as observation_client,
            OutcomeApiClient(
                "https://monitor.synthetic.invalid",
                SYNTHETIC_TOKEN,
                transport=httpx.MockTransport(bridge),
            ) as outcome_client,
        ):
            workflow = ProducerWorkflow(
                store=store,
                identity=identity,
                feature_mapper=FeatureMapper.identity(),
                observation_client=observation_client,
                outcome_client=outcome_client,
            )
            bookings = [_synthetic_booking(1), _synthetic_booking(2)]
            accepted = workflow.submit_observation_batch(BATCH_ID, bookings)

            assert len(accepted) == 2
            assert [item.state for item in accepted] == [
                WorkflowState.AWAITING_OUTCOME,
                WorkflowState.AWAITING_OUTCOME,
            ]
            assert [item.observation_id for item in accepted] == backend_observation_ids
            assert all(
                store.get_booking(booking.booking_id).observation_id == observation_id
                for booking, observation_id in zip(
                    bookings, backend_observation_ids, strict=True
                )
            )

            for booking, observation_id, final_status, expected_label in zip(
                bookings,
                backend_observation_ids,
                ("NOT_CANCELLED", "CANCELLED"),
                (0, 1),
                strict=True,
            ):
                predicted_at = store.get_booking(
                    booking.booking_id
                ).backend_prediction_timestamp
                assert predicted_at is not None
                outcome = FinalOutcomeEvent(
                    source_event_id=f"SYNTHETIC-OUTCOME-EVENT-{expected_label}",
                    booking_id=booking.booking_id,
                    source_event_timestamp=predicted_at + timedelta(seconds=1),
                    final_status=final_status,
                )
                completed = workflow.submit_final_outcome(outcome)
                assert completed.state == WorkflowState.COMPLETED
                assert completed.observation_id == observation_id
                assert completed.outcome_label == expected_label
                assert completed.outcome_source_timestamp == outcome.source_event_timestamp
                assert completed.backend_outcome_received_at is not None

            assert bridge_requests == [
                "/api/v1/monitoring/observations",
                "/api/v1/monitoring/outcomes",
                "/api/v1/monitoring/outcomes",
            ]
            assert len(training_hook_calls) == 2
            outcome_records = [
                json.loads(line)
                for line in (directories["outcomes"] / f"{BATCH_ID}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            assert [record[TARGET_COLUMN] for record in outcome_records] == [0, 1]
            assert [
                record["observation_id"] for record in outcome_records
            ] == backend_observation_ids
            assert all(
                record["outcome_source"] == "production_api"
                and record["source"] == "production"
                for record in outcome_records
            )
            assert all(
                booking.booking_id not in json.dumps(outcome_records)
                for booking in bookings
            )
