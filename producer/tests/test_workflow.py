from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pytest
from pydantic import ValidationError

from producer.contracts import (
    ObservationReceipt,
    Operation,
    OutcomeReceipt,
    PreparedObservation,
    WorkflowState,
)
from producer.errors import (
    BatchConflict,
    InvalidWorkflowTransition,
    SubmissionAmbiguous,
)
from producer.feature_mapping import FeatureMapper
from producer.identity import ProducerIdentity
from producer.schemas import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    BookingEvent,
    FinalOutcomeEvent,
)
from producer.storage import SQLiteProducerStore
from producer.workflow import ProducerWorkflow

FIXED_NOW = datetime(2025, 3, 12, 12, 0, tzinfo=timezone.utc)
SYNTHETIC_KEY = "SYNTHETIC_TEST_KEY_DO_NOT_USE_IN_PRODUCTION_0001"


def synthetic_features() -> dict[str, object]:
    result: dict[str, object] = {}
    for field in FEATURE_NAMES:
        if field in CATEGORICAL_FEATURES:
            result[field] = "SYNTHETIC"
        elif field == "adr":
            result[field] = 99.5
        elif field == "arrival_date_year":
            result[field] = 2025
        elif field == "arrival_date_week_number":
            result[field] = 11
        elif field == "arrival_date_day_of_month":
            result[field] = 12
        else:
            result[field] = 0
    return result


def synthetic_booking() -> BookingEvent:
    return BookingEvent(
        source_event_id="SYNTHETIC-BOOKING-EVENT-1",
        booking_id="SYNTHETIC-BOOKING-1",
        source_event_timestamp=datetime(2025, 3, 12, 10, 0, tzinfo=timezone.utc),
        features=synthetic_features(),
    )


def synthetic_outcome(status: str = "CANCELLED") -> FinalOutcomeEvent:
    return FinalOutcomeEvent(
        source_event_id="SYNTHETIC-OUTCOME-EVENT-1",
        booking_id="SYNTHETIC-BOOKING-1",
        source_event_timestamp=datetime(2025, 3, 12, 12, 0, 30, tzinfo=timezone.utc),
        final_status=status,
    )


class FakeObservationClient:
    def __init__(self, results: list[ObservationReceipt | Exception]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, Sequence[PreparedObservation]]] = []

    def submit(
        self,
        batch_id: str,
        observations: Sequence[PreparedObservation],
    ) -> ObservationReceipt:
        self.calls.append((batch_id, observations))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeOutcomeClient:
    def __init__(self, results: list[OutcomeReceipt | Exception]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, int]] = []

    def submit(self, observation_id: str, is_canceled: int) -> OutcomeReceipt:
        self.calls.append((observation_id, is_canceled))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def make_workflow(
    tmp_path: Path,
    observation_results: list[ObservationReceipt | Exception],
    outcome_results: list[OutcomeReceipt | Exception] | None = None,
) -> tuple[ProducerWorkflow, SQLiteProducerStore, FakeObservationClient, FakeOutcomeClient]:
    identity = ProducerIdentity(SYNTHETIC_KEY)
    store = SQLiteProducerStore(tmp_path / "producer-state.sqlite3", identity)
    observation = FakeObservationClient(observation_results)
    outcome = FakeOutcomeClient(outcome_results or [])
    workflow = ProducerWorkflow(
        store=store,
        identity=identity,
        feature_mapper=FeatureMapper.identity(),
        observation_client=observation,
        outcome_client=outcome,
    )
    return workflow, store, observation, outcome


def observation_receipt(batch_id: str = "SYNTHETIC-BATCH-1") -> ObservationReceipt:
    return ObservationReceipt(
        batch_id=batch_id,
        observation_ids=("SYNTHETIC-OBSERVATION-1",),
        prediction_timestamp=FIXED_NOW,
    )


@pytest.mark.parametrize(
    ("final_status", "expected_label"),
    [("CANCELLED", 1), ("NOT_CANCELLED", 0)],
)
def test_successful_workflow_links_observation_and_authoritative_outcome(
    tmp_path: Path,
    final_status: str,
    expected_label: int,
) -> None:
    workflow, store, observation, outcome = make_workflow(
        tmp_path,
        [observation_receipt()],
        [
            OutcomeReceipt(
                batch_id="SYNTHETIC-BATCH-1",
                observation_id="SYNTHETIC-OBSERVATION-1",
                outcome_received_at=datetime(2025, 3, 12, 12, 1, tzinfo=timezone.utc),
            )
        ],
    )
    booking = synthetic_booking()
    final_outcome = synthetic_outcome(final_status)
    accepted = workflow.submit_observation_batch("SYNTHETIC-BATCH-1", [booking])
    assert accepted[0].state == WorkflowState.AWAITING_OUTCOME
    assert accepted[0].observation_id == "SYNTHETIC-OBSERVATION-1"
    assert accepted[0].source_event_timestamp == booking.source_event_timestamp
    assert accepted[0].backend_prediction_timestamp == FIXED_NOW
    completed = workflow.submit_final_outcome(final_outcome)
    assert completed.state == WorkflowState.COMPLETED
    assert completed.outcome_label == expected_label
    assert completed.outcome_source_timestamp == final_outcome.source_event_timestamp
    assert completed.backend_outcome_received_at == datetime(
        2025, 3, 12, 12, 1, tzinfo=timezone.utc
    )
    assert len(observation.calls) == 1
    assert outcome.calls == [("SYNTHETIC-OBSERVATION-1", expected_label)]
    assert workflow.submit_final_outcome(final_outcome) == completed
    assert len(outcome.calls) == 1
    assert store.get_booking(booking.booking_id) == completed

    with sqlite3.connect(store.path) as connection:
        stored_text = json.dumps(connection.execute(
            "SELECT * FROM booking_workflows"
        ).fetchall())
        schema = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' "
            "AND name = 'booking_workflows'"
        ).fetchone()[0]
    assert booking.booking_id not in stored_text
    assert "features" not in schema


def test_ambiguous_observation_requires_confirmed_absence_and_same_batch_id(
    tmp_path: Path,
) -> None:
    workflow, store, observation, _ = make_workflow(
        tmp_path,
        [
            SubmissionAmbiguous(Operation.OBSERVATION),
            observation_receipt(),
        ],
    )
    event = synthetic_booking()
    with pytest.raises(SubmissionAmbiguous):
        workflow.submit_observation_batch("SYNTHETIC-BATCH-1", [event])
    assert store.get_batch("SYNTHETIC-BATCH-1")[0].state == WorkflowState.AMBIGUOUS
    with pytest.raises(SubmissionAmbiguous):
        workflow.submit_observation_batch("SYNTHETIC-BATCH-1", [event])
    assert len(observation.calls) == 1

    with pytest.raises(ValueError, match="confirmed absence"):
        workflow.reconcile_observation_absent(
            "SYNTHETIC-BATCH-1",
            confirmed_absent=False,
        )
    workflow.reconcile_observation_absent(
        "SYNTHETIC-BATCH-1",
        confirmed_absent=True,
    )
    assert store.get_batch("SYNTHETIC-BATCH-1")[0].state == WorkflowState.VALIDATED
    accepted = workflow.submit_observation_batch("SYNTHETIC-BATCH-1", [event])
    assert accepted[0].state == WorkflowState.AWAITING_OUTCOME
    assert len(observation.calls) == 2
    assert all(call[0] == "SYNTHETIC-BATCH-1" for call in observation.calls)


def test_pending_outcome_is_not_resent_and_reconciliation_checks_label(
    tmp_path: Path,
) -> None:
    workflow, store, observation, outcome = make_workflow(
        tmp_path,
        [observation_receipt()],
        [SubmissionAmbiguous(Operation.OUTCOME)],
    )
    workflow.submit_observation_batch("SYNTHETIC-BATCH-1", [synthetic_booking()])
    event = synthetic_outcome()
    with pytest.raises(SubmissionAmbiguous):
        workflow.submit_final_outcome(event)
    with pytest.raises(SubmissionAmbiguous):
        workflow.submit_final_outcome(event)
    assert len(outcome.calls) == 1

    wrong_event = event.model_copy(update={"final_status": "NOT_CANCELLED"})
    with pytest.raises(BatchConflict, match="does not match"):
        workflow.reconcile_outcome_accepted(
            wrong_event,
            datetime(2025, 3, 12, 12, 1, tzinfo=timezone.utc),
        )
    with pytest.raises(InvalidWorkflowTransition, match="does not match"):
        workflow.reconcile_outcome_absent(wrong_event, confirmed_absent=True)
    assert store.get_booking(event.booking_id).state == WorkflowState.AMBIGUOUS

    resolved = workflow.reconcile_outcome_accepted(
        event,
        datetime(2025, 3, 12, 12, 1, tzinfo=timezone.utc),
    )
    assert resolved.state == WorkflowState.COMPLETED
    assert len(observation.calls) == 1
    assert len(outcome.calls) == 1


def test_changed_features_or_mapping_cannot_reuse_a_batch_id(tmp_path: Path) -> None:
    workflow, _, _, _ = make_workflow(
        tmp_path,
        [observation_receipt()],
    )
    event = synthetic_booking()
    workflow.submit_observation_batch("SYNTHETIC-BATCH-1", [event])
    changed = event.model_copy(
        update={"features": {**event.features, "lead_time": 4}}
    )
    with pytest.raises(BatchConflict, match="different request digest"):
        workflow.submit_observation_batch("SYNTHETIC-BATCH-1", [changed])


def test_invalid_features_fail_before_any_observation_submission(tmp_path: Path) -> None:
    workflow, store, observation, _ = make_workflow(tmp_path, [observation_receipt()])
    event = synthetic_booking().model_copy(
        update={"features": {**synthetic_features(), "lead_time": "not-an-integer"}}
    )
    with pytest.raises(ValidationError):
        workflow.submit_observation_batch("SYNTHETIC-BATCH-1", [event])
    assert observation.calls == []
    assert store.get_batch("SYNTHETIC-BATCH-1")[0].state == WorkflowState.FAILED
