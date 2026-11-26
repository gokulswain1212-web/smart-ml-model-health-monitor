from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Sequence
from uuid import uuid4

from pydantic import ValidationError

from producer.contracts import Operation, PreparedObservation, StoredBooking, WorkflowState
from producer.errors import (
    BatchConflict,
    InvalidWorkflowTransition,
    SubmissionAmbiguous,
    SubmissionRejected,
    UnknownBooking,
)
from producer.feature_mapping import FeatureMapper
from producer.identity import ProducerIdentity
from producer.interfaces import ObservationSubmitter, OutcomeSubmitter, ProducerStore
from producer.logging_utils import ProducerEvent, emit_event
from producer.schemas import BookingEvent, FinalOutcomeEvent


class ProducerWorkflow:
    """Coordinates durable local state with the existing monitoring API."""

    def __init__(
        self,
        *,
        store: ProducerStore,
        identity: ProducerIdentity,
        feature_mapper: FeatureMapper,
        observation_client: ObservationSubmitter,
        outcome_client: OutcomeSubmitter,
        logger: logging.Logger | None = None,
    ) -> None:
        self.store = store
        self.identity = identity
        self.feature_mapper = feature_mapper
        self.observation_client = observation_client
        self.outcome_client = outcome_client
        self.logger = logger or logging.getLogger("producer")

    def submit_observation_batch(
        self,
        batch_id: str,
        events: Sequence[BookingEvent],
    ) -> list[StoredBooking]:
        if not events:
            raise ValueError("an observation batch must contain at least one booking")
        stable_events = [event.model_copy(deep=True) for event in events]
        serialized_events = json.dumps(
            {
                "events": [
                    json.loads(event.model_dump_json()) for event in stable_events
                ],
                "feature_mapping": dict(self.feature_mapper.model1_to_source),
                "mapping_version": self.feature_mapper.mapping_version,
                "backend_booking_references": [
                    self.identity.backend_booking_reference(event.booking_id)
                    for event in stable_events
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest = self.identity.batch_digest(serialized_events)
        created = self.store.register_batch(batch_id, digest, stable_events)
        stored = self.store.get_batch(batch_id)
        if not stored:
            raise BatchConflict("producer store did not retain the registered batch")

        if not created:
            states = {item.state for item in stored}
            operations = {item.ambiguous_operation for item in stored}
            if states == {WorkflowState.AMBIGUOUS}:
                operation = next(iter(operations))
                if operation is None:
                    raise InvalidWorkflowTransition(
                        "ambiguous batch is missing its operation"
                    )
                raise SubmissionAmbiguous(operation)
            if states == {WorkflowState.OBSERVATION_PENDING}:
                self.store.transition_batch(
                    batch_id,
                    {WorkflowState.OBSERVATION_PENDING},
                    WorkflowState.AMBIGUOUS,
                    ambiguous_operation=Operation.OBSERVATION,
                )
                raise SubmissionAmbiguous(Operation.OBSERVATION)
            if states == {WorkflowState.OBSERVATION_ACCEPTED}:
                self.store.transition_batch(
                    batch_id,
                    {WorkflowState.OBSERVATION_ACCEPTED},
                    WorkflowState.AWAITING_OUTCOME,
                )
                return self.store.get_batch(batch_id)
            if not states.issubset(
                {WorkflowState.RECEIVED, WorkflowState.VALIDATED}
            ):
                raise BatchConflict(
                    "batch was already submitted or completed; do not resubmit it"
                )

        correlation_id = uuid4().hex
        emit_event(
            self.logger,
            ProducerEvent.BOOKING_RECEIVED,
            correlation_id=correlation_id,
            record_count=len(events),
        )
        stored = self.store.get_batch(batch_id)
        states = {item.state for item in stored}
        try:
            prepared = [
                PreparedObservation(
                    backend_booking_reference=self.identity.backend_booking_reference(
                        event.booking_id
                    ),
                    features=self.feature_mapper.map_event(event),
                )
                for event in stable_events
            ]
        except (ValidationError, ValueError):
            if states.issubset({WorkflowState.RECEIVED, WorkflowState.VALIDATED}):
                self.store.transition_batch(
                    batch_id,
                    {WorkflowState.RECEIVED, WorkflowState.VALIDATED},
                    WorkflowState.FAILED,
                )
            emit_event(
                self.logger,
                ProducerEvent.SUBMISSION_REJECTED,
                correlation_id=correlation_id,
                operation=Operation.OBSERVATION.value,
                state=WorkflowState.FAILED.value,
            )
            raise

        if WorkflowState.RECEIVED in states:
            self.store.transition_batch(
                batch_id,
                {WorkflowState.RECEIVED},
                WorkflowState.VALIDATED,
            )
        emit_event(
            self.logger,
            ProducerEvent.BOOKING_VALIDATED,
            correlation_id=correlation_id,
            record_count=len(prepared),
            state=WorkflowState.VALIDATED.value,
        )
        self.store.transition_batch(
            batch_id,
            {WorkflowState.VALIDATED},
            WorkflowState.OBSERVATION_PENDING,
        )
        emit_event(
            self.logger,
            ProducerEvent.OBSERVATION_SUBMISSION_STARTED,
            correlation_id=correlation_id,
            operation=Operation.OBSERVATION.value,
            state=WorkflowState.OBSERVATION_PENDING.value,
            record_count=len(prepared),
        )
        try:
            receipt = self.observation_client.submit(batch_id, prepared)
        except SubmissionAmbiguous as error:
            self.store.transition_batch(
                batch_id,
                {WorkflowState.OBSERVATION_PENDING},
                WorkflowState.AMBIGUOUS,
                ambiguous_operation=Operation.OBSERVATION,
            )
            emit_event(
                self.logger,
                ProducerEvent.SUBMISSION_AMBIGUOUS,
                correlation_id=correlation_id,
                operation=Operation.OBSERVATION.value,
                state=WorkflowState.AMBIGUOUS.value,
                status_code=error.status_code or 0,
            )
            raise
        except SubmissionRejected as error:
            self.store.transition_batch(
                batch_id,
                {WorkflowState.OBSERVATION_PENDING},
                WorkflowState.FAILED,
            )
            emit_event(
                self.logger,
                ProducerEvent.SUBMISSION_REJECTED,
                correlation_id=correlation_id,
                operation=Operation.OBSERVATION.value,
                state=WorkflowState.FAILED.value,
                status_code=error.status_code,
            )
            raise
        except ValueError:
            self.store.transition_batch(
                batch_id,
                {WorkflowState.OBSERVATION_PENDING},
                WorkflowState.FAILED,
            )
            emit_event(
                self.logger,
                ProducerEvent.SUBMISSION_REJECTED,
                correlation_id=correlation_id,
                operation=Operation.OBSERVATION.value,
                state=WorkflowState.FAILED.value,
            )
            raise

        self.store.record_observation_receipt(
            batch_id,
            receipt.observation_ids,
            receipt.prediction_timestamp,
            expected_state=WorkflowState.OBSERVATION_PENDING,
        )
        self.store.transition_batch(
            batch_id,
            {WorkflowState.OBSERVATION_ACCEPTED},
            WorkflowState.AWAITING_OUTCOME,
        )
        emit_event(
            self.logger,
            ProducerEvent.OBSERVATION_ACCEPTED,
            correlation_id=correlation_id,
            state=WorkflowState.AWAITING_OUTCOME.value,
            record_count=len(prepared),
        )
        return self.store.get_batch(batch_id)

    def reconcile_observation_accepted(
        self,
        batch_id: str,
        observation_ids: Sequence[str],
        prediction_timestamp: datetime,
    ) -> list[StoredBooking]:
        self.store.record_observation_receipt(
            batch_id,
            observation_ids,
            prediction_timestamp,
            expected_state=WorkflowState.AMBIGUOUS,
            expected_operation=Operation.OBSERVATION,
        )
        self.store.transition_batch(
            batch_id,
            {WorkflowState.OBSERVATION_ACCEPTED},
            WorkflowState.AWAITING_OUTCOME,
        )
        return self.store.get_batch(batch_id)

    def reconcile_observation_absent(
        self,
        batch_id: str,
        *,
        confirmed_absent: bool,
    ) -> None:
        self.store.resolve_observation_absent(
            batch_id,
            confirmed_absent=confirmed_absent,
        )

    def submit_final_outcome(self, event: FinalOutcomeEvent) -> StoredBooking:
        stored = self.store.get_booking(event.booking_id)
        if stored is None:
            raise UnknownBooking("outcome has no stored observation mapping")

        event_key = self.identity.source_event_key(
            event.source_event_id,
            event_kind="outcome",
        )
        if stored.state == WorkflowState.COMPLETED:
            if (
                stored.outcome_event_key == event_key
                and stored.outcome_label == event.is_canceled
                and stored.outcome_source_timestamp == event.source_event_timestamp
            ):
                return stored
            raise BatchConflict("booking already has a different completed outcome")
        if stored.state == WorkflowState.AMBIGUOUS:
            if stored.ambiguous_operation == Operation.OUTCOME:
                raise SubmissionAmbiguous(Operation.OUTCOME)
            raise InvalidWorkflowTransition(
                "booking is ambiguous for an operation other than outcome"
            )
        if stored.state == WorkflowState.OUTCOME_PENDING:
            self.store.mark_outcome_ambiguous(event.booking_id)
            raise SubmissionAmbiguous(Operation.OUTCOME)
        if stored.state != WorkflowState.AWAITING_OUTCOME:
            raise InvalidWorkflowTransition(
                "booking is not ready to accept a final outcome"
            )
        if stored.observation_id is None:
            raise InvalidWorkflowTransition(
                "booking is missing its accepted observation ID"
            )
        if (
            stored.backend_prediction_timestamp is None
            or event.source_event_timestamp < stored.backend_prediction_timestamp
        ):
            raise ValueError(
                "final outcome source timestamp must not predate the backend prediction"
            )

        pending = self.store.begin_outcome(
            event.booking_id,
            event.source_event_id,
            event.is_canceled,
            event.source_event_timestamp,
        )
        correlation_id = uuid4().hex
        emit_event(
            self.logger,
            ProducerEvent.OUTCOME_SUBMISSION_STARTED,
            correlation_id=correlation_id,
            operation=Operation.OUTCOME.value,
            state=WorkflowState.OUTCOME_PENDING.value,
        )
        try:
            receipt = self.outcome_client.submit(
                pending.observation_id or "",
                event.is_canceled,
            )
        except SubmissionAmbiguous as error:
            self.store.mark_outcome_ambiguous(event.booking_id)
            emit_event(
                self.logger,
                ProducerEvent.SUBMISSION_AMBIGUOUS,
                correlation_id=correlation_id,
                operation=Operation.OUTCOME.value,
                state=WorkflowState.AMBIGUOUS.value,
                status_code=error.status_code or 0,
            )
            raise
        except SubmissionRejected as error:
            self.store.mark_outcome_failed(event.booking_id)
            emit_event(
                self.logger,
                ProducerEvent.SUBMISSION_REJECTED,
                correlation_id=correlation_id,
                operation=Operation.OUTCOME.value,
                state=WorkflowState.FAILED.value,
                status_code=error.status_code,
            )
            raise
        except ValueError:
            self.store.mark_outcome_failed(event.booking_id)
            emit_event(
                self.logger,
                ProducerEvent.SUBMISSION_REJECTED,
                correlation_id=correlation_id,
                operation=Operation.OUTCOME.value,
                state=WorkflowState.FAILED.value,
            )
            raise

        if (
            receipt.observation_id != pending.observation_id
            or receipt.batch_id != pending.batch_id
        ):
            self.store.mark_outcome_ambiguous(event.booking_id)
            raise SubmissionAmbiguous(Operation.OUTCOME, status_code=200)
        self.store.complete_outcome(
            event.booking_id,
            receipt.outcome_received_at,
        )
        emit_event(
            self.logger,
            ProducerEvent.OUTCOME_COMPLETED,
            correlation_id=correlation_id,
            state=WorkflowState.COMPLETED.value,
        )
        completed = self.store.get_booking(event.booking_id)
        if completed is None:
            raise UnknownBooking("completed booking mapping disappeared from storage")
        return completed

    def reconcile_outcome_accepted(
        self,
        event: FinalOutcomeEvent,
        outcome_received_at: datetime,
    ) -> StoredBooking:
        self.store.complete_reconciled_outcome(
            event.booking_id,
            event_id=event.source_event_id,
            is_canceled=event.is_canceled,
            source_event_timestamp=event.source_event_timestamp,
            outcome_received_at=outcome_received_at,
        )
        stored = self.store.get_booking(event.booking_id)
        if stored is None:
            raise UnknownBooking("completed booking mapping disappeared from storage")
        return stored

    def reconcile_outcome_absent(
        self,
        event: FinalOutcomeEvent,
        *,
        confirmed_absent: bool,
    ) -> None:
        stored = self.store.get_booking(event.booking_id)
        expected_event_key = self.identity.source_event_key(
            event.source_event_id,
            event_kind="outcome",
        )
        if (
            stored is None
            or stored.state != WorkflowState.AMBIGUOUS
            or stored.ambiguous_operation != Operation.OUTCOME
            or stored.outcome_event_key != expected_event_key
            or stored.outcome_label != event.is_canceled
            or stored.outcome_source_timestamp != event.source_event_timestamp
        ):
            raise InvalidWorkflowTransition(
                "outcome event does not match the ambiguous submission"
            )
        self.store.resolve_outcome_absent(
            event.booking_id,
            confirmed_absent=confirmed_absent,
        )
