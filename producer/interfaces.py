from __future__ import annotations

from datetime import datetime
from typing import Protocol, Sequence

from producer.contracts import (
    ObservationReceipt,
    Operation,
    OutcomeReceipt,
    PreparedObservation,
    StoredBooking,
    WorkflowState,
)
from producer.schemas import BookingEvent


class ObservationSubmitter(Protocol):
    def submit(
        self,
        batch_id: str,
        observations: Sequence[PreparedObservation],
    ) -> ObservationReceipt: ...


class OutcomeSubmitter(Protocol):
    def submit(
        self,
        observation_id: str,
        is_canceled: int,
    ) -> OutcomeReceipt: ...


class ProducerStore(Protocol):
    def register_batch(
        self,
        batch_id: str,
        digest: str,
        events: Sequence[BookingEvent],
    ) -> bool: ...

    def get_batch(self, batch_id: str) -> list[StoredBooking]: ...

    def transition_batch(
        self,
        batch_id: str,
        expected: set[WorkflowState],
        new_state: WorkflowState,
        *,
        ambiguous_operation: Operation | None = None,
    ) -> None: ...

    def record_observation_receipt(
        self,
        batch_id: str,
        observation_ids: Sequence[str],
        prediction_timestamp: datetime,
        *,
        expected_state: WorkflowState,
        expected_operation: Operation | None = None,
    ) -> None: ...

    def get_booking(self, booking_id: str) -> StoredBooking | None: ...

    def begin_outcome(
        self,
        booking_id: str,
        event_id: str,
        is_canceled: int,
        source_event_timestamp: datetime,
    ) -> StoredBooking: ...

    def mark_outcome_ambiguous(self, booking_id: str) -> None: ...

    def complete_outcome(
        self,
        booking_id: str,
        outcome_received_at: datetime,
        *,
        expected_state: WorkflowState = WorkflowState.OUTCOME_PENDING,
    ) -> None: ...

    def complete_reconciled_outcome(
        self,
        booking_id: str,
        *,
        event_id: str,
        is_canceled: int,
        source_event_timestamp: datetime,
        outcome_received_at: datetime,
    ) -> None: ...

    def mark_outcome_ambiguous(self, booking_id: str) -> None: ...

    def mark_outcome_failed(self, booking_id: str) -> None: ...

    def resolve_observation_absent(
        self,
        batch_id: str,
        *,
        confirmed_absent: bool,
    ) -> None: ...

    def resolve_outcome_absent(
        self,
        booking_id: str,
        *,
        confirmed_absent: bool,
    ) -> None: ...


class RetryScheduler(Protocol):
    def schedule(
        self,
        operation_key: str,
        operation: Operation,
        attempt: int,
        not_before: datetime,
    ) -> None: ...


class RetryPolicy(Protocol):
    def delay_seconds(self, attempt: int) -> float | None: ...


class ExponentialBackoff:
    """Bounded delay calculator; ambiguous submissions are never auto-retried."""

    def __init__(
        self,
        *,
        maximum_attempts: int = 5,
        initial_delay_seconds: float = 1.0,
        maximum_delay_seconds: float = 60.0,
    ) -> None:
        if maximum_attempts < 1:
            raise ValueError("maximum_attempts must be at least one")
        if initial_delay_seconds <= 0 or maximum_delay_seconds <= 0:
            raise ValueError("retry delays must be positive")
        self.maximum_attempts = maximum_attempts
        self.initial_delay_seconds = initial_delay_seconds
        self.maximum_delay_seconds = maximum_delay_seconds

    def delay_seconds(self, attempt: int) -> float | None:
        if attempt < 1:
            raise ValueError("attempt must be at least one")
        if attempt >= self.maximum_attempts:
            return None
        return min(
            self.maximum_delay_seconds,
            self.initial_delay_seconds * (2 ** (attempt - 1)),
        )
