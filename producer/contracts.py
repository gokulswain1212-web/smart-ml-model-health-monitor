from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from producer.schemas import Model1Features


class WorkflowState(str, Enum):
    RECEIVED = "RECEIVED"
    VALIDATED = "VALIDATED"
    OBSERVATION_PENDING = "OBSERVATION_PENDING"
    OBSERVATION_ACCEPTED = "OBSERVATION_ACCEPTED"
    AWAITING_OUTCOME = "AWAITING_OUTCOME"
    OUTCOME_PENDING = "OUTCOME_PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    AMBIGUOUS = "AMBIGUOUS"


class Operation(str, Enum):
    OBSERVATION = "observation"
    OUTCOME = "outcome"


@dataclass(frozen=True)
class PreparedObservation:
    backend_booking_reference: str
    features: Model1Features


@dataclass(frozen=True)
class ObservationReceipt:
    batch_id: str
    observation_ids: tuple[str, ...]
    prediction_timestamp: datetime


@dataclass(frozen=True)
class OutcomeReceipt:
    batch_id: str
    observation_id: str
    outcome_received_at: datetime


@dataclass(frozen=True)
class StoredBooking:
    booking_key: str
    source_event_key: str
    batch_id: str
    ordinal: int
    source_event_timestamp: datetime
    state: WorkflowState
    ambiguous_operation: Operation | None
    observation_id: str | None
    backend_prediction_timestamp: datetime | None
    outcome_event_key: str | None
    outcome_label: int | None
    outcome_source_timestamp: datetime | None
    backend_outcome_received_at: datetime | None
