from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

from producer.contracts import Operation, StoredBooking, WorkflowState
from producer.errors import (
    BatchConflict,
    DuplicateBooking,
    InvalidWorkflowTransition,
    UnknownBooking,
)
from producer.identity import ProducerIdentity
from producer.schemas import BookingEvent

BATCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("stored timestamps must include a timezone")
    return value.astimezone(timezone.utc).isoformat()


def _parse_utc(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("stored timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def _required_utc(value: str) -> datetime:
    parsed = _parse_utc(value)
    if parsed is None:
        raise ValueError("required stored timestamp is missing")
    return parsed


class SQLiteProducerStore:
    """Durable local store that persists keyed IDs, states, and timestamps only."""

    def __init__(self, path: str | Path, identity: ProducerIdentity) -> None:
        self.path = Path(path)
        self.identity = identity
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS producer_batches (
                    batch_id TEXT PRIMARY KEY,
                    request_digest TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS booking_workflows (
                    booking_key TEXT PRIMARY KEY,
                    source_event_key TEXT NOT NULL UNIQUE,
                    batch_id TEXT NOT NULL REFERENCES producer_batches(batch_id),
                    ordinal INTEGER NOT NULL,
                    source_event_timestamp TEXT NOT NULL,
                    state TEXT NOT NULL,
                    ambiguous_operation TEXT,
                    observation_id TEXT UNIQUE,
                    backend_prediction_timestamp TEXT,
                    outcome_event_key TEXT,
                    outcome_label INTEGER,
                    outcome_source_timestamp TEXT,
                    backend_outcome_received_at TEXT,
                    updated_at TEXT NOT NULL,
                    UNIQUE(batch_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS booking_workflows_batch_idx
                    ON booking_workflows(batch_id, ordinal);
                """
            )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def register_batch(
        self,
        batch_id: str,
        digest: str,
        events: Sequence[BookingEvent],
    ) -> bool:
        if not BATCH_ID_PATTERN.fullmatch(batch_id):
            raise ValueError("batch_id must use the backend's safe identifier format")
        if not digest or not events:
            raise ValueError("a batch digest and at least one booking event are required")

        booking_keys = [self.identity.booking_key(event.booking_id) for event in events]
        event_keys = [
            self.identity.source_event_key(event.source_event_id, event_kind="booking")
            for event in events
        ]
        if len(set(booking_keys)) != len(booking_keys):
            raise DuplicateBooking("a booking may be observed only once by this producer")
        if len(set(event_keys)) != len(event_keys):
            raise DuplicateBooking("source booking event IDs must be unique within a batch")

        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connection() as connection:
                existing = connection.execute(
                    "SELECT request_digest FROM producer_batches WHERE batch_id = ?",
                    (batch_id,),
                ).fetchone()
                if existing is not None:
                    if existing["request_digest"] != digest:
                        raise BatchConflict(
                            "batch_id already exists with a different request digest"
                        )
                    return False

                connection.execute(
                    """
                    INSERT INTO producer_batches
                        (batch_id, request_digest, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (batch_id, digest, now, now),
                )
                for ordinal, (event, booking_key, event_key) in enumerate(
                    zip(events, booking_keys, event_keys, strict=True)
                ):
                    connection.execute(
                        """
                        INSERT INTO booking_workflows (
                            booking_key, source_event_key, batch_id, ordinal,
                            source_event_timestamp, state, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            booking_key,
                            event_key,
                            batch_id,
                            ordinal,
                            _utc_iso(event.source_event_timestamp),
                            WorkflowState.RECEIVED.value,
                            now,
                        ),
                    )
        except sqlite3.IntegrityError as error:
            raise DuplicateBooking(
                "booking or source event already exists in producer storage"
            ) from error
        return True

    def get_batch(self, batch_id: str) -> list[StoredBooking]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM booking_workflows
                WHERE batch_id = ?
                ORDER BY ordinal
                """,
                (batch_id,),
            ).fetchall()
        return [self._stored_booking(row) for row in rows]

    def get_booking(self, booking_id: str) -> StoredBooking | None:
        booking_key = self.identity.booking_key(booking_id)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM booking_workflows WHERE booking_key = ?",
                (booking_key,),
            ).fetchone()
        return self._stored_booking(row) if row is not None else None

    def transition_batch(
        self,
        batch_id: str,
        expected: set[WorkflowState],
        new_state: WorkflowState,
        *,
        ambiguous_operation: Operation | None = None,
    ) -> None:
        if not expected:
            raise ValueError("expected states cannot be empty")
        if (
            new_state == WorkflowState.AMBIGUOUS
            and ambiguous_operation is None
        ) or (
            new_state != WorkflowState.AMBIGUOUS
            and ambiguous_operation is not None
        ):
            raise ValueError("ambiguous_operation must match the new workflow state")
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT state FROM booking_workflows WHERE batch_id = ?",
                (batch_id,),
            ).fetchall()
            if not rows:
                raise BatchConflict("batch does not exist in producer storage")
            allowed = {state.value for state in expected}
            if any(row["state"] not in allowed for row in rows):
                raise InvalidWorkflowTransition(
                    "batch contains a booking in an unexpected workflow state"
                )
            now = datetime.now(timezone.utc).isoformat()
            connection.execute(
                """
                UPDATE booking_workflows
                SET state = ?, ambiguous_operation = ?, updated_at = ?
                WHERE batch_id = ?
                """,
                (
                    new_state.value,
                    ambiguous_operation.value if ambiguous_operation else None,
                    now,
                    batch_id,
                ),
            )
            connection.execute(
                "UPDATE producer_batches SET updated_at = ? WHERE batch_id = ?",
                (now, batch_id),
            )

    def record_observation_receipt(
        self,
        batch_id: str,
        observation_ids: Sequence[str],
        prediction_timestamp: datetime,
        *,
        expected_state: WorkflowState,
        expected_operation: Operation | None = None,
    ) -> None:
        if not observation_ids or any(not item for item in observation_ids):
            raise ValueError("observation IDs must be non-empty")
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("observation IDs must be unique")
        prediction_at = _utc_iso(prediction_timestamp)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT booking_key, state, ambiguous_operation
                FROM booking_workflows
                WHERE batch_id = ?
                ORDER BY ordinal
                """,
                (batch_id,),
            ).fetchall()
            if len(rows) != len(observation_ids):
                raise InvalidWorkflowTransition(
                    "observation response count does not match stored batch"
                )
            for row in rows:
                if row["state"] != expected_state.value:
                    raise InvalidWorkflowTransition(
                        "batch is not in the expected observation state"
                    )
                actual_operation = row["ambiguous_operation"]
                expected_value = (
                    expected_operation.value if expected_operation else None
                )
                if actual_operation != expected_value:
                    raise InvalidWorkflowTransition(
                        "ambiguous operation does not match reconciliation"
                    )

            now = datetime.now(timezone.utc).isoformat()
            for row, observation_id in zip(rows, observation_ids, strict=True):
                connection.execute(
                    """
                    UPDATE booking_workflows
                    SET observation_id = ?, backend_prediction_timestamp = ?,
                        state = ?, ambiguous_operation = NULL, updated_at = ?
                    WHERE booking_key = ?
                    """,
                    (
                        observation_id,
                        prediction_at,
                        WorkflowState.OBSERVATION_ACCEPTED.value,
                        now,
                        row["booking_key"],
                    ),
                )
            connection.execute(
                "UPDATE producer_batches SET updated_at = ? WHERE batch_id = ?",
                (now, batch_id),
            )

    def begin_outcome(
        self,
        booking_id: str,
        event_id: str,
        is_canceled: int,
        source_event_timestamp: datetime,
    ) -> StoredBooking:
        if type(is_canceled) is not int or is_canceled not in (0, 1):
            raise ValueError("is_canceled must be the integer 0 or 1")
        booking_key = self.identity.booking_key(booking_id)
        event_key = self.identity.source_event_key(event_id, event_kind="outcome")
        source_time = _utc_iso(source_event_timestamp)
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM booking_workflows WHERE booking_key = ?",
                (booking_key,),
            ).fetchone()
            if row is None:
                raise UnknownBooking("outcome has no stored observation mapping")
            if row["state"] != WorkflowState.AWAITING_OUTCOME.value:
                raise InvalidWorkflowTransition(
                    "booking is not awaiting an authoritative outcome"
                )
            if _parse_utc(row["source_event_timestamp"]) > _parse_utc(source_time):
                raise ValueError("outcome event cannot predate its booking observation")

            existing_event_key = row["outcome_event_key"]
            if existing_event_key is not None and (
                existing_event_key != event_key
                or row["outcome_label"] != is_canceled
                or row["outcome_source_timestamp"] != source_time
            ):
                raise BatchConflict(
                    "booking already has a different pending outcome event"
                )
            connection.execute(
                """
                UPDATE booking_workflows
                SET state = ?, ambiguous_operation = NULL,
                    outcome_event_key = ?, outcome_label = ?,
                    outcome_source_timestamp = ?, updated_at = ?
                WHERE booking_key = ?
                """,
                (
                    WorkflowState.OUTCOME_PENDING.value,
                    event_key,
                    is_canceled,
                    source_time,
                    now,
                    booking_key,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM booking_workflows WHERE booking_key = ?",
                (booking_key,),
            ).fetchone()
        return self._stored_booking(updated)

    def mark_outcome_ambiguous(self, booking_id: str) -> None:
        self._transition_booking(
            booking_id,
            expected={WorkflowState.OUTCOME_PENDING},
            new_state=WorkflowState.AMBIGUOUS,
            ambiguous_operation=Operation.OUTCOME,
        )

    def mark_outcome_failed(self, booking_id: str) -> None:
        self._transition_booking(
            booking_id,
            expected={WorkflowState.OUTCOME_PENDING},
            new_state=WorkflowState.FAILED,
        )

    def complete_outcome(
        self,
        booking_id: str,
        outcome_received_at: datetime,
        *,
        expected_state: WorkflowState = WorkflowState.OUTCOME_PENDING,
    ) -> None:
        self._transition_booking(
            booking_id,
            expected={expected_state},
            new_state=WorkflowState.COMPLETED,
            outcome_received_at=_utc_iso(outcome_received_at),
            expected_operation=(
                Operation.OUTCOME if expected_state == WorkflowState.AMBIGUOUS else None
            ),
        )

    def resolve_observation_absent(
        self,
        batch_id: str,
        *,
        confirmed_absent: bool,
    ) -> None:
        if not confirmed_absent:
            raise ValueError("an ambiguous observation may retry only after confirmed absence")
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT state, ambiguous_operation
                FROM booking_workflows
                WHERE batch_id = ?
                """,
                (batch_id,),
            ).fetchall()
            if not rows or any(
                row["state"] != WorkflowState.AMBIGUOUS.value
                or row["ambiguous_operation"] != Operation.OBSERVATION.value
                for row in rows
            ):
                raise InvalidWorkflowTransition(
                    "batch is not ambiguous for observation submission"
                )
            now = datetime.now(timezone.utc).isoformat()
            connection.execute(
                """
                UPDATE booking_workflows
                SET state = ?, ambiguous_operation = NULL, updated_at = ?
                WHERE batch_id = ?
                """,
                (WorkflowState.VALIDATED.value, now, batch_id),
            )
            connection.execute(
                "UPDATE producer_batches SET updated_at = ? WHERE batch_id = ?",
                (now, batch_id),
            )

    def resolve_outcome_absent(
        self,
        booking_id: str,
        *,
        confirmed_absent: bool,
    ) -> None:
        if not confirmed_absent:
            raise ValueError("an ambiguous outcome may retry only after confirmed absence")
        self._transition_booking(
            booking_id,
            expected={WorkflowState.AMBIGUOUS},
            new_state=WorkflowState.AWAITING_OUTCOME,
            expected_operation=Operation.OUTCOME,
        )

    def complete_reconciled_outcome(
        self,
        booking_id: str,
        *,
        event_id: str,
        is_canceled: int,
        source_event_timestamp: datetime,
        outcome_received_at: datetime,
    ) -> None:
        record = self.get_booking(booking_id)
        if record is None:
            raise UnknownBooking("booking has no stored observation mapping")
        event_key = self.identity.source_event_key(event_id, event_kind="outcome")
        if (
            record.outcome_event_key != event_key
            or record.outcome_label != is_canceled
            or record.outcome_source_timestamp != source_event_timestamp
        ):
            raise BatchConflict("reconciled outcome event does not match pending event")
        self.complete_outcome(
            booking_id,
            outcome_received_at,
            expected_state=WorkflowState.AMBIGUOUS,
        )

    def _transition_booking(
        self,
        booking_id: str,
        *,
        expected: set[WorkflowState],
        new_state: WorkflowState,
        ambiguous_operation: Operation | None = None,
        expected_operation: Operation | None = None,
        outcome_received_at: str | None = None,
    ) -> None:
        booking_key = self.identity.booking_key(booking_id)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT state, ambiguous_operation FROM booking_workflows WHERE booking_key = ?",
                (booking_key,),
            ).fetchone()
            if row is None:
                raise UnknownBooking("booking has no stored observation mapping")
            if row["state"] not in {state.value for state in expected}:
                raise InvalidWorkflowTransition(
                    "booking is not in the expected workflow state"
                )
            expected_operation_value = (
                expected_operation.value if expected_operation else None
            )
            if row["ambiguous_operation"] != expected_operation_value:
                raise InvalidWorkflowTransition(
                    "booking is not ambiguous for the expected operation"
                )
            now = datetime.now(timezone.utc).isoformat()
            connection.execute(
                """
                UPDATE booking_workflows
                SET state = ?, ambiguous_operation = ?, updated_at = ?,
                    backend_outcome_received_at = COALESCE(?, backend_outcome_received_at)
                WHERE booking_key = ?
                """,
                (
                    new_state.value,
                    ambiguous_operation.value if ambiguous_operation else None,
                    now,
                    outcome_received_at,
                    booking_key,
                ),
            )

    @staticmethod
    def _stored_booking(row: sqlite3.Row) -> StoredBooking:
        operation = row["ambiguous_operation"]
        return StoredBooking(
            booking_key=row["booking_key"],
            source_event_key=row["source_event_key"],
            batch_id=row["batch_id"],
            ordinal=row["ordinal"],
            source_event_timestamp=_required_utc(row["source_event_timestamp"]),
            state=WorkflowState(row["state"]),
            ambiguous_operation=Operation(operation) if operation else None,
            observation_id=row["observation_id"],
            backend_prediction_timestamp=_parse_utc(row["backend_prediction_timestamp"]),
            outcome_event_key=row["outcome_event_key"],
            outcome_label=row["outcome_label"],
            outcome_source_timestamp=_parse_utc(row["outcome_source_timestamp"]),
            backend_outcome_received_at=_parse_utc(
                row["backend_outcome_received_at"]
            ),
        )
