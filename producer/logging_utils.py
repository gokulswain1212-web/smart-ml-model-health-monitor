from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ProducerEvent(str, Enum):
    BOOKING_RECEIVED = "booking_received"
    BOOKING_VALIDATED = "booking_validated"
    OBSERVATION_SUBMISSION_STARTED = "observation_submission_started"
    OBSERVATION_ACCEPTED = "observation_accepted"
    OUTCOME_SUBMISSION_STARTED = "outcome_submission_started"
    OUTCOME_COMPLETED = "outcome_completed"
    SUBMISSION_AMBIGUOUS = "submission_ambiguous"
    SUBMISSION_REJECTED = "submission_rejected"


SAFE_CONTEXT_FIELDS = {
    "correlation_id",
    "operation",
    "state",
    "status_code",
    "attempt",
    "record_count",
}
SAFE_OPERATIONS = {"observation", "outcome"}
SAFE_STATES = {
    "RECEIVED",
    "VALIDATED",
    "OBSERVATION_PENDING",
    "OBSERVATION_ACCEPTED",
    "AWAITING_OUTCOME",
    "OUTCOME_PENDING",
    "COMPLETED",
    "FAILED",
    "AMBIGUOUS",
}
_CORRELATION_ID = re.compile(r"^[A-Fa-f0-9-]{16,64}$")
_SAFE_EVENTS = {event.value for event in ProducerEvent}


class SafeJsonFormatter(logging.Formatter):
    """Serialize an allowlist only; message, args, IDs, payloads, and secrets are omitted."""

    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "producer_event", "unclassified")
        if not isinstance(event, str) or event not in _SAFE_EVENTS:
            event = "unclassified"
        context = getattr(record, "producer_context", {})
        if not isinstance(context, dict):
            context = {}
        safe_context = {
            key: value
            for key, value in context.items()
            if key in SAFE_CONTEXT_FIELDS and self._is_safe_value(key, value)
        }
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": event,
        }
        payload.update(safe_context)
        return json.dumps(payload, separators=(",", ":"), allow_nan=False)

    @staticmethod
    def _is_safe_value(key: str, value: object) -> bool:
        if key == "correlation_id":
            return isinstance(value, str) and bool(_CORRELATION_ID.fullmatch(value))
        if key in {"status_code", "attempt", "record_count"}:
            return isinstance(value, int) and not isinstance(value, bool)
        if key in {"operation", "state"}:
            if not isinstance(value, str):
                return False
            return (
                value in SAFE_OPERATIONS
                if key == "operation"
                else value in SAFE_STATES
            )
        return False


def emit_event(
    logger: logging.Logger,
    event: ProducerEvent,
    **context: object,
) -> None:
    safe_context = {
        key: value
        for key, value in context.items()
        if key in SAFE_CONTEXT_FIELDS
        and SafeJsonFormatter._is_safe_value(key, value)
    }
    logger.info(
        event.value,
        extra={"producer_event": event.value, "producer_context": safe_context},
    )


def configure_logger(name: str = "producer") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(SafeJsonFormatter())
        logger.addHandler(handler)
    return logger
