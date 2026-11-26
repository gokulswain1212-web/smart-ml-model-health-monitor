from __future__ import annotations

import hashlib
import hmac
from uuid import uuid4
from pydantic import SecretStr


class ProducerIdentity:
    """Purpose-separated HMAC identifiers; raw source IDs are never persisted."""

    def __init__(self, hmac_key: str | bytes | SecretStr) -> None:
        if isinstance(hmac_key, SecretStr):
            key = hmac_key.get_secret_value().encode("utf-8")
        elif isinstance(hmac_key, str):
            key = hmac_key.encode("utf-8")
        else:
            key = hmac_key
        if len(key) < 32:
            raise ValueError("producer mapping HMAC key must contain at least 32 bytes")
        self.__key = key

    def __repr__(self) -> str:
        return "ProducerIdentity(<redacted>)"

    def _digest(self, purpose: bytes, value: str | bytes) -> str:
        message = value.encode("utf-8") if isinstance(value, str) else value
        return hmac.new(
            self.__key,
            purpose + b"\x00" + message,
            hashlib.sha256,
        ).hexdigest()

    def booking_key(self, booking_id: str) -> str:
        return self._digest(b"producer-booking-map-v1", booking_id)

    def backend_booking_reference(self, booking_id: str) -> str:
        return "producer-" + self._digest(b"backend-booking-reference-v1", booking_id)

    def source_event_key(self, event_id: str, *, event_kind: str) -> str:
        return self._digest(
            f"producer-source-event-v1:{event_kind}".encode("ascii"),
            event_id,
        )

    def batch_digest(self, canonical_payload: bytes) -> str:
        return self._digest(b"producer-batch-payload-v1", canonical_payload)


def new_batch_id() -> str:
    """Create a stable-format, non-identifying batch ID for a durable outbox."""
    return "batch-" + uuid4().hex
