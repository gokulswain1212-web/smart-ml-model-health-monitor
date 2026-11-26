from __future__ import annotations

import json
import logging

from producer.logging_utils import SafeJsonFormatter


def test_structured_logging_omits_personal_data_secrets_and_unknown_values() -> None:
    record = logging.makeLogRecord(
        {
            "name": "producer",
            "levelname": "INFO",
            "msg": "SYNTHETIC-BOOKING-1 token=SYNTHETIC-SECRET",
            "producer_event": ["untrusted"],
            "producer_context": {
                "correlation_id": "a" * 32,
                "operation": "outcome",
                "state": "AMBIGUOUS",
                "booking_id": "SYNTHETIC-BOOKING-1",
                "token": "SYNTHETIC-SECRET",
                "status_code": 503,
            },
        }
    )
    formatted = SafeJsonFormatter().format(record)
    parsed = json.loads(formatted)
    assert parsed["event"] == "unclassified"
    assert parsed["operation"] == "outcome"
    assert parsed["state"] == "AMBIGUOUS"
    assert parsed["status_code"] == 503
    assert "SYNTHETIC-BOOKING-1" not in formatted
    assert "SYNTHETIC-SECRET" not in formatted


def test_structured_logging_rejects_unknown_state_and_operation() -> None:
    record = logging.makeLogRecord(
        {
            "name": "producer",
            "levelname": "INFO",
            "msg": "ignored",
            "producer_event": "booking_received",
            "producer_context": {
                "operation": "arbitrary",
                "state": "made_up",
            },
        }
    )
    parsed = json.loads(SafeJsonFormatter().format(record))
    assert "operation" not in parsed
    assert "state" not in parsed
