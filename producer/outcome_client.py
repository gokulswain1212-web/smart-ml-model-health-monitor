from __future__ import annotations

from datetime import datetime, timezone
import re

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError

from producer.contracts import Operation, OutcomeReceipt
from producer.errors import SubmissionAmbiguous
from producer.http_common import JsonApiClient

MAX_OBSERVATION_ID_LENGTH = 128
BATCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class _OutcomeResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    observation_id: StrictStr = Field(
        min_length=1,
        max_length=MAX_OBSERVATION_ID_LENGTH,
    )
    batch_id: StrictStr = Field(
        min_length=1,
        max_length=128,
        pattern=BATCH_ID_PATTERN.pattern,
    )
    outcome_received_at: datetime


class OutcomeApiClient(JsonApiClient):
    """Client for the existing producer-token authenticated outcome route."""

    def submit(self, observation_id: str, is_canceled: int) -> OutcomeReceipt:
        if (
            not isinstance(observation_id, str)
            or not observation_id
            or len(observation_id) > MAX_OBSERVATION_ID_LENGTH
        ):
            raise ValueError(
                "observation_id must contain 1 to 128 characters"
            )
        if type(is_canceled) is not int or is_canceled not in (0, 1):
            raise ValueError("is_canceled must be the integer 0 or 1")
        body = self._post_json(
            Operation.OUTCOME,
            "/api/v1/monitoring/outcomes",
            {
                "observation_id": observation_id,
                "is_canceled": is_canceled,
            },
        )
        try:
            parsed = _OutcomeResponse.model_validate(body)
            timestamp = parsed.outcome_received_at
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("outcome receipt timestamp must include a timezone")
            if parsed.observation_id != observation_id:
                raise ValueError("outcome response does not match the submitted observation")
        except (ValidationError, ValueError, TypeError):
            raise SubmissionAmbiguous(Operation.OUTCOME, status_code=200) from None
        return OutcomeReceipt(
            batch_id=parsed.batch_id,
            observation_id=parsed.observation_id,
            outcome_received_at=timestamp.astimezone(timezone.utc),
        )

    def __enter__(self) -> OutcomeApiClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
