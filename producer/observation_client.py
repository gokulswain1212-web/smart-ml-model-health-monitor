from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Sequence

from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError

from producer.contracts import ObservationReceipt, Operation, PreparedObservation
from producer.errors import SubmissionAmbiguous
from producer.http_common import JsonApiClient
from producer.schemas import FEATURE_NAMES

BATCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class _ObservationResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    batch_id: StrictStr
    observation_ids: list[StrictStr]
    prediction_timestamp: datetime


class ObservationApiClient(JsonApiClient):
    """Client for the existing producer-token authenticated observation route."""

    def submit(
        self,
        batch_id: str,
        observations: Sequence[PreparedObservation],
    ) -> ObservationReceipt:
        if not BATCH_ID_PATTERN.fullmatch(batch_id):
            raise ValueError("batch_id must use the backend's safe identifier format")
        if not observations:
            raise ValueError("an observation batch must contain at least one booking")
        if len(observations) > 10000:
            raise ValueError("an observation batch cannot exceed 10000 bookings")

        records: list[dict[str, object]] = []
        for observation in observations:
            feature_values = observation.features.model_dump(mode="python")
            if tuple(feature_values) != FEATURE_NAMES:
                raise ValueError("mapped features do not match the Model 1 feature order")
            record: dict[str, object] = {
                "booking_id": observation.backend_booking_reference,
            }
            record.update(feature_values)
            records.append(record)

        body = self._post_json(
            Operation.OBSERVATION,
            "/api/v1/monitoring/observations",
            {
                "batch_id": batch_id,
                "data_source_type": "real",
                "records": records,
            },
        )
        try:
            parsed = _ObservationResponse.model_validate(body)
            timestamp = parsed.prediction_timestamp
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("prediction timestamp must include a timezone")
            observation_ids = tuple(parsed.observation_ids)
            if (
                parsed.batch_id != batch_id
                or len(observation_ids) != len(observations)
                or any(not item for item in observation_ids)
                or len(set(observation_ids)) != len(observation_ids)
            ):
                raise ValueError("observation response does not match the submitted batch")
        except (ValidationError, ValueError, TypeError):
            raise SubmissionAmbiguous(Operation.OBSERVATION, status_code=200) from None
        return ObservationReceipt(
            batch_id=batch_id,
            observation_ids=observation_ids,
            prediction_timestamp=timestamp.astimezone(timezone.utc),
        )

    def __enter__(self) -> ObservationApiClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
