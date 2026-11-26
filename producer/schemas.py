from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES_PATH = ROOT / "ml" / "models" / "feature_names.json"
FEATURE_NAMES = tuple(
    json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
)

NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
PositiveInt = Annotated[StrictInt, Field(ge=1)]
WeekNumber = Annotated[StrictInt, Field(ge=1, le=53)]
DayOfMonth = Annotated[StrictInt, Field(ge=1, le=31)]
BinaryInt = Annotated[StrictInt, Field(ge=0, le=1)]

CATEGORICAL_FEATURES = (
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
)


class Model1Features(BaseModel):
    """Canonical Model 1 input with no coercion, omission, or extra fields."""

    model_config = ConfigDict(extra="forbid", strict=True)

    hotel: StrictStr
    lead_time: NonNegativeInt
    arrival_date_year: PositiveInt
    arrival_date_month: StrictStr
    arrival_date_week_number: WeekNumber
    arrival_date_day_of_month: DayOfMonth
    stays_in_weekend_nights: NonNegativeInt
    stays_in_week_nights: NonNegativeInt
    adults: NonNegativeInt
    children: NonNegativeInt
    babies: NonNegativeInt
    meal: StrictStr
    country: StrictStr
    market_segment: StrictStr
    distribution_channel: StrictStr
    is_repeated_guest: BinaryInt
    previous_cancellations: NonNegativeInt
    previous_bookings_not_canceled: NonNegativeInt
    reserved_room_type: StrictStr
    assigned_room_type: StrictStr
    booking_changes: NonNegativeInt
    deposit_type: StrictStr
    agent: NonNegativeInt
    days_in_waiting_list: NonNegativeInt
    customer_type: StrictStr
    adr: StrictFloat
    required_car_parking_spaces: NonNegativeInt
    total_of_special_requests: NonNegativeInt

    @field_validator(*CATEGORICAL_FEATURES)
    @classmethod
    def categorical_values_must_be_nonempty(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("categorical feature values must be non-empty and trimmed")
        return value

    @field_validator("adr", mode="before")
    @classmethod
    def validate_adr(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("adr must be a finite non-negative number")
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError("adr must be a finite non-negative number")
        return float(value)


if tuple(Model1Features.model_fields) != FEATURE_NAMES:
    raise RuntimeError(
        "Producer Model1Features field order does not match "
        "ml/models/feature_names.json."
    )


def _validate_identifier(value: str, *, field_name: str) -> str:
    if not value or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be non-empty and trimmed")
    if len(value) > 256 or any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field_name} contains invalid characters or is too long")
    return value


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("source timestamps must include a timezone")
    return value.astimezone(timezone.utc)


class BookingEvent(BaseModel):
    """Provider-neutral envelope for a newly observed booking."""

    model_config = ConfigDict(extra="forbid")

    source_event_id: StrictStr = Field(min_length=1, max_length=256)
    booking_id: StrictStr = Field(min_length=1, max_length=256)
    source_event_timestamp: datetime
    features: dict[str, Any]

    @field_validator("source_event_id", "booking_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _validate_identifier(value, field_name=info.field_name)

    @field_validator("source_event_timestamp")
    @classmethod
    def normalize_source_timestamp(cls, value: datetime) -> datetime:
        return _normalize_utc(value)


class FinalOutcomeEvent(BaseModel):
    """Canonical final-status event; pending statuses have no representation."""

    model_config = ConfigDict(extra="forbid")

    source_event_id: StrictStr = Field(min_length=1, max_length=256)
    booking_id: StrictStr = Field(min_length=1, max_length=256)
    source_event_timestamp: datetime
    final_status: Literal["CANCELLED", "NOT_CANCELLED"]

    @field_validator("source_event_id", "booking_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _validate_identifier(value, field_name=info.field_name)

    @field_validator("source_event_timestamp")
    @classmethod
    def normalize_source_timestamp(cls, value: datetime) -> datetime:
        return _normalize_utc(value)

    @property
    def is_canceled(self) -> int:
        return int(self.final_status == "CANCELLED")
