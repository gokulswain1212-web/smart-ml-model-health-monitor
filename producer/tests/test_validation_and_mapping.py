from __future__ import annotations

import pytest
from pydantic import ValidationError

from producer.feature_mapping import FeatureMapper
from producer.schemas import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    BookingEvent,
    Model1Features,
)


def synthetic_features() -> dict[str, object]:
    values: dict[str, object] = {}
    for field in FEATURE_NAMES:
        if field in CATEGORICAL_FEATURES:
            values[field] = "SYNTHETIC"
        elif field == "adr":
            values[field] = 125.5
        elif field == "arrival_date_year":
            values[field] = 2025
        elif field == "arrival_date_week_number":
            values[field] = 10
        elif field == "arrival_date_day_of_month":
            values[field] = 12
        else:
            values[field] = 0
    return values


def synthetic_event(
    features: dict[str, object] | None = None,
) -> BookingEvent:
    return BookingEvent.model_validate(
        {
            "source_event_id": "SYNTHETIC-EVENT-001",
            "booking_id": "SYNTHETIC-BOOKING-001",
            "source_event_timestamp": "2025-03-12T10:00:00Z",
            "features": features if features is not None else synthetic_features(),
        }
    )


def test_canonical_model1_schema_has_exactly_28_features() -> None:
    assert len(FEATURE_NAMES) == 28
    assert tuple(Model1Features.model_fields) == FEATURE_NAMES
    assert tuple(Model1Features.model_validate(synthetic_features()).model_dump()) == FEATURE_NAMES


def test_model1_features_reject_missing_extra_and_coerced_values() -> None:
    missing = synthetic_features()
    missing.pop("lead_time")
    with pytest.raises(ValidationError):
        Model1Features.model_validate(missing)

    extra = synthetic_features()
    extra["guest_email"] = "synthetic@example.test"
    with pytest.raises(ValidationError):
        Model1Features.model_validate(extra)

    coerced = synthetic_features()
    coerced["lead_time"] = "3"
    with pytest.raises(ValidationError):
        Model1Features.model_validate(coerced)


def test_feature_mapper_requires_exact_one_to_one_mapping() -> None:
    with pytest.raises(ValueError, match="cover exactly"):
        FeatureMapper({"hotel": "provider_hotel"})

    duplicate_source = {field: field for field in FEATURE_NAMES}
    duplicate_source["lead_time"] = "hotel"
    with pytest.raises(ValueError, match="distinct source field"):
        FeatureMapper(duplicate_source)

    mapper = FeatureMapper.identity()
    with pytest.raises(TypeError):
        mapper.model1_to_source["hotel"] = "mutated_source"  # type: ignore[index]


def test_feature_mapper_maps_explicit_provider_fields_in_canonical_order() -> None:
    mapping = {field: f"provider_{field}" for field in FEATURE_NAMES}
    provider_features = {
        f"provider_{field}": value
        for field, value in synthetic_features().items()
    }
    mapped = FeatureMapper(mapping).map_features(provider_features)
    assert tuple(mapped.model_dump()) == FEATURE_NAMES
    assert mapped.lead_time == 0
    with pytest.raises(ValueError, match="missing mapped source fields"):
        FeatureMapper(mapping).map_event(synthetic_event())


def test_booking_event_rejects_naive_timestamps_and_untrimmed_ids() -> None:
    payload = {
        "source_event_id": "SYNTHETIC-EVENT-001",
        "booking_id": "SYNTHETIC-BOOKING-001",
        "source_event_timestamp": "2025-03-12T10:00:00",
        "features": synthetic_features(),
    }
    with pytest.raises(ValidationError):
        BookingEvent.model_validate(payload)

    payload["source_event_timestamp"] = "2025-03-12T10:00:00Z"
    payload["booking_id"] = " SYNTHETIC-BOOKING-001"
    with pytest.raises(ValidationError):
        BookingEvent.model_validate(payload)
