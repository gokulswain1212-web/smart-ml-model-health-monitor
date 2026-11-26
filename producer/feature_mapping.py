from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from producer.schemas import BookingEvent, FEATURE_NAMES, Model1Features


@dataclass(frozen=True)
class FeatureMapper:
    """Explicit target-feature to provider-field mapping."""

    model1_to_source: Mapping[str, str]
    mapping_version: str = "mapping-v1"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.mapping_version, str)
            or not self.mapping_version
            or self.mapping_version != self.mapping_version.strip()
            or len(self.mapping_version) > 64
        ):
            raise ValueError("mapping_version must be non-empty and at most 64 characters")
        mapping = dict(self.model1_to_source)
        if set(mapping) != set(FEATURE_NAMES):
            missing = sorted(set(FEATURE_NAMES) - set(mapping))
            unexpected = sorted(set(mapping) - set(FEATURE_NAMES))
            raise ValueError(
                f"Feature mapping must cover exactly Model 1's 28 fields; "
                f"missing={missing}, unexpected={unexpected}."
            )
        if any(not isinstance(source, str) or not source.strip() for source in mapping.values()):
            raise ValueError("Every Model 1 feature must map to a non-empty source field.")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("Each Model 1 feature must map to a distinct source field.")
        object.__setattr__(
            self,
            "model1_to_source",
            MappingProxyType(
                {feature: mapping[feature] for feature in FEATURE_NAMES}
            ),
        )

    @classmethod
    def identity(cls) -> FeatureMapper:
        """Use when the upstream adapter already emits canonical feature names."""
        return cls({feature: feature for feature in FEATURE_NAMES})

    def map_features(self, source_features: Mapping[str, object]) -> Model1Features:
        missing_source = [
            source_field
            for source_field in self.model1_to_source.values()
            if source_field not in source_features
        ]
        if missing_source:
            raise ValueError(
                f"Booking is missing mapped source fields: {sorted(missing_source)}."
            )
        canonical = {
            feature: source_features[source_field]
            for feature, source_field in self.model1_to_source.items()
        }
        return Model1Features.model_validate(canonical)

    def map_event(self, event: BookingEvent) -> Model1Features:
        return self.map_features(event.features)
