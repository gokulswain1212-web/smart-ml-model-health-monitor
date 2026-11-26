from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from pydantic import SecretStr


@dataclass(frozen=True, repr=False)
class ProducerSettings:
    backend_base_url: str
    ingestion_token: SecretStr = field(repr=False)
    mapping_hmac_key: SecretStr = field(repr=False)

    def __post_init__(self) -> None:
        parsed = urlsplit(self.backend_base_url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("producer backend URL must use HTTPS")
        if parsed.query or parsed.fragment:
            raise ValueError("producer backend URL must not include query or fragment data")
        if len(self.ingestion_token.get_secret_value()) < 32:
            raise ValueError("production ingestion token must contain at least 32 characters")
        if len(self.mapping_hmac_key.get_secret_value().encode("utf-8")) < 32:
            raise ValueError("producer mapping HMAC key must contain at least 32 bytes")

    @classmethod
    def from_environment(cls) -> ProducerSettings:
        required = {
            "PRODUCER_BACKEND_BASE_URL": os.getenv("PRODUCER_BACKEND_BASE_URL"),
            "PRODUCTION_INGESTION_TOKEN": os.getenv("PRODUCTION_INGESTION_TOKEN"),
            "PRODUCER_MAPPING_HMAC_KEY": os.getenv("PRODUCER_MAPPING_HMAC_KEY"),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(
                "Missing producer configuration: " + ", ".join(sorted(missing))
            )
        return cls(
            backend_base_url=required["PRODUCER_BACKEND_BASE_URL"] or "",
            ingestion_token=SecretStr(
                required["PRODUCTION_INGESTION_TOKEN"] or ""
            ),
            mapping_hmac_key=SecretStr(
                required["PRODUCER_MAPPING_HMAC_KEY"] or ""
            ),
        )
