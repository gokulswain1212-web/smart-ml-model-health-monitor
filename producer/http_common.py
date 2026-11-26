from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import SecretStr

from producer.contracts import Operation
from producer.errors import SubmissionAmbiguous, SubmissionRejected


class JsonApiClient:
    def __init__(
        self,
        base_url: str,
        ingestion_token: str | SecretStr,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("producer backend URL must use HTTPS")
        if parsed.query or parsed.fragment:
            raise ValueError("producer backend URL must not include query or fragment data")
        token = (
            ingestion_token.get_secret_value()
            if isinstance(ingestion_token, SecretStr)
            else ingestion_token
        )
        if len(token) < 32:
            raise ValueError("production ingestion token must contain at least 32 characters")
        self._token = SecretStr(token)
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            follow_redirects=False,
            verify=True,
            transport=transport,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<credentials redacted>)"

    def close(self) -> None:
        self._http.close()

    def _post_json(
        self,
        operation: Operation,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            response = self._http.post(
                path,
                json=payload,
                headers={
                    "X-Production-Ingestion-Token": self._token.get_secret_value(),
                    "Content-Type": "application/json",
                },
                follow_redirects=False,
            )
        except httpx.RequestError:
            raise SubmissionAmbiguous(operation) from None

        if response.status_code in {408, 409} or response.status_code >= 500:
            raise SubmissionAmbiguous(
                operation,
                status_code=response.status_code,
            )
        if not response.is_success:
            raise SubmissionRejected(operation, response.status_code)
        try:
            body = response.json()
        except ValueError:
            raise SubmissionAmbiguous(
                operation,
                status_code=response.status_code,
            ) from None
        if not isinstance(body, dict):
            raise SubmissionAmbiguous(
                operation,
                status_code=response.status_code,
            )
        return body
