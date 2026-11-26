from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from producer.contracts import PreparedObservation
from producer.errors import SubmissionAmbiguous, SubmissionRejected
from producer.feature_mapping import FeatureMapper
from producer.identity import ProducerIdentity
from producer.observation_client import ObservationApiClient
from producer.outcome_client import OutcomeApiClient
from producer.schemas import CATEGORICAL_FEATURES, FEATURE_NAMES

SYNTHETIC_TOKEN = "SYNTHETIC_TEST_INGESTION_TOKEN_DO_NOT_USE_0001"
SYNTHETIC_KEY = "SYNTHETIC_TEST_KEY_DO_NOT_USE_IN_PRODUCTION_0001"


def synthetic_features():
    values = {}
    for field in FEATURE_NAMES:
        if field in CATEGORICAL_FEATURES:
            values[field] = "SYNTHETIC"
        elif field == "adr":
            values[field] = 99.5
        elif field == "arrival_date_year":
            values[field] = 2025
        elif field == "arrival_date_week_number":
            values[field] = 11
        elif field == "arrival_date_day_of_month":
            values[field] = 12
        else:
            values[field] = 0
    return FeatureMapper.identity().map_features(values)


def test_observation_client_posts_only_to_mock_transport() -> None:
    identity = ProducerIdentity(SYNTHETIC_KEY)
    prepared = PreparedObservation(
        backend_booking_reference=identity.backend_booking_reference(
            "SYNTHETIC-BOOKING-1"
        ),
        features=synthetic_features(),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/monitoring/observations"
        assert request.headers["X-Production-Ingestion-Token"] == SYNTHETIC_TOKEN
        body = json.loads(request.content)
        assert body["data_source_type"] == "real"
        assert body["batch_id"] == "SYNTHETIC-BATCH-1"
        assert tuple(body["records"][0])[1:] == FEATURE_NAMES
        return httpx.Response(
            200,
            json={
                "batch_id": "SYNTHETIC-BATCH-1",
                "observation_ids": ["SYNTHETIC-OBSERVATION-1"],
                "prediction_timestamp": "2025-03-12T12:00:00Z",
            },
        )

    api = ObservationApiClient(
        "https://monitor.synthetic.invalid",
        SYNTHETIC_TOKEN,
        transport=httpx.MockTransport(handler),
    )
    receipt = api.submit("SYNTHETIC-BATCH-1", [prepared])
    assert receipt.observation_ids == ("SYNTHETIC-OBSERVATION-1",)
    assert receipt.prediction_timestamp == datetime(
        2025, 3, 12, 12, 0, tzinfo=timezone.utc
    )
    api.close()


@pytest.mark.parametrize("value", [True, -1, 2, "1"])
def test_outcome_client_rejects_non_binary_labels_without_http(value: object) -> None:
    api = OutcomeApiClient(
        "https://monitor.synthetic.invalid",
        SYNTHETIC_TOKEN,
        transport=httpx.MockTransport(
            lambda request: pytest.fail("invalid outcome must not make an HTTP request")
        ),
    )
    with pytest.raises(ValueError, match="integer 0 or 1"):
        api.submit("SYNTHETIC-OBSERVATION-1", value)  # type: ignore[arg-type]
    api.close()


@pytest.mark.parametrize("observation_id", ["", "x" * 129])
def test_outcome_client_matches_backend_observation_id_length_limit(
    observation_id: str,
) -> None:
    api = OutcomeApiClient(
        "https://monitor.synthetic.invalid",
        SYNTHETIC_TOKEN,
        transport=httpx.MockTransport(
            lambda request: pytest.fail("invalid observation ID must not make an HTTP request")
        ),
    )
    with pytest.raises(ValueError, match="1 to 128 characters"):
        api.submit(observation_id, 0)
    api.close()


def test_outcome_client_uses_only_mocked_route_and_redacts_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/monitoring/outcomes"
        assert request.headers["X-Production-Ingestion-Token"] == SYNTHETIC_TOKEN
        return httpx.Response(
            200,
            json={
                "observation_id": "SYNTHETIC-OBSERVATION-1",
                "batch_id": "SYNTHETIC-BATCH-1",
                "outcome_received_at": "2025-03-12T12:01:00Z",
            },
        )

    api = OutcomeApiClient(
        "https://monitor.synthetic.invalid",
        SYNTHETIC_TOKEN,
        transport=httpx.MockTransport(handler),
    )
    receipt = api.submit("SYNTHETIC-OBSERVATION-1", 0)
    assert receipt.batch_id == "SYNTHETIC-BATCH-1"
    assert receipt.observation_id == "SYNTHETIC-OBSERVATION-1"
    assert SYNTHETIC_TOKEN not in repr(api)
    api.close()


def test_server_error_is_ambiguous_and_redirect_is_not_followed() -> None:
    for status_code, expected_error in [
        (503, SubmissionAmbiguous),
        (307, SubmissionRejected),
    ]:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                status_code,
                headers={"Location": "https://attacker.synthetic.invalid/"},
            )

        api = OutcomeApiClient(
            "https://monitor.synthetic.invalid",
            SYNTHETIC_TOKEN,
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(expected_error):
            api.submit("SYNTHETIC-OBSERVATION-1", 1)
        assert len(requests) == 1
        assert requests[0].headers["X-Production-Ingestion-Token"] == SYNTHETIC_TOKEN
        api.close()


@pytest.mark.parametrize(
    "url",
    [
        "http://monitor.synthetic.invalid",
        "https://user:password@monitor.synthetic.invalid",
    ],
)
def test_api_clients_reject_insecure_or_credential_bearing_urls(url: str) -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        OutcomeApiClient(url, SYNTHETIC_TOKEN)
