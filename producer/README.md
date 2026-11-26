# Production producer foundation

This package is a provider-independent foundation, not a running connector. It
does not poll a hotel system, expose a webhook, start a worker, or submit data
until an application explicitly constructs `ProducerWorkflow` and calls it.
No live provider or backend calls are made by the tests.

## Components

- `schemas.py` defines booking and final-outcome envelopes and the strict,
  canonical 28-feature Model 1 input.
- `feature_mapping.py` requires an explicit one-to-one mapping from every
  canonical Model 1 feature to a provider field.
- `observation_client.py` and `outcome_client.py` call the existing authenticated
  monitoring endpoints over HTTPS.
- `storage.py` stores workflow state, HMAC-keyed source identifiers, returned
  observation IDs, batch IDs, labels, and source/backend timestamps in SQLite.
  Raw booking IDs and feature payloads are not persisted.
- `workflow.py` coordinates durable state transitions and leaves ambiguous
  submissions for explicit reconciliation rather than retrying automatically.
- `interfaces.py` defines provider, API-client, storage, and retry boundaries;
  no concrete hotel provider or retry daemon is included.

## Configuration and integration boundary

Construct `ProducerSettings` from server-side environment variables:

- `PRODUCER_BACKEND_BASE_URL`: HTTPS base URL for the monitoring API.
- `PRODUCTION_INGESTION_TOKEN`: server-side API credential.
- `PRODUCER_MAPPING_HMAC_KEY`: independent random key (at least 32 bytes) used
  to pseudonymize source identifiers and fingerprint batches.

Do not put either key in frontend code, logs, source control, or booking
payloads. Keep the SQLite file on access-controlled durable storage and back it
up securely; the database is not application-encrypted by this package. Losing
or rotating the HMAC key makes existing keyed mappings unresolvable; plan a
controlled migration before rotation.

A provider adapter must map only authorized, genuine booking events into
`BookingEvent` and `FinalOutcomeEvent`. It must retain the source event IDs and
timestamps and persist a non-identifying batch ID (for example,
`new_batch_id()`) with its outbox record before submission. Reuse that exact
batch ID and event payload during recovery. Outcomes may be sent only after the
workflow has stored the backend-returned observation ID.
`FinalOutcomeEvent` represents only a finalized cancellation or
non-cancellation, not a pending booking status.

## Workflow and recovery

The workflow uses `RECEIVED`, `VALIDATED`, `OBSERVATION_PENDING`,
`OBSERVATION_ACCEPTED`, `AWAITING_OUTCOME`, `OUTCOME_PENDING`, `COMPLETED`,
`FAILED`, and `AMBIGUOUS`. Timeouts, connection failures, conflict/timeout
responses, server errors, and malformed success responses are treated as
ambiguous. Do not retry an ambiguous operation with a new batch ID. An operator
or a future provider-specific reconciler must establish whether the backend
accepted the request, then call the corresponding reconciliation method with
the original stable identifiers. Retrying an observation after absence is
confirmed reuses the same batch ID and payload digest.

The current monitoring API does not expose a producer idempotency or
reconciliation endpoint. Until one is added, resolving ambiguity requires an
authorized operator to verify backend state through an approved secure
operational process; the producer does not guess or automatically resend.

Source event timestamps are stored separately from backend prediction and
outcome-receipt timestamps. The current backend does not accept producer source
timestamps, so the producer preserves them locally and sends only the fields
accepted by the existing API.

Structured logs contain an allowlist of event, state, operation, status, count,
and random correlation ID fields. They omit messages, booking IDs, observation
IDs, feature values, tokens, and HMAC keys.

## Isolated tests

Run from the repository root with the workspace virtual environment:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest producer\tests
```

Fixtures are synthetic and all HTTP tests use `httpx.MockTransport`. The tests
do not contact a hotel system or the monitoring backend and do not create
production evidence.
