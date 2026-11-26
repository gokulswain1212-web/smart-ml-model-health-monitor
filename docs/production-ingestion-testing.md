# Production ingestion: safe testing and provider workflow

This guide covers the existing server-to-server ingestion API. A hotel/PMS
provider has not been selected or authorized, so **do not submit observation
or outcome requests to the running API yet**. The production endpoints mark
accepted requests as `source: "production"` and `data_source_type: "real"`;
they cannot independently prove that the caller's data came from a real
provider. A valid token is not a substitute for provider authorization.

Use the isolated automated tests below for development. They use synthetic
test fixtures and temporary storage; do not copy fixture records into a
production request.

## What the endpoints require

Both `POST` routes require this header:

```http
X-Production-Ingestion-Token: <PRODUCTION_INGESTION_TOKEN>
```

The backend must have `PRODUCTION_INGESTION_TOKEN` configured with at least
32 characters. Keep it on the backend and authorized producer only; never put
it in React, Vite variables, browser requests, source control, or terminal
output. A missing backend token returns `503`; a missing or incorrect request
token returns `401`.

### Submit a prediction batch

`POST /api/v1/monitoring/observations` accepts:

```json
{
  "batch_id": "<unique-provider-batch-id>",
  "data_source_type": "real",
  "records": [
    {
      "booking_id": "<provider-booking-reference>",
      "hotel": "<provider-value>",
      "lead_time": "<provider-value>",
      "arrival_date_year": "<provider-value>",
      "arrival_date_month": "<provider-value>",
      "arrival_date_week_number": "<provider-value>",
      "arrival_date_day_of_month": "<provider-value>",
      "stays_in_weekend_nights": "<provider-value>",
      "stays_in_week_nights": "<provider-value>",
      "adults": "<provider-value>",
      "children": "<provider-value>",
      "babies": "<provider-value>",
      "meal": "<provider-value>",
      "country": "<provider-value>",
      "market_segment": "<provider-value>",
      "distribution_channel": "<provider-value>",
      "is_repeated_guest": "<provider-value>",
      "previous_cancellations": "<provider-value>",
      "previous_bookings_not_canceled": "<provider-value>",
      "reserved_room_type": "<provider-value>",
      "assigned_room_type": "<provider-value>",
      "booking_changes": "<provider-value>",
      "deposit_type": "<provider-value>",
      "agent": "<provider-value>",
      "days_in_waiting_list": "<provider-value>",
      "customer_type": "<provider-value>",
      "adr": "<provider-value>",
      "required_car_parking_spaces": "<provider-value>",
      "total_of_special_requests": "<provider-value>"
    }
  ]
}
```

This is a **shape-only template**, not a valid request: replace every
`<provider-value>` and identifier with authorized provider data before sending.
Each record must contain `booking_id` followed by the Model 1 fields in the
exact order listed in
[`ml/models/feature_names.json`](../ml/models/feature_names.json). Do not
include `is_canceled` or any other target/label in a prediction request.
`records` must contain 1–10,000 rows, with unique booking IDs within the batch.
`batch_id` is optional; when omitted, the API generates one.

The response contains the `batch_id`, prediction and monitoring timestamps,
and an `observation_ids` array in the same order as the submitted records. It
also returns per-prediction IDs and Model 1 metadata. The producer must
durably retain the mapping from each provider booking reference to its returned
`observation_id`. Do not rely on row order after the producer stores the
response.

### Submit actual outcomes

`POST /api/v1/monitoring/outcomes` accepts only an observation ID returned by
the production prediction route, plus the provider-confirmed integer outcome:

```json
{
  "observation_id": "<returned-observation-id>",
  "is_canceled": 0
}
```

`is_canceled` must be the integer `0` or `1` (not a string or boolean). Send an
outcome only after the provider can authoritatively determine the booking's
final cancellation status. Outcomes are linked by `observation_id`; the
backend resolves that ID to the saved prediction's `row_id` and `batch_id`.
Unknown IDs return `404`, invalid requests return `422`, and duplicate
outcomes return `409`.

Each accepted outcome updates the batch performance snapshot. Incomplete
batches remain unlabeled for Model 2. An accepted outcome also invokes the
existing Model 2 eligibility-gated training workflow; **do not send live
outcome requests if training must not run**. No training was run as part of
this guide or the tests.

## Safe development checks (no provider data)

From the project root, run the focused tests:

```powershell
.\.venv\Scripts\python.exe -m pytest ml\tests\test_performance_tracking.py -k "concurrent_duplicate_outcomes or outcome_matching_rejects or production_outcome_matches" --no-header -q
.\.venv\Scripts\python.exe -m pytest backend\tests\test_model1_api.py -k "production_observation or production_outcome or production_collection_to_complete_future_f1_label or training_status" --no-header -q
```

These tests exercise authentication, validation, prediction-to-outcome
linkage, duplicate rejection, provenance, partial outcomes, and the Model 2
status endpoint. Test-only records use temporary directories or mocked
services. Do not replace these fixtures with real records.

For a local API smoke check, start the backend using the existing local
configuration (this command does not print environment values):

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --env-file .env --host 127.0.0.1 --port 8000
```

Then check readiness and the generated request schema without sending any
observation or outcome:

```powershell
$Api = "http://127.0.0.1:8000"
Invoke-RestMethod -Uri "$Api/health" -Method Get
(Invoke-RestMethod -Uri "$Api/openapi.json" -Method Get).paths.Keys |
  Where-Object { $_ -like "/api/v1/monitoring/*" }
```

With no authorized provider, stop after these checks. Do not send synthetic
fixtures to either POST route, even on a developer machine: a successful
authenticated request is persisted with production provenance.

## Requests for a future authorized producer

Only use these requests after a provider is authorized, the connector is
configured, and the payload contains genuine provider records and outcomes.
Run the producer from a trusted server-side environment where the ingestion
token is available as `PRODUCTION_INGESTION_TOKEN`; do not run it in frontend
code or paste the token into a command literal.

```powershell
$Api = "https://<authorized-api-host>"
$Token = $env:PRODUCTION_INGESTION_TOKEN
if ([string]::IsNullOrWhiteSpace($Token)) {
  throw "PRODUCTION_INGESTION_TOKEN is not available to this trusted producer."
}
$Headers = @{ "X-Production-Ingestion-Token" = $Token }

# $ProviderBatch must come from the authorized provider connector.
# Preserve record order and exclude actual outcome/target fields.
$ObservationBody = [ordered]@{
  batch_id = $ProviderBatch.batch_id
  data_source_type = "real"
  records = @($ProviderBatch.records)
} | ConvertTo-Json -Depth 10

$Receipt = Invoke-RestMethod `
  -Uri "$Api/api/v1/monitoring/observations" `
  -Method Post `
  -Headers $Headers `
  -ContentType "application/json" `
  -Body $ObservationBody
$Receipt | Select-Object batch_id, prediction_timestamp, data_source_type,
  prediction_count, number_of_pending_outcomes
```

Persist `$Receipt.observation_ids` against the provider booking references
before processing outcomes. Submit only actual finalized provider outcomes:

```powershell
# $ProviderOutcome must be a final, provider-confirmed cancellation result.
$OutcomeBody = @{
  observation_id = $Receipt.observation_ids[$ProviderOutcome.record_index]
  is_canceled = [int]$ProviderOutcome.is_canceled
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "$Api/api/v1/monitoring/outcomes" `
  -Method Post `
  -Headers $Headers `
  -ContentType "application/json" `
  -Body $OutcomeBody
```

The producer must reconcile timeouts before retrying: a response timeout does
not prove that the outcome was not recorded. Do not blindly retry a request
that may have succeeded; a duplicate is rejected with `409`.

## Verify persisted data and eligibility

Use the local dashboard session (not the producer token) for read-only
verification:

```powershell
$Credential = Get-Credential
$LoginBody = @{
  username = $Credential.UserName
  password = $Credential.GetNetworkCredential().Password
} | ConvertTo-Json
$null = Invoke-RestMethod `
  -Uri "$Api/api/v1/auth/login" `
  -Method Post `
  -ContentType "application/json" `
  -Body $LoginBody `
  -SessionVariable DashboardSession

# The batch report must show data_source_type="real" and source="production".
Invoke-RestMethod `
  -Uri "$Api/api/v1/monitoring/reports/$($Receipt.batch_id)" `
  -Method Get `
  -WebSession $DashboardSession

# Verify returned observation IDs, timestamps, and any recorded actual outcomes.
Invoke-RestMethod `
  -Uri "$Api/api/v1/monitoring/observations?period_days=3650&limit=1000" `
  -Method Get `
  -WebSession $DashboardSession

# This is the authoritative Model 2 eligibility/status response.
Invoke-RestMethod `
  -Uri "$Api/api/v1/model2/training-status" `
  -Method Get `
  -WebSession $DashboardSession
```

The report and observation endpoints confirm the provenance recorded by the
backend; they do not independently authenticate the provider. Model 2 counts
distinct eligible monitoring batches, not booking rows. The minimum remains
20 fully labeled genuine production batches, with verified outcomes and both
target classes, plus the existing chronological and training-partition
checks. One batch containing 20 bookings is still only one batch.

Backend file-backed history is not a multi-worker transactional database.
Run a single API worker for this storage mode; the outcome duplicate guard
serializes writes within that process only. See
[the full API contract](./api.md#trusted-production-observation-and-outcome-collection).
