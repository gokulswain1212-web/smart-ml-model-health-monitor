# Backend API

Configure local credentials in `.env` and run the API from the project root:

```powershell
python -m uvicorn backend.main:app --env-file .env --reload --host 127.0.0.1 --port 8000
```

## Local development authentication

The local API requires `LOCAL_AUTH_USERNAME`, a password of at least 6
characters, and a random `LOCAL_AUTH_SECRET` of at least 32 characters.
`POST /api/v1/auth/login` accepts JSON username/password and sets a signed,
HttpOnly, SameSite=Lax session cookie with an eight-hour lifetime.
`GET /api/v1/auth/session` returns the authenticated username; `POST
/api/v1/auth/logout` clears the cookie. Dashboard and user-facing model and
monitoring routes require this session. `GET /health` is public and returns
readiness only. The two production ingestion routes use the separate producer
token instead of the browser session. Login requests with a browser `Origin`
outside `CORS_ORIGINS` are rejected.

This local session mechanism is not for public deployment. OIDC is a future
integration; do not expose the development API publicly.

## Route inventory and access

| Method | Route | Access |
|---|---|---|
| `GET` | `/health` | Public readiness |
| `POST` | `/api/v1/auth/login` | Local credentials; browser origin checked |
| `GET` | `/api/v1/auth/session` | Local session |
| `POST` | `/api/v1/auth/logout` | Browser origin checked |
| `GET` | `/api/v1/dashboard` | Local session |
| `GET` | `/api/v1/model2/training-status` | Local session |
| `GET` | `/api/v1/model2/offline/status` | Local session |
| `POST` | `/api/v1/model2/offline/predict` | Local session |
| `GET` | `/api/v1/monitoring/ingestion-status` | Local session |
| `POST` | `/api/v1/monitoring/observations` | Production ingestion token |
| `POST` | `/api/v1/monitoring/outcomes` | Production ingestion token |
| `GET` | `/api/v1/monitoring/observations` | Local session |
| `GET` | `/api/v1/monitoring/performance` | Local session |
| `GET` | `/api/v1/alerts` | Local session |
| `GET` | `/api/v1/monitoring/reports` | Local session |
| `GET` | `/api/v1/monitoring/reports/{batch_id}` | Local session |
| `POST` | `/api/v1/monitoring/batches/csv` | Local session |
| `POST` | `/api/v1/predictions/cancellation` | Local session |
| `POST` | `/api/v1/predictions/cancellation/csv` | Local session |
| `POST` | `/api/v1/outcomes/{batch_id}/csv` | Local session |
| `POST` | `/api/v1/predictions/performance-degradation` | Local session |
| `POST` | `/api/v1/model2/development/predict` | Local session |

Protected routes return `401` without the required session or producer token.
Validation failures generally return `422`; route-specific conflict, missing
artifact, missing record, and size-limit responses are described below.

`GET /health` returns application, Model 1, Model 2, feature-schema, and
production-ingestion readiness. `production_ingestion_configured` is a boolean
only; the token is never returned.

## Model 1 cancellation predictions

`POST /api/v1/predictions/cancellation` accepts a non-empty batch of Model 1
feature records. Each record must contain exactly the feature columns listed
in `ml/models/feature_names.json`, with optional `row_id` and `generated_at`
metadata. Supply a stable booking identifier as `row_id` if actual outcomes
will be imported later; otherwise the API generates a batch-local row ID and
the returned ID must be retained for outcome matching. The API assigns the
supplied `batch_id` to each record, applies the normal data-quality checks,
and logs predictions for later outcome tracking.

```json
{
  "batch_id": "hotel-bookings-2026-11-03",
  "records": [
    {
      "row_id": "booking-1001",
      "hotel": "City Hotel",
      "lead_time": 12,
      "arrival_date_year": 2026,
      "arrival_date_month": "November",
      "arrival_date_week_number": 45,
      "arrival_date_day_of_month": 3,
      "stays_in_weekend_nights": 1,
      "stays_in_week_nights": 2,
      "adults": 2,
      "children": 0,
      "babies": 0,
      "meal": "BB",
      "country": "PRT",
      "market_segment": "Online TA",
      "distribution_channel": "TA/TO",
      "is_repeated_guest": 0,
      "previous_cancellations": 0,
      "previous_bookings_not_canceled": 0,
      "reserved_room_type": "A",
      "assigned_room_type": "A",
      "booking_changes": 0,
      "deposit_type": "No Deposit",
      "agent": 0,
      "days_in_waiting_list": 0,
      "customer_type": "Transient",
      "adr": 100,
      "required_car_parking_spaces": 0,
      "total_of_special_requests": 1
    }
  ]
}
```

The response contains the predicted class (`1` means canceled), positive-class
probability, row ID, model version, artifact digest, and batch quality status.
Repeated use of a batch ID returns `409` rather than overwriting its logged
predictions. Invalid feature schemas and critically poor-quality batches return
`422`; unavailable model/configuration artifacts return `503`.

The browser workflow can upload a feature-only CSV using:

```text
POST /api/v1/predictions/cancellation/csv?batch_id=hotel-batch-2026-11
Content-Type: text/csv
```

Batch IDs must start with a letter or digit and contain at most 128 letters,
digits, dots, underscores, or hyphens. Prediction CSV uploads are limited to
10 MiB and 10,000 rows.

## Run monitoring on a CSV batch

```text
POST /api/v1/monitoring/batches/csv?batch_id=hotel-batch-2026-11&data_source_type=unknown
Content-Type: text/csv
```

The optional `batch_id` column in the CSV must match the query parameter; if
omitted, the API assigns that batch ID. `data_source_type` must be `real`,
`synthetic_test`, or `unknown` and defaults to `unknown`. A `real` value is an
operator declaration and is not independently verified. CSVs are limited to
10 MiB. The endpoint stores a monitoring report and corresponding alerts and
returns the report status and timestamp.

## Monitoring report history and download

`GET /api/v1/monitoring/reports?period_days=30&source=REAL&limit=100`
lists persisted reports, filterable by source (`REAL`, `SIMULATED`, or
`UNKNOWN`). With no source filter, all three are returned with provenance
preserved. Each item includes quality-rate and warning summaries, plus
feature-level PSI and KS statistics for historical dashboard trends.
`GET /api/v1/monitoring/reports/{batch_id}` downloads the latest report JSON for
that exact batch ID. Missing reports return `404`; invalid batch IDs return
`422`. The downloaded report preserves its source classification.

## Import actual cancellation outcomes

For a legacy or other non-production batch whose monitoring report is
explicitly marked `real`, import observed booking outcomes with:

```text
POST /api/v1/outcomes/{batch_id}/csv
Content-Type: text/csv

row_id,is_canceled
booking-1001,0
booking-1002,1
```

`row_id` must match a stored prediction in that batch. The optional `batch_id`
CSV column, when present, must match the path. Labels must be exactly `0` or
`1`; blank IDs, duplicate IDs, unmatched prediction IDs, and previously
recorded outcomes are rejected. The complete CSV is validated before any rows
are stored. Imports for seeded simulations, unknown-source batches, or batches
without an explicit real monitoring report are rejected. CSV-imported outcomes
are not trusted production evidence and cannot label observations for Model 2
production training.

Accepted outcomes are appended to
`data/monitoring/outcomes/{batch_id}.jsonl` with their receipt timestamp and
CSV-import provenance. A timestamped evaluation snapshot is written to
`data/monitoring/performance/`; accuracy, precision, recall, F1, and confusion
matrix use only outcomes matched to predictions. ROC-AUC is reported only when
both actual classes and valid prediction probabilities are present.
Re-importing a row is a conflict (`409`); malformed data and schema/label
errors return `422`, and a batch with no predictions returns `404`.

## Trusted production observation and outcome collection

For safe local checks and a step-by-step provider workflow, see the
[production ingestion testing guide](./production-ingestion-testing.md).

### Configure the producer credential

Generate a high-entropy token in a trusted terminal; for example, from
PowerShell at the project root:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Set the generated value as `PRODUCTION_INGESTION_TOKEN` in **both** the
backend's secret store and the trusted producer's secret store. For local
development only, replace the non-secret `your_secure_token_here` placeholder
in the root `.env` file, which is ignored by Git; restart the API after
changing it. The placeholder in `.env.example` is not a valid token and must
not be used for requests. In production, use the deployment platform's secret
manager/environment injection and restart or redeploy the API when rotating
the token. Never commit the generated value, put it in `frontend/.env*`,
browser code, URLs, request bodies, logs, or support tickets. Restrict secret
access, use HTTPS between producer and API, and rotate the value in both
systems together.

The backend checks that the configured value is at least 32 characters and
compares the supplied header with a constant-time comparison. The producer
sends it in `X-Production-Ingestion-Token`. Missing server configuration
returns `503`; a missing or incorrect token returns `401`. `GET /health`
reports only whether ingestion is configured; it never returns the token.
This credential identifies the configured trusted producer, but does not
independently verify the truthfulness of its input.

### Connect a trusted producer

**Provider integration status: PENDING.** No hotel/PMS provider has been
selected or authorized, and no official provider API documentation is
available. The project currently has **no hotel API connection or production
records**; do not configure or connect a provider-specific production
connector until an authorized source and its official event/API contract are
available.

The generic producer workflow and its local integration tests support
development of the observation/outcome contract. Those tests use explicitly
synthetic fixtures; they are test-only, do not represent a webhook from a
hotel, and must never be treated as genuine production observations or used
to satisfy Model 2 training eligibility. No provider-specific webhook adapter
is implemented.

The Kaggle Hotel Booking Demand dataset is an offline reference/training
dataset, not a source for genuine production observations. The producer
integration described here is a future contract, not evidence that an external
hotel is connected.

A hotel or its authorized integration provider must supply a server-side API,
webhook, or export with least-privilege access to booking fields and final
cancellation status. The provider should issue credentials to its integration
adapter; do not send hotel API credentials to this backend. Separately, an
operator must configure `PRODUCTION_INGESTION_TOKEN` in the backend's secret
manager and give the same value only to the authorized adapter. This API
currently supports one shared bearer token, not per-hotel OAuth identities or
role-based producer credentials. Deploy behind HTTPS, configure the exact
frontend CORS origins, and use managed secret storage before connecting a
real producer.

The adapter must map each booking to the 28 Model 1 fields in the exact order
listed in `ml/models/feature_names.json`. Send `booking_id` first, followed by
those features; do not send `is_canceled`, reservation outcome, or extra
columns. Translate upstream enums and missing values to the Model 1 schema
without inventing values. The dataset's `company`, `reservation_status`, and
`reservation_status_date` fields are not Model 1 prediction features.

1. Assign an opaque stable booking reference in the adapter and submit each
   booking once in a batch. Include `data_source_type: "real"` and a unique
   `batch_id` (or let the API generate one). The authenticated producer is
   responsible for sending genuinely sourced records; the token authenticates
   the producer but cannot independently prove the truth of a claim.
2. The API creates opaque row/observation IDs and stores a keyed HMAC of each
   source booking reference beside its prediction, without returning or
   logging the raw booking ID. Persist the returned `observation_id` against
   the source booking in the adapter's protected mapping store. The response
   preserves record order and returns one observation ID per record; use IDs,
   not order, for subsequent outcome calls.
3. Only after the hotel system reports an authoritative cancellation or
   completed non-cancellation outcome, send the matching observation ID and
   `is_canceled` as integer `0` or `1` to
   `POST /api/v1/monitoring/outcomes`. Do not infer an outcome for a booking
   whose status is still pending. A booking may remain pending until its
   cancellation window closes or the hotel can authoritatively determine the
   final status. Each accepted outcome is matched to its saved prediction by
   observation ID; row order is not used.
4. Retry/network behavior must account for duplicate submissions: a recorded
   outcome returns `409` if submitted again. Unknown observation IDs return
   `404`; invalid labels return `422`. Do not treat a timeout as proof that an
   outcome was not recorded—reconcile it before retrying.
5. Continue collecting across monitoring batches. A partial batch evaluation
   remains visible for monitoring but cannot label Model 2. Model 2 uses a
   future F1 target only after all predictions in the relevant evaluation
   batch have outcomes; reaching 20 booking outcomes in one or a few batches
   does not meet the minimum of 20 distinct labeled production batches.

The token-authenticated endpoints are server-to-server; do not call them from
the dashboard/browser or expose the credential to users.

`POST /api/v1/monitoring/observations` requires `data_source_type: "real"` and
a non-empty batch of records. Each record must have a non-empty `booking_id`
followed by exactly the ordered Model 1 feature schema. The caller may supply
a unique batch ID; otherwise one is generated. Client-supplied observation
IDs and target columns are rejected. The API assigns opaque unique row and
observation IDs, stores a keyed HMAC of the source booking ID, records Model 1
predictions, and saves an explicitly production-source monitoring report.
Known synthetic batch naming patterns are rejected, but the token-authenticated
producer must not submit historical/offline, simulated, or demo data.

```http
POST /api/v1/monitoring/observations
X-Production-Ingestion-Token: <configured producer token>
Content-Type: application/json
```

```json
{
  "batch_id": "hotel-batch-<unique-id>",
  "data_source_type": "real",
  "records": [
    {
      "booking_id": "<opaque-hotel-booking-reference>",
      "hotel": "City Hotel",
      "lead_time": 12,
      "arrival_date_year": 2026,
      "arrival_date_month": "November",
      "arrival_date_week_number": 45,
      "arrival_date_day_of_month": 3,
      "stays_in_weekend_nights": 1,
      "stays_in_week_nights": 2,
      "adults": 2,
      "children": 0,
      "babies": 0,
      "meal": "BB",
      "country": "PRT",
      "market_segment": "Online TA",
      "distribution_channel": "TA/TO",
      "is_repeated_guest": 0,
      "previous_cancellations": 0,
      "previous_bookings_not_canceled": 0,
      "reserved_room_type": "A",
      "assigned_room_type": "A",
      "booking_changes": 0,
      "deposit_type": "No Deposit",
      "agent": 0,
      "days_in_waiting_list": 0,
      "customer_type": "Transient",
      "adr": 100,
      "required_car_parking_spaces": 0,
      "total_of_special_requests": 1
    }
  ]
}
```

The response includes each opaque `observation_id`, immutable prediction,
batch ID, model version, monitoring result, and pending-outcome count. Retain
the observation ID mapped to the source booking reference in the producer's
protected store. The producer submits it with
`is_canceled` equal to integer `0` or `1`:

```http
POST /api/v1/monitoring/outcomes
X-Production-Ingestion-Token: <configured producer token>
Content-Type: application/json
```

```json
{"observation_id": "returned-observation-id", "is_canceled": 1}
```

Only IDs from trusted production predictions are accepted. Unknown IDs return
`404`, invalid labels return `422`, and a duplicate outcome returns `409`.
The persisted outcome keeps its `observation_id`, trusted API provenance, and
receipt timestamp. A production evaluation is eligible for Model 2 labels only
when all contributing outcomes are verified and their IDs match the
production prediction records; session-authenticated CSV outcomes and
operator-declared production sources are excluded.
Predictions are not changed when an outcome is recorded. Metrics are
recalculated from labeled rows only and saved as timestamped, batch-specific
performance snapshots; ROC-AUC is `null` unless both actual classes are
present. Partial snapshots are retained for monitoring, but they do not label
Model 2 observations. A snapshot is complete for Model 2 only when every
prediction in that batch has an actual outcome:
`number_of_actual_outcomes_received` and `number_of_evaluated_records` equal
`number_of_predictions`, and `number_of_pending_outcomes` is zero. After the
outcome and performance snapshot are persisted, the backend checks Model 2's
production-only training eligibility. It trains when all minimum-sample,
target-class, complete-F1, and chronological-split requirements pass, and
otherwise returns the exact eligibility reason without creating a production
model. Artifact files are staged before publication; training failures retain
the previous valid model and record a failed/not-trained attempt.
The response's `model2_training` field contains that result. Existing models
are not retrained unless the eligible labeled production observation count
increases.

Authenticated dashboard readers can query
`GET /api/v1/monitoring/observations?period_days=3650&limit=200` and
`GET /api/v1/monitoring/performance?period_days=3650&limit=200`. These endpoints
return production-source observations and performance snapshots only.
`GET /api/v1/model2/training-status` reports `NOT_TRAINED`, `READY`, or
`TRAINED` based on the existing Model 2 training artifact and eligible labeled
production monitoring history. Its response includes distinct production batch
count, verified labeled and unlabeled production batch counts, required
minimum, target class counts, collection progress, eligibility blockers,
evaluation metrics and evaluation size when trained, model version, training
timestamp, and the most recent training attempt. Collection progress describes
labeled batch count toward the minimum; it is not a live percentage through
estimator fitting. Synthetic/development rows are excluded from all production
counts. There is no separate training HTTP endpoint: after eligible genuine
outcomes arrive, the outcome workflow runs gated training automatically; the
same guarded attempt is available through `python -m ml.model2_training`.

`GET /api/v1/monitoring/ingestion-status` requires an authenticated dashboard
session and returns the safe configured/not-configured flag and the last 50
production observation/outcome request results for the current API process.
The bounded event history records only endpoint, time, HTTP status, and a
coarse result category; it never stores request bodies, observation values, or
the ingestion token. This in-memory history is cleared when the API restarts.

## Model 2 performance-degradation predictions

`POST /api/v1/predictions/performance-degradation` predicts Model 1 performance
degradation from either a stored `batch_id` or a `monitoring_features` object.
Stored-batch predictions require a batch present in the trusted production
observation registry; operator-declared reports are not accepted as
production prediction inputs.
See [Model 2 documentation](./model2.md) for the feature and response contract.
When no trained artifact is available, or its artifact files are invalid, the
prediction endpoint returns `503 Service Unavailable` with an explanatory
detail. Invalid feature input remains `422`, and an unknown batch remains
`404`. Inference also checks the saved artifact's genuine-observation evidence,
binary classes, canonical feature order, and fitted preprocessing pipeline;
an invalid artifact returns `503` rather than a risk score. Successfully
persisted production predictions appear in the dashboard's Model 2 prediction
history; ad-hoc operator feature predictions are not added to that history.

## Historical offline Model 2

`GET /api/v1/model2/offline/status` returns the separately trained historical
Model 2 status, data provenance, observation history, holdout metrics, confusion
matrix, chronological holdout predictions, and the identified Model 1 replay
version. Its source is labeled `HISTORICAL_OFFLINE`; these records do not enter
production eligibility. The dashboard shows its status and results separately
from production Model 2, which remains `NOT_TRAINED` until genuine production
eligibility passes.

`POST /api/v1/model2/offline/predict` accepts a
`{"monitoring_features": { ... }}` object using the Model 2 feature names.
It returns a historical offline estimate only. It is separate from
`POST /api/v1/predictions/performance-degradation`, never writes production
prediction history, and returns `503` when the offline artifact is unavailable.
The response includes the offline model and Model 1 replay versions, an
UTC prediction timestamp, risk class, and probability. The dashboard's
historical prediction table is the saved chronological holdout history;
ad-hoc endpoint predictions are not persisted. Invalid features return `422`.
The offline pipeline can be regenerated with
`python -m ml.offline_model2`; it writes versioned replay data and model
artifacts under `ml/models/offline_model2/` without changing the source dataset
or production artifacts.

## Dashboard

`GET /api/v1/dashboard?period_days=30` returns the aggregated dashboard data.
`period_days` must be between 1 and 3650.

The response includes the Model 1 held-out ROC-AUC and sampled ROC curve points,
plus its five highest Random Forest feature importances. These are computed
from the saved model's original held-out split and checked against the recorded
confusion matrix. Feature importance is descriptive, not causal. Historical
data performance is grouped by arrival month within that same random held-out
test split. This is a retrospective view of the original dataset, not a
time-based validation or live monitoring trend.

Operational monitoring panels include explicitly real reports; unknown-source
reports are shown separately. Simulated reports remain stored for audit and
provenance but are hidden from dashboard, drift, alert, and report views.
production observation totals and prediction history are restricted to
records created by the token-authenticated production ingestion flow.
Unknown-source results never contribute to genuine operational performance.
Model 2 training eligibility is further restricted to labeled, production
observation history with the future-performance target available. Manual
operator-declared real reports and original dataset labels are not Model 2
production training data. A distinct historical offline model may use
chronological Kaggle evaluation evidence, but those labels and predictions
never enter production Model 2 eligibility. The dashboard presents production,
historical offline, and synthetic development Model 2 status, metrics, and
predictions as separate sections. Ad-hoc production feature requests are
marked unverified and are not represented as production prediction history.

## Drift explanations and alert history

Each saved monitoring report includes severity-ranked feature explanations.
Numerical distributions use the same reference-quantile bins used by PSI;
categorical reports retain reference/current proportions and new-category
evidence. PSI warning/critical thresholds and KS significance/effect thresholds
come from `ml/monitoring_config.py`. Root-cause entries are evidence or possible
explanations, not confirmed causes. Recommendations do not claim model
performance and do not recommend retraining from drift alone.

Monitoring saves alerts to `data/monitoring/alerts/alerts.jsonl`. Alert IDs are
deterministic for a batch, provenance source, and condition, preventing repeat
monitoring writes from duplicating the same condition. Each entry contains its
timestamp, severity (`HEALTHY`, `WARNING`, or `CRITICAL`), affected feature,
reason, evidence, data source, and recommended action.
On the first alert-history request, existing monitoring reports are converted
into alert records from their stored evidence. Seeded batches remain simulated;
legacy reports without declared provenance remain unknown unless their batch
ID matches the established simulator naming pattern.

`GET /api/v1/alerts` returns eligible persistent history and supports optional
`severity`, exact `batch_id`, `source` (`REAL`, `SIMULATED`, or `UNKNOWN`),
`period_days` (1–3650), and `limit` (1–1000) filters. Simulated alerts are
always excluded from monitor views, including requests filtered to `SIMULATED`.
The dashboard's Alerts page displays real and unknown-source history. Data
Drift can select the latest report for real or unknown provenance. A missing
provenance value is never promoted to real.

After updating or retraining Model 1, regenerate its dashboard evaluation data
from the project root:

```powershell
python -m ml.model1_evaluation_visualizations
```

## Status and error behavior summary

- Login succeeds with `200`; invalid credentials return `401`, an unconfigured
  local login returns `503`, and a disallowed browser origin returns `403`.
- `GET /health` returns `200` with Model 1, schema, Model 2, and ingestion
  readiness fields. It does not expose a credential or token.
- Dashboard, history, and status GET routes return `200` for valid,
  authenticated requests. Invalid query filters/ranges return `422`.
- Model 1 prediction returns `200` for accepted records; invalid schema or
  quality returns `422`, a repeated batch ID returns `409`, and unavailable
  model/reference artifacts return `503`.
- CSV prediction and outcome uploads are limited to 10 MiB. Oversized bodies
  return `413`; malformed CSV/schema or rejected quality returns `422`.
- Monitoring report download returns `404` when no report matches and `422`
  for an invalid batch ID.
- Production observation/outcome routes return `503` if the server token is
  not configured, `401` for a missing/invalid token, and `422` for invalid
  payloads. A duplicate observation batch or duplicate outcome returns `409`;
  unknown production observation IDs return `404`.
- Model 2 prediction returns `503` while no usable trained production model
  exists. This is the expected current behavior, not a fabricated fallback
  prediction.

The route tests in `backend/tests/` exercise request validation and
authentication. A successful local smoke check does not establish production
availability, source provenance, deployment security, or operational
performance.
