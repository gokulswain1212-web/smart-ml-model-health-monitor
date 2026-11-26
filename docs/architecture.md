# Project Architecture

## Overview

The application is a local, single-process ML monitoring prototype with three
main layers:

1. **React + TypeScript frontend** in `frontend/src/` renders eleven sections:
   Overview, Model 1, Model 2, Data Quality, Data Drift, Model Performance,
   Production Data, Alerts, Reports, Batch Explorer, and System Health &
   Settings. Recharts is statically imported and emitted as a separate chart
   bundle by the production build.
2. **FastAPI backend** in `backend/main.py` validates requests, applies local
   session or producer-token authentication as appropriate, invokes model and
   monitoring services, and returns JSON responses.
3. **ML services** in `ml/` load Model 1, assess data quality and drift,
   persist monitoring evidence, evaluate genuine outcomes, and gate Model 2
   training/prediction.

## Data flow

### Model 1 prediction

The browser or a client submits a batch of feature-only records. The API
validates batch IDs and the exact 28-feature schema, applies the saved
scikit-learn pipeline, and records row IDs, classes, probabilities, model
version, and artifact digest. The predictor uses the pipeline's persisted
preprocessor; it does not retrain Model 1.

### Batch monitoring

The monitoring path loads the reference dataset and feature metadata, assesses
feature schema, missingness, invalid numeric values, infinities, negative
values, category membership, duplicate rates, and batch metadata. It computes
numerical PSI plus the two-sample KS statistic, and categorical PSI with rare
reference categories pooled. Configured thresholds classify overall status;
evidence is written to a timestamped report and corresponding alerts.

The shared defaults are PSI `0.10` warning and `0.20` critical; KS requires
`p < 0.05` and statistic `>= 0.10`; at least three critical-PSI features or
critical data quality make the batch CRITICAL. Missingness increases of 1 and
5 percentage points versus reference are warning and critical; excess
duplicate rates of 1% and 10% are warning and critical. These are screening
defaults from `ml/monitoring_config.py`, not thresholds calibrated to observed
model failures. PSI tiers and the KS minimum effect check are intended to
avoid treating statistical significance alone as operational impact. Validate
and tune the defaults against genuine labeled production history before
production use.

### Actual outcomes

For a manual batch, outcomes are accepted only when saved prediction row IDs
match and the latest report explicitly declares `real`. For production
observations, a configured producer token authorizes prediction and outcome
submission. An explicit `real` declaration is provenance metadata, not
independent verification of the data source. The production request pairs an
upstream booking ID with its generated opaque observation ID; only a keyed
HMAC of the source ID is stored with the prediction. The trusted producer must
keep the reversible booking-to-observation mapping in its own protected
storage and submit later outcomes by observation ID.

No external hotel system is connected in the current project. A future
integration uses a least-privilege hotel API credential in a producer-side
adapter, then sends feature-mapped bookings to the FastAPI production routes
over HTTPS using `PRODUCTION_INGESTION_TOKEN`. The hotel's authenticated
cancellation/final-status event is submitted later as the actual outcome.
Hotel credentials and raw booking identifiers are not sent to the dashboard.

### Production Model 2

The historical dataset builder selects the latest report per batch and uses
only production-source reports, predictions, and outcome-backed performance
records for training eligibility. Features are computed as of the observation
time. The target is a future F1 decline of at least 0.05 within 30 days.
Training requires at least 20 labeled production batches, both target classes
in every completed batch and in the purged chronological training partition,
a usable test partition, and valid timestamps. The current working copy has
zero eligible labeled rows; the expected status is `NOT_TRAINED`.

### HISTORICAL_OFFLINE Model 2

The offline workflow is independent of production eligibility and artifacts.
It uses the original chronological Kaggle partitions, refits Model 1's saved
preprocessing/classifier recipe on the earlier partition only, and predicts
the later partition. Actual `is_canceled` labels determine weekly F1. The
offline target uses the same 0.05 decline/30-day definition; rows record one
identified Model 1 replay version, and training purges overlapping target
windows before a chronological holdout. The Model 2 artifact, metadata,
versioned replay dataset, evaluation predictions, and dashboard status are
stored only under `ml/models/offline_model2/`. This experiment is trained but
has poor holdout metrics and is not evidence of production performance.

### Fixed-pipeline historical replay

`python -m ml.offline_historical_replay` is a separate retrospective analysis
of `data/raw/hotel_bookings.csv`. It uses the existing fitted Model 1 pipeline
and the exact `feature_names.json` order, excludes the cancellation target and
reservation-status fields, and writes versioned outputs only under
`data/offline_replay/runs/`. Arrival-week cohorts provide ordering because the
dataset has no booking-created timestamp. The report records duplicate and
conflicting-feature diagnostics and keeps identical feature groups within one
arrival window. Model 2 labels are derived from measured historical F1 using
the existing threshold/horizon, and offline Model 2 is trained only after a
purged chronological split meets sample-count and both-class gates.

This fixed Model 1 artifact was trained with a random split across the source
period, and some feature lifecycle availability is unknown. Consequently the
replay is exploratory retrospective evidence, not an independent out-of-time
or booking-time evaluation. Its `HISTORICAL_OFFLINE` provenance is distinct
from both production data and synthetic development data.

## Components and source files

- `backend/main.py`: authentication, readiness, dashboard, monitoring, report,
  alert, prediction, outcome, and ingestion HTTP routes.
- `backend/dashboard.py`: aggregates Model 1 evaluation and provenance-filtered
  monitoring/operational records for the UI.
- `ml/hotel_preprocessing.py`: Model 1 saved preprocessing behavior.
- `ml/performance_tracking.py`: Model 1 prediction persistence and actual
  outcome evaluation.
- `ml/data_quality.py`: schema and record-level quality checks.
- `ml/drift_detection.py`: PSI, KS, and distribution summaries.
- `ml/monitoring_config.py`: shared quality and drift defaults.
- `ml/monitoring.py`: complete batch report creation and status classification.
- `ml/alert_engine.py`: evidence-based explanations and persistent alerts.
- `ml/model2_dataset.py`: production-only historical feature/target dataset.
- `ml/model2_training.py`: training eligibility, chronological evaluation,
  artifact persistence, and retraining gates.
- `ml/model2_prediction.py`: safe production artifact discovery and inference.
- `ml/offline_model2.py`: isolated Kaggle replay, target generation,
  offline-only training, artifact loading, and prediction.
- `ml/model2_development_*.py`: isolated synthetic development experiment; not
  a production Model 2 or source of genuine training labels.
- `frontend/src/App.tsx`: authenticated dashboard, navigation, pages, upload
  workflows, and chart components.

## Local storage

- `data/monitoring/reports/*.json`: timestamped monitoring reports and PSI/KS
  evidence.
- `data/monitoring/alerts/alerts.jsonl`: deduplicated alert history.
- `data/monitoring/predictions/*.jsonl`: saved predictions and identifiers.
- `data/monitoring/outcomes/*.jsonl`: accepted actual outcomes.
- `data/monitoring/performance/*.json`: outcome-based evaluation snapshots.
- `data/monitoring/model2/*.csv`: prepared historical Model 2 datasets.
- `ml/models/*.pkl`: local Model 1 artifacts; covered by `.gitignore` rules.
- `ml/models/model2/`: Model 2 definitions, status records, and any trained
  production artifacts. Synthetic development outputs are isolated in
  `ml/models/model2/development/`.
- `ml/models/offline_model2/`: versioned historical replay data, offline
  Model 2 artifacts, evaluation results, and offline training status only.
- `data/offline_replay/runs/`: isolated fixed-pipeline historical replay
  outputs; never production monitoring evidence.

The source dataset and local binary artifacts are present in the inspected
working copy, and `.gitignore` has rules covering them. This workspace has no
Git metadata, so their tracked status could not be verified; a clean checkout
or source-only submission will need compatible artifacts supplied separately.
Runtime records are local files rather than a transactional database.
Concurrent multi-worker writes, automated retention, and backups are not
supported.

## Authentication and configuration

- Dashboard/data API routes use a single-user local session configured with
  `LOCAL_AUTH_USERNAME`, `LOCAL_AUTH_PASSWORD`, and `LOCAL_AUTH_SECRET`.
- `GET /health` is public and returns readiness booleans/status only, not
  credentials or secrets.
- Production observation/outcome routes use the separate
  `X-Production-Ingestion-Token` configured through
  `PRODUCTION_INGESTION_TOKEN`; they do not use the dashboard session.
- Browser origins are checked against the exact `CORS_ORIGINS` allowlist.
  CORS is not endpoint authentication.
- Local development uses HTTP on loopback; cookies are HttpOnly and SameSite
  Lax but are not Secure in the current local configuration.

These are development controls, not a production security boundary. OIDC,
explicit multi-user authorization, HTTPS termination, rate limiting, secure
secret management, and deployment-specific hardening have not been implemented
or verified.

## API route inventory

| Method | Route | Access |
|---|---|---|
| `GET` | `/health` | Public readiness |
| `POST` | `/api/v1/auth/login` | Local credentials and allowed browser origin |
| `GET` | `/api/v1/auth/session` | Local session |
| `POST` | `/api/v1/auth/logout` | Allowed browser origin |
| `GET` | `/api/v1/dashboard` | Local session |
| `GET` | `/api/v1/model2/training-status` | Local session |
| `GET` | `/api/v1/monitoring/ingestion-status` | Local session |
| `POST` | `/api/v1/monitoring/observations` | Producer token |
| `POST` | `/api/v1/monitoring/outcomes` | Producer token |
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

Detailed schemas and error responses are documented in [api.md](./api.md).
