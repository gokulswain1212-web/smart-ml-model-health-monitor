# Smart ML Model Health Monitor

An academic/portfolio prototype for predicting hotel-booking cancellations and
monitoring incoming data for quality problems and distribution drift. The
project combines a saved Model 1 pipeline, a FastAPI service, and a React
dashboard. Production Model 2 remains untrained until enough genuine,
labeled production observations exist; a separate historical offline Model 2
is trained from chronological Kaggle replay data and is not production evidence.

> **Current local status:** Model 1 and the dashboard are available in the
> inspected working copy. Monitoring history currently contains 15 simulated
> reports and 1 report with unknown provenance; there are no explicitly real
> monitored reports. Production ingestion is not configured, no hotel/PMS
> provider has been selected or authorized, and Model 2 has 0 of the 20 minimum
> genuine labeled production observations. The provider-specific webhook
> connector is pending official provider documentation and authorization.
> The separate `HISTORICAL_OFFLINE` Model 2 is trained on offline replay data,
> but its small holdout is weak and does not establish operational predictive
> value. Do not interpret local or offline history as live production
> performance.

## 1. Problem statement

Model quality can change when incoming records differ from the data used to
build a model. This project demonstrates how to validate new hotel-booking
records, compare their feature distributions with a reference dataset, surface
alerts, and collect genuine outcomes for operational evaluation.

## 2. Objectives

- Serve reproducible cancellation predictions from the existing Model 1
  pipeline.
- Check uploaded batches for schema, missingness, invalid values, duplicates,
  and unseen categories.
- Detect numerical and categorical distribution changes and preserve
  evidence for review.
- Keep simulated, unknown, operator-declared, and trusted-producer data
  distinguishable.
- Avoid claiming production performance or training Model 2 without genuine
  outcome labels.

## 3. Key features

- Local single-user sign-in with a signed, HttpOnly session cookie.
- Model 1 JSON and CSV batch prediction workflows.
- Quality and drift reports, evidence-based alerts, and downloadable report
  history.
- PSI-based numerical/categorical drift and a KS test for numerical features.
- Real-outcome evaluation for explicitly real monitored batches.
- Token-gated production observation/outcome routes and a production-only,
  eligibility-gated Model 2 training path.
- Eleven dashboard sections: Overview, Model 1, Model 2, Data Quality,
  Data Drift, Model Performance, Production Data, Alerts, Reports, Batch
  Explorer, and System Health & Settings.

## 4. System architecture

```text
Browser (React + TypeScript + Vite)
        | authenticated same-host HTTP requests
        v
FastAPI backend --------------------> saved Model 1 pipeline
        |                              (preprocessing + classifier)
        +--> quality and drift monitoring
        +--> alert/report JSON and JSONL files
        +--> prediction and genuine-outcome history
                                         |
                                         v
                              Model 2 dataset, eligibility,
                              training, and prediction
```

See [docs/architecture.md](./docs/architecture.md) for component boundaries,
data flow, storage, and production limitations.

## 5. Technology stack

- Python 3.10+; Python 3.13 was used for the current local validation.
- FastAPI, Pydantic, Uvicorn, pandas, NumPy, SciPy, scikit-learn, and joblib.
- React 19, TypeScript, Vite 8, Recharts, and lucide-react.
- pytest and FastAPI `TestClient` for Python tests; TypeScript build and ESLint
  for the frontend.

Backend dependency ranges are listed in [requirements.txt](./requirements.txt)
and [backend/requirements.txt](./backend/requirements.txt). Test dependencies
are in [requirements-dev.txt](./requirements-dev.txt); exact frontend
dependency resolution is recorded in [frontend/package-lock.json](./frontend/package-lock.json).

## 6. ML workflow

1. The Hotel Booking data is prepared and Model 1's feature and target schema
   are recorded in the metadata files.
2. The saved preprocessing/model pipeline consumes 28 ordered features and
   predicts `is_canceled` (`1` means canceled). The API rejects missing or
   unexpected features and does not accept a target column in prediction
   records.
3. For each monitoring batch, the backend validates quality and compares
   distributions to the reference data, then saves a timestamped report and
   alerts.
4. Actual outcomes can be matched to saved prediction row IDs only for batches
   explicitly declared real, or reported later for token-authenticated
   production observations.
5. Model 2 labels a production monitoring observation only when genuine F1
   measurements exist before and after that observation. Training remains
   blocked until the minimum, both-class, chronological-split, and leakage
   checks pass.
6. A separate offline replay refits Model 1 on the chronological historical
   partition, predicts the later Kaggle partition, calculates F1 from actual
   cancellation labels, and trains `HISTORICAL_OFFLINE` Model 2. Its data,
   artifact, status, predictions, and dashboard panel remain isolated from
   production Model 2.

The local working copy inspected for this report contains the source dataset
and Model 1 artifacts. `.gitignore` has rules covering them, but this workspace
has no Git metadata, so whether any are tracked could not be verified. Supply
compatible data and artifacts separately for a clean source checkout. Do not
include them in a submission unless their license and submission rules permit
it.

## 7. Data drift detection: PSI and KS

- **Population Stability Index (PSI)** compares reference and incoming
  proportions. Numerical features use reference-quantile bins; categorical
  features pool rare reference categories before comparing category
  proportions. A small epsilon smooths zero-count bins.
- Project defaults are PSI warning at `0.10` and critical at `0.20`.
- **Kolmogorov–Smirnov (KS)** compares the empirical cumulative distributions
  of valid numerical values. A KS result is treated as meaningful only when
  `p < 0.05` and the statistic is at least `0.10`; statistical significance
  alone is not enough.
- Unusable or too-small distributions are reported as unusable, rather than
  being treated as zero drift.
- Quality-critical batches are CRITICAL. At least three features at or above
  the critical PSI threshold also produce CRITICAL. Lesser drift, meaningful
  KS changes, or quality warnings produce WARNING. Otherwise the overall
  status is HEALTHY.

The numerical cutoffs are project-configured screening defaults, not values
calibrated against observed model failures. PSI uses warning/critical tiers;
the KS rule requires both a significance test and a minimum effect size so
that a small p-value alone does not trigger a finding. Quality-rate defaults
are +1 and +5 percentage points of missingness versus reference for warning
and critical respectively, and +1% and +10% excess duplicate rate for warning
and critical. These defaults need validation against genuine labeled
production history before operational use. None guarantees model failure;
drift evidence alone is not a performance measurement or a reason to retrain.

## 8. Model 1: cancellation prediction

Model 1 is the saved `RandomForestClassifier` pipeline. Its preprocessor
preserves the recorded feature order, fills the configured missing values,
applies explicit mappings for selected fields, and uses the saved categorical
encoder. Tests load both artifacts and compare pipeline predictions with the
separately saved model applied to transformed features.

The recorded held-out evaluation contains 23,878 records and reports 88.93%
accuracy and 0.847 F1 for the canceled class. This is a retrospective
held-out evaluation from the original dataset, not a live operational metric.
The evaluation metadata and artifacts must match the current local files.
Model 1 was not retrained during this finalization.

## 9. Model 2: performance degradation risk

Model 2 is designed to predict whether Model 1's future F1 score declines by
at least `0.05` within 30 days. It uses only quality, drift, prediction, and
performance features available at the monitoring observation time. Future F1
is used only to define the target, and overlapping target windows are purged
from the chronological training split.

The production training path requires at least 20 labeled, genuine production
monitoring batches, a usable chronological split, and both classes in the
training partition. Synthetic, unknown, and offline records are excluded. In
this working copy, production has **0 eligible labels** and remains
`NOT_TRAINED`; there are no production Model 2 accuracy or risk metrics to
report. The separately trained `HISTORICAL_OFFLINE` model has 41 valid labels
(24 stable / 17 degraded) from 45 outcome-week batches. Its nine-row holdout
scored 0.333 accuracy, 0 precision, 0 recall, 0 F1, and 0.400 ROC-AUC; it is
technically trainable but not validated as a useful predictor. The dashboard
shows the offline panel separately from production status. See
[docs/model2.md](./docs/model2.md) and the
[historical offline Model 2 report](./HISTORICAL_OFFLINE_MODEL2_REPORT.md).

To reproduce the offline replay and its separately versioned artifacts, run
from the project root:

```powershell
python -m ml.offline_model2
```

This reads the source partitions and Model 1 artifacts without changing them.
It does not train production Model 2 or make offline labels eligible for
production.

The fixed-pipeline retrospective analysis of the raw hotel booking dataset is
a separate experiment. It uses arrival-date windows, reports the saved
Model 1 artifact's temporal-overlap limitation, and writes only under
`data/offline_replay/runs/`. See the
[offline historical replay guide](./docs/offline-historical-replay.md) and run:

```powershell
python -m ml.offline_historical_replay
```

## Synthetic hotel development simulation

To exercise Model 1, drift monitoring, synthetic outcomes, and chronological
Model 2 development evaluation without a hotel provider, see the
[synthetic hotel simulator guide](./docs/synthetic-hotel-simulator.md). Its
outputs are explicitly synthetic and never count toward production eligibility.

## 10. API documentation

The service provides local authentication, health, dashboard, Model 1
prediction, monitoring, alerts, report history/download, actual outcomes,
production ingestion, and Model 2 routes. Interactive OpenAPI documentation is
available at `http://127.0.0.1:8000/docs` while the backend is running.
Request formats, validation, response behavior, and route inventory are in
[docs/api.md](./docs/api.md).

## 11. Installation

Requirements: Python 3.10+, Node.js 22.12+, npm, local dataset/model artifacts,
and PowerShell on Windows.

From the project root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
npm --prefix frontend ci
Copy-Item .env.example .env
```

Edit `.env`: set `LOCAL_AUTH_USERNAME`, a password of at least six characters,
and a random `LOCAL_AUTH_SECRET` of at least 32 characters. Keep `.env`
private. Leave `PRODUCTION_INGESTION_TOKEN` empty unless a trusted producer is
being configured.

Generate separate random secrets in PowerShell when needed:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

`VITE_API_URL` is optional and is not a secret. The frontend defaults to the
same hostname at port 8000 for local development.

## 12. Run the backend

From the project root in one terminal:

```powershell
python -m uvicorn backend.main:app --env-file .env --reload --host 127.0.0.1 --port 8000
```

## 13. Run the frontend

From the project root in another terminal:

```powershell
npm --prefix frontend run dev
```

Open the Vite URL (normally `http://localhost:5173`) and use the local
credentials from `.env`. Keep the backend bound to loopback for local
development.

## 14. Tests and checks

From the project root:

```powershell
python -m pytest -q
npm --prefix frontend run build
npm --prefix frontend run lint
```

See [docs/testing_report.md](./docs/testing_report.md) for the recorded test
results and scope. The production build emits the chart library as a separate
JavaScript chunk, but it is statically imported and downloaded with the
application; this currently limits individual chunk size, not initial
JavaScript download size.

Submission handoff: [completion report](./PROJECT_COMPLETION_REPORT.md),
[testing report](./TESTING_REPORT.md),
[known limitations](./KNOWN_LIMITATIONS.md), and
[demo guide](./DEMO_GUIDE.md). See the separate
[Model 2 completion report](./MODEL2_COMPLETION_REPORT.md) for production
eligibility, its genuine-data blocker, and implementation status.
See the separate [HISTORICAL_OFFLINE Model 2 report](./HISTORICAL_OFFLINE_MODEL2_REPORT.md)
for the latest offline replay training results and dashboard verification.

## 15. Screenshots

No application screenshots were found in the inspected working copy. For a
submission that requires screenshots, capture the running local dashboard
after sign-in and save them under `docs/screenshots/`. Label screenshots as
local/demo views and do not represent simulated or unknown-provenance data as
real production traffic.

## 16. Current limitations

- No verified real monitoring reports or genuine Model 2 labels are available
  in the inspected working copy.
- The production ingestion token is not configured; no trusted producer was
  connected or exercised against a real source.
- Local authentication is single-user development authentication, not OIDC or
  a production identity/authorization system. The local cookie is not marked
  Secure because the supported local workflow uses loopback HTTP.
- Monitoring state is JSON/JSONL on local disk. It is not transactional for
  concurrent workers and has no automated retention or backup system.
- No production deployment has been performed or verified.
- `.gitignore` has rules for local datasets and binary model artifacts, but
  their tracked status could not be verified because this workspace has no
  Git metadata. Provide required assets separately under applicable licensing
  rules.
- The project build separates the chart library into its own chunk, but the
  chunk is not lazy-loaded; both application and chart chunks are downloaded
  during initial app loading.
- The responsive layout has a 240 CSS pixel minimum width; narrower effective
  viewports are not supported or verified.

## 17. Future improvements

- Connect a trusted, authorized producer and collect genuine outcomes before
  enabling production Model 2 training.
- Add OIDC, role-based authorization, HTTPS, rate limiting, and deployment
  security controls.
- Move runtime records to a transactional database and add retention,
  backup, restore, and observability policies.
- Add authenticated end-to-end browser tests and curated, non-sensitive
  screenshots.
- Continue measuring bundle startup behavior on target hardware and split
  additional pages only when profiling justifies it.

## 18. Submission checklist

- [ ] Include source, `README.md`, `docs/`, requirements files, frontend
  `package.json` and lockfile, and `.env.example`.
- [ ] Do not submit `.env`, passwords, signing secrets, or ingestion tokens.
- [ ] Confirm the required dataset and compatible artifacts are available
  separately if the evaluator needs to run Model 1.
- [ ] Confirm data/artifact licenses and academic rules before redistributing
  them.
- [ ] Include screenshots only if they are genuine captures of the running
  application and their provenance is clear.
- [ ] Distinguish the weakly evaluated `HISTORICAL_OFFLINE` Model 2 from
  production Model 2 (`NOT_TRAINED`, 0/20 genuine labels); production ingestion
  is unconfigured.
- [ ] Include [docs/final_project_report.md](./docs/final_project_report.md),
  [docs/architecture.md](./docs/architecture.md),
  [docs/api.md](./docs/api.md), and
  [docs/testing_report.md](./docs/testing_report.md).
- [ ] Do not claim production deployment or real-data validation.
