# Final Project Audit Report

**Project:** Smart ML Model Health Monitor  
**Audit date:** 20 November 2026  
**Audit scope:** Existing source, configuration templates, local artifacts and
monitoring records, backend/API tests, frontend build/lint, and local smoke
checks. The separate historical offline Model 2 was regenerated; Model 1's
production artifact and the source dataset were not changed.

## Executive summary

The project is a substantially implemented local academic/portfolio
prototype. Model 1, backend APIs, monitoring services, Model 2 training gates,
and the 11-section frontend are present. Automated checks and local browser/API
smoke checks support that the application works in the inspected working
copy.

This is **not a production deployment**. Production Model 2 correctly remains
`NOT_TRAINED` because there are 0 of the required 20 eligible genuine labels;
the ingestion token is unset and no trusted producer is connected. A separate
`HISTORICAL_OFFLINE` Model 2 is trained on Kaggle replay data, but its holdout
metrics are poor and do not establish predictive utility.

## Audit findings

1. **Documentation consistency:** the old architecture document described
   eight dashboard sections; the UI has eleven. The README also described
   datasets and artifacts as absent without distinguishing a clean checkout
   from this working copy. These claims have been clarified.
2. **Submission docs:** a consolidated final report and testing report were
   missing; they are now included.
3. **Frontend payload:** the previous production build produced one
   approximately 760 kB JavaScript chunk. The production build now emits
   separate application and chart chunks. Recharts remains a static import, so
   this avoids one oversized chunk but does not defer the chart download or
   reduce the total initial JavaScript payload.
4. **Generated artifacts:** the ignore rules did not cover nested synthetic
   Model 2 development datasets and `.joblib` output. Specific ignore patterns
   were added; existing local files were preserved.
5. **Production evidence:** reports in the inspected 30-day history total 16:
   15 simulated and 1 unknown-source, with 0 explicitly real. Production
   ingestion is unconfigured and there are no production Model 2 labels.
6. **Repository boundary:** this working directory has no Git metadata.
   `.gitignore` rules were inspected, but tracked/ignored status could not be
   confirmed with Git.
7. **Security/deployment boundary:** local single-user authentication and
   producer-token checks are implemented, but OIDC, production authorization,
   HTTPS termination, database-backed transactional storage, backups, and
   production deployment are not implemented or verified.
8. **Implementation bugs:** the audit and smoke tests did not identify a
   confirmed user-facing functional defect in the tested local workflows.

## Feature status

Statuses use the requested vocabulary. `VERIFIED` means supported by
automated tests or a completed local smoke check, not by a production
deployment.

| Feature | Status | Evidence / boundary |
|---|---|---|
| Local sign-in, session cookie, logout, and origin checks | **VERIFIED** | Authentication route tests and local browser sign-in |
| Public health/readiness response | **VERIFIED** | Health test and local API response; Model 1 `READY`, application `OK` |
| Dashboard aggregation and section navigation | **VERIFIED** | Dashboard API tests; all 11 section headings opened in a local browser |
| Model 1 artifact loading and 28-feature order | **VERIFIED** | Artifact/preprocessor tests and actual authenticated API prediction |
| Model 1 input schema, output, and validation | **VERIFIED** | API tests; valid prediction returned class/probability, invalid feature schema rejected |
| Data quality checks and HEALTHY/WARNING/CRITICAL classification | **VERIFIED** | Monitoring tests cover schema, missing/invalid values, duplicates, metadata, and classification |
| Numerical PSI, KS, categorical PSI, and unusable samples | **VERIFIED** | Drift tests cover identical/shifted distributions, KS, unseen categories, and insufficient data |
| Alert persistence, provenance filtering, report listing/download | **VERIFIED** | Backend tests and local authenticated alert/report API reads |
| Manual CSV monitoring and outcome import | **VERIFIED** | Route and service tests; only explicitly real reports qualify for manual outcome evaluation |
| Production observation/outcome endpoints | **IMPLEMENTED BUT NOT FULLY VERIFIED** | Auth, validation, and processing paths have tests; no configured producer or genuine source was available |
| Model 2 target definition, as-of features, provenance/leakage gates | **VERIFIED** | Dataset/training tests use isolated fixtures; synthetic/unknown rows are excluded |
| HISTORICAL_OFFLINE Model 2 generation, training, artifact loading, prediction | **IMPLEMENTATION COMPLETE** | Separate versioned replay; weak holdout results are disclosed in the offline report |
| Production Model 2 fit and production inference | **BLOCKED BY MISSING REAL DATA OR CONFIGURATION** | No eligible genuine labeled production observations; status remains `NOT_TRAINED` (0/20) |
| Frontend production build, TypeScript, and lint | **VERIFIED** | Build and lint commands recorded in [testing_report.md](./testing_report.md) |
| Frontend responsive layout across devices | **IMPLEMENTED BUT NOT FULLY VERIFIED** | Desktop section navigation and mobile navigation at 390x844 were smoke-tested; a full device/browser matrix was not run |
| Application screenshots for submission | **NOT IMPLEMENTED** | No application screenshot files were found in the inspected working copy |
| OIDC, production HTTPS, multi-user authorization, database persistence, backups | **NOT IMPLEMENTED** | Deployment/security work remains future work |
| Live production monitoring/performance claims | **BLOCKED BY MISSING REAL DATA OR CONFIGURATION** | No verified real-source reports or live outcome measurements |

## Model 1 verification

The saved pipeline and its separate model/preprocessor artifacts load from the
local working copy. The recorded feature list contains 28 names. Artifact
tests compare pipeline predictions against the separately saved preprocessor
and estimator, including a reordered input frame; the output is restored to
the recorded feature order. An authenticated API smoke prediction on a
complete local dataset row returned HTTP `200`, one prediction, class `0`,
and probability `0.0617` (rounded). A malformed feature schema returned
`422`. The small smoke batch was flagged `WARNING` by quality monitoring; it
was not used as a quality or live-performance claim.

The recorded held-out evaluation has 23,878 records, 88.93% accuracy, and
0.8466 F1 for the canceled class. These are original-dataset held-out results,
not operational metrics. Model 1 was not retrained or overwritten.

## Monitoring verification

Quality assessment checks required/extra features, feature order, missing
values relative to reference rates, numeric conversion, infinities, configured
non-negative constraints, categorical values, duplicate rows, and required or
inconsistent batch metadata. Drift calculation uses finite numeric samples,
reference-quantile bins for numerical PSI, categorical PSI with rare
reference-category pooling, and a two-sample KS test for numerical features.
Reports retain sample counts, distributions, p-values, effect thresholds, and
provenance. Tests exercise representative healthy, warning, critical, and
unusable-input cases.

Monitoring reports are evidence about input data. They do not prove a model
has degraded; genuine outcome labels are needed for operational performance
evaluation.

## Model 2 verification and current statuses

The target is a decline of at least 0.05 in F1 within 30 days, relative to a
prior genuine F1 value. Training rows must be production-source observations
with real provenance and a future label. Future F1 is not an input feature;
the training split is chronological and purges overlapping target windows.
Unit fixtures test both eligible and ineligible pathways, but are not
production observations.

**Production:** `NOT_TRAINED`, **0 / 20** genuine labeled production
observations. The dashboard exposes the count, minimum, eligibility, and
reason. No production accuracy or risk score is reported.

**HISTORICAL_OFFLINE:** `TRAINED` as a separate experiment using 45 outcome
batches and 41 valid labels (24 stable / 17 degraded). Its chronological
split uses 31 training and 9 holdout rows. Holdout accuracy is 0.333, precision,
recall, and F1 are all 0, and ROC-AUC is 0.400. The dashboard displays its
status and historical holdout predictions separately from production. See
[the detailed offline report](../HISTORICAL_OFFLINE_MODEL2_REPORT.md).

## Security and configuration

- `.env.example` contains empty password, signing-secret, and ingestion-token
  values; `.env` is ignored. Do not submit a populated `.env`.
- Health and dashboard responses expose readiness/configuration booleans, not
  token values.
- User-facing data routes require a local signed session; production ingest
  uses a separate token.
- CORS is allowlisted but is not a substitute for authentication.
- Local cookies are HttpOnly and SameSite Lax. The local HTTP configuration
  does not set `Secure`.
- No OIDC, production identity service, TLS termination, rate limiting,
  production secret manager, or deployment hardening has been verified.

## Submission recommendation

Submit source, updated README, `docs/`, requirements files, frontend source,
`package.json`, and `package-lock.json`. Do not submit credentials, local
monitoring records, `node_modules`, or generated build output. Provide the
dataset and compatible model artifacts separately only when permitted by
their license and the evaluator's requirements. State plainly that Model 2 is
trained only for the historical offline experiment, with weak holdout results;
production Model 2 is `NOT_TRAINED` and production ingestion is not configured.

See [testing_report.md](./testing_report.md) for exact final command outputs,
test counts, and smoke-test scope.
