# Testing and Verification Report

**Validation date:** 20 November 2026  
**Environment:** Windows, Python 3.13, Node.js/npm available in the local
workspace. The results below are local verification, not production
availability or deployment evidence.

## Automated checks

| Command | Result |
|---|---|
| `python -m pytest -q` | **229 passed**, 0 failed, 5 warnings, 162.22 seconds |
| `npm --prefix frontend run test` | **8 passed** |
| `npm --prefix frontend run build` | **Passed**; TypeScript build and Vite production build succeeded |
| `npm --prefix frontend run lint` | **Passed**; ESLint reported no issues |

### Python warnings

- One Starlette deprecation warning notes that `httpx` with
  `starlette.testclient` is deprecated and suggests `httpx2`.
- Four SciPy/NumPy runtime warnings in offline multi-window drift tests state
  that an exact KS calculation could not complete and the test switched to
  the asymptotic method. The affected tests passed. No dependency was changed
  to suppress these warnings.

### Frontend build output

The successful production build emitted:

- Main entry: **367.32 kB** (102.73 kB gzip).
- Chart chunk: **415.59 kB** (117.62 kB gzip).
- Rolldown runtime: **0.72 kB** (0.42 kB gzip).
- CSS: **82.41 kB** (18.66 kB gzip).

There was no 500 kB chunk warning after adding Rolldown code splitting. The
chart library remains statically imported, so both JavaScript chunks are
downloaded during initial app loading. Splitting reduces the largest individual
chunk but does not reduce the total JavaScript download or defer chart code.
The frontend test suite passed all 8 tests; ESLint reported no issues.

## Backend/API smoke checks

The local API was started with temporary test-only credentials; no credential
or token value is recorded here.

| Check | Result |
|---|---|
| `GET /health` | `200`; application `OK`, Model 1 `READY`, schema `READY`, Model 2 `NOT_TRAINED` |
| Local login and authenticated session | Login `200`; dashboard request succeeded |
| `GET /api/v1/dashboard` | `200`; reported application `OK` and Model 2 `NOT_TRAINED` |
| `GET /api/v1/alerts` | `200`; persisted history returned |
| `GET /api/v1/monitoring/reports` | `200`; 16 reports in the inspected period |
| Actual Model 1 prediction route | `200`; one prediction with a finite probability in `[0,1]` |
| Malformed Model 1 feature schema | `422` |
| Production Model 2 status | `NOT_TRAINED`; 0/20 genuine eligible labels |
| HISTORICAL_OFFLINE Model 2 status | `TRAINED`; 41 valid labels |
| Offline Model 2 prediction | `200`; valid probability, model/replay version, UTC timestamp |
| Production ingestion readiness | Not configured; no token value exposed |

The Model 1 smoke input was a complete feature row from the local prepared
dataset. It returned class `0`, probability approximately `0.0617`, and
quality status `WARNING` for the one-row monitoring context. The temporary
smoke-test prediction record was removed; it is not part of project history.
This result verifies API execution only and is not an accuracy or live-traffic
claim.

The full backend test suite also covers authentication, request validation,
model artifact readiness, production-token failures, ingestion validation,
report/alert behavior, Model 2 eligibility and prediction behavior, and
provenance/leakage controls. Production endpoints have not been tested with a
real configured external producer.

## Frontend browser checks

The backend and Vite development server were opened in the integrated browser.
All eleven dashboard sections rendered; no page exceptions, console errors or
warnings, or failed non-cancelled requests were observed during stable
navigation. The Model 2 page showed `HISTORICAL_OFFLINE` first with 41 labels,
holdout results, history, and the Model 1 replay version, followed by the
separate production `NOT_TRAINED` panel with 0/20 genuine labels.
Authenticated live API requests returned the same statuses and a successful
offline prediction.

Mobile navigation succeeded at a configured 390x844 viewport. At an effective
242 CSS pixel width, an initial check found 7 pixels of top-bar overflow; after
reducing the responsive date-range control width, a repeat check showed no
horizontal overflow. A full browser/device/accessibility matrix remains
unverified.

Recharts is statically imported and grouped into a separate production chunk;
it is not deferred or lazy-loaded. A prior lazy-loading experiment caused a
React maximum-update-depth runtime error and was removed before final browser
verification. This validation is a local smoke check, not exhaustive
interaction coverage for every upload form, every viewport/browser,
screen-reader use, or a production deployment. Responsive CSS exists, but a
full device/browser matrix has not been run.

## Coverage boundaries

- The local dataset and model artifacts support Model 1 execution, but are
  covered by `.gitignore` rules and may not be present in a clean source
  checkout. The workspace has no Git metadata, so tracked status could not be
  confirmed.
- Existing history contains 15 simulated reports and 1 unknown-provenance
  report; there are no explicitly real reports in the inspected 30-day
  period.
- No production ingestion token or connected trusted producer is configured.
- Production Model 2 has 0/20 genuine eligible labels and no production
  metrics. The offline model has 41 labels but poor holdout results (F1 0,
  ROC-AUC 0.400); predictive utility is not established.
- The project has no production deployment to test.
