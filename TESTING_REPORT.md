# Testing and Verification Report

**Run date:** 20 November 2026  
**Environment:** Windows, Python 3.13, Node.js/npm in the local workspace.  
**Scope:** Local automated tests, production frontend build, lint, and
authenticated API/browser smoke checks. No production
deployment or trusted external producer was tested.

## Automated commands and results

| Command | Result |
|---|---|
| `python -m pytest -q` | **229 passed, 0 failed, 5 warnings in 162.22s** |
| `npm --prefix frontend run test` | **8 passed** |
| `npm --prefix frontend run build` | **Passed**; `tsc -b` and Vite production build succeeded |
| `npm --prefix frontend run lint` | **Passed**; ESLint reported no issues |

### Python warnings

- One Starlette deprecation warning: use of `httpx` with
  `starlette.testclient` is deprecated and suggests `httpx2`.
- Four SciPy/NumPy warnings: exact KS calculation could not complete and fell
  back to the asymptotic method. Affected tests passed.
- No tests were skipped or removed to obtain these results.

### Frontend production build

The latest build emitted:

| Asset | Size | Gzip |
|---|---:|---:|
| Main JS | 367.32 kB | 102.73 kB |
| Chart JS | 415.59 kB | 117.62 kB |
| Rolldown runtime | 0.72 kB | 0.42 kB |
| CSS | 82.41 kB | 18.66 kB |

No 500 kB chunk warning was emitted. Recharts is statically imported and
grouped into a separate chunk; both application and chart JavaScript are
downloaded during initial app loading. This limits individual chunk size but
does not defer the chart download or reduce the total JavaScript payload.

ESLint reported no issues. The frontend test suite passed all 8 tests.

## Backend/API checks

The existing automated suite covers authentication, validation, readiness,
monitoring, alerts/reports, Model 1 artifacts and prediction, ingestion
validation, Model 2 eligibility/prediction, and provenance/leakage controls.
The recorded local API smoke checks returned:

| Check | Result |
|---|---|
| `GET /health` | `200`; application `OK`, Model 1 `READY`, schema `READY`, Model 2 `NOT_TRAINED` |
| Local login/session | Login `200`; authenticated dashboard request succeeded |
| `GET /api/v1/dashboard` | `200` |
| `GET /api/v1/alerts` | `200` |
| `GET /api/v1/monitoring/reports` | `200`; saved reports returned |
| Model 2 production training status | `200`; `NOT_TRAINED`, 0 genuine eligible labels, minimum 20 |
| Model 2 offline status | `200`; `TRAINED`, 41 eligible historical labels |
| Model 2 offline prediction | `200`; valid probability, model/replay versions, UTC timestamp |
| Production ingestion readiness | Not configured |

The offline prediction smoke returned class `0` with probability `0.235` from
version `model2-offline-random-forest-20261120T104834452589Z` and replay
`historical-refit-87bf685e4dd39ad8`. It is an exploratory historical estimate,
not a production prediction or performance claim. The Python suite also
covers Model 1 prediction/artifact validation, monitoring, and drift behavior.

Production ingestion was not tested with an external trusted producer because
no token or producer integration is configured.

## Frontend browser checks

The backend and Vite development server were run locally and the authenticated
dashboard loaded. All eleven dashboard sections rendered; no console errors or
warnings, page exceptions, or failed non-cancelled requests were observed
during stable navigation. The Model 2 page showed the offline panel first,
with `TRAINED`, 41 labels, evaluation results, history, and the Model 1 replay
version, followed by production `NOT_TRAINED` with 0/20 genuine labels.
Authenticated live API requests returned those same distinct statuses and a
valid offline prediction.

Mobile navigation succeeded at a configured 390x844 viewport. At an
effective 242 CSS pixel width, the first check found 7 pixels of top-bar
overflow; the responsive select width was reduced, and a repeat check showed
no horizontal overflow. A full browser/device/accessibility matrix and
exhaustive form testing were not performed.

## Coverage limits

- Local history inspected: 15 simulated reports, 1 unknown-provenance report,
  0 explicitly verified real reports.
- Production Model 2: 0/20 genuine labeled production observations; no
  production training or performance metrics.
- HISTORICAL_OFFLINE Model 2: 45 batches, 41 valid labels (24 stable / 17
  degraded), 31 training rows, 9 holdout rows. Holdout F1 is 0 and ROC-AUC is
  0.400; predictive utility is not established.
- The source dataset and model artifacts are available in the inspected
  working copy but may be absent from a clean checkout. `.gitignore` rules
  were checked; because the workspace has no Git metadata, tracked status
  could not be verified.
- No production deployment, external producer, or production security setup
  was exercised.
