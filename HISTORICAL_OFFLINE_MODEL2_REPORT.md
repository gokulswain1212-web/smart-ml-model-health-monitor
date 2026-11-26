# HISTORICAL_OFFLINE Model 2 Completion Report

**Run date:** 20 November 2026  
**Scope:** Historical data replay, offline Model 2 training, artifact loading,
prediction, API, and dashboard integration.

## Status

- **Offline implementation:** Complete and verified.
- **Offline artifact:** `TRAINED`; it is a separate historical experiment and
  has weak holdout results. Training status does not imply useful predictive
  performance.
- **Production Model 2:** `NOT_TRAINED`; 0 genuine eligible production labels
  against the unchanged minimum of 20. No offline observation, label, or model
  counts toward production eligibility.

## Architecture and data preparation

Model 1 predicts hotel booking cancellation. The offline replay uses its
recorded preprocessing and classifier configuration, refit on the historical
partition only, then predicts the strictly later evaluation partition. The
saved Model 1 pipeline artifact was not applied to the future partition
because it had been fit using a random split containing later-period data.
Production Model 1 artifacts were not retrained or replaced.

The replay aligns Model 1 predictions to the actual historical `is_canceled`
labels, groups evaluation outcomes by `reservation_status_date`, and records
weekly aggregate monitoring observations. The existing target remains
unchanged: compare the latest eligible earlier F1 with the first eligible
later F1 within 30 days; label degradation when the absolute decline is at
least 0.05. Future F1 and all target evidence are excluded from Model 2 input
features. The chronological split purges training rows whose target window
would overlap the test period.

| Data result | Value |
|---|---:|
| Historical reference/training partition | 83,580 bookings, through 2016-11-06 |
| Later labeled evaluation partition | 35,810 bookings |
| Outcome-week monitoring observations | 45 |
| Observations with both actual cancellation classes | 43 |
| Valid offline target labels | 41 |
| Unlabeled observations | 4 |
| Stable / degraded labels | 24 / 17 |
| Dataset provenance | All 45 rows are `HISTORICAL_OFFLINE` |
| Model 2 features | 17, with no future-target fields |
| Model 1 replay version | `historical-refit-87bf685e4dd39ad8` (all 45 batches) |

On the later evaluation partition, the historical Model 1 replay achieved
approximately 0.790 accuracy, 0.709 precision, 0.673 recall, and 0.691 F1.
These scores describe the offline replay only.

Validation confirmed 45 distinct batch IDs and unique monotonically increasing
timestamps. All rows use one replay version and `HISTORICAL_OFFLINE`
provenance. Reconstructed targets matched the configured F1-drop rule, and
every future measurement was later than its observation and within 30 days.
The 17-feature schema excludes target/future fields. Six features contain
sparse missing values, which are imputed from the training partition only;
four feature pairs have absolute correlation at least 0.95. No feature was
removed because 31 training examples are too few to justify selection by
correlation alone.

These are historical Kaggle data, not production observations. The booking
chronology is derived from `estimated_booking_date`; the outcome-week date is
not a verified production receipt timestamp. The original source partitions
and Model 1 artifacts were read-only for the replay. SHA-256 checks were run
before and after; the displayed hash prefixes matched.

## Training and evaluation

The offline trainer required the existing minimum of 20 valid labels and both
target classes. It used a median imputer fitted on training rows and a
200-estimator, class-balanced `RandomForestClassifier` with `random_state=42`.
The split is chronological, with overlapping future label windows purged.

| Result | Value |
|---|---:|
| Model version | `model2-offline-random-forest-20261120T104834452589Z` |
| Training / test rows after eligibility and purge | 31 / 9 |
| Training time range | 2016-11-20 through 2017-06-18 |
| Latest training target time | 2017-06-25 |
| Test time range | 2017-07-02 through 2017-08-27 |
| Test classes (stable / degraded) | 4 / 5 |
| Accuracy | 0.333 |
| Precision / recall / F1 | 0 / 0 / 0 |
| ROC-AUC | 0.400 |
| Confusion matrix (stable/degraded order) | `[[3, 1], [5, 0]]` |

The results show poor holdout performance. The artifact is technically
trainable and loadable, but it should not be treated as a validated operational
early-warning model.

## Exploratory candidate comparison

Five fixed estimator candidates were compared with purged expanding-time
validation using only 18 pre-holdout out-of-fold predictions. Logistic
Regression had F1 0.615, precision 0.800, recall 0.500, ROC-AUC 0.675, and
PR-AUC 0.737. HistGradientBoosting reached recall 1.0 but predicted every
validation case positive. The final nine-row holdout was not used for this
comparison. Because the evidence is small and unstable, no candidate was
promoted and no model improvement is claimed. A broad hyperparameter search
was not performed because it would overfit this dataset.

## Artifacts and prediction verification

Artifacts are stored only under `ml/models/offline_model2/`, separate from
production Model 2 artifacts:

- Dataset: `ml/models/offline_model2/historical_offline_monitoring_dataset_historical-refit-87bf685e4dd39ad8.csv`
- Model: `ml/models/offline_model2/model2-offline-random-forest-20261120T104834452589Z.joblib`
- Metadata: `ml/models/offline_model2/model2-offline-random-forest-20261120T104834452589Z.json`
- Evaluation: `ml/models/offline_model2/evaluation_report_model2-offline-random-forest-20261120T104834452589Z.json`
- A versioned offline training-status JSON is also saved in that directory.

The saved artifact was loaded for inference. A smoke prediction from the
latest historical observation returned class `0` (stable risk), probability
`0.235`, explicit `HISTORICAL_OFFLINE` provenance, Model 1 replay version
`historical-refit-87bf685e4dd39ad8`, and a UTC prediction timestamp. The
holdout report also contains nine timestamped historical predictions. These
are exploratory offline results, not production prediction history.

## API and dashboard verification

- `GET /api/v1/model2/offline/status` and
  `POST /api/v1/model2/offline/predict` use the separate offline artifact and
  provenance.
- The dashboard payload returned `TRAINED`, the current offline model version
  and Model 1 replay version, 45 observations, 41 labels, class counts 24/17,
  45 history rows, and 9
  evaluation predictions.
- On the Model 2 dashboard page, the `HISTORICAL_OFFLINE` panel is now first,
  before the production eligibility panel, so the trained offline status is
  visible without mistaking the production `NOT_TRAINED` status for it.
- The frontend offline-panel component tests passed and verify the
  `HISTORICAL_OFFLINE` disclosure.
- Production readiness remained `NOT_TRAINED`, with the reason that 0 genuine
  production labels are available and 20 are required.
- The authenticated live API returned offline `TRAINED`/41 labels, production
  `NOT_TRAINED`/0 genuine labels, and a successful offline prediction response
  with a probability, model version, replay version, and timestamp.
- The local browser smoke navigated all eleven sections without console
  warnings/errors or failed non-cancelled requests. Mobile navigation worked;
  the dashboard has no horizontal overflow at an effective 242 CSS pixel
  width after a narrow top-bar adjustment.

## Tests and changed files

- `python -m pytest -q` — **229 passed**, 5 warnings.
- `npm --prefix frontend run test -- --run App.test.tsx` — **4 passed**.
- `npm --prefix frontend run test` — **8 passed**.
- `npm --prefix frontend run build` and `npm --prefix frontend run lint` — **passed**.
- The added isolated training test uses explicitly named synthetic offline
  fixtures in a temporary directory. It verifies training eligibility,
  chronological holdout, separate artifact serialization/loading, status, and
  inference; it does not write production data or artifacts.

Documentation and test changes:

- `ml/offline_model2.py` — persist and enforce a single versioned Model 1
  offline replay identity; write versioned replay data; include model versions
  and a UTC timestamp on offline predictions.
- `ml/tests/test_offline_model2.py`, `backend/tests/test_offline_model2_api.py`
  — verify version consistency, fixture training, artifact loading, and
  timestamped, scope-labeled prediction responses.
- `frontend/src/App.tsx` — show the replay version and trained offline panel
  ahead of the distinct production status.
- `docs/model2.md` — record the current offline model version, results, and
  separate completion report.
- `README.md` — link the separate offline report and distinguish offline from
  production status.
- `HISTORICAL_OFFLINE_MODEL2_REPORT.md` — this report.

## Remaining limitations

- Production Model 2 remains `NOT_TRAINED` until verified production ingestion
  supplies at least 20 distinct, fully labeled monitoring batches satisfying
  the existing baseline/future F1 and chronological requirements.
- Historical timestamps are derived from dataset fields, not verified live
  ingestion timestamps.
- The 41-label dataset and nine-row holdout are small, and the holdout metrics
  are poor. More historical or production evidence is needed before claiming
  predictive utility.
- The original source data and Model 1 artifacts are local inputs and are not
  bundled or reproduced by this report.
