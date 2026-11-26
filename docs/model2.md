# Model 2: Performance Degradation Risk

## Status

Model 2 is a separate, supervised classifier that is intended to predict whether
Model 1's future F1 score will decline materially. Production Model 2 remains
`NOT_TRAINED` until it receives eligible, verified outcomes from genuine
production batches. A separate `HISTORICAL_OFFLINE` Model 2 is trained from
chronological Kaggle evaluation evidence and is not production evidence.

## Architecture and data flow

1. `ml/monitoring.py` saves quality and drift reports with a data-source
   declaration (`real`, `synthetic_test`, or `unknown`). Seeded simulator batch
   IDs in older reports are recognized as synthetic.
2. `ml/model2_dataset.py` selects the latest report for each batch, aggregates
   its PSI and quality signals, and joins prediction/performance observations
   only when they are available by the Model 2 decision timestamp (the later of
   the monitoring-report timestamp and the batch prediction timestamp).
3. Production observations enter through
   `POST /api/v1/monitoring/observations` with the trusted producer token.
   The API runs Model 1, stores a production-source monitoring report and
   prediction rows, and returns a distinct `observation_id` for each booking.
   The producer must retain the ID-to-booking mapping and later send the
   booking's actual `is_canceled` value to
   `POST /api/v1/monitoring/outcomes`. Outcomes are associated with predictions
   by that ID, not by row order.
4. Only outcomes written by the token-authenticated production API are eligible
   production evidence. Each outcome is linked by its immutable
   `observation_id`, checked against a production-ingestion prediction, and
   records the producer provenance and receipt time. Session-authenticated CSV
   imports and operator-declared `real` reports cannot create production Model
   2 labels. Duplicate, unknown, and already-labeled observation IDs are
   rejected.
5. An observation remains unlabeled until a complete genuine performance
   evaluation exists both at/before its decision time and after it. A complete
   evaluation has actual outcomes for every prediction in that batch:
   `number_of_actual_outcomes_received` and
   `number_of_evaluated_records` equal `number_of_predictions`, and
   `number_of_pending_outcomes` is zero. Partial snapshots remain useful for
   monitoring but are not used to create Model 2 targets.
6. The baseline is the latest complete F1 evaluation at or before the
   observation's decision time. The future value is the first subsequent
   complete F1 evaluation for the same Model 1 version, no later than 30 days
   after that decision time. The target is `1` when baseline F1 minus future
   F1 is at least `0.05`, and `0` otherwise. Without both complete scores,
   `performance_degraded` stays missing; it is never inferred from partial
   outcomes.
7. `ml/model2_training.py` requires at least 20 labeled production monitoring
   observations (one training row per monitored batch, not per booking row),
   both target classes in the chronological training partition, valid
   timestamps, and enough non-overlapping future-label windows after purging.
   It then evaluates Logistic Regression as a baseline and Random Forest as
   the primary model.
   After each token-authenticated production outcome is recorded, the backend
   checks eligibility and trains only when the checks pass. An existing model
   is not retrained until the eligible labeled observation count increases.
   Training stages its model, preprocessing pipeline, feature schema, metadata,
   and evaluation report before publishing the versioned artifact set; metadata
   is published last as the commit marker. A failed attempt preserves the prior
   valid artifact and records a `NOT_TRAINED` status with the exact blocker.
8. `ml/model2_prediction.py` returns `NOT_TRAINED` when no trained artifact is
   available. Before inference, it verifies the artifact records at least 20
   genuine labeled production observations, matches the canonical feature
   order, contains both target classes, and includes the fitted median
   imputer. Trained artifacts are kept under `ml/models/model2/`, separate
   from Model 1.
9. `backend/main.py` exposes a dedicated prediction route without changing any
   Model 1 behavior. Successful stored-production predictions are persisted
   by `ml/performance_tracking.py`; the authenticated dashboard response
   exposes their history, probability, timestamp, risk category, and model
   version. There is no unauthenticated or manual HTTP training route: the
   gated CLI and production-outcome workflow use the same eligibility checks.

The local CSV monitoring command can still make an operator provenance
declaration, but those records are not eligible to train Model 2. CSV outcome
imports remain for legacy/non-production performance analysis only and cannot
label production batches. Production training data must enter through the
token-authenticated collection endpoints
documented in [api.md](./api.md#trusted-production-observation-and-outcome-collection).
Never call a locally generated batch `production`.

Manual monitoring of a batch may still declare `real`:

```powershell
python -m ml.monitoring path\to\batch.csv --data-source-type real
```

That declaration alone does not make the batch eligible for Model 2 training.
Do not use `real` for simulated or otherwise unverified inputs.

## Historical offline Model 2

The offline experiment uses only the existing Kaggle chronological partitions:

- Historical Model 1 training/reference partition: 83,580 bookings through
  2016-11-06 by `estimated_booking_date`.
- Future evaluation partition: 35,810 later bookings, with actual Kaggle
  `is_canceled` labels.
- Actual Model 1 predictions were not available as persisted row-level records.
  The saved Model 1 artifact was trained using a random split across the full
  period, so using it on the future partition would leak future-period training
  data. To avoid that leakage, the experiment refits the same saved pipeline
  recipe and classifier configuration using the historical partition only,
  then predicts every row in the later evaluation partition. The production
  Model 1 artifacts are not changed.
- Outcome-week batches are formed by Kaggle's `reservation_status_date`.
  Predictions and actual labels remain aligned in memory; only aggregate batch
  monitoring observations are persisted. Rows with incomplete outcomes or
  single-class actual outcomes are excluded from eligibility.
- Each observation records a deterministic `model1_replay_version` derived
  from the historical partition, Model 1 metadata and pipeline artifact,
  replay recipe revision, and scikit-learn version. Eligibility rejects missing
  or mixed replay versions. Replays produce a new versioned CSV rather than
  overwriting earlier derived datasets.
- PSI and numerical KS compare each offline outcome week against the historical
  feature reference. Performance and target labels use actual `is_canceled`
  labels. Model 2 inputs contain only current distribution/quality/prediction
  statistics and performance from earlier eligible batches; future F1 is used
  only to create `performance_degraded`.
- The existing target definition is unchanged: compare the latest eligible
  earlier F1 with the first eligible later F1 within 30 days; label degraded
  when the absolute decline is at least 0.05. The chronological Model 2
  evaluation split purges training rows whose future-target window reaches the
  test period.

The current offline run trained version
`model2-offline-random-forest-20261120T104834452589Z` with Model 1 replay
version `historical-refit-87bf685e4dd39ad8`. It produced 45
outcome-week observations, all marked `HISTORICAL_OFFLINE`; 43 had both actual
cancellation classes, 41 had valid future-F1 target labels (24 stable, 17
degraded), and 4 remained unlabeled. The chronological split and overlapping
target-window purge left 31 training rows and 9 test rows. The model uses a
median imputer fitted on training rows and a 200-tree class-balanced random
forest (seed 42). Holdout accuracy was 0.333, precision 0, recall 0, F1 0,
ROC-AUC 0.400, with confusion matrix `[[3, 1], [5, 0]]`. This tiny, weak
holdout is not evidence that the offline predictor is useful; its predictions
are exploratory only. A smoke prediction from the saved offline artifact
returned stable risk (class 0) with probability 0.235 and explicitly
`HISTORICAL_OFFLINE` provenance, with a UTC prediction timestamp. Its
versioned replay dataset is
`ml/models/offline_model2/historical_offline_monitoring_dataset_historical-refit-87bf685e4dd39ad8.csv`.

Five fixed estimator candidates were also compared with purged expanding-time
validation on only 18 pre-holdout rows. Logistic Regression had F1 0.615,
precision 0.800, recall 0.500, ROC-AUC 0.675, and PR-AUC 0.737; a
HistGradientBoosting candidate had recall 1.0 but predicted every case
positive. The validation sample is too small to select a replacement or claim
improvement. No broad hyperparameter search was run because it would overfit
the small dataset; the saved Random Forest remains the reproducible baseline,
not a validated choice.

The latest run's exact artifacts, split boundaries, test predictions, API and
dashboard verification, and limitations are recorded in the
[HISTORICAL_OFFLINE Model 2 report](../HISTORICAL_OFFLINE_MODEL2_REPORT.md).

These are historical dataset results, not independently verified production
outcomes. `estimated_booking_date` is derived from arrival date minus lead time,
and Kaggle's `reservation_status_date` is not a verified production receipt
timestamp. Do not interpret this experiment as a live temporal deployment
validation. No historical offline batch counts toward the production minimum.

Run or reproduce the separate experiment with:

```powershell
python -m ml.offline_model2
```

Its model, monitoring dataset, metadata, evaluation report, and status are kept
under `ml/models/offline_model2/`, separate from `ml/models/model2/`. The
authenticated endpoints `GET /api/v1/model2/offline/status` and
`POST /api/v1/model2/offline/predict` serve only this historical artifact.
Offline predictions are not written to production prediction history.
The dashboard labels the experiment `HISTORICAL_OFFLINE` and displays the
required notice:

> Historical offline model: trained and evaluated using Kaggle booking data and
> Model 1 offline evaluation results. Not validated using live production
> observations.

For legacy/non-production batches, actual outcomes can be imported from CSV
only when the batch has an explicitly real monitoring report, using
`POST /api/v1/outcomes/{batch_id}/csv`. The CSV must contain the stable
prediction `row_id` and `is_canceled` label (0 or 1) for each row. Uploads are
validated against the batch's saved predictions before any outcomes are
recorded. See [the API documentation](./api.md#import-actual-cancellation-outcomes)
for the CSV format and response details. Seeded and unknown-source batches
cannot be used for genuine performance evaluations.

## Features

The versioned feature list is in `ml/model2_config.py` and is written to
`ml/models/model2/feature_names.json` when the preparation/training command is
run. It includes aggregate PSI and drifted-feature count, missing/invalid and
duplicate rates, a data-quality score, prediction-class distribution and its
change from the preceding batch, and performance metrics/trend known by the
observation timestamp.

The quality score is `max(0, 100 - missing_value_percentage -
invalid_value_percentage)`. Invalid percentages aggregate invalid numerical,
infinite, negative, and unexpected categorical values over model feature cells.

Numerical missing values remain missing in the prepared dataset. Training
pipelines impute medians using the training partition only; Logistic Regression
also scales imputed features. Both candidates use class weights to account for
imbalance. Test data is the newest chronological partition and is not used to
fit preprocessing or estimators. Training rows whose future-label measurement
reaches the test period are purged to prevent their future outcomes from
crossing the chronological boundary.

## Target definition and leakage controls

The observation timestamp is the later of its production monitoring-report
timestamp and its Model 1 prediction time. `performance_degraded = 1` when the
first *complete* genuine F1 evaluation for the same Model 1 version after that
timestamp, within the configured 30-day horizon, is at least 0.05 below the
latest *complete* genuine F1 evaluation available at or before that timestamp.
Otherwise the target is `0`. An observation without both complete evaluations
remains unlabeled. A complete evaluation covers every prediction in its batch;
partial outcome snapshots never create labels or serve as baseline/future F1.

`target_future_f1`, the target drop, and the target itself are not model
features. Reports marked synthetic or unknown never receive training labels.
Only evaluation records with all actual outcomes received and available,
finite F1 metrics, verified production outcome provenance, and contributing
observation IDs matching the trusted production prediction records qualify as
Model 2 performance observations. Synthetic-test, unknown-provenance,
operator-declared-real, and CSV-imported outcomes are excluded. Historical
accuracy/precision/recall/F1 and trends are strictly as-of features, not future
values.

The metric threshold, future horizon, minimum labeled sample count, split
fraction, positive-class decision threshold (0.35 by default to favor recall),
and risk probability thresholds are configurable in
`ml/model2_config.py`.
The 20-row minimum applies to distinct, labeled production monitoring batches
whose complete future-F1 targets can be established; collecting 20 booking
outcomes from only one or a few batches does not by itself make Model 2
trainable. The eligible chronological training partition must also contain
both target classes after the overlapping target windows are purged.

## Training and artifacts

Run preparation and the gated training attempt with:

```powershell
python -m ml.model2_training
```

Production outcome submissions also trigger the same gated training workflow.
The response includes the current Model 2 status and, when trained, the model
version and holdout metrics. No synthetic, operator-declared, unknown-source,
or unlabeled observations are promoted to production training data.

Only prepare and inspect the historical dataset with:

```powershell
python -m ml.model2_training --prepare-only
```

### Isolated synthetic development experiment

The development experiment verifies that feature preprocessing, Random Forest
fitting, stratified holdout evaluation, and artifact serialization work without
production labels. It does not read the monitoring dataset or production
observation files. Its synthetic target is a reproducible Bernoulli sample
derived from monitoring features and is not the genuine future-F1 target.

Run it separately:

```powershell
python -m ml.model2_development_training
python -m ml.model2_development_training --samples 1000 --seed 42
```

Development datasets, models, metadata, and evaluation reports are stored
under `ml/models/model2/development/`. Each carries the
`SYNTHETIC_DEVELOPMENT_ONLY` provenance label. The holdout is stratified and
reports accuracy, precision, recall, F1, confusion matrix, and classification
report. Reusing the same seed reproduces the generated data and metrics.
These metrics measure only the synthetic experiment; they must never be
presented as production performance. Production artifact discovery does not
search this directory, and there is no automatic promotion path.

Timestamped prepared CSV files are saved under
`data/monitoring/model2/`. Target definition, feature names, training status,
trained estimators, and versioned metadata are saved under
`ml/models/model2/`. An untrained status is not a model artifact and contains
no fabricated evaluation metrics.

The Monitoring dashboard reports production observation count, verified
labeled count, minimum required count, eligibility blockers, class counts, and
the latest training attempt. For a trained production artifact it shows model
version, training time, dataset/train/test sizes, and chronological holdout
accuracy, precision, recall, F1, confusion matrix, and ROC-AUC when defined.
Production prediction history is recorded only for predictions made from a
trusted production batch and is labeled separately from development
predictions.

### Synthetic development explainability

Generate a feature-importance report from the isolated development Random
Forest with:

```powershell
python -m ml.model2_development_explainability
```

The command verifies that fitted feature names exactly match
`MODEL2_FEATURE_NAMES`, validates finite non-negative importance values and
their sum, then saves `ml/models/model2/development/feature_importance.json`.
The top-ten chart and feature explanations are written under
`ml/models/model2/development/reports/`.

The displayed importance is Random Forest mean decrease in impurity across
tree splits. It describes global model reliance in this synthetic experiment;
it does not indicate effect direction or prove that a feature causes
degradation. Correlated features can share or obscure importance, and
impurity-based importance can favor features with many possible split points.
All these values are labeled `SYNTHETIC_DEVELOPMENT_ONLY` and must not be
presented as production explainability or evidence of live model risk. The
dashboard keeps this section separate from production Model 2 status and
reports unavailable for missing or invalid development explainability files
instead of substituting production values.

The primary estimator is Random Forest; Logistic Regression is the baseline.
Reported holdout metrics include accuracy, precision, recall, F1, confusion
matrix, ROC-AUC, and average precision (PR-AUC) where both classes permit.
Recall is emphasized when assessing a missed degradation event. No metric is
reported as real-world performance unless it comes from the chronological
holdout of genuine labeled observations.

The minimum of 20 genuine labeled production observations remains mandatory
for this production path. Synthetic development data and its artifacts are
never eligible for production training or prediction. Development predictions
are available only when a valid development model artifact exists; production
predictions remain unavailable until production training succeeds.

Random Forest feature importance is included as a descriptive ranking, not as
causal evidence.

## API

`POST /api/v1/predictions/performance-degradation`

Run the API from the project root with:

```powershell
python -m uvicorn backend.main:app --env-file .env --reload --host 127.0.0.1 --port 8000
```

Request using stored batch signals:

```json
{"batch_id": "a-monitoring-batch"}
```

Or provide the feature vector directly:

```json
{"monitoring_features": {"dataset_drift_score": 0.12}}
```

Missing feature values may be `null`; extra keys and non-numeric values are
rejected. For stored production `batch_id` requests, a successful prediction is saved
with its timestamp and batch ID in production prediction history. Ad-hoc
operator-supplied feature predictions are marked unverified and are not saved
as production evidence. The response includes Model 2 status, binary risk,
probability, `LOW`/`MEDIUM`/`HIGH` risk category, mapped health status
(`HEALTHY`/`WARNING`/`CRITICAL`), ranked signals, explanation, prediction
timestamp, and model version. Before training, the API responds with HTTP `503` Service
Unavailable` and a reason; the internal prediction function reports
`NOT_TRAINED` with null prediction fields. Missing or invalid trained
artifacts also produce an explicit `503` response rather than a prediction.

### Synthetic development prediction

`POST /api/v1/model2/development/predict` is a separate, authenticated endpoint
for exercising the saved development Random Forest. It loads only the
versioned model and JSON metadata directly under
`ml/models/model2/development/`; it never calls the production Model 2 loader
and never falls back to either production model. Development artifacts are not
copied or promoted into the production artifact directory.

The request contains one complete feature object using the exact order in
`MODEL2_FEATURE_NAMES` (the dashboard obtains these names from its backend
response rather than defining a second schema):

```json
{
  "monitoring_features": {
    "dataset_drift_score": 0.1,
    "max_feature_drift_score": 0.2,
    "number_drifted_features": 1,
    "missing_value_percentage": 2.0,
    "invalid_value_percentage": 1.0,
    "data_quality_score": 97.0,
    "duplicate_rate": 0.02,
    "prediction_positive_rate": 0.4,
    "prediction_positive_rate_change": 0.05,
    "prediction_mean_probability": 0.5,
    "historical_accuracy": 0.8,
    "historical_precision": 0.7,
    "historical_recall": 0.7,
    "historical_f1_score": 0.7,
    "previous_f1_trend": -0.05,
    "historical_metrics_available": 1,
    "historical_performance_observations": 8
  }
}
```

All 17 values must be finite JSON numbers; null, missing, extra, or out-of-order
features are rejected. Rates and percentages must be within their documented
domains (rates in `[0, 1]`, percentages in `[0, 100]`, and F1 deltas in
`[-1, 1]`); counts must be non-negative whole numbers, and the availability
flag must be 0 or 1. The endpoint does not impute or synthesize missing
features. Invalid requests return `422`. Missing development files, corrupt or
incompatible metadata/model files, and prediction failures return `503` with
an explicit reason. There is no production fallback.

On success, the response includes `api_status: "DEVELOPMENT_PREDICTION"`, the
model version, `SYNTHETIC_DEVELOPMENT_ONLY` provenance, predicted class and
label, class probabilities, risk-assessment details, and a prediction
timestamp. The class mapping is read from the existing Model 2 target
definition (`0 = stable`, `1 = degraded`). `degradation_probability` is the
model's probability for class 1. The target definition provides no probability
thresholds for a risk category, so `risk_level` is null; the endpoint does not
invent LOW/MEDIUM/HIGH categories. The synthetic training target is generated
from a synthetic feature-based rule, not from genuine future F1 observations.
Consequently, a returned class/probability is only a development workflow
result—not production degradation status, production performance, or evidence
about the future performance of Model 1.

Example success response:

```json
{
  "api_status": "DEVELOPMENT_PREDICTION",
  "model_name": "Model 2 Synthetic Development Experiment",
  "model_version": "model2-development-random-forest-seed-42",
  "model_provenance": "SYNTHETIC_DEVELOPMENT_ONLY",
  "target_name": "performance_degraded",
  "prediction": 1,
  "predicted_class_label": "degraded",
  "class_probabilities": {"stable": 0.335, "degraded": 0.665},
  "risk_assessment": {
    "predicted_class": "degraded",
    "degradation_probability": 0.665,
    "risk_level": null
  },
  "explanation": "Synthetic development-only prediction; not production health status.",
  "prediction_timestamp": "2026-11-06T10:26:47.140040+00:00"
}
```

The dashboard's **Model 2 Development Risk Assessment** is visually separate
from production Model 2 status. It requires an operator to enter all feature
values, displays validation/API errors and unavailable-artifact states, and
labels each result as synthetic development only.

## Current limitations

- Current seeded batches have no reliable source-row-to-outcome mapping.
- Existing seeded performance snapshots contain zero actual outcomes.
- The original hotel booking CSV contains historical cancellation labels and
  supports retrospective Model 1 held-out analysis, but it does not provide
  outcome observations linked to real-time monitoring batches. It must not be
  used to fabricate Model 2 future-degradation labels.
- Existing monitoring reports predate the provenance field; seeded simulator
  IDs are recognized, while other undeclared sources are treated as unknown.
- Manual `real` provenance declarations are not sufficient for Model 2
  eligibility. Only records ingested by the configured trusted production
  producer are eligible.
- No production training, evaluation metrics, or degradation-risk predictions
  can be claimed until the required number of verified production batches have
  complete subsequent outcomes and pass the class-balance and chronological
  split requirements. Declaring a batch `real` or importing historical CSV
  outcomes is not sufficient.
