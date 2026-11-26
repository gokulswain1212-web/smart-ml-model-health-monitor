# Model 2 End-to-End Completion Report

**Review date:** 20 November 2026  
**Scope:** Production Model 2 dataset, eligibility, training, artifact loading,
prediction, API, frontend, and tests.

## Final status

- **Implementation status: IMPLEMENTATION COMPLETE.** The production Model 2
  pipeline and its API and dashboard integration are implemented and covered
  by tests.
- **Actual production training status: TRAINING BLOCKED BY INSUFFICIENT GENUINE
  DATA.** The current monitoring dataset has **0 of the required 20** genuine
  labeled production observations. No production Model 2 artifact exists, and
  the correct status is `NOT_TRAINED`.
- **TRAINED AND VALIDATED:** **Not for production.** A controlled, temporary
  fixture run verified fitting, evaluation, serialization, reloading, and
  inference. Its metrics are synthetic test results, not evidence of model
  usefulness. A separately labeled historical offline artifact also exists;
  it is not a production model.

## Architecture and data flow

Model 1 predicts booking cancellations and records production observations and
their later actual outcomes. Model 2 uses time-bounded monitoring signals to
predict whether Model 1's F1 score will decline by at least 0.05 within 30
days. Only verified outcomes from trusted production ingestion can create
training labels.

The dataset builder uses quality, drift, prediction, and prior performance
signals available at the monitoring decision timestamp. The future F1 score
is used only to create the target. Training uses a chronological holdout and
purges training labels whose future-evaluation window reaches the test period.
Synthetic, unknown, operator-declared, incomplete, and CSV-imported evidence
is excluded from production labels.

## Dataset and production training readiness

| Measure | Current result |
|---|---:|
| Historical monitoring observations | 7 |
| Source counts | 6 synthetic/test; 1 unknown |
| Genuine production observations | 0 |
| Genuine labeled production observations | **0 / 20** |
| Training eligibility | `NOT_TRAINED` |
| Production trained artifact | None |

The blocker is absence of trusted production-source observation and outcome
records. There are no complete genuine prior and future performance
evaluations from which to form target labels. The minimum requirement remains
20 distinct labeled production monitoring observations and is not reduced.

### Exact production data collection steps

1. Configure a secret `PRODUCTION_INGESTION_TOKEN` for the backend and the
   trusted producer. Do not use this token for test data or expose it in the
   frontend.
2. The producer sends each real booking's `booking_id` and Model 1 feature
   values to `POST /api/v1/monitoring/observations` with the ingestion-token
   header. Use the documented feature order and a distinct batch ID. Preserve
   each returned `observation_id` alongside the source booking.
3. When each real booking outcome is known, submit its `is_canceled` value
   (`0` or `1`) to `POST /api/v1/monitoring/outcomes` using that exact
   `observation_id`. Do not use CSV imports, operator declarations, or
   simulated data for production labels.
4. Complete all outcomes for a batch. A performance evaluation is eligible
   only when every prediction has an outcome, no outcomes are pending, both
   actual classes are present, and the outcome IDs match trusted production
   predictions.
5. Continue collecting complete outcomes for the same Model 1 version. An
   observation gets a target only if it has a complete baseline F1 at or
   before its decision time and a subsequent complete F1 within 30 days.
   The target is degraded (`1`) only when baseline F1 minus subsequent F1 is
   at least `0.05`; otherwise it is stable (`0`).
6. Accumulate at least 20 distinct labeled production monitoring batches.
   The chronological training partition must still contain both target
   classes after training rows whose future-label windows overlap the test
   period are purged.
7. Check `GET /api/v1/model2/training-status` for the current count and exact
   blockers. Once eligible, training runs automatically after production
   outcomes; the same gated workflow is available through
   `python -m ml.model2_training`.

These are real production collection steps. The test suite below uses only
isolated test fixtures and does not create production evidence.

## Training, evaluation, and artifact verification

The pipeline uses median imputation fitted on the training partition only.
Logistic Regression is the baseline; the predeclared Random Forest is the
primary candidate. Evaluation includes accuracy, precision, recall, F1,
confusion matrix, ROC-AUC, and PR-AUC when both test classes are present.
Versioned model, metadata, and evaluation artifacts are staged before
publication.

A controlled run used only temporary records named
`isolated-unit-fixture-*`; the records and resulting files were not written to
production monitoring history or the repository's production artifact
directory.

| Fixture result | Value |
|---|---:|
| Training rows after chronological split/purge | 29 |
| Test rows | 10 |
| Logistic Regression accuracy / F1 / ROC-AUC | 1.000 / 1.000 / 1.000 |
| Random Forest accuracy / F1 / ROC-AUC | 1.000 / 1.000 / 1.000 |
| Confusion matrix, each candidate | `[[5, 0], [0, 5]]` |
| Saved artifact reloaded and prediction completed | Yes |
| Fixture prediction probability / class | 0.025 / 0 |

These perfect fixture scores result from a deliberately simple generated
relationship. They verify the code path only; they are not production
evaluation metrics and do not establish that Model 2 is useful.

An additional isolated end-to-end fixture writes 20 explicitly
`SYNTHETIC_TEST_FIXTURE_ONLY` batches, derives all 20 labels from complete
baseline/future F1 records for one test Model 1 version, then trains with a
16-row chronological training partition and 4-row test partition. It saves,
reloads, and predicts with the temporary artifact. Fixture records and
artifacts remain under pytest's temporary directory. A separate API
integration test covers trusted observation ingestion, observation-ID outcome
matching, incomplete-versus-complete outcomes, and target generation; it
asserts that synthetic and unknown-source records stay unlabeled.

Prediction now rejects artifacts that do not verify the minimum genuine
production observation count, expected provenance, canonical feature order,
binary classes, fitted median-imputation pipeline, and valid feature
importances. Missing, corrupt, or incompatible artifacts do not yield risk
predictions.

## API and frontend integration

| Path | Behavior verified |
|---|---|
| `GET /api/v1/model2/training-status` | Returns actual readiness, count, minimum, blockers, and artifact/training information. |
| `POST /api/v1/predictions/performance-degradation` | Uses a trained artifact only; returns `503` if untrained/invalid, `422` for invalid features, and `404` for an unknown production batch. |
| Production outcome workflow | Runs gated training automatically when trusted outcomes arrive. There is no separate training HTTP endpoint. |
| Prediction history | Persisted production Model 2 predictions are included in authenticated dashboard data; there is no separate history route. |

The Model 2 dashboard shows status, genuine labeled count against the
minimum, eligibility blockers, artifact/version information, holdout metrics
only when trained, and persisted production prediction history. The
prediction UI is disabled until status is `TRAINED`; it validates and displays
the probability, risk classification, timestamp, and model version. Empty
history is shown as empty rather than fabricated.

## Separate historical offline experiment

The latest `HISTORICAL_OFFLINE` artifact has 45 historical observations, 41
eligible labels, 31 training rows, and 9 test rows. Its holdout results remain
weak: accuracy 0.333, precision 0, recall 0, F1 0, ROC-AUC 0.400, and confusion
matrix `[[3, 1], [5, 0]]`. This is a small, exploratory Kaggle-derived
historical experiment—not trusted production evidence, not a production
artifact, and not a substitute for the production 20-observation gate. See the
[separate historical offline report](./HISTORICAL_OFFLINE_MODEL2_REPORT.md)
for the current model version, full split results, and dashboard verification.

## Modified files

- `ml/model2_prediction.py` — reject ineligible metadata and incompatible
  estimator/preprocessing artifacts before inference.
- `ml/tests/test_model2_dataset.py` — add a full isolated dataset-to-training-
  to-prediction pipeline fixture with explicit test-only provenance.
- `ml/tests/test_model2_training_prediction.py` — test minimum training
  evidence, class mapping, feature order, fixture training, artifact
  serialization, and prediction.
- `backend/tests/test_dashboard_api.py` — align the valid artifact fixture
  with production metadata requirements.
- `docs/model2.md` and `docs/api.md` — document artifact validation, training
  triggers, API availability, and persisted prediction history.
- `README.md` — link to this completion report.

The dataset builder, gated production trainer, performance tracking,
FastAPI routes/dashboard aggregation, and React integration were reviewed and
required no further code changes in this pass.

## Verification

- `python -m pytest -q` — **229 passed**, 5 existing warnings.
- `npm --prefix frontend run test` — **2 files, 8 tests passed**.
- `npm --prefix frontend run build` — **passed**, including TypeScript build.
- `npm --prefix frontend run lint` — **passed**.
- Live dataset/readiness inspection — **0 / 20** genuine labeled production
  observations; no production Model 2 artifact.
- Isolated temporary training and prediction — **passed**, with test-only
  metrics as listed above.

Warnings in the Python suite were one Starlette/httpx deprecation and four
SciPy/NumPy KS-test fallbacks to the asymptotic method; no tests failed.

## Remaining limitations

- Production training and production risk predictions remain blocked until
  genuine trusted observations and delayed outcomes produce at least 20
  eligible labels and satisfy all chronology/class/split checks.
- No trusted producer or production ingestion token is configured or
  verified here.
- Offline and synthetic development models are separate experiments; neither
  can be promoted as production evidence.
- The application remains a local prototype; this report does not establish
  production deployment, operational security, or predictive utility.

**Conclusion:** Model 2's implementation and integration paths are technically
complete and verified. Smart ML Model Health Monitor is **not complete for
production use**; production Model 2 training is correctly blocked by
insufficient genuine data.
