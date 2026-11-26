# Model 2 Development Feature Importance

- **Provenance:** SYNTHETIC_DEVELOPMENT_ONLY
- **Model version:** model2-development-random-forest-seed-42
- **Generated:** 2026-11-06T09:58:23.465714+00:00
- **Method:** Random Forest mean decrease in impurity
- **Chart:** `feature_importance_top10.png`

## Ranked features

### 1. `historical_f1_score` — 0.110270

- **Measures:** Latest genuine F1 score available before the monitoring observation.
- **Why the model may use it:** F1 summarizes recent precision and recall and was directly used in the synthetic target-generation process.
- **Relation to risk:** Its prominence may reflect the synthetic label rule and must not be interpreted as measured production behavior.

### 2. `dataset_drift_score` — 0.107192

- **Measures:** Aggregate distribution drift between reference and current input data.
- **Why the model may use it:** A broad shift in input distributions may signal that the model is operating on less familiar data.
- **Relation to risk:** Synthetic labels were more likely as this score increased; this is a learned association in the development fixture.

### 3. `previous_f1_trend` — 0.081874

- **Measures:** Recent trend in genuine F1 values known as of the observation.
- **Why the model may use it:** A declining historical trend can provide an early warning signal.
- **Relation to risk:** The synthetic target rule explicitly used negative trend, so its importance partly reflects the constructed experiment.

### 4. `prediction_positive_rate_change` — 0.067024

- **Measures:** Change in positive prediction rate relative to the preceding batch.
- **Why the model may use it:** A sudden change in prediction mix can highlight behavior changes worth investigating.
- **Relation to risk:** This feature's importance represents predictive association in this trained synthetic model only.

### 5. `prediction_mean_probability` — 0.066433

- **Measures:** Mean predicted probability across records in a batch.
- **Why the model may use it:** A shift in confidence distribution can accompany changes in inputs or predictions.
- **Relation to risk:** It does not measure calibration or correctness by itself.

### 6. `data_quality_score` — 0.065764

- **Measures:** A bounded score derived from missing and invalid value percentages.
- **Why the model may use it:** It summarizes data-quality signals in a single feature.
- **Relation to risk:** Its importance may overlap with missingness and invalid-value features and should not be read as an independent effect.

### 7. `duplicate_rate` — 0.060844

- **Measures:** Fraction of duplicate records in the monitored batch.
- **Why the model may use it:** Duplicate records can alter the effective input population and batch summaries.
- **Relation to risk:** Any relationship shown here is specific to the synthetic development data.

### 8. `prediction_positive_rate` — 0.060802

- **Measures:** Fraction of Model 1 predictions assigned to the positive class.
- **Why the model may use it:** A changed prediction mix can reflect changes in inputs or model behavior.
- **Relation to risk:** It is a monitoring signal, not proof that the model's true performance has degraded.

### 9. `missing_value_percentage` — 0.058230

- **Measures:** Percentage of monitored feature values that are missing.
- **Why the model may use it:** Missingness can affect the inputs and preprocessing behavior seen by Model 1.
- **Relation to risk:** The synthetic label rule made higher missingness a risk signal; this is a designed association, not causation.

### 10. `invalid_value_percentage` — 0.058093

- **Measures:** Percentage of monitored feature cells with invalid, infinite, negative, or unexpected values.
- **Why the model may use it:** Invalid values can indicate changes in upstream data generation or quality.
- **Relation to risk:** The synthetic label rule made higher invalid-value rates a risk signal; production evidence is still required.

## Interpretation limits

Feature importance describes model reliance in this synthetic development experiment. It does not show that a feature causes performance degradation.
Random Forest impurity-based importance measures how much a feature contributes to reducing split impurity across the fitted trees. It can favor variables with many possible split points, can distribute importance among correlated features, and does not show effect direction or causal impact. All labels and metrics in this report are synthetic development results and must not be presented as production evidence.
