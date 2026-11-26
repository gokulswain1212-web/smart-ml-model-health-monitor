from __future__ import annotations

DEGRADATION_TARGET = "performance_degraded"
F1_DEGRADATION_THRESHOLD = 0.05
FUTURE_PERFORMANCE_WINDOW_DAYS = 30
MIN_LABELED_OBSERVATIONS = 20
TEST_FRACTION = 0.20
PSI_CRITICAL_THRESHOLD = 0.20
DEGRADATION_CLASSIFICATION_THRESHOLD = 0.35
LOW_RISK_MAX_PROBABILITY = 0.30
HIGH_RISK_MIN_PROBABILITY = 0.70

MODEL2_FEATURE_NAMES = [
    "dataset_drift_score",
    "max_feature_drift_score",
    "number_drifted_features",
    "missing_value_percentage",
    "invalid_value_percentage",
    "data_quality_score",
    "duplicate_rate",
    "prediction_positive_rate",
    "prediction_positive_rate_change",
    "prediction_mean_probability",
    "historical_accuracy",
    "historical_precision",
    "historical_recall",
    "historical_f1_score",
    "previous_f1_trend",
    "historical_metrics_available",
    "historical_performance_observations",
]

DATA_SOURCE_TYPES = {"real", "synthetic_test", "unknown"}
