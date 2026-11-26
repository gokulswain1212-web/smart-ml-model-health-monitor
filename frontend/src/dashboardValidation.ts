export type DevelopmentSimulationSummary = {
  data_source_type: "synthetic";
  source: "synthetic_simulator";
  environment: "development";
  is_synthetic: true;
  mode: "DEVELOPMENT SIMULATION";
  status: "NOT_RUN" | "AVAILABLE" | "UNAVAILABLE";
  run_id: string | null;
  batch_count: number;
  booking_count: number;
  seed: number | null;
  baseline_f1_score: number | null;
  minimum_batch_f1_score: number | null;
  drift_detected_batch_count: number;
  monitoring_status_counts: Record<string, number>;
  model2_training: {
    status: string;
    model_version: string;
    training_size: number;
    testing_size: number;
    split_method: string;
    training_class_counts: Record<string, number>;
    testing_class_counts: Record<string, number>;
    evaluation_metrics: {
      accuracy: number;
      precision: number;
      recall: number;
      f1_score: number;
    };
  } | null;
  reason: string | null;
};

export type HistoricalReplaySummary = {
  status: "NOT_RUN" | "AVAILABLE" | "UNAVAILABLE";
  model_scope: "HISTORICAL_OFFLINE";
  provenance: "HISTORICAL_OFFLINE";
  data_source_type: "historical_offline";
  source: "historical_hotel_booking_dataset";
  environment: "offline";
  is_synthetic: false;
  run_id: string | null;
  historical_record_count: number;
  arrival_window_count: number;
  arrival_date_range: string[];
  model1_metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
  } | null;
  model2_training_status: "TRAINED" | "NOT_ELIGIBLE" | null;
  model2_eligibility: {
    status: "READY" | "NOT_ELIGIBLE";
    eligible_observation_count: number;
    target_class_counts: Record<string, number>;
    training_size_after_purge: number;
    testing_size: number;
    training_class_counts: Record<string, number>;
    testing_class_counts: Record<string, number>;
    missing_requirements: string[];
    reason: string | null;
  } | null;
  prediction_scope: string | null;
  booking_created_timestamp_available: boolean | null;
  limitations: string[];
  reason: string | null;
};

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNumberRecord(
  value: unknown,
): value is Record<string, number> {
  return (
    isObject(value) &&
    Object.values(value).every(
      (item) => typeof item === "number" && Number.isFinite(item),
    )
  );
}

export function isDevelopmentSimulationSummary(
  value: unknown,
): value is DevelopmentSimulationSummary {
  if (
    !isObject(value) ||
    value.data_source_type !== "synthetic" ||
    value.source !== "synthetic_simulator" ||
    value.environment !== "development" ||
    value.is_synthetic !== true ||
    value.mode !== "DEVELOPMENT SIMULATION" ||
    !["NOT_RUN", "AVAILABLE", "UNAVAILABLE"].includes(String(value.status)) ||
    (value.reason !== null && typeof value.reason !== "string") ||
    typeof value.batch_count !== "number" ||
    !Number.isInteger(value.batch_count) ||
    typeof value.booking_count !== "number" ||
    !Number.isInteger(value.booking_count) ||
    typeof value.drift_detected_batch_count !== "number" ||
    !Number.isInteger(value.drift_detected_batch_count) ||
    !isFiniteNumberRecord(value.monitoring_status_counts) ||
    !Object.values(value.monitoring_status_counts).every(
      (count) => Number.isInteger(count) && count >= 0,
    )
  ) {
    return false;
  }

  if (value.status !== "AVAILABLE") {
    return (
      value.run_id === null &&
      value.batch_count === 0 &&
      value.booking_count === 0 &&
      value.seed === null &&
      value.baseline_f1_score === null &&
      value.minimum_batch_f1_score === null &&
      value.drift_detected_batch_count === 0 &&
      value.model2_training === null &&
      typeof value.reason === "string"
    );
  }

  const training = value.model2_training;
  if (
    typeof value.run_id !== "string" ||
    value.run_id.length === 0 ||
    value.batch_count < 20 ||
    value.booking_count <= 0 ||
    typeof value.seed !== "number" ||
    !Number.isInteger(value.seed) ||
    value.seed < 0 ||
    typeof value.baseline_f1_score !== "number" ||
    !Number.isFinite(value.baseline_f1_score) ||
    value.baseline_f1_score < 0 ||
    value.baseline_f1_score > 1 ||
    typeof value.minimum_batch_f1_score !== "number" ||
    !Number.isFinite(value.minimum_batch_f1_score) ||
    value.minimum_batch_f1_score < 0 ||
    value.minimum_batch_f1_score > 1 ||
    value.drift_detected_batch_count < 0 ||
    value.reason !== null ||
    !isObject(training)
  ) {
    return false;
  }

  const classCountsValid = (counts: unknown): boolean =>
    isFiniteNumberRecord(counts) &&
    Object.keys(counts).length === 2 &&
    ["0", "1"].every(
      (label) => Number.isInteger(counts[label]) && counts[label] > 0,
    );
  const metrics = training.evaluation_metrics;
  return (
    training.status === "DEVELOPMENT_SIMULATION_TRAINED" &&
    typeof training.model_version === "string" &&
    training.model_version.length > 0 &&
    typeof training.training_size === "number" &&
    Number.isInteger(training.training_size) &&
    training.training_size > 0 &&
    typeof training.testing_size === "number" &&
    Number.isInteger(training.testing_size) &&
    training.testing_size > 0 &&
    training.training_size + training.testing_size === value.batch_count &&
    training.split_method === "chronological holdout" &&
    classCountsValid(training.training_class_counts) &&
    classCountsValid(training.testing_class_counts) &&
    isObject(metrics) &&
    ["accuracy", "precision", "recall", "f1_score"].every(
      (name) =>
        typeof metrics[name] === "number" &&
        Number.isFinite(metrics[name]) &&
        metrics[name] >= 0 &&
        metrics[name] <= 1,
    )
  );
}

export function isHistoricalReplaySummary(
  value: unknown,
): value is HistoricalReplaySummary {
  if (
    !isObject(value) ||
    !["NOT_RUN", "AVAILABLE", "UNAVAILABLE"].includes(String(value.status)) ||
    value.model_scope !== "HISTORICAL_OFFLINE" ||
    value.provenance !== "HISTORICAL_OFFLINE" ||
    value.data_source_type !== "historical_offline" ||
    value.source !== "historical_hotel_booking_dataset" ||
    value.environment !== "offline" ||
    value.is_synthetic !== false ||
    typeof value.historical_record_count !== "number" ||
    !Number.isInteger(value.historical_record_count) ||
    value.historical_record_count < 0 ||
    typeof value.arrival_window_count !== "number" ||
    !Number.isInteger(value.arrival_window_count) ||
    value.arrival_window_count < 0 ||
    !Array.isArray(value.arrival_date_range) ||
    !value.arrival_date_range.every((date) => typeof date === "string") ||
    !Array.isArray(value.limitations) ||
    !value.limitations.every((item) => typeof item === "string") ||
    (value.reason !== null && typeof value.reason !== "string")
  ) {
    return false;
  }

  if (value.status !== "AVAILABLE") {
    return (
      value.run_id === null &&
      value.historical_record_count === 0 &&
      value.arrival_window_count === 0 &&
      value.arrival_date_range.length === 0 &&
      value.model1_metrics === null &&
      value.model2_training_status === null &&
      value.model2_eligibility === null &&
      value.prediction_scope === null &&
      value.booking_created_timestamp_available === null &&
      value.limitations.length === 0 &&
      typeof value.reason === "string"
    );
  }

  const metrics = value.model1_metrics;
  const eligibility = value.model2_eligibility;
  const validCounts = (counts: unknown): boolean =>
    isFiniteNumberRecord(counts) &&
    Object.values(counts).every(
      (count) => Number.isInteger(count) && count >= 0,
    );
  return (
    typeof value.run_id === "string" &&
    value.run_id.length > 0 &&
    value.historical_record_count > 0 &&
    value.arrival_window_count > 0 &&
    value.arrival_date_range.length === 2 &&
    value.arrival_date_range.every((date) => date.length > 0) &&
    value.reason === null &&
    value.prediction_scope ===
      "retrospective_historical_feature_replay_not_booking_time" &&
    value.booking_created_timestamp_available === false &&
    Array.isArray(value.limitations) &&
    value.limitations.length > 0 &&
    isObject(metrics) &&
    ["accuracy", "precision", "recall", "f1_score"].every(
      (name) =>
        typeof metrics[name] === "number" &&
        Number.isFinite(metrics[name]) &&
        metrics[name] >= 0 &&
        metrics[name] <= 1,
    ) &&
    (value.model2_training_status === "TRAINED" ||
      value.model2_training_status === "NOT_ELIGIBLE") &&
    isObject(eligibility) &&
    (eligibility.status === "READY" ||
      eligibility.status === "NOT_ELIGIBLE") &&
    [
      "eligible_observation_count",
      "training_size_after_purge",
      "testing_size",
    ].every(
      (name) =>
        typeof eligibility[name] === "number" &&
        Number.isInteger(eligibility[name]) &&
        eligibility[name] >= 0,
    ) &&
    validCounts(eligibility.target_class_counts) &&
    validCounts(eligibility.training_class_counts) &&
    validCounts(eligibility.testing_class_counts) &&
    Array.isArray(eligibility.missing_requirements) &&
    eligibility.missing_requirements.every(
      (item) => typeof item === "string",
    ) &&
    (eligibility.reason === null ||
      typeof eligibility.reason === "string")
  );
}
