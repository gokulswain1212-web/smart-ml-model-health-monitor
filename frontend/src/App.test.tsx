import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  DevelopmentSimulationPanel,
  HistoricalReplayDashboardPanel,
  OfflineModel2DashboardPanel,
  Model2DevelopmentRiskAssessment,
  MonitoringBatchRiskAssessment,
} from "./App";
import {
  isDevelopmentSimulationSummary,
  isHistoricalReplaySummary,
  type HistoricalReplaySummary,
} from "./dashboardValidation";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Model 2 development prediction controls", () => {
  it("disables manual development predictions when the artifact is unavailable", () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    render(
      <Model2DevelopmentRiskAssessment
        featureNames={["dataset_drift_score"]}
        available={false}
      />,
    );

    expect(screen.getByText("The DEVELOPMENT model is unavailable. Manual predictions are disabled.")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Run development assessment" })).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("does not load or assess batches when the development artifact is unavailable", () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    render(<MonitoringBatchRiskAssessment refreshKey={1} available={false} />);

    expect(screen.getByText("The DEVELOPMENT model is unavailable. Batch predictions are disabled.")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Assess selected batch" })).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("requests only SIMULATED batches for development assessment", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ reports: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    render(<MonitoringBatchRiskAssessment refreshKey={1} available />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());
    expect(fetchMock.mock.calls[0][0]).toContain("source=SIMULATED");
  });
});

describe("Historical offline Model 2 dashboard", () => {
  it("labels Kaggle metrics and predictions as offline and shows the required notice", () => {
    const model: Parameters<typeof OfflineModel2DashboardPanel>[0]["model"] = {
      status: "TRAINED",
      model_scope: "HISTORICAL_OFFLINE",
      provenance: "HISTORICAL_OFFLINE",
      model1_replay_version: "historical-refit-test",
      historical_observation_count: 45,
      eligible_observation_count: 41,
      target_class_counts: { "0": 24, "1": 17 },
      training_size: 31,
      testing_size: 9,
      evaluation_metrics: {
        accuracy: 1 / 3,
        precision: 0,
        recall: 0,
        f1_score: 0,
        roc_auc: 0.4,
        confusion_matrix: [[3, 1], [5, 0]],
        test_class_counts: { "0": 4, "1": 5 },
      },
      confusion_matrix: [[3, 1], [5, 0]],
      evaluation_predictions: [
        {
          batch_id: "historical-offline-outcomes-20170702",
          timestamp: "2017-07-02T23:59:59",
          actual_degradation_risk: 1,
          predicted_degradation_risk: 0,
          degradation_probability: 0.23,
        },
      ],
      observation_history: [
        {
          batch_id: "historical-offline-outcomes-20170702",
          timestamp: "2017-07-02T23:59:59",
          sample_count: 700,
          prediction_count: 700,
          actual_outcome_count: 700,
          psi_score: 0.24,
          ks_statistic: 0.11,
          f1_score: 0.62,
          accuracy: 0.8,
          precision: 0.6,
          recall: 0.65,
          performance_eligible: true,
          target: 1,
        },
      ],
      missing_requirements: [],
      reason: null,
      model_version: "offline-model2-test",
      training_timestamp: "2026-11-19T11:08:38.226838+00:00",
      target_definition: {
        rule: "Historical F1 drop rule.",
        f1_absolute_drop_threshold: 0.05,
        future_window_days: 30,
        target_source: "HISTORICAL_OFFLINE",
      },
      limitations: ["Not validated on production observations."],
    };

    render(<OfflineModel2DashboardPanel model={model} />);

    expect(screen.getByText("HISTORICAL OFFLINE Model 2")).toBeTruthy();
    expect(
      screen.getByText(
        "Historical offline model: trained and evaluated using Kaggle booking data and Model 1 offline evaluation results. Not validated using live production observations.",
      ),
    ).toBeTruthy();
    expect(screen.getByText("Model 1 offline replay version")).toBeTruthy();
    expect(screen.getByText("historical-refit-test")).toBeTruthy();
    expect(screen.getByText("Historical PSI trend")).toBeTruthy();
    expect(screen.getByText("Historical KS trend")).toBeTruthy();
    expect(screen.getByText("Model 1 F1 trend from actual Kaggle labels")).toBeTruthy();
    expect(screen.getByText("historical-offline-outcomes-20170702")).toBeTruthy();
  });
});

describe("Synthetic hotel development simulation dashboard", () => {
  it("clearly separates synthetic simulation metrics from production eligibility", () => {
    const simulation: Parameters<
      typeof DevelopmentSimulationPanel
    >[0]["simulation"] = {
      data_source_type: "synthetic",
      source: "synthetic_simulator",
      environment: "development",
      is_synthetic: true,
      mode: "DEVELOPMENT SIMULATION",
      status: "AVAILABLE",
      run_id: "simulation-test",
      batch_count: 24,
      booking_count: 3840,
      seed: 42,
      baseline_f1_score: 0.7692307692307693,
      minimum_batch_f1_score: 0.38333333333333336,
      drift_detected_batch_count: 24,
      monitoring_status_counts: { HEALTHY: 0, WARNING: 0, CRITICAL: 24 },
      model2_training: {
        status: "DEVELOPMENT_SIMULATION_TRAINED",
        model_version: "model2-development-simulation-test",
        training_size: 18,
        testing_size: 6,
        split_method: "chronological holdout",
        training_class_counts: { "0": 11, "1": 7 },
        testing_class_counts: { "0": 3, "1": 3 },
        evaluation_metrics: {
          accuracy: 0.8333333333333334,
          precision: 0.75,
          recall: 1.0,
          f1_score: 0.8571428571428571,
        },
      },
      reason: null,
    };

    expect(isDevelopmentSimulationSummary(simulation)).toBe(true);
    expect(
      isDevelopmentSimulationSummary({
        ...simulation,
        model2_training: undefined,
      }),
    ).toBe(false);
    render(<DevelopmentSimulationPanel simulation={simulation} />);

    expect(screen.getByText("DEVELOPMENT SIMULATION · synthetic only")).toBeTruthy();
    expect(
      screen.getByText("DEVELOPMENT SIMULATION · NOT REAL HOTEL DATA"),
    ).toBeTruthy();
    expect(screen.getByRole("status").textContent).toContain("synthetic_simulator");
    expect(screen.getByRole("status").textContent).toContain("run simulation-test");
    expect(screen.getByText("Synthetic batches / bookings")).toBeTruthy();
    expect(screen.getByText("24 / 3,840")).toBeTruthy();
    expect(screen.getByText("18 / 6")).toBeTruthy();
    expect(screen.getByText("83.3% / 85.7%")).toBeTruthy();
    expect(screen.getByText(/This run is tagged/).textContent).toContain(
      "cannot satisfy production Model 2 eligibility",
    );
  });

  it("shows a clear load error instead of an endless loading state", () => {
    render(
      <DevelopmentSimulationPanel
        simulation={undefined}
        error="Dashboard API returned HTTP 503"
      />,
    );

    expect(
      screen.getByText("Development simulation data could not be loaded"),
    ).toBeTruthy();
    expect(screen.getByRole("alert").textContent).toContain("HTTP 503");
    expect(
      screen.queryByText("Loading development simulation status…"),
    ).toBeNull();
  });
});

describe("Historical offline replay dashboard", () => {
  it("shows saved replay metrics and the ineligibility reason without implying production", () => {
    const replay: HistoricalReplaySummary = {
      status: "AVAILABLE",
      model_scope: "HISTORICAL_OFFLINE",
      provenance: "HISTORICAL_OFFLINE",
      data_source_type: "historical_offline",
      source: "historical_hotel_booking_dataset",
      environment: "offline",
      is_synthetic: false,
      run_id: "replay-test",
      historical_record_count: 119390,
      arrival_window_count: 114,
      arrival_date_range: ["2015-07-01", "2017-08-31"],
      model1_metrics: {
        accuracy: 0.974051428,
        precision: 0.974326444,
        recall: 0.95511487,
        f1_score: 0.964625011,
      },
      model2_training_status: "NOT_ELIGIBLE",
      model2_eligibility: {
        status: "NOT_ELIGIBLE",
        eligible_observation_count: 109,
        target_class_counts: { "0.0": 106, "1.0": 3 },
        training_size_after_purge: 86,
        testing_size: 22,
        training_class_counts: { "0": 83, "1": 3 },
        testing_class_counts: { "0": 22 },
        missing_requirements: [
          "Chronological Model 2 holdout windows lack both target classes.",
        ],
        reason: "Chronological Model 2 holdout windows lack both target classes.",
      },
      prediction_scope: "retrospective_historical_feature_replay_not_booking_time",
      booking_created_timestamp_available: false,
      limitations: [
        "The dataset has no recorded booking-created timestamp.",
        "Historical offline evidence cannot satisfy production Model 2 eligibility.",
      ],
      reason: null,
    };

    expect(isHistoricalReplaySummary(replay)).toBe(true);
    expect(
      isHistoricalReplaySummary({
        ...replay,
        data_source_type: "real",
      }),
    ).toBe(false);
    render(<HistoricalReplayDashboardPanel replay={replay} />);

    expect(screen.getByText("HISTORICAL OFFLINE · NOT PRODUCTION")).toBeTruthy();
    expect(screen.getByText(/1,?19,?390/)).toBeTruthy();
    expect(screen.getByText("97.41%")).toBeTruthy();
    expect(screen.getByText("96.46%")).toBeTruthy();
    expect(screen.getByText("Chronological Model 2 holdout windows lack both target classes.")).toBeTruthy();
    expect(screen.getByText("Historical offline evidence cannot satisfy production Model 2 eligibility.")).toBeTruthy();
  });
});
