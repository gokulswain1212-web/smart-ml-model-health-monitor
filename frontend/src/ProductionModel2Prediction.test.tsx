import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ProductionModel2Prediction,
} from "./ProductionModel2Prediction";

type ApiFetch = (
  url: string,
  init?: RequestInit,
) => Promise<Response>;

const prediction = {
  batch_id: "genuine-batch-1",
  prediction_id: "prediction-1",
  prediction_timestamp: "2026-04-04T10:01:00Z",
  source: "production",
  data_source_type: "real",
  model2_status: "TRAINED",
  predicted_degradation_risk: 1,
  degradation_probability: 0.82,
  risk_category: "HIGH",
  health_status: "CRITICAL",
  important_contributing_signals: [
    { feature: "historical_f1_score", importance: 0.4, observed_value: 0.71 },
  ],
  explanation: "Production model explanation.",
  model_version: "model2-random-forest-v1",
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function renderPrediction(
  apiFetch: ApiFetch,
  modelStatus = "TRAINED",
) {
  return render(
    <ProductionModel2Prediction
      apiBaseUrl="http://localhost:8000"
      apiFetch={apiFetch}
      modelStatus={modelStatus}
      modelReason="Collect genuine labeled batches first."
      eligibleObservationCount={10}
      minimumObservationCount={20}
    />,
  );
}

afterEach(cleanup);

describe("ProductionModel2Prediction", () => {
  it("does not fetch or offer predictions while Model 2 is not trained", () => {
    const apiFetch = vi.fn<ApiFetch>();
    renderPrediction(apiFetch, "NOT_TRAINED");

    expect(screen.getByText("Production prediction unavailable")).toBeTruthy();
    expect(screen.getByText(/10\/20 genuine labeled production observations/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Predict production risk" })).toBeNull();
    expect(apiFetch).not.toHaveBeenCalled();
  });

  it("keeps production predictions disabled while eligible but not yet trained", () => {
    const apiFetch = vi.fn<ApiFetch>();
    renderPrediction(apiFetch, "READY");

    expect(screen.getByText("Production prediction unavailable")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Predict production risk" })).toBeNull();
    expect(apiFetch).not.toHaveBeenCalled();
  });

  it("predicts from a distinct real-source batch and displays validated output", async () => {
    const apiFetch: ApiFetch = vi.fn(async (url) => {
      if (url.includes("/api/v1/monitoring/observations?")) {
        return jsonResponse({
          observations: [
            {
              batch_id: "genuine-batch-1",
              timestamp: "2026-04-04T10:00:00Z",
              source: "production",
            },
            {
              batch_id: "synthetic-batch",
              timestamp: "2026-04-04T09:00:00Z",
              source: "synthetic_test",
            },
            {
              batch_id: "genuine-batch-1",
              timestamp: "2026-04-03T10:00:00Z",
              source: "production",
            },
          ],
        });
      }
      return jsonResponse(prediction);
    });

    renderPrediction(apiFetch);
    const select = screen.getByLabelText(
      "PRODUCTION genuine monitoring batch",
    ) as HTMLSelectElement;
    await waitFor(() => expect(select.options).toHaveLength(2));
    expect(screen.queryByRole("option", { name: /synthetic-batch/ })).toBeNull();

    fireEvent.change(select, { target: { value: "genuine-batch-1" } });
    fireEvent.click(screen.getByRole("button", { name: "Predict production risk" }));

    expect(await screen.findByText("CRITICAL")).toBeTruthy();
    expect(screen.getByText("82.0%")).toBeTruthy();
    expect(screen.getByText("PRODUCTION · HIGH risk · genuine-batch-1")).toBeTruthy();
    expect(
      screen.getByText(/saved in the PRODUCTION Model 2 prediction history/i),
    ).toBeTruthy();
    expect(
      screen.getByText("Prediction timestamp").parentElement?.textContent,
    ).toMatch(/4 Apr 2026/);

    const requests = vi.mocked(apiFetch).mock.calls;
    expect(requests[0][0]).toContain("/api/v1/monitoring/observations?");
    expect(requests[1][0]).toContain(
      "/api/v1/predictions/performance-degradation",
    );
    expect(requests[1][1]).toMatchObject({
      method: "POST",
      body: JSON.stringify({ batch_id: "genuine-batch-1" }),
    });
  });

  it("shows API errors instead of a success-shaped prediction", async () => {
    const apiFetch: ApiFetch = vi.fn(async (url) =>
      url.includes("/api/v1/monitoring/observations?")
        ? jsonResponse({
            observations: [
              {
                batch_id: "genuine-batch-1",
                timestamp: "2026-04-04T10:00:00Z",
                source: "production",
              },
            ],
          })
        : jsonResponse({ detail: "Model 2 artifact is unavailable." }, 503),
    );

    renderPrediction(apiFetch);
    const select = screen.getByLabelText(
      "PRODUCTION genuine monitoring batch",
    ) as HTMLSelectElement;
    await waitFor(() => expect(select.options).toHaveLength(2));
    fireEvent.change(select, { target: { value: "genuine-batch-1" } });
    fireEvent.click(screen.getByRole("button", { name: "Predict production risk" }));

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toBe("Model 2 artifact is unavailable.");
    expect(screen.queryByText("Degradation risk detected")).toBeNull();
  });
});
