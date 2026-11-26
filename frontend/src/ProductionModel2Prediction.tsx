import { useEffect, useState } from "react";

type ApiFetch = (url: string, init?: RequestInit) => Promise<Response>;

type ProductionBatch = {
  batch_id: string;
  timestamp: string;
};

type ProductionModel2PredictionResult = {
  batch_id: string;
  prediction_id: string;
  prediction_timestamp: string;
  source: "production";
  data_source_type: "real";
  model2_status: "TRAINED";
  predicted_degradation_risk: 0 | 1;
  degradation_probability: number;
  risk_category: "LOW" | "MEDIUM" | "HIGH";
  health_status: "HEALTHY" | "WARNING" | "CRITICAL";
  important_contributing_signals: {
    feature: string;
    importance: number;
    observed_value: number | null;
  }[];
  explanation: string;
  model_version: string;
};

type ProductionModel2PredictionProps = {
  apiBaseUrl: string;
  apiFetch: ApiFetch;
  modelStatus: string;
  modelReason: string | null;
  eligibleObservationCount: number;
  minimumObservationCount: number;
};

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isPredictionResult(
  value: unknown,
): value is ProductionModel2PredictionResult {
  if (!isObject(value) || !Array.isArray(value.important_contributing_signals)) {
    return false;
  }
  return (
    typeof value.batch_id === "string" &&
    typeof value.prediction_id === "string" &&
    typeof value.prediction_timestamp === "string" &&
    Number.isFinite(Date.parse(value.prediction_timestamp)) &&
    value.source === "production" &&
    value.data_source_type === "real" &&
    value.model2_status === "TRAINED" &&
    (value.predicted_degradation_risk === 0 ||
      value.predicted_degradation_risk === 1) &&
    typeof value.degradation_probability === "number" &&
    Number.isFinite(value.degradation_probability) &&
    value.degradation_probability >= 0 &&
    value.degradation_probability <= 1 &&
    ["LOW", "MEDIUM", "HIGH"].includes(String(value.risk_category)) &&
    ["HEALTHY", "WARNING", "CRITICAL"].includes(String(value.health_status)) &&
    typeof value.explanation === "string" &&
    typeof value.model_version === "string" &&
    value.important_contributing_signals.every(
      (signal) =>
        isObject(signal) &&
        typeof signal.feature === "string" &&
        typeof signal.importance === "number" &&
        Number.isFinite(signal.importance) &&
        (signal.observed_value === null ||
          (typeof signal.observed_value === "number" &&
            Number.isFinite(signal.observed_value))),
    )
  );
}

function isProductionBatch(value: unknown): value is ProductionBatch {
  return (
    isObject(value) &&
    typeof value.batch_id === "string" &&
    typeof value.timestamp === "string" &&
    value.source === "production"
  );
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function detailMessage(value: unknown, fallback: string): string {
  return isObject(value) && typeof value.detail === "string"
    ? value.detail
    : fallback;
}

function formatTimestamp(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : new Intl.DateTimeFormat(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      }).format(date);
}

export function ProductionModel2Prediction({
  apiBaseUrl,
  apiFetch,
  modelStatus,
  modelReason,
  eligibleObservationCount,
  minimumObservationCount,
}: ProductionModel2PredictionProps) {
  const [batches, setBatches] = useState<ProductionBatch[]>([]);
  const [selectedBatchId, setSelectedBatchId] = useState("");
  const [loadingBatches, setLoadingBatches] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [prediction, setPrediction] =
    useState<ProductionModel2PredictionResult | null>(null);
  const isTrained = modelStatus === "TRAINED";

  useEffect(() => {
    const controller = new AbortController();
    let active = true;

    if (!isTrained) {
      return () => controller.abort();
    }

    const params = new URLSearchParams({
      period_days: "3650",
      limit: "1000",
    });
    const baseUrl = apiBaseUrl.replace(/\/+$/, "");
    apiFetch(`${baseUrl}/api/v1/monitoring/observations?${params}`, {
      signal: controller.signal,
    })
      .then(async (response) => {
        const result: unknown = await readJson(response);
        if (!response.ok) {
          throw new Error(
            detailMessage(
              result,
              `Production batch list failed (HTTP ${response.status}).`,
            ),
          );
        }
        if (!isObject(result) || !Array.isArray(result.observations)) {
          throw new Error("Production observations returned an invalid response.");
        }

        const seenBatchIds = new Set<string>();
        const productionBatches: ProductionBatch[] = [];
        for (const observation of result.observations) {
          if (
            !isProductionBatch(observation) ||
            seenBatchIds.has(observation.batch_id)
          ) {
            continue;
          }
          seenBatchIds.add(observation.batch_id);
          productionBatches.push({
            batch_id: observation.batch_id,
            timestamp: observation.timestamp,
          });
        }
        if (active) setBatches(productionBatches);
      })
      .catch((requestError: unknown) => {
        if (
          !active ||
          (requestError instanceof DOMException &&
            requestError.name === "AbortError")
        ) {
          return;
        }
        setError(
          requestError instanceof Error
            ? requestError.message
            : "Could not load production monitoring batches.",
        );
      })
      .finally(() => {
        if (active) setLoadingBatches(false);
      });

    return () => {
      active = false;
      controller.abort();
    };
  }, [apiBaseUrl, apiFetch, isTrained]);

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!isTrained || !selectedBatchId) return;

    setError("");
    setPrediction(null);
    setSubmitting(true);
    try {
      const baseUrl = apiBaseUrl.replace(/\/+$/, "");
      const response = await apiFetch(
        `${baseUrl}/api/v1/predictions/performance-degradation`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ batch_id: selectedBatchId }),
        },
      );
      const result: unknown = await readJson(response);
      if (!response.ok) {
        throw new Error(
          detailMessage(
            result,
            `Production prediction failed (HTTP ${response.status}).`,
          ),
        );
      }
      if (!isPredictionResult(result)) {
        throw new Error("Production Model 2 returned an invalid prediction.");
      }
      setPrediction(result);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Could not request a production prediction.",
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div>
      {!isTrained ? (
        <div className="report-status-banner warning" role="status">
          <span>
            <strong>Production prediction unavailable</strong>
            <small>
              {eligibleObservationCount}/{minimumObservationCount} genuine labeled production observations.
              {modelReason ? ` ${modelReason}` : " Add genuine labels to enable training."}
            </small>
          </span>
        </div>
      ) : (
        <>
          <form
            className="workflow-upload-form"
            onSubmit={(event) => void submit(event)}
          >
            <label>
              PRODUCTION genuine monitoring batch
              <select
                value={selectedBatchId}
                onChange={(event) => {
                  setSelectedBatchId(event.target.value);
                  setPrediction(null);
                  setError("");
                }}
                disabled={loadingBatches || batches.length === 0 || submitting}
                required
              >
                <option value="">
                  {loadingBatches
                    ? "Loading production batches…"
                    : batches.length
                      ? "Select a production batch"
                      : "No genuine production batches available"}
                </option>
                {batches.map((batch) => (
                  <option
                    key={batch.batch_id}
                    value={batch.batch_id}
                  >
                    {batch.batch_id} · {formatTimestamp(batch.timestamp)}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="submit"
              className="outline-action"
              disabled={
                loadingBatches ||
                submitting ||
                batches.length === 0 ||
                !selectedBatchId
              }
            >
              {submitting ? "Predicting…" : "Predict production risk"}
            </button>
          </form>
          {loadingBatches && (
            <p className="data-note" role="status">
              Loading genuine production batches…
            </p>
          )}
          {!loadingBatches && !error && batches.length === 0 && (
            <p className="data-note">
              No genuine production batch is available for prediction.
            </p>
          )}
        </>
      )}
      {error && <p className="alert-error" role="alert">{error}</p>}
      {prediction && (
        <section className="detail-list" aria-label="PRODUCTION Model 2 prediction">
          <div className="report-status-banner">
            <span>
              <strong>
                    {prediction.health_status}
              </strong>
                  <small>
                    PRODUCTION · {prediction.risk_category} risk · {prediction.batch_id}
                  </small>
            </span>
          </div>
          <div className="detail-row">
            <span>Degradation probability</span>
            <strong>{(prediction.degradation_probability * 100).toFixed(1)}%</strong>
          </div>
          <div className="detail-row">
            <span>Prediction timestamp</span>
            <strong>{formatTimestamp(prediction.prediction_timestamp)}</strong>
          </div>
          <div className="detail-row">
            <span>Data source</span>
            <strong>PRODUCTION · {prediction.source}</strong>
          </div>
          <div className="detail-row">
            <span>Model version</span>
            <strong>{prediction.model_version}</strong>
          </div>
          <p className="data-note">{prediction.explanation}</p>
          {prediction.important_contributing_signals.length > 0 && (
            <div>
              <strong>Contributing signals</strong>
              <ul>
                {prediction.important_contributing_signals.map((signal) => (
                  <li key={signal.feature}>
                    {signal.feature.replaceAll("_", " ")} · importance{" "}
                    {signal.importance.toFixed(3)} · observed{" "}
                    {signal.observed_value == null
                      ? "unavailable"
                      : signal.observed_value}
                  </li>
                ))}
              </ul>
            </div>
          )}
          <p className="data-note">
            This production prediction is saved in the PRODUCTION Model 2 prediction history.
          </p>
        </section>
      )}
    </div>
  );
}
