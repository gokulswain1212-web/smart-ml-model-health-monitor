from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from ml.model2_config import MODEL2_FEATURE_NAMES
from ml.model2_development_training import DEVELOPMENT_DIR, DEVELOPMENT_LABEL

DEVELOPMENT_REPORTS_DIR = DEVELOPMENT_DIR / "reports"
FEATURE_IMPORTANCE_PATH = DEVELOPMENT_DIR / "feature_importance.json"
FEATURE_IMPORTANCE_CHART_PATH = (
    DEVELOPMENT_REPORTS_DIR / "feature_importance_top10.png"
)
FEATURE_EXPLANATIONS: dict[str, dict[str, str]] = {
    "dataset_drift_score": {
        "measures": "Aggregate distribution drift between reference and current input data.",
        "model_rationale": "A broad shift in input distributions may signal that the model is operating on less familiar data.",
        "risk_relation": "Synthetic labels were more likely as this score increased; this is a learned association in the development fixture.",
    },
    "max_feature_drift_score": {
        "measures": "The largest drift score among monitored features.",
        "model_rationale": "A single feature can shift sharply even when aggregate drift remains moderate.",
        "risk_relation": "A larger value can flag a concentrated input shift; importance does not establish a production relationship.",
    },
    "number_drifted_features": {
        "measures": "Count of features exceeding configured drift thresholds.",
        "model_rationale": "The number of affected features can provide context about the breadth of a distribution shift.",
        "risk_relation": "The synthetic target may associate broad drift with risk, but that relationship is not validated on production outcomes.",
    },
    "missing_value_percentage": {
        "measures": "Percentage of monitored feature values that are missing.",
        "model_rationale": "Missingness can affect the inputs and preprocessing behavior seen by Model 1.",
        "risk_relation": "The synthetic label rule made higher missingness a risk signal; this is a designed association, not causation.",
    },
    "invalid_value_percentage": {
        "measures": "Percentage of monitored feature cells with invalid, infinite, negative, or unexpected values.",
        "model_rationale": "Invalid values can indicate changes in upstream data generation or quality.",
        "risk_relation": "The synthetic label rule made higher invalid-value rates a risk signal; production evidence is still required.",
    },
    "data_quality_score": {
        "measures": "A bounded score derived from missing and invalid value percentages.",
        "model_rationale": "It summarizes data-quality signals in a single feature.",
        "risk_relation": "Its importance may overlap with missingness and invalid-value features and should not be read as an independent effect.",
    },
    "duplicate_rate": {
        "measures": "Fraction of duplicate records in the monitored batch.",
        "model_rationale": "Duplicate records can alter the effective input population and batch summaries.",
        "risk_relation": "Any relationship shown here is specific to the synthetic development data.",
    },
    "prediction_positive_rate": {
        "measures": "Fraction of Model 1 predictions assigned to the positive class.",
        "model_rationale": "A changed prediction mix can reflect changes in inputs or model behavior.",
        "risk_relation": "It is a monitoring signal, not proof that the model's true performance has degraded.",
    },
    "prediction_positive_rate_change": {
        "measures": "Change in positive prediction rate relative to the preceding batch.",
        "model_rationale": "A sudden change in prediction mix can highlight behavior changes worth investigating.",
        "risk_relation": "This feature's importance represents predictive association in this trained synthetic model only.",
    },
    "prediction_mean_probability": {
        "measures": "Mean predicted probability across records in a batch.",
        "model_rationale": "A shift in confidence distribution can accompany changes in inputs or predictions.",
        "risk_relation": "It does not measure calibration or correctness by itself.",
    },
    "historical_accuracy": {
        "measures": "Latest genuine accuracy available before the monitoring observation.",
        "model_rationale": "Past observed performance can provide context for subsequent performance risk.",
        "risk_relation": "Historical association does not establish that accuracy changes cause future degradation.",
    },
    "historical_precision": {
        "measures": "Latest genuine positive-class precision available before the observation.",
        "model_rationale": "Precision captures one aspect of recent model behavior.",
        "risk_relation": "The development importance is not a production performance estimate.",
    },
    "historical_recall": {
        "measures": "Latest genuine positive-class recall available before the observation.",
        "model_rationale": "Recall captures the model's recent ability to identify positive cases.",
        "risk_relation": "Its use by the synthetic development model is not causal evidence.",
    },
    "historical_f1_score": {
        "measures": "Latest genuine F1 score available before the monitoring observation.",
        "model_rationale": "F1 summarizes recent precision and recall and was directly used in the synthetic target-generation process.",
        "risk_relation": "Its prominence may reflect the synthetic label rule and must not be interpreted as measured production behavior.",
    },
    "previous_f1_trend": {
        "measures": "Recent trend in genuine F1 values known as of the observation.",
        "model_rationale": "A declining historical trend can provide an early warning signal.",
        "risk_relation": "The synthetic target rule explicitly used negative trend, so its importance partly reflects the constructed experiment.",
    },
    "historical_metrics_available": {
        "measures": "Indicator that usable historical performance measurements are available.",
        "model_rationale": "Availability can distinguish batches with performance context from those without it.",
        "risk_relation": "This indicator can capture missingness patterns and does not imply degradation.",
    },
    "historical_performance_observations": {
        "measures": "Number of prior performance observations available to monitoring.",
        "model_rationale": "The count can represent how much historical evidence is available.",
        "risk_relation": "The synthetic data's relationship is illustrative, not evidence about live operations.",
    },
}


class DevelopmentExplainabilityError(ValueError):
    """A development model cannot produce a trustworthy importance report."""


def _read_development_model(
    development_dir: Path,
) -> tuple[Path, dict[str, Any], Pipeline]:
    candidates = sorted(
        development_dir.glob("model2-development-random-forest-*.json"),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No Model 2 synthetic development metadata found under {development_dir}."
        )

    metadata_path = candidates[0]
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DevelopmentExplainabilityError(
            f"Unable to read development model metadata {metadata_path.name}: {error}"
        ) from error
    if not isinstance(metadata, dict):
        raise DevelopmentExplainabilityError(
            f"Development metadata {metadata_path.name} must be a JSON object."
        )
    if metadata.get("data_provenance") != DEVELOPMENT_LABEL:
        raise DevelopmentExplainabilityError(
            "Development model metadata is missing the synthetic-only provenance."
        )
    if metadata.get("model2_status") != "DEVELOPMENT_TRAINED":
        raise DevelopmentExplainabilityError(
            "Development metadata does not identify a trained development model."
        )
    if metadata.get("feature_names") != MODEL2_FEATURE_NAMES:
        raise DevelopmentExplainabilityError(
            "Development model feature names do not match the Model 2 feature order."
        )

    model_version = metadata.get("model_version")
    if not isinstance(model_version, str) or model_version != metadata_path.stem:
        raise DevelopmentExplainabilityError(
            "Development metadata model version does not match its filename."
        )
    model_path = development_dir / f"{model_version}.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Development model artifact is missing: {model_path}.")
    try:
        model = joblib.load(model_path)
    except (
        AttributeError,
        EOFError,
        ImportError,
        KeyError,
        OSError,
        pickle.UnpicklingError,
        RuntimeError,
        ValueError,
        TypeError,
    ) as error:
        raise DevelopmentExplainabilityError(
            f"Unable to load development model artifact {model_path.name}: {error}"
        ) from error
    if not isinstance(model, Pipeline) or "classifier" not in model.named_steps:
        raise DevelopmentExplainabilityError(
            "Development artifact is not the expected Random Forest pipeline."
        )
    return model_path, metadata, model


def extract_development_feature_importance(
    *,
    development_dir: Path = DEVELOPMENT_DIR,
    top_n: int = 10,
) -> dict[str, Any]:
    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    model_path, metadata, model = _read_development_model(development_dir)
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps.get("imputer")
    fitted_feature_names = getattr(
        preprocessor,
        "feature_names_in_",
        getattr(classifier, "feature_names_in_", None),
    )
    if fitted_feature_names is None or list(fitted_feature_names) != MODEL2_FEATURE_NAMES:
        raise DevelopmentExplainabilityError(
            "Fitted estimator feature order does not match the Model 2 schema."
        )
    raw_importances = getattr(classifier, "feature_importances_", None)
    if raw_importances is None:
        raise DevelopmentExplainabilityError(
            "Development Random Forest does not expose feature_importances_."
        )
    try:
        importances = np.asarray(raw_importances, dtype=float)
    except (TypeError, ValueError) as error:
        raise DevelopmentExplainabilityError(
            "Development feature importances must be numeric."
        ) from error
    if importances.ndim != 1 or len(importances) != len(MODEL2_FEATURE_NAMES):
        raise DevelopmentExplainabilityError(
            "Development feature importance length does not match Model 2 features."
        )
    if not np.isfinite(importances).all() or (importances < 0).any():
        raise DevelopmentExplainabilityError(
            "Development feature importances must be finite and non-negative."
        )
    if not np.isclose(float(importances.sum()), 1.0, rtol=1e-5, atol=1e-8):
        raise DevelopmentExplainabilityError(
            "Development feature importances must sum approximately to 1."
        )

    ranked = sorted(
        zip(MODEL2_FEATURE_NAMES, importances.tolist(), strict=True),
        key=lambda item: (-item[1], item[0]),
    )
    feature_rows = [
        {
            "feature": feature_name,
            "importance": float(importance),
            "rank": rank,
            **FEATURE_EXPLANATIONS[feature_name],
        }
        for rank, (feature_name, importance) in enumerate(ranked, start=1)
    ]
    return {
        "provenance": DEVELOPMENT_LABEL,
        "model_version": metadata["model_version"],
        "feature_names": MODEL2_FEATURE_NAMES.copy(),
        "importance_sum": float(importances.sum()),
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "importance_method": "Random Forest mean decrease in impurity",
        "causality_warning": (
            "Feature importance describes model reliance in this synthetic "
            "development experiment. It does not show that a feature causes "
            "performance degradation."
        ),
        "features": feature_rows,
        "top_features": feature_rows[:top_n],
        "model_path": str(model_path),
    }


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _save_chart(
    importance_report: dict[str, Any],
    chart_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(reversed(importance_report["top_features"]))
    labels = [row["feature"].replace("_", " ") for row in rows]
    values = [row["importance"] for row in rows]
    figure, axis = plt.subplots(figsize=(11, 6.5), facecolor="#0b1220")
    axis.set_facecolor("#0b1220")
    bars = axis.barh(labels, values, color="#22c6a8", edgecolor="#64e3d0", height=0.68)
    axis.set_title(
        "Model 2 Development — Top 10 Feature Importance\n"
        "SYNTHETIC_DEVELOPMENT_ONLY",
        color="#f4f7fb",
        fontsize=14,
        fontweight="bold",
        pad=18,
    )
    axis.set_xlabel("Mean decrease in impurity importance", color="#c7d2e2")
    axis.set_ylabel("Model 2 feature", color="#c7d2e2")
    axis.tick_params(axis="x", colors="#c7d2e2")
    axis.tick_params(axis="y", colors="#e2e8f0", labelsize=9)
    axis.grid(axis="x", color="#243248", linestyle="--", linewidth=0.7)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            value + max(values) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            color="#dce7f5",
            fontsize=9,
        )
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#43536b")
    axis.set_xlim(0, max(values) * 1.18)
    figure.tight_layout()
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(chart_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def _save_explanation_report(
    importance_report: dict[str, Any],
    report_path: Path,
    chart_path: Path,
) -> None:
    lines = [
        "# Model 2 Development Feature Importance",
        "",
        f"- **Provenance:** {importance_report['provenance']}",
        f"- **Model version:** {importance_report['model_version']}",
        f"- **Generated:** {importance_report['generation_timestamp']}",
        f"- **Method:** {importance_report['importance_method']}",
        f"- **Chart:** `{chart_path.name}`",
        "",
        "## Ranked features",
        "",
    ]
    for row in importance_report["top_features"]:
        lines.extend(
            [
                f"### {row['rank']}. `{row['feature']}` — {row['importance']:.6f}",
                "",
                f"- **Measures:** {row['measures']}",
                f"- **Why the model may use it:** {row['model_rationale']}",
                f"- **Relation to risk:** {row['risk_relation']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation limits",
            "",
            importance_report["causality_warning"],
            "Random Forest impurity-based importance measures how much a feature "
            "contributes to reducing split impurity across the fitted trees. It can "
            "favor variables with many possible split points, can distribute "
            "importance among correlated features, and does not show effect direction "
            "or causal impact. All labels and metrics in this report are synthetic "
            "development results and must not be presented as production evidence.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def generate_development_explainability(
    *,
    development_dir: Path = DEVELOPMENT_DIR,
    report_dir: Path | None = None,
) -> dict[str, Any]:
    report_dir = report_dir or development_dir / "reports"
    importance_report = extract_development_feature_importance(
        development_dir=development_dir
    )
    chart_path = report_dir / "feature_importance_top10.png"
    explanation_path = report_dir / "feature_importance_explanations.md"
    importance_report["chart_path"] = str(chart_path)
    importance_report["explanation_report_path"] = str(explanation_path)
    _save_json(development_dir / "feature_importance.json", importance_report)
    _save_chart(importance_report, chart_path)
    _save_explanation_report(importance_report, explanation_path, chart_path)
    return importance_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and visualize development-only Model 2 feature importance."
    )
    parser.add_argument("--development-dir", type=Path, default=DEVELOPMENT_DIR)
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    result = generate_development_explainability(
        development_dir=args.development_dir,
        report_dir=args.report_dir,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
