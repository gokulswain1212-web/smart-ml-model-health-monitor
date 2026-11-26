from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ml.monitoring_config import DEFAULT_THRESHOLDS
from ml.offline_drift_analysis import compare_offline_features

ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES_PATH = ROOT / "ml" / "models" / "feature_names.json"
MODEL_METADATA_PATH = ROOT / "ml" / "models" / "model_metadata.json"
HISTORICAL_PATH = ROOT / "data" / "processed" / "hotel_bookings_historical.csv"
FUTURE_PATH = ROOT / "data" / "processed" / "hotel_bookings_future_evaluation.csv"
REPORT_PATH = ROOT / "reports" / "multiwindow_drift_validation.md"
DATE_COLUMN = "estimated_booking_date"
FORBIDDEN_FEATURES = {
    "is_canceled",
    "reservation_status",
    "reservation_status_date",
    "offline_row_id",
    DATE_COLUMN,
}


@dataclass(frozen=True)
class ComparisonWindow:
    name: str
    start_date: date
    end_date: date


DEFAULT_WINDOWS = (
    ComparisonWindow("2016-Nov-Dec", date(2016, 11, 7), date(2016, 12, 31)),
    ComparisonWindow("2017-Jan-Mar", date(2017, 1, 1), date(2017, 3, 31)),
    ComparisonWindow("2017-Apr-Jun", date(2017, 4, 1), date(2017, 6, 30)),
    ComparisonWindow("2017-Jul-Aug", date(2017, 7, 1), date(2017, 8, 31)),
)


def _validated_dates(frame: pd.DataFrame, frame_name: str, date_column: str) -> pd.Series:
    if date_column not in frame:
        raise ValueError(f"{frame_name} data is missing date column {date_column!r}.")
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{frame_name} data contains missing or invalid dates.")
    if not dates.is_monotonic_increasing:
        raise ValueError(f"{frame_name} dates must be in chronological order.")
    return dates


def _validate_windows(windows: Sequence[ComparisonWindow]) -> None:
    if not windows:
        raise ValueError("At least one comparison window is required.")
    previous_end: date | None = None
    for window in windows:
        if window.start_date > window.end_date:
            raise ValueError(f"Window {window.name!r} starts after it ends.")
        if previous_end is not None and window.start_date <= previous_end:
            raise ValueError("Comparison windows must be chronological and non-overlapping.")
        previous_end = window.end_date


def analyze_multiwindow_drift(
    reference_data: pd.DataFrame,
    future_data: pd.DataFrame,
    feature_names: list[str],
    numerical_features: list[str],
    categorical_features: list[str],
    *,
    windows: Sequence[ComparisonWindow] = DEFAULT_WINDOWS,
    date_column: str = DATE_COLUMN,
) -> dict[str, Any]:
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Feature names must be unique.")
    forbidden = FORBIDDEN_FEATURES.intersection(feature_names)
    if forbidden:
        raise ValueError(f"Target/helper columns cannot be drift features: {sorted(forbidden)}")
    _validate_windows(windows)

    for frame_name, frame in (
        ("historical reference", reference_data),
        ("future comparison", future_data),
    ):
        missing = [column for column in [date_column, *feature_names] if column not in frame]
        if missing:
            raise ValueError(f"{frame_name} data is missing columns: {missing}")
    reference_dates = _validated_dates(reference_data, "Historical reference", date_column)
    future_dates = _validated_dates(future_data, "Future comparison", date_column)

    window_results: list[dict[str, Any]] = []
    for window in windows:
        start = pd.Timestamp(window.start_date)
        end = pd.Timestamp(window.end_date)
        future_mask = future_dates.between(start, end, inclusive="both")
        window_future = future_data.loc[future_mask]
        window_dates = future_dates.loc[future_mask]

        if window_future.empty:
            window_results.append(
                {
                    "name": window.name,
                    "start_date": start.date().isoformat(),
                    "end_date": end.date().isoformat(),
                    "status": "empty",
                    "future_rows": 0,
                    "future_date_count": 0,
                    "reference_rows": len(reference_data),
                    "seasonal_reference_rows": 0,
                    "features": {},
                    "seasonal_features": {},
                }
            )
            continue

        months = set(window_dates.dt.month.unique().tolist())
        seasonal_reference_mask = reference_dates.dt.month.isin(months)
        seasonal_reference = reference_data.loc[seasonal_reference_mask]
        if seasonal_reference.empty:
            raise ValueError(
                f"No historical reference rows match the calendar months for {window.name!r}."
            )

        future_features = window_future.loc[:, feature_names]
        features = compare_offline_features(
            reference_data.loc[:, feature_names],
            future_features,
            feature_names,
            numerical_features,
            categorical_features,
        )
        seasonal_features = compare_offline_features(
            seasonal_reference.loc[:, feature_names],
            future_features,
            feature_names,
            numerical_features,
            categorical_features,
        )
        window_results.append(
            {
                "name": window.name,
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
                "status": "ok",
                "future_rows": len(window_future),
                "future_date_count": int(window_dates.nunique()),
                "reference_rows": len(reference_data),
                "seasonal_reference_rows": len(seasonal_reference),
                "seasonal_reference_date_count": int(
                    reference_dates.loc[seasonal_reference_mask].nunique()
                ),
                "features": features,
                "seasonal_features": seasonal_features,
            }
        )

    return {
        "reference_rows": len(reference_data),
        "reference_date_start": reference_dates.min().date().isoformat(),
        "reference_date_end": reference_dates.max().date().isoformat(),
        "future_rows": len(future_data),
        "future_date_start": future_dates.min().date().isoformat(),
        "future_date_end": future_dates.max().date().isoformat(),
        "feature_names": list(feature_names),
        "windows": window_results,
    }


def _format_metric(result: dict[str, Any] | None) -> str:
    if not result:
        return "—"
    psi = f"{result['psi']:.4f}"
    ks = result.get("ks")
    if not ks or not ks["available"]:
        return f"{psi} / KS —"
    p_value = "<1e-300" if ks["p_value"] == 0 else f"{ks['p_value']:.3g}"
    return f"{psi} / D {ks['statistic']:.4f}, p {p_value}"


def build_multiwindow_report(results: dict[str, Any]) -> str:
    windows = results["windows"]
    feature_names = results["feature_names"]
    usable_windows = [window for window in windows if window["status"] == "ok"]
    lines = [
        "# Offline Multi-Window Drift Validation",
        "",
        "> **Offline historical experiment only. These findings are not genuine "
        "production drift monitoring and do not assign health statuses.**",
        "",
        "## Strategy and coverage",
        "",
        "The historical partition is the reference and the later evaluation "
        "partition is divided into consecutive, non-overlapping windows. Each "
        "window is compared both with the full historical reference and with a "
        "season-matched historical subset containing rows whose estimated "
        "booking-date calendar month(s) match that window. PSI and KS are "
        "calculated by the existing offline implementation without changing "
        "its binning, missingness, unseen-category, or smoothing rules.",
        "",
        "Only the 28 Model 1 features, in `feature_names.json` order, are "
        "analyzed. `is_canceled`, reservation outcome/status fields, row IDs, "
        "and estimated dates are not drift features. Estimated dates are used "
        "only to order and select windows. No preprocessing was fitted and no "
        "model was trained.",
        "",
        f"- Historical reference: **{results['reference_rows']:,}** rows, "
        f"{results['reference_date_start']} to {results['reference_date_end']}",
        f"- Later evaluation data: **{results['future_rows']:,}** rows, "
        f"{results['future_date_start']} to {results['future_date_end']}",
        f"- Features: **{len(feature_names)}**",
        "",
        "| Window | Inclusive dates | Evaluation rows / dates | Full reference rows | Same-month reference rows / dates |",
        "|---|---|---:|---:|---:|",
    ]
    for window in windows:
        seasonal_dates = window.get("seasonal_reference_date_count", 0)
        lines.append(
            f"| {window['name']} ({window['status']}) | "
            f"{window['start_date']}–{window['end_date']} | "
            f"{window['future_rows']:,} / {window['future_date_count']:,} | "
            f"{window['reference_rows']:,} | "
            f"{window['seasonal_reference_rows']:,} / {seasonal_dates:,} |"
        )

    lines.extend(
        [
            "",
            "## PSI and KS by feature and window",
            "",
            "Each cell reports PSI and, for numerical features, KS D and p-value. "
            "Categorical features have no KS result. PSI bands (<0.10, "
            "0.10–<0.20, ≥0.20) are configuration-based descriptive "
            "conventions only, not validated thresholds.",
            "",
            "| Feature | Type | "
            + " | ".join(f"{window['name']} full-reference" for window in windows)
            + " |",
            "|---|---|"
            + "---:|" * len(windows),
        ]
    )
    for feature in feature_names:
        feature_type = next(
            (
                window["features"][feature]["type"]
                for window in usable_windows
                if feature in window["features"]
            ),
            "unavailable",
        )
        cells = [
            _format_metric(window["features"].get(feature))
            for window in windows
        ]
        lines.append(f"| `{feature}` | {feature_type} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "### Season-matched reference results",
            "",
            "These comparisons use the same calendar month numbers in the "
            "historical reference. They are sensitivity checks, not independent "
            "data or a seasonal adjustment. Month coverage and sample "
            "composition in the historical subset are uneven.",
            "",
            "| Feature | Type | "
            + " | ".join(f"{window['name']} same-month reference" for window in windows)
            + " |",
            "|---|---|"
            + "---:|" * len(windows),
        ]
    )
    for feature in feature_names:
        feature_type = next(
            (
                window["seasonal_features"][feature]["type"]
                for window in usable_windows
                if feature in window["seasonal_features"]
            ),
            "unavailable",
        )
        cells = [
            _format_metric(window["seasonal_features"].get(feature))
            for window in windows
        ]
        lines.append(f"| `{feature}` | {feature_type} | " + " | ".join(cells) + " |")

    threshold = DEFAULT_THRESHOLDS.psi_critical
    summary_rows: list[tuple[str, str, int, float, float]] = []
    for feature in feature_names:
        values = [
            window["features"][feature]["psi"]
            for window in usable_windows
            if feature in window["features"]
        ]
        seasonal_values = [
            window["seasonal_features"][feature]["psi"]
            for window in usable_windows
            if feature in window["seasonal_features"]
        ]
        if not values:
            continue
        high_count = sum(value >= threshold for value in values)
        spread = max(values) - min(values)
        summary_rows.append((feature, f"{min(values):.4f}–{max(values):.4f}", high_count, spread, max(seasonal_values) - min(seasonal_values) if seasonal_values else 0.0))

    persistent = [
        row for row in summary_rows
        if row[2] >= max(1, (len(usable_windows) * 3 + 3) // 4)
    ]
    variable = sorted(summary_rows, key=lambda row: row[3], reverse=True)
    notably_variable = [row for row in variable if row[3] >= 0.20]
    lines.extend(
        [
            "",
            "## Persistence and between-window variation",
            "",
            f"For descriptive screening only, “high” means PSI ≥ "
            f"{threshold:.2f} (the current configuration boundary), and "
            "“persistent” means meeting that boundary in at least 75% of "
            "non-empty full-reference windows. A PSI range is max minus min "
            "across windows; it is descriptive, not a significance test. "
            "These rules do not imply production alert levels.",
            "",
            "| Feature | Full-reference PSI range | Windows ≥ 0.20 | Range of same-month PSI |",
            "|---|---:|---:|---:|",
        ]
    )
    for feature, value_range, count, _, seasonal_spread in summary_rows:
        lines.append(
            f"| `{feature}` | {value_range} | {count}/{len(usable_windows)} | "
            f"{seasonal_spread:.4f} |"
        )
    lines.extend(
        [
            "",
            "### Persistent high PSI (descriptive)",
            "",
            (
                ", ".join(f"`{row[0]}` ({row[2]}/{len(usable_windows)} windows)"
                          for row in persistent)
                if persistent
                else "No feature met the descriptive persistence rule."
            ),
            "",
            "### Largest full-reference PSI ranges",
            "",
            "| Feature | PSI range | Range width |",
            "|---|---:|---:|",
        ]
    )
    for row in variable[:10]:
        lines.append(f"| `{row[0]}` | {row[1]} | {row[3]:.4f} |")
    lines.extend(
        [
            "",
            "### Features with notable between-window PSI variation",
            "",
            "A max-minus-min PSI range of at least 0.20 is used here only as a "
            "descriptive screening marker. It is not a p-value or a claim of "
            "statistical significance.",
            "",
            (
                ", ".join(f"`{row[0]}` (range {row[3]:.4f})"
                          for row in notably_variable)
                if notably_variable
                else "No feature met the descriptive range marker."
            ),
        ]
    )

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- PSI and KS describe marginal input differences; neither "
            "establishes model degradation or concept drift.",
            "- The comparison windows are non-overlapping, but they are "
            "compared against the same historical data. Their results are "
            "therefore not independent replicates.",
            "- The available later period spans only November 2016 through "
            "August 2017. September and October later-period windows cannot be "
            "compared, and same-month reference coverage is uneven.",
            "- The split date is estimated from arrival date minus `lead_time`, "
            "not a recorded booking creation timestamp. This can induce "
            "structural relationships with features and limits chronology.",
            "- This historical dataset contains repeated records. KS p-values "
            "assume independent samples, are sensitive to sample size, and "
            "have no multiple-feature adjustment here.",
            "- Results depend on reference-derived numeric bins, category "
            "pooling, missing/unseen buckets, and smoothing inherited from the "
            "existing offline method. Season-matched results reduce some "
            "calendar mismatch but do not control for year, source, hotel, or "
            "other composition changes.",
            "- Historical results do not represent genuine production data; "
            "no HEALTHY/WARNING/CRITICAL classification is made.",
            "",
        ]
    )
    return "\n".join(lines)


def run_multiwindow_analysis() -> Path:
    feature_names = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("feature_names") != feature_names:
        raise ValueError("Model metadata feature order does not match feature_names.json.")
    if FORBIDDEN_FEATURES.intersection(feature_names):
        raise ValueError("Feature schema contains forbidden target or helper columns.")
    usecols = [DATE_COLUMN, *feature_names]
    historical = pd.read_csv(HISTORICAL_PATH, usecols=usecols)
    future = pd.read_csv(FUTURE_PATH, usecols=usecols)
    results = analyze_multiwindow_drift(
        historical,
        future,
        feature_names,
        metadata["numerical_features"],
        metadata["categorical_features"],
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_multiwindow_report(results), encoding="utf-8")
    return REPORT_PATH


if __name__ == "__main__":
    print(run_multiwindow_analysis())
