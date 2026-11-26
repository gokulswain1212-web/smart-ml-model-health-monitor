from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MonitoringThresholds:
    psi_warning: float = 0.10
    psi_critical: float = 0.20
    ks_significance_level: float = 0.05
    ks_min_effect_size: float = 0.10
    critical_drifted_features: int = 3
    missing_rate_warning_delta: float = 0.01
    missing_rate_critical_delta: float = 0.05
    duplicate_warning_rate: float = 0.01
    duplicate_critical_rate: float = 0.10
    psi_bins: int = 10
    psi_epsilon: float = 1e-6
    categorical_min_expected_count: int = 20


DEFAULT_THRESHOLDS = MonitoringThresholds()

NON_NEGATIVE_FEATURES = frozenset(
    {
        "lead_time",
        "arrival_date_year",
        "arrival_date_week_number",
        "arrival_date_day_of_month",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "adults",
        "children",
        "babies",
        "is_repeated_guest",
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "booking_changes",
        "agent",
        "days_in_waiting_list",
        "required_car_parking_spaces",
        "total_of_special_requests",
    }
)
