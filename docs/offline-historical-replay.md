# Offline Historical Hotel Replay

This command runs an isolated retrospective replay against
`data/raw/hotel_bookings.csv`. It uses the saved Model 1 pipeline and the exact
feature order in `ml/models/feature_names.json`. It does not call production
ingestion endpoints, use `PRODUCTION_INGESTION_TOKEN`, update production
monitoring history, or modify production Model 2 artifacts/status.

## Run

From the project root:

```powershell
python -m ml.offline_historical_replay
```

By default, each invocation creates a unique
`data/offline_replay/runs/<run_id>/` directory. A different dataset or output
root can be supplied explicitly:

```powershell
python -m ml.offline_historical_replay `
  --dataset data/raw/hotel_bookings.csv `
  --output-root data/offline_replay/runs
```

The run contains:

- `model1_predictions.csv` and `historical_outcomes.csv`, linked by an offline
  record ID and tagged `HISTORICAL_OFFLINE`.
- `model1_evaluation_windows.csv` with chronological arrival-week Model 1
  accuracy, precision, recall, F1, class counts, and drift summaries.
- `model2_historical_dataset.csv` with observed monitoring features and a
  future-only degradation target.
- `summary.json` with data provenance, duplicate diagnostics, Model 2
  eligibility/reasons, metrics, artifact hashes, and limitations.
- Model 2 model/metadata/evaluation files only when its offline eligibility
  gates pass.

  ## Dashboard

  The overview reads the latest validated `summary.json` from the offline replay
  run directory and displays its Model 1 metrics, evaluation-window count, and
  Model 2 eligibility reason in a separate **HISTORICAL OFFLINE replay** panel.
  The production outcome trend and production Model 2 panels remain reserved for
  verified production data; offline replay values never populate them.

## Interpretation and limitations

The source has no booking-created timestamp. Windows therefore use arrival
dates as a retrospective ordering proxy; they are not booking-time windows.
The lifecycle availability of features such as `assigned_room_type`,
`booking_changes`, `days_in_waiting_list`, `adr`, and special-request fields is
not established. Do not interpret these results as booking-time performance.

The saved Model 1 artifact was trained using a random split spanning the
historical data period. Replaying that fixed artifact chronologically does not
produce independent out-of-time Model 1 metrics. The report marks the metrics
exploratory. A leakage-free temporal Model 1 backtest would require a separate
historical-only fit, which this pipeline intentionally does not perform.

The input contains exact repeated rows and repeated Model 1 feature groups
with conflicting cancellation labels. Without a booking ID, exact repeated
rows cannot be proven to be duplicate bookings, so the replay retains them,
reports them, and keeps identical feature groups within one arrival-week
window. These repeated rows may affect metric weighting.

Model 2 labels use measured current-window and later-window F1 values with the
threshold and future window in `ml/models/model2/target_definition.json`.
Future target evidence is not included in Model 2 features. Training uses a
chronological holdout and purges training rows whose target windows overlap
the holdout. At least 20 labeled historical windows and both target classes in
the eligible data, training partition, and holdout are required. If any gate
fails, the report records `NOT_ELIGIBLE` with the exact reasons and does not
write a Model 2 model.

All results are historical offline evidence, not genuine production
observations and not synthetic development data.
