# Known Limitations

**Reviewed:** 20 November 2026

This file describes observed boundaries, not promises of future production
behavior.

## Data and ML evidence

- The inspected monitoring history contains 15 simulated reports and one
  report with unknown provenance; none is explicitly verified as real
  production traffic.
- Production ingestion is not configured and no trusted producer has been
  connected.
- Production Model 2 requires at least 20 eligible, genuinely labeled
  production observations and currently has **0**. It correctly remains
  `NOT_TRAINED`; there are no production risk scores or evaluation metrics.
- The separate `HISTORICAL_OFFLINE` Model 2 is technically trained from 41
  valid labels (24 stable / 17 degraded), but its nine-observation holdout
  scored 0.333 accuracy, 0 precision, 0 recall, 0 F1, and 0.400 ROC-AUC. It is
  not validated as a useful predictor and must not be presented as production.
- Exploratory candidate comparison used only 18 expanding-time out-of-fold
  predictions. That evidence is insufficient to promote a different model or
  claim improvement; no broad hyperparameter search was justified.
- Offline observation chronology comes from Kaggle-derived booking and
  reservation-status dates, not verified production receipt timestamps.
- Model 1's reported accuracy/F1 are from its historical held-out dataset.
  They are not current production or live monitoring metrics.
- Distribution drift and quality findings are not proof of model performance
  degradation. Genuine outcomes are needed for operational performance
  evaluation.
- The dataset and Model 1 binary artifacts are local assets. A clean
  source-only checkout may not include them. Their licensing and packaging
  permissions must be checked before redistribution.

## Application and infrastructure

- Authentication is local single-user development authentication, not OIDC
  or a production multi-user authorization system.
- The local HTTP cookie is HttpOnly and SameSite Lax but is not marked Secure
  in the local loopback configuration.
- No production TLS termination, rate limiting, secret manager, deployment
  hardening, or production authorization configuration was verified.
- Monitoring state is stored in local JSON/JSONL files. Transactional
  multi-worker writes, retention, automated backup, and recovery are not
  implemented as production services.
- CORS is allowlisted in local configuration; CORS is not an authentication
  control.
- Production ingestion routes have tests but no real producer integration
  test because no external producer/token is configured.

## Frontend and packaging

- The chart library is statically imported, although the build emits a
  separate chart chunk. It is not lazy-loaded, so total initial JavaScript
  download is not reduced by the split.
- A prior lazy-loading experiment caused a React maximum-update-depth runtime
  error and was reverted. Do not reintroduce it without an end-to-end
  production-build browser test.
- The offline dashboard section is shown separately from the production
  training panel. Browser verification covers local startup/login and the
  Model 2 summary, not a full browser/device/accessibility matrix.
- The document minimum width is 240 CSS pixels. Smaller effective widths are
  not supported or verified.
- No application screenshots were found in the inspected working copy.
- The workspace has no Git metadata. `.gitignore` patterns were inspected,
  but whether local assets or other files are tracked cannot be confirmed.

## Submission and demo precautions

- Do not submit a populated `.env`, passwords, signing secrets, or ingestion
  tokens.
- Clearly label simulated and unknown-provenance records in any screenshots
  or presentation.
- Do not present held-out results as live performance or describe the app as
  production-ready.
- Do not describe production Model 2 as trained; state its `NOT_TRAINED`
  status and 0/20 genuine-label count. If presenting the offline artifact,
  label it `HISTORICAL_OFFLINE` and disclose its weak holdout results.
