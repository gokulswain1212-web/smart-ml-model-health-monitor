# Local Demonstration Guide

This guide demonstrates the existing local prototype. It does not configure
production ingestion or create genuine production evidence.

## 1. Prepare the environment

From the project root in PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
npm --prefix frontend ci
Copy-Item .env.example .env
```

Edit `.env` locally. Set a local login password of at least six characters
and a random signing secret of at least 32 characters. Keep the ingestion
token blank for a demo by removing the non-working placeholder copied from
`.env.example`. Set a generated token only when a trusted producer integration
is explicitly being configured. Never paste `.env` values into slides,
screenshots, logs, or source control.

The inspected working copy contains the Model 1 dataset/artifacts, but a clean
source-only checkout may not. Confirm required local assets and their license
before the demo; do not download or redistribute restricted assets without
permission.

## 2. Start the application

In the first PowerShell terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --env-file .env --reload --host 127.0.0.1 --port 8000
```

In a second terminal, from the project root:

```powershell
npm --prefix frontend run dev
```

Open the Vite URL (normally `http://localhost:5173`) and sign in using the
local credentials configured in `.env`. Keep the backend bound to loopback for
this local demo.

## 3. Suggested demonstration flow

1. **Sign in and overview:** show the dashboard connection and current model
   readiness. Explain that historical monitoring cards include simulated or
   unknown-provenance records rather than live production traffic.
2. **Model 1:** show the saved Hotel Booking Cancellation model, its ordered
   feature schema and clearly labeled held-out evaluation. If demonstrating
   prediction, use a permitted feature-only CSV/record and identify it as a
   local demonstration input. Do not describe a single prediction as a
   performance evaluation.
3. **Data Quality and Data Drift:** explain that quality checks cover schema,
   missing/invalid values and duplicates; PSI compares feature distributions,
   and KS checks numerical distributions. The documented thresholds and
   interpretation are in the README and architecture docs. A drift alert is
   evidence of input change, not proof of degraded model accuracy.
4. **Batch Explorer, Reports, and Alerts:** inspect existing local history and
   show source/provenance labels. Do not relabel old simulated or unknown
   records as real. Use report download only for an existing report if no
   new demo batch is intended.
5. **Model 2:** show the `HISTORICAL_OFFLINE` panel first. Explain that it is
   trained on chronological Kaggle replay data, with 41 valid labels and weak
   holdout results (accuracy 0.333, F1 0, ROC-AUC 0.400); it is not a validated
   early-warning model. Then show the separate production panel:
   `NOT_TRAINED`, **0/20** genuine eligible labels, and its eligibility reason.
   Synthetic test fixtures never qualify for production training.
6. **Production Data and System Health & Settings:** show that ingestion is
   not configured and that health responses report configuration presence,
   not secret values. Do not enter a production token during a presentation.
7. **Close:** state that the app is a local prototype. No trusted external
   producer, production deployment, or live performance evaluation has been
   verified.

## 4. Optional non-production monitoring exercise

The UI/API supports CSV monitoring and prediction workflows. If using a
sample/test batch, label it `synthetic_test` or `unknown` as appropriate,
use a unique batch identifier, and explain its provenance. Never use a
synthetic or test batch to claim real production behavior or make it eligible
for Model 2 training. Check the report and alerts after submission and retain
the source classification.

Avoid this optional exercise if the existing local monitoring history must
remain unchanged for the submission. Do not create a batch labeled `real`
unless it is genuinely real and its provenance is supportable.

## 5. End the demo

Stop the two local server processes with `Ctrl+C`. Keep `.env` local and do
not include generated local monitoring records or build output in the
submission.

## 6. Demo checklist

- [ ] Backend and frontend start locally.
- [ ] Login and dashboard load.
- [ ] Model 1 is described with held-out metrics clearly labeled.
- [ ] Drift, data quality, reports, and alerts are shown with provenance.
- [ ] Historical offline Model 2 is labeled `HISTORICAL_OFFLINE` and its poor
  holdout results are disclosed.
- [ ] Production Model 2 is honestly presented as `NOT_TRAINED` (0/20).
- [ ] Production ingestion and security limitations are disclosed.
- [ ] No credential or token is shown or submitted.
