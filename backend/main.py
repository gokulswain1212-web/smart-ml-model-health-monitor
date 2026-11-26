from __future__ import annotations

import base64
from collections import deque
import hashlib
import hmac
import io
import json
import os
import re
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import Response

from backend.dashboard import (
    REPORTS_DIR,
    _model2_summary,
    _report_source,
    build_dashboard_data,
    monitoring_report_trend_summary,
)
from ml.alert_engine import ALERTS_PATH, backfill_alert_history, load_alert_history
from ml.model2_prediction import (
    Model2ArtifactError,
    features_for_batch,
    predict_batch_degradation,
    predict_degradation,
)
from ml.offline_model2 import (
    OfflineModel2ArtifactError,
    offline_model2_status,
    predict_offline_degradation,
)
from ml.model2_development_prediction import (
    DevelopmentFeatureError,
    DevelopmentModelError,
    predict_development_degradation,
)
from ml.model2_training import run_training_if_eligible
from ml.monitoring import monitor_batch_data
from ml.performance_tracking import (
    DuplicateOutcomeError,
    FEATURE_NAMES_PATH,
    MODEL_METADATA_PATH,
    PoorQualityBatchError,
    PREDICTIONS_DIR,
    REFERENCE_DATA_PATH,
    evaluate_batch_performance,
    import_actual_outcomes_csv,
    list_production_observations,
    list_production_performance,
    log_batch_predictions,
    record_model2_production_prediction,
    record_production_outcome,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PIPELINE_PATH = ROOT / "ml" / "models" / "hotel_cancellation_pipeline.pkl"
MAX_OUTCOME_CSV_BYTES = 10 * 1024 * 1024
MAX_BATCH_ID_LENGTH = 128
SESSION_COOKIE_NAME = "smart_ml_session"
SESSION_TTL_SECONDS = 8 * 60 * 60
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
)


def _configured_origins() -> list[str]:
    configured = os.getenv("CORS_ORIGINS")
    if configured is None:
        return list(DEFAULT_CORS_ORIGINS)
    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    if "*" in origins:
        raise RuntimeError("CORS_ORIGINS must not contain a wildcard origin.")
    return origins


def _local_auth_configuration() -> tuple[str, str, bytes]:
    username = os.getenv("LOCAL_AUTH_USERNAME", "").strip()
    password = os.getenv("LOCAL_AUTH_PASSWORD", "")
    secret = os.getenv("LOCAL_AUTH_SECRET", "")
    if not username or len(password) < 6 or len(secret) < 32:
        raise HTTPException(
            status_code=503,
            detail=(
                "Local authentication is not configured. Set "
                "LOCAL_AUTH_USERNAME, a LOCAL_AUTH_PASSWORD of at least "
                "6 characters, and a random LOCAL_AUTH_SECRET of at least "
                "32 characters."
            ),
        )
    return username, password, secret.encode("utf-8")


def _validate_browser_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin and origin not in _configured_origins():
        raise HTTPException(status_code=403, detail="Request origin is not allowed.")


def _create_session_token(username: str, secret: bytes) -> str:
    payload = json.dumps(
        {
            "sub": username,
            "exp": int(datetime.now(timezone.utc).timestamp())
            + SESSION_TTL_SECONDS,
            "nonce": secrets.token_urlsafe(16),
        },
        separators=(",", ":"),
    ).encode("utf-8")
    encoded_payload = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    signature = hmac.new(
        secret, encoded_payload.encode("ascii"), hashlib.sha256
    ).digest()
    encoded_signature = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
    return f"{encoded_payload}.{encoded_signature}"


def require_authenticated_user(request: Request) -> str:
    username, _, secret = _local_auth_configuration()
    if request.method != "GET":
        _validate_browser_origin(request)

    token = request.cookies.get(SESSION_COOKIE_NAME, "")
    try:
        encoded_payload, provided_signature = token.split(".", maxsplit=1)
        expected_signature = hmac.new(
            secret, encoded_payload.encode("ascii"), hashlib.sha256
        ).digest()
        expected_encoded_signature = (
            base64.urlsafe_b64encode(expected_signature).decode("ascii").rstrip("=")
        )
        if not hmac.compare_digest(
            provided_signature.encode("ascii"),
            expected_encoded_signature.encode("ascii"),
        ):
            raise ValueError("Invalid session signature.")

        payload_bytes = base64.urlsafe_b64decode(
            encoded_payload + "=" * (-len(encoded_payload) % 4)
        )
        payload = json.loads(payload_bytes)
    except (ValueError, UnicodeError):
        raise HTTPException(
            status_code=401, detail="Authentication is required."
        ) from None

    if (
        not isinstance(payload, dict)
        or payload.get("sub") != username
        or type(payload.get("exp")) is not int
        or payload["exp"] <= int(datetime.now(timezone.utc).timestamp())
    ):
        raise HTTPException(status_code=401, detail="Session has expired.")
    return username


def _model1_readiness() -> str:
    if not MODEL_PIPELINE_PATH.exists():
        return "UNAVAILABLE"
    if not FEATURE_NAMES_PATH.exists() or not MODEL_METADATA_PATH.exists():
        return "UNAVAILABLE"
    try:
        feature_names = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
        metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return "UNAVAILABLE"
    if not isinstance(feature_names, list) or not isinstance(metadata, dict):
        return "UNAVAILABLE"
    if metadata.get("feature_names") != feature_names:
        return "UNAVAILABLE"
    return "READY"


def _schema_readiness() -> str:
    if not FEATURE_NAMES_PATH.exists() or not MODEL_METADATA_PATH.exists():
        return "UNAVAILABLE"
    if not REFERENCE_DATA_PATH.exists():
        return "UNAVAILABLE"
    try:
        feature_names = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
        metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return "UNAVAILABLE"
    if not isinstance(feature_names, list) or not isinstance(metadata, dict):
        return "UNAVAILABLE"
    if metadata.get("feature_names") != feature_names:
        return "UNAVAILABLE"
    try:
        required_feature_set = set(feature_names)
        metadata_feature_set = set(metadata.get("numerical_features", [])) | set(
            metadata.get("categorical_features", [])
        )
        if metadata_feature_set != required_feature_set:
            return "UNAVAILABLE"
        reference_data = pd.read_csv(
            REFERENCE_DATA_PATH,
            usecols=lambda column: column in required_feature_set,
            nrows=0,
        )
    except (OSError, TypeError, ValueError, pd.errors.ParserError):
        return "UNAVAILABLE"
    if list(reference_data.columns) != feature_names:
        return "UNAVAILABLE"
    return "READY"


def _application_status() -> str:
    if _model1_readiness() == "READY" and _schema_readiness() == "READY":
        return "OK"
    return "DEGRADED"


def _production_ingestion_configured() -> bool:
    return len(os.getenv("PRODUCTION_INGESTION_TOKEN", "")) >= 32


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        _local_auth_configuration()
    except HTTPException as error:
        raise RuntimeError(error.detail) from error
    yield


async def _read_limited_body(request: Request) -> bytes:
    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > MAX_OUTCOME_CSV_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"CSV must not exceed {MAX_OUTCOME_CSV_BYTES} bytes.",
            )
        content.extend(chunk)
    return bytes(content)


def _validate_batch_id(batch_id: str) -> str:
    if len(batch_id) > MAX_BATCH_ID_LENGTH or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", batch_id
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "batch_id must start with a letter or digit and contain only "
                "letters, digits, dots, underscores, or hyphens (maximum 128 characters)."
            ),
        )
    return batch_id


app = FastAPI(
    title="Smart ML Model Health Monitor API",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_configured_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_credentials=True,
)

PRODUCTION_INGESTION_PATHS = {
    "/api/v1/monitoring/observations": "observations",
    "/api/v1/monitoring/outcomes": "outcomes",
}
PRODUCTION_INGESTION_EVENT_HISTORY: deque[dict[str, Any]] = deque(maxlen=50)


@app.middleware("http")
async def record_production_ingestion_event(request: Request, call_next: Any) -> Response:
    response = await call_next(request)
    endpoint = PRODUCTION_INGESTION_PATHS.get(request.url.path)
    if endpoint is None or request.method != "POST":
        return response

    status_code = response.status_code
    category = (
        "accepted"
        if status_code < 400
        else {
            401: "authentication_rejected",
            404: "observation_not_found",
            409: "duplicate_submission",
            422: "validation_rejected",
            503: "ingestion_not_configured_or_unavailable",
        }.get(status_code, "request_rejected")
    )
    PRODUCTION_INGESTION_EVENT_HISTORY.appendleft(
        {
            "event_id": uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "status_code": status_code,
            "category": category,
        }
    )
    return response


class LocalLoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=1, max_length=1024)


class PerformanceDegradationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str | None = None
    monitoring_features: dict[str, float | None] | None = None


class OfflineModel2PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    monitoring_features: dict[str, float | None]


class Model2DevelopmentPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_BATCH_ID_LENGTH,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    monitoring_features: dict[str, Any] | None = None


class CancellationPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str = Field(
        min_length=1,
        max_length=MAX_BATCH_ID_LENGTH,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    records: list[dict[str, Any]] = Field(min_length=1, max_length=10000)


class ProductionObservationsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_BATCH_ID_LENGTH,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    data_source_type: Literal["real"]
    records: list[dict[str, Any]] = Field(min_length=1, max_length=10000)


class ProductionOutcomeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_id: str = Field(min_length=1, max_length=128)
    is_canceled: int = Field(strict=True, ge=0, le=1)


def require_production_ingestion_token(request: Request) -> None:
    configured_token = os.getenv("PRODUCTION_INGESTION_TOKEN", "")
    if len(configured_token) < 32:
        raise HTTPException(
            status_code=503,
            detail=(
                "Production ingestion is not configured. Set "
                "PRODUCTION_INGESTION_TOKEN to a random secret of at least "
                "32 characters and provide it to the trusted production source."
            ),
        )
    provided_token = request.headers.get("X-Production-Ingestion-Token", "")
    if not hmac.compare_digest(
        provided_token.encode("utf-8"), configured_token.encode("utf-8")
    ):
        raise HTTPException(
            status_code=401,
            detail="A valid production ingestion token is required.",
        )


@app.post("/api/v1/auth/login", dependencies=[Depends(_validate_browser_origin)])
def login_local_user(
    request: LocalLoginRequest,
    response: Response,
) -> dict[str, str]:
    username, password, secret = _local_auth_configuration()
    username_matches = hmac.compare_digest(
        request.username.encode("utf-8"), username.encode("utf-8")
    )
    password_matches = hmac.compare_digest(
        request.password.encode("utf-8"), password.encode("utf-8")
    )
    if not (username_matches and password_matches):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=_create_session_token(username, secret),
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=False,
        samesite="lax",
        path="/",
    )
    return {"username": username}


@app.get("/api/v1/auth/session")
def read_local_session(
    username: str = Depends(require_authenticated_user),
) -> dict[str, str]:
    return {"username": username}


@app.post(
    "/api/v1/auth/logout",
    status_code=204,
    dependencies=[Depends(_validate_browser_origin)],
)
def logout_local_user(response: Response) -> Response:
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        httponly=True,
        secure=False,
        samesite="lax",
        path="/",
    )
    response.status_code = 204
    return response


@app.get("/health")
def get_health() -> dict[str, Any]:
    model2_status = _model2_summary()
    model1_readiness = _model1_readiness()
    schema_readiness = _schema_readiness()
    return {
        "application_status": (
            "OK"
            if model1_readiness == "READY" and schema_readiness == "READY"
            else "DEGRADED"
        ),
        "model1_readiness": model1_readiness,
        "model2_readiness": model2_status["status"],
        "schema_readiness": schema_readiness,
        "production_ingestion_configured": _production_ingestion_configured(),
    }


@app.get(
    "/api/v1/dashboard",
    dependencies=[Depends(require_authenticated_user)],
)
def get_dashboard(
    period_days: int = Query(default=30, ge=1, le=3650),
) -> dict[str, Any]:
    dashboard = build_dashboard_data(period_days)
    model1_readiness = _model1_readiness()
    schema_readiness = _schema_readiness()
    dashboard["system"].update(
        {
            "application_status": (
                "OK"
                if model1_readiness == "READY" and schema_readiness == "READY"
                else "DEGRADED"
            ),
            "model1_readiness": model1_readiness,
            "schema_readiness": schema_readiness,
            "production_ingestion_configured": _production_ingestion_configured(),
        }
    )
    return dashboard


@app.get(
    "/api/v1/model2/training-status",
    dependencies=[Depends(require_authenticated_user)],
)
def get_model2_training_status() -> dict[str, Any]:
    return _model2_summary()


@app.get(
    "/api/v1/model2/offline/status",
    dependencies=[Depends(require_authenticated_user)],
)
def get_offline_model2_status() -> dict[str, Any]:
    try:
        return offline_model2_status()
    except OfflineModel2ArtifactError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post(
    "/api/v1/model2/offline/predict",
    dependencies=[Depends(require_authenticated_user)],
)
def predict_offline_model2(
    request: OfflineModel2PredictionRequest,
) -> dict[str, Any]:
    try:
        status = offline_model2_status()
        if status["status"] != "TRAINED":
            raise HTTPException(
                status_code=503,
                detail=status["reason"]
                or "Historical offline Model 2 is not trained.",
            )
        return predict_offline_degradation(request.monitoring_features)
    except OfflineModel2ArtifactError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get(
    "/api/v1/monitoring/ingestion-status",
    dependencies=[Depends(require_authenticated_user)],
)
def get_production_ingestion_status() -> dict[str, Any]:
    return {
        "configured": _production_ingestion_configured(),
        "retention_limit": PRODUCTION_INGESTION_EVENT_HISTORY.maxlen,
        "events": list(PRODUCTION_INGESTION_EVENT_HISTORY),
    }


@app.post(
    "/api/v1/monitoring/observations",
    dependencies=[Depends(require_production_ingestion_token)],
)
def ingest_production_observations(
    request: ProductionObservationsRequest,
) -> dict[str, Any]:
    batch_id = request.batch_id or (
        "prod-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "-"
        + uuid4().hex[:12]
    )
    _validate_batch_id(batch_id)
    if re.fullmatch(
        r"(?:normal|drifted|poor_quality)-.+-seed-\d+",
        batch_id,
        flags=re.IGNORECASE,
    ):
        raise HTTPException(
            status_code=422,
            detail="Production observations cannot use seeded simulator batch IDs.",
        )
    prediction_path = PREDICTIONS_DIR / f"{batch_id}.jsonl"
    if prediction_path.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Predictions already exist for batch {batch_id!r}.",
        )

    prediction_written = False
    ingestion_complete = False
    try:
        feature_names = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
        booking_ids: list[str] = []
        internal_row_ids: list[str] = []
        booking_id_hmacs: list[str] = []
        ingestion_token = os.environ["PRODUCTION_INGESTION_TOKEN"].encode("utf-8")
        for index, record in enumerate(request.records):
            keys = list(record)
            expected_keys = ["booking_id", *feature_names]
            if keys != expected_keys:
                raise ValueError(
                    f"Observation {index} must contain booking_id followed by "
                    "the Model 1 features in the exact documented order."
                )
            booking_id = record["booking_id"]
            if (
                not isinstance(booking_id, str)
                or not booking_id.strip()
                or len(booking_id) > 256
            ):
                raise ValueError(
                    f"Observation {index} booking_id must be a non-empty string "
                    "of at most 256 characters."
                )
            booking_ids.append(booking_id.strip())
            internal_row_ids.append(uuid4().hex)
            booking_id_hmacs.append(
                hmac.new(
                    ingestion_token,
                    booking_id.strip().encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()
            )
        if len(set(booking_ids)) != len(booking_ids):
            raise ValueError("booking_id values must be unique within the batch.")
        batch = pd.DataFrame(
            [
                {
                    **{feature: record[feature] for feature in feature_names},
                    "row_id": row_id,
                    "source_booking_id_hmac": booking_hmac,
                }
                for record, row_id, booking_hmac in zip(
                    request.records, internal_row_ids, booking_id_hmacs, strict=True
                )
            ]
        )
        batch.insert(0, "batch_id", batch_id)
        prediction = log_batch_predictions(batch, source="production")
        prediction_written = True
        report, report_path = monitor_batch_data(
            batch,
            data_source_type="real",
            source="production",
        )
        performance = evaluate_batch_performance(batch_id)
        ingestion_complete = True
    except PoorQualityBatchError as error:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "poor_quality_batch",
                "message": str(error),
                "quality_report": error.quality_report,
            },
        ) from error
    except FileExistsError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="Model prediction or monitoring reference data is unavailable.",
        ) from error
    except (TypeError, ValueError, pd.errors.ParserError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        if prediction_written and not ingestion_complete:
            prediction_path.unlink(missing_ok=True)

    return {
        "batch_id": batch_id,
        "prediction_timestamp": prediction["prediction_timestamp"],
        "source": "production",
        "data_source_type": report["data_source_type"],
        "monitoring_timestamp": report["monitoring_timestamp"],
        "report_path": report_path.name,
        "prediction_count": prediction["prediction_count"],
        "observation_ids": prediction["observation_ids"],
        "predictions": prediction["predictions"],
        "model_version": prediction["model_version"],
        "model_artifact_sha256": prediction["model_artifact_sha256"],
        "quality_status": prediction["quality_status"],
        "number_of_pending_outcomes": performance["number_of_pending_outcomes"],
    }


@app.post(
    "/api/v1/monitoring/outcomes",
    dependencies=[Depends(require_production_ingestion_token)],
)
def record_production_booking_outcome(
    request: ProductionOutcomeRequest,
) -> dict[str, Any]:
    try:
        result = record_production_outcome(
            request.observation_id,
            request.is_canceled,
        )
    except DuplicateOutcomeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    training = run_training_if_eligible()
    return {
        "observation_id": result["observation_id"],
        "batch_id": result["batch_id"],
        "outcome_received_at": result["outcome"]["outcome_received_at"],
        "performance": result["performance"],
        "model2_training": training,
    }


@app.get(
    "/api/v1/monitoring/observations",
    dependencies=[Depends(require_authenticated_user)],
)
def get_production_observations(
    period_days: int = Query(default=3650, ge=1, le=3650),
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    observations = list_production_observations(
        period_days=period_days,
        limit=limit,
    )
    return {"count": len(observations), "observations": observations}


@app.get(
    "/api/v1/monitoring/performance",
    dependencies=[Depends(require_authenticated_user)],
)
def get_production_performance(
    period_days: int = Query(default=3650, ge=1, le=3650),
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    performance = list_production_performance(
        period_days=period_days,
        limit=limit,
    )
    return {"count": len(performance), "performance": performance}


@app.get(
    "/api/v1/alerts",
    dependencies=[Depends(require_authenticated_user)],
)
def get_alert_history(
    severity: str | None = Query(default=None),
    batch_id: str | None = Query(default=None, min_length=1),
    source: str | None = Query(default=None),
    period_days: int = Query(default=30, ge=1, le=3650),
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    severity_filter = severity.upper() if severity else None
    source_filter = source.upper() if source else None
    if severity_filter not in {None, "HEALTHY", "WARNING", "CRITICAL"}:
        raise HTTPException(
            status_code=422,
            detail="severity must be HEALTHY, WARNING, or CRITICAL.",
        )
    if source_filter not in {None, "REAL", "SIMULATED", "UNKNOWN"}:
        raise HTTPException(
            status_code=422,
            detail="source must be REAL, SIMULATED, or UNKNOWN.",
        )
    cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)
    backfill_alert_history(alerts_path=ALERTS_PATH)
    alerts = load_alert_history(
        ALERTS_PATH,
        severity=severity_filter,
        batch_id=batch_id,
        source=source_filter,
    )
    alerts = [
        alert
        for alert in alerts
        if (
            timestamp := datetime.fromisoformat(
                str(alert["timestamp"]).replace("Z", "+00:00")
            )
        )
        >= cutoff
    ]
    return {"count": len(alerts), "alerts": alerts[:limit]}


@app.get(
    "/api/v1/monitoring/reports",
    dependencies=[Depends(require_authenticated_user)],
)
def get_monitoring_reports(
    period_days: int = Query(default=30, ge=1, le=3650),
    source: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    source_filter = source.upper() if source else None
    if source_filter not in {None, "REAL", "SIMULATED", "UNKNOWN"}:
        raise HTTPException(
            status_code=422,
            detail="source must be REAL, SIMULATED, or UNKNOWN.",
        )
    cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)
    reports = []
    if REPORTS_DIR.exists():
        for path in REPORTS_DIR.glob("*.json"):
            report = json.loads(path.read_text(encoding="utf-8"))
            timestamp = datetime.fromisoformat(
                str(report.get("monitoring_timestamp", "")).replace("Z", "+00:00")
            )
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            report_source = _report_source(report)
            source_name = {
                "real": "REAL",
                "synthetic_test": "SIMULATED",
                "unknown": "UNKNOWN",
            }[report_source]
            if timestamp < cutoff or (
                source_filter and source_name != source_filter
            ):
                continue
            reports.append(
                {
                    "batch_id": report["batch_id"],
                    "timestamp": timestamp.isoformat(),
                    "status": report.get("overall_status", "UNKNOWN"),
                    "quality_status": report.get("quality", {}).get(
                        "quality_status", "UNKNOWN"
                    ),
                    "source": source_name,
                    "rows": report.get("quality", {}).get("total_rows", 0),
                    "download_url": f"/api/v1/monitoring/reports/{report['batch_id']}",
                    **monitoring_report_trend_summary(report),
                }
            )
    reports.sort(key=lambda report: report["timestamp"], reverse=True)
    return {"count": len(reports), "reports": reports[:limit]}


@app.get(
    "/api/v1/monitoring/reports/{batch_id}",
    dependencies=[Depends(require_authenticated_user)],
)
def download_monitoring_report(batch_id: str) -> Response:
    _validate_batch_id(batch_id)
    if not REPORTS_DIR.exists():
        raise HTTPException(status_code=404, detail="Monitoring report not found.")
    candidates = []
    for path in REPORTS_DIR.glob("*.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("batch_id") == batch_id:
            candidates.append((report.get("monitoring_timestamp", ""), path))
    if not candidates:
        raise HTTPException(status_code=404, detail="Monitoring report not found.")
    _, report_path = max(candidates, key=lambda item: item[0])
    return Response(
        content=report_path.read_text(encoding="utf-8"),
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="monitoring_{batch_id}.json"'
        },
    )


@app.post(
    "/api/v1/monitoring/batches/csv",
    dependencies=[Depends(require_authenticated_user)],
)
async def monitor_batch_csv(
    request: Request,
    batch_id: str = Query(
        min_length=1,
        max_length=MAX_BATCH_ID_LENGTH,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
    data_source_type: str = Query(default="unknown"),
) -> dict[str, Any]:
    if data_source_type not in {"real", "synthetic_test", "unknown"}:
        raise HTTPException(
            status_code=422,
            detail="data_source_type must be real, synthetic_test, or unknown.",
        )
    content_type = request.headers.get("content-type", "").split(";", maxsplit=1)[0]
    if content_type not in {"text/csv", "application/csv"}:
        raise HTTPException(
            status_code=415,
            detail="Send the monitoring batch with Content-Type: text/csv.",
        )
    content = await _read_limited_body(request)
    try:
        batch = pd.read_csv(io.BytesIO(content))
        _validate_batch_id(batch_id)
        if batch.empty:
            raise ValueError("Monitoring CSV contains no records.")
        if "batch_id" in batch:
            batch_ids = batch["batch_id"].dropna().astype(str).unique().tolist()
            if (
                len(batch_ids) != 1
                or batch_ids[0] != batch_id
                or batch["batch_id"].isna().any()
            ):
                raise ValueError("CSV batch_id values do not match the query batch_id.")
        else:
            batch.insert(0, "batch_id", batch_id)
        report, report_path = monitor_batch_data(
            batch, data_source_type=data_source_type
        )
    except FileExistsError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="Monitoring reference data or model metadata is unavailable.",
        ) from error
    except (TypeError, ValueError, pd.errors.ParserError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "batch_id": report["batch_id"],
        "timestamp": report["monitoring_timestamp"],
        "status": report["overall_status"],
        "quality_status": report["quality"]["quality_status"],
        "data_source_type": report["data_source_type"],
        "rows": report["quality"]["total_rows"],
        "report_path": report_path.name,
        "reasons": report["reasons"],
    }


@app.post(
    "/api/v1/predictions/cancellation",
    dependencies=[Depends(require_authenticated_user)],
)
def predict_cancellations(
    request: CancellationPredictionRequest,
) -> dict[str, Any]:
    try:
        batch = pd.DataFrame(request.records)
        batch.insert(0, "batch_id", request.batch_id)
        result = log_batch_predictions(batch)
    except PoorQualityBatchError as error:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "poor_quality_batch",
                "message": str(error),
                "quality_report": error.quality_report,
            },
        ) from error
    except FileExistsError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="Model prediction service is not configured.",
        ) from error
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {
        "batch_id": result["batch_id"],
        "prediction_timestamp": result["prediction_timestamp"],
        "prediction_count": result["prediction_count"],
        "predictions": result["predictions"],
        "model_version": result["model_version"],
        "model_artifact_sha256": result["model_artifact_sha256"],
        "quality_status": result["quality_status"],
    }


@app.post(
    "/api/v1/predictions/cancellation/csv",
    dependencies=[Depends(require_authenticated_user)],
)
async def predict_cancellations_csv(
    request: Request,
    batch_id: str = Query(
        min_length=1,
        max_length=MAX_BATCH_ID_LENGTH,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "").split(";", maxsplit=1)[0]
    if content_type not in {"text/csv", "application/csv"}:
        raise HTTPException(
            status_code=415,
            detail="Send prediction records with Content-Type: text/csv.",
        )
    content = await _read_limited_body(request)
    _validate_batch_id(batch_id)
    try:
        batch = pd.read_csv(io.BytesIO(content))
        if "batch_id" in batch:
            ids = batch["batch_id"].dropna().astype(str).unique().tolist()
            if len(ids) != 1 or ids[0] != batch_id or batch["batch_id"].isna().any():
                raise ValueError("CSV batch_id values do not match the query batch_id.")
        else:
            batch.insert(0, "batch_id", batch_id)
        if len(batch) > 10000:
            raise ValueError("Prediction CSV must contain at most 10,000 rows.")
        result = log_batch_predictions(batch)
    except PoorQualityBatchError as error:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "poor_quality_batch",
                "message": str(error),
                "quality_report": error.quality_report,
            },
        ) from error
    except FileExistsError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="Model prediction service is not configured.",
        ) from error
    except (TypeError, ValueError, pd.errors.ParserError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "batch_id": result["batch_id"],
        "prediction_timestamp": result["prediction_timestamp"],
        "prediction_count": result["prediction_count"],
        "predictions": result["predictions"],
        "model_version": result["model_version"],
        "model_artifact_sha256": result["model_artifact_sha256"],
        "quality_status": result["quality_status"],
    }


@app.post(
    "/api/v1/outcomes/{batch_id}/csv",
    dependencies=[Depends(require_authenticated_user)],
)
async def import_batch_outcomes_csv(
    batch_id: str,
    request: Request,
) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "").split(";", maxsplit=1)[0]
    if content_type not in {"text/csv", "application/csv"}:
        raise HTTPException(
            status_code=415,
            detail="Send the outcome file with Content-Type: text/csv.",
        )

    csv_content = await _read_limited_body(request)

    try:
        result = import_actual_outcomes_csv(batch_id, csv_content)
    except DuplicateOutcomeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    performance = result["performance"]
    return {
        "batch_id": batch_id,
        "outcome_count": result["outcome"]["outcome_count"],
        "outcome_received_at": result["outcome"]["outcome_received_at"],
        "evaluation_timestamp": performance["evaluation_timestamp"],
        "number_of_evaluated_records": performance["number_of_evaluated_records"],
        "number_of_pending_outcomes": performance["number_of_pending_outcomes"],
        "metrics_available": performance["metrics_available"],
        "accuracy": performance["accuracy"],
        "precision": performance["precision"],
        "recall": performance["recall"],
        "f1_score": performance["f1_score"],
        "confusion_matrix": performance["confusion_matrix"],
        "data_source_type": performance["data_source_type"],
    }


@app.post(
    "/api/v1/predictions/performance-degradation",
    dependencies=[Depends(require_authenticated_user)],
)
def predict_performance_degradation(
    request: PerformanceDegradationRequest,
) -> dict[str, Any]:
    if request.monitoring_features is None and not request.batch_id:
        raise HTTPException(
            status_code=422,
            detail="Provide either batch_id or monitoring_features.",
        )
    readiness = _model2_summary()
    if readiness["status"] != "TRAINED":
        raise HTTPException(
            status_code=503,
            detail=(
                readiness["reason"]
                or "Production Model 2 is not trained and is unavailable for predictions."
            ),
        )
    if request.monitoring_features is not None:
        try:
            result = predict_degradation(request.monitoring_features)
        except Model2ArtifactError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    else:
        production_batches = {
            observation["batch_id"]
            for observation in list_production_observations(limit=1000)
        }
        if request.batch_id not in production_batches:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No trusted production observations were found for batch "
                    f"{request.batch_id!r}."
                ),
            )
        try:
            result = predict_batch_degradation(request.batch_id or "")
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except Model2ArtifactError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    if result["model2_status"] == "NOT_TRAINED":
        raise HTTPException(status_code=503, detail=result["explanation"])
    if request.monitoring_features is not None:
        return {
            **({"batch_id": request.batch_id} if request.batch_id else {}),
            **result,
            "source": "operator_supplied",
            "data_source_type": "unverified",
        }

    production_result = {
        **result,
        "source": "production",
        "data_source_type": "real",
    }
    try:
        stored_prediction = record_model2_production_prediction(production_result)
    except (OSError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail=f"Production Model 2 prediction could not be persisted: {error}",
        ) from error
    return {
        **production_result,
        "prediction_id": stored_prediction["prediction_id"],
    }


@app.post(
    "/api/v1/model2/development/predict",
    dependencies=[Depends(require_authenticated_user)],
)
def predict_model2_development(
    request: Model2DevelopmentPredictionRequest,
) -> dict[str, Any]:
    if (request.batch_id is None) == (request.monitoring_features is None):
        raise HTTPException(
            status_code=422,
            detail="Provide exactly one of batch_id or monitoring_features.",
        )
    try:
        features = (
            features_for_batch(
                request.batch_id,
                expected_source="synthetic_test",
            )
            if request.batch_id is not None
            else request.monitoring_features
        )
        result = predict_development_degradation(features or {})
    except DevelopmentFeatureError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (DevelopmentModelError, FileNotFoundError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"batch_id": request.batch_id, **result} if request.batch_id else result
