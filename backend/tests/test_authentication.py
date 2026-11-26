from pathlib import Path

from fastapi.testclient import TestClient

import backend.main as main


def test_env_example_has_blank_local_credentials_and_token_placeholder():
    env_example_path = Path(__file__).resolve().parents[2] / ".env.example"
    credential_names = {
        "LOCAL_AUTH_USERNAME",
        "LOCAL_AUTH_PASSWORD",
        "LOCAL_AUTH_SECRET",
        "PRODUCTION_INGESTION_TOKEN",
    }
    configured_values = {}

    for line in env_example_path.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition("=")
        if separator and name in credential_names:
            configured_values[name] = value

    assert configured_values == {
        "LOCAL_AUTH_USERNAME": "",
        "LOCAL_AUTH_PASSWORD": "",
        "LOCAL_AUTH_SECRET": "",
        "PRODUCTION_INGESTION_TOKEN": "your_secure_token_here",
    }


def test_data_endpoints_require_authenticated_session():
    client = TestClient(main.app)
    requests = [
        ("GET", "/api/v1/dashboard", {}),
        ("GET", "/api/v1/alerts", {}),
        ("GET", "/api/v1/monitoring/reports", {}),
        ("GET", "/api/v1/monitoring/ingestion-status", {}),
        ("GET", "/api/v1/model2/training-status", {}),
        ("GET", "/api/v1/monitoring/reports/unknown-batch", {}),
        (
            "POST",
            "/api/v1/monitoring/batches/csv?batch_id=unauthenticated",
            {"content": b"", "headers": {"Content-Type": "text/csv"}},
        ),
        (
            "POST",
            "/api/v1/predictions/cancellation",
            {"json": {"batch_id": "unauthenticated", "records": [{"hotel": "City Hotel"}]}},
        ),
        (
            "POST",
            "/api/v1/predictions/cancellation/csv?batch_id=unauthenticated",
            {"content": b"", "headers": {"Content-Type": "text/csv"}},
        ),
        (
            "POST",
            "/api/v1/outcomes/unauthenticated/csv",
            {"content": b"", "headers": {"Content-Type": "text/csv"}},
        ),
        (
            "POST",
            "/api/v1/predictions/performance-degradation",
            {"json": {"batch_id": "unauthenticated"}},
        ),
    ]

    for method, url, kwargs in requests:
        assert client.request(method, url, **kwargs).status_code == 401


def test_local_login_session_cookie_and_logout():
    client = TestClient(main.app)
    response = client.post(
        "/api/v1/auth/login",
        json={
            "username": "tests@example.com",
            "password": "test-only-password-123",
        },
        headers={"Origin": "http://127.0.0.1:5173"},
    )

    assert response.status_code == 200
    assert response.json() == {"username": "tests@example.com"}
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
    assert response.headers["access-control-allow-credentials"] == "true"
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie
    assert client.get("/api/v1/auth/session").json() == {
        "username": "tests@example.com"
    }
    initial_dashboard = client.get("/api/v1/dashboard?period_days=30")
    refreshed_dashboard = client.get("/api/v1/dashboard?period_days=7")
    assert initial_dashboard.status_code == 200
    assert refreshed_dashboard.status_code == 200
    assert initial_dashboard.json()["model2"]["status"] == "NOT_TRAINED"
    assert refreshed_dashboard.json()["model2"]["minimum_labeled_observations"] == 20

    logout = client.post(
        "/api/v1/auth/logout",
        headers={"Origin": "http://127.0.0.1:5173"},
    )

    assert logout.status_code == 204
    assert "Max-Age=0" in logout.headers["set-cookie"]
    assert client.get("/api/v1/auth/session").status_code == 401
    assert client.get("/api/v1/dashboard").status_code == 401


def test_local_login_allows_development_origins():
    client = TestClient(main.app)
    origins = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    )

    for origin in origins:
        preflight = client.options(
            "/api/v1/auth/login",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == origin
        assert preflight.headers["access-control-allow-credentials"] == "true"

        response = client.post(
            "/api/v1/auth/login",
            json={
                "username": "tests@example.com",
                "password": "test-only-password-123",
            },
            headers={"Origin": origin},
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin


def test_local_auth_accepts_six_character_password(monkeypatch):
    monkeypatch.setenv("LOCAL_AUTH_USERNAME", "tests@example.com")
    monkeypatch.setenv("LOCAL_AUTH_PASSWORD", "123456")
    monkeypatch.setenv("LOCAL_AUTH_SECRET", "test-only-session-signing-secret-1234567890")

    assert main._local_auth_configuration() == (
        "tests@example.com",
        "123456",
        b"test-only-session-signing-secret-1234567890",
    )


def test_invalid_credentials_and_untrusted_origins_are_rejected(monkeypatch):
    client = TestClient(main.app)

    invalid_credentials = client.post(
        "/api/v1/auth/login",
        json={"username": "tests@example.com", "password": "wrong"},
    )
    untrusted_origin = client.post(
        "/api/v1/auth/login",
        json={
            "username": "tests@example.com",
            "password": "test-only-password-123",
        },
        headers={"Origin": "https://untrusted.example"},
    )

    assert invalid_credentials.status_code == 401
    assert untrusted_origin.status_code == 403

    untrusted_preflight = client.options(
        "/api/v1/auth/login",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert untrusted_preflight.status_code == 400
    assert "access-control-allow-origin" not in untrusted_preflight.headers

    monkeypatch.delenv("LOCAL_AUTH_USERNAME")
    unconfigured = client.post(
        "/api/v1/auth/login",
        json={
            "username": "tests@example.com",
            "password": "test-only-password-123",
        },
    )
    assert unconfigured.status_code == 503
