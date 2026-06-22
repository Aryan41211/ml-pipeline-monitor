"""Coverage-targeted tests for services/api/app.py login path."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from services.api.app import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret-12345")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("JWT_EXPIRATION_MINUTES", "60")
    monkeypatch.setenv("MLMONITOR_API_KEY", "test-api-key")


class TestLoginCoverage:
    def test_login_success_path(self):
        with patch("src.auth._check_login", return_value=(True, "")):
            with patch("src.auth._resolve_user", return_value="admin"):
                with patch("src.auth._credentials", return_value={"admin": {"password": "x", "role": "admin"}}):
                    r = client.post("/v1/auth/login", json={"username": "admin", "password": "password", "refresh": True})
        assert r.status_code == 200
        body = r.json()
        assert "access_token" in body
        assert body.get("token_type") == "bearer"
        assert body.get("role") == "admin"

    def test_login_invalid_password(self):
        with patch("src.auth._check_login", return_value=(False, "Invalid credentials")):
            r = client.post("/v1/auth/login", json={"username": "admin", "password": "wrong"})
        assert r.status_code == 401

    def test_login_missing_user(self):
        with patch("src.auth._check_login", return_value=(True, "")):
            with patch("src.auth._resolve_user", return_value=None):
                with patch("src.auth._credentials", return_value={}):
                    r = client.post("/v1/auth/login", json={"username": "nobody", "password": "pw"})
        assert r.status_code in (401, 404)
