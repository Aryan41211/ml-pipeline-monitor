"""Integration tests for FastAPI API endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from services.api.app import app
from src.jwt_auth import create_access_token

client = TestClient(app)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret-12345")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("JWT_EXPIRATION_MINUTES", "60")
    monkeypatch.setenv("MLMONITOR_API_KEY", "test-api-key")


def _auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _get_token(username: str = "testuser", role: str = "admin") -> str:
    return create_access_token(sub=username, role=role)


class TestHealth:
    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert "status" in body
        assert "version" in body

    def test_health_live(self):
        r = client.get("/health/live")
        assert r.status_code == 200
        assert r.json() == {"status": "alive"}

    def test_health_ready(self):
        r = client.get("/health/ready")
        assert r.status_code == 200
        assert "status" in r.json()

    def test_health_detailed(self):
        r = client.get("/health/detailed")
        assert r.status_code == 200
        body = r.json()
        assert "system" in body
        assert "database" in body

    def test_metrics(self):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "text/plain" in r.headers.get("content-type", "")


class TestAuthV1:
    def test_login_requires_credentials(self):
        r = client.post("/v1/auth/login", json={"username": "", "password": ""})
        assert r.status_code in (401, 422)

    def test_refresh_requires_token(self):
        r = client.post("/v1/auth/refresh", json={"refresh_token": "bad"})
        assert r.status_code == 401

    def test_me_requires_auth(self):
        r = client.get("/v1/auth/me")
        assert r.status_code == 401

    def test_me_with_valid_token(self):
        token = _get_token()
        r = client.get("/v1/auth/me", headers=_auth_header(token))
        assert r.status_code == 200
        body = r.json()
        assert body["sub"] == "testuser"
        assert body["role"] == "admin"


class TestPredictV1:
    def test_predict_requires_auth(self):
        r = client.post("/v1/predict", json={"features": {"x": 1.0}})
        assert r.status_code == 401

    def test_predict_success(self):
        import numpy as np
        token = _get_token()
        fake_model = MagicMock()
        fake_model.predict.return_value = np.array([1])
        fake_scaler = MagicMock()
        fake_scaler.transform.return_value = np.array([[1.0, 2.0]])

        with patch("services.api.app.get_latest_production_model", return_value=(fake_model, fake_scaler, "path")):
            r = client.post(
                "/v1/predict",
                json={"features": {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5}, "dataset": "Iris Species"},
                headers=_auth_header(token),
            )
            assert r.status_code == 200

    def test_predict_no_model(self):
        token = _get_token()
        with patch("services.api.app.get_latest_production_model", return_value=None):
            r = client.post(
                "/v1/predict",
                json={"features": {"x": 1.0}},
                headers=_auth_header(token),
            )
            assert r.status_code == 404


class TestLegacyPredict:
    def test_legacy_requires_api_key(self):
        r = client.post("/predict", json={"features": {"x": 1.0}})
        assert r.status_code == 401

    def test_legacy_with_api_key(self):
        with patch("services.model_service.predict_from_payload", return_value={"model_id": "m1", "predictions": [1]}):
            r = client.post("/predict", json={"features": {"x": 1.0}}, headers={"X-API-Key": "test-api-key"})
            assert r.status_code == 200

    def test_legacy_validation_error(self):
        with patch("services.model_service.predict_from_payload", side_effect=ValueError("bad input")):
            r = client.post("/predict", json={"features": {"x": 1.0}}, headers={"X-API-Key": "test-api-key"})
            assert r.status_code == 400
